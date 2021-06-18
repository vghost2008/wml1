#include <stdio.h>
#include <cfloat>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <boost/algorithm/clamp.hpp>
#include <random>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"
//#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include "bboxes.h"
#include "wtoolkit.h"
#include "wtoolkit_cuda.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
/*
 * num_classes: 类别数，不含背景
 * gaussian_iou: 一般为0.7
 * gbboxes: groundtruth bbox, [B,N,4] 相对坐标
 * glabels: 标签[B,N], 背景为0
 * glength: 有效的groundtruth bbox数量
 * output_size: 输出图的大小[2]=(OH,OW) 
 *
 * output:
 * output_heatmaps_c: center heatmaps [B,OH,OW,num_classes]
 * output_hw_offset: [B,OH,OW,4], (h,w,yoffset,xoffset)
 * output_mask: [B,OH,OW,2] (hw_mask,offset_mask)
 */
REGISTER_OP("Center2BoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("num_classes:int")
	.Attr("gaussian_iou:float=0.7")
    .Input("gbboxes: T")
    .Input("glabels: int32")
    .Input("glength: int32")
    .Input("output_size: int32")
	.Output("output_heatmaps_c:T")
	.Output("output_hw_offset:T")
	.Output("output_hw_offset_mask:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            int num_classes;
            c->GetAttr("num_classes",&num_classes);
            auto shape0 = c->MakeShape({batch_size,-1,-1,num_classes});
            auto shape1 = c->MakeShape({batch_size,-1,-1,4});
            auto shape2 = c->MakeShape({batch_size,-1,-1,2});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape2);

			return Status::OK();
			});

template <typename Device,typename T>
class Center2BoxesEncodeOp: public OpKernel {
	public:
		explicit Center2BoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
			OP_REQUIRES_OK(context, context->GetAttr("gaussian_iou", &gaussian_iou_));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("Center2BoxesEncode");
            const Tensor &_gbboxes    = context->input(0);
            const Tensor &_glabels    = context->input(1);
            const Tensor &_gsize      = context->input(2);
            auto          gbboxes     = _gbboxes.template tensor<T,3>();
            auto          glabels     = _glabels.template tensor<int,2>();
            auto          gsize       = _gsize.template tensor<int,1>();
            auto          output_size = context->input(3).template flat<int>().data();
            const auto    batch_size  = _gbboxes.dim_size(0);

            OP_REQUIRES(context, _gbboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimension"));
            OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimension"));
            OP_REQUIRES(context, _gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimension"));

            int           dims_4d0[4]            = {int(batch_size),output_size[0],output_size[1],num_classes_};
            int           dims_4d1[4]            = {int(batch_size),output_size[0],output_size[1],4};
            int           dims_4d2[4]            = {int(batch_size),output_size[0],output_size[1],2};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_heatmaps_c  = NULL;
            Tensor      *output_hw_offset      = NULL;
            Tensor      *output_mask = NULL;

            TensorShapeUtils::MakeShape(dims_4d0, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_4d1, 4, &outshape1);
            TensorShapeUtils::MakeShape(dims_4d2, 4, &outshape2);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_heatmaps_c));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_hw_offset));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape2, &output_mask));

            auto heatmaps_c  = output_heatmaps_c->template tensor<T,4>();
            auto hw_offsets     = output_hw_offset->template tensor<T,4>();
            auto o_mask = output_mask->template tensor<T,4>();
            Eigen::Tensor<float,2,Eigen::RowMajor> max_probs(output_size[0],output_size[1]);

            heatmaps_c.setZero();
            hw_offsets.setZero();
            o_mask.setZero();

            for(auto i=0; i<batch_size; ++i) {
                max_probs.setZero();
                for(auto j=0; j<gsize(i); ++j) {
                    const auto fytl = gbboxes(i,j,0)*(output_size[0]-1);
                    const auto fxtl = gbboxes(i,j,1)*(output_size[1]-1);
                    const auto fybr = gbboxes(i,j,2)*(output_size[0]-1);
                    const auto fxbr = gbboxes(i,j,3)*(output_size[1]-1);
                    const auto fyc = (fytl+fybr)/2;
                    const auto fxc = (fxtl+fxbr)/2;
                    const auto yc = int(fyc+0.5);
                    const auto xc = int(fxc+0.5);
                    const auto r0 = get_gaussian_radius(fybr-fytl,fxbr-fxtl,gaussian_iou_);
                    const auto label = glabels(i,j);
                    const auto h = fybr-fytl;
                    const auto w = fxbr-fxtl;

                    if(yc<0||xc<0||yc>=output_size[0]||xc>=output_size[1]) {
                        cout<<"ERROR bboxes data: "<<gbboxes(i,j,0)<<","<<gbboxes(i,j,1)<<","<<gbboxes(i,j,2)<<","<<gbboxes(i,j,3)<<endl;
                        continue;
                    }

                    draw_gaussian(heatmaps_c,xc,yc,r0,i,label,5);
                    draw_gaussianv2(max_probs,hw_offsets,xc,yc,r0,h,w,i,5);

                    hw_offsets(i,yc,xc,2) = fyc-yc;
                    hw_offsets(i,yc,xc,3) = fxc-xc;
                    o_mask(i,yc,xc,1) = 1.0;
                }
                o_mask.chip(i,0).chip(0,2) = max_probs;
            }
        }
        template<typename DT>
        static void draw_gaussian(DT& data,int cx,int cy,float radius,int batch_index,int class_index,float delta=6,float k=1.0)
        {
            const auto width   = data.dimension(2);
            const auto height  = data.dimension(1);
            const auto xtl     = max(0,int(cx-radius));
            const auto ytl     = max(0,int(cy-radius));
            const auto xbr     = min<int>(width,int(cx+radius+1));
            const auto ybr     = min<int>(height,int(cy+radius+1));
            const auto sigma   = (2*radius+1)/delta;
            const auto c_index = class_index-1;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    auto v = exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*k;
                    data(batch_index,y,x,c_index) = max(data(batch_index,y,x,c_index),v);
                }
            }
        }
        template<typename DT0,typename DT1>
        static void draw_gaussianv2(DT0& data0,DT1& data1,int cx,int cy,float radius,float h,float w,int batch_index,int class_index,float delta=6,float k=1.0)
        {
            const auto width   = data1.dimension(2);
            const auto height  = data1.dimension(1);
            const auto xtl     = max(0,int(cx-radius));
            const auto ytl     = max(0,int(cy-radius));
            const auto xbr     = min<int>(width,int(cx+radius+1));
            const auto ybr     = min<int>(height,int(cy+radius+1));
            const auto sigma   = (2 *radius+1)/delta;
            const auto c_index = class_index-1;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    auto v = exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*k;
                    if(data0(y,x)<v) {
                        data0(y,x) = v;    
                        data1(batch_index,y,x,0) = h;
                        data1(batch_index,y,x,1) = w;
                    }
                }
            }
        }
	private:
        int   num_classes_  = 80;
        float gaussian_iou_ = 0.7f;
};
REGISTER_KERNEL_BUILDER(Name("Center2BoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), Center2BoxesEncodeOp<CPUDevice, float>);

REGISTER_OP("Center2BoxesDecode")
    .Attr("T: {float,double,int32,int64}")
    .Attr("k:int")
    .Attr("threshold:float")
    .Input("heatmaps: T")
    .Input("offset: T")
    .Input("hw: T")
	.Output("output_bboxes:T")
	.Output("output_labels:int32")
	.Output("output_probs:T")
	.Output("output_index:int32")
	.Output("output_lens:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            int k;
            c->GetAttr("k",&k);
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            auto shape0 = c->MakeShape({batch_size,k,4});
            auto shape1 = c->MakeShape({batch_size,k});
            auto shape2 = c->MakeShape({batch_size});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape1);
			c->set_output(3, shape1);
			c->set_output(4, shape2);
			return Status::OK();
			});

template <typename Device,typename T>
class Center2BoxesDecodeOp: public OpKernel {
    private:
        struct InterData{
            InterData(int y,int x,int z,float s):y(y),x(x),z(z),score(s){}
            float score;
            int y,x,z;
            bool operator<(const InterData& rhv)const{
                return score<rhv.score;
            }
        };
        struct Box
        {
            float ymin;
            float xmin;
            float ymax;
            float xmax;
            float prob;
            int classes;
            int index;
        };
    public:
        explicit Center2BoxesDecodeOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
        }
        void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("Center2BoxesDecode");
            const Tensor &_heatmaps_c = context->input(0);
            const Tensor &_offset_c   = context->input(1);
            const Tensor &_hw         = context->input(2);

            OP_REQUIRES(context, _heatmaps_c.dims() == 4, errors::InvalidArgument("heatmap data must be 4-dimension"));
            OP_REQUIRES(context, _offset_c.dims() == 4, errors::InvalidArgument("offset data must be 4-dimension"));
            OP_REQUIRES(context, _hw.dims() == 4, errors::InvalidArgument("hw data must be 4-dimension"));

            auto          heatmaps_c_r  = _heatmaps_c.template tensor<T,4>();
            auto          offset_c    = _offset_c.template tensor<T,4>();
            auto          hw          = _hw.template tensor<T,4>();

            const auto batch_size = _heatmaps_c.dim_size(0);
            vector<vector<Box>> res_boxes;

            auto heatmaps_c = batch_sim_max_pool(heatmaps_c_r);

            for(auto i=0; i<batch_size; ++i) {
                auto c = get_top_k(heatmaps_c, i,k_);
                auto tboxes = get_boxes(c,i,hw,offset_c);
                res_boxes.push_back(std::move(tboxes));
            }

            auto         box_nr        = k_;
            int          dims_3d[3]    = {int(batch_size),box_nr,4};
            int          dims_2d[2]    = {int(batch_size),box_nr};
            int          dims_1d[1]    = {int(batch_size)};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_boxes  = NULL;
            Tensor      *output_labels = NULL;
            Tensor      *output_probs  = NULL;
            Tensor      *output_indexs = NULL;
            Tensor      *output_lens   = NULL;

            TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);
            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape2);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_probs));
            OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_indexs));
            OP_REQUIRES_OK(context, context->allocate_output(4, outshape2, &output_lens));

            auto o_boxes = output_boxes->template tensor<T,3>();
            auto o_labels = output_labels->template tensor<int,2>();
            auto o_probs = output_probs->template tensor<T,2>();
            auto o_indexs = output_indexs->template tensor<int,2>();
            auto o_lens = output_lens->template tensor<int,1>();

            o_boxes.setZero();
            o_labels.setZero();
            o_probs.setZero();
            o_lens.setZero();
            o_indexs.setZero();

            for(auto i=0; i<batch_size; ++i) {
                auto b_it = next(res_boxes.begin(),i);
                for(auto j=0; j<b_it->size(); ++j) {
                    auto& box = (*b_it)[j];
                    o_boxes(i,j,0) = box.ymin;
                    o_boxes(i,j,1) = box.xmin;
                    o_boxes(i,j,2) = box.ymax;
                    o_boxes(i,j,3) = box.xmax;
                    o_labels(i,j) = box.classes;
                    o_probs(i,j) = box.prob;
                    o_indexs(i,j) = box.index;
                }
                o_lens(i) = b_it->size();
            }
        }

        template<typename DT>
        Eigen::Tensor<T,4,Eigen::RowMajor> batch_sim_max_pool(DT& data,int k=3,float neg_value = 0.0f) {
            Eigen::Tensor<T,4,Eigen::RowMajor> res_data(Eigen::array<Eigen::Index,4>(data.dimensions()));
            for(auto i=0; i<data.dimension(0); ++i) {
                for(auto j=0; j<data.dimension(3); ++j) {
                    Eigen::Tensor<T,2,Eigen::RowMajor> ldata = data.chip(i,0).chip(j,2);
                    sim_max_pool(ldata,k,neg_value);
                    res_data.chip(i,0).chip(j,2) = ldata;
                }
            }
            return res_data;
        }
        template<typename DT>
        void sim_max_pool(DT& data,int k=3,float neg_value = 0.0f) {

            float buffer[k *k];
            int   nr;
            auto  H            = data.dimension(0);
            auto  W            = data.dimension(1);
            auto  max_i        = -1;
            auto  max_j        = -1;
            auto  max_v        = neg_value;

            for(auto i=0; i<H; ++i) {
                for(auto j=0; j<W; ++j) {
                    auto i_min = max<int>(0,i-k);
                    auto j_min = max<int>(0,j-k);
                    auto i_max = min<int>(H-1,i+k+1);
                    auto j_max = min<int>(W-1,j+k+1);

                    max_i = -1; 
                    max_v = neg_value;
                    for(auto ii=i_min; ii<i_max; ++ii) {
                        for(auto jj=j_min; jj<j_max; ++jj) {
                            if(data(ii,jj)>max_v) {
                                max_i = ii;
                                max_j = jj;
                                max_v = data(ii,jj);
                            }
                        }
                    }//end ii
                    if((max_i != i) || (max_j != j)) {
                        data(i,j) = neg_value;
                    }
                }
            }
        }

        template<typename DT>
            vector<InterData> get_top_k(const DT& heatmaps,int batch_index,int k) {
                const auto H = heatmaps.dimension(1);
                const auto W = heatmaps.dimension(2);
                const auto C = heatmaps.dimension(3);
                vector<InterData> res;
                res.reserve(H*W/4);
                for(auto y=0; y<H; ++y) {
                    for(auto x=0; x<W; ++x) {
                        for(auto z=0; z<C; ++z) {
                            auto score = heatmaps(batch_index,y,x,z);
                            if(score>threshold_)
                                res.emplace_back(y,x,z,heatmaps(batch_index,y,x,z));
                        }
                    }
                }
                auto mid = res.begin()+min<int>(k,res.size());
                partial_sort(res.begin(),mid,res.end(),[this](auto lhv,auto rhv){ return lhv.score>rhv.score;});
                res.erase(mid,res.end());
                return res;
            }

        template<typename DT0,typename DT1>
        vector<Box> get_boxes(const vector<InterData>& c,int batch_index,const DT0& HW,const DT1& offset) {
            vector<Box> boxes;
            const auto H = HW.dimension(1);
            const auto W = HW.dimension(2);

            for(auto& id:c) {
                auto cx = id.x+offset(batch_index,id.y,id.x,1);
                auto cy = id.y+offset(batch_index,id.y,id.x,0);
                auto hh = HW(batch_index,id.y,id.x,0)/2;
                auto hw = HW(batch_index,id.y,id.x,1)/2;
                Box box;
                box.xmin = (cx-hw)/(W-1);
                box.ymin = (cy-hh)/(H-1);
                box.xmax = (cx+hw)/(W-1);
                box.ymax = (cy+hh)/(H-1);
                box.prob = id.score;
                box.index = id.x+id.y*W;
                box.classes = id.z+1;
                boxes.push_back(box);
            }
            return boxes;
        }
        static inline pair<int,int> index_to_yx(int index,int H,int W) {
            return make_pair(index/W,index%W);
        }
	private:
        int k_ = 0;
        float threshold_ = 1e-3;
};
REGISTER_KERNEL_BUILDER(Name("Center2BoxesDecode").Device(DEVICE_CPU).TypeConstraint<float>("T"), Center2BoxesDecodeOp<CPUDevice, float>);
