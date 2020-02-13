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
REGISTER_OP("CenterBoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("max_box_nr:int")
	.Attr("num_classes:int")
	.Attr("gaussian_iou:float")
    .Input("gbboxes: T")
    .Input("glabels: int32")
    .Input("glength: int32")
    .Input("output_size: int32")
	.Output("output_heatmaps_tl:T")
	.Output("output_heatmaps_br:T")
	.Output("output_heatmaps_c:T")
	.Output("output_offset:T")
	.Output("output_tags:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            int num_classes;
            int max_box_nr;
            c->GetAttr("num_classes",&num_classes);
            c->GetAttr("max_box_nr",&max_box_nr);
            auto shape0 = c->MakeShape({batch_size,-1,-1,num_classes});
            auto shape1 = c->MakeShape({batch_size,max_box_nr,6});
            auto shape2 = c->MakeShape({batch_size,max_box_nr,3});

            for(auto i=0; i<3; ++i)
			    c->set_output(i, shape0);
			c->set_output(3, shape1);
			c->set_output(4, shape2);
			return Status::OK();
			});

template <typename Device,typename T>
class CenterBoxesEncodeOp: public OpKernel {
	public:
		explicit CenterBoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("max_box_nr", &max_box_nr_));
			OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
			OP_REQUIRES_OK(context, context->GetAttr("gaussian_iou", &gaussian_iou_));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("CenterBoxesEncode");
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

            int           dims_4d[4]            = {int(batch_size),output_size[0],output_size[1],num_classes_};
            int           dims_3d0[3]           = {int(batch_size),max_box_nr_,6};
            int           dims_3d1[3]           = {int(batch_size),max_box_nr_,3};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_heatmaps_tl = NULL;
            Tensor      *output_heatmaps_br = NULL;
            Tensor      *output_heatmaps_c  = NULL;
            Tensor      *output_tags        = NULL;
            Tensor      *output_offset      = NULL;

            TensorShapeUtils::MakeShape(dims_4d, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_3d0, 3, &outshape1);
            TensorShapeUtils::MakeShape(dims_3d1, 3, &outshape2);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_heatmaps_tl));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape0, &output_heatmaps_br));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape0, &output_heatmaps_c));
            OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_offset));
            OP_REQUIRES_OK(context, context->allocate_output(4, outshape2, &output_tags));

            auto heatmaps_tl = output_heatmaps_tl->template tensor<T,4>();
            auto heatmaps_br = output_heatmaps_br->template tensor<T,4>();
            auto heatmaps_c  = output_heatmaps_c->template tensor<T,4>();
            auto offsets     = output_offset->template tensor<T,3>();
            auto tags        = output_tags->template tensor<int,3>();
            tags.setZero();
            offsets.setZero();
            heatmaps_tl.setZero();
            heatmaps_br.setZero();
            heatmaps_c.setZero();
            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<gsize(i); ++j) {
                    const auto fytl = gbboxes(i,j,0)*(output_size[0]-1);
                    const auto fxtl = gbboxes(i,j,1)*(output_size[1]-1);
                    const auto fybr = gbboxes(i,j,2)*(output_size[0]-1);
                    const auto fxbr = gbboxes(i,j,3)*(output_size[1]-1);
                    const auto fyc = (fytl+fybr)/2;
                    const auto fxc = (fxtl+fxbr)/2;
                    const auto ytl = int(fytl+0.5);
                    const auto xtl = int(fxtl+0.5);
                    const auto ybr = int(fybr+0.5);
                    const auto xbr = int(fxbr+0.5);
                    const auto yc = int(fyc+0.5);
                    const auto xc = int(fxc+0.5);
                    const auto r0 = get_gaussian_radius(fybr-fytl,fxbr-fxtl,gaussian_iou_);
                    const auto r1 = min<float>(r0,min(fybr-fytl,fxbr-fxtl)/2.0);
                    const auto label = glabels(i,j);
                    draw_gaussian(heatmaps_tl,xtl,ytl,r0,i,label);
                    draw_gaussian(heatmaps_br,xbr,ybr,r0,i,label);
                    draw_gaussian(heatmaps_c,fxc,fyc,r1,i,label);
                    offsets(i,j,0) = fytl-ytl;
                    offsets(i,j,1) = fxtl-xtl;
                    offsets(i,j,2) = fybr-ybr;
                    offsets(i,j,3) = fxbr-xbr;
                    offsets(i,j,4) = fyc-yc;
                    offsets(i,j,5) = fxc-xc;
                    tags(i,j,0) = ytl*output_size[1]+xtl;
                    tags(i,j,1) = ybr*output_size[1]+xbr;
                    tags(i,j,2) = yc*output_size[1]+xc;
                }
            }
        }
        template<typename DT>
        static void draw_gaussian(DT& data,int cx,int cy,float radius,int batch_index,int class_index,float k=1.0)
        {
            const auto width  = data.dimension(2);
            const auto height = data.dimension(1);
            const auto xtl    = max(0,int(cx-radius));
            const auto ytl    = max(0,int(cy-radius));
            const auto xbr    = min<int>(width,int(cx+radius+1));
            const auto ybr    = min<int>(height,int(cy+radius+1));
            const auto sigma  = radius/3;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    auto v = exp(-(dx*dx+dy*dy)/(2*sigma*sigma))*k;
                    data(batch_index,y,x,class_index) = max(data(batch_index,y,x,class_index),v);
                }
            }
        }
	private:
        int max_box_nr_;
        int num_classes_;
        float gaussian_iou_ = 0.7f;
};
REGISTER_KERNEL_BUILDER(Name("CenterBoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), CenterBoxesEncodeOp<CPUDevice, float>);

REGISTER_OP("CenterBoxesDecode")
    .Attr("T: {float,double,int32,int64}")
    .Attr("k:int")
    .Input("heatmaps_tl: T")
    .Input("heatmaps_br: T")
    .Input("heatmaps_c: T")
    .Input("offset_tl: T")
    .Input("offset_br: T")
    .Input("offset_c: T")
	.Output("output_bboxes:T")
	.Output("output_labels:int32")
	.Output("output_probs:T")
	.Output("output_index:int32")
	.Output("output_lens:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            auto shape0 = c->MakeShape({batch_size,-1,4});
            auto shape1 = c->MakeShape({batch_size,-1});
            auto shape2 = c->MakeShape({batch_size});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape1);
			c->set_output(3, shape1);
			c->set_output(4, shape2);
			return Status::OK();
			});

template <typename Device,typename T>
class CenterBoxesDecodeOp: public OpKernel {
    private:
        struct InterData{
            InterData(int i,float s):index(i),score(s){}
            int index;
            float score;
            float y;
            float x;
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
            int index;
        };
    public:
        explicit CenterBoxesDecodeOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
        }
        void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("CenterBoxesDecode");
            const Tensor &_heatmaps_tl = context->input(0);
            const Tensor &_heatmaps_br = context->input(1);
            const Tensor &_heatmaps_c  = context->input(2);
            const Tensor &_offset_tl   = context->input(3);
            const Tensor &_offset_br   = context->input(4);
            const Tensor &_offset_c    = context->input(5);
            auto          heatmaps_tl  = _heatmaps_tl.template tensor<T,4>();
            auto          heatmaps_br  = _heatmaps_br.template tensor<T,4>();
            auto          heatmaps_c   = _heatmaps_c.template tensor<T,4>();
            auto          offset_tl    = _offset_tl.template tensor<T,4>();
            auto          offset_br    = _offset_br.template tensor<T,4>();
            auto          offset_c     = _offset_c.template tensor<T,4>();

            const auto    batch_size  = _heatmaps_tl.dim_size(0);
            const auto num_classes = _heatmaps_tl.dim_size(3);
            const auto dw = 1.0/(heatmaps_tl.dimension(2)-1);
            const auto dh = 1.0/(heatmaps_tl.dimension(1)-1);
            const auto H = heatmaps_tl.dimension(1);
            const auto W = heatmaps_tl.dimension(2);

            OP_REQUIRES(context, _heatmaps_tl.dims() == 4, errors::InvalidArgument("heatmap data must be 4-dimension"));
            OP_REQUIRES(context, _heatmaps_br.dims() == 4, errors::InvalidArgument("heatmap data must be 4-dimension"));
            OP_REQUIRES(context, _heatmaps_c.dims() == 4, errors::InvalidArgument("heatmap data must be 4-dimension"));
            OP_REQUIRES(context, _offset_tl.dims() == 4, errors::InvalidArgument("offset data must be 4-dimension"));
            OP_REQUIRES(context, _offset_br.dims() == 4, errors::InvalidArgument("offset data must be 4-dimension"));
            OP_REQUIRES(context, _offset_c.dims() == 4, errors::InvalidArgument("offset data must be 4-dimension"));

            list<vector<Box>> res_boxes;
            list<vector<int>> res_labels;
            for(auto i=0; i<batch_size; ++i) {
                vector<Box> boxes;
                vector<int> labels;
                for(auto j=0; j<num_classes; ++j) {
                    vector<InterData> tl;
                    vector<InterData> br;
                    vector<InterData> c;
                    tl.reserve(k_);
                    br.reserve(k_);
                    c.reserve(k_);
                    tl = get_top_k(heatmaps_tl,i,j,k_);
                    br = get_top_k(heatmaps_br,i,j,k_);
                    c  = get_top_k(heatmaps_c,i,j,k_);
                    fill_pos(offset_tl,tl,i,j);
                    fill_pos(offset_br,br,i,j);
                    fill_pos(offset_c,c,i,j);
                    auto tboxes = get_boxes(tl,br,c);
                    vector<int> tlabels(tboxes.size(),j);
                    boxes.insert(boxes.end(),tboxes.begin(),tboxes.end());
                    labels.insert(labels.end(),tlabels.begin(),tlabels.end());
                }
                res_boxes.push_back(boxes);
                res_labels.push_back(labels);
            }
            auto box_nr = max_element(res_boxes.begin(),res_boxes.end(),[](const auto& lhv,const auto& rhv){ return lhv.size()<rhv.size();})->size();
            int           dims_3d[3]            = {int(batch_size),box_nr,4};
            int           dims_2d[2]           = {int(batch_size),box_nr};
            int           dims_1d[1]           = {int(batch_size)};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_boxes = NULL;
            Tensor      *output_labels = NULL;
            Tensor      *output_probs = NULL;
            Tensor      *output_indexs = NULL;
            Tensor      *output_lens = NULL;

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
            o_lens.setZero();

            for(auto i=0; i<batch_size; ++i) {
                auto b_it = next(res_boxes.begin(),i);
                auto l_it = next(res_labels.begin(),i);
                for(auto j=0; j<b_it->size(); ++j) {
                    auto& box = (*b_it)[j];
                    o_boxes(i,j,0) = box.ymin;
                    o_boxes(i,j,1) = box.xmin;
                    o_boxes(i,j,2) = box.ymax;
                    o_boxes(i,j,3) = box.xmax;
                    o_labels(i,j) = (*l_it)[j];
                    o_probs(i,j) = box.prob;
                    o_indexs(i,j) = box.index;
                }
                o_lens(i) = b_it->size();
            }
        }
        template<typename DT>
            vector<InterData> get_top_k(const DT& heatmaps,int batch_index,int class_index,int k) {
                const auto H = heatmaps.dimension(1);
                const auto W = heatmaps.dimension(2);
                vector<InterData> res;
                res.reserve(H*W);
                for(auto y=0; y<H; ++y) {
                    for(auto x=0; x<W; ++x) {
                        res.emplace_back(y*W+x,heatmaps(batch_index,y,x,class_index));
                    }
                }
                auto it = partition(res.begin(),res.end(),[this](auto v){ return v.score>threshold_;});
                res.erase(it,res.end());
                auto mid = res.begin()+std::min<int>(k,res.size());
                partial_sort(res.begin(),mid,res.end(),[](const auto& lhv,const auto& rhv){ return lhv.score>rhv.score;});
                res.erase(mid,res.end());
                return res;
            }
        template<typename DT>
            void fill_pos(const DT& offsets,vector<InterData>& data,int batch_index,int class_index) {
                const auto H = offsets.dimension(1);
                const auto W = offsets.dimension(2);
                const auto dw = 1.0/(H-1);
                const auto dh = 1.0/(W-1);
                int y;
                int x;

                for(auto&d :data) {
                    std::tie(y,x) = index_to_yx(d.index,H,W);
                    auto yoffset = offsets(batch_index,y,x,0)*dh;
                    auto xoffset = offsets(batch_index,y,x,1)*dw;
                    d.y = float(y)/(H-1)+yoffset;
                    d.x = float(x)/(W-1)+xoffset;
                }
            }
        vector<Box> get_boxes(const vector<InterData>& tl,const vector<InterData>& br,const vector<InterData>& c) {
            const auto nr_i = tl.size();
            const auto nr_j = br.size();
            const auto nr_k = c.size();
            Eigen::Tensor<float,2,Eigen::RowMajor> scores(nr_i,nr_j);
            Eigen::Tensor<int,2,Eigen::RowMajor> c_index(nr_i,nr_j);

            scores.setConstant(-1.0);
            c_index.setZero();

            for(auto i=0; i<nr_i; ++i) {
                const auto& tl_d = tl[i];
                for(auto j=0; j<nr_j; ++j) {
                    const auto& br_d = br[j];
                    float box[4] = {tl_d.y,tl_d.x,br_d.y,br_d.x};

                    if((box[0]>=box[2]) || (box[1]>=box[3])) 
                        continue;

                    for(auto k=0; k<nr_k; ++k) {
                        const auto& c_d = c[k];
                        if(is_in_box_center(box,c_d.y,c_d.x,1)) {
                            auto score = 1.0 - (box[2]-box[0])*(box[3]-box[1]);
                            if(score>scores(i,j)) {
                                scores(i,j) = score;
                                c_index(i,j) = k;
                            }
                        }
                    }
                }
            }
            vector<Box> boxs;
            const auto loop_nr = min({nr_i,nr_j,nr_k});
            for(auto x=0; x<loop_nr; ++x) {
                auto max_i = 0;
                auto max_j = 0;
                auto max_score = -1.0;
                for(auto i=0; i<nr_i; ++i) {
                    for(auto j=0; j<nr_j; ++j) {
                        if(scores(i,j)>max_score) {
                            max_score = scores(i,j);
                            max_i = i;
                            max_j = j;
                        }
                    }
                }
                if(max_score<0) break;
                Box box;
                box.ymin = tl[max_i].y;
                box.xmin = tl[max_i].x;
                box.ymax = br[max_j].y;
                box.xmax = br[max_j].x;
                const auto k = c_index(max_i,max_j);
                box.index = c[k].index;
                box.prob = (tl[max_i].score+br[max_j].score+c[k].score)/3.0f;
                boxs.push_back(box);
                for(auto j=0; j<nr_j; ++j)
                    scores(max_i,j) = -1.0;
                for(auto i=0; i<nr_i; ++i)
                    scores(i,max_j) = -1.0;
            }
            return boxs;
        }
       static inline bool is_in_box_center(const float* box,float y,float x,int grid_size=3) {
           const auto h = box[2]-box[0];
           const auto w = box[3]-box[1];
           const auto h_grid_size = float(int(grid_size/2))/grid_size;
           if((y<box[0]+h*h_grid_size)||
                   (y>box[2]-h*h_grid_size)||
                   (x<box[1]+w*h_grid_size) ||
                   (x>box[3]-w*h_grid_size))
               return false;
           return true;
       }
        static inline pair<int,int> index_to_yx(int index,int H,int W) {
            return make_pair(index/W,index%W);
        }
	private:
        int k_ = 0;
        float threshold_ = 1e-3;
};
REGISTER_KERNEL_BUILDER(Name("CenterBoxesDecode").Device(DEVICE_CPU).TypeConstraint<float>("T"), CenterBoxesDecodeOp<CPUDevice, float>);
