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
#include "open_pose_decode_imp.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
/*
 * gaussian_delta: 一般为2
 * keypoints: [B,N,num_points_nr,2] 相对坐标,x,y
 * glength: 有效的groundtruth instance数量
 * output_size: 输出图的大小[2]=(OH,OW) 
 *
 * output:
 * output_conf_map: heatmaps [B,OH,OW,num_points_nr]
 * output_indexs: [B,N,num_points_nr] //-1表示点无效
 */
REGISTER_OP("HrNetPe")
    .Attr("T: {float,double,int32,int64}")
	.Attr("gaussian_delta:float=8.0")
    .Input("keypoints: T")
    .Input("output_size: int32")
    .Input("glength: int32")
	.Output("output_conf_map:T")
	.Output("output_indexs:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto N = c->Value(c->Dim(input_shape0,1));
            const auto points_nr = c->Value(c->Dim(input_shape0,2));
            const auto batch_size = c->Dim(input_shape0,0);
            auto shape0 = c->MakeShape({batch_size,-1,-1,points_nr});
            auto shape1 = c->MakeShape({batch_size,N,points_nr});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			return Status::OK();
			});

template <typename Device,typename T>
class HRNetPEOp: public OpKernel {
	public:
		explicit HRNetPEOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("gaussian_delta", &gaussian_delta_));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("HRNetPE");
            const Tensor &_keypoints = context->input(0);
            const Tensor &_gsize      = context->input(2);
            auto          output_size = context->input(1).template flat<int>().data();
            const auto    batch_size  = _keypoints.dim_size(0);
            const auto num_keypoints = _keypoints.dim_size(2);

            OP_REQUIRES(context, _keypoints.dims() == 4, errors::InvalidArgument("keypoints data must be 4-dimension"));
            OP_REQUIRES(context, _gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimension"));

            auto         keypoints       = _keypoints.template tensor<T,4>();
            auto         gsize           = _gsize.template tensor<int,1>();
            int          dims_4d0[4]     = {int(batch_size),output_size[0],output_size[1],num_keypoints};
            int          dims_3d0[3]     = {int(batch_size),_keypoints.dim_size(1),num_keypoints};
            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_conf_map = NULL;
            Tensor      *output_indexs = NULL;
            const auto   max_data_nr     = _keypoints.dim_size(1);

            TensorShapeUtils::MakeShape(dims_4d0, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_3d0, 3, &outshape1);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_conf_map));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_indexs));

            auto heatmaps_conf = output_conf_map->template tensor<T,4>();
            auto heatmaps_indexs = output_indexs->template tensor<int,3>();
            const auto block_size = output_size[0]*output_size[1];

            heatmaps_conf.setZero();
            heatmaps_indexs.setConstant(-1);


            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<gsize(i); ++j) {
                    for(auto k=0; k<num_keypoints; ++k) {
                        const auto x0 = keypoints(i,j,k,0)*(output_size[1]-1);
                        const auto y0 = keypoints(i,j,k,1)*(output_size[0]-1);

						if((x0>=0) && (y0>=0)) {
                        	draw_gaussian(heatmaps_conf,i,x0,y0,k,gaussian_delta_);
                            const auto ix0 = int(x0+0.5);
                            const auto iy0 = int(y0+0.5);
                            const auto index = ix0*num_keypoints+iy0*output_size[1]*num_keypoints+k;
                            heatmaps_indexs(i,j,k) = index;
                        }
                    }
                }
            }
        }

        template<typename DT>
        static void draw_gaussian(DT& data,int batch_index,float cx,float cy,int k,float radius)
        {
            const auto th           = 4.6052;
            const auto spread_range = radius *sqrt(2*th);
            const auto width        = data.dimension(2);
            const auto height       = data.dimension(1);
            const auto xtl          = max(0,int(cx-spread_range));
            const auto ytl          = max(0,int(cy-spread_range));
            const auto xbr          = min<int>(width,int(cx+spread_range+1));
            const auto ybr          = min<int>(height,int(cy+spread_range+1));
            const auto sigma_p      = 2 *radius*radius;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    const auto d = (dx*dx+dy*dy);
                    const auto expv = d/sigma_p;
                    if (expv>th)
                        continue;
                    auto v = exp(-expv);
                    data(batch_index,y,x,k) = max(data(batch_index,y,x,k),v);
                }
            }
        }
	private:
       float       gaussian_delta_ = 2;
};
REGISTER_KERNEL_BUILDER(Name("HrNetPe").Device(DEVICE_CPU).TypeConstraint<float>("T"), HRNetPEOp<CPUDevice, float>);

/*
 * ans: [B,N,num_keypoints,3+tag_C]
 * det: [B,H,W,num_keypoints]
 * tag: [B,H,W,num_keypoints,tag_C]
 * output:
 * output_ans: [B,N,num_keypoints,3+tag_C]
 */
REGISTER_OP("HrNetRefine")
    .Attr("T: {float,double,int32,int64}")
    .Input("ans: T")
    .Input("det: T")
    .Input("tag: T")
    .Output("output_ans: T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto shape0 = c->input(0);
			c->set_output(0, shape0);
			return Status::OK();
			});

template <typename Device,typename T>
class HRNetRefine: public OpKernel {
    private:
        using Tensor1d_t = Eigen::Tensor<float,1,Eigen::RowMajor>;
        using Tensor2d_t = Eigen::Tensor<float,2,Eigen::RowMajor>;
        using Tensor3d_t = Eigen::Tensor<float,3,Eigen::RowMajor>;
        using Tensor4d_t = Eigen::Tensor<float,4,Eigen::RowMajor>;
	public:
		explicit HRNetRefine(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("HRNetPE");
            const Tensor &_ans = context->input(0);
            const Tensor &_det = context->input(1);
            const Tensor &_tag = context->input(2);

            OP_REQUIRES(context, _ans.dims() == 4, errors::InvalidArgument("ans data must be 4-dimension"));
            OP_REQUIRES(context, _det.dims() == 4, errors::InvalidArgument("dete data must be 4-dimension"));
            OP_REQUIRES(context, _tag.dims() == 5, errors::InvalidArgument("tag data must be 5-dimension"));

            const auto  batch_size    = _ans.dim_size(0);
            const auto  num_keypoints = _ans.dim_size(2);
            auto        ans           = _ans.template tensor<T,4>();
            auto        det           = _det.template tensor<T,4>();
            auto        tag           = _tag.template tensor<T,5>();
            Tensor     *output_ans    = NULL;


            OP_REQUIRES_OK(context, context->allocate_output(0, _ans.shape(), &output_ans));

            auto o_ans = output_ans->template tensor<T,4>();

            o_ans.setZero();

            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<_ans.dim_size(1); ++j) {
                    auto n_ans = refine(ans.chip(i,0).chip(j,0),det.chip(i,0),tag.chip(i,0));
                    o_ans.chip(i,0).chip(j,0) = n_ans;
                }
            }
        }

        Tensor2d_t refine(const Tensor2d_t& keypoints,const Tensor3d_t& det,const Tensor4d_t& tag)
        {
            Tensor2d_t res_keypoints = keypoints;
            const auto num_keypoints = keypoints.dimension(0);
            const auto tag_C = keypoints.dimension(1)-3;
            Tensor2d_t tags(num_keypoints,tag_C);
            const auto H = det.dimension(0);
            const auto W = det.dimension(1);
            float tags_nr = 1e-3;

            tags.setZero();
            for(auto i=0; i<num_keypoints; ++i) {
                if(keypoints(i,2)>0.01) {
                    const int x = keypoints(i,0);
                    const int y = keypoints(i,1);

                    tags.chip(i,0) = tag.chip(y,0).chip(x,0).chip(i,0);
                    tags_nr += 1;
                }
            }

            if(tags_nr<1)
                return res_keypoints;

            Tensor1d_t _tags_mean0 = tags.sum(Eigen::array<int,1>({0}));
            Tensor1d_t _tags_mean = _tags_mean0/_tags_mean0.constant(tags_nr);
            Eigen::array<int,3> three_dims{{1, 1, tag_C}};
            Tensor3d_t _prev_tag = _tags_mean.reshape(three_dims);
            Eigen::array<int, 3> bcast({det.dimension(0), det.dimension(1),1});
            Tensor3d_t prev_tag = _prev_tag.broadcast(bcast);
            vector<tuple<float,float,float>> ans;

            for(auto i=0; i<num_keypoints; ++i) {
                Tensor2d_t tmp_det = det.chip(i,2);
                Tensor2d_t _tt0;

                if(tag.dimension(2)>1) {
                    _tt0 = (tag.chip(i,2)-prev_tag).square().sum(Eigen::array<int,1>({2})).pow(0.5);
                } else {
                    _tt0 = (tag.chip(i,2)-prev_tag).abs().reshape(Eigen::array<int,2>({det.dimension(0),det.dimension(1)}));
                }

                Tensor2d_t _tt = _tt0+_tt0.constant(0.5);
                Tensor2d_t tt = _tt.cast<int>().cast<float>();
                Tensor2d_t tmp_det2 = tmp_det-tt*tt.constant(100);
                int xx,yy;
                float y,x;

                tie(yy,xx) = argmax(tmp_det2);

                const auto val = tmp_det(yy,xx);

                y = yy;
                x = xx;
                
                x += 0.5;
                y += 0.5;

                if(tmp_det(yy,min<int>(xx+1,W-1))>tmp_det(yy,max<int>(xx-1,0))) {
                    x += 0.25;
                } else {
                    x -= 0.25;
                }
                if(tmp_det(min<int>(yy+1,H-1),xx)>tmp_det(max<int>(yy-1,0),xx)) {
                    y += 0.25;
                } else {
                    y -= 0.25;
                }
                ans.emplace_back(make_tuple(x,y,val));
            }
            for(auto i=0; i<num_keypoints; ++i) {
                if((std::get<2>(ans[i])>0) && (keypoints(i,2)<=0)) {
                    res_keypoints(i,0) = std::get<0>(ans[i]);
                    res_keypoints(i,1) = std::get<1>(ans[i]);
                    res_keypoints(i,2) = std::get<2>(ans[i]);
                }
            }
            return res_keypoints;
        }

        pair<int,int> argmax(const Tensor2d_t& tensor) {
            int row = 0;
            int col = 0;
            float max_val = tensor(0,0);
            for(auto i=0; i<tensor.dimension(0); ++i) {
                for(auto j=0; j<tensor.dimension(1); ++j) {
                    if(tensor(i,j)>max_val) {
                        row = i;
                        col = j;
                        max_val = tensor(i,j);
                    }
                }
            }
            return make_pair(row,col);
        }
};
REGISTER_KERNEL_BUILDER(Name("HrNetRefine").Device(DEVICE_CPU).TypeConstraint<float>("T"), HRNetRefine<CPUDevice, float>);
