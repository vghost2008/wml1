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
 * input:
 * min_size: pos box的最小值
 * max_size: pos box的最大值, FCOS通过这两个值将box分配到不同的层
 * fm: (H,W) 预测层的feature map, 主要用于确定输出大小及stride, 目前没有使用相应的值
 *  gbboxes: [B,box_nr,4] relative coordinate
 * glabels: [B,box_nr]
 * glength: [B]
 * img_size:[img_H,img_W]
 * output:
 * regression: [B,H,W,4] (t,l,b,r) labsolute coordinate
 * center_ness: [B,H,W]
 * gt_boxes: [B,H,W,4] (ymin,xmin,ymax,xmax) absolute coordinate, 
 * classes: [B,H,W] ground truth classes, -1 indict neg samples
 */
REGISTER_OP("FcosBoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("min_size:float")
	.Attr("max_size:float")
    .Input("fm_shape: int32")
    .Input("gbboxes: T")
    .Input("glabels: int32")
    .Input("glength: int32")
    .Input("img_size: int32")
	.Output("regression:T")
	.Output("center_ness:T")
	.Output("gt_boxes:T")
	.Output("classes:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(1);
            const auto batch_size = c->Dim(input_shape0,0);
            const auto H  = -1;
            const auto W  = -1;

            auto shape0 = c->MakeShape({batch_size,H,W,4});
            auto shape1 = c->MakeShape({batch_size,H,W});
            auto shape2 = c->MakeShape({batch_size,H,W});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape0);
			c->set_output(3, shape2);
			return Status::OK();
			});

template <typename Device,typename T>
class FcosBoxesEncodeOp: public OpKernel {
	public:
		explicit FcosBoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("min_size", &min_size_));
			OP_REQUIRES_OK(context, context->GetAttr("max_size", &max_size_));

            if(max_size_<0)
                max_size_ = 1e20;
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("FcosBoxesEncode");
            const auto    _fm_shape  = context->input(0).template flat<int>().data();
            const Tensor &_gbboxes   = context->input(1);
            const Tensor &_glabels   = context->input(2);
            const Tensor &_glength   = context->input(3);
            const auto    img_size   = context->input(4).template flat<int>().data();
            auto          gbboxes    = _gbboxes.template tensor<T,3>();
            auto          glabels    = _glabels.template tensor<int,2>();
            auto          glength    = _glength.template tensor<int,1>();
            const auto    batch_size = _gbboxes.dim_size(0);
            const auto    H          = _fm_shape[0];
            const auto    W          = _fm_shape[1];
            const auto    img_H      = img_size[0];
            const auto    img_W      = img_size[1];
            const auto    x_delta    = float(img_W)/W;
            const auto    y_delta    = float(img_H)/H;

            OP_REQUIRES(context, _gbboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimension"));
            OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimension"));
            OP_REQUIRES(context, _glength.dims() == 1, errors::InvalidArgument("glength data must be 1-dimension"));

            int           dims_4d0[4]            = {int(batch_size),H,W,4};
            int           dims_3d0[3]           = {int(batch_size),H,W};
            int           dims_3d1[3]           = {int(batch_size),H,W};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            Tensor      *output_regression  = NULL;
            Tensor      *output_center_ness = NULL;
            Tensor      *output_gt_boxes    = NULL;
            Tensor      *output_classes     = NULL;

            TensorShapeUtils::MakeShape(dims_4d0, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_3d0, 3, &outshape1);
            TensorShapeUtils::MakeShape(dims_3d1, 3, &outshape2);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_regression));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_center_ness));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape0, &output_gt_boxes));
            OP_REQUIRES_OK(context, context->allocate_output(3, outshape2, &output_classes));

            auto regression_tensor = output_regression->template tensor<T,4>();
            auto center_ness_tensor = output_center_ness->template tensor<T,3>();
            auto gt_bboxes   = output_gt_boxes->template tensor<T,4>();
            auto classes_tensor = output_classes->template tensor<int,3>();
            Eigen::Tensor<float,2>  box_size(H,W);

            
            
            classes_tensor.setConstant(-1);
            center_ness_tensor.setZero();
            regression_tensor.setZero();
            gt_bboxes.setZero();

            for(auto i=0; i<batch_size; ++i) {
                box_size.setConstant(1e8);
                for(auto j=0; j<glength(i); ++j) {
                    const auto beg_yf  = gbboxes(i,j,0) *(img_H-1);
                    const auto beg_xf  = gbboxes(i,j,1) *(img_W-1);
                    const auto end_yf  = gbboxes(i,j,2) *(img_H-1);
                    const auto end_xf  = gbboxes(i,j,3) *(img_W-1);
                    const int  beg_y   = gbboxes(i,j,0) *(H-1)+0.5;
                    const int  beg_x   = gbboxes(i,j,1) *(W-1)+0.5;
                    const int  end_y   = gbboxes(i,j,2) *H+0.5;
                    const int  end_x   = gbboxes(i,j,3) *W+0.5;
                    const auto classes = glabels(i,j);
                    /*const auto max_loc = std::max(end_yf-beg_yf,end_xf-beg_xf);

                    if((max_loc<min_size_) 
                       || (max_loc>max_size_)) 
                        continue;*/

                    for(auto k=beg_x; k<end_x; ++k) {
                        for(auto l=beg_y; l<end_y; ++l) {
                            const auto l_dis = -beg_xf+(k+0.5)*x_delta;
                            const auto r_dis = end_xf-(k+0.5)*x_delta;
                            const auto t_dis = -beg_yf+(l+0.5)*y_delta;
                            const auto b_dis = end_yf-(l+0.5)*y_delta;
                            const auto max_loc = std::max({l_dis,r_dis,t_dis,b_dis});

                            if((max_loc>box_size(l,k)) 
                                    || (max_loc<min_size_) 
                                    || (max_loc>max_size_) 
                                    || (std::min({r_dis,b_dis,l_dis,t_dis})<0)) {
                                continue;
                            }

                            classes_tensor(i,l,k) = classes;

                            center_ness_tensor(i,l,k) = sqrt(std::min(l_dis,r_dis)*std::min(t_dis,b_dis)/(std::max(l_dis,r_dis)*std::max(b_dis,t_dis)+1e-8));

                            regression_tensor(i,l,k,0) = t_dis;
                            regression_tensor(i,l,k,1) = l_dis;
                            regression_tensor(i,l,k,2) = b_dis;
                            regression_tensor(i,l,k,3) = r_dis;

                            gt_bboxes(i,l,k,0) = beg_yf;
                            gt_bboxes(i,l,k,1) = beg_xf;
                            gt_bboxes(i,l,k,2) = end_yf;
                            gt_bboxes(i,l,k,3) = end_xf;

                            box_size(l,k) = max_loc;
                        }
                    }
                }
            }
        }
	private:
        float min_size_;
        float max_size_;
};
REGISTER_KERNEL_BUILDER(Name("FcosBoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), FcosBoxesEncodeOp<CPUDevice, float>);
