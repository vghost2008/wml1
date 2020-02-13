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
#include "matcher.h"
#include "wtoolkit.h"
#include "wtoolkit_cuda.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

/*
 * max_overlap_as_pos:是否将与ground truth bbox交叉面积最大的box设置为正样本
 * bottom_boxes:[1,X,4]/[batch_size,X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[batch_size,Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[batch_size,Y] 0为背景
 * bottom_glength:[batch_size] 为每一个batch中gboxes的有效数量
 * output_labels:[batch_size,X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分,-1表示忽略
 * output_scores:[batch_size,X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 * output_indict:[batch_size,X], 当anchorbox有效时，与它对应的gboxes(从0开始)序号,无效时为-1
 */
REGISTER_OP("Matcher")
    .Attr("T: {float,double,int32,int64}")
	.Attr("pos_threshold:float")
	.Attr("neg_threshold:float")
    .Attr("max_overlap_as_pos:bool")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
    .Input("bottom_glength: int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("indict:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto input_shape1 = c->input(1);
            const auto batch_size = c->Dim(input_shape1,0);
            const auto boxes_nr  = c->Dim(input_shape0,1);
            auto shape = c->MakeShape({batch_size,boxes_nr});

            for(auto i=0; i<3; ++i)
			    c->set_output(i, shape);
			return Status::OK();
			});

template <typename Device, typename T>
class MatcherOp: public OpKernel {
};
template <typename T>
class MatcherOp<CPUDevice,T>: public OpKernel {
	public:
		explicit MatcherOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("Matcher");
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_glabels = context->input(2);
			const Tensor &_bottom_gsize   = context->input(3);
			auto          bottom_boxes    = _bottom_boxes.template tensor<T,3>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,3>();
			auto          bottom_glabels  = _bottom_glabels.template tensor<int,2>();
			auto          bottom_gsize    = _bottom_gsize.template tensor<int,1>();
			const auto    batch_size      = _bottom_gboxes.dim_size(0);
			const auto    data_nr         = _bottom_boxes.dim_size(1);

			OP_REQUIRES(context, _bottom_boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimensional"));

			int           dims_2d[2]            = {int(batch_size),int(data_nr)};
			TensorShape   outshape1;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_indict         = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);


			OP_REQUIRES_OK(context, context->allocate_output(0, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_indict));

			auto output_labels_tensor         =  output_labels->template tensor<int,2>();
			auto output_scores_tensor         =  output_scores->template tensor<T,2>();
			auto output_indict_tensor         =  output_indict->template tensor<int,2>();

            MatcherUnit<CPUDevice,T> encode_unit(pos_threshold,neg_threshold,max_overlap_as_pos_);
            auto shard = [&](int64 start,int64 limit){
                for(auto i=start; i<limit; ++i) {

                    auto size     = bottom_gsize(i);
                    auto boxes    = bottom_boxes.chip(bottom_boxes.dimension(0)==batch_size?i:0,0);
                    auto _gboxes  = bottom_gboxes.chip(i,0);
                    auto _glabels = bottom_glabels.chip(i,0);
                    Eigen::array<long,2> offset={0,0};
                    Eigen::array<long,2> extents={size,4};
                    Eigen::array<long,1> offset1={0};
                    Eigen::array<long,1> extents1={size};
                    auto gboxes             = _gboxes.slice(offset,extents);
                    auto glabels            = _glabels.slice(offset1,extents1);
                    auto out_labels         = output_labels_tensor.chip(i,0);
                    auto out_scores         = output_scores_tensor.chip(i,0);
                    auto out_indices        = output_indict_tensor.chip(i,0);
                    auto res                = encode_unit(boxes,gboxes,glabels);

                    out_labels          =  std::get<0>(res);
                    out_scores          =  std::get<1>(res);
                    out_indices         =  std::get<2>(res);
                }
            };
            list<future<void>> results;
            for(auto i=0; i<batch_size; ++i) {
                results.emplace_back(async(launch::async,[i,&shard](){ shard(i,i+1);}));
            }
        }
	private:
		float         pos_threshold;
		float         neg_threshold;
        bool          max_overlap_as_pos_ = true;
};
#ifdef GOOGLE_CUDA
template <typename T>
class MatcherOp<GPUDevice,T>: public OpKernel {
	public:
		explicit MatcherOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("MatcherGPU");
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_glabels = context->input(2);
			const Tensor &_bottom_gsize   = context->input(3);
			auto          bottom_boxes    = _bottom_boxes.template tensor<T,3>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,3>();
			auto          bottom_glabels  = _bottom_glabels.template tensor<int,2>();
			auto          d_bottom_gsize  = _bottom_gsize.template tensor<int,1>();
			const auto    batch_size      = _bottom_gboxes.dim_size(0);
			const auto    data_nr         = _bottom_boxes.dim_size(1);
		    Eigen::Tensor<int,1,Eigen::RowMajor> bottom_gsize;
            assign_tensor<CPUDevice,GPUDevice>(bottom_gsize,d_bottom_gsize);

			OP_REQUIRES(context, _bottom_boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimensional"));

			int           dims_2d[2]            = {int(batch_size),int(data_nr)};
			TensorShape   outshape1;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_indict         = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_indict));

			auto output_labels_tensor         =  output_labels->template tensor<int,2>();
			auto output_scores_tensor         =  output_scores->template tensor<T,2>();
			auto output_indict_tensor         =  output_indict->template tensor<int,2>();

            MatcherUnit<GPUDevice,T> encode_unit(pos_threshold,neg_threshold,max_overlap_as_pos_);
            for(auto i=0; i<batch_size; ++i) {
                    auto size    = bottom_gsize(i);
                    auto boxes   = bottom_boxes.dimension(0)==batch_size?chip_data(bottom_boxes,i):chip_data(bottom_boxes,0);
                    auto gboxes  = chip_data(bottom_gboxes,i);
                    auto glabels = chip_data(bottom_glabels,i);
                    encode_unit(boxes,gboxes,glabels,
                            chip_data(output_labels_tensor,i),
                            chip_data(output_scores_tensor,i),
                            chip_data(output_indict_tensor,i),
                            size,bottom_boxes.dimension(1)
                            );
                }
        }
	private:
		float         pos_threshold;
		float         neg_threshold;
        bool          max_overlap_as_pos_ = true;
};
#endif
REGISTER_KERNEL_BUILDER(Name("Matcher").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatcherOp<CPUDevice, float>);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Matcher").Device(DEVICE_GPU).TypeConstraint<float>("T"), MatcherOp<GPUDevice, float>);
#endif
