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
REGISTER_OP("RightPool")
    .Attr("T: {float,double}")
    .Input("tensor: T")
	.Output("output_tensor:T")
	.SetShapeFn(shape_inference::UnchangedShape);
REGISTER_OP("RightPoolGrad")
    .Attr("T: {float,double}")
    .Input("tensor: T")
    .Input("fw_output: T")
    .Input("backprop: T")
	.Output("output_tensor:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class RightPoolOp: public OpKernel {
};
template <typename T>
class RightPoolOp<CPUDevice,T>: public OpKernel {
	public:
		explicit RightPoolOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("RightPool");
			const Tensor &_tensor = context->input(0);

			OP_REQUIRES(context, _tensor.dims() == 4, errors::InvalidArgument("tensor must be 4-dimensional"));

			auto         tensor        = _tensor.template tensor<T,4>();
			TensorShape  outshape      = _tensor.shape();
			Tensor      *output_tensor = nullptr;

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_tensor));

			auto           output      = output_tensor->template tensor<T,4>();
			constexpr auto kProcessDim = 2;
			const auto     process_nr  = _tensor.dim_size(kProcessDim);

            output.chip(0,kProcessDim) = tensor.chip(0,kProcessDim);

            for(auto i=1; i<process_nr; ++i){
                output.chip(i,kProcessDim) = output.chip(i-1,kProcessDim).cwiseMax(tensor.chip(i,kProcessDim));
            }
        }
};
template <typename Device, typename T>
class RightPoolGradOp: public OpKernel {
};
template <typename T>
class RightPoolGradOp<CPUDevice,T>: public OpKernel {
	public:
		explicit RightPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("RightPoolGrad");
			const Tensor &_tensor = context->input(0);
			const Tensor &_fw_output = context->input(1);
			const Tensor &_backprop = context->input(2);

			OP_REQUIRES(context, _tensor.dims() == 4, errors::InvalidArgument("tensor must be 4-dimensional"));
			OP_REQUIRES(context, _fw_output.dims() == 4, errors::InvalidArgument("forward output must be 4-dimensional"));
			OP_REQUIRES(context, _backprop.dims() == 4, errors::InvalidArgument("backprop must be 4-dimensional"));

			auto         tensor        = _tensor.template tensor<T,4>();
			auto         fw_output     = _fw_output.template tensor<T,4>();
			auto         backprop      = _backprop.template tensor<T,4>();
			TensorShape  outshape      = _tensor.shape();
			Tensor      *output_tensor = nullptr;

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_tensor));

			auto           output      = output_tensor->template tensor<T,4>();
			constexpr auto kProcessDim = 2;
			const auto     process_nr  = _tensor.dim_size(kProcessDim);
            Eigen::Tensor<int,3> index(_tensor.dim_size(0),_tensor.dim_size(1),_tensor.dim_size(3));

            output.setZero();
            output.chip(0,kProcessDim) = backprop.chip(0,kProcessDim);
            index.setConstant(0);

            for(auto i=1; i<process_nr; ++i){
                Eigen::Tensor<bool,3,Eigen::RowMajor> t = (fw_output.chip(i-1,kProcessDim) < tensor.chip(i,kProcessDim));
                assign_grad(t,i,index,backprop,output);
            }
        }

        template<typename IT,typename BPT,typename OT,typename ST>
        void assign_grad(const ST& select,int cur_index,IT& index, const BPT& backprop,OT& output) {
            auto idx = 0;
            for(auto i=0; i<index.dimension(0); ++i) {
                for(auto j=0; j<index.dimension(1); ++j) {
                    for(auto k=0; k<index.dimension(2); ++k) {
                        if(select(i,j,k)){
                                idx = cur_index;
                                index(i,j,k) = idx;
                          } else {
                                idx = index(i,j,k);
                          }
                        output(i,j,idx,k) += backprop(i,j,idx,k);
                    }
                }
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("RightPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), RightPoolOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RightPool").Device(DEVICE_CPU).TypeConstraint<double>("T"), RightPoolOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("RightPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), RightPoolGradOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RightPoolGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"), RightPoolGradOp<CPUDevice, double>);
