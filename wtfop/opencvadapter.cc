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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include <opencv2/opencv.hpp>
#include "bboxes.h"
#include "wtoolkit.h"

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("MedianBlur")
    .Attr("T: {uint8}")
	.Attr("ksize:int")
    .Input("image: T")
	.Output("outimage:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class MedianBlurOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<T,4,Eigen::RowMajor>;
	public:
		explicit MedianBlurOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
		}
        void process_one_image(T* data, int width,int height)
        {
            cv::Mat img(height,width,CV_8UC1,data);
            cv::medianBlur(img,img,ksize_);
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_image  = context->input(0);
			TensorShape   output_shape  = _input_image.shape();
			Tensor       *output_tensor = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            if(_input_image.dims() == 3) {
                output_tensor->template tensor<T,3>() = _input_image.template tensor<T,3>();
                OP_REQUIRES(context, _input_image.dim_size(2) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                process_one_image(output_tensor->template flat<T>().data(),_input_image.dim_size(1),_input_image.dim_size(0));
            } else if(_input_image.dims()==4) {
                output_tensor->template tensor<T,4>() = _input_image.template tensor<T,4>();
                OP_REQUIRES(context, _input_image.dim_size(3) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                const auto img_size = _input_image.dim_size(1)*_input_image.dim_size(2);
                auto data = output_tensor->template flat<T>().data();
                for(auto i=0; i<_input_image.dim_size(0); ++i) {
                    auto d = data+i*img_size;
                    process_one_image(d,_input_image.dim_size(2),_input_image.dim_size(1));
                }
            } else {
                OP_REQUIRES(context, _input_image.dims() == 3, errors::InvalidArgument("Error dims size."));
            }
        }

	private:
        int ksize_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("MedianBlur").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MedianBlurOp<CPUDevice, uint8_t>);

REGISTER_OP("BilateralFilter")
    .Attr("T: {float,uint8}")
	.Attr("d:int")
	.Attr("sigmaColor:float")
	.Attr("sigmaSpace:float")
    .Input("image: T")
	.Output("outimage:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class BilateralFilterOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<T,4,Eigen::RowMajor>;
	public:
		explicit BilateralFilterOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("d", &d_));
			OP_REQUIRES_OK(context, context->GetAttr("sigmaColor", &sigmaColor_));
			OP_REQUIRES_OK(context, context->GetAttr("sigmaSpace", &sigmaSpace_));
		}
        template<typename TT>
        void process_one_image(const TT* data_i, const TT* data_o,int width,int height,int channel)
        {
            assert(false);
        }
        void process_one_image(const uint8_t* data_i, uint8_t* data_o,int width,int height,int channel)
        {
            if(1==channel) {
                __process_one_image<CV_8UC1>(data_i,data_o,width,channel);
            } else {
                __process_one_image<CV_8UC3>(data_i,data_o,width,channel);
            }
        }
        void process_one_image(const float* data_i, float* data_o,int width,int height,int channel)
        {
            cout<<data_i<<","<<data_o<<endl;
            if(1==channel) {
                __process_one_image<CV_32FC1>(data_i,data_o,width,height,channel);
            } else {
                __process_one_image<CV_32FC3>(data_i,data_o,width,height,channel);
            }
        }
        template<int type,typename TT>
        void __process_one_image(const TT* data_i, TT* data_o,int width,int height,int channel)
        {
            cv::Mat img_i(height,width,type,const_cast<TT*>(data_i));
            cv::Mat img_o(height,width,type,data_o);
            cv::bilateralFilter(img_i,img_o,d_,sigmaColor_,sigmaSpace_);
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_image  = context->input(0);
			TensorShape   output_shape  = _input_image.shape();
			Tensor       *output_tensor = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            if(_input_image.dims() == 3) {
                output_tensor->template tensor<T,3>() = _input_image.template tensor<T,3>();
                OP_REQUIRES(context, _input_image.dim_size(2) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                auto data_o = output_tensor->template flat<T>().data();
                auto data_i = _input_image.template flat<T>().data();
                process_one_image(data_i,data_o,_input_image.dim_size(1),_input_image.dim_size(0),_input_image.dim_size(2));
            } else if(_input_image.dims()==4) {
                output_tensor->template tensor<T,4>() = _input_image.template tensor<T,4>();
                OP_REQUIRES(context, _input_image.dim_size(3) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                const auto img_size = _input_image.dim_size(1)*_input_image.dim_size(2);
                auto data_o = output_tensor->template flat<T>().data();
                auto data_i = _input_image.template flat<T>().data();
                for(auto i=0; i<_input_image.dim_size(0); ++i) {
                    auto d_i = data_i+i*img_size;
                    auto d_o = data_o+i*img_size;
                    process_one_image(d_i,d_o,_input_image.dim_size(2),_input_image.dim_size(1),_input_image.dim_size(3));
                }
            } else {
                OP_REQUIRES(context, _input_image.dims() == 3, errors::InvalidArgument("Error dims size."));
            }
        }

	private:
        int d_ = 0;
        float sigmaColor_ = 0;
        float sigmaSpace_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MedianBlurOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_CPU).TypeConstraint<float>("T"), BilateralFilterOp<CPUDevice, float>);
