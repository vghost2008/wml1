#include <stdio.h>
#include <cfloat>
#include <iostream>
#include <boost/algorithm/clamp.hpp>
#include <third_party/eigen3/unsupported/Eigen/CXX11/Tensor>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace std;
using namespace boost::algorithm;
typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * bottom_data:输入网络，rois的绝对坐标以bottom_data的空间大小为限
 * bottom_rois应该是2维的shape为[N,5],最后一维依次为[batch_index,w_min,h_min,w_max,h_max] 绝对坐标
 * 包含边界(min及max)
 * 输出output:[N,pool_height,pool_width,channels]
 */

REGISTER_OP("RoiPooling")
    .Attr("T: {float, double}")
    .Attr("pool_height: int")
    .Attr("pool_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("argmax: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto dims_data     = c->input(0);
            auto channels      = c->Value(c->Dim(dims_data,3));
            auto dims_rois     = c->input(1);
            auto num_rois      = c->Value(c->Dim(dims_rois,0));
            int  pooled_height = 0;
            int  pooled_width  = 0;

            c->GetAttr("pool_width",&pooled_width);
            c->GetAttr("pool_height",&pooled_height);

            auto output_shape = c->MakeShape({num_rois, pooled_height, pooled_width, channels});

            c->set_output(0,output_shape);

            return Status::OK();
            });


REGISTER_OP("RoiPoolingGrad")
    .Attr("T: {float, double}")
    .Attr("pool_height: int")
    .Attr("pool_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("argmax: int32")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class RoiPoolingOp : public OpKernel {
 public:
  explicit RoiPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("pool_height", &pool_height_));
    OP_REQUIRES(context, pool_height_ >= 0,
                errors::InvalidArgument("Need pool_height >= 0, got ",
                                        pool_height_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("pool_width", &pool_width_));
    OP_REQUIRES(context, pool_width_ >= 0,
                errors::InvalidArgument("Need pool_width >= 0, got ",
                                        pool_width_));
	/*
	 * 对bottom_rois指定的区域进行缩放
	 */
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    const Tensor &bottom_data      = context->input(0);
    const Tensor &bottom_rois      = context->input(1);
    auto          bottom_data_flat = bottom_data.flat<T>();
    auto          bottom_rois_flat = bottom_rois.flat<T>();

    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    int num_rois = bottom_rois.dim_size(0);
    int batch_size = bottom_data.dim_size(0);
    int data_height = bottom_data.dim_size(1);
    int data_width = bottom_data.dim_size(2);
    int num_channels = bottom_data.dim_size(3);

    int dims[4];

    dims[0] = num_rois;
    dims[1] = pool_height_;
    dims[2] = pool_width_;
    dims[3] = num_channels;

    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    Tensor* output_tensor = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    auto output = output_tensor->template flat<T>();

    Tensor* argmax_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
    auto argmax = argmax_tensor->template flat<int>();

    int   pool_height   = pool_height_;
    int   pool_width    = pool_width_;
    float spatial_scale = spatial_scale_;

	if((num_rois<=0) || (batch_size<=0)) {
		while(1)
			printf("in box nr :%d,data nr%d\n",num_rois,batch_size);
	}

    auto shard = [pool_height, pool_width, spatial_scale,
                  num_rois, batch_size, data_height, data_width, num_channels,
                  &bottom_data_flat, &bottom_rois_flat, &output, &argmax]
                  (int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b)
      {
        // (n, ph, pw, c) is an element in the pooled output
        int n = b;
        int c = n % num_channels;
        n /= num_channels;
        int pw = n % pool_width;
        n /= pool_width;
        int ph = n % pool_height;
        n /= pool_height;

        const float* bottom_rois = bottom_rois_flat.data() + n * 5;
		/*
		 * 确定roi区域
		 */
        int roi_batch_ind = bottom_rois[0];
        int roi_start_w   = round(bottom_rois[1] *spatial_scale);
        int roi_start_h   = round(bottom_rois[2] *spatial_scale);
        int roi_end_w     = round(bottom_rois[3] *spatial_scale);
        int roi_end_h     = round(bottom_rois[4] *spatial_scale);

        int     roi_width  = std::max(roi_end_w - roi_start_w + 1, 1);
        int     roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pool_height);
        const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pool_width);

        int hstart  =  static_cast<int>(floor(ph * bin_size_h));
        int wstart  =  static_cast<int>(floor(pw * bin_size_w));
        int hend    =  static_cast<int>(ceil((ph + 1) * bin_size_h));
        int wend    =  static_cast<int>(ceil((pw + 1) * bin_size_w));

        hstart  =  clamp(hstart + roi_start_h, 0, data_height);
        hend    =  clamp(hend + roi_start_h, 0, data_height);
        wstart  =  clamp(wstart + roi_start_w, 0, data_width);
        wend    =  clamp(wend + roi_start_w, 0, data_width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        float maxval = is_empty ? 0 : -FLT_MAX;
        int maxidx = -1;
        const float* bottom_data = bottom_data_flat.data() + roi_batch_ind * num_channels * data_height * data_width;
		//printf("process:%d, total=%d\n",b,output.size());
		/*
		if((hend>hstart+1)||(wend>wstart+1))
			printf("hstart=%d,wstart=%d,hend=%d,wend=%d\n",hstart,wstart,hend,wend);
			*/
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = (h * data_width + w) * num_channels + c;
            if (bottom_data[bottom_index] > maxval) {
              maxval = bottom_data[bottom_index];
              maxidx = bottom_index;
            }
          }
        }
        output(b) = maxval;
        argmax(b) = maxidx;
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        num_rois * num_channels * pool_height * pool_width * spatial_scale;
    Shard(worker_threads.num_threads, worker_threads.workers,
          output.size(), shard_cost, shard);
  }
 private:
  int pool_height_;
  int pool_width_;
  float spatial_scale_;
};
//梯度
template <class Device, class T>
class RoiPoolingGradOp : public OpKernel {
	public:
		explicit RoiPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {

			OP_REQUIRES_OK(context,
					context->GetAttr("pool_height", &pool_height_));
			OP_REQUIRES(context, pool_height_ >= 0,
					errors::InvalidArgument("Need pool_height >= 0, got ",
						pool_height_));
			OP_REQUIRES_OK(context,
					context->GetAttr("pool_width", &pool_width_));
			OP_REQUIRES(context, pool_width_ >= 0,
					errors::InvalidArgument("Need pool_width >= 0, got ",
						pool_width_));
			OP_REQUIRES_OK(context,
					context->GetAttr("spatial_scale", &spatial_scale_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_data  = context->input(0);
			const Tensor &bottom_rois  = context->input(1);
			const Tensor &argmax_data  = context->input(2);
			const Tensor &out_backprop = context->input(3);

			auto bottom_data_flat  = bottom_data.flat<T>();
			auto bottom_rois_flat  = bottom_rois.flat<T>();
			auto argmax_data_flat  = argmax_data.flat<int32>();
			auto out_backprop_flat = out_backprop.flat<T>();

			OP_REQUIRES(context, bottom_data.dims() == 4,
					errors::InvalidArgument("data must be 4-dimensional"));

			OP_REQUIRES(context, bottom_rois.dims() == 2,
					errors::InvalidArgument("rois must be 2-dimensional"));

			OP_REQUIRES(context, argmax_data.dims() == 4,
					errors::InvalidArgument("argmax_data must be 4-dimensional"));

			OP_REQUIRES(context, out_backprop.dims() == 4,
					errors::InvalidArgument("out_backprop must be 4-dimensional"));

			int num_rois = bottom_rois.dim_size(0);
			int batch_size = bottom_data.dim_size(0);
			int data_height = bottom_data.dim_size(1);
			int data_width = bottom_data.dim_size(2);
			int num_channels = bottom_data.dim_size(3);

			TensorShape output_shape = bottom_data.shape();

			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

			auto  output        = output_tensor->template flat<T>();
			int   pool_height   = pool_height_;
			int   pool_width    = pool_width_;
			float spatial_scale = spatial_scale_;

			auto shard = [pool_height, pool_width, spatial_scale,
				 num_rois, batch_size, data_height, data_width, num_channels,
				 &bottom_data_flat, &bottom_rois_flat, &argmax_data_flat,
				 &out_backprop_flat, &output](int64 start, int64 limit) {
					 for (int64 b = start; b < limit; ++b)
					 {
						 // (n, h, w, c) coords in bottom data
						 int n = b;
						 int c = n % num_channels;
						 n /= num_channels;
						 int w = n % data_width;
						 n /= data_width;
						 int h = n % data_height;
						 n /= data_height;

						 float gradient = 0.0;
						 //Accumulate gradient over all ROIs that pooled this element
						 for (int roi_n = 0; roi_n < num_rois; ++roi_n)
						 {
							 const float *offset_bottom_rois = bottom_rois_flat.data() + roi_n *5;
							 int          roi_batch_ind      = offset_bottom_rois[0];
							 if (n != roi_batch_ind) {
								 continue;
							 }

							 int roi_start_w = round(offset_bottom_rois[1] *spatial_scale);
							 int roi_start_h = round(offset_bottom_rois[2] *spatial_scale);
							 int roi_end_w   = round(offset_bottom_rois[3] *spatial_scale);
							 int roi_end_h   = round(offset_bottom_rois[4] *spatial_scale);


							 const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
									 h >= roi_start_h && h <= roi_end_h);
							 if (!in_roi) {
								 continue;
							 }

							 int          offset             = roi_n *pool_height *pool_width *num_channels;
							 const float *offset_top_diff    = out_backprop_flat.data() + offset;
							 const int   *offset_argmax_data = argmax_data_flat.data() + offset;

							 int roi_width  = std::max(roi_end_w - roi_start_w + 1, 1);
							 int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);

							 const T bin_size_h = static_cast<T>(roi_height)
								 / static_cast<T>(pool_height);
							 const T bin_size_w = static_cast<T>(roi_width)
								 / static_cast<T>(pool_width);

							 int phstart  =  floor(static_cast<int>(h - roi_start_h) / bin_size_h);
							 int phend    =  ceil(static_cast<int>(h - roi_start_h + 1) / bin_size_h);
							 int pwstart  =  floor(static_cast<int>(w - roi_start_w) / bin_size_w);
							 int pwend    =  ceil(static_cast<int>(w - roi_start_w + 1) / bin_size_w);

							 phstart  =  clamp(phstart, 0, pool_height);
							 phend    =  clamp(phend, 0, pool_height);
							 pwstart  =  clamp(pwstart, 0, pool_width);
							 pwend    =  clamp(pwend, 0, pool_width);

							 for (int ph = phstart; ph < phend; ++ph) {
								 for (int pw = pwstart; pw < pwend; ++pw) {
									 if (offset_argmax_data[(ph * pool_width + pw) * num_channels + c] == (h * data_width + w) * num_channels + c)
									 {
										 gradient += offset_top_diff[(ph * pool_width + pw) * num_channels + c];
									 }
								 }
							 }
						 }
						 output(b) = gradient;
					 }
				 };

			const DeviceBase::CpuWorkerThreads& worker_threads =
				*(context->device()->tensorflow_cpu_worker_threads());
			const int64 shard_cost =
				num_rois * num_channels * pool_height * pool_width * spatial_scale;
			Shard(worker_threads.num_threads, worker_threads.workers,
					output.size(), shard_cost, shard);
		}
	private:
		int   pool_height_;
		int   pool_width_;
		float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiPooling").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolingOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolingGradOp<CPUDevice, float>);
