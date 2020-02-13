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
/*
data:输入Tensor,shape为[X,Y]
输出:output shape为[X*(1+expand_nr),Y]
如输入[[1,2],
[3,4]]
expand_nr = 2:
输出:
[[1,2],
[1,2],
[1,2],
[3,4],
[3,4],
[3,4]]
*/
REGISTER_OP("ExpandTensor")
    .Attr("T: {int32, int64,float32,float64}")
	.Attr("expand_nr:int")
    .Input("data: T")
	.Output("output:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto dims_data0 = c->input(0);
            int expand_nr = 0;
            c->GetAttr("expand_nr",&expand_nr);
            auto batch_size = c->Value(c->Dim(dims_data0,0))*(1+expand_nr);
            auto output_shape0 = c->Matrix(batch_size,c->Dim(dims_data0,1));

            c->set_output(0,output_shape0);
            return Status::OK();
            });

template <typename Device, typename T>
class ExpandTensorOp: public OpKernel {
	public:
		explicit ExpandTensorOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("expand_nr", &expand_nr));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &data= context->input(0);
			auto          data_flat = data.flat<T>();

			OP_REQUIRES(context, data.dims() == 2, errors::InvalidArgument("data data must be 2-dimensional"));

			const auto batch_size   = data.dim_size(0);
			const auto num_output   = batch_size *(1+expand_nr);
			const auto data_len = data.dim_size(1);

			TensorShape output_shape0({num_output,data_len});

			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_data));

			auto oq_flat = output_data->flat<T>();

			for(auto i=0; i<batch_size; ++i) {
				auto bq_i = data_flat.data()+data_len*i;
				auto bq_o = oq_flat.data()+data_len*i*(expand_nr+1);
				for(auto k=0; k<=expand_nr; ++k) {
					for(auto j=0; j<data_len; ++j) {
						bq_o[j] = bq_i[j];
					}
					bq_o += data_len;
				}
			}
		}
	private:
		int expand_nr;
};
REGISTER_KERNEL_BUILDER(Name("ExpandTensor").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), ExpandTensorOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("ExpandTensor").Device(DEVICE_CPU).TypeConstraint<float>("T"), ExpandTensorOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("ExpandTensor").Device(DEVICE_CPU).TypeConstraint<double>("T"), ExpandTensorOp<CPUDevice, double>);
Status slide_batch_shape(shape_inference::InferenceContext* c) 
{
    auto                         shape        = c->input(0);
    auto                         filter_shape = c->input(1);
    string                       padding;
    vector<int>                  strides;
    shape_inference::ShapeHandle output;

    c->GetAttr("padding",&padding);
    c->GetAttr("strides",&strides);

    auto org_h = c->Value(c->Dim(shape,0));
    auto org_w = c->Value(c->Dim(shape,1));

    if(padding == "SAME") {
        org_h += (c->Value(c->Dim(filter_shape,0))-1);
        org_w += (c->Value(c->Dim(filter_shape,1))-1);
    }

    const auto h_size =  (org_h-c->Value(c->Dim(filter_shape,0)))/strides[0]+1;
    const auto w_size =  (org_w-c->Value(c->Dim(filter_shape,1)))/strides[1]+1;


    c->Concatenate(c->MakeShape({h_size,w_size}),shape,&output);
    c->set_output(0,output);

    return Status::OK();
}
/*
 * 输入一个[H,W,C]或[H,W]的tensor
 * 输出一个[H1,W1,H,W,C]的tensor
 * filter:[h,w,c]或[h,w]
 * H1,W1指定的每一个tensor都是原tensor在相应位置与filter相乘的结果
 */
REGISTER_OP("SlideBatch")
    .Attr("T: {int32, int64,float32,float64}")
	.Attr("strides:list(int)")
	.Attr("padding:string")
    .Input("data: T")
    .Input("filter: T")
	.Output("output:T")
	.SetShapeFn(slide_batch_shape);

template <typename Device, typename T>
class SlideBatchOp: public OpKernel {
	public:
		explicit SlideBatchOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
			OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
		}
        inline bool is_same() const {
            return padding_ == "SAME";
        }
        inline bool is_valid()const {
            return !is_same();
        }
        inline int h_stride()const { return strides_[0]; }
        inline int w_stride()const { return strides_[1]; }
		inline int h_begin(int fh)const { 
			if(is_same())
				return (1-fh)/2;
			else
				return 0;
		}
		inline int h_end(int fh,int h)const { 
			if(is_same())
				return h-1;
			else
				return h-fh+1;
		}
        inline int w_begin(int fw)const { 
			if(is_same())
				return (1-fw)/2;
			else
				return 0;
        }
        inline int w_end(int fw,int w)const { 
			if(is_same())
				return w-1;
			else
				return w-fw+1;
        }
        inline size_t output_h_size(int fh,int h) const {
            return (h_end(fh,h)-h_begin(fh))/h_stride();
        }
        inline size_t output_w_size(int fw,int w) const {
            return (w_end(fw,w)-w_begin(fw))/w_stride();
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &data   = context->input(0);
			const Tensor &filter = context->input(1);
			const int     fh     = filter.dim_size(0);
			const int     fw     = filter.dim_size(1);
            Eigen::Tensor<T, 3,Eigen::RowMajor>  data_t = data.template tensor<T,3>();
            Eigen::Tensor<T,3,Eigen::RowMajor> filter_t = filter.template tensor<T,3>();

			auto      data_flat   = data.flat<T>();
			auto      filter_flat = filter.flat<T>();
			const int h           = data_t.dimension(0);
			const int w           = data_t.dimension(1);
			const int c           = data_t.dimension(2);
			const int oh          = output_h_size(fh,h);
			const int ow          = output_w_size(fw,w);

			TensorShape output_shape0({oh,ow,data.dim_size(0),data.dim_size(1),c});
			cout<<"oh="<<oh<<", ow="<<ow<<endl;

			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_data));

			auto      oq_tensor = output_data->template tensor<T,5>();
			const int size[]    = {h,w,c};

            for(auto i=0; i<oh; ++i) {
                for(auto j=0; j<ow; ++j) {
                    Eigen::array<int, 3> offsets = {h_begin(fh)+h_stride()*i,w_begin(fw)+w_stride()*j,0};
                    Eigen::array<int, 3> extents = {fh, fw,c};
                    Eigen::array<int, 3> offsets1 = {0,0};

					correct(offsets,extents,offsets1,size);

                    auto v      = data_t.slice(offsets, extents) *filter_t.slice(offsets1,extents);
                    auto target = oq_tensor.chip(i,0).chip(j,0);

					target                         =  data_t;
					target.slice(offsets,extents)  =  v;
                }
            }
		}
		static void correct(Eigen::array<int,3>& offsets,Eigen::array<int,3>& extents, Eigen::array<int,3>& offsets1,const int size[3] ) {
			for(auto i=0; i<3; ++i) {
				if(offsets[i]< 0) {
				extents[i] = offsets[i]+extents[i];
				offsets1[i] = -offsets[i];
				offsets[i] = 0;
				} else  if(offsets[i]+extents[i] > size[i]) {
					extents[i] = size[i]-offsets[i];
				}
			}
		}
	private:
		vector<int> strides_;
		string      padding_;
};
REGISTER_KERNEL_BUILDER(Name("SlideBatch").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), SlideBatchOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("SlideBatch").Device(DEVICE_CPU).TypeConstraint<float>("T"), SlideBatchOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SlideBatch").Device(DEVICE_CPU).TypeConstraint<double>("T"), SlideBatchOp<CPUDevice, double>);

/*
 * 输入data 1D:Tensor
 * padding:表示在axis 0上进行对称padding的数量,负数或0表示无操作
 * 如果原有数据的数量为0， 则使用0填充
 */
REGISTER_OP("WPad")
    .Attr("T: {int32, int64,float32,float64}")
    .Input("tensor: T")
    .Input("padding: int32")
	.Output("data:T");

template <typename Device, typename T>
class WPadOp: public OpKernel {
	public:
		explicit WPadOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_data    = context->input(0);
			const Tensor &_padding = context->input(1);
			auto          data     = _data.template flat<T>().data();
			auto          padding  = _padding.template flat<int>().data();
			const auto    data_nr  = _data.dim_size(0);
			const auto    out_size = data_nr+ std::max<int>(0,padding[0])+std::max<int>(0,padding[1]);

			OP_REQUIRES(context, _data.dims()<=1, errors::InvalidArgument("tensor data must be 1-dimensional"));
			OP_REQUIRES(context, _padding.dims()<=1, errors::InvalidArgument("padding data must be 1-dimensional"));

			TensorShape output_shape0({out_size});
			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_data));

			auto      oq_tensor = output_data->template flat<T>().data();

			/*
			 * 如果原始数据中没有内容，使用0填充
			 */
			if(data_nr == 0) {
				for(auto i=0; i<out_size; ++i) {
					oq_tensor[i] = 0.0f;
				}
				return;
			} 
			/*
			 * 原始数据中有内容，对称填充
			 */

            for(auto i=0; i<padding[0]; ++i) {
                oq_tensor[i] = data[(padding[0]-i-1)%data_nr];
            }
            auto base_index = std::max<int>(0,padding[0]);
            for(auto i=0; i<data_nr; ++i) {
                oq_tensor[i+base_index] = data[i];
            }
            base_index = std::max<int>(0,padding[0])+data_nr;
            auto src_index = data_nr-1;
            for(auto i=0; i<padding[1]; ++i,--src_index) {
                if(src_index<0)
                    src_index = data_nr-1;
                oq_tensor[i+base_index] = data[src_index];
            }
		}
};
REGISTER_KERNEL_BUILDER(Name("WPad").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), WPadOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("WPad").Device(DEVICE_CPU).TypeConstraint<float>("T"), WPadOp<CPUDevice, float>);

/*
 * 输入tensor [X,Y,Z,...,M,N,..]tensor
 * 输入v[M,N,...] tensor
 * 输入index[num]，依次表示[X,Y,Z,...]维度的值
 * 将tensor中由index指定的值设置为
 * example:
 * tensor shape=[2,3,4,2,2]
 * v shape=[2,2]
 * index=[0,1,3]
 * 那么tensor[0,1,3]=v
 */
REGISTER_OP("SetValue")
    .Attr("T: {int32,int64,float32,float64,bool}")
    .Input("tensor: T")
    .Input("v: T")
    .Input("index: int32")
	.Output("data:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class SetValueOp: public OpKernel {
	public:
		explicit SetValueOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_tensor        = context->input(0);
			const Tensor &_v             = context->input(1);
			const Tensor &_index         = context->input(2);
			auto          tensor         = _tensor.template flat<T>().data();
			auto          v              = _v.template flat<T>().data();
			auto          index          = _index.template flat<int>().data();
			auto          dim_nr         = _tensor.dims();
			auto          skip_dim_nr    = _index.dim_size(0);
			auto          offset         = 0;
			const auto    block_size     = _v.NumElements();
			auto          cur_block_size = block_size;

			for(auto i=skip_dim_nr-1; i>=0; --i) {
				offset += index[i]*cur_block_size;
				cur_block_size *= _tensor.dim_size(i);
			}

			OP_REQUIRES(context, _index.dims()==1, errors::InvalidArgument("index must be 1-dimensional"));

			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, _tensor.shape(), &output_data));

			auto      oq_tensor = output_data->template flat<T>().data();
            copy(tensor,tensor+_tensor.NumElements(),oq_tensor);

			/*
			 * 如果原始数据中没有内容，使用0填充
			 */
			 copy(v,v+block_size,oq_tensor+offset);
		}
};
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), SetValueOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<float>("T"), SetValueOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<double>("T"), SetValueOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<bool>("T"), SetValueOp<CPUDevice, bool>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), SetValueOp<CPUDevice, tensorflow::int64>);
/*
 * 输入mask[H,W,N]
 * 输入labels[N]
 * set_background:如果一个位置没有标签，则默认为背景
 * attr:num_classes
 * 输出mask[W,H,num_classes]
 */
REGISTER_OP("SparseMaskToDense")
    .Attr("T: {int32,bool,int8,uint8}")
	.Attr("num_classes:int")
	.Attr("set_background:bool")
    .Input("mask: T")
    .Input("labels: int32")
	.Output("data:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			int num_classes = 0;
			c->GetAttr("num_classes",&num_classes);
			auto w = c->Value(c->Dim(c->input(0),0));
			auto h = c->Value(c->Dim(c->input(0),1));
            auto output_shape = c->MakeShape({h, w, num_classes});
			c->set_output(0,output_shape);
			return Status::OK();
			});

template <typename Device, typename T>
class SparseMaskToDenseOp: public OpKernel {
	public:
		explicit SparseMaskToDenseOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("num_classes", &num_classes_));
			OP_REQUIRES_OK(context,
					context->GetAttr("set_background", &set_background_));
		}
		void Compute(OpKernelContext* context) override
        {
            const Tensor &_mask   = context->input(0);
            const Tensor &_labels = context->input(1);
            auto          mask    = _mask.template tensor<T,3>();
            auto          labels  = _labels.template flat<int>().data();
            auto          h       = _mask.dim_size(0);
            auto          w       = _mask.dim_size(1);
            auto          nr      = _mask.dim_size(2);
            auto          nr1     = _labels.dim_size(0);

            OP_REQUIRES(context, _labels.dims()==1, errors::InvalidArgument("labels must be 1-dimensional"));
            OP_REQUIRES(context, _mask.dims()==3, errors::InvalidArgument("mask must be 3-dimensional"));
            OP_REQUIRES(context, nr==nr1, errors::InvalidArgument("size unmatch."));

            int          dims3d[]     = {int(h),int(w),num_classes_};
            Tensor      *output_data  = NULL;
            TensorShape  output_shape;

            TensorShapeUtils::MakeShape(dims3d, 3, &output_shape);

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_data));

            auto      oq_tensor = output_data->template tensor<T,3>();
            oq_tensor.setZero();
			using tensor_t = Eigen::Tensor<T,2,Eigen::RowMajor>;

            for(auto i=0; i<nr; ++i) {
				const auto label = labels[i];
				if((label<0) || (label>=num_classes_)) {
					cout<<"Error label "<<label<<", not in range [0,"<<num_classes_<<")"<<endl;
					continue;
				}
                auto t = oq_tensor.chip(label,2);
                t = (t||mask.chip(i,2)).template cast<T>();
            }

			if(!set_background_) return;

			for(auto i=0; i<h; ++i) {
				for(auto j=0; j<w; ++j) {
					bool have_label = false;
					for(auto k=0; k<nr; ++k) {
						const auto label = labels[k];

						if((label<0) || (label>=num_classes_)) continue;

						if(mask(i,j,k) != 0) {
							have_label = true;
							break;
						}
					}
					if(have_label) continue;

					oq_tensor(i,j,0) = T(true);
				}
			}
        }
	private:
		int num_classes_ = 0;
		bool set_background_ = false;
};
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), SparseMaskToDenseOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<bool>("T"), SparseMaskToDenseOp<CPUDevice, bool>);
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), SparseMaskToDenseOp<CPUDevice, int8_t>);
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), SparseMaskToDenseOp<CPUDevice, uint8_t>);
