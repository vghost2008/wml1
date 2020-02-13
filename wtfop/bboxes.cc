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
#include "bboxes.h"
#include "wtoolkit.h"

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;

/*
 * 
 * bottom_box:[batch_size,Nr,4](y,x,h,w)
 * bottom_pred:[batch_size,Nr,num_class]
 * output_box:[X,4]
 * output_classes:[X]
 * output_scores:[X]
 * output_batch_index:[X]
 */
REGISTER_OP("BoxesSelect")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("ignore_first:bool")
    .Input("bottom_box: T")
    .Input("bottom_pred: T")
	.Output("output_box:T")
	.Output("output_classes:T")
	.Output("output_scores:T")
	.Output("output_batch_index:T");

template <typename Device, typename T>
class BoxesSelectOp: public OpKernel {
	public:
		explicit BoxesSelectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("ignore_first", &ignore_first));
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box        = context->input(0);
			const Tensor &bottom_pred       = context->input(1);
			auto          bottom_box_flat   = bottom_box.flat<T>();
			auto          bottom_pred_flat  = bottom_pred.flat<T>();

			OP_REQUIRES(context, bottom_box.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, bottom_pred.dims() == 3, errors::InvalidArgument("pred data must be 2-dimensional"));

			const auto     batch_nr  = bottom_box.dim_size(0);
			const auto     data_nr   = bottom_box.dim_size(1);
			const auto     class_nr  = bottom_pred.dim_size(2);
			using Outtype=tuple<const float*,int,float,int>; //box,class,scores,batch_index
			vector<Outtype>  tmp_outdata;
			auto type_func = ignore_first?type_without_first:type_with_first;
			auto shard = [this, &bottom_box_flat,&bottom_pred_flat,class_nr,data_nr,batch_nr,&tmp_outdata,type_func]
				(int64 start, int64 limit) {
					for (int64 b = start; b < limit; ++b) {
						int batch_ind = b;
						int data_ind  = batch_ind%data_nr;
						batch_ind /= data_nr;
						const auto   base_offset0 = batch_ind *data_nr *4+data_ind *4;
						const auto   base_offset1 = batch_ind *data_nr *class_nr+data_ind *class_nr;
						const float *box_data    = bottom_box_flat.data()+base_offset0;
						const float *pred        = bottom_pred_flat.data()+base_offset1;
						const auto   type        = type_func(pred,class_nr);
						const auto   scores      = pred[type];

						if((scores >= threshold) && (box_area(box_data)>1E-4)) 
							tmp_outdata.emplace_back(box_data,type,scores,batch_ind);
					}
				};

			const DeviceBase::CpuWorkerThreads& worker_threads =
			*(context->device()->tensorflow_cpu_worker_threads());
			const int64 total_cost= batch_nr*data_nr;
			//Shard(worker_threads.num_threads, worker_threads.workers,total_cost,total_cost, shard);
			shard(0,total_cost);
			int dims_2d[2] = {int(tmp_outdata.size()),4};
			int dims_1d[1] = {int(tmp_outdata.size())};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;
			Tensor      *output_batch_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_batch_index));

			auto obox         = output_box->template flat<T>();
			auto oclasses     = output_classes->template flat<T>();
			auto oscores      = output_scores->template flat<T>();
			auto obatch_index = output_batch_index->template flat<T>();

			for(int i=0; i<tmp_outdata.size(); ++i) {
				auto& data = tmp_outdata[i];
				auto box = get<0>(data);
				std::copy(box,box+4,obox.data()+4*i);
				oclasses(i) = get<1>(data);
				oscores(i) = get<2>(data);
				obatch_index(i) = get<3>(data);
			}
		}
		static int type_with_first(const float* data,size_t size) {
			auto it = max_element(data,data+size);
			return it-data;
		}
		static int type_without_first(const float* data,size_t size) {
			auto it = max_element(data+1,data+size);
			return it-data;
		}
	private:
		bool  ignore_first;
		float threshold;
};
REGISTER_KERNEL_BUILDER(Name("BoxesSelect").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesSelectOp<CPUDevice, float>);

/*
 * bottom_boxes:[Nr,4](ymin,xmin,ymax,xmax) proposal box
 * width:float
 * height:float
 * output:[Nr,4] 相对坐标(ymin,xmin,ymax,xmax)
 */
REGISTER_OP("BoxesRelativeToAbsolute")
    .Attr("T: {float, double}")
	.Attr("width: int")
	.Attr("height: int")
    .Input("bottom_boxes: T")
	.Output("output:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class BoxesRelativeToAbsoluteOp: public OpKernel {
	public:
		explicit BoxesRelativeToAbsoluteOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
			OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_boxes      = context->input(0);
			auto          bottom_boxes_flat = bottom_boxes.flat<T>();

			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("boxes data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dim_size(1)==4, errors::InvalidArgument("Boxes second dim size must be 4."));
			const auto nr = bottom_boxes.dim_size(0);

			TensorShape output_shape = bottom_boxes.shape();
			// Create output tensors
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

			auto output = output_tensor->template flat<T>();
			auto shard = [this, &bottom_boxes_flat,&output]
					 (int64 start, int64 limit) {
						 for (int64 b = start; b < limit; ++b) {
							 const auto base_offset = b *4;
							 const auto box_data    = bottom_boxes_flat.data()+base_offset;
							 const auto output_data = output.data()+base_offset;
							 output_data[0] = box_data[0]*(height_-1)+0.5f;
							 output_data[1] = box_data[1]*(width_-1)+0.5f;
							 output_data[2] = box_data[2]*(height_-1)+0.5f;
							 output_data[3] = box_data[3]*(width_-1)+0.5f;
						 }
					 };

			const DeviceBase::CpuWorkerThreads& worker_threads =
				*(context->device()->tensorflow_cpu_worker_threads());
			Shard(worker_threads.num_threads, worker_threads.workers,
					nr, 1000, shard);
		}
	private:
		int width_;
		int height_;
};
REGISTER_KERNEL_BUILDER(Name("BoxesRelativeToAbsolute").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesRelativeToAbsoluteOp<CPUDevice, float>);
/*
 * 在image中剪切出一个ref_box定义的区域，同时对原来在image中的boxes进行处理，如果boxes与剪切区域jaccard得分小于threshold的会被删除
 * ref_box:shape=[4],[ymin,xmin,ymax,xmax] 参考box,相对坐标
 * boxes:[Nr,4],(ymin,xmin,ymax,xmax),需要处理的box
 * threshold:阀值
 * output:[Y,4]
 */
REGISTER_OP("CropBoxes")
    .Attr("T: {float, double}")
	.Attr("threshold:float")
    .Input("ref_box: T")
    .Input("boxes: T")
	.Output("output:T")
	.Output("mask:bool")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(c->Dim(c->input(1), 0)));
			return Status::OK();
			});

template <typename Device, typename T>
class CropBoxesOp: public OpKernel {
	public:
		explicit CropBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &ref_box      = context->input(0);
			const Tensor &boxes        = context->input(1);
			auto          ref_box_flat = ref_box.flat<T>();
			auto          boxes_flat   = boxes.flat<T>();
			const auto    nr           = boxes.dim_size(0);
			vector<int>   good_index;
			vector<bool>  good_mask(nr,false);

			OP_REQUIRES(context, ref_box.dims() == 1, errors::InvalidArgument("ref box must be 1-dimensional"));
			OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

            good_index.reserve(nr);

            for(int i=0; i<nr; ++i) {
                auto cur_box = boxes_flat.data()+4*i;
                if(bboxes_jaccard_of_box0(cur_box,ref_box_flat.data()) < threshold) continue;
                good_index.push_back(i);
				good_mask[i] = true;
            }
            const int   out_nr       = good_index.size();
            const int   dims_2d[]    = {out_nr,4};
            const int   dims_1d[]    = {int(nr)};
            TensorShape output_shape;
            TensorShape output_shape1;

			TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
			TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape1);

			Tensor *output_tensor = NULL;
			Tensor *output_mask   = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_mask));

			auto output           = output_tensor->template flat<T>();
			auto output_mask_flat = output_mask->template flat<bool>();

			for(auto i=0; i<nr; ++i)
				output_mask_flat.data()[i] = good_mask[i];

            for(int i=0; i<out_nr; ++i) {
                cut_box(ref_box_flat.data(),boxes_flat.data()+good_index[i]*4,output.data()+i*4);
            }
		}
	private:
		float threshold=1.0;
};
REGISTER_KERNEL_BUILDER(Name("CropBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), CropBoxesOp<CPUDevice, float>);

/*
 * 删除与边框的jaccard得分大于threshold的box
 * size:shape=[2],[h,w] 相对坐标
 * boxes:[Nr,4],(ymin,xmin,ymax,xmax),需要处理的box
 * threshold:阀值
 * output:[Y,4]
 */
REGISTER_OP("RemoveBoundaryBoxes")
    .Attr("T: {float, double}")
	.Attr("threshold:float")
    .Input("size: T")
    .Input("boxes: T")
	.Output("output:T")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(c->UnknownDim(), 4));
			c->set_output(1, c->Vector(c->UnknownDim()));
			return Status::OK();
			});

template <typename Device, typename T>
class RemoveBoundaryBoxesOp: public OpKernel {
	public:
		explicit RemoveBoundaryBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &size         = context->input(0);
			const Tensor &boxes        = context->input(1);
			auto          size_flat    = size.flat<T>();
			auto          boxes_flat   = boxes.flat<T>();
			vector<int>   good_index;
			T             ref_boxes0[] = {0.0f,0.0f,1.0f,size_flat.data()[1]};
			T             ref_boxes1[] = {0.0f,0.0f,size_flat.data()[0],1.0f};
			T             ref_boxes2[] = {0.0f,1.0f-size_flat.data()[1],1.0f,1.0f};
			T             ref_boxes3[] = {1.0f-size_flat.data()[0],0.0f,1.0f,1.0f};

			OP_REQUIRES(context, size.dims() == 1, errors::InvalidArgument("size must be 1-dimensional"));
			OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

			const auto nr = boxes.dim_size(0);
            good_index.reserve(nr);

            for(int i=0; i<nr; ++i) {
                auto cur_box = boxes_flat.data()+4*i;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes0) > threshold) continue;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes1) > threshold) continue;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes2) > threshold) continue;
                if(bboxes_jaccard_of_box0(cur_box,ref_boxes3) > threshold) continue;
                good_index.push_back(i);
            }
            const int   out_nr       = good_index.size();
            const int   dims_2d[]    = {out_nr,4};
            const int   dims_1d[]    = {out_nr};
            TensorShape output_shape;
            TensorShape output_shape1;

			TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
			TensorShapeUtils::MakeShape(dims_1d, 1, &output_shape1);

			Tensor* output_tensor = NULL;
			Tensor* output_tensor_index = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_tensor_index));

			auto output       = output_tensor->template flat<T>();
			auto output_index = output_tensor_index->template flat<int32_t>();

            for(int i=0; i<out_nr; ++i) {
                copy_box(boxes_flat.data()+good_index[i]*4,output.data()+i*4);
				output_index.data()[i] = good_index[i];
            }
		}
	private:
		float threshold=1.0;
};
REGISTER_KERNEL_BUILDER(Name("RemoveBoundaryBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), RemoveBoundaryBoxesOp<CPUDevice, float>);

Status distored_boxes_shape(shape_inference::InferenceContext* c) 
{
    auto shape = c->input(0);

    if(!c->FullyDefined(shape)) {
        c->set_output(0,c->Matrix(c->UnknownDim(),4));
        return Status::OK();
    }

    auto          data_nr  = c->Value(c->Dim(shape,0));
    bool          keep_org;
    int           res_nr   = 0;
    vector<float> xoffset;
    vector<float> yoffset;
    vector<float> scale;

    c->GetAttr("keep_org",&keep_org);
    c->GetAttr("xoffset",&xoffset);
    c->GetAttr("yoffset",&yoffset);
    c->GetAttr("scale",&scale);
    if(keep_org)
        res_nr = data_nr;
    else
        res_nr = 0;
    res_nr += (xoffset.size()+yoffset.size()+scale.size())*data_nr;
    c->set_output(0,c->Matrix(res_nr,4));
    return Status::OK();
}
/*
 * 对Boxes:[Nr,4]进行多样化处理
 * scale:对box进行缩放处理
 * offset:对box进行上下左右的平移处理
 */
REGISTER_OP("DistoredBoxes")
    .Attr("T: {float, double}")
	.Attr("scale:list(float)")
	.Attr("xoffset:list(float)")
	.Attr("yoffset:list(float)")
	.Attr("keep_org:bool")
    .Input("boxes: T")
	.Output("output_boxes:T")
	.SetShapeFn(distored_boxes_shape);

template <typename Device, typename T>
class DistoredBoxesOp: public OpKernel {
	private:
	public:
		explicit DistoredBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
			OP_REQUIRES_OK(context, context->GetAttr("xoffset", &xoffset_));
			OP_REQUIRES_OK(context, context->GetAttr("yoffset", &yoffset_));
			OP_REQUIRES_OK(context, context->GetAttr("keep_org", &keep_org_));
		}
        inline int get_output_nr(int nr)const {
            auto    output_nr  = 0; 
            if(keep_org_)
                output_nr += nr;
            output_nr += (xoffset_.size()+yoffset_.size())*nr;
            output_nr += scale_.size()*nr;
            return output_nr;
        }

		void Compute(OpKernelContext* context) override
        {
            const Tensor &boxes      = context->input(0);
            auto          boxes_flat = boxes.flat<T>();
            const auto    nr         = boxes.dim_size(0);
            const auto    output_nr  = get_output_nr(nr);

            OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

            const int   dims_2d[]     = {int(output_nr),4};
            TensorShape output_shape0;

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape0);

            Tensor *output_boxes  = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_boxes));

            auto outputboxes  = output_boxes->template flat<T>();
            auto output_index = 0;

            if(keep_org_) {
                copy_boxes(boxes_flat.data(),outputboxes.data(),nr);
                output_index = nr;
            }
            processOffsetX(boxes_flat.data(),outputboxes.data(),xoffset_,nr,output_index);
            processOffsetY(boxes_flat.data(),outputboxes.data(),yoffset_,nr,output_index);

            for(auto i=0; i<scale_.size(); ++i) {
                processScale(boxes_flat.data(),outputboxes.data(),scale_[i],nr,output_index);
            }
        }
		void processOffsetX(const T* src_boxes,T* out_boxes,const vector<float>& xoffset,const int nr,int& output_index) 
		{
			for(auto k=0; k<nr; ++k) {
				const auto src_box = src_boxes+4*k;
				for(auto i=0; i<xoffset.size(); ++i) {
					const auto os      = xoffset[i];
					const auto dx      = (src_box[3]-src_box[1]) *os;
					auto       cur_box = out_boxes+4 *output_index++;

					copy_box(src_box,cur_box);
					cur_box[1] += dx;
					cur_box[3] += dx;
				}
			}
		}
		void processOffsetY(const T* src_boxes,T* out_boxes,const vector<float>& yoffset,const int nr,int& output_index) 
		{
			for(auto k=0; k<nr; ++k) {
				const auto src_box = src_boxes+4*k;
				for(auto i=0; i<yoffset.size(); ++i) {
					const auto os      = yoffset[i];
					const auto dy      = (src_box[2]-src_box[0]) *os;
					auto       cur_box = out_boxes+4 *output_index++;

					copy_box(src_box,cur_box);
					cur_box[0] -= dy;
					cur_box[2] -= dy;
				}
			}
		}
		void processScale(const T* src_boxes,T* out_boxes,const float scale,const int nr,int& output_index) 
		{
			for(auto k=0; k<nr; ++k) {
				const auto src_box = src_boxes+4 *k;
				const auto dx      = (src_box[3]-src_box[1]) *(scale-1.0)/2.;
				const auto dy      = (src_box[2]-src_box[0]) *(scale-1.0)/2.;
				auto       cur_box = out_boxes+4 *output_index++;

				copy_box(src_box,cur_box);
				cur_box[0] -= dy;
				cur_box[1] -= dx;
				cur_box[2] += dy;
				cur_box[3] += dx;
			}
		}
	private:
		vector<float> scale_;
		vector<float> xoffset_;
		vector<float> yoffset_;
		bool          keep_org_;
};
REGISTER_KERNEL_BUILDER(Name("DistoredBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), DistoredBoxesOp<CPUDevice, float>);

Status random_distored_boxes_shape(shape_inference::InferenceContext* c) 
{
    auto shape = c->input(0);

    if(!c->FullyDefined(shape)) {
        c->set_output(0,c->Matrix(c->UnknownDim(),4));
        return Status::OK();
    }

    auto data_nr  = c->Value(c->Dim(shape,0));
    bool keep_org;
    int  res_nr   = 0;
    int  size;

    c->GetAttr("keep_org",&keep_org);
    c->GetAttr("size",&size);
    if(keep_org)
        res_nr = data_nr;
    else
        res_nr = 0;
    res_nr += size*data_nr;
    c->set_output(0,c->Matrix(res_nr,4));
    return Status::OK();
}
/*
 * 对Boxes:[Nr,4]进行多样化处理
 * limits:[xoffset,yoffset,scale]大小限制
 * size:[产生xoffset,yoffset,scale的数量]
 */
REGISTER_OP("RandomDistoredBoxes")
    .Attr("T: {float, double}")
	.Attr("limits:list(float)")
	.Attr("size:int")
	.Attr("keep_org:bool")
    .Input("boxes: T")
	.Output("output_boxes:T")
	.SetShapeFn(random_distored_boxes_shape);

template <typename Device, typename T>
class RandomDistoredBoxesOp: public OpKernel {
	private:
	public:
		explicit RandomDistoredBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("limits", &limits_));
			OP_REQUIRES_OK(context, context->GetAttr("size", &size_));
			OP_REQUIRES_OK(context, context->GetAttr("keep_org", &keep_org_));
		}
        inline int get_output_nr(int nr)const {
            auto    output_nr  = 0; 
            if(keep_org_)
                output_nr += nr;
            output_nr += size_*nr;
            return output_nr;
        }

		void Compute(OpKernelContext* context) override
		{
			const Tensor &boxes      = context->input(0);
			auto          boxes_flat = boxes.flat<T>();
			const auto    nr         = boxes.dim_size(0);
			const auto    output_nr  = get_output_nr(nr);

			OP_REQUIRES(context, boxes.dims() == 2, errors::InvalidArgument("boxes must be 2-dimensional"));

			const int   dims_2d[]     = {int(output_nr),4};
			TensorShape output_shape0;

			TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape0);

			Tensor *output_boxes  = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_boxes));

			auto outputboxes  = output_boxes->template flat<T>();
			auto output_index = 0;

			if(keep_org_) {
				copy_boxes(boxes_flat.data(),outputboxes.data(),nr);
				output_index = nr;
			}
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> xoffset(-limits_[0],limits_[0]);
			std::uniform_real_distribution<> yoffset(-limits_[1],limits_[1]);
			std::uniform_real_distribution<> scale(-limits_[2],limits_[2]);
			for(auto i=0; i<size_; ++i) {
				/*process(boxes_flat.data(),outputboxes.data(),
						nr,
						xoffset(gen),
						yoffset(gen),
						scale(gen),
						output_index);*/
				process(boxes_flat.data(),outputboxes.data(),
						nr,
						xoffset,
						yoffset,
						scale,
                        gen,
						output_index);
			}
		}
		void process(const T* src_boxes,T* out_boxes,int nr,float xoffset,float yoffset,float scale,int& output_index) 
		{
            for(auto k=0; k<nr; ++k) {
                const auto src_box = src_boxes+4*k;
				const auto sdx      = (src_box[3]-src_box[1]) *scale/2.;
				const auto sdy      = (src_box[2]-src_box[0]) *scale/2.;
                const auto dx      = (src_box[3]-src_box[1])*xoffset;
                const auto dy      = (src_box[2]-src_box[0])*yoffset;
                auto       cur_box = out_boxes+4 *output_index++;

                copy_box(src_box,cur_box);
				cur_box[0] += dy-sdy;
				cur_box[1] += dx-sdx;
				cur_box[2] += dy+sdy;
				cur_box[3] += dx+sdx;
            }
		}
        template<typename dis_t,typename gen_t>
		void process(const T* src_boxes,T* out_boxes,int nr,dis_t& xoffset,dis_t& yoffset,dis_t& scale,gen_t& gen,int& output_index) 
		{
            for(auto k=0; k<nr; ++k) {
                const auto src_box = src_boxes+4*k;
				const auto sdx     = (src_box[3]-src_box[1]) *scale(gen)/2.;
				const auto sdy     = (src_box[2]-src_box[0]) *scale(gen)/2.;
                const auto dx      = (src_box[3]-src_box[1])*xoffset(gen);
                const auto dy      = (src_box[2]-src_box[0])*yoffset(gen);
                auto       cur_box = out_boxes+4 *output_index++;

                copy_box(src_box,cur_box);
				cur_box[0] += dy-sdy;
				cur_box[1] += dx-sdx;
				cur_box[2] += dy+sdy;
				cur_box[3] += dx+sdx;
            }
		}
	private:
		int           size_;
		vector<float> limits_;
		bool          keep_org_;
};
REGISTER_KERNEL_BUILDER(Name("RandomDistoredBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), RandomDistoredBoxesOp<CPUDevice, float>);

/*
 * 将boxes中与gboxes IOU最大且不小于threadshold的标记为相应的label, 即1个gboxes最多与一个boxes相对应
 * boxes:[batch_size,N,4](y,x,h,w)
 * gboxes:[batch_size,M,4]
 * glabels:[batch_size,M] 0表示背景
 * glens:[batch_size] 用于表明gboxes中的有效boxes数量
 * output_labels:[batch_size,N]
 * output_scores:[batch_size,N]
 * threashold: IOU threshold
 */
REGISTER_OP("BoxesMatch")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
    .Input("boxes: T")
    .Input("gboxes: T")
	.Input("glabels:int32")
	.Input("glens:int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesMatchOp: public OpKernel {
	public:
		explicit BoxesMatchOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_boxes   = context->input(0);
			const Tensor &_gboxes  = context->input(1);
			const Tensor &_glabels = context->input(2);
			const Tensor &_glens   = context->input(3);
			auto          boxes    = _boxes.tensor<T,3>();
			auto          gboxes   = _gboxes.tensor<T,3>();
			auto          glabels  = _glabels.tensor<int,2>();
			auto          glens    = _glens.tensor<int,1>();

			OP_REQUIRES(context, _boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _gboxes.dims() == 3, errors::InvalidArgument("gboxes data must be 3-dimensional"));
			OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("glabels data must be 2-dimensional"));
			OP_REQUIRES(context, _glens.dims() == 1, errors::InvalidArgument("glens data must be 1-dimensional"));

			const int batch_nr  = _boxes.dim_size(0);
			const int boxes_nr  = _boxes.dim_size(1);
			const int gboxes_nr = _gboxes.dim_size(1);

			int dims_2d[2] = {batch_nr,boxes_nr};
			TensorShape  outshape;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();

            oclasses.setZero();
            oscores.setZero();

            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;

                    for(auto k=0; k<boxes_nr; ++k) {
                        if(oclasses(i,k) != 0) continue;

                        Eigen::Tensor<T,1,Eigen::RowMajor> gbox = gboxes.chip(i,0).chip(j,0);
                        Eigen::Tensor<T,1,Eigen::RowMajor> box = boxes.chip(i,0).chip(k,0);
                        auto jaccard = bboxes_jaccardv1(gbox,box);

                        if((jaccard>threshold_) && (jaccard>max_scores)) {
                            max_scores = jaccard;
                            index = k;
                        }
                    }
                    if(index>=0) {
                        oclasses(i,index) = glabels(i,j);
                        oscores(i,index) = max_scores;
                    }
                }
            }
		}
	private:
		float threshold_;
};
REGISTER_KERNEL_BUILDER(Name("BoxesMatch").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesMatchOp<CPUDevice, float>);

/*
 * 将boxes先与pred_bboxes匹配（通过pred_labels指定）如果IOU大于指定的阀值则为预测正确，没有与pred_bboxes匹配的bboxes与gboxes匹配，
 * IOU最大且不小于threadshold的标记为相应的label, 即1个gboxes最多与一个boxes相对应
 * boxes:[batch_size,N,4](y,x,h,w)
 * plabels:[batch_size,N]预测的类别
 * gboxes:[batch_size,M,4]
 * glabels:[batch_size,M] 0表示背景
 * glens:[batch_size] 用于表明gboxes中的有效boxes数量
 * output_labels:[batch_size,N]
 * output_scores:[batch_size,N]
 * threashold: IOU threshold
 */
REGISTER_OP("BoxesMatchWithPred")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
    .Input("boxes: T")
    .Input("plabels: int32")
    .Input("gboxes: T")
	.Input("glabels:int32")
	.Input("glens:int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesMatchWithPredOp: public OpKernel {
	public:
		explicit BoxesMatchWithPredOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_boxes   = context->input(0);
			const Tensor &_plabels = context->input(1);
			const Tensor &_gboxes  = context->input(2);
			const Tensor &_glabels = context->input(3);
			const Tensor &_glens   = context->input(4);
			auto          boxes    = _boxes.tensor<T,3>();
			auto          plabels  = _plabels.tensor<int,2>();
			auto          gboxes   = _gboxes.tensor<T,3>();
			auto          glabels  = _glabels.tensor<int,2>();
			auto          glens    = _glens.tensor<int,1>();

			OP_REQUIRES(context, _boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _gboxes.dims() == 3, errors::InvalidArgument("gboxes data must be 3-dimensional"));
			OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("glabels data must be 2-dimensional"));
			OP_REQUIRES(context, _plabels.dims() == 2, errors::InvalidArgument("plabels data must be 2-dimensional"));
			OP_REQUIRES(context, _glens.dims() == 1, errors::InvalidArgument("glens data must be 1-dimensional"));

			const int batch_nr  = _boxes.dim_size(0);
			const int boxes_nr  = _boxes.dim_size(1);
			const int gboxes_nr = _gboxes.dim_size(1);
            Eigen::Tensor<bool,2> process_mask(batch_nr,glabels.dimension(1));
			int dims_2d[2] = {batch_nr,boxes_nr};
			TensorShape  outshape;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;
            if(plabels.dimension(1) != boxes.dimension(1)) {
                cout<<"Error plabels dimension 1"<<endl;
            }

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();

            process_mask.setZero();
            oclasses.setZero();
            oscores.setZero();

            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;

                    for(auto k=0; k<boxes_nr; ++k) {
                        const auto plabel = plabels(i,k);
                        if(plabel != glabels(i,j)) continue;
                        if(oclasses(i,k) != 0) continue;

                        Eigen::Tensor<T,1,Eigen::RowMajor> gbox = gboxes.chip(i,0).chip(j,0);
                        Eigen::Tensor<T,1,Eigen::RowMajor> box = boxes.chip(i,0).chip(k,0);
                        auto jaccard = bboxes_jaccardv1(gbox,box);

                        if((jaccard>threshold_) && (jaccard>max_scores)) {
                            max_scores = jaccard;
                            index = k;
                        }
                    }
                    if(index>=0) {
                        oclasses(i,index) = glabels(i,j);
                        oscores(i,index) = max_scores;
                        process_mask(i,j) = true;
                    }
                }
            }
            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;
                    if(process_mask(i,j)) continue;

                    for(auto k=0; k<boxes_nr; ++k) {
                        if(oclasses(i,k) != 0) continue;

                        Eigen::Tensor<T,1,Eigen::RowMajor> gbox = gboxes.chip(i,0).chip(j,0);
                        Eigen::Tensor<T,1,Eigen::RowMajor> box = boxes.chip(i,0).chip(k,0);
                        auto jaccard = bboxes_jaccardv1(gbox,box);

                        if((jaccard>threshold_) && (jaccard>max_scores)) {
                            max_scores = jaccard;
                            index = k;
                        }
                    }
                    if(index>=0) {
                        oclasses(i,index) = glabels(i,j);
                        oscores(i,index) = max_scores;
                    }
                }
            }
		}
	private:
		float threshold_;
};
REGISTER_KERNEL_BUILDER(Name("BoxesMatchWithPred").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesMatchWithPredOp<CPUDevice, float>);

/*
 * 将boxes先与pred_bboxes匹配（通过pred_labels指定）如果IOU大于指定的阀值则为预测正确，没有与pred_bboxes匹配的bboxes与gboxes匹配，
 * 与上面的版本相比会多输出一个boxex encode
 * IOU最大且不小于threadshold的标记为相应的label, 即1个gboxes最多与一个boxes相对应
 * boxes:[batch_size,N,4](y,x,h,w)
 * plabels:[batch_size,N]预测的类别
 * gboxes:[batch_size,M,4]
 * glabels:[batch_size,M] 0表示背景
 * glens:[batch_size] 用于表明gboxes中的有效boxes数量
 * output_labels:[batch_size,N]
 * output_scores:[batch_size,N]
 * threashold: IOU threshold
 */
REGISTER_OP("BoxesMatchWithPred2")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
 	.Attr("prio_scaling: list(float)")
    .Input("boxes: T")
    .Input("plabels: int32")
    .Input("gboxes: T")
	.Input("glabels:int32")
	.Input("glens:int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("output_encode:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			c->set_output(2, boxes_shape);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesMatchWithPred2Op: public OpKernel {
	public:
		explicit BoxesMatchWithPred2Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_boxes   = context->input(0);
			const Tensor &_plabels = context->input(1);
			const Tensor &_gboxes  = context->input(2);
			const Tensor &_glabels = context->input(3);
			const Tensor &_glens   = context->input(4);
			auto          boxes    = _boxes.tensor<T,3>();
			auto          plabels  = _plabels.tensor<int,2>();
			auto          gboxes   = _gboxes.tensor<T,3>();
			auto          glabels  = _glabels.tensor<int,2>();
			auto          glens    = _glens.tensor<int,1>();

			OP_REQUIRES(context, _boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _gboxes.dims() == 3, errors::InvalidArgument("gboxes data must be 3-dimensional"));
			OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("glabels data must be 2-dimensional"));
			OP_REQUIRES(context, _plabels.dims() == 2, errors::InvalidArgument("plabels data must be 2-dimensional"));
			OP_REQUIRES(context, _glens.dims() == 1, errors::InvalidArgument("glens data must be 1-dimensional"));

			const int batch_nr  = _boxes.dim_size(0);
			const int boxes_nr  = _boxes.dim_size(1);
			const int gboxes_nr = _gboxes.dim_size(1);
            Eigen::Tensor<bool,2> process_mask(batch_nr,glabels.dimension(1));
			int dims_2d[2] = {batch_nr,boxes_nr};
			int dims_3d[3] = {batch_nr,boxes_nr,4};
			TensorShape  outshape;
			TensorShape  outshape1;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;
			Tensor      *output_encode      = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);
			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_encode));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();
			auto oencodes     = output_encode->template tensor<T,3>();

            process_mask.setZero();
            oclasses.setZero();
            oscores.setZero();
            oencodes.setZero();

            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;
                    Eigen::Tensor<T,1,Eigen::RowMajor> gbox = gboxes.chip(i,0).chip(j,0);

                    for(auto k=0; k<boxes_nr; ++k) {
                        const auto plabel = plabels(i,k);
                        if(plabel != glabels(i,j)) continue;
                        if(oclasses(i,k) != 0) continue;

                        Eigen::Tensor<T,1,Eigen::RowMajor> box = boxes.chip(i,0).chip(k,0);
                        auto jaccard = bboxes_jaccardv1(gbox,box);

                        if((jaccard>threshold_) && (jaccard>max_scores)) {
                            max_scores = jaccard;
                            index = k;
                        }
                    }
                    if(index>=0) {
                        oclasses(i,index) = glabels(i,j);
                        oscores(i,index) = max_scores;
                        oencodes.chip(i,0).chip(index,0) = encode_one_boxes<T>(gbox,boxes.chip(i,0).chip(index,0),prio_scaling);
                        process_mask(i,j) = true;
                    }
                }
            }
            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;
                    if(process_mask(i,j)) continue;
                    Eigen::Tensor<T,1,Eigen::RowMajor> gbox = gboxes.chip(i,0).chip(j,0);

                    for(auto k=0; k<boxes_nr; ++k) {
                        if(oclasses(i,k) != 0) continue;

                        Eigen::Tensor<T,1,Eigen::RowMajor> box = boxes.chip(i,0).chip(k,0);
                        auto jaccard = bboxes_jaccardv1(gbox,box);

                        if((jaccard>threshold_) && (jaccard>max_scores)) {
                            max_scores = jaccard;
                            index = k;
                        }
                    }
                    if(index>=0) {
                        oclasses(i,index) = glabels(i,j);
                        oscores(i,index) = max_scores;
                        oencodes.chip(i,0).chip(index,0) = encode_one_boxes<T>(gbox,boxes.chip(i,0).chip(index,0),prio_scaling);
                    }
                }
            }
		}
	private:
		float threshold_;
		vector<float> prio_scaling;
};
REGISTER_KERNEL_BUILDER(Name("BoxesMatchWithPred2").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesMatchWithPred2Op<CPUDevice, float>);
