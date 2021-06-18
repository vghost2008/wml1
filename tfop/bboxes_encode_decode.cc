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
#include "bboxes_encode.h"
#include "wtoolkit.h"
#include "wtoolkit_cuda.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
/*
 * scale_weights:[4](基意义与BoxesEncode的prio_scaling互为倒数
 * boxes:[1,X,4]/[batch_size,X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * gboxes:[batch_size,Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * labels:[batch_size,X] 0为背景,-1表示忽略
 * indices:[batch_size,X] 与box相对应的gtbox索引
 * output_boxes:[batch_size,X,4] regs(cy,cx,h,w)
 */
REGISTER_OP("GetBoxesDeltas")
    .Attr("T: {float,double,int32,int64}")
 	.Attr("scale_weights: list(float)")
    .Input("boxes: T")
    .Input("gboxes: T")
    .Input("labels: int32")
    .Input("indices: int32")
	.Output("output_boxes:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			const auto batch_size = c->Dim(c->input(1),0);
			const auto box_nr = c->Dim(c->input(0),1);
			const auto box_dim = c->Dim(c->input(0),2);
            auto shape0 = c->MakeShape({batch_size,box_nr,box_dim});

			c->set_output(0, shape0);
			return Status::OK();
			});

template <typename Device, typename T>
class GetBoxesDeltasOp: public OpKernel {
	public:
		using tensor_2d = Eigen::Tensor<T,2,Eigen::RowMajor>;
	public:
		explicit GetBoxesDeltasOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("scale_weights", &scale_weights));
			OP_REQUIRES(context, scale_weights.size() == 4, errors::InvalidArgument("scale_weights data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_labels  = context->input(2);
			const Tensor &_bottom_indices = context->input(3);

			OP_REQUIRES(context, _bottom_boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_labels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_indices.dims() == 2, errors::InvalidArgument("index data must be 1-dimensional"));

			auto          bottom_boxes    = _bottom_boxes.template tensor<T,3>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,3>();
			auto          bottom_labels   = _bottom_labels.template tensor<int,2>();
			auto          bottom_indices  = _bottom_indices.template tensor<int,2>();
			const auto    batch_size      = _bottom_gboxes.dim_size(0);
			const auto    data_nr         = _bottom_boxes.dim_size(1);


			int dims_3d[] = {batch_size,data_nr,_bottom_boxes.dim_size(2)};
			TensorShape   outshape0 = _bottom_boxes.shape();
			Tensor       *output_boxes          = NULL;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));

			auto output_boxes_tensor          =  output_boxes->template tensor<T,3>();

			output_boxes_tensor.setZero();

            auto shard = [&](int64 start,int64 limit){
				for(auto i=start; i<limit; ++i) {

					auto boxes   = tensor_2d(bottom_boxes.chip(bottom_boxes.dimension(0)==batch_size?i:0,0));
					auto gboxes  = tensor_2d(bottom_gboxes.chip(i,0));
					/*
					 * 计算所有正样本gbox到所对应的box的回归参数
					 */
					for(auto j=0; j<data_nr; ++j) {
						if((bottom_labels(i,j)<1)) {
							continue;
						}
						Eigen::Tensor<T,1,Eigen::RowMajor> box     = boxes.chip(j,0);
						Eigen::Tensor<T,1,Eigen::RowMajor> gbox    = gboxes.chip(bottom_indices(i,j),0);
						auto  yxhw    = box_minmax_to_cxywh(box.data());
						auto  yref    = std::get<0>(yxhw);
						auto  xref    = std::get<1>(yxhw);
						auto  href    = std::get<2>(yxhw);
						auto  wref    = std::get<3>(yxhw);

						if((href<1E-8) || (wref<1E-8)) {
							continue;
						}

						auto gyxhw = box_minmax_to_cxywh(gbox.data());
						auto gcy =  std::get<0>(gyxhw);
						auto gcx =  std::get<1>(gyxhw);
						auto gh  =  std::get<2>(gyxhw);
						auto gw  =  std::get<3>(gyxhw);
						auto feat_cy  =  scale_weights[0]*(gcy-yref)/href;
						auto feat_cx  =  scale_weights[1]*(gcx-xref)/wref;
						auto feat_h   =  log(gh/href)*scale_weights[2];
						auto feat_w   =  log(gw/wref)*scale_weights[3];

						output_boxes_tensor(i,j,0) = feat_cy;
						output_boxes_tensor(i,j,1) = feat_cx;
						output_boxes_tensor(i,j,2) = feat_h;
						output_boxes_tensor(i,j,3) = feat_w;
					}
				}
            };
            list<future<void>> results;
            for(auto i=0; i<batch_size; ++i) {
                results.emplace_back(async(launch::async,[i,&shard](){ shard(i,i+1);}));
            }
        }
	private:
		vector<float> scale_weights;
};
REGISTER_KERNEL_BUILDER(Name("GetBoxesDeltas").Device(DEVICE_CPU).TypeConstraint<float>("T"), GetBoxesDeltasOp<CPUDevice, float>);
///////////
/*
 * prio_scaling:[4]
 * max_overlap_as_pos:是否将与ground truth bbox交叉面积最大的box设置为正样本
 * bottom_boxes:[1,X,4]/[batch_size,X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[batch_size,Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[batch_size,Y] 0为背景
 * bottom_glength:[batch_size] 为每一个batch中gboxes的有效数量
 * output_boxes:[batch_size,X,4] regs(cy,cx,h,w)
 * output_labels:[batch_size,X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分
 * output_scores:[batch_size,X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 * output_remove_indict:[batch_size,X], anchorbox是否有效(一般为iou处理中间部分的无效)
 * output_indict:[batch_size,X], 当anchorbox有效时，与它对应的gboxes(从0开始)序号,无效时为-1
 */
REGISTER_OP("BoxesEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("pos_threshold:float")
	.Attr("neg_threshold:float")
 	.Attr("prio_scaling: list(float)")
    .Attr("max_overlap_as_pos:bool")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
    .Input("bottom_glength: int32")
	.Output("output_boxes:T")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("remove_indict:bool")
	.Output("indict:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto input_shape1 = c->input(1);
            const auto batch_size = c->Dim(input_shape1,0);
            const auto boxes_nr  = c->Dim(input_shape0,1);
            auto shape0 = c->MakeShape({batch_size,boxes_nr,4});
            auto shape1 = c->MakeShape({batch_size,boxes_nr});

			c->set_output(0, shape0);
            for(auto i=1; i<5; ++i)
			    c->set_output(i, shape1);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesEncodeOp: public OpKernel {
};
template <typename T>
class BoxesEncodeOp<CPUDevice,T>: public OpKernel {
	public:
		explicit BoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("BoxesEncode");
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

			int           dims_3d[3]            = {int(batch_size),int(data_nr),4};
			int           dims_2d[2]            = {int(batch_size),int(data_nr)};
			TensorShape   outshape0;
			TensorShape   outshape1;
			Tensor       *output_boxes          = NULL;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_remove_indict  = NULL;
			Tensor       *output_indict         = NULL;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);


			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_remove_indict));
			OP_REQUIRES_OK(context, context->allocate_output(4, outshape1, &output_indict));

			auto output_boxes_tensor          =  output_boxes->template tensor<T,3>();
			auto output_labels_tensor         =  output_labels->template tensor<int,2>();
			auto output_scores_tensor         =  output_scores->template tensor<T,2>();
			auto output_remove_indict_tensor  =  output_remove_indict->template tensor<bool,2>();
			auto output_indict_tensor         =  output_indict->template tensor<int,2>();

            BoxesEncodeUnit<CPUDevice,T> encode_unit(pos_threshold,neg_threshold,prio_scaling,max_overlap_as_pos_);
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
                    auto out_boxes          = output_boxes_tensor.chip(i,0);
                    auto out_labels         = output_labels_tensor.chip(i,0);
                    auto out_scores         = output_scores_tensor.chip(i,0);
                    auto out_remove_indices = output_remove_indict_tensor.chip(i,0);
                    auto out_indices        = output_indict_tensor.chip(i,0);
                    auto res                = encode_unit(boxes,gboxes,glabels);

                    out_boxes           =  std::get<0>(res);
                    out_labels          =  std::get<1>(res);
                    out_scores          =  std::get<2>(res);
                    out_remove_indices  =  std::get<3>(res);
                    out_indices         =  std::get<4>(res);
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
		vector<float> prio_scaling;
        bool          max_overlap_as_pos_ = true;
};
#ifdef GOOGLE_CUDA
template <typename T>
class BoxesEncodeOp<GPUDevice,T>: public OpKernel {
	public:
		explicit BoxesEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("BoxesEncodeGPU");
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

			int           dims_3d[3]            = {int(batch_size),int(data_nr),4};
			int           dims_2d[2]            = {int(batch_size),int(data_nr)};
			TensorShape   outshape0;
			TensorShape   outshape1;
			Tensor       *output_boxes          = NULL;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_remove_indict  = NULL;
			Tensor       *output_indict         = NULL;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_remove_indict));
			OP_REQUIRES_OK(context, context->allocate_output(4, outshape1, &output_indict));

			auto output_boxes_tensor          =  output_boxes->template tensor<T,3>();
			auto output_labels_tensor         =  output_labels->template tensor<int,2>();
			auto output_scores_tensor         =  output_scores->template tensor<T,2>();
			auto output_remove_indict_tensor  =  output_remove_indict->template tensor<bool,2>();
			auto output_indict_tensor         =  output_indict->template tensor<int,2>();

            BoxesEncodeUnit<GPUDevice,T> encode_unit(pos_threshold,neg_threshold,prio_scaling,max_overlap_as_pos_);
            for(auto i=0; i<batch_size; ++i) {
                    
                    auto size    = bottom_gsize(i);
                    auto boxes   = bottom_boxes.dimension(0)==batch_size?chip_data(bottom_boxes,i):chip_data(bottom_boxes,0);
                    auto gboxes  = chip_data(bottom_gboxes,i);
                    auto glabels = chip_data(bottom_glabels,i);
                    encode_unit(boxes,gboxes,glabels,
                            chip_data(output_boxes_tensor,i),
                            chip_data(output_labels_tensor,i),
                            chip_data(output_scores_tensor,i),
                            chip_data(output_remove_indict_tensor,i),
                            chip_data(output_indict_tensor,i),
                            size,bottom_boxes.dimension(1)
                            );
                }
        }
	private:
		float         pos_threshold;
		float         neg_threshold;
		vector<float> prio_scaling;
        bool          max_overlap_as_pos_ = true;
};
#endif
REGISTER_KERNEL_BUILDER(Name("BoxesEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesEncodeOp<CPUDevice, float>);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("BoxesEncode").Device(DEVICE_GPU).TypeConstraint<float>("T"), BoxesEncodeOp<GPUDevice, float>);
#endif
/*
 * prio_scaling:[4]
 * bottom_boxes:[X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[Y] 0为背景
 * output_boxes:[X,4] regs(cy,cx,h,w)
 * output_labels:[X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分
 * output_scores:[X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 */
REGISTER_OP("BoxesEncode1")
    .Attr("T: {float,double,int32,int64}")
	.Attr("pos_threshold:float")
	.Attr("neg_threshold:float")
 	.Attr("prio_scaling: list(float)")
 	.Attr("max_overlap_as_pos: bool")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
	.Output("output_boxes:T")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("remove_indict:bool")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto shape0 = c->input(0);
            auto shape1 = c->Vector(c->Dim(shape0,0));

            c->set_output(0,shape0);

            for(auto i=1; i<4; ++i) c->set_output(i,shape1);

            return Status::OK();
            });

template <typename Device, typename T>
class BoxesEncode1Op{
};
template <typename T>
class BoxesEncode1Op<CPUDevice,T>: public OpKernel {
	public:
        struct IOUIndex{
            int index;
            float iou;
        };
		explicit BoxesEncode1Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("BoxesEncode1");
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_glabels = context->input(2);
			auto          bottom_boxes    = _bottom_boxes.template tensor<T,2>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,2>();
			auto          bottom_glabels  = _bottom_glabels.template tensor<int,1>();
			const auto    data_nr         = _bottom_boxes.dim_size(0);

			OP_REQUIRES(context, _bottom_boxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_glabels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));

			int           dims_2d[2]            = {int(data_nr),4};
			int           dims_1d[1]            = {int(data_nr)};
			TensorShape   outshape0;
			TensorShape   outshape1;
			Tensor       *output_boxes          = NULL;
			Tensor       *output_labels         = NULL;
			Tensor       *output_scores         = NULL;
			Tensor       *output_remove_indict  = NULL;
			vector<IOUIndex>   iou_indexs(data_nr,IOUIndex({-1,0.0})); //默认box不与任何ground truth box相交，iou为0

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);


			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_remove_indict));
			auto output_boxes_tensor         = output_boxes->template tensor<T,2>();
			auto output_labels_tensor        = output_labels->template tensor<int,1>();
			auto output_scores_tensor        = output_scores->template tensor<T,1>();
			auto output_remove_indict_tensor = output_remove_indict->template tensor<bool,1>();

			BoxesEncodeUnit<CPUDevice,T> encode_unit(pos_threshold,neg_threshold,prio_scaling,max_overlap_as_pos_);
			auto &boxes              = bottom_boxes;
			auto &gboxes             = bottom_gboxes;
			auto &glabels            = bottom_glabels;
			auto &out_boxes          = output_boxes_tensor;
			auto &out_labels         = output_labels_tensor;
			auto &out_scores         = output_scores_tensor;
			auto &out_remove_indices = output_remove_indict_tensor;
			auto  res                = encode_unit(boxes,gboxes,glabels);

			out_boxes           =  std::get<0>(res);
			out_labels          =  std::get<1>(res);
			out_scores          =  std::get<2>(res);
			out_remove_indices  =  std::get<3>(res);
		}
	private:
		float         pos_threshold;
		float         neg_threshold;
		vector<float> prio_scaling;
        bool max_overlap_as_pos_ = true;
};
template <typename T>
class BoxesEncode1Op<GPUDevice,T>: public OpKernel {
	public:
        struct IOUIndex{
            int index;
            float iou;
        };
		explicit BoxesEncode1Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("BoxesEncode1");
            const Tensor &_bottom_boxes   = context->input(0);
            const Tensor &_bottom_gboxes  = context->input(1);
            const Tensor &_bottom_glabels = context->input(2);
            auto          bottom_boxes    = _bottom_boxes.template tensor<T,2>();
            auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,2>();
            auto          bottom_glabels  = _bottom_glabels.template tensor<int,1>();
            const auto    data_nr         = _bottom_boxes.dim_size(0);

            OP_REQUIRES(context, _bottom_boxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
            OP_REQUIRES(context, _bottom_gboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
            OP_REQUIRES(context, _bottom_glabels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));

            int           dims_2d[2]            = {int(data_nr),4};
            int           dims_1d[1]            = {int(data_nr)};
            TensorShape   outshape0;
            TensorShape   outshape1;
            Tensor       *output_boxes          = NULL;
            Tensor       *output_labels         = NULL;
            Tensor       *output_scores         = NULL;
            Tensor       *output_remove_indict  = NULL;
            vector<IOUIndex>   iou_indexs(data_nr,IOUIndex({-1,0.0})); //默认box不与任何ground truth box相交，iou为0

            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_boxes));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_scores));
            OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_remove_indict));
            auto output_boxes_tensor         = output_boxes->template tensor<T,2>();
            auto output_labels_tensor        = output_labels->template tensor<int,1>();
            auto output_scores_tensor        = output_scores->template tensor<T,1>();
            auto output_remove_indict_tensor = output_remove_indict->template tensor<bool,1>();

            BoxesEncodeUnit<GPUDevice,T> encode_unit(pos_threshold,neg_threshold,prio_scaling,max_overlap_as_pos_);
            auto size    = bottom_gboxes.dimension(0);
            auto boxes   = bottom_boxes.data();
            auto gboxes  = bottom_gboxes.data();
            auto glabels = bottom_glabels.data();
            encode_unit(boxes,gboxes,glabels,
                    output_boxes_tensor.data(),
                    output_labels_tensor.data(),
                    output_scores_tensor.data(),
                    output_remove_indict_tensor.data(),
                    nullptr,
                    size,bottom_boxes.dimension(0)
                    );
        }
	private:
		float         pos_threshold;
		float         neg_threshold;
		vector<float> prio_scaling;
        bool max_overlap_as_pos_ = true;
};

REGISTER_OP("BoxesEncode1Grad")
    .Attr("T: {float,double,int32,int64}")
	.Attr("threshold:float")
 	.Attr("prio_scaling: list(float)")
    .Input("bottom_boxes: T")
    .Input("bottom_gboxes: T")
    .Input("bottom_glabels: int32")
    .Input("grad: T")
	.Output("output:T");
template <typename Device, typename T>
class BoxesEncode1GradOp: public OpKernel {
	public:
		explicit BoxesEncode1GradOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
			OP_REQUIRES_OK(context, context->GetAttr("prio_scaling", &prio_scaling));
			OP_REQUIRES(context, prio_scaling.size() == 4, errors::InvalidArgument("prio scaling data must be shape[4]"));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_boxes        = context->input(0);

			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));

			TensorShape  outshape0   = bottom_boxes.shape();
			Tensor      *output_grad = nullptr;

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_grad));

			auto output_grad_flat = output_grad->template flat<T>();
			auto num_elements     = output_grad->NumElements();
			//cout<<"num elements:"<<num_elements<<","<<bottom_boxes.dims()<<","<<bottom_boxes.dim_size(0)<<","<<bottom_boxes.dim_size(1)<<endl;

            for(auto i=0; i<num_elements; ++i)
                output_grad_flat(i) = 0.0f;
		}
	private:
		float threshold;
		vector<float> prio_scaling;
};
REGISTER_KERNEL_BUILDER(Name("BoxesEncode1").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesEncode1Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("BoxesEncode1Grad").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesEncode1GradOp<CPUDevice, float>);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("BoxesEncode1").Device(DEVICE_GPU).TypeConstraint<float>("T"), BoxesEncode1Op<GPUDevice, float>);
#endif
/*
 * bottom_boxes:[Nr,4](ymin,xmin,ymax,xmax) proposal box,相对坐标
 * bottom_regs:[Nr,4],(y,x,h,w)
 * prio_scaling:[4]
 * output:[Nr,4] 相对坐标(ymin,xmin,ymax,xmax)
 */
REGISTER_OP("DecodeBoxes1")
    .Attr("T: {float, double}")
	.Attr("prio_scaling:list(float)")
    .Input("bottom_boxes: T")
    .Input("bottom_regs: T")
	.Output("output:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class DecodeBoxes1Op{
};
template <typename T>
class DecodeBoxes1Op<CPUDevice,T>: public OpKernel {
	public:
		explicit DecodeBoxes1Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("prio_scaling", &prio_scaling));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("DecodeBoxes1");
			const Tensor &bottom_boxes       = context->input(0);
			const Tensor &bottom_regs        = context->input(1);
			auto          bottom_regs_flat   = bottom_regs.flat<T>();
			auto          bottom_boxes_flat  = bottom_boxes.flat<T>();

			OP_REQUIRES(context, bottom_regs.dims() == 2, errors::InvalidArgument("regs data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("pos data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dim_size(0)==bottom_regs.dim_size(0), errors::InvalidArgument("First dim size must be equal."));
			OP_REQUIRES(context, bottom_boxes.dim_size(1)==4, errors::InvalidArgument("Boxes second dim size must be 4."));
			OP_REQUIRES(context, bottom_regs.dim_size(1)==4, errors::InvalidArgument("Regs second dim size must be 4."));
			const auto nr = bottom_regs.dim_size(0);

			TensorShape output_shape = bottom_regs.shape();
			// Create output tensors
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

			auto output = output_tensor->template flat<T>();
			for (auto b = 0; b < nr; ++b) {
				const auto base_offset = b *4;
				const auto regs_data   = bottom_regs_flat.data()+base_offset;
				const auto box_data    = bottom_boxes_flat.data()+base_offset;
				float      y;
				float      x;
				float      href;
				float      wref;

				std::tie(y,x,href,wref) = box_minmax_to_cxywh(box_data);

				auto cy          = boost::algorithm::clamp<T>(regs_data[0]*prio_scaling[0],-10.0f,10.0f)*href+y;
				auto cx          = boost::algorithm::clamp<T>(regs_data[1]*prio_scaling[1],-10.0f,10.0f)*wref+x;
				auto h           = href *exp(boost::algorithm::clamp<T>(regs_data[2]*prio_scaling[2],-10.0,10.0));
				auto w           = wref *exp(boost::algorithm::clamp<T>(regs_data[3]*prio_scaling[3],-10.0,10.0));
				auto output_data = output.data() + base_offset;

				std::tie(output_data[0],output_data[1],output_data[2],output_data[3]) = box_cxywh_to_minmax(cy,cx,h,w);
				transform(output_data,output_data+4,output_data,[](T& v) { return boost::algorithm::clamp<T>(v,0.0,1.0);});
				if(output_data[0]>output_data[2]) 
					output_data[2] = output_data[0];
				if(output_data[1]>output_data[3])
					output_data[3] = output_data[1];
			}
		}
	private:
		std::vector<float> prio_scaling;
};
template <typename T>
class DecodeBoxes1Op<GPUDevice,T>: public OpKernel {
	public:
		explicit DecodeBoxes1Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("prio_scaling", &prio_scaling));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("DecodeBoxesGPU1");
			const Tensor &bottom_boxes       = context->input(0);
			const Tensor &bottom_regs        = context->input(1);
			auto          bottom_regs_flat   = bottom_regs.flat<T>();
			auto          bottom_boxes_flat  = bottom_boxes.flat<T>();

			OP_REQUIRES(context, bottom_regs.dims() == 2, errors::InvalidArgument("regs data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dims() == 2, errors::InvalidArgument("pos data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_boxes.dim_size(0)==bottom_regs.dim_size(0), errors::InvalidArgument("First dim size must be equal."));
			OP_REQUIRES(context, bottom_boxes.dim_size(1)==4, errors::InvalidArgument("Boxes second dim size must be 4."));
			OP_REQUIRES(context, bottom_regs.dim_size(1)==4, errors::InvalidArgument("Regs second dim size must be 4."));
			const auto nr = bottom_regs.dim_size(0);

			TensorShape output_shape = bottom_regs.shape();
			// Create output tensors
			Tensor* output_tensor = NULL;
			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

			auto output = output_tensor->template flat<T>();
            if(nr>0)
                bboxes_decode_by_gpu(bottom_boxes_flat.data(),bottom_regs_flat.data(),prio_scaling.data(),output.data(),nr);
		}
	private:
		std::vector<float> prio_scaling;
};
REGISTER_KERNEL_BUILDER(Name("DecodeBoxes1").Device(DEVICE_CPU).TypeConstraint<float>("T"), DecodeBoxes1Op<CPUDevice, float>);
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("DecodeBoxes1").Device(DEVICE_GPU).TypeConstraint<float>("T"), DecodeBoxes1Op<GPUDevice, float>);
#endif
