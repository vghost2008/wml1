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
#include "bboxes_encode.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

/*
 * max_overlap_as_pos:是否将与ground truth bbox交叉面积最大的box设置为正样本
 * force_in_gtbox: 是否强制正样本的中心点必须在相应的gtbox内
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
    .Attr("force_in_gtbox:bool=False")
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
			OP_REQUIRES_OK(context, context->GetAttr("force_in_gtbox", &force_in_gtbox_));
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

            MatcherUnit<CPUDevice,T> encode_unit(pos_threshold,neg_threshold,max_overlap_as_pos_,force_in_gtbox_);
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
		float pos_threshold;
		float neg_threshold;
		bool  max_overlap_as_pos_ = true;
		bool  force_in_gtbox_     = false;
};
#ifdef GOOGLE_CUDA
template <typename T>
class MatcherOp<GPUDevice,T>: public OpKernel {
	public:
		explicit MatcherOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("pos_threshold", &pos_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("neg_threshold", &neg_threshold));
			OP_REQUIRES_OK(context, context->GetAttr("max_overlap_as_pos", &max_overlap_as_pos_));
			OP_REQUIRES_OK(context, context->GetAttr("force_in_gtbox", &force_in_gtbox_));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("MatcherGPU");
			const Tensor &_bottom_boxes   = context->input(0);
			const Tensor &_bottom_gboxes  = context->input(1);
			const Tensor &_bottom_glabels = context->input(2);
			const Tensor &_bottom_gsize   = context->input(3);

			OP_REQUIRES(context, _bottom_boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_gboxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _bottom_glabels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _bottom_gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimensional"));

			auto          bottom_boxes    = _bottom_boxes.template tensor<T,3>();
			auto          bottom_gboxes   = _bottom_gboxes.template tensor<T,3>();
			auto          bottom_glabels  = _bottom_glabels.template tensor<int,2>();
			auto          d_bottom_gsize  = _bottom_gsize.template tensor<int,1>();
			const auto    batch_size      = _bottom_gboxes.dim_size(0);
			const auto    data_nr         = _bottom_boxes.dim_size(1);
		    Eigen::Tensor<int,1,Eigen::RowMajor> bottom_gsize;
            assign_tensor<CPUDevice,GPUDevice>(bottom_gsize,d_bottom_gsize);


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

            MatcherUnit<GPUDevice,T> encode_unit(pos_threshold,neg_threshold,max_overlap_as_pos_,force_in_gtbox_);

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
		float pos_threshold;
		float neg_threshold;
		bool  max_overlap_as_pos_ = true;
		bool  force_in_gtbox_     = false;
};
#endif
REGISTER_KERNEL_BUILDER(Name("Matcher").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatcherOp<CPUDevice, float>);
#ifdef GOOGLE_CUDA
//REGISTER_KERNEL_BUILDER(Name("Matcher").Device(DEVICE_GPU).TypeConstraint<float>("T"), MatcherOp<GPUDevice, float>);
#endif
/*
 * 将boxes中与gboxes IOU最大且不小于threadshold的标记为相应的label, 即1个gboxes最多与一个boxes相对应
 * boxes:[batch_size,N,4](y,x,h,w)
 * gboxes:[batch_size,M,4]
 * glabels:[batch_size,M] 0表示背景
 * glens:[batch_size] 用于表明gboxes中的有效boxes数量
 * output_labels:[batch_size,N]
 * output_scores:[batch_size,N]
 * output_indexs:[batch_size,N] #gbboxes的index
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
	.Output("output_indexs:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			c->set_output(2, outshape);
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
			Tensor      *output_indexs      = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape, &output_indexs));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();
			auto oindexs      = output_indexs->template tensor<int,2>();

            oclasses.setZero();
            oscores.setZero();
            oindexs.setConstant(-1);

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
                        oindexs(i,index) = j;
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
 * output_indexs:[batch_size,N] #gbboxes的index
 * threashold: IOU threshold
 */
REGISTER_OP("BoxesMatchWithPred")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("is_binary_plabels:bool=False")
    .Input("boxes: T")
    .Input("plabels: int32")
    .Input("pprobs: T")
    .Input("gboxes: T")
	.Input("glabels:int32")
	.Input("glens:int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("output_indexs:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			c->set_output(2, outshape);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesMatchWithPredOp: public OpKernel {
    private:
        struct PInfo {
            int index;
            float probs;
        };
	public:
		explicit BoxesMatchWithPredOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
			OP_REQUIRES_OK(context, context->GetAttr("is_binary_plabels", &is_binary_plabels_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_boxes   = context->input(0);
			const Tensor &_plabels = context->input(1);
			const Tensor &_pprobs = context->input(2);
			const Tensor &_gboxes  = context->input(3);
			const Tensor &_glabels = context->input(4);
			const Tensor &_glens   = context->input(5);

			OP_REQUIRES(context, _boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _gboxes.dims() == 3, errors::InvalidArgument("gboxes data must be 3-dimensional"));
			OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("glabels data must be 2-dimensional"));
			OP_REQUIRES(context, _plabels.dims() == 2, errors::InvalidArgument("plabels data must be 2-dimensional"));
			OP_REQUIRES(context, _pprobs.dims() == 2, errors::InvalidArgument("pprobs data must be 2-dimensional"));
			OP_REQUIRES(context, _glens.dims() == 1, errors::InvalidArgument("glens data must be 1-dimensional"));

			auto          boxes    = _boxes.tensor<T,3>();
			auto          plabels  = _plabels.tensor<int,2>();
			auto          pprobs   = _pprobs.tensor<T,2>();
			auto          gboxes   = _gboxes.tensor<T,3>();
			auto          glabels  = _glabels.tensor<int,2>();
			auto          glens    = _glens.tensor<int,1>();

			const int batch_nr  = _boxes.dim_size(0);
			const int boxes_nr  = _boxes.dim_size(1);
			const int gboxes_nr = _gboxes.dim_size(1);
            Eigen::Tensor<bool,2> process_mask(batch_nr,glabels.dimension(1));
			int dims_2d[2] = {batch_nr,boxes_nr};
			TensorShape  outshape;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;
			Tensor      *output_indexs      = NULL;

            if(plabels.dimension(1) != boxes.dimension(1)) {
                cout<<"Error plabels dimension 1"<<endl;
            }

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape, &output_indexs));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();
			auto oindexs      = output_indexs->template tensor<int,2>();

            process_mask.setZero();
            oclasses.setZero();
            oscores.setZero();
            oindexs.setConstant(-1);

            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;
                    vector<PInfo> infos(boxes_nr);
                    for(auto k=0; k<boxes_nr; ++k) {
                        infos[k].index = k;
                        infos[k].probs = pprobs(i,k);
                    }
                    sort(infos.begin(),infos.end(),[](const auto& lhv,const auto& rhv){ return rhv.probs<lhv.probs;});

                    for(auto _k=0; _k<boxes_nr; ++_k) {
                        const auto k = infos[_k].index;
                        const auto plabel = plabels(i,k);
                        if((is_binary_plabels_ && (plabel == 0)) || ((!is_binary_plabels_) && (plabel != glabels(i,j))))
                            continue;
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
                        oindexs(i,index) = j;
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
                        oindexs(i,index) = j;
                    }
                }
            }
		}
	private:
		float threshold_;
        bool is_binary_plabels_ = false;
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
 * output_indexs:[batch_size,N] #gbboxes的index
 * threashold: IOU threshold
 */
REGISTER_OP("BoxesMatchWithPredV3")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("is_binary_plabels:bool=False")
	.Attr("sort_by_probs:bool=True")
    .Input("boxes: T")
    .Input("plabels: int32")
    .Input("pprobs: T")
    .Input("gboxes: T")
	.Input("glabels:int32")
	.Input("glens:int32")
	.Output("output_labels:int32")
	.Output("output_scores:T")
	.Output("output_indexs:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto boxes_shape = c->input(0);
			shape_inference::ShapeHandle outshape;
			c->Subshape(boxes_shape,0,2,&outshape);
			c->set_output(0, outshape);
			c->set_output(1, outshape);
			c->set_output(2, outshape);
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesMatchWithPredV3Op: public OpKernel {
    private:
        struct PInfo {
            int index;
            float probs;
        };
	public:
		explicit BoxesMatchWithPredV3Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
			OP_REQUIRES_OK(context, context->GetAttr("is_binary_plabels", &is_binary_plabels_));
			OP_REQUIRES_OK(context, context->GetAttr("sort_by_probs", &sort_by_probs_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_boxes   = context->input(0);
			const Tensor &_plabels = context->input(1);
			const Tensor &_pprobs = context->input(2);
			const Tensor &_gboxes  = context->input(3);
			const Tensor &_glabels = context->input(4);
			const Tensor &_glens   = context->input(5);

			OP_REQUIRES(context, _boxes.dims() == 3, errors::InvalidArgument("box data must be 3-dimensional"));
			OP_REQUIRES(context, _gboxes.dims() == 3, errors::InvalidArgument("gboxes data must be 3-dimensional"));
			OP_REQUIRES(context, _glabels.dims() == 2, errors::InvalidArgument("glabels data must be 2-dimensional"));
			OP_REQUIRES(context, _plabels.dims() == 2, errors::InvalidArgument("plabels data must be 2-dimensional"));
			OP_REQUIRES(context, _pprobs.dims() == 2, errors::InvalidArgument("pprobs data must be 2-dimensional"));
			OP_REQUIRES(context, _glens.dims() == 1, errors::InvalidArgument("glens data must be 1-dimensional"));

			auto          boxes    = _boxes.tensor<T,3>();
			auto          plabels  = _plabels.tensor<int,2>();
			auto          pprobs   = _pprobs.tensor<T,2>();
			auto          gboxes   = _gboxes.tensor<T,3>();
			auto          glabels  = _glabels.tensor<int,2>();
			auto          glens    = _glens.tensor<int,1>();

			const int batch_nr  = _boxes.dim_size(0);
			const int boxes_nr  = _boxes.dim_size(1);
			const int gboxes_nr = _gboxes.dim_size(1);
			int dims_2d[2] = {batch_nr,boxes_nr};
			TensorShape  outshape;
			Tensor      *output_classes     = NULL;
			Tensor      *output_scores      = NULL;
			Tensor      *output_indexs      = NULL;

            if(plabels.dimension(1) != boxes.dimension(1)) {
                cout<<"Error plabels dimension 1"<<endl;
            }

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape, &output_scores));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape, &output_indexs));

			auto oclasses     = output_classes->template tensor<int,2>();
			auto oscores      = output_scores->template tensor<T,2>();
			auto oindexs      = output_indexs->template tensor<int,2>();

            oclasses.setZero();
            oscores.setZero();
            oindexs.setConstant(-1);

            for(auto i=0; i<batch_nr; ++i) {
                for(auto j=0; j<glens(i); ++j) {
                    auto max_scores = -1.0;
                    auto index      = -1;
                    vector<PInfo> infos(boxes_nr);
                    for(auto k=0; k<boxes_nr; ++k) {
                        infos[k].index = k;
                        infos[k].probs = pprobs(i,k);
                    }
                    if(sort_by_probs_)
                        sort(infos.begin(),infos.end(),[](const auto& lhv,const auto& rhv){ return rhv.probs<lhv.probs;});

                    for(auto _k=0; _k<boxes_nr; ++_k) {
                        const auto k = infos[_k].index;
                        const auto plabel = plabels(i,k);
                        if((is_binary_plabels_ && (plabel == 0)) || ((!is_binary_plabels_) && (plabel != glabels(i,j))))
                            continue;
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
                        oindexs(i,index) = j;
                    }
                }
            }
		}
	private:
		float threshold_;
        bool is_binary_plabels_ = false;
        bool sort_by_probs_ = false;
};
REGISTER_KERNEL_BUILDER(Name("BoxesMatchWithPredV3").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesMatchWithPredV3Op<CPUDevice, float>);
/*
 * 如果boxes与gboxes的iou>threshold[0], 并且交叉面积占gboxes的面积大小threshold[1]的为正样本，其余的为负样本
 * bottom_boxes:[1,X,4]/[batch_size,X,4](ymin,xmin,ymax,xmax) 候选box,相对坐标
 * bottom_gboxes:[batch_size,Y,4](ymin,xmin,ymax,xmax)ground truth box相对坐标
 * bottom_glabels:[batch_size,Y] 0为背景
 * bottom_glength:[batch_size] 为每一个batch中gboxes的有效数量
 * output_labels:[batch_size,X], 当前anchorbox的标签，背景为0,不为背景时为相应最大jaccard得分,-1表示忽略
 * output_scores:[batch_size,X], 当前anchorbox与groundtruthbox的jaccard得分，当jaccard得分高于threshold时就不为背影
 * output_indict:[batch_size,X], 当anchorbox有效时，与它对应的gboxes(从0开始)序号,无效时为-1
 */
REGISTER_OP("MatcherV2")
    .Attr("T: {float,double,int32,int64}")
	.Attr("threshold:list(float)")
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
class MatcherV2Op: public OpKernel {
};
template <typename T>
class MatcherV2Op<CPUDevice,T>: public OpKernel {
	public:
		explicit MatcherV2Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("MatcherV2");
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

            MatcherV2Unit<CPUDevice,T> encode_unit(threshold_);
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
		vector<float> threshold_;
};
REGISTER_KERNEL_BUILDER(Name("MatcherV2").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatcherV2Op<CPUDevice, float>);
