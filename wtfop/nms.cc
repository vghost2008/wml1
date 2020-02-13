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
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * k:结果最多只包含k个
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("BoxesNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("classes_wise:bool")
	.Attr("k:int")
    .Input("bottom_box: T")
    .Input("classes: int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesNmsOp: public OpKernel {
	public:
		explicit BoxesNmsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
			OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THIS();
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr           = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end          = data_nr-1;

			for(auto i=0; i<loop_end; ++i) {
				if(keep_mask[i]) {
					const auto iclass = bottom_classes_flat(i);
					for(auto j=i+1; j<data_nr; ++j) {
						if(classes_wise && (bottom_classes_flat(j) != iclass)) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
					}
				}
			}

			const auto _out_size = count(keep_mask.begin(),keep_mask.end(),true);
			const auto out_size = k_>0?std::min<int>(k_,_out_size):_out_size;
			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();

			for(int i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
				if(j>=out_size)
					break;
			}
		}
	private:
		float threshold    = 0.2;
		bool  classes_wise = true;
		int   k_           = -1;
};
REGISTER_KERNEL_BUILDER(Name("BoxesNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesNmsOp<CPUDevice, float>);
/*
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * group:[Z,2]分组信息，分别为一个组里标签的开始与结束编号，不在分组信息的的默认为一个组
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("GroupBoxesNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
    .Input("bottom_box: T")
    .Input("classes: int32")
    .Input("group: int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class GroupBoxesNmsOp: public OpKernel {
	public:
		explicit GroupBoxesNmsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			const Tensor &_group              = context->input(2);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();
			auto          group               = _group.template tensor<int,2>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));
			OP_REQUIRES(context, _group.dims() == 2, errors::InvalidArgument("group data must be 2-dimensional"));

			const auto   data_nr           = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end          = data_nr-1;

			for(auto i=0; i<loop_end; ++i) {
				if(keep_mask[i]) {
					const auto iclass = bottom_classes_flat(i);
                    const auto igroup = get_group(group,iclass);
					for(auto j=i+1; j<data_nr; ++j) {
						if(igroup != get_group(group,bottom_classes_flat(j))) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
					}
				}
			}

			const auto out_size = count(keep_mask.begin(),keep_mask.end(),true);
			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();

			for(int i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
			}
		}
        template<typename TM>
        int get_group(const TM& group_data,int label)
        {
            for(auto i=0; i<group_data.dimension(0); ++i) {
                if((label>=group_data(i,0)) && (label<=group_data(i,1)))
                    return i;
            }
            return -1;
        }
	private:
		float threshold    = 0.2;
};
REGISTER_KERNEL_BUILDER(Name("GroupBoxesNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), GroupBoxesNmsOp<CPUDevice, float>);
/*
 * 数据不需要排序
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * confidence:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("BoxesSoftNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
	.Attr("classes_wise:bool")
	.Attr("delta:float")
    .Input("bottom_box: T")
    .Input("classes: int32")
    .Input("confidence:float")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesSoftNmsOp: public OpKernel {
    public:
        struct InterData
        {
            int index;
            float score;
            bool operator<(const InterData& v)const {
                return score<v.score;
            }
        };
        explicit BoxesSoftNmsOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold));
            OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise));
            OP_REQUIRES_OK(context, context->GetAttr("delta", &delta));
        }

        void Compute(OpKernelContext* context) override
        {
            TIME_THIS();
            const Tensor &bottom_box          = context->input(0);
            const Tensor &bottom_classes      = context->input(1);
            const Tensor &confidence          = context->input(2);
            auto          bottom_box_flat     = bottom_box.flat<T>();
            auto          bottom_classes_flat = bottom_classes.flat<int32>();
            auto          confidence_flat     = confidence.flat<float>();

            OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
            OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));
            OP_REQUIRES(context, confidence.dims() == 1, errors::InvalidArgument("confidence data must be 1-dimensional"));

            const auto   data_nr           = bottom_box.dim_size(0);
            vector<InterData> set_D(data_nr,InterData({0,0.0f}));
            vector<InterData> set_B;
            const auto   loop_end          = data_nr-1;

            for(auto i=0; i<data_nr; ++i) {
                set_D[i].index = i;
                set_D[i].score = confidence_flat.data()[i];
            }
            set_B.reserve(data_nr);

            for(auto i=0; i<data_nr; ++i) {
                auto it = max_element(set_D.begin(),set_D.end());
                if(it->score<threshold)
                    break;
                auto M = *it;
                set_D.erase(it);
                set_B.push_back(M);
                const auto index = M.index;
                const auto iclass = bottom_classes_flat(index);
                for(auto& data:set_D) {
                    const auto j = data.index;
                    if(classes_wise && (bottom_classes_flat(j) != iclass)) continue;
                    const auto iou = bboxes_jaccard(bottom_box_flat.data()+index*4,bottom_box_flat.data()+j*4);
                    if(iou>1e-2)
                        data.score *= exp(-iou*iou/delta);
                }
            }
            sort(set_B.begin(),set_B.end(),[](const InterData& lhv, const InterData& rhv){ return lhv.index<rhv.index;});
            const auto out_size = set_B.size();
            int dims_2d[2] = {int(out_size),4};
            int dims_1d[1] = {int(out_size)};
            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_box         = NULL;
            Tensor      *output_classes     = NULL;
            Tensor      *output_index = NULL;

            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

            auto obox     = output_box->template flat<T>();
            auto oclasses = output_classes->template flat<int32>();
            auto oindex   = output_index->template flat<int32>();

            for(auto j=0; j<out_size; ++j) {
                const auto i = set_B[j].index;
                auto box = bottom_box_flat.data()+i*4;
                std::copy(box,box+4,obox.data()+4*j);
                oclasses(j) = bottom_classes_flat(i);
                oindex(j) = i;
            }
        }
    private:
        float threshold    = 0.2;
        float delta        = 2.0;
        bool  classes_wise = true;
};
REGISTER_KERNEL_BUILDER(Name("BoxesSoftNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesSoftNmsOp<CPUDevice, float>);
/*
 * 与BoxesNms的主要区别为BoxesNmsNr使用输出人数来进行处理
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_indices:[X]
 * 输出时的相对位置不能改变
 * 程序会自动改变threshold的方式来使输出box的数量为k个
 */
REGISTER_OP("BoxesNmsNr")
    .Attr("T: {float, double,int32}")
	.Attr("classes_wise:bool")
	.Attr("k:int")
	.Attr("max_loop:int")
    .Input("bottom_box: T")
    .Input("classes:int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			int k = 0;
			c->GetAttr("k",&k);
			c->set_output(0, c->Matrix(k, 4));
			c->set_output(1, c->Vector(k));
			c->set_output(2, c->Vector(k));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesNmsNrOp: public OpKernel {
	public:
		explicit BoxesNmsNrOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise_));
			OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
			OP_REQUIRES_OK(context, context->GetAttr("max_loop", &max_loop_));
            if(k_ <= 0) k_ = 8;
            if(max_loop_ <= 0) max_loop_ = 4;
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THIS();
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr               = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			vector<bool> old_keep_mask(data_nr,true);
			const auto   loop_end              = data_nr-1;
			int          old_nr                = 0;

            auto loop_fn = [&](float threshold) {
                if(old_nr>k_)
				    std::swap(old_keep_mask,keep_mask);
				fill(keep_mask.begin(),keep_mask.end(),true);
                auto keep_nr = keep_mask.size();
				for(auto i=0; i<loop_end; ++i) {
					if(!keep_mask.at(i)) continue;
					const auto iclass = bottom_classes_flat(i);
					for(auto j=i+1; j<data_nr; ++j) {
						if(classes_wise_ && (bottom_classes_flat(j) != iclass)) continue;
						if(!keep_mask.at(j)) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
						--keep_nr;
					}
				}
				//cout<<"keep nr:"<<keep_nr<<", threshold "<<threshold<<endl;
			   //keep_nr = count(keep_mask.begin(),keep_mask.end(),true);
               return keep_nr;
            };

            auto threshold_low   = 0.0;
            auto threshold_hight = 1.0;

            for(auto i=0; i<max_loop_; ++i) {
                auto threshold = (threshold_low+threshold_hight)/2.0;
                auto nr = loop_fn(threshold);
                old_nr = nr;
                if(nr == k_) break;
                if(nr>k_)
                    threshold_hight = threshold;
                else
                    threshold_low = threshold;
            }

			auto out_size = count(keep_mask.begin(),keep_mask.end(),true);

            if(k_>data_nr) k_ = data_nr;

            if(out_size<k_) {
                auto delta = k_-out_size;
                for(auto it=keep_mask.begin(); it!=keep_mask.end(); ++it) 
                    if((*it) == false) {
                        (*it) = true;
                        --delta;
                        if(0 == delta) break;
                    }
                out_size = count(keep_mask.begin(),keep_mask.end(),true);
            } else if(out_size>k_) {
                auto nr = out_size-k_;
                for(auto it = keep_mask.rbegin(); it!=keep_mask.rend(); ++it) {
                    if((*it) == true) {
                        *it = false;
                        --nr;
                        if(0 == nr) break;
                    }
                }
            }
            out_size = k_;

			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();
			int  j        = 0;
			int  i        = 0;

			for(i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
				if(j>=out_size) break;
			}
            if(j<out_size) {
				cout<<"out size = "<<out_size<<", in size = "<<data_nr<<", j= "<<j<<std::endl;
                auto i = data_nr-1;
                for(;j<out_size; ++j) {
                    auto box = bottom_box_flat.data()+i*4;
                    std::copy(box,box+4,obox.data()+4*j);
                    oclasses(j) = bottom_classes_flat(i);
                    oindex(j) = i;
                }
            }
		}
	private:
		bool  classes_wise_ = true;
		int   k_            = 0;
		int   max_loop_     = 4;
};
REGISTER_KERNEL_BUILDER(Name("BoxesNmsNr").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesNmsNrOp<CPUDevice, float>);
/*
 * 与BoxesNmsNr的主要区别, 使用输入的theshold进行处理，选靠前的nr个boxes, 如果NMS后没有足够的boxes部分被删除的boxes会重新加入进来
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_indices:[X]
 * 输出时的相对位置不能改变
 * 程序会自动改变threshold的方式来使输出box的数量为k个
 */
REGISTER_OP("BoxesNmsNr2")
    .Attr("T: {float, double,int32}")
	.Attr("classes_wise:bool")
	.Attr("k:int")
	.Attr("threshold:float")
    .Input("bottom_box: T")
    .Input("classes:int32")
    .Input("confidence:T")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			int k = 0;
			c->GetAttr("k",&k);
			c->set_output(0, c->Matrix(k, 4));
			c->set_output(1, c->Vector(k));
			c->set_output(2, c->Vector(k));
			return Status::OK();
			});

template <typename Device, typename T>
class BoxesNmsNr2Op: public OpKernel {
    private:
        struct InterData
        {
            InterData(int i,float s):index(i),score(s){}
            int index;
            float score;
            bool operator<(const InterData& v)const {
                return score<v.score;
            }
        };
	public:
		explicit BoxesNmsNr2Op(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise_));
			OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
			OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
            if(k_ <= 0) k_ = 1;
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			const Tensor &confidence          = context->input(2);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();
			auto          confidence_flat     = confidence.flat<T>();


			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr               = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end              = data_nr-1;

            auto loop_fn = [&](float threshold) {
				for(auto i=0; i<loop_end; ++i) {
					if(!keep_mask.at(i)) continue;
					const auto iclass = bottom_classes_flat(i);
					for(auto j=i+1; j<data_nr; ++j) {
						if(classes_wise_ && (bottom_classes_flat(j) != iclass)) continue;
						if(!keep_mask.at(j)) continue;
						if(bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold) continue;
						keep_mask[j] = false;
					}
				}
            };

            if(k_>data_nr) {
                cout<<"Error input size is less than require ("<<data_nr<<" vs "<<k_<<")."<<endl;
            }

            loop_fn(threshold_);
			auto out_size = count(keep_mask.begin(),keep_mask.end(),true);

            if(out_size<k_) {
                auto delta = k_-out_size;
                vector<InterData> datas;
                datas.reserve((data_nr-out_size)/2);

                for(auto it=keep_mask.begin(); it!=keep_mask.end(); ++it) 
                    if((*it) == false) {
                        auto index = std::distance(keep_mask.begin(),it);
                        auto score = confidence_flat.data()[index];
                        for(auto kt=keep_mask.begin(); kt != it; ++kt) {
                            if((*kt) == false) 
                                continue;
                            auto index0 = std::distance(keep_mask.begin(),kt);
                            auto iou = bboxes_jaccard(bottom_box_flat.data()+index*4,bottom_box_flat.data()+index0*4);
                            score *= (1.0-iou);
                        }
                        datas.emplace_back(int(index),score);
                    }
                for(auto x=0; x<delta; ++x) {
                    auto jt = max_element(datas.begin(),datas.end());
                    const auto index = jt->index;
                    keep_mask[jt->index] = true;
                    datas.erase(jt);
                    for(auto jt=datas.begin(); jt!=datas.end(); ++jt) {
                        auto index0 = jt->index;
                        auto iou = bboxes_jaccard(bottom_box_flat.data()+index*4,bottom_box_flat.data()+index0*4);
                        jt->score *= (1.0-iou);
                    }
                }
          } else if(out_size>k_) {
                auto nr = out_size-k_;
                for(auto it = keep_mask.rbegin(); it!=keep_mask.rend(); ++it) {
                    if((*it) == true) {
                        *it = false;
                        --nr;
                        if(0 == nr) break;
                    }
                }
            }
            out_size = k_;

			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();
			int  j        = 0;
			int  i        = 0;

            obox.setZero();
            oindex.setZero();
            oclasses.setZero();

			for(i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
				if(j>=out_size) break;
			}
            if(j<out_size) {
				cout<<"out size = "<<out_size<<", in size = "<<data_nr<<", j= "<<j<<std::endl;
                auto i = data_nr-1;
                for(;j<out_size; ++j) {
                    auto box = bottom_box_flat.data()+i*4;
                    std::copy(box,box+4,obox.data()+4*j);
                    oclasses(j) = bottom_classes_flat(i);
                    oindex(j) = i;
                }
            }
		}
	private:
		bool  classes_wise_ = true;
		float threshold_    = 0.0;
		int   k_            = 0;
};
REGISTER_KERNEL_BUILDER(Name("BoxesNmsNr2").Device(DEVICE_CPU).TypeConstraint<float>("T"), BoxesNmsNr2Op<CPUDevice, float>);
/*
 * 要求数据已经按重要呈度从0到data_nr排好了序(从大到小）
 * 用于处理目标不会重叠的情况
 * bottom_box:[X,4](ymin,xmin,ymax,xmax)
 * classes:[X]
 * output_box:[Y,4]
 * output_classes:[Y]
 * output_index:[Y]
 * 输出时的相对位置不能改变
 */
REGISTER_OP("NoOverlapBoxesNms")
    .Attr("T: {float, double,int32}")
	.Attr("threshold0:float")
	.Attr("threshold1:float")
	.Attr("classes_wise:bool")
    .Input("bottom_box: T")
    .Input("classes: int32")
	.Output("output_box:T")
	.Output("output_classes:int32")
	.Output("output_index:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			c->set_output(0, c->Matrix(-1, 4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			return Status::OK();
			});

template <typename Device, typename T>
class NoOverlapBoxesNmsOp: public OpKernel {
	public:
		explicit NoOverlapBoxesNmsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("threshold0", &threshold0));
			OP_REQUIRES_OK(context, context->GetAttr("threshold1", &threshold1));
			OP_REQUIRES_OK(context, context->GetAttr("classes_wise", &classes_wise));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THIS();
			const Tensor &bottom_box          = context->input(0);
			const Tensor &bottom_classes      = context->input(1);
			auto          bottom_box_flat     = bottom_box.flat<T>();
			auto          bottom_classes_flat = bottom_classes.flat<int32>();

			OP_REQUIRES(context, bottom_box.dims() == 2, errors::InvalidArgument("box data must be 2-dimensional"));
			OP_REQUIRES(context, bottom_classes.dims() == 1, errors::InvalidArgument("classes data must be 1-dimensional"));

			const auto   data_nr           = bottom_box.dim_size(0);
			vector<bool> keep_mask(data_nr,true);
			const auto   loop_end          = data_nr-1;

			for(auto i=0; i<loop_end; ++i) {
				if(keep_mask[i]) {
					const auto iclass = bottom_classes_flat(i);
                    for(auto j=i+1; j<data_nr; ++j) {
                        if(classes_wise && (bottom_classes_flat(j) != iclass)) continue;
                        if((bboxes_jaccard(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4) < threshold0)
                                && (bboxes_jaccard_of_box0(bottom_box_flat.data()+i*4,bottom_box_flat.data()+j*4)<threshold1)
                                && (bboxes_jaccard_of_box0(bottom_box_flat.data()+j*4,bottom_box_flat.data()+i*4)<threshold1)
                          ) continue;
                        keep_mask[j] = false;
                    }
				}
			}

			const auto out_size = count(keep_mask.begin(),keep_mask.end(),true);
			int dims_2d[2] = {int(out_size),4};
			int dims_1d[1] = {int(out_size)};
			TensorShape  outshape0;
			TensorShape  outshape1;
			Tensor      *output_box         = NULL;
			Tensor      *output_classes     = NULL;
			Tensor      *output_index = NULL;

			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_box));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_classes));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape1, &output_index));

			auto obox     = output_box->template flat<T>();
			auto oclasses = output_classes->template flat<int32>();
			auto oindex   = output_index->template flat<int32>();

			for(int i=0,j=0; i<data_nr; ++i) {
				if(!keep_mask[i]) continue;
				auto box = bottom_box_flat.data()+i*4;
				std::copy(box,box+4,obox.data()+4*j);
				oclasses(j) = bottom_classes_flat(i);
				oindex(j) = i;
				++j;
			}
		}
	private:
		float threshold0    = 0.2;
		float threshold1    = 0.8;
		bool  classes_wise = true;
};
REGISTER_KERNEL_BUILDER(Name("NoOverlapBoxesNms").Device(DEVICE_CPU).TypeConstraint<float>("T"), NoOverlapBoxesNmsOp<CPUDevice, float>);
