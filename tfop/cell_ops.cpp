#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <future>
#include <assert.h>
#include <boost/algorithm/clamp.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
REGISTER_OP("CellEncodeLabel")
    .Attr("T: {int32, int64}")
 	.Attr("num_classes: int")
    .Input("tensor: T")
	.Output("tensor_o:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			const auto batch_size = c->Dim(c->input(0),0);
            int num_classes = 0;
            c->GetAttr("num_classes",&num_classes);
            auto shape = c->MakeShape({batch_size,num_classes});
            c->set_output(0,shape);
    return Status::OK();
    });

template <typename Device, typename T>
class CellEncodeLabelOp: public OpKernel {
	public:
		explicit CellEncodeLabelOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
            //第一级分鳞，腺，TRI,CC,....
            //0-7表示第一级
            label_map_level0_[1] = 0;
            label_map_level0_[2] = 0;
            label_map_level0_[3] = 2;
            label_map_level0_[4] = 3;
            label_map_level0_[5] = 1;
            label_map_level0_[6] = 4;
            label_map_level0_[7] = 5;
            label_map_level0_[8] = 0;
            label_map_level0_[9] = 6;
            label_map_level0_[10] = 7;
            label_map_level0_[11] = 0;
            label_map_level0_[12] = 0;
            //第二级分低级别鳞状病变,高级别鳞状病变
            //8-9表示第二级
            label_map_level1_[1] = 9;
            label_map_level1_[11] = 9;
            label_map_level1_[2] = 8;
            label_map_level1_[8] = 8;
            label_map_level1_[12] = 8;
            //第三级分（ASCUS,LSIL0,(ASCH,HSIL,SCC)
            //10-14表示第三级
            label_map_level2_[1] = 11;
            label_map_level2_[11] = 10;
            label_map_level2_[2] = 13;
            label_map_level2_[8] = 14;
            label_map_level2_[12] = 12;
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor     = context->input(0);
            auto          tensor_flat = _tensor.flat<T>().data();
            const auto    batch_size  = _tensor.dim_size(0);
            const auto    data_nr     = _tensor.NumElements();

			OP_REQUIRES(context, _tensor.dims() == 1, errors::InvalidArgument("input must be 1-dimensional"));

            int dims_2d[] = {batch_size,num_classes_};

            TensorShape outshape;
            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

            Tensor      *output_tensor = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,outshape,&output_tensor));

            auto o_tensor = output_tensor->template tensor<T,2>();

            o_tensor.setZero();

            for(auto i=0; i<data_nr; ++i) {
                const auto l = tensor_flat[i];

                if(0 == l)
                    continue;

                auto kt = label_map_level0_.find(l);
                if(kt == label_map_level0_.end()) {
                    cout<<"Error label "<<l<<endl;
                    continue;
                }

                o_tensor(i,kt->second) = 1;

                auto it = label_map_level1_.find(l);

                if(label_map_level1_.end() == it) 
                    continue;
                o_tensor(i,it->second) = 1;
                auto jt = label_map_level2_.find(l);
                if(label_map_level2_.end() == jt) 
                    continue;
                o_tensor(i,jt->second) = 1;
            }
        }
	private:
        int num_classes_ = 0;
        unordered_map<int,int> label_map_level0_;
        unordered_map<int,int> label_map_level1_;
        unordered_map<int,int> label_map_level2_;
};
REGISTER_KERNEL_BUILDER(Name("CellEncodeLabel").Device(DEVICE_CPU).TypeConstraint<int>("T"), CellEncodeLabelOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("CellEncodeLabel").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), CellEncodeLabelOp<CPUDevice, tensorflow::int64>);

REGISTER_OP("CellDecodeLabel")
    .Attr("T: {float, double}")
    .Input("tensor: T") //probability
	.Output("tensor_o:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class CellDecodeLabelOp: public OpKernel {
	public:
		explicit CellDecodeLabelOp(OpKernelConstruction* context) : OpKernel(context) {
            //第一级分鳞，腺，TRI,CC,....
            //0-7表示第一级
            label_map_level0_[1] = 5;
            label_map_level0_[2] = 3;
            label_map_level0_[3] = 4;
            label_map_level0_[4] = 6;
            label_map_level0_[5] = 7;
            label_map_level0_[6] = 9;
            label_map_level0_[7] = 10;
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor    = context->input(0);

			OP_REQUIRES(context, _tensor.dims() == 2, errors::InvalidArgument("input must be 2-dimensional"));

            auto          tensor     = _tensor.template tensor<T,2>();
            const auto    batch_size = _tensor.dim_size(0);
            const auto    C          = _tensor.dim_size(1);
            const auto    kThreadNr  = 512;


            TensorShape  output_shape  = _tensor.shape();
            Tensor      *output_tensor = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

            auto o_tensor = output_tensor->template tensor<T,2>();
            auto i_data = _tensor.template flat<T>().data();

            o_tensor.setZero();

            auto fn = [&](int begin,int end) {
                for(auto i=begin; i<end; ++i) {
                    for(auto it=label_map_level0_.begin(); it!=label_map_level0_.end(); ++it) {
                        o_tensor(i,it->second) = tensor(i,it->first);
                    }
                    if(tensor(i,8)>=tensor(i,9)) { //H
                        auto d = i_data+i*C+12;
                        auto it = max_element(d,d+3);
                        auto dis = it-d;
                        //auto p = max<T>(d[dis],tensor(i,0));
                        //auto p = tensor(i,0);
                        auto p = min<T>(d[dis],tensor(i,0));

                        if(dis==0){
                            o_tensor(i,12) = p;
                        } else if(dis==1) {
                            o_tensor(i,2) = p;
                        } else {
                            o_tensor(i,8) = p;
                        }
                    } else {
                        if(tensor(i,10)>=tensor(i,11)) {
                            //o_tensor(i,11) = max<T>(tensor(i,0),tensor(i,10));
                            //o_tensor(i,11) = tensor(i,0);
                            o_tensor(i,11) = min<T>(tensor(i,0),tensor(i,10));
                        } else {
                            //o_tensor(i,1) = max<T>(tensor(i,0),tensor(i,11));
                            //o_tensor(i,1) = tensor(i,0);
                            o_tensor(i,1) = min<T>(tensor(i,0),tensor(i,11));
                        }
                    }
                }
            };

            list<future<void>> furs;

            for(auto i=0; i<batch_size; i += kThreadNr) {
                furs.push_back(std::async(std::launch::async,fn,i,min<int>(i+kThreadNr,batch_size)));
            }
        }
	private:
        unordered_map<int,int> label_map_level0_;
};
REGISTER_KERNEL_BUILDER(Name("CellDecodeLabel").Device(DEVICE_CPU).TypeConstraint<float>("T"), CellDecodeLabelOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CellDecodeLabel").Device(DEVICE_CPU).TypeConstraint<double>("T"), CellDecodeLabelOp<CPUDevice, double>);

typedef Eigen::ThreadPoolDevice CPUDevice;
REGISTER_OP("CellEncodeLabel2")
    .Attr("T: {int32, int64}")
    .Input("tensor: T")
	.Output("tensor_o0:T")
	.Output("tensor_o1:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			const auto batch_size = c->Dim(c->input(0),0);
            int num_classes = 0;
            auto shape = c->MakeShape({batch_size});
            c->set_output(0,shape);
            auto shape1 = c->MakeShape({batch_size});
            c->set_output(1,shape1);
    return Status::OK();
    });

template <typename Device, typename T>
class CellEncodeLabel2Op: public OpKernel {
	public:
		explicit CellEncodeLabel2Op(OpKernelConstruction* context) : OpKernel(context) {
            //第一级分鳞，腺，TRI,CC,....
            //0-8表示第一级, 0 表示背景
            label_map_level0_[1] = 8;
            label_map_level0_[2] = 8;
            label_map_level0_[3] = 2;
            label_map_level0_[4] = 3;
            label_map_level0_[5] = 1;
            label_map_level0_[6] = 4;
            label_map_level0_[7] = 5;
            label_map_level0_[8] = 8;
            label_map_level0_[9] = 6;
            label_map_level0_[10] = 7;
            label_map_level0_[11] = 8;
            label_map_level0_[12] = 8;
            //第二级分低级别鳞状病变,高级别鳞状病变
            //8-9表示第二级
            label_map_level1_[11] = 0;
            label_map_level1_[1] = 1;
            label_map_level1_[12] = 3; //增加了LSIL与ASCH之间的距离
            label_map_level1_[2] = 4;
            label_map_level1_[8] = 5;
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor     = context->input(0);
            auto          tensor_flat = _tensor.flat<T>().data();
            const auto    batch_size  = _tensor.dim_size(0);
            const auto    data_nr     = _tensor.NumElements();

			OP_REQUIRES(context, _tensor.dims() == 1, errors::InvalidArgument("input must be 1-dimensional"));

            int dims_1d0[] = {batch_size};
            int dims_1d1[] = {batch_size};

            TensorShape outshape0;
            TensorShape outshape1;
            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d1, 1, &outshape1);

            Tensor      *output_tensor0 = nullptr;
            Tensor      *output_tensor1 = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,outshape0,&output_tensor0));
            OP_REQUIRES_OK(context,context->allocate_output(1,outshape1,&output_tensor1));

            auto o_tensor0 = output_tensor0->template tensor<T,1>();
            auto o_tensor1 = output_tensor1->template tensor<T,1>();

            o_tensor0.setZero();
            o_tensor1.setZero();

            for(auto i=0; i<data_nr; ++i) {
                const auto l = tensor_flat[i];
                auto tp = label_map_level0_[l];

                o_tensor0(i) = tp;
                if(8 == tp) {
                    auto it = label_map_level1_.find(l);

                    if(label_map_level1_.end() == it) {
                        cout<<"Error label "<<l<<endl;
                        continue;
                    }
                    o_tensor1(i) = it->second;
                } else {
                    o_tensor1(i) = -1; //如果不是鳞状病变，默认为-1
                }
            }
        }
	private:
        unordered_map<int,int> label_map_level0_;
        unordered_map<int,int> label_map_level1_;
};
REGISTER_KERNEL_BUILDER(Name("CellEncodeLabel2").Device(DEVICE_CPU).TypeConstraint<int>("T"), CellEncodeLabel2Op<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("CellEncodeLabel2").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), CellEncodeLabel2Op<CPUDevice, tensorflow::int64>);

REGISTER_OP("CellDecodeLabel2")
    .Attr("T: {float, double}")
 	.Attr("num_classes: int")
    .Input("tensor0: T") //probability
    .Input("tensor1: T") //regs
	.Output("tensor_o:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			const auto batch_size = c->Dim(c->input(0),0);
            int num_classes = 0;
            c->GetAttr("num_classes",&num_classes);
            auto shape = c->MakeShape({batch_size,num_classes});
            c->set_output(0,shape);
    return Status::OK();
    });

template <typename Device, typename T>
class CellDecodeLabel2Op: public OpKernel {
	public:
		explicit CellDecodeLabel2Op(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("num_classes", &num_classes_));
            //第一级分鳞，腺，TRI,CC,....
            //0-7表示第一级
            label_map_level0_[1] = 5;
            label_map_level0_[2] = 3;
            label_map_level0_[3] = 4;
            label_map_level0_[4] = 6;
            label_map_level0_[5] = 7;
            label_map_level0_[6] = 9;
            label_map_level0_[7] = 10;
            //
            label_map_level1_[0] = 11;
            label_map_level1_[1] = 1;
            label_map_level1_[3] = 12;
            label_map_level1_[4] = 2;
            label_map_level1_[5] = 8;
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor0    = context->input(0);
            const Tensor &_tensor1    = context->input(1);

			OP_REQUIRES(context, _tensor0.dims() == 2, errors::InvalidArgument("input0 must be 2-dimensional"));
			OP_REQUIRES(context, _tensor1.dims() == 2, errors::InvalidArgument("input1 must be 2-dimensional"));

            auto          tensor0     = _tensor0.template tensor<T,2>();
            auto          tensor1     = _tensor1.template tensor<T,2>();
            const auto    batch_size = _tensor0.dim_size(0);
            const auto    kThreadNr  = 512;


            int dims_2d[] = {batch_size,num_classes_};
            TensorShape outshape;
            Tensor      *output_tensor = nullptr;

            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape);

            OP_REQUIRES_OK(context,context->allocate_output(0,outshape,&output_tensor));

            auto o_tensor = output_tensor->template tensor<T,2>();

            o_tensor.setZero();

            auto fn = [&](int begin,int end) {
                for(auto i=begin; i<end; ++i) {
                    for(auto it=label_map_level0_.begin(); it!=label_map_level0_.end(); ++it) {
                        o_tensor(i,it->second) = tensor0(i,it->first);
                    }
                    auto index = int(tensor1(i,0)+0.5);
                    float scale = 1.0;
                    if(2 == index) {
                        index = int(tensor1(i,0));
                        if(2 == index)
                            index = 3;
                    } else if(index<=-1) {
                        scale = 0.5;
                    }
                    index = max(min(index,5),0);
                    auto idx = label_map_level1_[index];
                    o_tensor(i,idx) = tensor0(i,8)*scale;
                }
            };

            list<future<void>> furs;

            for(auto i=0; i<batch_size; i += kThreadNr) {
                furs.push_back(std::async(std::launch::async,fn,i,min<int>(i+kThreadNr,batch_size)));
            }
        }
	private:
        unordered_map<int,int> label_map_level0_;
        unordered_map<int,int> label_map_level1_;
        int num_classes_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("CellDecodeLabel2").Device(DEVICE_CPU).TypeConstraint<float>("T"), CellDecodeLabel2Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("CellDecodeLabel2").Device(DEVICE_CPU).TypeConstraint<double>("T"), CellDecodeLabel2Op<CPUDevice, double>);
