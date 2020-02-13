#include <stdio.h>
#include <cfloat>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <stdint.h>
#include <boost/algorithm/clamp.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"
#include "bboxes.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
REGISTER_OP("DistoredQa")
    .Attr("T: {int32, int64}")
	.Attr("expand_nr:int")
    .Input("question: T")
    .Input("answer: T")
	.Output("output_question:T")
	.Output("output_answer:T")
	.Output("output_labels:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			auto dims_data0  =  c->input(0);
			auto dims_data1  =  c->input(1);
			auto expand_nr   =  0;

			c->GetAttr("expand_nr",&expand_nr);

			auto batch_size     =  c->Value(c->Dim(dims_data0,0))*(1+expand_nr);
			auto output_shape0  =  c->Matrix(batch_size,c->Dim(dims_data0,1));
			auto output_shape1  =  c->Matrix(batch_size,c->Dim(dims_data1,1));
			auto output_shape2  =  c->Vector(batch_size);

			c->set_output(0,output_shape0);
			c->set_output(1,output_shape1);
			c->set_output(2,output_shape2);
			return Status::OK();
			});

template <typename Device, typename T>
class DistoredQaOp: public OpKernel {
	public:
		explicit DistoredQaOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("expand_nr", &expand_nr));
			srand(::time(nullptr));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &question= context->input(0);
			const Tensor &answer = context->input(1);
			auto          question_flat = question.flat<T>();
			auto          answer_flat = answer.flat<T>();

			OP_REQUIRES(context, question.dims() == 2, errors::InvalidArgument("question data must be 2-dimensional"));
			OP_REQUIRES(context, answer.dims() == 2, errors::InvalidArgument("answer data must be 2-dimensional"));

			const auto batch_size   = question.dim_size(0);
			const auto num_output   = batch_size *(1+expand_nr);
			const auto question_len = question.dim_size(1);
			const auto answer_len   = answer.dim_size(1);

			TensorShape output_shape0({num_output,question_len});
			TensorShape output_shape1({num_output,answer_len});
			TensorShape output_shape2({num_output});

			Tensor* output_question = NULL;
			Tensor* output_answer = NULL;
			Tensor* output_labels= NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_question));
			OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_answer));
			OP_REQUIRES_OK(context, context->allocate_output(2, output_shape2, &output_labels));

			auto oq_flat = output_question->flat<T>();
			auto oa_flat = output_answer->flat<T>();
			auto ol_flat = output_labels->flat<T>();

			for(auto i=0; i<batch_size; ++i) {
				auto bq_i = question_flat.data()+question_len*i;
				auto ba_i = answer_flat.data()+answer_len*i;
				auto bq_o = oq_flat.data()+question_len*i*(expand_nr+1);
				auto ba_o = oa_flat.data()+answer_len*i*(expand_nr+1);
				auto bl_o = ol_flat.data()+i*(expand_nr+1);
				for(auto j=0; j<question_len; ++j) {
					bq_o[j] = bq_i[j];
				}
				for(auto j=0; j<answer_len; ++j) {
					ba_o[j] = ba_i[j];
				}
				*bl_o = 1;
				for(auto k=0; k<expand_nr; ++k) {
					bq_o += question_len;
					ba_o += answer_len;
					bl_o += 1;

					auto index = rand()%(batch_size-1);
					if(index==i)
						++index;

					ba_i = answer_flat.data()+answer_len*index;

					for(auto j=0; j<question_len; ++j) {
						bq_o[j] = bq_i[j];
					}
					for(auto j=0; j<answer_len; ++j) {
						ba_o[j] = ba_i[j];
					}

					*bl_o = 0;
				}
			}
		}
	private:
		int expand_nr;
};
REGISTER_KERNEL_BUILDER(Name("DistoredQa").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), DistoredQaOp<CPUDevice, int32_t>);

/*
 * 方法来源:ATTENTION IS ALL YOU NEED
 */
REGISTER_OP("PositionEmbedding")
    .Attr("T: {int32, int64}")
    .Input("size: T")
	.Output("output:float");

template <typename Device, typename T>
class PositionEmbeddingOp: public OpKernel {
	public:
		explicit PositionEmbeddingOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_size = context->input(0);
            const auto    size  = _size.template tensor<T,1>();

            OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size data must be 1-dimensional"));
            OP_REQUIRES(context, _size.dim_size(0) == 2, errors::InvalidArgument("size dim size must be 2."));

            const auto batch_size     = 1;
            const auto max_seq_len    = size(0);
            const auto embedding_size = size(1);

            TensorShape output_shape0({batch_size,max_seq_len,embedding_size});
            Tensor* output= NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output));

            auto       o_tensor = output->template tensor<float,3>();
            const auto h_es     = embedding_size/2;

            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<max_seq_len; ++j) {
                    for(auto k=0; k<h_es; ++k) {
                        const auto v = float(j)/pow(10000.,2.*k/embedding_size);
                        o_tensor(i,j,k<<1) = sin(v);
                        o_tensor(i,j,(k<<1)+1) = cos(v);
                    }
                    if(embedding_size&1) {
                        const auto v = float(j)/pow(10000.,2*h_es/embedding_size);
                        o_tensor(i,j,embedding_size) = sin(v);
                    }
                }
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("PositionEmbedding").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), PositionEmbeddingOp<CPUDevice, int32_t>);

REGISTER_OP("PlanePositionEmbedding")
    .Attr("T: {int32, int64}")
    .Input("size: T")
	.Output("output:float");

template <typename Device, typename T>
class PlanePositionEmbeddingOp: public OpKernel {
	public:
		explicit PlanePositionEmbeddingOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_size = context->input(0);
            const auto    size  = _size.template tensor<T,1>();

            OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size data must be 1-dimensional"));
            OP_REQUIRES(context, _size.dim_size(0) == 2, errors::InvalidArgument("size dim size must be 2."));

            const auto batch_size = 1;
            const auto width      = size(1);
            const auto height     = size(0);

            TensorShape output_shape0({batch_size,height,width});
            Tensor* output= NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output));

            auto       o_tensor = output->template tensor<float,3>();

            static const auto a = 0.1;
            static const auto b = 0.2;
            static const auto c = 0.3;

            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<height; ++j) {
                    for(auto k=0; k<width; ++k) {
                        const auto x = float(k)/width;
                        const auto y = float(j)/width;
                        const auto v = (1-x/a-y/b)*c;
                        o_tensor(i,j,k) = v;
                    }
                }
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("PlanePositionEmbedding").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), PlanePositionEmbeddingOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("PlanePositionEmbedding").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), PlanePositionEmbeddingOp<CPUDevice, tensorflow::int64>);
