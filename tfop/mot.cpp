#include <stdio.h>
#include <cfloat>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"
//#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "wtoolkit.h"
#include "jde_tracker.h"

using namespace tensorflow;
using namespace std;
using namespace MOT;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
REGISTER_OP("FairMot")
    .Attr("det_thredh:float=0.1")
    .Attr("frame_rate:int=30")
    .Attr("track_buffer:int=30")
    .Input("bboxes: float32") //(ymin,xmin,ymax,xmax) absolute coordinate
    .Input("probs: float32")
    .Input("embedding: float32")
    .Input("is_first_frame: bool")
	.Output("output_track_id:int32")
	.Output("output_bboxes:float32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            auto shape0 = c->MakeShape({-1});
            auto shape1 = c->MakeShape({-1,4});

			c->set_output(0, shape0);
			c->set_output(1, shape1);

			return Status::OK();
			});
template <typename Device>
class FairMOTOp: public OpKernel {
    public:
        explicit FairMOTOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("det_thredh", &det_thredh_));
			OP_REQUIRES_OK(context, context->GetAttr("track_buffer", &track_buffer_));
			OP_REQUIRES_OK(context, context->GetAttr("frame_rate", &frame_rate_));
        }

        void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("FairMOT");
            const Tensor &_bboxes        = context->input(0);
            const Tensor &_probs         = context->input(1);
            const Tensor &_embedding     = context->input(2);
            const bool    is_first_frame = context->input(3).template flat<bool>().data()[0];

            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimension"));
            OP_REQUIRES(context, _probs.dims() == 1, errors::InvalidArgument("probs data must be 1-dimension"));
            OP_REQUIRES(context, _embedding.dims() == 2, errors::InvalidArgument("embedding data must be 2-dimension"));

            if(is_first_frame || (jde_tracker_ == nullptr)) {
                if(!is_first_frame) {
                    cout<<"WARNING: First frame should set is_first_frame to true."<<endl;
                }
                jde_tracker_ = make_shared<MOT::JDETracker>(det_thredh_,frame_rate_,track_buffer_);
            }

            auto bboxes    = _bboxes.template tensor<float,2>();
            auto probs     = _probs.template tensor<float,1>();
            auto embedding = _embedding.template tensor<float,2>();


            auto jt_bboxes    = Eigen::Map<const BBoxes_t>(bboxes.data(),bboxes.dimension(0),4);
            auto jt_probs     = Eigen::Map<const Probs_t>(probs.data(),probs.dimension(0),1);
            auto jt_embedding = Eigen::Map<const Embeddings_t>(embedding.data(),embedding.dimension(0),embedding.dimension(1));
            auto tracks       = jde_tracker_->update(jt_bboxes,jt_probs,jt_embedding);
            auto data_nr      = tracks.size();
            int  dims_1d0[1]  = {data_nr};
            int  dims_2d0[2]  = {data_nr,4};

            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_bboxes = NULL;
            Tensor      *output_ids    = NULL;

            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            TensorShapeUtils::MakeShape(dims_2d0, 2, &outshape1);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_ids));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_bboxes));

            auto ids = output_ids->template tensor<int,1>();
            auto o_bboxes = output_bboxes->template tensor<float,2>();

            ids.setZero();
            o_bboxes.setZero();

            for(auto i=0; i<data_nr; ++i) {
                auto& track = tracks[i];
                const auto& bbox = track->get_latest_yminxminymaxxmax_bbox();
                ids(i) = track->track_id();
                for(auto j=0; j<4; ++j)
                    o_bboxes(i,j) = bbox(j);
            }
        }
    private:
        shared_ptr<MOT::JDETracker> jde_tracker_;
        float det_thredh_   = 0.1;
        int   frame_rate_   = 30;
        int   track_buffer_ = 30;
};
REGISTER_KERNEL_BUILDER(Name("FairMot").Device(DEVICE_CPU), FairMOTOp<CPUDevice>);

REGISTER_OP("SortMot")
    .Attr("det_thredh:float=0.1")
    .Attr("frame_rate:int=30")
    .Attr("track_buffer:int=30")
    .Input("bboxes: float32") //(ymin,xmin,ymax,xmax) absolute coordinate
    .Input("probs: float32")
    .Input("is_first_frame: bool")
	.Output("output_track_id:int32")
	.Output("output_bboxes:float32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            auto shape0 = c->MakeShape({-1});
            auto shape1 = c->MakeShape({-1,4});

			c->set_output(0, shape0);
			c->set_output(1, shape1);

			return Status::OK();
			});

template <typename Device>
class SORTMOTOp: public OpKernel {
    public:
        explicit SORTMOTOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("det_thredh", &det_thredh_));
			OP_REQUIRES_OK(context, context->GetAttr("track_buffer", &track_buffer_));
			OP_REQUIRES_OK(context, context->GetAttr("frame_rate", &frame_rate_));
        }

        void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("SORTMOT");
            const Tensor &_bboxes        = context->input(0);
            const Tensor &_probs         = context->input(1);
            const bool    is_first_frame = context->input(2).template flat<bool>().data()[0];

            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("box data must be 2-dimension"));
            OP_REQUIRES(context, _probs.dims() == 1, errors::InvalidArgument("probs data must be 1-dimension"));

            if(is_first_frame || (jde_tracker_ == nullptr)) {
                if(!is_first_frame) {
                    cout<<"WARNING: First frame should set is_first_frame to true."<<endl;
                }
                jde_tracker_ = make_shared<MOT::JDETracker>(det_thredh_,frame_rate_,track_buffer_);
            }

            auto bboxes    = _bboxes.template tensor<float,2>();
            auto probs     = _probs.template tensor<float,1>();


            auto jt_bboxes    = Eigen::Map<const BBoxes_t>(bboxes.data(),bboxes.dimension(0),4);
            auto jt_probs     = Eigen::Map<const Probs_t>(probs.data(),probs.dimension(0),1);
            auto tracks       = jde_tracker_->update(jt_bboxes,jt_probs);
            auto data_nr      = tracks.size();
            int  dims_1d0[1]  = {data_nr};
            int  dims_2d0[2]  = {data_nr,4};

            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_bboxes = NULL;
            Tensor      *output_ids    = NULL;

            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            TensorShapeUtils::MakeShape(dims_2d0, 2, &outshape1);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_ids));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_bboxes));

            auto ids = output_ids->template tensor<int,1>();
            auto o_bboxes = output_bboxes->template tensor<float,2>();

            ids.setZero();
            o_bboxes.setZero();

            for(auto i=0; i<data_nr; ++i) {
                auto& track = tracks[i];
                const auto& bbox = track->get_latest_yminxminymaxxmax_bbox();
                ids(i) = track->track_id();
                for(auto j=0; j<4; ++j)
                    o_bboxes(i,j) = bbox(j);
            }
        }
    private:
        shared_ptr<MOT::JDETracker> jde_tracker_;
        float det_thredh_   = 0.1;
        int   frame_rate_   = 30;
        int   track_buffer_ = 30;
};
REGISTER_KERNEL_BUILDER(Name("SortMot").Device(DEVICE_CPU), SORTMOTOp<CPUDevice>);
