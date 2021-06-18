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
#include "wtoolkit.h"
#include "open_pose_decode_imp.h"
#include <future>

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
/*
 * gaussian_delta: 一般为2
 * keypoints: [B,N,num_points_nr,2] 相对坐标,x,y
 * glength: 有效的groundtruth instance数量
 * output_size: 输出图的大小[2]=(OH,OW) 
 *
 * output:
 * output_conf_map: top left heatmaps [B,OH,OW,num_points_nr]
 * output_paf_map: bottom right heatmaps [B,OH,OW,num_points_nr*2]
 */
REGISTER_OP("OpenPoseEncode")
    .Attr("T: {float,double,int32,int64}")
	.Attr("l_delta:float=8.0")
	.Attr("gaussian_delta:float=8.0")
	.Attr("keypoints_pair:list(int)")
    .Input("keypoints: T")
    .Input("output_size: int32")
    .Input("glength: int32")
	.Output("output_conf_map:T")
	.Output("output_paf_map:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto points_nr = c->Value(c->Dim(input_shape0,2));
            const auto batch_size = c->Dim(input_shape0,0);
            vector<int> keypoints_pair;
			c->GetAttr("keypoints_pair", &keypoints_pair);
            auto shape0 = c->MakeShape({batch_size,-1,-1,points_nr});
            auto shape1 = c->MakeShape({batch_size,-1,-1,keypoints_pair.size()});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			return Status::OK();
			});

template <typename Device,typename T>
class OpenPoseEncodeOp: public OpKernel {
	public:
		explicit OpenPoseEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("l_delta", &l_delta_));
			OP_REQUIRES_OK(context, context->GetAttr("gaussian_delta", &gaussian_delta_));
			OP_REQUIRES_OK(context, context->GetAttr("keypoints_pair", &keypoints_pair_));
		}

		void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("OpenPoseEncode");
            const Tensor &_keypoints = context->input(0);
            const Tensor &_gsize      = context->input(2);
            auto          output_size = context->input(1).template flat<int>().data();
            const auto    batch_size  = _keypoints.dim_size(0);
            const auto num_keypoints = _keypoints.dim_size(2);

            OP_REQUIRES(context, _keypoints.dims() == 4, errors::InvalidArgument("keypoints data must be 4-dimension"));
            OP_REQUIRES(context, _gsize.dims() == 1, errors::InvalidArgument("gsize data must be 1-dimension"));

            auto         keypoints       = _keypoints.template tensor<T,4>();
            auto         gsize           = _gsize.template tensor<int,1>();
            int          dims_4d0[4]     = {int(batch_size),output_size[0],output_size[1],num_keypoints};
            int          dims_4d1[4]     = {int(batch_size),output_size[0],output_size[1],keypoints_pair_.size()};
            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_conf_map = NULL;
            Tensor      *output_paf_map  = NULL;
            const auto   max_data_nr     = _keypoints.dim_size(1);

            TensorShapeUtils::MakeShape(dims_4d0, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_4d1, 4, &outshape1);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_conf_map));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_paf_map));

            auto heatmaps_conf = output_conf_map->template tensor<T,4>();
            auto heatmaps_paf = output_paf_map->template tensor<T,4>();

            heatmaps_conf.setZero();
            heatmaps_paf.setZero();

            for(auto i=0; i<batch_size; ++i) {
                for(auto j=0; j<gsize(i); ++j) {
                    for(auto k=0; k<num_keypoints; ++k) {
                        const auto x0 = keypoints(i,j,k,0)*(output_size[1]-1);
                        const auto y0 = keypoints(i,j,k,1)*(output_size[0]-1);

						if((x0>=0) && (y0>=0))
                        	draw_gaussian(heatmaps_conf,i,x0,y0,k,gaussian_delta_);
                    }
                }
            }
            if(keypoints_pair_.size()%2 != 0) {
                cout<<"ERROR keypoints pair size, "<<keypoints_pair_.size()<<endl;
            }
            const auto kpoints_pair_nr = keypoints_pair_.size()/2;
            Eigen::Tensor<float,4,Eigen::RowMajor> data_ct(batch_size,output_size[0],output_size[1],num_keypoints);
            for(auto i=0; i<batch_size; ++i) {
                const auto nr = min<int>(max_data_nr,gsize(i));
                for(auto j=0; j<nr; ++j) {
                    for(auto k=0; k<kpoints_pair_nr; ++k) {
                        const auto index0 = keypoints_pair_[k*2];
                        const auto index1 = keypoints_pair_[k*2+1];
                        const auto x0 = keypoints(i,j,index0,0)*(output_size[1]-1);
                        const auto y0 = keypoints(i,j,index0,1)*(output_size[0]-1);
                        const auto x1 = keypoints(i,j,index1,0)*(output_size[1]-1);
                        const auto y1 = keypoints(i,j,index1,1)*(output_size[0]-1);
						if((index0>num_keypoints) || (index1>num_keypoints)) {
							cout<<"ERROR: OpenPoseEncode: error keypoints pair index "<<index0<<" and "<<index1<<", Keypoints num is "<<num_keypoints<<endl;
							continue;
						}

						if((x0>=0) && (y0>=0) && (x1>=0) && (y1>=0))
                        	draw_paf(heatmaps_paf,data_ct,i,x0,y0,x1,y1,k);
                    }
                }
            }
        }

        template<typename DT>
        static void draw_gaussian(DT& data,int batch_index,float cx,float cy,int k,float radius)
        {
            const auto th           = 4.6052;
            const auto spread_range = radius *sqrt(2*th);
            const auto width        = data.dimension(2);
            const auto height       = data.dimension(1);
            const auto xtl          = max(0,int(cx-spread_range));
            const auto ytl          = max(0,int(cy-spread_range));
            const auto xbr          = min<int>(width,int(cx+spread_range+1));
            const auto ybr          = min<int>(height,int(cy+spread_range+1));
            const auto sigma_p      = 2 *radius*radius;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-cx;
                    auto dy = y-cy;
                    const auto d = (dx*dx+dy*dy);
                    const auto expv = d/sigma_p;
                    if (expv>th)
                        continue;
                    auto v = exp(-expv);
                    data(batch_index,y,x,k) = max(data(batch_index,y,x,k),v);
                }
            }
        }
        template<typename DT,typename CT>
        void draw_paf(DT& data,CT& ct,int batch_index,float x0,float y0,float x1,float y1,int k)
        {
            const auto dis     = distance(x0,y0,x1,y1);
            const auto l_delta = max<float>(dis *0.125,l_delta_);
            const auto width   = data.dimension(2);
            const auto height  = data.dimension(1);
            const auto xtl     = max<int>(min(x0,x1)-l_delta,0);
            const auto ytl     = max<int>(min(y0,y1)-l_delta,0);
            const auto xbr     = min<int>(max<int>(x0,x1)+l_delta+1,width);
            const auto ybr     = min<int>(max<int>(y0,y1)+l_delta+1,height);
            const auto vx      = (x1-x0)/dis;
            const auto vy      = (y1-y0)/dis;
            const auto A       = y1-y0;
            const auto B       = x0-x1;
            const auto C       = y0 *x1-x0 *y1;
            const auto D       = sqrt(A *A+B *B+1e-8);
            bool       set_any = false;

            for(auto x=xtl; x<xbr; ++x) {
                for(auto y=ytl; y<ybr; ++y) {
                    auto dx = x-x0;
                    auto dy = y-y0;
                    auto j0 = dx*vx+dy*vy;

                    if((j0<0) || (j0>dis))
                        continue;

                    if(fabs((x*A+y*B+C)/D)>l_delta)
                        continue;

                    set_any = true;

                    ct(batch_index,y,x,k) = ct(batch_index,y,x,k)+1;

                    const auto nr = ct(batch_index,y,x,k);

                    if(nr>1) {
                        auto old_vx = data(batch_index,y,x,k*2);
                        auto old_vy = data(batch_index,y,x,k*2+1);
                        auto new_vx = (old_vx*(nr-1)+vx)/nr;
                        auto new_vy = (old_vy*(nr-1)+vy)/nr;

                        data(batch_index,y,x,k*2) = new_vx;
                        data(batch_index,y,x,k*2+1) = new_vy;
                    } else {
                        data(batch_index,y,x,k*2) = vx;
                        data(batch_index,y,x,k*2+1) = vy;
                    }
                }
            }
            if(!set_any) {
                auto x = int((x0+x1)/2+0.5);
                auto y = int((y0+y1)/2+0.5);

                ct(batch_index,y,x,k) = ct(batch_index,y,x,k)+1;

                const auto nr = ct(batch_index,y,x,k);

                if(nr>1) {
                    auto old_vx = data(batch_index,y,x,k*2);
                    auto old_vy = data(batch_index,y,x,k*2+1);
                    auto new_vx = (old_vx*(nr-1)+vx)/nr;
                    auto new_vy = (old_vy*(nr-1)+vy)/nr;

                    data(batch_index,y,x,k*2) = new_vx;
                    data(batch_index,y,x,k*2+1) = new_vy;
                } else {
                    data(batch_index,y,x,k*2) = vx;
                    data(batch_index,y,x,k*2+1) = vy;
                }
            }
        }
        inline float distance(float x0,float y0,float x1,float y1) {
            const auto dx = x1-x0;
            const auto dy = y1-y0;

            return sqrt(dx*dx+dy*dy+1e-8);
        }
	private:
       float       l_delta_        = 2;
       float       gaussian_delta_ = 2;
       vector<int> keypoints_pair_;
};
REGISTER_KERNEL_BUILDER(Name("OpenPoseEncode").Device(DEVICE_CPU).TypeConstraint<float>("T"), OpenPoseEncodeOp<CPUDevice, float>);

/*
 * keypoints_th: threshold for detection keypoints on conf_maps
 * interp_samples: interp samples in paf map line test
 * paf_score_th: threshold for detection points on paf map
 * conf_th: percent of valid point on paf map line
 * max detection: max persion in one image
 * keypoints_pair: [limb0_begin_id,limb0_end_id,limb1_begin_id,....]
 * conf_maps: typical: [B,H,W,points_nr]
 * paf_maps: typical: [B,H,W,len(keypoints_pair)]
 * return:
 * output_keypoints:[B,max_detection,points_nr,2]
 * output_lengths:[B]
 */
REGISTER_OP("OpenPoseDecode")
    .Attr("T: {float,double,int32,int64}")
    .Attr("keypoints_th:float=0.1")
    .Attr("interp_samples:int=10")
    .Attr("paf_score_th:float=0.1")
    .Attr("conf_th:float=0.7")
    .Attr("max_detection:int=100")
	.Attr("keypoints_pair:list(int)")
    .Input("conf_maps: T")
    .Input("paf_maps: T")
	.Output("output_keypoints:T")
	.Output("output_lengths:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            const auto points_nr = c->Dim(input_shape0,3);
            int max_detection;

            c->GetAttr("max_detection",&max_detection);

            auto shape0 = c->MakeShape({batch_size,max_detection,points_nr,2});
            auto shape1 = c->MakeShape({batch_size});

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			return Status::OK();
			});

template <typename Device,typename T>
class OpenPoseDecodeOp: public OpKernel {
    public:
        explicit OpenPoseDecodeOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("keypoints_th", &keypoints_th_));
            OP_REQUIRES_OK(context, context->GetAttr("interp_samples", &interp_samples_));
            OP_REQUIRES_OK(context, context->GetAttr("paf_score_th", &paf_score_th_));
            OP_REQUIRES_OK(context, context->GetAttr("conf_th", &conf_th_));
            OP_REQUIRES_OK(context, context->GetAttr("max_detection", &max_detection_));
            vector<int> _keypoints_pair;
            OP_REQUIRES_OK(context, context->GetAttr("keypoints_pair", &_keypoints_pair));

            if(_keypoints_pair.size()%2 != 0) {
                cout<<"ERROR keypoints pair size: "<<_keypoints_pair.size()<<endl;
            }

            for(auto i=0; i<_keypoints_pair.size()/2; ++i) {
                keypoints_pair_.emplace_back(_keypoints_pair[i*2],_keypoints_pair[i*2+1]);
                map_idx_.emplace_back(i*2,i*2+1);
            }
        }
        void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("OpenPoseDecode");
            const Tensor &_conf_map = context->input(0);
            const Tensor &_paf_map  = context->input(1);

            OP_REQUIRES(context, _conf_map.dims() == 4, errors::InvalidArgument("conf map data must be 4-dimension"));
            OP_REQUIRES(context, _paf_map.dims() == 4, errors::InvalidArgument("paf map data must be 4-dimension"));

            auto       conf_map   = _conf_map.template tensor<T,4>();
            auto       paf_map    = _paf_map.template tensor<T,4>();
            const auto batch_size = _conf_map.dim_size(0);
            const auto points_nr  = _conf_map.dim_size(3);
            const auto H          = conf_map.dimension(1);
            const auto W          = conf_map.dimension(2);

            int           dims_4d[4]            = {int(batch_size),max_detection_,points_nr,2};
            int           dims_1d[1]           = {int(batch_size)};
            TensorShape  outshape0;
            TensorShape  outshape1;
            Tensor      *output_keypoints = NULL;
            Tensor      *output_lens = NULL;

            TensorShapeUtils::MakeShape(dims_4d, 4, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape1);


            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_keypoints));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_lens));

            auto o_keypoints = output_keypoints->template tensor<T,4>();
            auto o_lens = output_lens->template tensor<int,1>();

            o_keypoints.setZero();
            o_lens.setZero();
            int size_conf_map[] = {points_nr,H,W};
            int size_paf_map[] = {paf_map.dimension(3),H,W};
            Eigen::array<int, 3> shuffling({2, 0,1});

            for(auto i=0; i<batch_size; ++i) {
                Eigen::Tensor<float,3,Eigen::RowMajor> l_conf_map = conf_map.chip(i,0);
                Eigen::Tensor<float,3,Eigen::RowMajor> l_paf_map = paf_map.chip(i,0);
                Eigen::Tensor<float,3,Eigen::RowMajor> l_conf_map_t = l_conf_map.shuffle(shuffling);
                Eigen::Tensor<float,3,Eigen::RowMajor> l_paf_map_t = l_paf_map.shuffle(shuffling);
                cv::Mat paf_map_mat = cv::Mat(3,size_paf_map,CV_32FC1,l_paf_map_t.data());
                cv::Mat conf_map_mat = cv::Mat(3,size_conf_map,CV_32FC1,l_conf_map_t.data());
                auto persion_wise_keypoints = openpose_decode_imp(conf_map_mat,paf_map_mat,
                        map_idx_,keypoints_pair_,
                        keypoints_th_,
                        interp_samples_,
                        paf_score_th_,
                        conf_th_);
                auto data_nr = min<int>(persion_wise_keypoints.size(),max_detection_);
                for(auto j=0; j<data_nr; ++j) {
                    auto& keypoints = persion_wise_keypoints[j];
                    for(auto k=0; k<keypoints.size(); ++k) {
                        auto x = keypoints[k].first;
                        auto y = keypoints[k].second;
                        if(x>0)
                            x = x/(W-1);
                        if(y>0)
                            y = y/(H-1);
                        o_keypoints(i,j,k,0) = x;
                        o_keypoints(i,j,k,1) = y;
                    }
                }
                o_lens(i) = data_nr;
            }
        }
    private:
        float keypoints_th_   = 0.1;
        int interp_samples_ = 10;
        float paf_score_th_   = 0.1;
        float conf_th_        = 0.7;
        int   max_detection_  = 100;
        vector<pair<int,int>> keypoints_pair_;
        vector<pair<int,int>> map_idx_;
};
REGISTER_KERNEL_BUILDER(Name("OpenPoseDecode").Device(DEVICE_CPU).TypeConstraint<float>("T"), OpenPoseDecodeOp<CPUDevice, float>);
