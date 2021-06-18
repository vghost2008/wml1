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
#include <bitset>
#include "bboxes.h"
#include <opencv2/opencv.hpp>
#include <boost/algorithm/clamp.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * bboxes: [N,4], absolute coordinate
 * labels: [N]
 * mask: [N,X,X]
 * probability: [N]
 * min_area_bboxes: [N,3,2]
 */
REGISTER_OP("QcPostProcess")
    .Attr("T: {float, double}")
	.Attr("angle_threshold:float=3.0")
	.Attr("knife_label:int=1")
	.Attr("merge:bool=False")
    .Input("bboxes: T")
    .Input("labels: int32")
    .Input("probability: T")
    .Input("mask: uint8")
    .Input("min_area_bboxes: T")
	.Output("r_bboxes:T")
	.Output("r_labels:int32")
	.Output("r_probability:T")
	.Output("r_mask:uint8")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            c->set_output(0,c->Matrix(-1,4));
			c->set_output(1, c->Vector(-1));
			c->set_output(2, c->Vector(-1));
			c->set_output(3, c->input(3));
			return Status::OK();
			});

template <typename Device, typename T>
class QcPostProcessOp: public OpKernel {
    private:
        using bbox_t = Eigen::Tensor<T,1,Eigen::RowMajor>;
        using mask_t = Eigen::Tensor<uint8_t,2,Eigen::RowMajor>;
    private:
        static constexpr auto kMinKnifeRatio = 5.0f;
        static constexpr auto kDecayKnifeRatio = 1.0f;
        enum KMI_Status
        {
            KMIS_NEED_UPDATE_PROBABILITY,
            KMIS_NEED_MERGE,
        };
        struct KnifeMaskInfo {
            float center_x;
            float center_y;
            float angle;
            float width;
            float height;
            float probability;
            float tmp;
            int index=0;
            bitset<32> statu;
            int merge_id = 0;
        };
    public:
        explicit QcPostProcessOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("angle_threshold", &angle_threshold_));
            OP_REQUIRES_OK(context, context->GetAttr("knife_label", &knife_label_));
            OP_REQUIRES_OK(context, context->GetAttr("merge", &merge_));
        }
        void Compute(OpKernelContext* context) override
        {
            const Tensor &_bboxes          = context->input(0);
            const Tensor &_labels          = context->input(1);
            const Tensor &_probability     = context->input(2);
            const Tensor &_mask            = context->input(3);
            const Tensor &_min_area_bboxes = context->input(4);

            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes data must be 2-dimensional"));
            OP_REQUIRES(context, _labels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));
            OP_REQUIRES(context, _probability.dims() == 1, errors::InvalidArgument("probability must be 1-dimensional"));
            OP_REQUIRES(context, _min_area_bboxes.dims() == 3, errors::InvalidArgument("min_area_bboxes must be 3-dimensional"));
            OP_REQUIRES(context, _mask.dims() == 3, errors::InvalidArgument("mask be 3-dimensional"));

            auto       bboxes          = _bboxes.template tensor<T,2>();
            auto       labels          = _labels.template tensor<int,1>();
            auto       probability     = _probability.template tensor<T,1>();
            auto       min_area_bboxes = _min_area_bboxes.template tensor<T,3>();
            auto       mask            = _mask.template tensor<uint8_t,3>();
            const auto data_nr         = _labels.dim_size(0);

            Tensor      *output_bboxes = nullptr;
            Tensor      *output_labels = nullptr;
            Tensor      *output_probability = nullptr;
            Tensor      *output_mask   = nullptr;


            OP_REQUIRES_OK(context,context->allocate_output(0,_bboxes.shape(),&output_bboxes));
            OP_REQUIRES_OK(context,context->allocate_output(1,_labels.shape(),&output_labels));
            OP_REQUIRES_OK(context,context->allocate_output(2,_probability.shape(),&output_probability));
            OP_REQUIRES_OK(context,context->allocate_output(3,_mask.shape(),&output_mask));

            if(!output_bboxes->CopyFrom(_bboxes,_bboxes.shape()))
                return;
            if(!output_labels->CopyFrom(_labels,_labels.shape()))
                return;
            if(!output_probability->CopyFrom(_probability,_probability.shape()))
                return;
            if(!output_mask->CopyFrom(_mask,_mask.shape()))
                return;

            vector<KnifeMaskInfo> knife_mask_info;

            for(auto i=0; i<data_nr; ++i) {
                if(labels(i) == knife_label_) {
                    KnifeMaskInfo info;
                    info.center_x = min_area_bboxes(i,0,0);
                    info.center_y = min_area_bboxes(i,0,1);
                    const auto w = min_area_bboxes(i,1,0);
                    const auto h = min_area_bboxes(i,1,1);
                    if(w>h) {
                        info.width = w;
                        info.height = h;
                        info.angle = min_area_bboxes(i,2,0);
                    } else {
                        info.width = h;
                        info.height = w;
                        info.angle = min_area_bboxes(i,2,0)+90;
                    }
                    info.probability = probability(i);
                    info.index = i;
                    knife_mask_info.push_back(info);
                }
            }
            sort(knife_mask_info.begin(),knife_mask_info.end(),[](const KnifeMaskInfo& lhv,const KnifeMaskInfo& rhv) {
                return lhv.center_x<rhv.center_x;
            });
            process_knife_mask(knife_mask_info);

            auto o_probability = output_probability->template tensor<T,1>();
            auto o_bboxes = output_bboxes->template tensor<T,2>();
            auto o_mask = output_mask->template tensor<uint8_t,3>();

            for(auto & info:knife_mask_info) {
                if(info.statu.test(KMIS_NEED_UPDATE_PROBABILITY)) {
                    o_probability(info.index) = info.probability;
                }
            }
            if(merge_ && (knife_mask_info.size()>1)) {
                vector<int> indexs;
                for(auto i=0; i<knife_mask_info.size()-1; ++i) {
                    auto& info0 = knife_mask_info[i];
                    
                    if(!info0.statu.test(KMIS_NEED_MERGE))
                        continue;

                    indexs.clear();
                    indexs.push_back(i);
                    for(auto j=i+1; j<knife_mask_info.size(); ++j) {
                        auto& info1 = knife_mask_info[j];

                        if(!info1.statu.test(KMIS_NEED_MERGE))
                            continue;

                        if((info0.merge_id == info1.merge_id) 
                                && (info0.merge_id>0)) {
                            indexs.push_back(j);
                        }
                    }

                    if(indexs.size()<2)
                        continue;

                    float x_min = 1e8;
                    float y_min = 1e8;
                    float x_max = -1;
                    float y_max = -1;
                    float prob = 0.0f;
                    vector<bbox_t> merge_bboxes_data;
                    vector<mask_t> merge_mask_data;

                    for(auto j:indexs) {
                        auto r_index = knife_mask_info[j].index;
                        merge_bboxes_data.push_back(o_bboxes.chip(r_index,0));
                        merge_mask_data.push_back(o_mask.chip(r_index,0));

                        prob = max<float>(prob,o_probability(r_index));
                    }
                    auto merged_data = merge_bboxes(merge_bboxes_data,merge_mask_data);

                    auto r_index = knife_mask_info[indexs[0]].index;

                    o_bboxes.chip(r_index,0) = get<0>(merged_data);
                    o_mask.chip(r_index,0) = get<1>(merged_data);
                    o_probability(r_index) = prob;

                    for(auto j=1; j<indexs.size(); ++j) {
                        auto r_index = knife_mask_info[indexs[j]].index;
                        o_probability(r_index) = 0.0f;
                    }
                }
            }
        }
        void process_knife_mask(vector<KnifeMaskInfo>& infos) {
            vector<int> indexs;

            if(infos.empty()) 
                return;


            for(auto i=0; i<infos.size()-1; ++i) {
                auto info0 = infos[i];

                if(info0.statu.test(KMIS_NEED_MERGE))
                    continue;

                indexs.clear();
                indexs.push_back(i);
                for(auto j=i+1; j<infos.size(); ++j) {
                    KnifeMaskInfo ref_info = infos[indexs.back()];
                    const auto info1 = infos[j];
                    const auto angle = ref_info.angle;
                    if(fabs(info1.angle-angle)>angle_threshold_)
                        continue;
                     auto delta_x = info1.center_x-ref_info.center_x;
                     auto delta_y = info1.center_y-ref_info.center_y;
                     auto pos_angle = atan2(delta_y,delta_x)*180/M_PI;
                     if(pos_angle<=-90) {
                         pos_angle = 180.0+pos_angle;
                     } else if(pos_angle>90) {
                         pos_angle = pos_angle-180.0;
                     }
                    if(fabs(pos_angle-angle)>angle_threshold_)
                        continue;
                    indexs.push_back(j);
                }
                if(indexs.size()<2) {
                    /*
                     * 孤立的刀痕需要在宽高比上有所要求
                     */
                    if(info0.width<info0.height*kMinKnifeRatio) {
                        info0.statu.set(KMIS_NEED_UPDATE_PROBABILITY);
                        info0.probability = info0.probability/2;
                    } else if(info0.width<info0.height*kDecayKnifeRatio) {
                        info0.statu.set(KMIS_NEED_UPDATE_PROBABILITY);
                        info0.probability = info0.probability-0.1;
                    }
                    continue;
                }
                for(auto j:indexs) {
                    auto& info = infos[j];
                    info.merge_id = i+1;
                }
                enhance_knife_mask(indexs,infos);
            }
        }
        void enhance_knife_mask(const vector<int>& indexs,vector<KnifeMaskInfo>& infos) {
            for(auto i=0; i<indexs.size()-1; ++i) {
                auto& info0 = infos[indexs[i]];
                auto& info1 = infos[indexs[i+1]];
                const auto dx = info0.center_x-info1.center_x;
                const auto dy = info0.center_y-info1.center_y;
                const auto distance = sqrt(dx*dx+dy*dy);
                const auto fill_value = (info0.width+info1.width)/2;

                if(fill_value<distance/4) {
                    continue;
                }


                const auto min_p = min(info0.probability,info1.probability);
                const auto max_p = max(info0.probability,info1.probability);
                const auto alpha = (fill_value/distance);
                const auto new_p = min_p+(max_p-min_p)*alpha;

                if(info0.probability<new_p) {
                    info0.tmp = new_p;
                    info0.statu.set(KMIS_NEED_UPDATE_PROBABILITY);
                }

                info0.statu.set(KMIS_NEED_MERGE);
                info1.statu.set(KMIS_NEED_MERGE);

                if(info1.probability<new_p) {
                    info1.tmp = new_p;
                    info1.statu.set(KMIS_NEED_UPDATE_PROBABILITY);
                }
            }
            for(auto i=0; i<indexs.size(); ++i) {
                auto& info0 = infos[indexs[i]];
                if(info0.statu.test(KMIS_NEED_UPDATE_PROBABILITY))
                    info0.probability = info0.tmp;
            }
        }
        tuple<bbox_t,mask_t> merge_bboxes(const vector<bbox_t>& boxes,const vector<mask_t>& masks)
        {
            const auto mask_size = masks[0].dimension(0);
            bbox_t  env_bbox(4);
            cv::Mat res_mask = cv::Mat::zeros(mask_size,mask_size,CV_8UC1);
            cv::Mat res_mask1 = cv::Mat::zeros(mask_size,mask_size,CV_8UC1);

            bboxes_envelope(boxes,env_bbox);

            const auto env_bbox_w = env_bbox(3)-env_bbox(1);
            const auto env_bbox_h = env_bbox(2)-env_bbox(0);

            for(auto i=0; i<boxes.size(); ++i) {
                vector<vector<cv::Point>> contours;
                vector<vector<cv::Point>> new_contours;
                vector<cv::Vec4i> hierarchy;
                vector<cv::Point> points;
                cv::Mat dst_img = to_mat(masks[i]);
                const auto cur_bbox = boxes[i];
                const auto cur_bbox_w = cur_bbox(3)-cur_bbox(1);
                const auto cur_bbox_h = cur_bbox(2)-cur_bbox(0);

                cv::findContours(dst_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));

                if(contours.size()==0) {
                    continue;
                }

                for(auto& points:contours) {
                    vector<cv::Point> new_points;
                    for(auto& p:points) {
                        auto x = (p.x*cur_bbox_w/mask_size+cur_bbox(1)-env_bbox(1))*mask_size/env_bbox_w;
                        auto y = (p.y*cur_bbox_h/mask_size+cur_bbox(0)-env_bbox(0))*mask_size/env_bbox_h;
                        new_points.push_back(cv::Point(x,y));
                    }
                    new_contours.push_back(new_points);
                }
                cv::drawContours(res_mask,new_contours,-1,cv::Scalar(1),CV_FILLED,8,hierarchy);

            }

            vector<vector<cv::Point>> contours;
            vector<vector<cv::Point>> new_contours;
            vector<cv::Vec4i> hierarchy;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));

            cv::morphologyEx(res_mask, res_mask1, CV_MOP_CLOSE, kernel);

            mask_t r_mask(mask_size,mask_size);
            memcpy(r_mask.data(),res_mask1.data,mask_size*mask_size);
            return make_tuple(env_bbox,r_mask);
        }
        static cv::Mat to_mat(const mask_t& msk) 
        {
            cv::Mat res(msk.dimension(0),msk.dimension(1),CV_8UC1,(void*)msk.data());
            return res.clone();
        }
    private:
        float angle_threshold_ = 3.0f;
        int   knife_label_     = 1;
        bool  merge_           = false;
};
REGISTER_KERNEL_BUILDER(Name("QcPostProcess").Device(DEVICE_CPU).TypeConstraint<float>("T"), QcPostProcessOp<CPUDevice, float>);
