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
#include "wtoolkit_cuda.h"
#include <future>
#include "mot_matching.h"

using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
/*
 * tag_k: [num_keypoints,N,tag_C] tag值
 * loc_k: [num_keypoints,N,2], x,y
 * val_k: [num_keypoints,N]
 *
 * output:
 * output: [N,num_keypoints,3+tag_C] (x,y,val,tag..)
 */
static uint32_t s_dict_key = 0;
inline uint32_t get_dict_key() {
    return s_dict_key++;
}
REGISTER_OP("MatchByTag")
    .Attr("T: {float,double,int32,int64}")
    .Attr("detection_threshold: float=0.1")
    .Attr("tag_threshold:float=1.0")
    .Attr("use_detection_val: bool=true")
    .Input("tag_k: T")
    .Input("loc_k: T")
    .Input("val_k: T")
	.Output("output:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(0);
            const auto num_keypoints = c->Value(c->Dim(input_shape0,0));
            const auto N = c->Dim(input_shape0,1);
            const auto tag_C = c->Value(c->Dim(input_shape0,2));
            auto shape0 = c->MakeShape({N,num_keypoints,3+tag_C});

			c->set_output(0, shape0);

			return Status::OK();
			});

template <typename Device,typename T>
class MatchByTagOp: public OpKernel {
    private:
        using Tensor1d_t = Eigen::Tensor<T,1,Eigen::RowMajor>;
        using Tensor2d_t = Eigen::Tensor<T,2,Eigen::RowMajor>;
        using Tensor3d_t = Eigen::Tensor<T,3,Eigen::RowMajor>;
    public:
        explicit MatchByTagOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("detection_threshold", &detection_threshold_));
            OP_REQUIRES_OK(context, context->GetAttr("use_detection_val", &use_detection_val_));
            OP_REQUIRES_OK(context, context->GetAttr("tag_threshold", &tag_threshold_));
        }

        void Compute(OpKernelContext* context) override
        {
            TIME_THISV1("MatchByTag");
            const Tensor &_tag_k = context->input(0);
            const Tensor &_loc_k = context->input(1);
            const Tensor &_val_k = context->input(2);

            OP_REQUIRES(context, _tag_k.dims() == 3, errors::InvalidArgument("tag_k data must be 3-dimension"));
            OP_REQUIRES(context, _loc_k.dims() == 3, errors::InvalidArgument("loc_k data must be 3-dimension"));
            OP_REQUIRES(context, _val_k.dims() == 2, errors::InvalidArgument("val_k data must be 2-dimension"));

            auto          tag_k = _tag_k.template tensor<T,3>();
            auto          loc_k = _loc_k.template tensor<T,3>();
            auto          val_k = _val_k.template tensor<T,2>();

            const auto   num_keypoints = _tag_k.dim_size(0);
            const auto   N             = _tag_k.dim_size(1);
            const auto   tag_C         = _tag_k.dim_size(2);
            const auto   out_C         = 3+tag_C;
            int          dims_3d0[3]   = {N,num_keypoints,out_C};
            TensorShape  outshape0;
            Tensor      *output        = NULL;

            TensorShapeUtils::MakeShape(dims_3d0, 3, &outshape0);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output));

            auto output_t = output->template tensor<T,3>();
            map<uint32_t,Eigen::Tensor<T,2,Eigen::RowMajor>> joints_dict;
            map<uint32_t,vector<Tensor1d_t>> tag_dict;
            Eigen::Tensor<T,2,Eigen::RowMajor> g_default_joints(num_keypoints,out_C);
            g_default_joints.setZero();

            output_t.setZero();

            for(auto i=0; i<num_keypoints; ++i) {
                auto valid_nr = 0;
                for(auto j=0; j<N; ++j) {
                    if(val_k(i,j)>detection_threshold_)
                        ++valid_nr;
                }
                if(0 == valid_nr)
                    continue;

                Eigen::Tensor<T,2,Eigen::RowMajor> joints(valid_nr,3+tag_C);
                vector<Tensor1d_t> tags;

                joints.setZero();
                for(auto j=0,k=0; j<N; ++j) {
                    if(val_k(i,j)<=detection_threshold_)
                        continue;
                    joints(k,0) = loc_k(i,j,0);
                    joints(k,1) = loc_k(i,j,1);
                    joints(k,2) = val_k(i,j);
                    for(auto l=0; l<tag_C; ++l)
                        joints(k,3+l) = tag_k(i,j,l);
                    ++k;
                    tags.push_back(Tensor1d_t(tag_k.chip(i,0).chip(j,0)));
                }
                if(joints_dict.empty()) {
                    for(auto j=0; j<valid_nr; ++j) {
                        auto key = get_dict_key();
                        Eigen::Tensor<T,2,Eigen::RowMajor> default_joints = g_default_joints;
                        vector<Eigen::Tensor<T,1,Eigen::RowMajor>> l_tags;

                        default_joints.chip(i,0) = joints.chip(j,0);
                        joints_dict[key] = default_joints;

                        l_tags.push_back(Eigen::Tensor<T,1,Eigen::RowMajor>(tags.at(j)));
                        tag_dict[key] = l_tags;
                    }
                } else {
                    vector<uint32_t> grouped_keys;
                    vector<Tensor1d_t> grouped_tags;
                    for(auto it=joints_dict.begin(); it!=joints_dict.end(); ++it) {
                        auto key = it->first;
                        auto tag = tags_mean(tag_dict[key],tag_C);

                        grouped_keys.push_back(key);
                        grouped_tags.emplace_back(std::move(tag));
                    }
                    auto diff_normed = get_normalized_dis_matrix(tags,grouped_tags);
                    Tensor2d_t diff_normed_saved = diff_normed;
                    if(use_detection_val_) 
                        diff_normed = apply_detection_value(diff_normed,joints);
                    const auto row = diff_normed.dimension(0);
                    const auto col = diff_normed.dimension(1);
                    const auto nr = max<int>(row,col);
                    Tensor2d_t big_diff_normed(nr,nr);
                    big_diff_normed.setConstant(1e10);
                    Eigen::array<int, 2> offsets = {0, 0};
                    Eigen::array<int, 2> extents = {row, col};

                    big_diff_normed.slice(offsets,extents) = diff_normed;

                    vector<pair<int,int>> matches;
                    int i_row,i_col;

                    MOT::linear_assignment(big_diff_normed.data(),nr,nullptr,&matches);

                    for(auto& d:matches) {

                        tie(i_row,i_col) = d;

                        if((i_row<row) && (i_col<col) && (diff_normed_saved(i_row,i_col)<tag_threshold_)) {
                            auto key = grouped_keys[i_col];

                            joints_dict[key].chip(i,0) = joints.chip(i_row,0);
                            tag_dict[key].push_back(tags.at(i_row));
                        } else if(i_row<row) {
                            auto key = get_dict_key();
                            Eigen::Tensor<T,2,Eigen::RowMajor> default_joints = g_default_joints;
                            vector<Tensor1d_t> l_tags;

                            default_joints.chip(i,0) = joints.chip(i_row,0);
                            joints_dict[key] = default_joints;

                            l_tags.push_back(Eigen::Tensor<T,1,Eigen::RowMajor>(tags.at(i_row)));
                            tag_dict[key] = l_tags;
                        }
                    }
                }
            }
            vector<pair<float,uint32_t>> scores_keys;
            for(auto it=joints_dict.begin(); it!=joints_dict.end(); ++it) {
                const auto key = it->first;
                Tensor1d_t scores = it->second.chip(2,1);
                scores_keys.emplace_back(Eigen::Tensor<float, 0,Eigen::RowMajor>(scores.mean()).data()[0],key);
            }
            sort(scores_keys.begin(),scores_keys.end(),[](const auto& lhv, const auto& rhv) {
                return lhv.first>rhv.first;
            });
            for(auto i=0; i<min<int>(scores_keys.size(),N); ++i ) {
                auto data = joints_dict[scores_keys[i].second];
                output_t.chip(i,0) = data;
            }
        }
        Eigen::Tensor<T,1,Eigen::RowMajor> tags_mean(const vector<Eigen::Tensor<T,1,Eigen::RowMajor>>& tensors,int tag_dim) {
            if(tensors.size()==1)
                return tensors.at(0);

            Eigen::Tensor<T,2,Eigen::RowMajor> all_data(tensors.size(),tag_dim);

            for(auto i=0; i<tensors.size(); ++i) {
                all_data.chip(i,0) = tensors.at(i);
            }
            return all_data.mean(Eigen::array<int,1>({0}));
        }
        Tensor2d_t get_normalized_dis_matrix(const vector<Tensor1d_t>& data0,const vector<Tensor1d_t>& data1) {
            Tensor2d_t res(data0.size(),data1.size());
            for(auto i=0; i<data0.size(); ++i) {
                for(auto j=0; j<data1.size(); ++j) {
                    res(i,j) = normalized_dis(data0.at(i),data1.at(j));
                }
            }
            return res;
        }
        T normalized_dis(const Tensor1d_t& data0,const Tensor1d_t& data1) {
            if(data0.dimension(0) == 1) {
                return fabs(data0(0)-data1(0));
            } else {
                T sum = 0;
                for(auto i=0; i<data0.dimension(0); ++i) {
                    const auto diff = data0(i)-data1(i);
                    sum += (diff*diff);
                }
                return sqrt(sum);
            }
        }
        Tensor2d_t apply_detection_value(const Tensor2d_t& diff,const Tensor2d_t& joints) {
            Tensor2d_t res(diff.dimension(0),diff.dimension(1));
            res.setZero();

            for(auto i=0; i<diff.dimension(0); ++i) {
                const auto val = joints(i,2);
                for(auto j=0; j<diff.dimension(1); ++j) {
                    //res(i,j) = int(diff(i,j)+0.5)*100-val;
                    res(i,j) = int(diff(i,j)*2+0.5)*50-val;
                }
            }
            return res;
        }
    private:
        float detection_threshold_ = 0.1f;
        float tag_threshold_       = 1.0f;
        bool  use_detection_val_   = true;
};
REGISTER_KERNEL_BUILDER(Name("MatchByTag").Device(DEVICE_CPU).TypeConstraint<float>("T"), MatchByTagOp<CPUDevice, float>);
