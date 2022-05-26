#include <torch/script.h>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "match_by_tag.h"
#include "hrnet_pe.h"
#include <iostream>

torch::Tensor match_by_tag(torch::Tensor tag_k,torch::Tensor loc_k,torch::Tensor val_k,double detection_threshold=0.1,double tag_threshold=1.0,bool use_detection_val=true)
{
    using Tensor3d_t    = Eigen::Tensor<float,3,Eigen::RowMajor>;
    using TensorMap3d_t = Eigen::TensorMap<Tensor3d_t>;
    using Tensor2d_t    = Eigen::Tensor<float,2,Eigen::RowMajor>;
    using TensorMap2d_t = Eigen::TensorMap<Tensor2d_t>;

    auto          op           = MatchByTagOp<float>::instance(detection_threshold,tag_threshold,use_detection_val);
    TensorMap3d_t _tag_k(tag_k.data_ptr<float>(),tag_k.size(0),tag_k.size(1),tag_k.size(2));
    TensorMap3d_t _loc_k(loc_k.data_ptr<float>(),loc_k.size(0),loc_k.size(1),loc_k.size(2));
    TensorMap2d_t _val_k(val_k.data_ptr<float>(),val_k.size(0),val_k.size(1));
    auto          _output      = op.Compute(_tag_k,_loc_k,_val_k);
    torch::Tensor output       = torch::from_blob(_output.data(), /*sizes= */{_output.dimension(0), _output.dimension(1),_output.dimension(2)});

    return output.clone();
}
torch::Tensor hrnet_refine(torch::Tensor ans,torch::Tensor det,torch::Tensor tag)
{
    using Tensor4d_t    = Eigen::Tensor<float,4,Eigen::RowMajor>;
    using TensorMap4d_t = Eigen::TensorMap<Tensor4d_t>;
    using Tensor5d_t    = Eigen::Tensor<float,5,Eigen::RowMajor>;
    using TensorMap5d_t = Eigen::TensorMap<Tensor5d_t>;

    auto          op           = HRNetRefine<float>::instance();
    std::cout<<det.size(0)<<","<<det.size(1)<<","<<det.size(2)<<","<<det.size(3)<<std::endl;
    TensorMap4d_t _ans(ans.data_ptr<float>(),ans.size(0),ans.size(1),ans.size(2),ans.size(3));
    TensorMap4d_t _det(det.data_ptr<float>(),det.size(0),det.size(1),det.size(2),det.size(3));
    TensorMap5d_t _tag(tag.data_ptr<float>(),tag.size(0),tag.size(1),tag.size(2),tag.size(3),tag.size(4));
    auto          _output      = op.Compute(_ans,_det,_tag);
    torch::Tensor output       = torch::from_blob(_output.data(), /*sizes= */{_output.dimension(0), _output.dimension(1),_output.dimension(2),_output.dimension(3)});

    return output.clone();
}
TORCH_LIBRARY(hrnet_ops, m) {
      m.def("match_by_tag", match_by_tag);
      m.def("hrnet_refine", hrnet_refine);
}
