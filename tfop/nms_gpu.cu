#include "wtoolkit_cuda.h"
#include "bboxes.cu.h"
constexpr auto kBlockSize = 32;
#ifdef GOOGLE_CUDA
__global__ void do_nms_for_a_bbox_cw(const float* src_box,const int src_classes,const float* bboxes,const int* classes,const float threshold,bool* keep_mask,int data_nr,int begin_index)
{
    const auto block_size = blockDim.x *gridDim.x;
    const auto repeat_nr = (data_nr+block_size-1)/block_size;
    __shared__ float _src_box[4];

    if(threadIdx.x==0)
        for(auto i=0; i<4; ++i)
            _src_box[i] = src_box[i];

    for(auto i=0; i<repeat_nr; ++i) {
        auto       index      = blockDim.x *blockIdx.x+threadIdx.x+begin_index+i*block_size;

        if(index>=data_nr)
            return;

        if(classes[index] != src_classes)
            return;

        const auto cs = cuda_bboxes_jaccard(_src_box,bboxes+index*4);

        if(cs<threshold)
            return;

        keep_mask[index] = false;
    }
}
__global__ void do_nms_classes_wise_kernel(const float* bboxes, const int* classes,const float threshold,bool* keep_mask,int data_nr)
{
    const auto end = data_nr-1;
    for(auto i=0; i<end; ++i) {
        //auto grid_nr = std::min<int>(std::max<int>(1,(data_nr-i-1)/kBlockSize),32);
        auto grid_nr = std::max<int>(1,(data_nr-i-1)/kBlockSize);
        if(keep_mask[i]) {
            do_nms_for_a_bbox_cw<<<grid_nr,kBlockSize>>>(bboxes+i*4,classes[i],bboxes,classes,threshold,keep_mask,data_nr,i+1);
            cudaDeviceSynchronize();
        }
    }
}
void do_nms_classes_wise(const float* _bboxes, const int* _classes,const float threshold,bool* _keep_mask,int data_nr)
{
    auto bboxes = make_cuda_unique(_bboxes,data_nr);
    auto classes = make_cuda_unique(_classes,data_nr);
    auto keep_mask = make_cuda_unique<bool>(data_nr);

    CHECK_OK(cudaMemset(keep_mask.get(),0x1,sizeof(bool)*data_nr));
    //
    do_nms_classes_wise_kernel<<<1,1>>>(bboxes.get(),classes.get(),threshold,keep_mask.get(),data_nr);
    cudaDeviceSynchronize();
    //
    CHECK_OK(cudaMemcpy(_keep_mask,keep_mask.get(),data_nr*sizeof(bool),cudaMemcpyDeviceToHost));
}
#endif
