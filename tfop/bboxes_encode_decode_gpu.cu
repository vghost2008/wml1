#include <vector>
#include "wtoolkit_cuda.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "bboxes.cu.h"
#include "wmacros.h"
using namespace std;
constexpr auto kBlockSize = 64;

#ifdef GOOGLE_CUDA
/*
 * 找到与每一个anchor boxes相对就的最大的gbboxes
 *
 * gbboxes:[gb_size,4] (ymin,xmin,ymax,xmax)表示ground truth box
 * anchor_bboxes:[ab_size,4] (ymin,xmin,ymax,xmax)表示待匹配的box
 * 输出:
 * scores:[ab_size]相应的iou得分
 * indexs:[ab_size]与之相对应的gbboxes索引
 * is_boundary_box:表示anchor_bboxes是否在边界上
 */
__global__ void get_scores_and_indexs(const float* gbboxes,const float* anchor_bboxes,float* scores,int* indexs,bool* is_boundary_box,size_t gb_size,size_t ab_size)
{
    const auto       a_index                = blockIdx.x;
    const auto       g_offset               = threadIdx.x;
    auto             max_i                  = -1;
    auto             max_s                  = 1e-8;
    float            abbox[4];
    float            gbbox[4];
    __shared__ int   max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];
    /*
     * gbboxes按kBlockSize划分为多个组，下面的代码找到在同一个组中与给定anchor box(a_index)对应的最大ground truth box(max_i,max_s)
     */

    for(auto i=0; i<4; ++i)
        abbox[i] = (anchor_bboxes+(a_index<<2))[i];
#ifdef PROCESS_BOUNDARY_ANCHORS
    if(cuda_is_cross_boundaries(abbox)) { 
        is_boundary_box[a_index] = true;
        return;
    }
#endif

    for(auto i=g_offset; i<gb_size; i += blockDim.x) {
        for(auto j=0; j<4; ++j)
            gbbox[j] = (gbboxes+(i<<2))[j];
        const auto cs = cuda_bboxes_jaccard(abbox,gbbox);
        //const auto cs = cuda_bboxes_jaccard(abbox,gbboxes+(i<<2));
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    max_index[g_offset] = max_i;
    max_scores[g_offset] = max_s;

    __syncthreads();

    if(g_offset != 0) return; 

    /*
     * 线程0在所有的组中找到最大的一个
     */
    max_i = -1;
    max_s = 1e-8;
    for(auto i=0; i<blockDim.x; ++i) {
        const auto cs = max_scores[i];
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    if(max_i>=0) {
        indexs[a_index] = max_index[max_i];
        scores[a_index] = max_s;
    }
}
/*
 * 找到与每一个ground truth box相对应的最大的anchor box
 * gbboxes:[gb_size,4]
 * anchor_bboxes: [ab_size,4]
 * is_boundary_box:[ab_size]
 * 输出:
 * is_max_score:[ab_size]
 * scores0:[gb_size]
 * indexs0:[gb_size]
 */
__global__ void find_max_score_index(const float* gbboxes,const float* anchor_bboxes,const bool* is_boundary_box,bool* is_max_score,size_t ab_size)
{
    const auto       g_index                = blockIdx.x;
    const auto       a_offset               = threadIdx.x;
    auto             max_i                  = -1;
    auto             max_s                  = MIN_SCORE_FOR_POS_BOX;
    float            gbbox[4];
    float            abbox[4];
    __shared__ int   max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];

    /*
     * anchor bboxes按kBlockSize分组，这部分找到在一个组里与指定的gbboxes(g_index)对应的最大的anchor boxes
     */
    for(auto i=0; i<4; ++i)
        gbbox[i] = (gbboxes+(g_index<<2))[i];

    for(auto i=a_offset; i<ab_size; i += blockDim.x) {
        for(auto j=0; j<4; ++j)
            abbox[j] = (anchor_bboxes+(i<<2))[j];
        if(is_boundary_box[i]) continue;
        const auto cs = cuda_bboxes_jaccard(gbbox,abbox);
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    max_index[a_offset] = max_i;
    max_scores[a_offset] = max_s;

    __syncthreads();

    if(a_offset != 0) return;

    /*
     * 线程0找到唯一的最大anchor box索引
     */
    max_i = -1;
    max_s = MIN_SCORE_FOR_POS_BOX;
    for(auto i=0; i<blockDim.x; ++i) {
        const auto cs = max_scores[i];
        if(cs>max_s) {
            max_i = i;
            max_s = cs;
        }
    }
    if(max_i>=0) 
        is_max_score[max_index[max_i]] = true;
}
__global__ void get_labels_and_remove_indices(int* indexs,float* scores,const bool* is_max_score,const int* glabels,int* out_labels,bool* remove_indices,float neg_threshold,float pos_threshold)
{
    auto        a_index = blockIdx.x;
    const auto &index   = indexs[a_index];
    const auto  score   = scores[a_index];

    if((score>=pos_threshold) || (score<neg_threshold) || is_max_score[a_index]) {
        remove_indices[a_index] = false;
        if((score>=pos_threshold) || is_max_score[a_index]) {
            out_labels[a_index] = glabels[index];
        } else {
            out_labels[a_index]  =  0;
            indexs[a_index]      =  -1;
            scores[a_index]      =  0;
        }
    } else {
        remove_indices[a_index]  =  true;
        indexs[a_index]          =  -1;
        scores[a_index]          =  0.0f;
    }
}
__global__ void get_bboxes_regression(float* out_boxes,const float* anchor_bboxes,const float* gbboxes,const int* out_labels,const bool* out_remove_indices,const int* out_index,float* prio_scaling)
{
    auto j = blockIdx.x; //a_index

    auto  outbox  = out_boxes+j*4;
    if((out_labels[j]<1) || (out_remove_indices[j])) {
        return;
    }
    auto box  = anchor_bboxes+j *4;
    auto gbox = gbboxes+out_index[j] *4;
    auto yxhw = cuda_box_minmax_to_cxywh(box);
    auto yref = std::get<0>(yxhw);
    auto xref = std::get<1>(yxhw);
    auto href = std::get<2>(yxhw);
    auto wref = std::get<3>(yxhw);

    if((href<1E-8) || (wref<1E-8)) {
        return;
    }

    auto gyxhw = cuda_box_minmax_to_cxywh(gbox);

    auto feat_cy  =  std::get<0>(gyxhw);
    auto feat_cx  =  std::get<1>(gyxhw);
    auto feat_h   =  std::get<2>(gyxhw);
    auto feat_w   =  std::get<3>(gyxhw);

    outbox[0] =  (feat_cy-yref)/(href*prio_scaling[0]);
    outbox[1] =  (feat_cx-xref)/(wref*prio_scaling[1]);
    outbox[2] =  log(feat_h/href)/prio_scaling[2];
    outbox[3] =  log(feat_w/wref)/prio_scaling[3];
}
__global__ void bboxes_decode_kernel(const float* anchor_bboxes,const float* regs,const float* prio_scaling,float* out_bboxes,size_t data_nr)
{
    const auto b           = threadIdx.x+blockIdx.x *blockDim.x;

    if(b>=data_nr) return;

    const auto base_offset = b *4;
    const auto regs_data   = regs+base_offset;
    const auto box_data    = anchor_bboxes+base_offset;
    float      y;
    float      x;
    float      href;
    float      wref;
    auto       xywh        = cuda_box_minmax_to_cxywh(box_data);

    y = std::get<0>(xywh);
    x = std::get<1>(xywh);
    href = std::get<2>(xywh);
    wref = std::get<3>(xywh);

    auto       cy          = clamp<float>(regs_data[0] *prio_scaling[0],-10.0f,10.0f) *href+y;
    auto       cx          = clamp<float>(regs_data[1] *prio_scaling[1],-10.0f,10.0f) *wref+x;
    auto       h           = href *exp(clamp<float>(regs_data[2] *prio_scaling[2],-10.0,10.0));
    auto       w           = wref *exp(clamp<float>(regs_data[3] *prio_scaling[3],-10.0,10.0));
    auto       output_data = out_bboxes + base_offset;
    const auto minmax      = cuda_box_cxywh_to_minmax(cy,cx,h,w);

    output_data[0] = clamp<float>(std::get<0>(minmax),0.0,1.0);
    output_data[1] = clamp<float>(std::get<1>(minmax),0.0,1.0);
    output_data[2] = clamp<float>(std::get<2>(minmax),0.0,1.0);
    output_data[3] = clamp<float>(std::get<3>(minmax),0.0,1.0); 

    if(output_data[0]>output_data[2]) 
        output_data[2] = output_data[0];
    if(output_data[1]>output_data[3])
        output_data[3] = output_data[1];
}
__host__ void get_encodes(const float* gbboxes,const float* anchor_bboxes,const int* glabels,
float* out_boxes,float* out_scores,int* out_labels,bool* out_remove_indices,int* out_index,const float* prio_scaling,
size_t gb_size,size_t ab_size,float neg_threshold,float pos_threshold,bool max_overlap_as_pos=true)
{
    cuda_unique_ptr<int> g_out_index;

    if(nullptr == out_index) {
        g_out_index = make_cuda_unique<int>(ab_size);
        out_index = g_out_index.get(); 
    }

    CHECK_OK(cudaMemset(out_boxes,0,sizeof(float)*4*ab_size));
    CHECK_OK(cudaMemset(out_scores,0,sizeof(float)*ab_size));
    CHECK_OK(cudaMemset(out_index,0xff,sizeof(int)*ab_size));
    CHECK_OK(cudaMemset(out_labels,0,sizeof(int)*ab_size));

    dim3      grid(ab_size);
    dim3      grid1(gb_size);
    auto      d_is_boundary_box = make_cuda_unique<bool>((unsigned char)(0x00),ab_size);
    cuda_unique_ptr<bool> d_is_max_score = make_cuda_unique<bool>((unsigned char)(0x00),ab_size);

    get_scores_and_indexs<<<grid,std::min<size_t>(kBlockSize,gb_size)>>>(gbboxes,anchor_bboxes,out_scores,out_index,d_is_boundary_box.get(),gb_size,ab_size);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();


    if(max_overlap_as_pos) {
        find_max_score_index<<<grid1,std::min<size_t>(kBlockSize,ab_size)>>>(gbboxes,anchor_bboxes,d_is_boundary_box.get(),d_is_max_score.get(),ab_size);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    get_labels_and_remove_indices<<<grid,1>>>(out_index,out_scores,d_is_max_score.get(),glabels,out_labels,out_remove_indices,neg_threshold,pos_threshold);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cuda_unique_ptr<float> d_prio_scaling = make_cuda_unique<float>(prio_scaling,4);

    get_bboxes_regression<<<grid,1>>>(out_boxes,anchor_bboxes,gbboxes,out_labels,out_remove_indices,out_index,d_prio_scaling.get());

    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}
void bboxes_decode_by_gpu(const float* anchor_bboxes,const float* regs,const float* prio_scaling,float* out_bboxes,size_t data_nr)
{
    if(0 == data_nr) 
        return;
    cuda_unique_ptr<float> d_prio_scaling = make_cuda_unique<float>(prio_scaling,4);
    const auto block_size = std::min<size_t>(data_nr,128);
    const auto grid_size = (data_nr+block_size-1)/block_size;

    bboxes_decode_kernel<<<grid_size,block_size>>>(anchor_bboxes,regs,d_prio_scaling.get(),out_bboxes,data_nr);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize(); 
}
__global__ void boxes_pair_jaccard_kernel(const float* bboxes, float* jaccard, int data_nr)
{
    const auto index     = threadIdx.x+blockIdx.x *blockDim.x;
    const auto src_index = index/data_nr;
    const auto dst_index = index%data_nr;

    if(index>= data_nr*data_nr)
        return;

    jaccard[index] = cuda_bboxes_jaccard(bboxes+(src_index<<2),bboxes+(dst_index<<2));
}
void boxes_pair_jaccard(const float* _bboxes,float* _jaccard,int data_nr)
{
    auto bboxes = make_cuda_unique(_bboxes,data_nr);
    auto jaccard = make_cuda_unique<float>(data_nr*data_nr);

    boxes_pair_jaccard_gpu_mem(bboxes.get(),jaccard.get(),data_nr);

    CHECK_OK(cudaMemcpy(_jaccard,jaccard.get(),data_nr*data_nr*sizeof(float),cudaMemcpyDeviceToHost));
}
void boxes_pair_jaccard_gpu_mem(const float* bboxes,float* jaccard,int data_nr)
{
    const auto grid_size = (data_nr*data_nr+kBlockSize-1)/kBlockSize;

    boxes_pair_jaccard_kernel<<<data_nr,kBlockSize>>>(bboxes,jaccard,data_nr);

    cudaDeviceSynchronize(); 
}
#endif
