#include <vector>
#include "wtoolkit_cuda.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include "wmacros.h"
#include "bboxes.cu.h"
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
 */
__global__ void matcher_get_scores_and_indexs(const float* gbboxes,const float* anchor_bboxes,float* scores,int* indexs,size_t gb_size,size_t ab_size)
{
    const auto       a_index                = blockIdx.x;
    const auto       g_offset               = threadIdx.x;
    auto             max_i                  = -1;
    auto             max_s                  = MIN_SCORE_FOR_POS_BOX;
    float            abbox[4];
    float            gbbox[4];
    __shared__ int   max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];
    /*
     * gbboxes按kBlockSize划分为多个组，下面的代码找到在同一个组中与给定anchor box(a_index)对应的最大ground truth box(max_i,max_s)
     */

    for(auto i=0; i<4; ++i)
        abbox[i] = (anchor_bboxes+(a_index<<2))[i];

    for(auto i=g_offset; i<gb_size; i += blockDim.x) {
        for(auto j=0; j<4; ++j)
            gbbox[j] = (gbboxes+(i<<2))[j];
        const auto cs = cuda_bboxes_jaccard(abbox,gbbox);
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
__global__ void matcher_get_scores_and_indexsv2(const float* gbboxes,const float* anchor_bboxes,float* scores,int* indexs,size_t gb_size,size_t ab_size)
{
    const auto       a_index                = blockIdx.x;
    const auto       g_offset               = threadIdx.x;
    auto             max_i                  = -1;
    auto             max_s                  = MIN_SCORE_FOR_POS_BOX;
    float            abbox[4];
    float            gbbox[4];
    __shared__ int   max_index[kBlockSize];
    __shared__ float max_scores[kBlockSize];
    /*
     * gbboxes按kBlockSize划分为多个组，下面的代码找到在同一个组中与给定anchor box(a_index)对应的最大ground truth box(max_i,max_s)
     */
    for(auto i=0; i<4; ++i)
        abbox[i] = (anchor_bboxes+(a_index<<2))[i];

    for(auto i=g_offset; i<gb_size; i += blockDim.x) {
        for(auto j=0; j<4; ++j)
            gbbox[j] = (gbboxes+(i<<2))[j];
        if(!cuda_is_in_gtbox(abbox,gbbox))
            continue;
        const auto cs = cuda_bboxes_jaccard(abbox,gbbox);
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
 * 输出:
 * is_max_score:[ab_size]
 * scores0:[gb_size]
 * indexs0:[gb_size]
 */
__global__ void matcher_find_max_score_index(const float* gbboxes,const float* anchor_bboxes,bool* is_max_score,size_t ab_size)
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
__global__ void matcher_get_labels(int* indexs,float* scores,const bool* is_max_score,const int* glabels,int* out_labels,float neg_threshold,float pos_threshold)
{
    auto        a_index = blockIdx.x;
    const auto &index   = indexs[a_index];
    const auto  score   = scores[a_index];

    if((score>=pos_threshold) || (score<neg_threshold) || is_max_score[a_index]) {
        if((score>=pos_threshold) || is_max_score[a_index]) {
            out_labels[a_index] = glabels[index];
        } else {
            out_labels[a_index]  =  0;
            indexs[a_index]      =  -1;
            scores[a_index]      =  0;
        }
    } else {
        indexs[a_index]      =  -1;
        scores[a_index]      =  0.0f;
        out_labels[a_index]  =  -1;
    }
}
__host__ void matcher_by_gpu(const float* gbboxes,const float* anchor_bboxes,const int* glabels,
float* out_scores,int* out_labels,int* out_index,
size_t gb_size,size_t ab_size,float neg_threshold,float pos_threshold,bool max_overlap_as_pos=true,bool force_in_gtbox=false)
{
    cuda_unique_ptr<int> g_out_index;

    if(nullptr == out_index) {
        g_out_index = make_cuda_unique<int>(ab_size);
        out_index = g_out_index.get(); 
    }

    CHECK_OK(cudaMemset(out_scores,0,sizeof(float)*ab_size));
    CHECK_OK(cudaMemset(out_index,0xff,sizeof(int)*ab_size));
    CHECK_OK(cudaMemset(out_labels,0,sizeof(int)*ab_size));

    dim3      grid(ab_size);
    dim3      grid1(gb_size);
    cuda_unique_ptr<bool> d_is_max_score = make_cuda_unique<bool>((unsigned char)(0x00),ab_size);

    if(force_in_gtbox)
        matcher_get_scores_and_indexsv2<<<grid,std::min<size_t>(kBlockSize,gb_size)>>>(gbboxes,anchor_bboxes,out_scores,out_index,gb_size,ab_size);
    else
        matcher_get_scores_and_indexs<<<grid,std::min<size_t>(kBlockSize,gb_size)>>>(gbboxes,anchor_bboxes,out_scores,out_index,gb_size,ab_size);

    auto res = cudaPeekAtLastError();
    if(res != cudaError::cudaSuccess) {
        CHECK_CUDA_ERRORS(res);
        cout<<"CUDAERROR INFO:"<<ab_size<<","<<kBlockSize<<","<<gb_size<<endl;
    }
    cudaDeviceSynchronize();

    if(max_overlap_as_pos) {
        matcher_find_max_score_index<<<grid1,std::min<size_t>(kBlockSize,ab_size)>>>(gbboxes,anchor_bboxes,d_is_max_score.get(),ab_size);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());

    matcher_get_labels<<<grid,1>>>(out_index,out_scores,d_is_max_score.get(),glabels,out_labels,neg_threshold,pos_threshold);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}
#endif
