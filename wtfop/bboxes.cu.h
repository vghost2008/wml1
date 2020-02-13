_Pragma("once")
#include "wtoolkit_cuda.h"
#ifdef GOOGLE_CUDA
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_minmax_to_cxywh(const T* box)
{
	return std::make_tuple((box[0]+box[2])/2.0,(box[1]+box[3])/2.0,box[2]-box[0],box[3]-box[1]);
}
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_cxywh_to_minmax(const T* box)
{
	return std::make_tuple(box[0]-box[2]/2.0,box[1]-box[3]/2.0,box[0]+box[2]/2.,box[1]+box[3]/2);
}
template<typename T>
__device__ std::tuple<T,T,T,T> cuda_box_cxywh_to_minmax(T cy, T cx, T h, T w)
{
	return std::make_tuple(cy-h/2.0,cx-w/2.,cy+h/2.0,cx+w/2.0);
}
template<typename T0,typename T1>
__device__ float cuda_bboxes_jaccard(const T0* box0, const T1* box1)
{
	const auto  int_ymin  = std::max(box0[0],box1[0]);
	const auto  int_xmin  = std::max(box0[1],box1[1]);
	const auto  int_ymax  = std::min(box0[2],box1[2]);
	const auto  int_xmax  = std::min(box0[3],box1[3]);
	const float int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const float int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto  int_vol   = int_h *int_w;
	const auto  vol1      = (box0[2]-box0[0]) *(box0[3]-box0[1]);
	const auto  vol2      = (box1[2]-box1[0]) *(box1[3]-box1[1]);
	const auto  union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return int_vol/union_vol;
}
template<typename T>
__device__ T clamp(T v,T min, T max)
{
    if(v<min) return min;
    if(v>max) return max;
    return v;
}
template<typename T>
__device__ inline bool cuda_is_cross_boundaries(const T* box) {
    return (box[0]<0.0) || (box[1]<0.0) || (box[2]>1.0) ||(box[3]>1.0);
}
#endif
