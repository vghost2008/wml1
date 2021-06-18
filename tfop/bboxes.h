#pragma once
#include <math.h>
#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>
#include <boost/algorithm/clamp.hpp>
#include <mutex>
#include <future>
#include <list>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "wmacros.h"
/*
 * box:(ymin,xmin,ymax,xmax)
 */
template<typename T0,typename T1>
auto bboxes_jaccard(const T0& box0, const T1& box1)
{
	const auto int_ymin  = std::max(box0[0],box1[0]);
	const auto int_xmin  = std::max(box0[1],box1[1]);
	const auto int_ymax  = std::min(box0[2],box1[2]);
	const auto int_xmax  = std::min(box0[3],box1[3]);
	const auto int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const auto int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto int_vol   = int_h *int_w;
	const auto vol1      = (box0[2]-box0[0]) *(box0[3]-box0[1]);
	const auto vol2      = (box1[2]-box1[0]) *(box1[3]-box1[1]);
	const auto union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return int_vol/union_vol;
}
template<typename T0,typename T1>
auto bboxes_jaccardv1(const T0& box0, const T1& box1)
{
	const auto int_ymin  = std::max(box0(0),box1(0));
	const auto int_xmin  = std::max(box0(1),box1(1));
	const auto int_ymax  = std::min(box0(2),box1(2));
	const auto int_xmax  = std::min(box0(3),box1(3));
	const auto int_h     = std::max<float>(int_ymax-int_ymin,0.);
	const auto int_w     = std::max<float>(int_xmax-int_xmin,0.);
	const auto int_vol   = int_h *int_w;
	const auto vol1      = (box0(2)-box0(0)) *(box0(3)-box0(1));
	const auto vol2      = (box1(2)-box1(0)) *(box1(3)-box1(1));
	const auto union_vol = vol1+vol2-int_vol;

	if(union_vol<1E-6) return 0.0f;

	return float(int_vol/union_vol);
}
template<typename T0,typename T1>
auto bboxes_jaccardv2(const T0& box0, const T1& box1,bool is_h)
{
    if(is_h) {
        const auto int_ymin    =  std::max(box0(0),box1(0));
        const auto int_ymax    =  std::min(box0(2),box1(2));
        const auto union_ymin  =  std::min(box0(0),box1(0));
        const auto union_ymax  =  std::max(box0(2),box1(2));
        const auto int_h       =  std::max<float>(int_ymax-int_ymin,0.);
        const auto union_h     =  union_ymax-union_ymin;
        if(union_h<1e-8)
            return 0.0f;
        return int_h/union_h;
    } else {
        const auto int_xmin    =  std::max(box0(1),box1(1));
        const auto int_xmax    =  std::min(box0(3),box1(3));
        const auto union_xmin  =  std::min(box0(1),box1(1));
        const auto union_xmax  =  std::max(box0(3),box1(3));
        const auto int_w       =  std::max<float>(int_xmax-int_xmin,0.);
        const auto union_w     =  union_xmax-union_xmin;
        if(union_w<1e-8)
            return 0.0f;
        return int_w/union_w;
    }
}
/*
 * box:(ymin,xmin,ymax,xmax)
 * 仅计算两个交叉的box的交叉面积占box0的百分比
 */
template<typename T>
T bboxes_jaccard_of_box0(const T* box0, const T* box1)
{
	const auto int_ymin = std::max(box0[0],box1[0]);
	const auto int_xmin = std::max(box0[1],box1[1]);
	const auto int_ymax = std::min(box0[2],box1[2]);
	const auto int_xmax = std::min(box0[3],box1[3]);
	const auto int_h    = std::max<T>(int_ymax-int_ymin,0.);
	const auto int_w    = std::max<T>(int_xmax-int_xmin,0.);
	const auto int_vol  = int_h *int_w;
	const auto box0_vol = (box0[2]-box0[0]) *(box0[3]-box0[1]);

	if(box0_vol<1E-6) return 0.0f;

	return int_vol/box0_vol;
}
template<typename T0,typename T1>
float bboxes_jaccard_of_box0v1(const T0& box0, const T1& box1)
{
	const auto int_ymin = std::max(box0(0),box1(0));
	const auto int_xmin = std::max(box0(1),box1(1));
	const auto int_ymax = std::min(box0(2),box1(2));
	const auto int_xmax = std::min(box0(3),box1(3));
	const auto int_h    = std::max<float>(int_ymax-int_ymin,0.);
	const auto int_w    = std::max<float>(int_xmax-int_xmin,0.);
	const auto int_vol  = int_h *int_w;
	const auto box0_vol = (box0(2)-box0(0)) *(box0(3)-box0(1));

	if(box0_vol<1E-6) return 0.0f;

	return int_vol/box0_vol;
}
template<typename T0,typename T1,typename T2>
void bboxes_envelope(const T0& box0, const T1& box1,T2& box2)
{
	const auto ymin = std::min(box0(0),box1(0));
	const auto xmin = std::min(box0(1),box1(1));
	const auto ymax = std::max(box0(2),box1(2));
	const auto xmax = std::max(box0(3),box1(3));

    box2(0) = ymin;
    box2(1) = xmin;
    box2(2) = ymax;
    box2(3) = xmax;
}
template<typename T0,typename T2>
void bboxes_envelope(const std::vector<T0>& bboxes, T2& box2)
{
    if(bboxes.size()==1) {
        box2 = bboxes[0];
        return;
    }
    if(bboxes.size()==2) {
        return bboxes_envelope(bboxes[0],bboxes[1],box2);
    }
    box2 = bboxes[0];
    for(auto i=1; i<bboxes.size(); ++i) {
        bboxes_envelope(box2,bboxes[i],box2);
    }
}

/*
 * box:ymin,xmin,ymax,xmax
 * return:cy,cx,h,w
 */
template<typename T>
auto box_minmax_to_cxywh(const T& box)
{
	return std::make_tuple((box[0]+box[2])/2.0,(box[1]+box[3])/2.0,box[2]-box[0],box[3]-box[1]);
}
/*
 * box:cy,cx,h,w
 * return:ymin,xmin,ymax,xmax
 */
template<typename T>
std::tuple<T,T,T,T> box_cxywh_to_minmax(const T* box)
{
	return std::make_tuple(box[0]-box[2]/2.0,box[1]-box[3]/2.0,box[0]+box[2]/2.,box[1]+box[3]/2);
}
template<typename T>
std::tuple<T,T,T,T> box_cxywh_to_minmax(T cy, T cx, T h, T w)
{
	return std::make_tuple(cy-h/2.0,cx-w/2.,cy+h/2.0,cx+w/2.0);
}
/*
 * box:ymin,xmin,ymax,xmax
 */
template<typename T>
T box_area(const T* box) {
	const auto h = std::max<T>(0,box[2]-box[0]);
	const auto w = std::max<T>(0,box[3]-box[1]);
	return h*w;
}
/*
 * ref_box,target_box为在原图中的box
 * output为将原图按ref_box剪切后，target_box在新图中应该的大小与位置
 * output会截断到[0,1]
 */
template<typename T>
bool cut_box(const T* ref_box,const T* target_box,T* output) {

	bzero(output,sizeof(T)*4);
	if((target_box[0]>=ref_box[2])
			|| (target_box[2]<=ref_box[0])
			|| (target_box[1]>=ref_box[3])
			|| (target_box[3]<=ref_box[1]))
		return false;

	const auto w = ref_box[3]-ref_box[1];
	const auto h = ref_box[2]-ref_box[0];

	if((w<1E-5) 
			|| (h<1E-5))
		return false;

	output[0] = (target_box[0]-ref_box[0])/h;
	output[2] = (target_box[2]-ref_box[0])/h;
	output[1] = (target_box[1]-ref_box[1])/w;
	output[3] = (target_box[3]-ref_box[1])/w;

	for(auto i=0; i<4; ++i) {
		output[i] = boost::algorithm::clamp<float>(output[i],0.0f,1.0f);
	}
	return true;
}

template<typename T>
void copy_box(const T* input,T* output) {
	for(int i=0; i<4; ++i)
		output[i] = input[i];
}
template<typename T>
void copy_boxes(const T* input,T* output,int nr) {
	for(int i=0; i<nr; ++i) {
		copy_box(input,output);
		input += 4;
		output += 4;
	}
}
template<typename T>
T box_size(const T* input) {
	return (input[2]-input[0])*(input[3]-input[1]);
}
template<typename T>
float box_sizev1(const T& input) {
	return (input(2)-input(0))*(input(3)-input(1));
}
inline float get_gaussian_radius(float height,float width,float min_overlap)
{
    auto a1 = 1;
    auto b1 = (height+width);
    auto c1 = width*height*(1-min_overlap)/(1+min_overlap);
    auto sq1 = sqrt(b1*b1-4*a1*c1);
    auto r1 = (b1+sq1)/2;

    auto a2 = 4;
    auto b2 = 2*(height+width);
    auto c2 = width*height*(1-min_overlap);
    auto sq2 = sqrt(b2*b2-4*a2*c2);
    auto r2 = (b2+sq2)/2;

    auto a3 = 4*min_overlap;
    auto b3 = -2*min_overlap*(height+width);
    auto c3 = -width*height*(1-min_overlap);
    auto sq3 = sqrt(b3*b3-4*a3*c3);
    auto r3 = (b3+sq3)/2;
    return std::max<float>(0,std::min<float>({r1,r2,r3}));
}
