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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include <opencv2/opencv.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include "bboxes.h"
#include "wtoolkit.h"

using namespace tensorflow;
using namespace std;
namespace bm=boost::mpl;
cv::RotatedRect getRotatedRect(const cv::Mat& img)
{
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    vector<cv::Point> points;

    cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));

    for (auto &cont:contours) 
        points.insert(points.end(),cont.begin(),cont.end());

    return  cv::minAreaRect(points);
}
cv::RotatedRect getRotatedRect(const uint8_t* data,int height,int width)
{
    const cv::Mat img(height,width,CV_8UC1,(uint8_t*)data);
    return getRotatedRect(img);
}

typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("MedianBlur")
    .Attr("T: {uint8}")
	.Attr("ksize:int")
    .Input("image: T")
	.Output("outimage:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class MedianBlurOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<T,4,Eigen::RowMajor>;
	public:
		explicit MedianBlurOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
		}
        void process_one_image(T* data, int width,int height)
        {
            cv::Mat img(height,width,CV_8UC1,data);
            cv::medianBlur(img,img,ksize_);
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_image  = context->input(0);
			TensorShape   output_shape  = _input_image.shape();
			Tensor       *output_tensor = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            if(_input_image.dims() == 3) {
                output_tensor->template tensor<T,3>() = _input_image.template tensor<T,3>();
                OP_REQUIRES(context, _input_image.dim_size(2) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                process_one_image(output_tensor->template flat<T>().data(),_input_image.dim_size(1),_input_image.dim_size(0));
            } else if(_input_image.dims()==4) {
                output_tensor->template tensor<T,4>() = _input_image.template tensor<T,4>();
                OP_REQUIRES(context, _input_image.dim_size(3) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                const auto img_size = _input_image.dim_size(1)*_input_image.dim_size(2);
                auto data = output_tensor->template flat<T>().data();
                for(auto i=0; i<_input_image.dim_size(0); ++i) {
                    auto d = data+i*img_size;
                    process_one_image(d,_input_image.dim_size(2),_input_image.dim_size(1));
                }
            } else {
                OP_REQUIRES(context, _input_image.dims() == 3, errors::InvalidArgument("Error dims size."));
            }
        }

	private:
        int ksize_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("MedianBlur").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MedianBlurOp<CPUDevice, uint8_t>);

REGISTER_OP("BilateralFilter")
    .Attr("T: {float,uint8}")
	.Attr("d:int")
	.Attr("sigmaColor:float")
	.Attr("sigmaSpace:float")
    .Input("image: T")
	.Output("outimage:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class BilateralFilterOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<T,4,Eigen::RowMajor>;
	public:
		explicit BilateralFilterOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("d", &d_));
			OP_REQUIRES_OK(context, context->GetAttr("sigmaColor", &sigmaColor_));
			OP_REQUIRES_OK(context, context->GetAttr("sigmaSpace", &sigmaSpace_));
		}
        template<typename TT>
        void process_one_image(const TT* data_i, const TT* data_o,int width,int height,int channel)
        {
            assert(false);
        }
        void process_one_image(const uint8_t* data_i, uint8_t* data_o,int width,int height,int channel)
        {
            if(1==channel) {
                __process_one_image<CV_8UC1>(data_i,data_o,width,channel);
            } else {
                __process_one_image<CV_8UC3>(data_i,data_o,width,channel);
            }
        }
        void process_one_image(const float* data_i, float* data_o,int width,int height,int channel)
        {
            cout<<data_i<<","<<data_o<<endl;
            if(1==channel) {
                __process_one_image<CV_32FC1>(data_i,data_o,width,height,channel);
            } else {
                __process_one_image<CV_32FC3>(data_i,data_o,width,height,channel);
            }
        }
        template<int type,typename TT>
        void __process_one_image(const TT* data_i, TT* data_o,int width,int height,int channel)
        {
            cv::Mat img_i(height,width,type,const_cast<TT*>(data_i));
            cv::Mat img_o(height,width,type,data_o);
            cv::bilateralFilter(img_i,img_o,d_,sigmaColor_,sigmaSpace_);
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_image  = context->input(0);
			TensorShape   output_shape  = _input_image.shape();
			Tensor       *output_tensor = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

            if(_input_image.dims() == 3) {
                output_tensor->template tensor<T,3>() = _input_image.template tensor<T,3>();
                OP_REQUIRES(context, _input_image.dim_size(2) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                auto data_o = output_tensor->template flat<T>().data();
                auto data_i = _input_image.template flat<T>().data();
                process_one_image(data_i,data_o,_input_image.dim_size(1),_input_image.dim_size(0),_input_image.dim_size(2));
            } else if(_input_image.dims()==4) {
                output_tensor->template tensor<T,4>() = _input_image.template tensor<T,4>();
                OP_REQUIRES(context, _input_image.dim_size(3) == 1, errors::InvalidArgument("Error channel size, should be 1"));
                const auto img_size = _input_image.dim_size(1)*_input_image.dim_size(2);
                auto data_o = output_tensor->template flat<T>().data();
                auto data_i = _input_image.template flat<T>().data();
                for(auto i=0; i<_input_image.dim_size(0); ++i) {
                    auto d_i = data_i+i*img_size;
                    auto d_o = data_o+i*img_size;
                    process_one_image(d_i,d_o,_input_image.dim_size(2),_input_image.dim_size(1),_input_image.dim_size(3));
                }
            } else {
                OP_REQUIRES(context, _input_image.dims() == 3, errors::InvalidArgument("Error dims size."));
            }
        }

	private:
        int d_ = 0;
        float sigmaColor_ = 0;
        float sigmaSpace_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MedianBlurOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_CPU).TypeConstraint<float>("T"), BilateralFilterOp<CPUDevice, float>);

/*
 * image:[N,H,W], 二值图像, 整图尺寸的mask
 * 当res_points==True时
 * return:
 * [N,4,2] 分别为[[[p0.x,p0.y],[p1.x,p1.y],..[p3.x,p3.y]],[....]]]
 * 当res_points==False时:
 * [N,3,2] 分别为[[[center.x,center.y],[width,height],[angle,_]],[.....]]]
 * 旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角。并且这个边的边长是width，另一条边边长是height。也就是说，在这里，width与height不是按照长短来定义的。
 * 在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。所以，θ∈（-90度，0]
 */
REGISTER_OP("MinAreaRect")
    .Attr("T: {uint8}")
    .Attr("res_points: bool = True")
    .Input("image: T")
	.Output("box:float32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            bool res_points = true;
            c->GetAttr("res_points",&res_points);
            auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
                auto shape = c->MakeShape({batch_size,res_points?4:3,2});
                c->set_output(0, shape);
			return Status::OK();
            });

template <typename Device, typename T>
class MinAreaRectOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<T,4,Eigen::RowMajor>;
	public:
		explicit MinAreaRectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("res_points", &res_points_));
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_image  = context->input(0);

			OP_REQUIRES(context, _input_image.dims() == 3, errors::InvalidArgument("image must be at 3-dimensional"));

            auto         input_image   = _input_image.template flat<T>().data();
            const auto   batch_size    = _input_image.dim_size(0);
            const auto   img_height    = _input_image.dim_size(1);
            const auto   img_width     = _input_image.dim_size(2);
            const int    dim3d[]       = {batch_size,res_points_?4:3,2};
            TensorShape  output_shape;
            Tensor      *output_tensor = nullptr;

            TensorShapeUtils::MakeShape(dim3d,3,&output_shape);

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto      o_tensor = output_tensor->template tensor<float,3>();

            o_tensor.setZero();

            for(auto i=0; i<batch_size; ++i) {
                try{
                    auto rect = getRotatedRect(input_image+i*img_height*img_width,img_height,img_width);

                    if(res_points_) {
                        cv::Point2f P[4];
                        rect.points(P);
                        for(auto j=0; j<4; ++j) {
                            o_tensor(i,j,0) = P[j].x;
                            o_tensor(i,j,1) = P[j].y;
                        }
                    } else {
                        o_tensor(i,0,0) = rect.center.x;
                        o_tensor(i,0,1) = rect.center.y;
                        o_tensor(i,1,0) = rect.size.width;
                        o_tensor(i,1,1) = rect.size.height;
                        o_tensor(i,2,0) = rect.angle;
                        o_tensor(i,2,1) = 0;
                    }
                } catch(...) {
                }
            }
        }

	private:
        bool res_points_ = true;
};
REGISTER_KERNEL_BUILDER(Name("MinAreaRect").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MinAreaRectOp<CPUDevice, uint8_t>);

/*
 * mask:[N,H,W], 二值图像, 仅instance bboxes内部部分
 * bboxes:[N,4]绝对坐标
 * 当res_points==True时
 * return:
 * [N,4,2] 分别为[[[p0.x,p0.y],[p1.x,p1.y],..[p3.x,p3.y]],[....]]]
 * 当res_points==False时:
 * [N,3,2] 分别为[[[center.x,center.y],[width,height],[angle,_]],[.....]]]
 * 旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角。并且这个边的边长是width，另一条边边长是height。也就是说，在这里，width与height不是按照长短来定义的。
 * 在opencv中，坐标系原点在左上角，相对于x轴，逆时针旋转角度为负，顺时针旋转角度为正。所以，θ∈（-90度，0]
 */
REGISTER_OP("MinAreaRectWithBboxes")
    .Attr("res_points: bool = True")
    .Attr("size_limit: list(int)= [63,1024]")
    .Input("mask: uint8")
    .Input("bboxes: float")
	.Output("box:float")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            bool res_points = true;
            c->GetAttr("res_points",&res_points);
            auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
                auto shape = c->MakeShape({batch_size,res_points?4:3,2});
                c->set_output(0, shape);
			return Status::OK();
            });

template <typename Device>
class MinAreaRectWithBboxesOp: public OpKernel {
    private:
        using bbox_t = Eigen::Tensor<float,1,Eigen::RowMajor>;
        using mask_t = Eigen::Tensor<uint8_t,2,Eigen::RowMajor>;
        using Tensor3D = Eigen::Tensor<uint8_t,3,Eigen::RowMajor>;
        using Tensor4D = Eigen::Tensor<uint8_t,4,Eigen::RowMajor>;
	public:
		explicit MinAreaRectWithBboxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("res_points", &res_points_));
			OP_REQUIRES_OK(context, context->GetAttr("size_limit", &size_limit_));
		}
        static cv::Mat to_mat(const mask_t& msk) 
        {
            cv::Mat res(msk.dimension(0),msk.dimension(1),CV_8UC1,(void*)msk.data());
            return res.clone();
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_mask= context->input(0);
			const Tensor &_input_bboxes = context->input(1);

			OP_REQUIRES(context, _input_mask.dims() == 3, errors::InvalidArgument("mask must be at 3-dimensional"));
			OP_REQUIRES(context, _input_bboxes.dims() == 2, errors::InvalidArgument("bboxes must be at 2-dimensional"));

            auto         input_mask = _input_mask.template tensor<uint8_t,3>();
            auto         input_bboxes = _input_bboxes.template tensor<float,2>();
            const auto   batch_size    = _input_mask.dim_size(0);
            const auto   msk_height    = _input_mask.dim_size(1);
            const auto   msk_width     = _input_mask.dim_size(2);
            const int    dim3d[]       = {batch_size,res_points_?4:3,2};
            TensorShape  output_shape;
            Tensor      *output_tensor = nullptr;

            TensorShapeUtils::MakeShape(dim3d,3,&output_shape);

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
            auto      o_tensor = output_tensor->template tensor<float,3>();

            o_tensor.setZero();

            for(auto i=0; i<batch_size; ++i) {
                try{
                    mask_t r_mask = input_mask.chip(i,0);
                    auto m_mask = to_mat(r_mask);
                    bbox_t box = input_bboxes.chip(i,0);
                    auto size = get_size(box);
                    cv::Mat n_mask;

                    cv::resize(m_mask,n_mask,cv::Size(get<1>(size),get<0>(size)),0,0,CV_INTER_LINEAR);

                    auto rect = getRotatedRect(n_mask);
                    auto scale = get<2>(size);

                    if(res_points_) {
                        cv::Point2f P[4];
                        rect.points(P);
                        for(auto j=0; j<4; ++j) {
                            o_tensor(i,j,0) = P[j].x/scale+box(1);
                            o_tensor(i,j,1) = P[j].y/scale+box(0);
                        }
                    } else {
                        o_tensor(i,0,0) = rect.center.x/scale+box(1);
                        o_tensor(i,0,1) = rect.center.y/scale+box(0);
                        o_tensor(i,1,0) = rect.size.width/scale;
                        o_tensor(i,1,1) = rect.size.height/scale;
                        o_tensor(i,2,0) = rect.angle;
                        o_tensor(i,2,1) = 0;
                    }
                } catch(...) {
                }
            }
        }
        auto get_size(const bbox_t& box) {
            auto         w              = box(3)-box(1);
            auto         h              = box(2)-box(0);
            const double min_size_limit = size_limit_[0] *size_limit_[0];
            const double max_size_limit = size_limit_[1] *size_limit_[1];
            double       scale          = 1.0;

            if(h*w>max_size_limit) {
                scale = sqrt(max_size_limit/(h*w));
            } else if(h*w<min_size_limit) {
                scale = sqrt(min_size_limit/(h*w));
            }
            h = h*scale;
            w = w*scale;
            return make_tuple(h,w,scale);
        }

	private:
        bool        res_points_ = true;
        vector<int> size_limit_;
};
REGISTER_KERNEL_BUILDER(Name("MinAreaRectWithBboxes").Device(DEVICE_CPU), MinAreaRectWithBboxesOp<CPUDevice>);

/*
 * 对输入image[H,W,C] C=1 or C=3 旋转任意角度
 */
REGISTER_OP("TensorRotate")
    .Attr("T: {uint8,float}")
    .Input("image: T")
    .Input("angle: float")
	.Output("o_image:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto input_shape0 = c->input(0);
            c->set_output(0, input_shape0);
			return Status::OK();
            });

template <typename Device, typename T>
class TensorRotateOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using type_to_int_c1 = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
        using type_to_int_c3 = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC3>>
                  , bm::pair<float,bm::int_<CV_32FC3>>
             >;
	public:
		explicit TensorRotateOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_input_img = context->input(0);
			const Tensor &_angle     = context->input(1);

			OP_REQUIRES(context, _input_img.dims() == 3, errors::InvalidArgument("tensor must be a 3-dimensional tensor"));
			OP_REQUIRES(context, _angle.dims() == 0, errors::InvalidArgument("angle be a 0-dimensional tensor"));

            const auto img_height  = _input_img.dim_size(0);
            const auto img_width   = _input_img.dim_size(1);
            const auto img_channel = _input_img.dim_size(2);

			OP_REQUIRES(context, (img_channel == 1)||(img_channel==3), errors::InvalidArgument("image channel must be 1 or 3."));

            auto        input_img     = _input_img.template flat<T>().data();
            auto        angle         = _angle.template flat<float>().data()[0];
            Tensor     *output_tensor = nullptr;
            const auto  cv_type       = (img_channel==1?bm::at<type_to_int_c1,T>::type::value:bm::at<type_to_int_c3,T>::type::value);

			OP_REQUIRES_OK(context, context->allocate_output(0, _input_img.shape(), &output_tensor));

            auto      o_tensor = output_tensor->template flat<T>().data();
            cv::Mat i_img(img_height,img_width,cv_type,(uint8_t*)input_img);
            cv::Mat o_img(img_height,img_width,cv_type,(uint8_t*)o_tensor);
            const cv::Point2f cp(img_width/2,img_height/2);
            cv::Mat r = cv::getRotationMatrix2D(cp,angle,1.0);

            cv::warpAffine(i_img,o_img,r,cv::Size(img_width,img_height));
        }
};
REGISTER_KERNEL_BUILDER(Name("TensorRotate").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), TensorRotateOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("TensorRotate").Device(DEVICE_CPU).TypeConstraint<float>("T"), TensorRotateOp<CPUDevice, float>);

/*
 * 输入绝对坐标
 */
REGISTER_OP("BboxesRotate")
    .Attr("T: {uint8,float}")
    .Attr("type:int=0")
    .Input("bboxes: T")
    .Input("angle: float")
    .Input("img_size: int32")
	.Output("o_bboxes:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto input_shape0 = c->input(0);
            c->set_output(0, input_shape0);
			return Status::OK();
            });

template <typename Device, typename T>
class BboxesRotateOp: public OpKernel {
	public:
		explicit BboxesRotateOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("type", &type_));
		}
		void Compute(OpKernelContext* context) override
		{
			const auto &_input_bboxes = context->input(0);
			const auto &_angle     = context->input(1);
			const auto &_size = context->input(2);

			OP_REQUIRES(context, _input_bboxes.dims() == 2, errors::InvalidArgument("tensor must be a 2-dimensional tensor"));
			OP_REQUIRES(context, _angle.dims() == 0, errors::InvalidArgument("angle be a 0-dimensional tensor"));
			OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size be a 1-dimensional tensor"));

            const auto  img_size      = _size.template flat<int32>().data();
            const auto  img_height    = img_size[0];
            const auto  img_width     = img_size[1];
            const auto  boxes_nr      = _input_bboxes.dim_size(0);
            auto        input_bboxes  = _input_bboxes.template tensor<T,2>();
            auto        angle         = _angle.template flat<float>().data()[0];
            Tensor     *output_tensor = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, _input_bboxes.shape(), &output_tensor));

            auto      o_tensor = output_tensor->template tensor<T,2>();
            const cv::Point2f cp(img_width/2,img_height/2);
            cv::Mat r = cv::getRotationMatrix2D(cp,angle,1.0);
            for(auto i=0; i<boxes_nr; ++i) {
                transform(input_bboxes,i,r,o_tensor);
            }
        }
        template<typename BT0,typename BT1>
        void transform(const BT0& input_bboxes,int index, const cv::Mat& r,BT1& output_bboxes) {
            auto points = get_points(input_bboxes,index);
            vector<cv::Point> out_points(points.size());
            cv::transform(points,out_points,r);
            auto xmin = 1e10;
            auto xmax = 0;
            auto ymin = 1e10;
            auto ymax = 0;
            for(auto& p:out_points) {
                if(p.x<xmin)
                    xmin = p.x;
                if(p.x>xmax)
                    xmax = p.x;
                if(p.y<ymin)
                    ymin = p.y;
                if(p.y>ymax)
                    ymax = p.y;
            }
            output_bboxes(index,0) = ymin;
            output_bboxes(index,1) = xmin;
            output_bboxes(index,2) = ymax;
            output_bboxes(index,3) = xmax;
        }
        template<typename BT> 
            vector<cv::Point> get_points(const BT& bboxes,int index) {
                const auto kNr = 100;
                vector<cv::Point> points;
                switch(type_) {
                    case 0:
                        {
                            auto ymin = bboxes(index,0);
                            auto xmin = bboxes(index,1);
                            auto ymax = bboxes(index,2);
                            auto xmax = bboxes(index,3);
                            const auto bw = xmax-xmin;
                            const auto bh = ymax-ymin;
                            const auto cx = (xmax+xmin)/2;
                            const auto cy = (ymax+ymin)/2;
                            const auto a = bw/2;
                            const auto b = bh/2;
                            for(auto i=0; i<kNr; ++i) {
                                const auto theta = 2*M_PI*i/kNr;
                                const auto x = cx+a*cos(theta);
                                const auto y = cy+b*sin(theta);
                                points.emplace_back(x,y);
                            }
                        }
                        break;
                    case 1:
                    default:
                        {
                            auto ymin = bboxes(index,0);
                            auto xmin = bboxes(index,1);
                            auto ymax = bboxes(index,2);
                            auto xmax = bboxes(index,3);
                            const auto bw = xmax-xmin;
                            const auto bh = ymax-ymin;
                            const auto cx = (xmax+xmin)/2;
                            const auto cy = (ymax+ymin)/2;
                            const auto a = bw/2;
                            const auto b = bh/2;
                            points.emplace_back(xmin,ymin);
                            points.emplace_back(xmin,ymax);
                            points.emplace_back(xmax,ymax);
                            points.emplace_back(xmax,ymin);
                        }
                        break;

                }
                return points;
            }
    private:
        int type_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("BboxesRotate").Device(DEVICE_CPU).TypeConstraint<double>("T"), BboxesRotateOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("BboxesRotate").Device(DEVICE_CPU).TypeConstraint<float>("T"), BboxesRotateOp<CPUDevice, float>);
