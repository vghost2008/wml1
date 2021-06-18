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
#include<opencv2/opencv.hpp>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include <boost/algorithm/clamp.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/at.hpp>
#include "bboxes.h"
#include <future>
#include "wtoolkit.h"

using namespace tensorflow;
using namespace std;
namespace ba=boost::algorithm;
namespace bm=boost::mpl;

typedef Eigen::ThreadPoolDevice CPUDevice;

/*
 * 
 * masks:[batch_size,Nr,h,w]
 * labels: [batch_size,Nr]
 * lens:[batch_size]
 * output_bboxes:[batch_size,nr,4] (ymin,xmin,ymax,xmax)
 * output_labels:[batch_size,nr]
 * output_lens:[batch_size]
 * output_ids:[batch_size,nr] 用于表示实例的编号，如第一个batch中的第二个实例所生成的所有的box的ids为3(id的编号从1开始)
 */
REGISTER_OP("MaskLineBboxes")
    .Attr("T: {int64,int32}")
	.Attr("max_output_nr:int")
    .Input("mask: uint8")
    .Input("labels: T")
    .Input("lens: int32")
	.Output("output_bboxes:float")
	.Output("output_labels:T")
	.Output("output_lens:int32")
	.Output("output_ids:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            int        nr = -1;

            c->GetAttr("max_output_nr",&nr);

            const auto batch_size = c->Value(c->Dim(c->input(0),0));
            const auto shape0     = c->MakeShape({batch_size,nr,4});
            const auto shape1     = c->Matrix(batch_size,nr);
            const auto shape2     = c->Vector(batch_size);

			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape2);
			c->set_output(3, shape1);
			return Status::OK();
			});

template <typename Device,typename T>
class MaskLineBboxesOp: public OpKernel {
    private:
        using bbox_t = tuple<float,float,float,float>;
	public:
		explicit MaskLineBboxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("max_output_nr", &max_output_nr_));
		}

		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("MaskLineBboxes");
			const Tensor &_mask= context->input(0);
			const Tensor &_labels= context->input(1);
			const Tensor &_lens = context->input(2);
			auto mask= _mask.template tensor<uint8_t,4>();
            auto labels = _labels.template tensor<T,2>();
            auto lens = _lens.template tensor<int32_t,1>();

			OP_REQUIRES(context, _mask.dims() == 4, errors::InvalidArgument("mask data must be 4-dimensional"));
			OP_REQUIRES(context, _labels.dims() == 2, errors::InvalidArgument("labels data must be 2-dimensional"));
			OP_REQUIRES(context, _lens.dims() == 1, errors::InvalidArgument("lens data must be 1-dimensional"));

			const auto     batch_size = _mask.dim_size(0);
			const auto     data_nr   = _mask.dim_size(1);
            list<vector<bbox_t>> out_bboxes;
            list<vector<int>> out_labels;
            list<vector<int>> out_ids;

            for(auto i=0; i<batch_size; ++i) {
                vector<bbox_t> res;
                vector<int> res_labels;
                vector<int> res_ids;
                res.reserve(1024);
                for(auto j=0; j<lens(i); ++j) {
                    const auto label = labels(i,j);
                    auto res0 = get_bboxes(mask.chip(i,0).chip(j,0));
                    if(!res0.empty()) {
                        res.insert(res.end(),res0.begin(),res0.end());
                        res_labels.insert(res_labels.end(),res0.size(),label);
                        res_ids.insert(res_ids.end(),res0.size(),j+1);
                    }
                }
			    OP_REQUIRES(context, res.size() == res_labels.size(), errors::InvalidArgument("size of bboxes should equal size of labels."));
                out_bboxes.push_back(std::move(res));
                out_labels.push_back(std::move(res_labels));
                out_ids.push_back(std::move(res_ids));
            }

            auto output_nr = max_output_nr_;

            if(output_nr<=0) {
                auto it = max_element(out_labels.begin(),out_labels.end(),[](const auto& v0,const auto& v1){ return v0.size()<v1.size();});
                output_nr = it->size();
            } 

			int dims_3d[3] = {batch_size,output_nr,4};
			int dims_2d[2] = {batch_size,output_nr};
			int dims_1d[1] = {batch_size};
			TensorShape  outshape0;
			TensorShape  outshape1;
			TensorShape  outshape2;
			Tensor      *output_bbox   = NULL;
			Tensor      *output_labels = NULL;
			Tensor      *output_lens   = NULL;
			Tensor      *output_ids    = NULL;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
			TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);
			TensorShapeUtils::MakeShape(dims_1d, 1, &outshape2);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_bbox));
			OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_labels));
			OP_REQUIRES_OK(context, context->allocate_output(2, outshape2, &output_lens));
			OP_REQUIRES_OK(context, context->allocate_output(3, outshape1, &output_ids));

			auto obbox   = output_bbox->template tensor<float,3>();
			auto olabels = output_labels->template tensor<T,2>();
			auto olens   = output_lens->template tensor<int32_t,1>();
			auto oids    = output_ids->template tensor<int32_t,2>();

            obbox.setZero();
            olabels.setZero();
            auto itb = out_bboxes.begin();
            auto itl = out_labels.begin();
            auto iti = out_ids.begin();

			for(int i=0; i<batch_size; ++i,++itb,++itl,++iti) {
                olens(i) = itl->size();
                for(auto j=0; j<olens(i); ++j) {
                    obbox(i,j,0) = std::get<0>((*itb)[j]);
                    obbox(i,j,1) = std::get<1>((*itb)[j]);
                    obbox(i,j,2) = std::get<2>((*itb)[j]);
                    obbox(i,j,3) = std::get<3>((*itb)[j]);
                    olabels(i,j) = (*itl)[j];
                    oids(i,j) = (*iti)[j];
                }
			}
		}
        /*
         * mask: [h,w]
         */
        vector<bbox_t> get_bboxes(const Eigen::Tensor<uint8_t,2,Eigen::RowMajor>& mask) {
            const auto h = mask.dimension(0);
            const auto w = mask.dimension(1);
            const auto y_delta = 1.0/h;
            const auto x_delta = 1.0/w;
            vector<bbox_t> res;
            res.reserve(256);

            for(auto i=0; i<h; ++i) {
                const auto ymin = i*y_delta;
                const auto ymax = (i+1)*y_delta;
                for(auto j=0; j<w; ++j) {
                    if(mask(i,j)<1) continue;
                    auto begin_j = j;
                    while((mask(i,j)>0) && (j<w))++j;
                    const auto xmin = begin_j*x_delta;
                    const auto xmax = (j==w)?1.0:j*x_delta;
                    res.emplace_back(ymin,xmin,ymax,xmax);
                }
            }
            return res;
        }
	private:
        int max_output_nr_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("MaskLineBboxes").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), MaskLineBboxesOp<CPUDevice,int32_t>);
REGISTER_KERNEL_BUILDER(Name("MaskLineBboxes").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), MaskLineBboxesOp<CPUDevice,tensorflow::int64>);

/*
 * 
 * masks:[Nr,h,w]
 * bboxes:[Nr,4]
 * size:[2]={H,W}
 * output_masks:[Nr,H,W]
 */
REGISTER_OP("FullSizeMask")
    .Attr("T: {float32,uint8}")
    .Input("mask: T")
    .Input("bboxes: float32")
    .Input("size: int32")
	.Output("output_masks:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto nr = c->Value(c->Dim(c->input(0),0));
            const auto shape0     = c->MakeShape({nr,-1,-1});
			c->set_output(0, shape0);
			return Status::OK();
			});

template <typename Device,typename T>
class FullSizeMaskOp: public OpKernel {
    private:
        using bbox_t = tuple<float,float,float,float>;
        using type_to_int = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
        using tensor_t = Eigen::Tensor<T,2,Eigen::RowMajor>;
        using tensor_map_t = Eigen::TensorMap<tensor_t>;
	public:
		explicit FullSizeMaskOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_mask= context->input(0);
			const Tensor &_bboxes = context->input(1);
			const Tensor &_size= context->input(2);

			OP_REQUIRES(context, _mask.dims() == 3, errors::InvalidArgument("mask data must be 3-dimensional"));
			OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes data must be 2-dimensional"));
			OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size data must be 1-dimensional"));

			auto         mask        = _mask.template flat<T>();
			auto         bboxes      = _bboxes.template tensor<float,2>();
			auto         size        = _size.template tensor<int,1>();
			const auto   mh          = _mask.dim_size(1);
			const auto   mw          = _mask.dim_size(2);
			const auto   H           = size(0);
			const auto   W           = size(1);
			const auto   data_nr     = _mask.dim_size(0);
			int          dims_3d[3]  = {data_nr,H,W};
			Tensor      *output_mask = NULL;
			TensorShape  outshape0;

			TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);

			OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_mask));

            auto o_tensor = output_mask->template tensor<T,3>();
            const auto H_max = H-1;
            const auto W_max = W-1;
            constexpr auto kMinSize = 1e-3;

            o_tensor.setZero();

            for(auto i=0; i<data_nr; ++i) {
                if((fabs(bboxes(i,3)-bboxes(i,1))<kMinSize)
                    || (fabs(bboxes(i,2)-bboxes(i,0))<kMinSize))
                    continue;

                long xmin = ba::clamp(bboxes(i,1)*W_max,0,W_max);
                long ymin = ba::clamp(bboxes(i,0)*H_max,0,H_max);
                long xmax = ba::clamp(bboxes(i,3)*W_max,0,W_max);
                long ymax = ba::clamp(bboxes(i,2)*H_max,0,H_max);
                const cv::Mat input_mask(mh,mw,bm::at<type_to_int,T>::type::value,(void*)(mask.data()+i*mh*mw));
                cv::Mat dst_mask(ymax-ymin+1,xmax-xmin+1,bm::at<type_to_int,T>::type::value);

                cv::resize(input_mask,dst_mask,cv::Size(xmax-xmin+1,ymax-ymin+1),0,0,CV_INTER_LINEAR);


                tensor_map_t src_map((T*)dst_mask.data,dst_mask.rows,dst_mask.cols);
                Eigen::array<long,2> offset = {ymin,xmin};
                Eigen::array<long,2> extents = {dst_mask.rows,dst_mask.cols};

                o_tensor.chip(i,0).slice(offset,extents) = src_map;

                /*if(((xmax-xmin>mw) || (ymax-ymin>mh)) && (xmax>xmin) && (ymax>ymin)) {
                    cv::Mat dst_mask(H,W,bm::at<type_to_int,T>::type::value,output_mask->template flat<T>().data()+H*W*i);
                    cv::Mat src_mask = dst_mask.clone();
                    const auto k = max<int>(3,sqrt((xmax-xmin)*(ymax-ymin)/(mh*mw))+1);
                    cv::medianBlur(src_mask,dst_mask,(k/2)*2+1);
                }*/
                
            }
		}
};
REGISTER_KERNEL_BUILDER(Name("FullSizeMask").Device(DEVICE_CPU).TypeConstraint<float>("T"), FullSizeMaskOp<CPUDevice,float>);
REGISTER_KERNEL_BUILDER(Name("FullSizeMask").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), FullSizeMaskOp<CPUDevice,uint8_t>);

/*
 * 对mask [Nr,H,W] 旋转指定角度，同时返回相应instance的bbox
 * bbox [N,4],[ymin,xmin,ymax,xmax], 绝对坐标
 */
REGISTER_OP("MaskRotate")
    .Attr("T: {uint8,float}")
    .Input("mask: T")
    .Input("angle: float")
	.Output("o_image:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto input_shape0 = c->input(0);
            c->set_output(0, input_shape0);
			return Status::OK();
            });

template <typename Device, typename T>
class MaskRotateOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using type_to_int = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
	public:
		explicit MaskRotateOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("RotateMask");
			const Tensor &_input_img = context->input(0);
			const Tensor &_angle = context->input(1);

			OP_REQUIRES(context, _input_img.dims() == 3, errors::InvalidArgument("tensor must be a 3-dimensional tensor"));
			OP_REQUIRES(context, _angle.dims() == 0, errors::InvalidArgument("angle be a 0-dimensional tensor"));

            auto         input_img     = _input_img.template flat<T>().data();
            auto         angle         = _angle.template flat<float>().data()[0];
            const auto   img_channel   = _input_img.dim_size(0);
            const auto   img_height    = _input_img.dim_size(1);
            const auto   img_width     = _input_img.dim_size(2);
            Tensor      *output_tensor = nullptr;


			OP_REQUIRES_OK(context, context->allocate_output(0, _input_img.shape(), &output_tensor));

            auto          o_tensor     = output_tensor->template flat<T>().data();
            const cv::Point2f cp(img_width/2,img_height/2);
            const cv::Mat r            = cv::getRotationMatrix2D(cp,angle,1.0);
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            auto fn  = [img_height,img_width,cv_type,r](T* i_data,T* o_data) {
                cv::Mat i_img(img_height,img_width,cv_type,i_data);
                cv::Mat o_img(img_height,img_width,cv_type,o_data);

                cv::warpAffine(i_img,o_img,r,cv::Size(img_width,img_height));
            };
            list<future<void>> futures;

            for(auto i=0; i<img_channel; ++i) {
                auto i_data = input_img+i *img_width *img_height;
                auto o_data = o_tensor+i *img_width *img_height;
                futures.emplace_back(async(launch::async,fn,(T*)i_data,o_data));
                if(futures.size()>8)
                    futures.pop_front();
            }
            futures.clear();
        }
};
REGISTER_KERNEL_BUILDER(Name("MaskRotate").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), MaskRotateOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("MaskRotate").Device(DEVICE_CPU).TypeConstraint<float>("T"), MaskRotateOp<CPUDevice, float>);

/*
 * mask [Nr,H,W] 中查找目标instance的bbox
 * bbox [N,4],[ymin,xmin,ymax,xmax], 绝对坐标
 */
REGISTER_OP("GetBboxesFromMask")
    .Attr("T: {uint8,float}")
    .Input("mask: T")
	.Output("bbox:float")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto input_shape0 = c->input(0);
            auto data_nr = c->Dim(input_shape0,0);
            auto output_shape = c->MakeShape({data_nr,4});
            c->set_output(0, output_shape);
			return Status::OK();
            });

template <typename Device, typename T>
class GetBboxesFromMaskOp: public OpKernel {
    private:
        using Tensor3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
        using type_to_int = bm::map<
              bm::pair<uint8_t,bm::int_<CV_8UC1>>
                  , bm::pair<float,bm::int_<CV_32FC1>>
             >;
	public:
		explicit GetBboxesFromMaskOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
            TIME_THISV1("GetBboxesFromMask");
			const Tensor &_input_img = context->input(0);

			OP_REQUIRES(context, _input_img.dims() == 3, errors::InvalidArgument("tensor must be a 3-dimensional tensor"));

            auto         input_img     = _input_img.template flat<T>().data();
            const auto   img_channel   = _input_img.dim_size(0);
            const auto   img_height    = _input_img.dim_size(1);
            const auto   img_width     = _input_img.dim_size(2);
            const int    dim2d[]       = {img_channel,4};
            TensorShape  output_shape;
            Tensor      *output_bbox   = nullptr;

            TensorShapeUtils::MakeShape(dim2d,2,&output_shape);

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_bbox));

            auto          o_bbox       = output_bbox->template flat<float>().data();
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            auto fn  = [img_height,img_width,cv_type](T* i_data,float* bbox) {
                cv::Mat i_img(img_height,img_width,cv_type,i_data);
                getBBox(i_img,bbox);
            };
            list<future<void>> futures;

            for(auto i=0; i<img_channel; ++i) {
                auto i_data = input_img+i *img_width *img_height;
                auto bbox   = o_bbox+i *4;
                futures.emplace_back(async(launch::async,fn,(T*)i_data,bbox));
                if(futures.size()>8)
                    futures.pop_front();
            }
            futures.clear();
        }

        static void getBBox(const cv::Mat& img,float* bbox)
        {
            vector<vector<cv::Point>> contours;
            vector<cv::Vec4i> hierarchy;
            vector<cv::Point> points;
            const auto    cv_type      = bm::at<type_to_int,T>::type::value;
            cv::Mat dst_img(img.rows,img.cols,cv_type);


            if(cv_type == CV_32FC1) {
                cv::threshold(img,dst_img,0.5,255,CV_THRESH_BINARY);
                cv::Mat dst_img1;
                dst_img.convertTo(dst_img1,CV_8UC1);
                cv::findContours(dst_img1, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
            } else {
                cv::threshold(img,dst_img,127,255,CV_THRESH_BINARY);
                cv::findContours(dst_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));
            }

            for (auto &cont:contours) 
                points.insert(points.end(),cont.begin(),cont.end());

            if(points.size()<2) {
               memset(bbox,0,sizeof(float)*4);
               return;
            }

            const auto rect = cv::boundingRect(points);

            bbox[0] = rect.y;
            bbox[1] = rect.x;
            bbox[2] = rect.y+rect.height;
            bbox[3] = rect.x+rect.width;
        }
};
REGISTER_KERNEL_BUILDER(Name("GetBboxesFromMask").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), GetBboxesFromMaskOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("GetBboxesFromMask").Device(DEVICE_CPU).TypeConstraint<float>("T"), GetBboxesFromMaskOp<CPUDevice, float>);
/*
 * 输入mask[H,W,N]
 * 输入labels[N]
 * set_background:如果一个位置没有标签，则默认为背景
 * attr:num_classes
 * 输出mask[W,H,num_classes]
 */
REGISTER_OP("SparseMaskToDense")
    .Attr("T: {int32,bool,int8,uint8}")
	.Attr("num_classes:int")
	.Attr("set_background:bool")
    .Input("mask: T")
    .Input("labels: int32")
	.Output("data:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
			int num_classes = 0;
			c->GetAttr("num_classes",&num_classes);
			auto w = c->Value(c->Dim(c->input(0),0));
			auto h = c->Value(c->Dim(c->input(0),1));
            auto output_shape = c->MakeShape({h, w, num_classes});
			c->set_output(0,output_shape);
			return Status::OK();
			});

template <typename Device, typename T>
class SparseMaskToDenseOp: public OpKernel {
	public:
		explicit SparseMaskToDenseOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("num_classes", &num_classes_));
			OP_REQUIRES_OK(context,
					context->GetAttr("set_background", &set_background_));
		}
		void Compute(OpKernelContext* context) override
        {
            const Tensor &_mask   = context->input(0);
            const Tensor &_labels = context->input(1);
            auto          mask    = _mask.template tensor<T,3>();
            auto          labels  = _labels.template flat<int>().data();
            auto          h       = _mask.dim_size(0);
            auto          w       = _mask.dim_size(1);
            auto          nr      = _mask.dim_size(2);
            auto          nr1     = _labels.dim_size(0);

            OP_REQUIRES(context, _labels.dims()==1, errors::InvalidArgument("labels must be 1-dimensional"));
            OP_REQUIRES(context, _mask.dims()==3, errors::InvalidArgument("mask must be 3-dimensional"));
            OP_REQUIRES(context, nr==nr1, errors::InvalidArgument("size unmatch."));

            int          dims3d[]     = {int(h),int(w),num_classes_};
            Tensor      *output_data  = NULL;
            TensorShape  output_shape;

            TensorShapeUtils::MakeShape(dims3d, 3, &output_shape);

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_data));

            auto      oq_tensor = output_data->template tensor<T,3>();
            oq_tensor.setZero();
			using tensor_t = Eigen::Tensor<T,2,Eigen::RowMajor>;

            for(auto i=0; i<nr; ++i) {
				const auto label = labels[i];
				if((label<0) || (label>=num_classes_)) {
					cout<<"Error label "<<label<<", not in range [0,"<<num_classes_<<")"<<endl;
					continue;
				}
                auto t = oq_tensor.chip(label,2);
                t = (t||mask.chip(i,2)).template cast<T>();
            }

			if(!set_background_) return;

			for(auto i=0; i<h; ++i) {
				for(auto j=0; j<w; ++j) {
					bool have_label = false;
					for(auto k=0; k<nr; ++k) {
						const auto label = labels[k];

						if((label<0) || (label>=num_classes_)) continue;

						if(mask(i,j,k) != 0) {
							have_label = true;
							break;
						}
					}
					if(have_label) continue;

					oq_tensor(i,j,0) = T(true);
				}
			}
        }
	private:
		int num_classes_ = 0;
		bool set_background_ = false;
};
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), SparseMaskToDenseOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<bool>("T"), SparseMaskToDenseOp<CPUDevice, bool>);
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), SparseMaskToDenseOp<CPUDevice, int8_t>);
REGISTER_KERNEL_BUILDER(Name("SparseMaskToDense").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), SparseMaskToDenseOp<CPUDevice, uint8_t>);

/*
 * mask_size: 输出的mask像素大小
 * masks:[Nr,h,w]
 * test_threshold: 距离小于test_threshold的bbox才进行是否可以合并的检测
 * kerner_size: 闭运算kernel大小
 * bboxes:[Nr,4] absolute coordinate
 * labels:[Nr]
 * mask:[Nr,H,W]
 * output:
 * out_bboxes: [out_Nr,4]
 * out_labels: [out_Nr]
 * out_masks: [out_Nr,mask_size,mask_size]
 * out_indices: [Nr] 相同的值表示相应位置的输入合并在了一起
 */
REGISTER_OP("MergeInstanceByMask")
    .Attr("T: {float32,uint8}")
    .Attr("mask_size:int=63")
    .Attr("test_threshold:int=8")
    .Attr("kernel_size:int=3")
    .Input("bboxes: T")
    .Input("labels: int32")
    .Input("probability: T")
    .Input("mask: uint8")
	.Output("out_bboxes:T")
	.Output("out_labels:int32")
	.Output("out_probability:T")
	.Output("out_masks:uint8")
	.Output("out_indices:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            int mask_size = 0;
            c->GetAttr("mask_size",&mask_size);
            const auto nr = c->Value(c->Dim(c->input(0),0));
            const auto shape0     = c->MakeShape({-1,4});
            const auto shape1     = c->MakeShape({-1});
            const auto shape2     = c->MakeShape({nr});
            const auto shape3     = c->MakeShape({-1,mask_size,mask_size});
			c->set_output(0, shape0);
			c->set_output(1, shape1);
			c->set_output(2, shape1);
			c->set_output(3, shape3);
			c->set_output(4, shape2);
			return Status::OK();
			});

template <typename Device,typename T>
class MergeInstanceByMaskOp: public OpKernel {
    private:
        using bbox_t = Eigen::Tensor<T,1,Eigen::RowMajor>;
        using mask_t = Eigen::Tensor<uint8_t,2,Eigen::RowMajor>;
    public:
        explicit MergeInstanceByMaskOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("mask_size", &mask_size_));
            OP_REQUIRES_OK(context, context->GetAttr("test_threshold", &test_threshold_));
            OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kerner_size_));
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &_bboxes = context->input(0);
            const Tensor &_labels= context->input(1);
            const Tensor &_probability = context->input(2);
            const Tensor &_mask= context->input(3);

            OP_REQUIRES(context, _mask.dims() == 3, errors::InvalidArgument("mask data must be 3-dimensional"));
            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes data must be 2-dimensional"));
            OP_REQUIRES(context, _labels.dims() == 1, errors::InvalidArgument("labels data must be 1-dimensional"));
            OP_REQUIRES(context, _probability.dims() == 1, errors::InvalidArgument("probability data must be 1-dimensional"));

            auto       mask        = _mask.template tensor<uint8_t,3>();
            auto       bboxes      = _bboxes.template tensor<T,2>();
            auto       labels      = _labels.template tensor<int,1>();
            auto       probability = _probability.template tensor<T,1>();
            const auto data_nr     = _mask.dim_size(0);

            bool is_merged = false;
            vector<bbox_t> in_bboxes_data;
            vector<int> in_labels_data;
            vector<mask_t> in_mask_data;
            vector<int> in_indices_data;
            vector<T> in_probability;
            vector<bool> need_remove(data_nr,false);

            for(auto i=0; i<data_nr; ++i) {
                in_bboxes_data.push_back(bboxes.chip(i,0));
                in_labels_data.push_back(labels(i));
                in_mask_data.push_back(mask.chip(i,0));
                in_indices_data.push_back(i);
                in_probability.push_back(probability(i));
            }

            do {
                is_merged = false;
                if(in_bboxes_data.size()>1) {
                    for(auto i=0; i<in_bboxes_data.size()-1; ++i) {
                        if(need_remove[i])
                            continue;
                        auto& bbox0 = in_bboxes_data[i];
                        for(auto j=i+1; j<in_bboxes_data.size(); ++j) {
                            if(need_remove[j])
                                continue;
                            if(in_labels_data[i] != in_labels_data[j]) 
                                continue;

                            auto &bbox1 = in_bboxes_data[j];
                            auto  dis   = bboxes_distance(in_bboxes_data[i],in_bboxes_data[j]);

                            if(dis > test_threshold_)
                                continue;

                            try {
                                auto res = merge_bboxes({bbox0,bbox1},{in_mask_data[i],in_mask_data[j]});

                                in_mask_data[i] = get<1>(res);
                                in_bboxes_data[i] = get<0>(res);
                                in_indices_data[j] = in_indices_data[i];
                                in_probability[i] = max(in_probability[i],in_probability[j]);
                                need_remove[j] = true;
                                is_merged = true;
                            }catch(...) {
                            }
                        }
                    }
                }
            }while(is_merged);

            auto         total_nr           = count(need_remove.begin(),need_remove.end(),false);
            int          dims_1d[1]         = {total_nr};
            int          dims_2d[2]         = {total_nr,4};
            int          dims_1d2[1]        = {data_nr};
            int          dims_3d[3]         = {total_nr,mask_size_,mask_size_};
            TensorShape  outshape0;
            TensorShape  outshape1;
            TensorShape  outshape2;
            TensorShape  outshape3;
            Tensor      *output_bboxes      = NULL;
            Tensor      *output_labels      = NULL;
            Tensor      *output_probability = NULL;
            Tensor      *output_mask        = NULL;
            Tensor      *output_indices     = NULL;

            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape0);
            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape1);
            TensorShapeUtils::MakeShape(dims_1d2, 1, &outshape2);
            TensorShapeUtils::MakeShape(dims_3d, 3, &outshape3);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape1, &output_bboxes));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape0, &output_labels));
            OP_REQUIRES_OK(context, context->allocate_output(2, outshape0, &output_probability));
            OP_REQUIRES_OK(context, context->allocate_output(3, outshape3, &output_mask));
            OP_REQUIRES_OK(context, context->allocate_output(4, outshape2, &output_indices));

            auto o_mask        = output_mask->template tensor<uint8_t,3>();
            auto o_bboxes      = output_bboxes->template tensor<T,2>();
            auto o_labels      = output_labels->template tensor<int,1>();
            auto o_probability = output_probability->template tensor<T,1>();
            auto o_indices     = output_indices->template tensor<int,1>();

            for(auto i=0,j=0; i<data_nr; ++i) {
                o_indices(i) = in_indices_data[i];
                if(need_remove[i])
                    continue;
                o_bboxes.chip(j,0) = in_bboxes_data[i];
                o_labels(j) = in_labels_data[i];
                o_probability(j) = in_probability[i];
                if(in_mask_data[i].dimension(0) == mask_size_) {
                    o_mask.chip(j,0) = in_mask_data[i];
                } else {
                    o_mask.chip(j,0) = resize_mask(in_mask_data[i],mask_size_);
                }
                ++j;
            }
        }
        mask_t resize_mask(const mask_t& msk,size_t size) {
            const cv::Mat input_mask(msk.dimension(0),msk.dimension(1),CV_8UC1,(void*)(msk.data()));
            cv::Mat dst_mask(size,size,CV_8UC1);
            mask_t res_mask(size,size);

            cv::resize(input_mask,dst_mask,cv::Size(size,size),0,0,CV_INTER_LINEAR);
            memcpy(res_mask.data(),dst_mask.data,size*size);

            return res_mask;
        }
        static vector<bbox_t> make_around_bboxes(const bbox_t& box0,const bbox_t& box1) {

            bbox_t res_box0(4);
            bbox_t res_box1(4);
            bbox_t res_box2(4);
            bbox_t res_box3(4);

            res_box0(0) = box0(0)-(box1(2)-box1(0));
            res_box0(1) = box0(1);
            res_box0(2) = box0(0);
            res_box0(3) = box0(3);

            res_box1(0) = box0(0);
            res_box1(1) = box0(3);
            res_box1(2) = box0(2);
            res_box1(3) = box0(3)+(box1(3)-box1(1));

            res_box2(0) = box0(2);
            res_box2(1) = box0(1);
            res_box2(2) = box0(2)+(box1(2)-box1(0));;
            res_box2(3) = box0(3);

            res_box3(0) = box0(0);
            res_box3(1) = box0(1)-(box1(3)-box1(1));
            res_box3(2) = box0(2);
            res_box3(3) = box0(1);
            return {box0,res_box0,res_box1,res_box2,res_box3};
            
        }
        float bboxes_distance(const bbox_t& box0,const bbox_t& box1) {
            auto  bboxes  = make_around_bboxes(box0,box1);
            int   index   = -1;
            float max_iou = -1.0;

            for(auto i=0; i<bboxes.size(); ++i) {
                auto v0 = bboxes_jaccard_of_box0v1(bboxes[i],box1);
                auto v1 = bboxes_jaccard_of_box0v1(box1,bboxes[i]);
                auto v = max(v0,v1);

                if(v>max_iou) {
                    max_iou = v;
                    index = i;
                }
            }
            if(max_iou<0.4)
                return 1e8;
            switch(index) {
                case 0:
                    return 0;
                case 1:
                    return fabs(box1(2)-box0(0));
                case 2:
                    return fabs(box1(1)-box0(3));
                case 3:
                    return fabs(box1(0)-box0(2));
                case 4:
                    return fabs(box1(3)-box0(1));
                default:
                    cout<<"Error type."<<endl;
                    return 1e8;
            }
        }
        void show_box(const bbox_t& box) {
            cout<<box(0)<<","<<box(1)<<","<<box(2)<<","<<box(3)<<endl;
        }
        static cv::Mat to_mat(const mask_t& msk) 
        {
            cv::Mat res(msk.dimension(0),msk.dimension(1),CV_8UC1,(void*)msk.data());
            return res.clone();
        }
        tuple<bbox_t,mask_t> merge_bboxes(const vector<bbox_t>& boxes,const vector<mask_t>& masks) noexcept(false)
        {
            bbox_t  env_bbox(4);
            cv::Mat res_mask = cv::Mat::zeros(mask_size_,mask_size_,CV_8UC1);
            cv::Mat res_mask1 = cv::Mat::zeros(mask_size_,mask_size_,CV_8UC1);
            int total_nr = 0;

            bboxes_envelope(boxes[0],boxes[1],env_bbox);

            const auto env_bbox_w = env_bbox(3)-env_bbox(1);
            const auto env_bbox_h = env_bbox(2)-env_bbox(0);

            for(auto i=0; i<boxes.size(); ++i) {
                vector<vector<cv::Point>> contours;
                vector<vector<cv::Point>> new_contours;
                vector<cv::Vec4i> hierarchy;
                vector<cv::Point> points;
                cv::Mat dst_img = to_mat(masks[i]);
                const auto cur_bbox = boxes[i];
                const auto cur_mask_h = masks[i].dimension(0);
                const auto cur_mask_w = masks[i].dimension(1);
                const auto cur_bbox_w = cur_bbox(3)-cur_bbox(1);
                const auto cur_bbox_h = cur_bbox(2)-cur_bbox(0);

                cv::findContours(dst_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));

                if(contours.size()==0) {
                    cout<<"Error contours size."<<endl;
                    throw std::runtime_error("error contours size.");
                }

                for(auto& points:contours) {
                    vector<cv::Point> new_points;
                    for(auto& p:points) {
                        auto x = (p.x*cur_bbox_w/cur_mask_w+cur_bbox(1)-env_bbox(1))*mask_size_/env_bbox_w;
                        auto y = (p.y*cur_bbox_h/cur_mask_h+cur_bbox(0)-env_bbox(0))*mask_size_/env_bbox_h;
                        new_points.push_back(cv::Point(x,y));
                    }
                    new_contours.push_back(new_points);
                }
                cv::drawContours(res_mask,new_contours,-1,cv::Scalar(1),CV_FILLED,8,hierarchy);

                total_nr += contours.size();
            }

            vector<vector<cv::Point>> contours;
            vector<vector<cv::Point>> new_contours;
            vector<cv::Vec4i> hierarchy;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kerner_size_, kerner_size_), cv::Point(-1, -1));

            cv::morphologyEx(res_mask, res_mask1, CV_MOP_CLOSE, kernel);
            cv::findContours(res_mask1.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0,0));

            if((contours.size()==1)
                    || (contours.size()<total_nr-2)) {
                mask_t r_mask(mask_size_,mask_size_);
                memcpy(r_mask.data(),res_mask1.data,mask_size_*mask_size_);
                return make_tuple(env_bbox,r_mask);
            }
            throw std::runtime_error("merge faild.");
        }
    private:
        int mask_size_      = 63;
        int test_threshold_ = 8;
        int kerner_size_    = 3;
};
REGISTER_KERNEL_BUILDER(Name("MergeInstanceByMask").Device(DEVICE_CPU).TypeConstraint<float>("T"), MergeInstanceByMaskOp<CPUDevice,float>);
