#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <future>
#include <boost/algorithm/clamp.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
REGISTER_OP("DrawPoints")
    .Attr("T: {float, double}")
	.Attr("color:list(float)")
	.Attr("point_size:int")
    .Input("image: T")
    .Input("points: T")
	.Output("output:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class DrawPointsOp: public OpKernel {
	public:
		explicit DrawPointsOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("color", &color_));
			OP_REQUIRES_OK(context, context->GetAttr("point_size", &point_size_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_image      = context->input(0);
			const Tensor &_points     = context->input(1);
			auto          image_flat  = _image.flat<T>();
			auto          points_flat = _points.flat<T>();
			const auto    points_nr   = _points.dim_size(0);
			const auto    width       = _image.dim_size(1);
			const auto    height      = _image.dim_size(0);
			const auto    channels    = _image.dim_size(2);

			OP_REQUIRES(context, _image.dims() == 3, errors::InvalidArgument("images data must be 3-dimensional"));
			OP_REQUIRES(context, _points.dims() == 2, errors::InvalidArgument("points data must be 2-dimensional"));
			OP_REQUIRES(context, color_.size() > 0, errors::InvalidArgument("empty color"));

			TensorShape  output_shape  = _image.shape();
			Tensor      *output_tensor = nullptr;

			OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

			if(!output_tensor->CopyFrom(_image,output_shape))
				return;

			auto       image  = output_tensor->tensor<T,3>();      
			const auto points = _points.tensor<T,2>();

			while(color_.size()<channels)
				color_.push_back(color_.back());

			auto shard = [&]
				(int64 start, int64 limit) {
					for(auto i=start; i<limit; ++i) {

						const auto x     = points(i,1);
						const auto y     = points(i,0);
						const auto beg_x = std::max<int>(x-point_size_,0);
						const auto end_x = std::min<int>(x+point_size_,width-1);
						const auto beg_y = std::max<int>(y-point_size_,0);
						const auto end_y = std::min<int>(y+point_size_,height-1);

						for(auto j=beg_x; j<=end_x; ++j) {
							for(auto k=beg_y; k<=end_y; ++k) {
								for(auto m=0; m<channels; ++m) {
									image(k,j,m) = color_.at(m);
								}
							}
						}
					}
				};
			shard(0,points_nr);
			/*const DeviceBase::CpuWorkerThreads& worker_threads =
			*(context->device()->tensorflow_cpu_worker_threads());
			const int64 total         = points_nr;
			const int64 cost_per_unit = 2;*/
			//Shard(worker_threads.num_threads, worker_threads.workers,total,cost_per_unit, shard);
		}
	private:
		std::vector<float> color_;
		int point_size_;
};
REGISTER_KERNEL_BUILDER(Name("DrawPoints").Device(DEVICE_CPU).TypeConstraint<float>("T"), DrawPointsOp<CPUDevice, float>);
/*
 * phy_max:返回的begin_index与end_index之间最多差phy_max
 * max：begin_index,end_index的最大值，
 * hint:提示值，生成的区间至少要包含hint中的一个值
 * 输出:
 * [begin_index,end_index)
 * hint:输入的hint中在[begin_index,end_index之间的部分
 */
REGISTER_OP("RandomRange")
	.Attr("phy_max:int")
    .Input("max: int32")
    .Input("hint: int32")
	.Output("oindex:int32")
	.Output("ohint:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        auto shape0 = c->Vector(2);
        auto shape1 = c->Vector(c->UnknownDim());
        c->set_output(0,shape0);
        c->set_output(1,shape1);
		return Status::OK();
    });

class RandomRange: public OpKernel {
	public:
		explicit RandomRange(OpKernelConstruction* context) : OpKernel(context) {
            std::srand(::time(nullptr));
			OP_REQUIRES_OK(context, context->GetAttr("phy_max", &phy_max_));
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_max      = context->input(0);
            const Tensor &_hint     = context->input(1);
            auto          max       = _max.flat<int>().data()[0];
            auto          hint_flat = _hint.flat<int>();
            const auto    hint_size = _hint.dim_size(0);
            int           index     = 0;
            int           beg_index = 0;
            int           end_index = max;

            OP_REQUIRES(context, _max.dims() <=1, errors::InvalidArgument("max data must be 1/0-dimensional"));
            OP_REQUIRES(context, _hint.dims() == 1, errors::InvalidArgument("hint data must be 1-dimensional"));

            TensorShape  output_shape0;
            Tensor      *output_tensor0 = nullptr;
            Tensor      *output_tensor1 = nullptr;
            const int    dim0[]           = {2};

            TensorShapeUtils::MakeShape(dim0,1,&output_shape0);

            OP_REQUIRES_OK(context,context->allocate_output(0,output_shape0,&output_tensor0));

            auto oindex = output_tensor0->flat<int>();

            if(max> phy_max_) {
                const auto index = std::rand()%hint_size;
                const auto base_index = hint_flat.data()[index];
                vector<int> outdata;

                beg_index = base_index-(phy_max_/2);
                end_index = beg_index+phy_max_;
                if(beg_index<0) {
                    beg_index = 0;
                    end_index = phy_max_;
                } else if (end_index>=max) {
                    end_index = max;
                    beg_index = max-phy_max_;
                }
                std::copy_if(hint_flat.data(),hint_flat.data()+hint_size,std::back_inserter(outdata),[beg_index,end_index](int v){ return (v>=beg_index)&& (v<end_index); });

                TensorShape output_shape1;
                const int   dim1[]        = {int(outdata.size())};
                TensorShapeUtils::MakeShape(dim1,1,&output_shape1);

                OP_REQUIRES_OK(context,context->allocate_output(1,output_shape1,&output_tensor1));
                auto ohint = output_tensor1->flat<int>();
                std::copy(outdata.begin(),outdata.end(),ohint.data());

            } else {
                TensorShape  output_shape1  = _hint.shape();
                OP_REQUIRES_OK(context,context->allocate_output(1,output_shape1,&output_tensor1));

                output_tensor1->CopyFrom(_hint,output_shape1);
            }
            oindex.data()[0] = beg_index;
            oindex.data()[1] = end_index;
        }
	private:
		int phy_max_;
};
REGISTER_KERNEL_BUILDER(Name("RandomRange").Device(DEVICE_CPU), RandomRange);

/*
 * 将输入的整数按指定的方式进行映射
 */
REGISTER_OP("IntHash")
	.Attr("T:{int32,int64}")
	.Attr("key:list(int)")
	.Attr("value:list(int)")
    .Input("input:T")
	.Output("output:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class IntHash: public OpKernel {
	public:
		explicit IntHash(OpKernelConstruction* context) : OpKernel(context) {
            vector<int> key;
            vector<int> value;
			OP_REQUIRES_OK(context, context->GetAttr("key", &key));
			OP_REQUIRES_OK(context, context->GetAttr("value", &value));
            const auto nr = std::min(key.size(),value.size());
            for(auto i=0; i<nr; ++i)
                dict_[key[i]] = value[i];
		}

		void Compute(OpKernelContext* context) override
        {
            const Tensor &_input = context->input(0);
            Tensor      *output_tensor0 = nullptr;

            OP_REQUIRES_OK(context,context->allocate_output(0,_input.shape(),&output_tensor0));

            auto input  = _input.flat<T>();
            auto output = output_tensor0->flat<T>();

            for(auto i=0; i<input.size(); ++i) {
                const auto v = input.data()[i];
                const auto it = dict_.find(v);
                if(it != dict_.end()) 
                    output.data()[i] = it->second;
                else
                    output.data()[i] = 65536;
            }
        }
	private:
		map<int,int> dict_;
};
REGISTER_KERNEL_BUILDER(Name("IntHash").Device(DEVICE_CPU).TypeConstraint<int>("T"), IntHash<CPUDevice, int>);
/*
 * 对Boxes的概率进行调整
 * 具体方法为：
 * 1，如果最大概率不在指定的类型中则不调整
 * 2，否则将指定的类型中非最大概率的一半值分配给最大概率
 * classes:指定需要调整的类别,如果为空则表示使用所有的非背景类别
 * probs:概率，[X,N]
 */
REGISTER_OP("ProbabilityAdjust")
    .Attr("T: {float, double}")
	.Attr("classes:list(int)")
    .Input("probs: T")
	.Output("output_probs:T")
	.SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class ProbabilityAdjustOp: public OpKernel {
	public:
		explicit ProbabilityAdjustOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("classes", &classes_));
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &probs = context->input(0);
			auto          probs_flat = probs.flat<T>();
			const auto    nr         = probs.dim_size(0);
			const auto    classes_nr = probs.dim_size(1);

			OP_REQUIRES(context, probs.dims() == 2, errors::InvalidArgument("probs must be 2-dimensional"));

			TensorShape output_shape = probs.shape();

			if(classes_.empty()) {
				for(auto i=1; i<classes_nr; ++i) {
					classes_.push_back(i);
				}
			}
			auto it = remove_if(classes_.begin(),classes_.end(),[classes_nr](int i){ return (i<0) || (i>=classes_nr);});

			classes_.erase(it,classes_.end());

			Tensor *output_probs = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_probs));

			output_probs->CopyFrom(probs,output_shape);

			auto output = output_probs->template flat<T>();

			for(int i=0; i<nr; ++i) {
				auto       v     = output.data()+i*classes_nr;
				const auto it    = max_element(v,v+classes_nr);
				const int  index = distance(v,it);
				auto       jt    = find(classes_.begin(),classes_.end(),index);
				auto       sum   = 0.;

                if(jt ==classes_.end()) continue;

                for(auto k:classes_) {
                    if(k==index)continue;
                    sum += v[k]/2.;
                    v[k] = v[k]/2.;
                }
                v[index] = v[index]+sum;
			}
		}
	private:
		vector<int> classes_;
};
REGISTER_KERNEL_BUILDER(Name("ProbabilityAdjust").Device(DEVICE_CPU).TypeConstraint<float>("T"), ProbabilityAdjustOp<CPUDevice, float>);

/*
 * labels:[batch_size,nr]
 * ids:[batch_size,nr]
 * output:[batch_size,sample_nr,3]
 */
REGISTER_OP("SampleLabels")
    .Attr("T: {int32, int64}")
	.Attr("sample_nr:int")
    .Input("labels: T")
    .Input("ids: T")
	.Output("data:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            int sample_nr = 0;
            const auto input_shape0 = c->input(0);
            const auto batch_size = c->Dim(input_shape0,0);
            c->GetAttr("sample_nr",&sample_nr);
            auto shape0 = c->MakeShape({batch_size,sample_nr,3});
			c->set_output(0, shape0);
			return Status::OK();
			});

template <typename Device, typename T>
class SampleLabelsOp: public OpKernel {
    public:
        explicit SampleLabelsOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("sample_nr", &sample_nr_));
        }
        void Compute(OpKernelContext* context) override
        {
            const Tensor &_labels     =  context->input(0);
            const Tensor &_ids        =  context->input(1);
            auto          labels      =  _labels.tensor<T,2>();
            auto          ids         =  _ids.tensor<T,2>();
            auto          batch_size  =  labels.dimension(0);

            OP_REQUIRES(context, _labels.dims() == 2, errors::InvalidArgument("labels must be 2-dimensional"));
            OP_REQUIRES(context, _ids.dims() == 2, errors::InvalidArgument("ids must be 2-dimensional"));


            int dims_3d[] = {batch_size,sample_nr_,3};
            TensorShape outshape0;

            TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);
            Tensor *output_data = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_data));
            auto out_tensor = output_data->tensor<T,3>();
            list<future<vector<tuple<T,T,T>>>> res;
            for(auto i=0; i<batch_size; ++i) {
                res.emplace_back(async(launch::async,&SampleLabelsOp<Device,T>::sample_one_batch,Eigen::Tensor<T,1,Eigen::RowMajor>(ids.chip(i,0)),
                Eigen::Tensor<T,1,Eigen::RowMajor>(labels.chip(i,0)),
                sample_nr_));
            }
            for(auto i=0; i<batch_size; ++i) {
                auto data = next(res.begin(),i)->get();
                for(auto j=0; j<sample_nr_; ++j) {
                    out_tensor(i,j,0) = std::get<0>(data[j]);
                    out_tensor(i,j,1) = std::get<1>(data[j]);
                    out_tensor(i,j,2) = std::get<2>(data[j]);
                }
            }
        }
        static vector<tuple<T,T,T>> sample_one_batch(const Eigen::Tensor<T,1,Eigen::RowMajor>& ids,const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,int sample_nr) {
            map<T,vector<int>> datas;
            map<T,T> id_to_label;
            const auto data_nr = ids.dimension(0);
            for(auto i=0; i<ids.dimension(0); ++i) {
                const auto id = ids(i);
                if(id == 0) continue;
                auto it = datas.find(id);
                if(it != datas.end()) {
                    it->second.push_back(i);
                } else {
                    datas[id] = vector<int>({i});
                    id_to_label[id] = labels[i];
                }
            }
            if(datas.size() == 0) {
                vector<tuple<T,T,T>> res(sample_nr,make_tuple(0,0,1));
                return res;
            } else if(datas.size() ==1){
                vector<tuple<T,T,T>> res(sample_nr);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, datas.begin()->second.size()-1);
                std::uniform_int_distribution<> dis1(0, data_nr-1);
                const auto& data = datas.begin()->second;
                auto o_r = minmax_element(data.begin(),data.end());
                auto o_v = 0;
                if((*o_r.first)>0)
                    o_v = 0;
                else if(*o_r.second+1<data_nr)
                    o_v=*o_r.second+1;
                else
                    o_v = dis1(gen);
                generate(res.begin(),res.end(),[&gen,&dis,o_v,&data](){
                        int index0;
                        int index1;
                        std::tie(index0,index1) = sample_two_int(dis.b(),[&dis,&gen]{ return dis(gen);});
                        const auto v0 = data[index0];
                        const auto v1 = data[index1];
                        return make_tuple(v0,v1,o_v);
                        });
                return res;
            } else {
                vector<tuple<T,T,T>> res(sample_nr);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, datas.size()-1);
                std::uniform_int_distribution<> dis1;
                generate(res.begin(),res.end(),[&gen,&dis,&dis1,&datas,&id_to_label](){
                        int        index0  = dis(gen);
                        int        index1;
                        int        index00;
                        int        index01;
                        const auto id0     = next(datas.begin(),index0)->first;
                        const auto label0  = id_to_label[id0];
                        vector<int> id1s;

                        for(auto& item:id_to_label) {
                            if((item.first != id0) && (item.second == label0)) {
                                id1s.push_back(item.first);
                            }
                        }
                        if(!id1s.empty()) {
                            const auto id1 = id1s[dis1(gen)%id1s.size()];
                            index1 = distance(datas.begin(),datas.find(id1));
                        } else {
                            index1 = dis(gen);
                            if(index1 == index0) {
                                if(index1>0)
                                    index1 = 0;
                                else
                                    index1 = 1;
                            }
                        }

                        const auto& data0 = next(datas.begin(),index0)->second;
                        const auto& data1 = next(datas.begin(),index1)->second;

                        std::tie(index00,index01) = sample_two_int(data0.size()-1,[&dis1,&gen,&data0]{ return dis1(gen)%data0.size();});

                        const auto index10 = dis1(gen)%data1.size();
                        const auto v0 = data0[index00];
                        const auto v1 = data0[index01];
                        const auto v2 = data1[index10];
                        return make_tuple(v0,v1,v2);
                        });
                return res;
            }
        }
        static pair<int,int> sample_two_int(int max_val,auto func) {
            const int v0 = func();
            int v1 = func();
            if((0 == max_val) || (v0 != v1))
                return make_pair(v0,v1);
            if(v1>0)
                return make_pair(v0,v1-1);
            else
                return make_pair(v0,1);
        };
    private:
        int sample_nr_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("SampleLabels").Device(DEVICE_CPU).TypeConstraint<int>("T"), SampleLabelsOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("SampleLabels").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), SampleLabelsOp<CPUDevice, tensorflow::int64>);

/*
 * data:[batch_size,nr,X]
 * labels:[batch_size,nr]
 * bboxes:[batch_size,nr,4]
 * lens:[batch_size]
 * threshold:
 * dis_threshold:[2](x,y)
 * output:[batch_size,nr]
 */
REGISTER_OP("MergeLineBoxes")
    .Attr("T: {int32, int64}")
	.Attr("threshold:float")
	.Attr("dis_threshold:list(float)")
    .Input("data: float")
    .Input("labels: T")
    .Input("bboxes:float")
    .Input("lens:T")
	.Output("ids:T")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            const auto input_shape0 = c->input(1);
			c->set_output(0, input_shape0);
			return Status::OK();
			});

template <typename Device, typename T>
class MergeLineBoxesOp: public OpKernel {
    public:
        explicit MergeLineBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("dis_threshold", &dis_threshold_));
            OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
            OP_REQUIRES(context, dis_threshold_.size() == 2, errors::InvalidArgument("Threshold must be contains two elements."));
        }
        void Compute(OpKernelContext* context) override
        {
            const Tensor &_data      = context->input(0);
            const Tensor &_labels    = context->input(1);
            const Tensor &_bboxes    = context->input(2);
            const Tensor &_lens      = context->input(3);
            auto          data       = _data.tensor<float,3>();
            auto          bboxes     = _bboxes.tensor<float,3>();
            auto          labels     = _labels.tensor<T,2>();
            auto          lens       = _lens.tensor<T,1>();
            auto          batch_size = labels.dimension(0);

            OP_REQUIRES(context, _labels.dims() == 2, errors::InvalidArgument("labels must be 2-dimensional"));
            OP_REQUIRES(context, _data.dims() == 3, errors::InvalidArgument("data must be 3-dimensional"));
            OP_REQUIRES(context, _bboxes.dims() == 3, errors::InvalidArgument("bboxes must be 3-dimensional"));
            OP_REQUIRES(context, _lens.dims() == 1, errors::InvalidArgument("lens must be 1-dimensional"));


            const auto data_nr = labels.dimension(1);
            int dims_2d[] = {batch_size,data_nr};
            TensorShape outshape0;

            TensorShapeUtils::MakeShape(dims_2d, 2, &outshape0);
            Tensor *output_data = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_data));
            auto out_tensor = output_data->tensor<T,2>();
            list<future<vector<int>>> res;
            for(auto i=0; i<batch_size; ++i) {
                res.emplace_back(async(launch::async,[this,i,&data,&labels,&bboxes,&lens]{
                    return process_one_batch(data.chip(i,0),labels.chip(i,0),bboxes.chip(i,0),lens(i));
                    }));
            }
            out_tensor.setConstant(0);
            for(auto i=0; i<batch_size; ++i) {
                auto data = next(res.begin(),i)->get();
                cout<<"LEN:"<<lens(i)<<endl;
                for(auto j=0; j<lens(i); ++j) {
                    out_tensor(i,j) = data[j];
                    cout<<j<<":"<<data[j]<<endl;
                }
            }
        }
        static auto get_distance(const Eigen::Tensor<float,1,Eigen::RowMajor>& box0,
        const Eigen::Tensor<float,1,Eigen::RowMajor>& box1
        ) {
            float xdis;
            const float ydis = fabs(box0(0)+box0(2)-box1(0)-box1(2))/2.0f;

            if(box0(1)>=box1(3)) {
                xdis = box0(1)-box1(3);
            } else if(box0(3)<=box1(1)) {
                xdis = box1(1)-box0(3);
            } else {
                xdis = 0.0f;
            }
            return make_pair(xdis,ydis);
        }
        static auto get_distance_matrix(const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& bboxes,int data_nr) {
            const auto kMaxDis = 1e8;
            Eigen::Tensor<float,3,Eigen::RowMajor> dis(data_nr,data_nr,2); //dis(x,y)
            dis.setConstant(kMaxDis);
            for(auto i=0; i<data_nr; ++i) {
                dis(i,i,0) = 0;
                dis(i,i,1) = 0;
                for(auto j=i+1; j<data_nr; ++j) {
                    if(labels(i) != labels(j)) continue;
                    const auto b_dis = get_distance(bboxes.chip(i,0),bboxes.chip(j,0));
                    dis(i,j,0) = b_dis.first;
                    dis(i,j,1) = b_dis.second;
                    dis(j,i,0) = b_dis.first;
                    dis(j,i,1) = b_dis.second;
                }
            }
            return dis;
        }
        static float feature_map_distance(const Eigen::Tensor<float,1,Eigen::RowMajor>& data0, const Eigen::Tensor<float,1,Eigen::RowMajor>& data1) {
            const Eigen::Tensor<float,0,Eigen::RowMajor> dis = (data0-data1).square().mean();
            //return  1.0-2.0/(1+exp(dis(0)));
            auto res = 1.0-2.0/(1+exp(dis(0)));
            cout<<"FMD:"<<dis(0)<<","<<res<<endl;
            return res;
        }
        template<typename DT>
        void label_one(const DT& dis_matrix,
        int index,
        const Eigen::Tensor<float,1,Eigen::RowMajor>& data_index,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& data,
        vector<int>& ids,
        int data_nr) {
            for(auto j=0; j<data_nr; ++j) {
                if(ids[j]>0) continue;
                if((dis_matrix(index,j,0) < dis_threshold_[0])
                        &&(dis_matrix(index,j,1) < dis_threshold_[1])
                        && (feature_map_distance(data_index,data.chip(j,0))<threshold_)){
                    ids[j] = ids[index];
                    label_one(dis_matrix,j,data.chip(j,0),data,ids,data_nr);
                }
            }
        }
        vector<int> process_one_batch(const Eigen::Tensor<float,2,Eigen::RowMajor>& data,
        const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& bboxes,int data_nr) {
            const auto dis_matrix = get_distance_matrix(labels,bboxes,data_nr);
            vector<int> ids(data_nr,0);
            int id = 0;

            for(auto i=0; i<data_nr; ++i) {
                if(ids[i] == 0) {
                    ids[i] = ++id;
                }
                const Eigen::Tensor<float,1,Eigen::RowMajor> data_i = data.chip(i,0);
                label_one(dis_matrix,i,data_i,data,ids,data_nr);
            }
            return ids;
        }
    private:
        vector<float> dis_threshold_;
        float threshold_ = 0.0f;
};
REGISTER_KERNEL_BUILDER(Name("MergeLineBoxes").Device(DEVICE_CPU).TypeConstraint<int>("T"), MergeLineBoxesOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("MergeLineBoxes").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), MergeLineBoxesOp<CPUDevice, tensorflow::int64>);

/*
 * limit:[min_size,max_size], satisfy max(out_size)<=max_size,min(out_size)>=min_size, if min_size/max_size is -1 or 1, means no limit
 * if both min_size and max_size return the input size
 * align:satisfy out_size[0]%align[0] == 0 and out_size[1]%align[1] == 0
 * Try to keep the ratio constant
 */
REGISTER_OP("GetImageResizeSize")
    .Input("size: int32")
	.Input("limit:int32") 
	.Input("align:int32")
	.Output("output_size:int32")
	.SetShapeFn(shape_inference::UnchangedShape);

class GetImageResizeSizeOp: public OpKernel {
        public:
		explicit GetImageResizeSizeOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_size = context->input(0);
			const Tensor &_limit= context->input(1);
			const Tensor &_align= context->input(2);

			OP_REQUIRES(context, _size.dims() == 1, errors::InvalidArgument("size must be 1-dimensional"));
			OP_REQUIRES(context, _limit.dims() == 1, errors::InvalidArgument("limit must be 1-dimensional"));
			OP_REQUIRES(context, _align.dims() == 1, errors::InvalidArgument("align must be 1-dimensional"));

			auto          size= _size.tensor<int,1>();
            auto          limit = _limit.flat<int>().data();
            auto          align = _align.flat<int>().data();
            int           out_size[2];
            auto scale = 1.0;
            if((limit[0]<1) && (limit[1]<1)) {
                out_size[0] = size(0);
                out_size[1] = size(1);
            } else if((limit[0]>0) && (limit[1]>0)) {
                if(size(0)<size(1))
                    scale = std::min(float(limit[0])/size(0),float(limit[1])/size(1));
                else
                    scale = std::min(float(limit[0])/size(1),float(limit[1])/size(0));
            } else if(limit[1]<1) {
                if(size(0)<size(1))
                    scale = float(limit[0])/size(0);
                else
                    scale = float(limit[0])/size(1);
            } else if(limit[0]<1) {
                if(size(0)<size(1))
                    scale = float(limit[1])/size(1);
                else
                    scale = float(limit[1])/size(0);
            }
            out_size[0] = size(0)*scale+0.5;
            out_size[1] = size(1)*scale+0.5;
            if(limit[0]>0) {
                if(out_size[0]<limit[0]) 
                    out_size[0] = limit[0];
                else if(out_size[1]<limit[0])
                    out_size[1] = limit[0];
            }

            if(align[1]>1)
                out_size[1] = ((out_size[1]+align[1]-1)/align[1])*align[1];
            if(align[0]>1)
                out_size[0] = ((out_size[0]+align[0]-1)/align[0])*align[0];

            TensorShape  outshape0;
            Tensor      *output_size = nullptr;
            int          dims_1d0[1]  = {2};
            TensorShapeUtils::MakeShape(dims_1d0, 1, &outshape0);
            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_size));
			auto       output_tensor = output_size->tensor<int,1>();      
            output_tensor(0) = out_size[0];
            output_tensor(1) = out_size[1];
        }
};
REGISTER_KERNEL_BUILDER(Name("GetImageResizeSize").Device(DEVICE_CPU), GetImageResizeSizeOp);
