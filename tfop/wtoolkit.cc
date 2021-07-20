#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <future>
#include <assert.h>
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
 * phy_max:返回的begin_index与end_index之间最多差phy_max(强制限制)
 * max：begin_index,end_index的最大值，
 * hint:提示值，生成的区间至少要包含hint中的一个值, 要求其值位于[0,max)之间
 * 输出:
 * oindex:[begin_index,end_index) shape=[2]的tensor用于表示一个范围
 * ohint:输入的hint中在[begin_index,end_index之间的部分
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
REGISTER_KERNEL_BUILDER(Name("IntHash").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), IntHash<CPUDevice, tensorflow::int64>);
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
 * line_no[batch_size,nr]
 *return: 
 * output:[batch_size,sample_nr,3] (id0,id1_pos,id2_neg) 内容为相应的索引
 */
REGISTER_OP("SampleLabels")
    .Attr("T: {int32, int64}")
	.Attr("sample_nr:int")
    .Input("labels: T")
    .Input("ids: T")
    .Input("line_no: T")
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
            const Tensor &_line_no    =  context->input(2);

            OP_REQUIRES(context, _labels.dims() == 2, errors::InvalidArgument("labels must be 2-dimensional"));
            OP_REQUIRES(context, _line_no.dims() == 2, errors::InvalidArgument("line no must be 2-dimensional"));
            OP_REQUIRES(context, _ids.dims() == 2, errors::InvalidArgument("ids must be 2-dimensional"));


            auto          labels      =  _labels.tensor<T,2>();
            auto          ids         =  _ids.tensor<T,2>();
            auto          line_no     =  _line_no.tensor<T,2>();
            auto          batch_size  =  labels.dimension(0);
            const auto    line_no_br  =  line_no.dimension(0);
            int dims_3d[] = {batch_size,sample_nr_,3};
            TensorShape outshape0;
            Tensor *output_data = NULL;

            TensorShapeUtils::MakeShape(dims_3d, 3, &outshape0);

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_data));

            auto out_tensor = output_data->tensor<T,3>();
            list<future<vector<tuple<T,T,T>>>> res;

            for(auto i=0; i<batch_size; ++i) {
                res.emplace_back(async(launch::async,&SampleLabelsOp<Device,T>::sample_one_batch,Eigen::Tensor<T,1,Eigen::RowMajor>(ids.chip(i,0)),
                Eigen::Tensor<T,1,Eigen::RowMajor>(labels.chip(i,0)),
                line_no.chip(line_no_br>1?i:0,0),
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
        static vector<tuple<T,T,T>> sample_one_batch(const Eigen::Tensor<T,1,Eigen::RowMajor>& ids,
        const Eigen::Tensor<T,1,Eigen::RowMajor>& labels,
        const Eigen::Tensor<T,1,Eigen::RowMajor>& line_no,
        int sample_nr) {
            //instance id->box index
            map<T,vector<int>> datas;
            map<T,int> id_to_label;
            map<int,vector<T>> label_to_id;
           const auto kDelta = 3;

            assert(ids.dimension(0)>0);
            const auto data_nr = ids.dimension(0);
            auto default_neg = data_nr-1;

            for(auto i=0; i<data_nr; ++i) {
                auto id = ids(i);
                if((id<1) || (labels(i)<1)) continue;
                auto it = datas.find(id);
                if(it == datas.end()) {
                    datas[id] = vector<int>({i});
                } else {
                    it->second.push_back(i);
                }
                const auto l = labels[i];
                id_to_label[id] = l;
            }
            for(auto it=id_to_label.begin(); it!=id_to_label.end(); ++it) {
                const auto id = it->first;
                const auto l = it->second;
                if(label_to_id.find(l) == label_to_id.end()) {
                    label_to_id[l] = vector<T>({id});
                } else {
                    label_to_id[l].push_back(id);
                }
            }
            /*
             * 用于简化采样时的操作
             */
            for(auto it=datas.begin(); it!=datas.end(); ++it) {
                if(it->second.size()==1) {
                    it->second.push_back(it->second[0]);
                }
            }

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, data_nr);

            if(datas.size() == 1) {
                auto it0 = datas.begin();
                auto ids = datas.begin()->second;
                int v0;
                int v1;
                std::tie(v0,v1) = sample_two_pos_int(ids,kDelta,line_no,[&dis,&gen]{return dis(gen);});
                auto neg_idx = (data_nr-1);
                if(find(ids.begin(),ids.end(),neg_idx) != ids.end()) {
                    neg_idx = 0;
                    if(find(ids.begin(),ids.end(),neg_idx) != ids.end()) {
                        cout<<"No neg idx find in sample_one_batch."<<endl;
                    }
                }
                vector<tuple<T,T,T>> res(sample_nr,make_tuple(v0,v1,neg_idx));
                return res;
            } else {
                /*
                 * 至少有两个以上的目标
                 */
                vector<tuple<T,T,T>> res(sample_nr);

                generate(res.begin(),res.end(),[&gen,&dis,&datas,&label_to_id,&id_to_label,&line_no](){
                        const auto id_index0 = dis(gen)%datas.size();
                        const auto id_index1 = sample_neg_data(datas,id_to_label,label_to_id,id_index0,[&dis,&gen]{return dis(gen);});
                        auto it0 = next(datas.begin(),id_index0);
                        auto it1 = next(datas.begin(),id_index1);
                        int v0;
                        int v1;
                        std::tie(v0,v1) = sample_two_pos_int(it0->second,kDelta,line_no,[&dis,&gen]{return dis(gen);});
                        auto id1_idx = dis(gen)%it1->second.size();
                        auto v2 = it1->second[id1_idx];
                        return make_tuple(v0,v1,v2);
                        });
                return res;
            } 
        }
        template<typename RFunc>
        static int sample_neg_data(const map<T,vector<int>>& id_to_index,const map<T,int>& id_to_label,const map<int,vector<T>>& label_to_id,int id_index,RFunc func) {
            /*
             * 尽量从具有相同label的实例中采样
             */
            auto id = next(id_to_index.begin(),id_index)->first;
            const auto label = id_to_label.at(id);
            auto ids = label_to_id.at(label);
            if(ids.size() == 1) {
                return sample_int_exclude(id_to_index.size(),id_index,func);
            } else {
                auto _index = distance(ids.begin(),find(ids.begin(),ids.end(),id));
                auto index = sample_int_exclude(ids.size(),_index,func);
                auto id1 = ids[index];
                assert(id_to_label.at(id1)==label);
                assert(id1!=id);
                auto it = id_to_index.find(id1);
                return distance(id_to_index.begin(),it);
            }

        }
        static pair<int,int> sample_two_int(int max_val,int delta,auto func) {
            const int v0 = func()%max_val;
            if(max_val<=delta) {  
                const auto v1 = sample_int_exclude(max_val,v0,func);
                return make_pair(v0,v1);
            }
            auto d_v1 = (func()%delta)+1;
            if(v0<max_val-1) {
                auto v1 = min(v0+d_v1,max_val-1);
                return make_pair(v0,v1);
            } else {
                return make_pair(v0,v0-d_v1);
            }
        };
        static pair<int,int> sample_two_pos_int(const vector<int>& indexs,int delta,
            const Eigen::Tensor<T,1,Eigen::RowMajor>& line_no,
            auto func) {
            /*
             * 尽量在不同的行采样
             */
            const int v0 = func()%indexs.size();
            const int index0 = indexs[v0];
            const int line_no0 = line_no(index0);
            vector<int> a_indexs;

            a_indexs.reserve(indexs.size());

            copy_if(indexs.begin(),indexs.end(),back_inserter(a_indexs),[line_no0,delta,&line_no](int v) {
                auto line_no1 = line_no(v);
                if(line_no1==line_no0) return false;
                return fabs(line_no1-line_no0)<=delta;
            });

            if(a_indexs.size()==0) {
                const auto v1 = sample_int_exclude(indexs.size(),v0,func);
                const int index1 = indexs[v1];
                return make_pair(index0,index1);
            }

            const auto v1 = func()%a_indexs.size();
            const auto index1 = a_indexs[v1];

            return make_pair(index0,index1);
        };
        template<typename RFunc>
        static int sample_int_exclude(int max_val,int exclude_v,RFunc func)
        {
            assert(max_val>0);
            auto res = func()%(max_val-1);
            return (res==exclude_v)?res+1:res;
        }
    private:
        int sample_nr_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("SampleLabels").Device(DEVICE_CPU).TypeConstraint<int>("T"), SampleLabelsOp<CPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("SampleLabels").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), SampleLabelsOp<CPUDevice, tensorflow::int64>);

/*
 * data:[nr,nr] (i,j)表示i到j的距离
 * labels:[nr]
 * bboxes:[nr,4]
 * threshold:
 * dis_threshold:[2](x,y)
 * output:[nr]
 */
REGISTER_OP("MergeLineBoxes")
    .Attr("T: {int32, int64}")
	.Attr("threshold:float")
	.Attr("dis_threshold:list(float)")
    .Input("data: float")
    .Input("labels: T")
    .Input("bboxes:float")
	.Output("ids:T")
	.Output("unique_ids:T")
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

            OP_REQUIRES(context, _labels.dims() == 1, errors::InvalidArgument("labels must be 1-dimensional"));
            OP_REQUIRES(context, _data.dims() == 2, errors::InvalidArgument("data must be 2-dimensional"));
            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes must be 2-dimensional"));

            auto          data       = _data.tensor<float,2>();
            auto          bboxes     = _bboxes.tensor<float,2>();
            auto          labels     = _labels.tensor<T,1>();
            auto          batch_size = labels.dimension(0);
            const auto     data_nr = labels.dimension(0);
            list<future<vector<int>>> res;
            auto res_data = process(data,labels,bboxes,data_nr);
            vector<int> res_data1 = res_data;

            sort(res_data1.begin(),res_data1.end());

            auto last = unique(res_data1.begin(),res_data1.end());
            res_data1.erase(last,res_data1.end());

            int dims_1d[] = {data_nr};
            int dims_1d2[] = {res_data1.size()};
            TensorShape outshape0;
            TensorShape outshape1;

            TensorShapeUtils::MakeShape(dims_1d, 1, &outshape0);
            TensorShapeUtils::MakeShape(dims_1d2, 1, &outshape1);

            Tensor *output_data = NULL;
            Tensor *output_data1 = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, outshape0, &output_data));
            OP_REQUIRES_OK(context, context->allocate_output(1, outshape1, &output_data1));

            auto out_tensor = output_data->tensor<T,1>();
            auto out_tensor1 = output_data1->tensor<T,1>();

            out_tensor.setConstant(0);

            for(auto j=0; j<data_nr; ++j) {
                out_tensor(j) = res_data[j];
            }
            for(auto j=0; j<res_data1.size(); ++j) {
                out_tensor1(j) = res_data1[j];
            }
        }
        static auto get_distance(const Eigen::Tensor<float,1,Eigen::RowMajor>& box0,
        const Eigen::Tensor<float,1,Eigen::RowMajor>& box1
        ) {
            float xdis;
            const float ydis = fabs(box0(0)+box0(2)-box1(0)-box1(2))/2.0f;
            const float box_h = (box0(2)-box0(0));

            if(fabs(box_h-(box1(2)-box1(0)))>1e-2) {
                cout<<"Error box height "<<box_h<<", "<<(box1(2)-box1(0))<<endl;
            }

            if(ydis<0.8*box_h)
                return make_pair(1e8f,1e8f);

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
        template<typename DT>
        void label_one(const DT& dis_matrix,
        int index,
        const Eigen::Tensor<float,2,Eigen::RowMajor>& data,
        vector<int>& ids,
        int data_nr) {
            for(auto j=0; j<data_nr; ++j) {
                if(ids[j]>0) continue;
                if((dis_matrix(index,j,0) < dis_threshold_[0])
                        &&(dis_matrix(index,j,1) < dis_threshold_[1])
                        && (data(index,j) <threshold_)){
                    ids[j] = ids[index];
                    label_one(dis_matrix,j,data,ids,data_nr);
                }
            }
        }
        vector<int> process(const Eigen::Tensor<float,2,Eigen::RowMajor>& data,
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
                label_one(dis_matrix,i,data,ids,data_nr);
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
    .SetShapeFn([](shape_inference::InferenceContext* c){
        auto shape0 = c->Vector(2);
        c->set_output(0,shape0);
		return Status::OK();
    });

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

/*
 * image[H,W,C]
 * boxes[N,4], 绝对坐标
 */
REGISTER_OP("FillBBoxes")
    .Attr("T: {float, double}")
    .Attr("v: float")
    .Attr("include_last: bool=True")
    .Input("image: T")
    .Input("bboxes: T")
	.Output("output:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->input(0));
		return Status::OK();
    });

template <typename Device, typename T>
class FillBoxesOp: public OpKernel {
	public:
		explicit FillBoxesOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("v", &v_));
			OP_REQUIRES_OK(context, context->GetAttr("include_last", &include_last_));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &_image      = context->input(0);
			const Tensor &_bboxes     = context->input(1);

			OP_REQUIRES(context, _image.dims() == 3, errors::InvalidArgument("images data must be 3-dimensional"));
			OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("boxes data must be 2-dimensional"));

			auto          image       = _image.tensor<T,3>();
			const auto    bboxes      = _bboxes.tensor<T,2>();
            const auto    box_nr      = _bboxes.dim_size(0);

			TensorShape  output_shape  = _image.shape();
			Tensor      *output_tensor = nullptr;

			OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

            auto out_tensor = output_tensor->tensor<T,3>();
            out_tensor = image;
            for(auto i=0; i<box_nr; ++i) 
                draw_a_box(out_tensor,bboxes.chip(i,0));

		}
        template<typename IT>
            void draw_a_box(IT& image,const Eigen::Tensor<T,1,Eigen::RowMajor>& box) {
                //使用float结束，结果更准确
                const auto xmin = max<int>(0,box(1));
                const auto xmax = min<float>(image.dimension(1),box(3));
                const auto ymin = max<int>(0,box(0));
                const auto ymax = min<float>(image.dimension(0),box(2));
                const auto channel = image.dimension(2);

                if(include_last_)
                    for(int x=xmin; x<=xmax; ++x) {
                        for(int y=ymin; y<=ymax; ++y) {
                            for(auto z=0; z<channel; ++z)
                                image(y,x,z) = v_;
                        }
                    }
                else
                    for(int x=xmin; x<xmax; ++x) {
                        for(int y=ymin; y<ymax; ++y) {
                            for(auto z=0; z<channel; ++z)
                                image(y,x,z) = v_;
                        }
                    }
            }
	private:
		float v_ = 1.0;
        bool include_last_ = true;

};
REGISTER_KERNEL_BUILDER(Name("FillBBoxes").Device(DEVICE_CPU).TypeConstraint<float>("T"), FillBoxesOp<CPUDevice, float>);

/*
 * 在data最后一维为True的位置随机选择出nr个,并将其它的设置为False
 * data: [D0,D1,...,Dn] a bool tensor
 * indices:[D0,D1,...,nr] 与返回值相对应的indices
 */
REGISTER_OP("RandomSelect")
    .Attr("nr: int")
    .Attr("sort_indices: bool = False")
    .Input("data: bool")
	.Output("output:bool")
	.Output("indices:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            auto input_shape0 = c->input(0);
            const auto dims = c->Rank(input_shape0);
            int nr;

            c->GetAttr("nr",&nr);

            shape_inference::ShapeHandle tmp_shape0;
            shape_inference::ShapeHandle tmp_shape1 = c->MakeShape({nr});
            shape_inference::ShapeHandle output_shape1;

            c->Subshape(input_shape0,0,-1,&tmp_shape0);
            c->Concatenate(tmp_shape0,tmp_shape1,&output_shape1);
			c->set_output(0, input_shape0);
			c->set_output(1, output_shape1);
			return Status::OK();
			});

template <typename Device>
class RandomSelectOp: public OpKernel {
	public:
		explicit RandomSelectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("nr", &nr_));
			OP_REQUIRES_OK(context, context->GetAttr("sort_indices", &sort_indices_));
		}

		void Compute(OpKernelContext* context) override
		{
            const Tensor &_tensor        = context->input(0);
            auto          tensor         = _tensor.template flat<bool>().data();
            auto          dim_nr         = _tensor.dims();
            const auto    block_size     = _tensor.dim_size(dim_nr-1);
            const auto    total_nr       = _tensor.NumElements()/block_size;


            Tensor* output_data = NULL;
            Tensor* output_indices = NULL;
            TensorShape output_shape1 = _tensor.shape();

            output_shape1.set_dim(dim_nr-1,nr_);

			OP_REQUIRES(context, _tensor.dims() >= 1, errors::InvalidArgument("data must be at lest 1-dimensional"));
            OP_REQUIRES_OK(context, context->allocate_output(0, _tensor.shape(), &output_data));
            OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_indices));


            auto       oq_tensor    = output_data->template flat<bool>().data();
            auto       oi_tensor    = output_indices->template flat<int>().data();
            const auto kMaxThreadNr = 100;
            std::vector<std::future<void>> res;
            const auto kDataNrPerThread = 20000;
            const auto kBatchSizePerThread = std::max<int>(1,kDataNrPerThread/block_size);

            output_indices->template flat<int>().setZero();
            copy(tensor,tensor+_tensor.NumElements(),oq_tensor);

            for(auto i=0; i<total_nr; i+=kBatchSizePerThread) {
                res.emplace_back(std::move(std::async(std::launch::async,
                                process_one_batch,oq_tensor+i*block_size,oi_tensor+i*nr_,
                                std::min<int>(kBatchSizePerThread,total_nr-i),
                                block_size,nr_,sort_indices_
                                )));
                if(res.size()>kMaxThreadNr)
                    res.clear();
            }
            res.clear();
		}
        static void process_one_batch(bool* data,int* o_indices,int batch_size,int size,int nr,bool sort_indices){
            for(auto i=0; i<batch_size; ++i) {
                 process_one_block(data+i*size,o_indices+i*nr,size,nr,sort_indices);
            }
        }
        static void process_one_block(bool* data,int* o_indices,int size,int nr,bool sort_indices){
            vector<int> indices;
            indices.reserve(nr*2);
            for(auto i=0; i<size; ++i){
                if(data[i])
                    indices.push_back(i);
            }
            if(indices.size()>=nr) {
                std::random_shuffle(indices.begin(),indices.end());
                for(auto i=nr; i<indices.size(); ++i) {
                    data[indices[i]] = false;
                }
            } 
            nr = std::min<int>(nr,indices.size());

            if(sort_indices)
                std::sort(indices.begin(),std::next(indices.begin(),nr));

            for(auto i=0; i<nr; ++i) {
                o_indices[i] = indices[i];
            }
        }
	private:
        int  nr_           = 1;
        bool sort_indices_ = false;

};
REGISTER_KERNEL_BUILDER(Name("RandomSelect").Device(DEVICE_CPU), RandomSelectOp<CPUDevice>);
/*
 * 将输入数据按其值的大小均分为his_nr个值域，从每个值域中选出select_nr/his_nr个值（如果某个值域中没有足够
 * 的数据，同一个值可能被选择多次)
 * 返回为所选择的数据的index, 如果需要排序则按index的大小从小到大排
 * data: [N]
 * output: [select_nr]
 */
REGISTER_OP("HisRandomSelect")
    .Attr("T: {float, double,int32}")
    .Attr("his_nr: int=0")
    .Attr("min: float=0")
    .Attr("max: float=0")
    .Attr("const_min_max: bool=True")
    .Attr("sort_indices: bool = False")
    .Input("data: T")
    .Input("select_nr: int32")
	.Output("indices:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
            shape_inference::ShapeHandle tmp_shape = c->MakeShape({-1});

			c->set_output(0, tmp_shape);
			return Status::OK();
			});

template <typename Device,typename T>
class HisRandomSelectOp: public OpKernel {
	public:
		explicit HisRandomSelectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("his_nr", &his_nr_));
			OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
			OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
			OP_REQUIRES_OK(context, context->GetAttr("const_min_max", &const_min_max_));
			OP_REQUIRES_OK(context, context->GetAttr("sort_indices", &sort_indices_));
            std::srand(std::time(nullptr)); // use current time as seed for random generator
		}

		void Compute(OpKernelContext* context) override
		{
            const Tensor &_tensor        = context->input(0);
            auto          tensor         = _tensor.template flat<T>().data();
            auto          data_nr        = _tensor.dim_size(0);
            const Tensor &_select_nr     = context->input(1);
            auto          select_nr      = _select_nr.template flat<int>().data()[0];
            Tensor       *output_indices = NULL;
            int           dim1[]         = {select_nr};
            TensorShape   output_shape;

            TensorShapeUtils::MakeShape(dim1,1,&output_shape);

			OP_REQUIRES(context, _tensor.dims() == 1, errors::InvalidArgument("data must be at 1-dimensional"));

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_indices));


            if(!const_min_max_) {
                auto res = minmax_element(tensor,tensor+data_nr);
                min_ = *res.first;
                max_ = *res.second;
            }

            vector<vector<int>> indices(his_nr_);
            const auto delta = (max_-min_)/his_nr_;
            const auto bin_nr = select_nr/his_nr_;
            auto    out_data = output_indices->template flat<int>().data();
            vector<int> unused_index;

            unused_index.reserve(max<int>(16,data_nr-select_nr));

            for(auto i=0; i<data_nr; ++i) {
                int idx = (tensor[i]-min_)/delta;
                idx = min(idx,his_nr_-1);
                idx = max(idx,0);
                indices[idx].push_back(i);
            }

            int total_res_nr = 0;
            for(auto i=0; i<his_nr_; ++i) {
                auto& tmp_indices = indices[i];
                auto tmp_data = random_select(tmp_indices.begin(),tmp_indices.end(),bin_nr,&unused_index);
                copy(tmp_data.begin(),tmp_data.end(),out_data+total_res_nr);
                total_res_nr += tmp_data.size();
            }
            if(total_res_nr<select_nr) {
                const auto tmp_sel_nr = select_nr-total_res_nr;
                if(unused_index.size()>=tmp_sel_nr) {
                    auto tmp_data = random_select(unused_index.begin(),unused_index.end(),tmp_sel_nr);
                    copy(tmp_data.begin(),tmp_data.end(),out_data+total_res_nr);
                } else {
                    vector<int> tmp_tensor(data_nr);
                    //copy(tensor,tensor+data_nr,tmp_tensor.begin());
                    generate(tmp_tensor.begin(),tmp_tensor.end(),[n=0]()mutable{return n++;});
                    auto tmp_data = random_select_v2(tmp_tensor.begin(),tmp_tensor.end(),tmp_sel_nr);
                    copy(tmp_data.begin(),tmp_data.end(),out_data+total_res_nr);
                }
            }
            if(sort_indices_)
                sort(out_data,out_data+select_nr);
		}
        template<typename IT>
        vector<int> random_select(IT begin, IT end, int nr,vector<int>* unused_index=nullptr) {
            vector<int> res;
            const auto input_nr = distance(begin,end);

            if((0 == input_nr) || (nr==0))
                return res;

            res.reserve(nr);

            std::random_shuffle(begin,end);

            if(nr<input_nr) {
                res.insert(res.end(),begin,next(begin,nr));
                if(nullptr != unused_index)
                    unused_index->insert(unused_index->end(),next(begin,nr),end);
            } else {
                res.insert(res.end(),begin,end);
            }
            return res;
        }
        template<typename IT>
        vector<int> random_select_v2(IT begin, IT end, int nr) {
            vector<int> res;
            const auto input_nr = distance(begin,end);

            if((0 == input_nr) || (nr==0))
                return res;

            res.reserve(nr);

            std::random_shuffle(begin,end);

            if(nr<=input_nr) {
                res.insert(res.end(),begin,next(begin,nr));
            } else {
                auto repeat_nr = nr/input_nr;
                for(auto i=0; i<repeat_nr; ++i) {
                    res.insert(res.end(),begin,end);
                }
                
                repeat_nr = nr-res.size();
                res.insert(res.end(),begin,next(begin,repeat_nr));
            }
            return res;
        }
	private:
        int   his_nr_        = 1;
        bool  const_min_max_ = true;
        float min_           = 0;
        float max_           = 0;
        bool  sort_indices_  = false;
};
/*
 * int特化版本
 * 针对每一个值取相同的数量的样本
 */
template <typename Device>
class HisRandomSelectOp<Device,int>: public OpKernel {
    using T = int;
	public:
		explicit HisRandomSelectOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("sort_indices", &sort_indices_));
            std::srand(std::time(nullptr)); // use current time as seed for random generator
		}

		void Compute(OpKernelContext* context) override
		{
            const Tensor &_tensor        = context->input(0);
            auto          tensor         = _tensor.template flat<T>().data();
            auto          data_nr        = _tensor.dim_size(0);
            const Tensor &_select_nr     = context->input(1);
            auto          select_nr      = _select_nr.template flat<int>().data()[0];
            Tensor       *output_indices = NULL;
            int           dim1[]         = {select_nr};
            TensorShape   output_shape;

            TensorShapeUtils::MakeShape(dim1,1,&output_shape);

			OP_REQUIRES(context, _tensor.dims() == 1, errors::InvalidArgument("data must be at 1-dimensional"));

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_indices));

            map<int,vector<int>> indices;
            auto    out_data = output_indices->template flat<int>().data();
            vector<int> unused_index;

            unused_index.reserve(max<int>(16,data_nr-select_nr));

            for(auto i=0; i<data_nr; ++i) {
                const int key = tensor[i];
                if(indices.find(key) == indices.end())
                    indices[key] = vector<int>();
                indices[key].push_back(i);
            }

            int total_res_nr = 0;
            const int bin_nr = select_nr/indices.size();

            if(bin_nr>0) {
                for(auto it=indices.begin(); it != indices.end(); ++it) {
                    auto& tmp_indices = it->second;
                    auto tmp_data = random_select(tmp_indices.begin(),tmp_indices.end(),bin_nr,&unused_index);
                    copy(tmp_data.begin(),tmp_data.end(),out_data+total_res_nr);
                    total_res_nr += tmp_data.size();
                }
            }

            if(total_res_nr<select_nr) {
                const auto tmp_sel_nr = select_nr-total_res_nr;
                if(unused_index.size()>=tmp_sel_nr) {
                    auto tmp_data = random_select(unused_index.begin(),unused_index.end(),tmp_sel_nr);
                    copy(tmp_data.begin(),tmp_data.end(),out_data+total_res_nr);
                } else {
                    vector<int> tmp_tensor(data_nr);
                    generate(tmp_tensor.begin(),tmp_tensor.end(),[n=0]()mutable{return n++;});
                    auto tmp_data = random_select_v2(tmp_tensor.begin(),tmp_tensor.end(),tmp_sel_nr);
                    copy(tmp_data.begin(),tmp_data.end(),out_data+total_res_nr);
                }
            }
            if(sort_indices_)
                sort(out_data,out_data+select_nr);
		}

        template<typename IT>
        vector<int> random_select(IT begin, IT end, int nr,vector<int>* unused_index=nullptr) {
            vector<int> res;
            const auto input_nr = distance(begin,end);

            if((0 == input_nr) || (nr==0))
                return res;

            res.reserve(nr);

            std::random_shuffle(begin,end);

            if(nr<input_nr) {
                res.insert(res.end(),begin,next(begin,nr));
                if(nullptr != unused_index)
                    unused_index->insert(unused_index->end(),next(begin,nr),end);
            } else {
                res.insert(res.end(),begin,end);
            }
            return res;
        }
        template<typename IT>
        vector<int> random_select_v2(IT begin, IT end, int nr) {
            vector<int> res;
            const auto input_nr = distance(begin,end);

            if((0 == input_nr) || (nr==0))
                return res;

            res.reserve(nr);

            std::random_shuffle(begin,end);

            if(nr<=input_nr) {
                res.insert(res.end(),begin,next(begin,nr));
            } else {
                auto repeat_nr = nr/input_nr;
                for(auto i=0; i<repeat_nr; ++i) {
                    res.insert(res.end(),begin,end);
                }
                
                repeat_nr = nr-res.size();
                res.insert(res.end(),begin,next(begin,repeat_nr));
            }
            return res;
        }
	private:
        bool  sort_indices_  = false;
};
REGISTER_KERNEL_BUILDER(Name("HisRandomSelect").Device(DEVICE_CPU).TypeConstraint<float>("T"), HisRandomSelectOp<CPUDevice,float>);
REGISTER_KERNEL_BUILDER(Name("HisRandomSelect").Device(DEVICE_CPU).TypeConstraint<int>("T"), HisRandomSelectOp<CPUDevice,int>);
/*
data:输入Tensor,shape为[X,Y]
输出:output shape为[X*(1+expand_nr),Y]
如输入[[1,2],
[3,4]]
expand_nr = 2:
输出:
[[1,2],
[1,2],
[1,2],
[3,4],
[3,4],
[3,4]]
*/
REGISTER_OP("ExpandTensor")
    .Attr("T: {int32, int64,float32,float64}")
	.Attr("expand_nr:int")
    .Input("data: T")
	.Output("output:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            auto dims_data0 = c->input(0);
            int expand_nr = 0;
            c->GetAttr("expand_nr",&expand_nr);
            auto batch_size = c->Value(c->Dim(dims_data0,0))*(1+expand_nr);
            auto output_shape0 = c->Matrix(batch_size,c->Dim(dims_data0,1));

            c->set_output(0,output_shape0);
            return Status::OK();
            });

template <typename Device, typename T>
class ExpandTensorOp: public OpKernel {
	public:
		explicit ExpandTensorOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context,
					context->GetAttr("expand_nr", &expand_nr));
		}

		void Compute(OpKernelContext* context) override
		{
			const Tensor &data= context->input(0);
			auto          data_flat = data.flat<T>();

			OP_REQUIRES(context, data.dims() == 2, errors::InvalidArgument("data data must be 2-dimensional"));

			const auto batch_size   = data.dim_size(0);
			const auto num_output   = batch_size *(1+expand_nr);
			const auto data_len = data.dim_size(1);

			TensorShape output_shape0({num_output,data_len});

			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_data));

			auto oq_flat = output_data->flat<T>();

			for(auto i=0; i<batch_size; ++i) {
				auto bq_i = data_flat.data()+data_len*i;
				auto bq_o = oq_flat.data()+data_len*i*(expand_nr+1);
				for(auto k=0; k<=expand_nr; ++k) {
					for(auto j=0; j<data_len; ++j) {
						bq_o[j] = bq_i[j];
					}
					bq_o += data_len;
				}
			}
		}
	private:
		int expand_nr;
};
REGISTER_KERNEL_BUILDER(Name("ExpandTensor").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), ExpandTensorOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("ExpandTensor").Device(DEVICE_CPU).TypeConstraint<float>("T"), ExpandTensorOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("ExpandTensor").Device(DEVICE_CPU).TypeConstraint<double>("T"), ExpandTensorOp<CPUDevice, double>);
Status slide_batch_shape(shape_inference::InferenceContext* c) 
{
    auto                         shape        = c->input(0);
    auto                         filter_shape = c->input(1);
    string                       padding;
    vector<int>                  strides;
    shape_inference::ShapeHandle output;

    c->GetAttr("padding",&padding);
    c->GetAttr("strides",&strides);

    auto org_h = c->Value(c->Dim(shape,0));
    auto org_w = c->Value(c->Dim(shape,1));

    if(padding == "SAME") {
        org_h += (c->Value(c->Dim(filter_shape,0))-1);
        org_w += (c->Value(c->Dim(filter_shape,1))-1);
    }

    const auto h_size =  (org_h-c->Value(c->Dim(filter_shape,0)))/strides[0]+1;
    const auto w_size =  (org_w-c->Value(c->Dim(filter_shape,1)))/strides[1]+1;


    c->Concatenate(c->MakeShape({h_size,w_size}),shape,&output);
    c->set_output(0,output);

    return Status::OK();
}
/*
 * 输入一个[H,W,C]或[H,W]的tensor
 * 输出一个[H1,W1,H,W,C]的tensor
 * filter:[h,w,c]或[h,w]
 * H1,W1指定的每一个tensor都是原tensor在相应位置与filter相乘的结果
 */
REGISTER_OP("SlideBatch")
    .Attr("T: {int32, int64,float32,float64}")
	.Attr("strides:list(int)")
	.Attr("padding:string")
    .Input("data: T")
    .Input("filter: T")
	.Output("output:T")
	.SetShapeFn(slide_batch_shape);

template <typename Device, typename T>
class SlideBatchOp: public OpKernel {
	public:
		explicit SlideBatchOp(OpKernelConstruction* context) : OpKernel(context) {
			OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
			OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
		}
        inline bool is_same() const {
            return padding_ == "SAME";
        }
        inline bool is_valid()const {
            return !is_same();
        }
        inline int h_stride()const { return strides_[0]; }
        inline int w_stride()const { return strides_[1]; }
		inline int h_begin(int fh)const { 
			if(is_same())
				return (1-fh)/2;
			else
				return 0;
		}
		inline int h_end(int fh,int h)const { 
			if(is_same())
				return h-1;
			else
				return h-fh+1;
		}
        inline int w_begin(int fw)const { 
			if(is_same())
				return (1-fw)/2;
			else
				return 0;
        }
        inline int w_end(int fw,int w)const { 
			if(is_same())
				return w-1;
			else
				return w-fw+1;
        }
        inline size_t output_h_size(int fh,int h) const {
            return (h_end(fh,h)-h_begin(fh))/h_stride();
        }
        inline size_t output_w_size(int fw,int w) const {
            return (w_end(fw,w)-w_begin(fw))/w_stride();
        }
		void Compute(OpKernelContext* context) override
		{
			const Tensor &data   = context->input(0);
			const Tensor &filter = context->input(1);
			const int     fh     = filter.dim_size(0);
			const int     fw     = filter.dim_size(1);
            Eigen::Tensor<T, 3,Eigen::RowMajor>  data_t = data.template tensor<T,3>();
            Eigen::Tensor<T,3,Eigen::RowMajor> filter_t = filter.template tensor<T,3>();

			auto      data_flat   = data.flat<T>();
			auto      filter_flat = filter.flat<T>();
			const int h           = data_t.dimension(0);
			const int w           = data_t.dimension(1);
			const int c           = data_t.dimension(2);
			const int oh          = output_h_size(fh,h);
			const int ow          = output_w_size(fw,w);

			TensorShape output_shape0({oh,ow,data.dim_size(0),data.dim_size(1),c});
			cout<<"oh="<<oh<<", ow="<<ow<<endl;

			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_data));

			auto      oq_tensor = output_data->template tensor<T,5>();
			const int size[]    = {h,w,c};

            for(auto i=0; i<oh; ++i) {
                for(auto j=0; j<ow; ++j) {
                    Eigen::array<int, 3> offsets = {h_begin(fh)+h_stride()*i,w_begin(fw)+w_stride()*j,0};
                    Eigen::array<int, 3> extents = {fh, fw,c};
                    Eigen::array<int, 3> offsets1 = {0,0};

					correct(offsets,extents,offsets1,size);

                    auto v      = data_t.slice(offsets, extents) *filter_t.slice(offsets1,extents);
                    auto target = oq_tensor.chip(i,0).chip(j,0);

					target                         =  data_t;
					target.slice(offsets,extents)  =  v;
                }
            }
		}
		static void correct(Eigen::array<int,3>& offsets,Eigen::array<int,3>& extents, Eigen::array<int,3>& offsets1,const int size[3] ) {
			for(auto i=0; i<3; ++i) {
				if(offsets[i]< 0) {
				extents[i] = offsets[i]+extents[i];
				offsets1[i] = -offsets[i];
				offsets[i] = 0;
				} else  if(offsets[i]+extents[i] > size[i]) {
					extents[i] = size[i]-offsets[i];
				}
			}
		}
	private:
		vector<int> strides_;
		string      padding_;
};
REGISTER_KERNEL_BUILDER(Name("SlideBatch").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), SlideBatchOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("SlideBatch").Device(DEVICE_CPU).TypeConstraint<float>("T"), SlideBatchOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SlideBatch").Device(DEVICE_CPU).TypeConstraint<double>("T"), SlideBatchOp<CPUDevice, double>);

/*
 * 输入data 1D:Tensor
 * padding:表示在axis 0上进行对称padding的数量,负数或0表示无操作
 * 如果原有数据的数量为0， 则使用0填充
 */
REGISTER_OP("WPad")
    .Attr("T: {int32, int64,float32,float64}")
    .Input("tensor: T")
    .Input("padding: int32")
	.Output("data:T");

template <typename Device, typename T>
class WPadOp: public OpKernel {
	public:
		explicit WPadOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
		{
			const Tensor &_data    = context->input(0);
			const Tensor &_padding = context->input(1);
			auto          data     = _data.template flat<T>().data();
			auto          padding  = _padding.template flat<int>().data();
			const auto    data_nr  = _data.dim_size(0);
			const auto    out_size = data_nr+ std::max<int>(0,padding[0])+std::max<int>(0,padding[1]);

			OP_REQUIRES(context, _data.dims()<=1, errors::InvalidArgument("tensor data must be 1-dimensional"));
			OP_REQUIRES(context, _padding.dims()<=1, errors::InvalidArgument("padding data must be 1-dimensional"));

			TensorShape output_shape0({out_size});
			Tensor* output_data = NULL;

			OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_data));

			auto      oq_tensor = output_data->template flat<T>().data();

			/*
			 * 如果原始数据中没有内容，使用0填充
			 */
			if(data_nr == 0) {
				for(auto i=0; i<out_size; ++i) {
					oq_tensor[i] = 0.0f;
				}
				return;
			} 
			/*
			 * 原始数据中有内容，对称填充
			 */

            for(auto i=0; i<padding[0]; ++i) {
                oq_tensor[i] = data[(padding[0]-i-1)%data_nr];
            }
            auto base_index = std::max<int>(0,padding[0]);
            for(auto i=0; i<data_nr; ++i) {
                oq_tensor[i+base_index] = data[i];
            }
            base_index = std::max<int>(0,padding[0])+data_nr;
            auto src_index = data_nr-1;
            for(auto i=0; i<padding[1]; ++i,--src_index) {
                if(src_index<0)
                    src_index = data_nr-1;
                oq_tensor[i+base_index] = data[src_index];
            }
		}
};
REGISTER_KERNEL_BUILDER(Name("WPad").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), WPadOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("WPad").Device(DEVICE_CPU).TypeConstraint<float>("T"), WPadOp<CPUDevice, float>);

/*
 * 设置多个子tensor的值
 * 输入tensor [X,Y,Z,...,M,N,..]tensor
 * 输入v[M,N,...] tensor
 * 输入index[num]，依次表示[X,Y,Z,...]维度的值
 * 将tensor中由index指定的值设置为
 * example:
 * tensor shape=[2,3,4,2,2]
 * v shape=[2,2]
 * index=[[0,1,3]]
 * 那么tensor[0,1,3]=v
 */
REGISTER_OP("SetValue")
    .Attr("T: {int32,int64,float32,float64,bool}")
    .Input("tensor: T")
    .Input("v: T")
    .Input("index: int32")
	.Output("data:T")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class SetValueOp: public OpKernel {
	public:
		explicit SetValueOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor        = context->input(0);
            const Tensor &_v             = context->input(1);
            const Tensor &_index         = context->input(2);

            OP_REQUIRES(context, _index.dims()==2, errors::InvalidArgument("index must be 2-dimensional"));

            auto          tensor         = _tensor.template flat<T>().data();
            auto          v              = _v.template flat<T>().data();
            auto          indexs         = _index.template tensor<int,2>();
            auto          dim_nr         = _tensor.dims();
            auto          skip_dim_nr    = _index.dim_size(1);
            const auto    block_size     = _v.NumElements();

            Tensor* output_data = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, _tensor.shape(), &output_data));

            auto      oq_tensor = output_data->template flat<T>().data();
            copy(tensor,tensor+_tensor.NumElements(),oq_tensor);

            /*
             * 如果原始数据中没有内容，使用0填充
             */
            const auto nr = _index.dim_size(0);

            for(auto i=0; i<nr; ++i) {
                auto offset         = 0;
                auto cur_block_size = block_size;

                for(auto j=skip_dim_nr-1; j>=0; --j) {
                    offset += indexs(i,j)*cur_block_size;
                    cur_block_size *= _tensor.dim_size(j);
                }
                copy(v,v+block_size,oq_tensor+offset);
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), SetValueOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<float>("T"), SetValueOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<double>("T"), SetValueOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<bool>("T"), SetValueOp<CPUDevice, bool>);
REGISTER_KERNEL_BUILDER(Name("SetValue").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), SetValueOp<CPUDevice, tensorflow::int64>);
/*
 * 实现类似于numpy tensor[index[0][0]:index[0][1],index[1][0]:index[1][1],..] = v的功能
 * 输入tensor [X,Y,Z,]tensor
 * 输入 v，值
 * 输入index [Nr,2] 分别表示tensor X,Y,Z...维度的起始及终止范围tensor
 * example:
 * tensor shape=[2,3,4,2,2]
 * v =[0,9]
 * index=[[1,3]]
 * 那么tensor = [2,0,9,2,2]
 */
REGISTER_OP("ItemAssign")
    .Attr("T: {int32,int64,float32,float64,bool,uint8,int8}")
    .Input("tensor: T")
    .Input("v: T")
    .Input("index: int32")
	.Output("data:T")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->input(0));
		return Status::OK();
    });

template <typename Device, typename T>
class ItemAssignOp: public OpKernel {
	public:
		explicit ItemAssignOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor     = context->input(0);
            const Tensor &_v          = context->input(1);
            const Tensor &_index      = context->input(2);
            auto          tensor_flat = _tensor.template flat<T>().data();
            auto          indexs      = _index.template tensor<int,2>();
            auto          dim_nr      = _tensor.dims();
            auto          index_nr    = _index.dim_size(0);

            OP_REQUIRES(context, _index.dims()==2, errors::InvalidArgument("index must be 2-dimensional"));
            OP_REQUIRES(context, index_nr==dim_nr, errors::InvalidArgument("index size must equal tensor's dim size"));

            Tensor* output_data = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, _tensor.shape(), &output_data));

            auto      oq_tensor = output_data->template flat<T>().data();

            copy(tensor_flat,tensor_flat+_tensor.NumElements(),oq_tensor);

            for(auto i=0; i<index_nr; ++i) {
                if(indexs(i,0)==indexs(i,1))
                    return;
                if((indexs(i,0)>indexs(i,1))
                    || (indexs(i,0)<0)
                    || (indexs(i,1)>_tensor.dim_size(i))) {
                    cout<<"Error index value ("<<indexs(i,0)<<","<<indexs(i,1)<<"), tensor dim size: "<<_tensor.dim_size(i)<<endl;
                    return;
                }
            }

            if(1 == dim_nr) {
                auto tensor = output_data->template tensor<T,1>();
                auto v      = _v.template tensor<T,1>();
                Eigen::array<long,1> offset={indexs(0,0)};
                Eigen::array<long,1> extents={indexs(0,1)-indexs(0,0)};

                tensor.slice(offset,extents) = v;
            } else if(2 == dim_nr) {
                auto tensor = output_data->template tensor<T,2>();
                auto v      = _v.template tensor<T,2>();
                Eigen::array<long,2> offset={indexs(0,0),indexs(1,0)};
                Eigen::array<long,2> extents={indexs(0,1)-indexs(0,0),indexs(1,1)-indexs(1,0)};

                tensor.slice(offset,extents) = v;
            } else if(3 == dim_nr) {
                auto tensor = output_data->template tensor<T,3>();
                auto v      = _v.template tensor<T,3>();
                Eigen::array<long,3> offset={indexs(0,0),indexs(1,0),indexs(2,0)};
                Eigen::array<long,3> extents={indexs(0,1)-indexs(0,0),indexs(1,1)-indexs(1,0),indexs(2,1)-indexs(2,0)};

                tensor.slice(offset,extents) = v;
            } else if(4 == dim_nr) {
                auto tensor = output_data->template tensor<T,4>();
                auto v      = _v.template tensor<T,4>();
                Eigen::array<long,4> offset={indexs(0,0),indexs(1,0),indexs(2,0),indexs(3,0)};
                Eigen::array<long,4> extents={indexs(0,1)-indexs(0,0),indexs(1,1)-indexs(1,0),indexs(2,1)-indexs(2,0),indexs(3,1)-indexs(3,0)};

                tensor.slice(offset,extents) = v;
            } else {
                cout<<"Error unimplement for dim_nr = "<<dim_nr<<endl;
            }
        }
};
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), ItemAssignOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<float>("T"), ItemAssignOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<double>("T"), ItemAssignOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<bool>("T"), ItemAssignOp<CPUDevice, bool>);
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), ItemAssignOp<CPUDevice, tensorflow::int64>);
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), ItemAssignOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("ItemAssign").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), ItemAssignOp<CPUDevice, int8_t>);
/*
 * 对输入tensor的值计数，如输入[1,1,2,4,3,1,1,] max_value=7,输出
 * [0,4,1,1,1,0,0,0]
 */
REGISTER_OP("Counting")
    .Attr("T: {int32,int64,uint8,int8}")
	.Attr("max_value:int")
    .Input("tensor: T")
	.Output("data:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
            int max_value = 0;
            c->GetAttr("max_value",&max_value);
            auto shape0 = c->MakeShape({max_value+1});
			c->set_output(0, shape0);
			return Status::OK();
		return Status::OK();
    });

template <typename Device, typename T>
class CountingOp: public OpKernel {
	public:
		explicit CountingOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("max_value", &max_value_));
		}
		void Compute(OpKernelContext* context) override
        {
            const Tensor &_tensor     = context->input(0);
            auto          tensor_flat = _tensor.template flat<T>().data();
            auto          total_nr = _tensor.NumElements();


            Tensor* output_data = NULL;
            TensorShape  output_shape;
            const int    dim0[]           = {max_value_+1};

            TensorShapeUtils::MakeShape(dim0,1,&output_shape);

            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_data));

            auto      oq_tensor = output_data->template flat<int>();

            oq_tensor.setZero();

            for(auto i=0; i<total_nr; ++i) {
                const auto v = tensor_flat[i];
                oq_tensor(v) = oq_tensor(v)+1;
            }
        }
   private:
     int max_value_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("Counting").Device(DEVICE_CPU).TypeConstraint<int32_t>("T"), CountingOp<CPUDevice, int32_t>);
REGISTER_KERNEL_BUILDER(Name("Counting").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), CountingOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("Counting").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), CountingOp<CPUDevice, int8_t>);
REGISTER_KERNEL_BUILDER(Name("Counting").Device(DEVICE_CPU).TypeConstraint<tensorflow::int64>("T"), CountingOp<CPUDevice, tensorflow::int64>);
