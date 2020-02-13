#include <stdio.h>
#include <cfloat>
#include <list>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <boost/algorithm/clamp.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/work_sharder.h"
#include <boost/geometry/io/wkt/write.hpp>
#include "bboxes.h"

using namespace boost;
using namespace tensorflow;
using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef adjacency_matrix<undirectedS, no_property, property <edge_weight_t, float>> Graph;
typedef graph_traits <Graph>::edge_descriptor Edge;
typedef std::pair<int, int> E;

/*
 * 根据输入的boxes, labels生成一个邻接矩阵
 * min_nr: 每个节点至少与min_nr个节点相连接
 * min_dis:如果两个节点间的距离小于min_dis, 他们之间应该需要一个连接
 * boxes: [N,4], ymin,xmin,ymax,xmax相对坐标
 * output:
 * matrix[N,N] 第i行,j列表示第i个点与第j个点之间的连接
 */
REGISTER_OP("AdjacentMatrixGenerator")
    .Attr("T: {float, double,int32}")
	.Attr("theta:float")
	.Attr("scale:float")
	.Attr("coord_scales:list(float)")
    .Input("bboxes: T")
	.Output("matrix:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto box_nr = c->Dim(c->input(0),0);
			c->set_output(0, c->Matrix(box_nr,box_nr));
			return Status::OK();
			});

template <typename Device, typename T>
class AdjacentMatrixGeneratorOp: public OpKernel {
    public:
        explicit AdjacentMatrixGeneratorOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("theta", &theta_));
            OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
            OP_REQUIRES_OK(context, context->GetAttr("coord_scales", &coord_scales_));
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &_bboxes  = context->input(0);
            auto          bboxes   = _bboxes.template tensor<T,2>();

            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes data must be 2-dimensional"));
            const auto bboxes_nr    = _bboxes.dim_size(0);
            Eigen::Tensor<float,2,Eigen::RowMajor> dis_matrix(bboxes_nr,bboxes_nr);

            dis_matrix.setZero();

            for(auto i=0; i<bboxes_nr-1; ++i) {
                const Eigen::Tensor<T,1,Eigen::RowMajor> box_data0 = bboxes.chip(i,0);

                dis_matrix(i,i) = 0.;
                for(auto j=i+1; j<bboxes_nr; ++j) {
                    const Eigen::Tensor<T,1,Eigen::RowMajor> box_data1 = bboxes.chip(j,0);
                    const auto dis = distance(box_data0,box_data1);
                    dis_matrix(i,j) = dis;
                    dis_matrix(j,i) = dis;
                }
            }

            auto res = make_graph(bboxes,dis_matrix,theta_,scale_);

            Tensor      *output_matrix = NULL;
            TensorShape  output_shape;
            const int    dims_2d[]     = {int(bboxes_nr),int(bboxes_nr)};

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_matrix));

            auto output           = output_matrix->template tensor<int,2>();
            for(auto i=0; i<bboxes_nr; ++i) {
                output(i,i) = 0;
                for(auto j=i+1; j<bboxes_nr; ++j) {
                    output(i,j) = res(i,j);
                    output(j,i) = res(i,j);
                }
            }

            auto total_nr = 0;
            for(auto i=0; i<bboxes_nr; ++i) {
                for(auto j=0; j<bboxes_nr; ++j) {
                    if(output(i,j)>0)++total_nr;
                }
            }
            cout<<"Total edge number: "<<total_nr<<", boxes nr:"<<bboxes_nr<<endl<<endl;
        }
        template<typename _T,typename M>
            Eigen::Tensor<int,2,Eigen::RowMajor> make_graph(const _T& bboxes,const M& dis_m,float theta,float rate=1.0) {
                const auto data_nr = bboxes.dimension(0);
                Graph g(data_nr);
                auto weightmap = get(edge_weight,g);
                for(auto i=0; i<data_nr; ++i) {
                    for(auto j=i+1; j<data_nr; ++j) {
                        Edge e;
                        bool inserted;
                        boost::tie(e,inserted) = add_edge(i,j,g);
                        weightmap[e] = dis_m(i,j);
                    }
                }
                std::vector < Edge > spanning_tree;
                kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
                Eigen::Tensor<int,2,Eigen::RowMajor> res(data_nr,data_nr);
                res.setZero();
                for(auto e:spanning_tree) {
                    auto si = source(e,g);
                    auto ti = target(e,g);
                    res(si,ti) = 1;
                    res(ti,si) = 1;
                }
                vector<float> weight(data_nr);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::exponential_distribution<> d(1.0/rate);
                generate(weight.begin(),weight.end(),[&d,&gen]() {
                        return d(gen);
                        });
                for(auto i=0; i<data_nr; ++i) {
                    for(auto j=i+1; j<data_nr; ++j) {
                        const auto v = theta*dis_m(i,j)*dis_m(i,j);
                        if(weight[i]+weight[j]>v) {
                            res(i,j) = 1;
                            res(j,i) = 1;
                        }
                    }
                }
                return res;
            }
        template<typename _T>
            inline float distance(const _T& box0, const _T& box1) {
                float cx0,cy0,cx1,cy1;
                tie(cx0,cy0) = get_cxy(box0);
                tie(cx1,cy1) = get_cxy(box1);
                const auto iou = bboxes_jaccardv1(box0,box1);
                const auto dx = coord_scales_[0]*(cx0-cx1);
                const auto dy = coord_scales_[1]*(cy0-cy1);
                const auto dz = coord_scales_[2]*iou;
                return sqrt(dx*dx+dy*dy+dz*dz);
            }
        template<typename _T>
        static inline std::pair<float,float> get_cxy(const _T& box) {
            return make_pair((box(1)+box(3))/2.0f,(box(0)+box(2))/2.0f);
        }
    private:
        float theta_;
        float scale_;
        vector<float> coord_scales_;
};
REGISTER_KERNEL_BUILDER(Name("AdjacentMatrixGenerator").Device(DEVICE_CPU).TypeConstraint<float>("T"), AdjacentMatrixGeneratorOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AdjacentMatrixGenerator").Device(DEVICE_CPU).TypeConstraint<double>("T"), AdjacentMatrixGeneratorOp<CPUDevice, double>);

/*
 * 根据输入的boxes生成一个邻接矩阵
 * boxes: [N,4], ymin,xmin,ymax,xmax相对坐标
 * output:
 * matrix[N,N] 第i行,j列表示第i个点与第j个点之间的连接
 */
REGISTER_OP("AdjacentMatrixGeneratorByIou")
    .Attr("T: {float, double,int32}")
	.Attr("threshold:float")
    .Attr("keep_connect: bool")
    .Input("bboxes: T")
	.Output("matrix:int32")
	.SetShapeFn([](shape_inference::InferenceContext* c) {
		    auto box_nr = c->Dim(c->input(0),0);
			c->set_output(0, c->Matrix(box_nr,box_nr));
			return Status::OK();
			});

template <typename Device, typename T>
class AdjacentMatrixGeneratorByIouOp: public OpKernel {
    public:
        explicit AdjacentMatrixGeneratorByIouOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("threshold", &threshold_));
            OP_REQUIRES_OK(context, context->GetAttr("keep_connect", &keep_connect_));
        }

        void Compute(OpKernelContext* context) override
        {
            const Tensor &_bboxes  = context->input(0);
            auto          bboxes   = _bboxes.template tensor<T,2>();

            OP_REQUIRES(context, _bboxes.dims() == 2, errors::InvalidArgument("bboxes data must be 2-dimensional"));
            const auto bboxes_nr    = _bboxes.dim_size(0);
            Eigen::Tensor<float,2,Eigen::RowMajor> dis_matrix(bboxes_nr,bboxes_nr);

            dis_matrix.setZero();

            if(keep_connect_) {
                for(auto i=0; i<bboxes_nr-1; ++i) {
                    const Eigen::Tensor<T,1,Eigen::RowMajor> box_data0 = bboxes.chip(i,0);

                    dis_matrix(i,i) = 0.;
                    for(auto j=i+1; j<bboxes_nr; ++j) {
                        const Eigen::Tensor<T,1,Eigen::RowMajor> box_data1 = bboxes.chip(j,0);
                        const auto dis = distance(box_data0,box_data1);
                        dis_matrix(i,j) = dis;
                        dis_matrix(j,i) = dis;
                    }
                }
            }

            auto res = make_graph(bboxes,dis_matrix);

            Tensor      *output_matrix = NULL;
            TensorShape  output_shape;
            const int    dims_2d[]     = {int(bboxes_nr),int(bboxes_nr)};

            TensorShapeUtils::MakeShape(dims_2d, 2, &output_shape);
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_matrix));

            auto output           = output_matrix->template tensor<int,2>();
            for(auto i=0; i<bboxes_nr; ++i) {
                output(i,i) = 0;
                for(auto j=i+1; j<bboxes_nr; ++j) {
                    output(i,j) = res(i,j);
                    output(j,i) = res(i,j);
                }
            }

            auto total_nr = 0;
            for(auto i=0; i<bboxes_nr; ++i) {
                for(auto j=0; j<bboxes_nr; ++j) {
                    if(output(i,j)>0)++total_nr;
                }
            }
        }
        template<typename _T,typename M>
            Eigen::Tensor<int,2,Eigen::RowMajor> make_graph(const _T& bboxes,const M& dis_m) {
                const auto data_nr = bboxes.dimension(0);
                Graph g(data_nr);
                Eigen::Tensor<int,2,Eigen::RowMajor> res(data_nr,data_nr);
                res.setZero();

                if(keep_connect_) {
                    auto weightmap = get(edge_weight,g);
                    for(auto i=0; i<data_nr; ++i) {
                        for(auto j=i+1; j<data_nr; ++j) {
                            Edge e;
                            bool inserted;
                            boost::tie(e,inserted) = add_edge(i,j,g);
                            weightmap[e] = dis_m(i,j);
                        }
                    }
                    std::vector < Edge > spanning_tree;
                    kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
                    for(auto e:spanning_tree) {
                        auto si = source(e,g);
                        auto ti = target(e,g);
                        res(si,ti) = 1;
                        res(ti,si) = 1;
                    }
                }
                for(auto i=0; i<data_nr; ++i) {
                    const Eigen::Tensor<T,1,Eigen::RowMajor> box_data0 = bboxes.chip(i,0);
                    for(auto j=i+1; j<data_nr; ++j) {
                        const Eigen::Tensor<T,1,Eigen::RowMajor> box_data1 = bboxes.chip(j,0);
                        //if(bboxes_jaccardv1(box_data0,box_data1)>threshold_) {
                        if((bboxes_jaccard_of_box0v1(box_data0,box_data1)>threshold_) ||
                                (bboxes_jaccard_of_box0v1(box_data1,box_data0)>threshold_)) {
                            res(i,j) = 1;
                            res(j,i) = 1;
                        }
                    }
                }
                return res;
            }
        template<typename _T>
            inline float distance(const _T& box0, const _T& box1) {
                float cx0,cy0,cx1,cy1;
                tie(cx0,cy0) = get_cxy(box0);
                tie(cx1,cy1) = get_cxy(box1);
                const auto dx = (cx0-cx1);
                const auto dy = (cy0-cy1);
                return sqrt(dx*dx+dy*dy);
            }
        template<typename _T>
        static inline std::pair<float,float> get_cxy(const _T& box) {
            return make_pair((box(1)+box(3))/2.0f,(box(0)+box(2))/2.0f);
        }
    private:
        float threshold_ = 0.3f;
        bool keep_connect_ = false;
};
REGISTER_KERNEL_BUILDER(Name("AdjacentMatrixGeneratorByIou").Device(DEVICE_CPU).TypeConstraint<float>("T"), AdjacentMatrixGeneratorByIouOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AdjacentMatrixGeneratorByIou").Device(DEVICE_CPU).TypeConstraint<double>("T"), AdjacentMatrixGeneratorByIouOp<CPUDevice, double>);
