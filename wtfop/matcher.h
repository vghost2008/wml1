_Pragma("once")
#include "bboxes.h"
#include "wmacros.h"

template <typename Device,typename T>
class MatcherUnit {
};
template <typename T>
class MatcherUnit<Eigen::ThreadPoolDevice,T> {
	public:
		struct IOUIndex{
			int index; //与当前box对应的gbbox序号
			float iou;//与其对应的gbbox的IOU
		};
		explicit MatcherUnit(float pos_threshold,float neg_threshold,bool max_overlap_as_pos=true) 
			:pos_threshold_(pos_threshold)
			,neg_threshold_(neg_threshold)
            ,max_overlap_as_pos_(max_overlap_as_pos){
			 }
#ifdef PROCESS_BOUNDARY_ANCHORS
        template<typename _T>
        inline bool is_cross_boundaries(const _T& box) {
            return (box(0)<0.0) || (box(1)<0.0) || (box(2)>1.0) ||(box(3)>1.0);
        }
#endif
		   auto operator()(
		   const Eigen::Tensor<T,2,Eigen::RowMajor>& boxes,
		   const Eigen::Tensor<T,2,Eigen::RowMajor>& gboxes,
		   const Eigen::Tensor<int,1,Eigen::RowMajor>& glabels)
			{
                int                   data_nr              = boxes.dimension(0);
                int                   gdata_nr             = gboxes.dimension(0);
                auto                  out_labels           = Eigen::Tensor<int,1,Eigen::RowMajor>(data_nr);
                auto                  out_scores           = Eigen::Tensor<T,1,Eigen::RowMajor>(data_nr);
                auto                  outindex             = Eigen::Tensor<int,1,Eigen::RowMajor>(data_nr);
                std::vector<bool>     is_max_score(data_nr   ,false);
                std::vector<IOUIndex> iou_indexs(data_nr     ,IOUIndex({-1,0.0}));                          //默认box不与任何ground truth box相交，iou为0

				for(auto i=0; i<data_nr; ++i) {
					out_labels(i) = 0;  //默认所有的都为背景
					out_scores(i) = 0;
                    outindex(i) = -1;
				}
                const auto kThreadNr = std::max(1,std::min({40,gdata_nr*data_nr/20000,gdata_nr}));
                const auto kDataPerThread = gdata_nr/kThreadNr;
				/*
				 * 遍历每一个ground truth box
				 */
                std::mutex mtx;
                std::list<std::future<void>> results;
                auto shard = [&](int start,int limit) {
				for(auto i=start; i<limit; ++i) {
					const      Eigen::Tensor<T,1,Eigen::RowMajor> gbox= gboxes.chip(i,0);
					const auto glabel          = glabels(i);
					auto       max_index       = -1;
					auto       max_scores      = -1.0;

					/*
					 * 计算ground truth box与每一个候选box的jaccard得分
					 */
					for(auto j=0; j<data_nr; ++j) {
						const Eigen::Tensor<T,1,Eigen::RowMajor> box       = boxes.chip(j,0);
#ifdef PROCESS_BOUNDARY_ANCHORS
                        /*
                         * Faster-RCNN原文认为边界框上的bounding box不仔细处理会引起很大的不会收敛的误差
                         * 在Detectron2的实现中默认并没有区别边界情况
                         */
                        if(is_cross_boundaries(box)) continue;
#endif

						auto        jaccard   = bboxes_jaccardv1(gbox,box);

						if(jaccard<1E-8) continue;

						if(jaccard>max_scores) {
							max_scores = jaccard;
							max_index = j;
						}

						auto       &iou_index = iou_indexs[j];

                        {
                            std::lock_guard<std::mutex> g(mtx);
                            if(jaccard>iou_index.iou) {
                                iou_index.iou = jaccard;
                                iou_index.index = i;
                                out_scores(j)  =  jaccard;
                                out_labels(j)  =  glabel;
                                outindex(j)    =  i;
                            }
                        }
					}
					if(max_scores<1E-8) continue;
					/*
					 * 面积交叉最大的给于标签
					 */
					auto j = max_index;

                    if(max_overlap_as_pos_) {
                        std::lock_guard<std::mutex> g(mtx);
                        is_max_score[j] = true;
                    }
				} //end for
                };
                for(auto i=0; i<gdata_nr; i+=kDataPerThread) {
                    results.emplace_back(std::async(std::launch::async,[i,gdata_nr,kDataPerThread,&shard](){shard(i,std::min(gdata_nr,i+kDataPerThread));}));
                }
                results.clear();

				for(auto j=0; j<data_nr; ++j) {
					const auto& iou_index = iou_indexs[j];
					if((iou_index.iou>=pos_threshold_) || is_max_score[j]) {
						continue;
					} else if(iou_index.iou<neg_threshold_) {
                        outindex(j) = -1;
                        out_labels(j) = 0;
					} else {
                        outindex(j) = -1;
                        out_labels(j) = -1;
                    }
				}

				return std::make_tuple(out_labels,out_scores,outindex);
			}
	private:
		const float              pos_threshold_;
		const float              neg_threshold_;
		bool                     max_overlap_as_pos_ = true;
};
#ifdef GOOGLE_CUDA
void matcher_by_gpu(const float* gbboxes,const float* anchor_bboxes,const int* glabels,
float* out_scores,int* out_labels,int* out_index,
size_t gb_size,size_t ab_size,float neg_threshold,float pos_threshold,bool max_overlap_as_pos);
template <typename T>
class MatcherUnit<Eigen::GpuDevice,T> {
    public:
        explicit MatcherUnit(float pos_threshold,float neg_threshold,bool max_overlap_as_pos) 
            :pos_threshold_(pos_threshold)
             ,neg_threshold_(neg_threshold)
             ,max_overlap_as_pos_(max_overlap_as_pos){
             }
        void operator()(
                const T* boxes,const T* gboxes,const int* glabels,
                float* out_bboxes,int* out_labels,float* out_scores,bool* out_remove_indict,int* out_index,
                int gdata_nr,int data_nr
                )
        {
            matcher_by_gpu(gboxes,boxes,glabels,
                    out_scores,out_labels,out_index,
                    gdata_nr,data_nr,neg_threshold_,pos_threshold_,max_overlap_as_pos_);
        }
    private:
        const float              pos_threshold_;
        const float              neg_threshold_;
        bool                     max_overlap_as_pos_ = true;
};
#else
//#error "No cuda support"
#endif
