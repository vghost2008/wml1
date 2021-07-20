#pragma once
#include "bboxes.h"

template <typename T,typename T1>
auto encode_one_boxes(const Eigen::Tensor<T,1,Eigen::RowMajor>& gbox,const Eigen::Tensor<T,1,Eigen::RowMajor>& ref_box,const T1& prio_scaling) {
    Eigen::Tensor<T,1,Eigen::RowMajor> out_box(4);
    auto &feat_cy = out_box(0);
    auto &feat_cx = out_box(1);
    auto &feat_h  = out_box(2);
    auto &feat_w  = out_box(3);
    auto  yxhw    = box_minmax_to_cxywh(ref_box.data());
    auto  yref    = std::get<0>(yxhw);
    auto  xref    = std::get<1>(yxhw);
    auto  href    = std::get<2>(yxhw);
    auto  wref    = std::get<3>(yxhw);

    if((href<1E-8) || (wref<1E-8)) {
        feat_cy = feat_cx = feat_h = feat_w = 0.0;
        return out_box;
    }

    auto gyxhw = box_minmax_to_cxywh(gbox.data());

    feat_cy  =  std::get<0>(gyxhw);
    feat_cx  =  std::get<1>(gyxhw);
    feat_h   =  std::get<2>(gyxhw);
    feat_w   =  std::get<3>(gyxhw);

    feat_cy  =  (feat_cy-yref)/(href*prio_scaling[0]);
    feat_cx  =  (feat_cx-xref)/(wref*prio_scaling[1]);
    feat_h   =  log(feat_h/href)/prio_scaling[2];
    feat_w   =  log(feat_w/wref)/prio_scaling[3];
    return out_box;
}
template <typename Device,typename T>
class BoxesEncodeUnit {
};
template <typename T>
class BoxesEncodeUnit<Eigen::ThreadPoolDevice,T> {
	public:
		struct IOUIndex{
			int index; //与当前box对应的gbbox序号
			float iou;//与其对应的gbbox的IOU
		};
		explicit BoxesEncodeUnit(float pos_threshold,float neg_threshold,const std::vector<float>& prio_scaling,bool max_overlap_as_pos=true) 
			:pos_threshold_(pos_threshold)
			 ,neg_threshold_(neg_threshold)
			 ,prio_scaling_(prio_scaling)
             ,max_overlap_as_pos_(max_overlap_as_pos){
				 assert(prio_scaling_.size() == 4);
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
                auto                  out_boxes            = Eigen::Tensor<T,2,Eigen::RowMajor>(data_nr,4);
                auto                  out_labels           = Eigen::Tensor<int,1,Eigen::RowMajor>(data_nr);
                auto                  out_scores           = Eigen::Tensor<T,1,Eigen::RowMajor>(data_nr);
                auto                  out_remove_indices   = Eigen::Tensor<bool,1,Eigen::RowMajor>(data_nr);
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

						if(jaccard<MIN_SCORE_FOR_POS_BOX) continue;

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
					if(max_scores<MIN_SCORE_FOR_POS_BOX) continue;
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
						out_remove_indices(j) = false;
					} else if(iou_index.iou<neg_threshold_) {
						out_remove_indices(j) = false;
                        outindex(j) = -1;
                        out_labels(j) = 0;
					} else {
						out_remove_indices(j) = true;
                        outindex(j) = -1;
                        out_labels(j) = 0;
                    }
				}

				/*
				 * 计算所有正样本gbox到所对应的box的回归参数
				 */
				for(auto j=0; j<data_nr; ++j) {
					auto  outbox  = out_boxes.chip(j,0);

					if((out_labels(j)<1) || (out_remove_indices(j))) {
                        outbox.setZero();
						continue;
					}
					auto &feat_cy = out_boxes(j,0);
					auto &feat_cx = out_boxes(j,1);
					auto &feat_h  = out_boxes(j,2);
					auto &feat_w  = out_boxes(j,3);
					Eigen::Tensor<T,1,Eigen::RowMajor> box     = boxes.chip(j,0);
					Eigen::Tensor<T,1,Eigen::RowMajor> gbox    = gboxes.chip(outindex(j),0);
					auto  yxhw    = box_minmax_to_cxywh(box.data());
					auto  yref    = std::get<0>(yxhw);
					auto  xref    = std::get<1>(yxhw);
					auto  href    = std::get<2>(yxhw);
					auto  wref    = std::get<3>(yxhw);

					if((href<1E-8) || (wref<1E-8)) {
						feat_cy = feat_cx = feat_h = feat_w = 0.0;
						continue;
					}

					auto gyxhw = box_minmax_to_cxywh(gbox.data());

					feat_cy  =  std::get<0>(gyxhw);
					feat_cx  =  std::get<1>(gyxhw);
					feat_h   =  std::get<2>(gyxhw);
					feat_w   =  std::get<3>(gyxhw);

					feat_cy  =  (feat_cy-yref)/(href*prio_scaling_[0]);
					feat_cx  =  (feat_cx-xref)/(wref*prio_scaling_[1]);
					feat_h   =  log(feat_h/href)/prio_scaling_[2];
					feat_w   =  log(feat_w/wref)/prio_scaling_[3];
				}
				return std::make_tuple(out_boxes,out_labels,out_scores,out_remove_indices,outindex);
			}
	private:
		const float              pos_threshold_;
		const float              neg_threshold_;
		const std::vector<float> prio_scaling_;
		bool                     max_overlap_as_pos_ = true;
};
#ifdef GOOGLE_CUDA
void get_encodes(const float* gbboxes,const float* anchor_bboxes,const int* glabels,
float* out_boxes,float* out_scores,int* out_labels,bool* out_remove_indices,int* out_index,const float* prio_scaling,
size_t gb_size,size_t ab_size,float neg_threshold,float pos_threshold,bool max_overlap_as_pos);
template <typename T>
class BoxesEncodeUnit<Eigen::GpuDevice,T> {
    public:
        explicit BoxesEncodeUnit(float pos_threshold,float neg_threshold,const std::vector<float>& prio_scaling,bool max_overlap_as_pos) 
            :pos_threshold_(pos_threshold)
             ,neg_threshold_(neg_threshold)
             ,prio_scaling_(prio_scaling)
             ,max_overlap_as_pos_(max_overlap_as_pos){
                 assert(prio_scaling_.size() == 4);
             }
        void operator()(
                const T* boxes,const T* gboxes,const int* glabels,
                float* out_bboxes,int* out_labels,float* out_scores,bool* out_remove_indict,int* out_index,
                int gdata_nr,int data_nr
                )
        {
            get_encodes(gboxes,boxes,glabels,
                    out_bboxes,out_scores,out_labels,out_remove_indict,out_index,prio_scaling_.data(),
                    gdata_nr,data_nr,neg_threshold_,pos_threshold_,max_overlap_as_pos_);
        }
    private:
        const float              pos_threshold_;
        const float              neg_threshold_;
        const std::vector<float> prio_scaling_;
        bool                     max_overlap_as_pos_ = true;
};
void bboxes_decode_by_gpu(const float* anchor_bboxes,const float* regs,const float* prio_scaling,float* out_bboxes,size_t data_nr);
#else
//#error "No cuda support"
#endif
