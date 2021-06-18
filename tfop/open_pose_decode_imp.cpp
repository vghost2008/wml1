#include<opencv2/imgproc.hpp>
#include "open_pose_decode_imp.h"
#include <opencv2/opencv.hpp>
#include<iostream>
#include<chrono>
#include<random>
#include<set>
#include<cmath>
using namespace std;
using namespace cv;

////////////////////////////////
struct WKeyPoint{
	WKeyPoint(cv::Point point,float probability){
		this->id = -1;
		this->point = point;
		this->probability = probability;
	}

	int id;
	cv::Point point;
	float probability;
};

std::ostream& operator << (std::ostream& os, const WKeyPoint& kp)
{
	os << "Id:" << kp.id << ", Point:" << kp.point << ", Prob:" << kp.probability << std::endl;
	return os;
}

////////////////////////////////
struct ValidPair{
	ValidPair(int aId,int bId,float score){
		this->aId = aId;
		this->bId = bId;
		this->score = score;
	}

	int aId;
	int bId;
	float score;
};

std::ostream& operator << (std::ostream& os, const ValidPair& vp)
{
	os << "A:" << vp.aId << ", B:" << vp.bId << ", score:" << vp.score << std::endl;
	return os;
}

////////////////////////////////

template < class T > std::ostream& operator << (std::ostream& os, const std::vector<T>& v)
{
    os << "[";
	bool first = true;
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
    {
		if(!first) os << ",";
        os << " " << *ii;
    }
    os << "]";
    return os;
}

template < class T > std::ostream& operator << (std::ostream& os, const std::set<T>& v)
{
    os << "[";
	bool first = true;
    for (typename std::set<T>::const_iterator ii = v.begin(); ii != v.end(); ++ii, first = false)
    {
		if(!first) os << ",";
        os << " " << *ii;
    }
    os << "]";
    return os;
}

/*
 * probMap:[H,W]
 *
 */
void getWKeyPoints(cv::Mat& probMap,double threshold,std::vector<WKeyPoint>& keyPoints){
	cv::Mat smoothProbMap;
	cv::GaussianBlur( probMap, smoothProbMap, cv::Size( 3, 3 ), 0, 0 );

	cv::Mat maskedProbMap;
	cv::threshold(smoothProbMap,maskedProbMap,threshold,255,cv::THRESH_BINARY);

	maskedProbMap.convertTo(maskedProbMap,CV_8U,1);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(maskedProbMap,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size();++i){
		cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows,smoothProbMap.cols,smoothProbMap.type());

		cv::fillConvexPoly(blobMask,contours[i],cv::Scalar(1));

		double maxVal;
		cv::Point maxLoc;

		cv::minMaxLoc(smoothProbMap.mul(blobMask),0,&maxVal,0,&maxLoc);

		keyPoints.push_back(WKeyPoint(maxLoc, probMap.at<float>(maxLoc.y,maxLoc.x)));
	}
}

void populateColorPalette(std::vector<cv::Scalar>& colors,int nColors){
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis1(64, 200);
    std::uniform_int_distribution<> dis2(100, 255);
    std::uniform_int_distribution<> dis3(100, 255);

	for(int i = 0; i < nColors;++i){
		colors.push_back(cv::Scalar(dis1(gen),dis2(gen),dis3(gen)));
	}
}

/*
 * netOutputBlob: [N,h,w]
 * outputs:
 * vector<Mat>:Nx[H,W]
 */
vector<Mat> splitNetOutputBlobToParts(const cv::Mat& netOutputBlob){
    vector<Mat> netOutputParts;
	int nParts = netOutputBlob.size[0];
	int h = netOutputBlob.size[1];
	int w = netOutputBlob.size[2];

	for(int i = 0; i< nParts;++i){
		cv::Mat part(h, w, CV_32F, (void*)netOutputBlob.ptr(i));
		netOutputParts.push_back(part);
	}
    return netOutputParts;
}

void populateInterpPoints(const cv::Point& a,const cv::Point& b,int numPoints,std::vector<cv::Point>& interpCoords){
	float xStep = ((float)(b.x - a.x))/(float)(numPoints-1);
	float yStep = ((float)(b.y - a.y))/(float)(numPoints-1);

	interpCoords.push_back(a);

	for(int i = 1; i< numPoints-1;++i){
		interpCoords.push_back(cv::Point(a.x + xStep*i,a.y + yStep*i));
	}

	interpCoords.push_back(b);
}


/*
 * map_idx: corresponding map id of pos_pairs, map_idx.size()==pos_pairs.size()
 * valid_pairs: return valid key points pairs, valid_pairs.size()==pos_pairs.size(), valid_pairs[k] include all valid pair keypoints of pos_pairs[i]
 *              for example neck -> right-shoulder
 * invalid_pairs: if valid_pairs[k].empty(), then k will be insert to invalid_pairs, invalid_pairs.size()<=pos_pairs.size()
 */
void getValidPairs(const cv::Mat& _paf_map,
				   const std::vector<std::vector<WKeyPoint>>& detectedKeypoints,
                   const vector<pair<int,int>>& map_idx,
                   const vector<pair<int,int>>& pose_pairs,
				   std::vector<std::vector<ValidPair>>& valid_pairs,
				   std::set<int>& invalid_pairs,
                   const int interp_samples=10,
                   const float paf_score_th=0.1,
                   const float conf_th=0.7) {


	auto paf_map = splitNetOutputBlobToParts(_paf_map);

	for(int k = 0; k < map_idx.size();++k ){

		//A->B constitute a limb
		cv::Mat pafX = paf_map[map_idx[k].first];
		cv::Mat pafY = paf_map[map_idx[k].second];
		//Find the keypoints for the first and second limb
		const std::vector<WKeyPoint>& candA = detectedKeypoints[pose_pairs[k].first];
		const std::vector<WKeyPoint>& candB = detectedKeypoints[pose_pairs[k].second];
		int nA = candA.size();
		int nB = candB.size();

		/*
		  # If keypoints for the joint-pair is detected
		  # check every joint in candA with every joint in candB
		  # Calculate the distance vector between the two joints
		  # Find the PAF values at a set of interpolated points between the joints
		  # Use the above formula to compute a score to mark the connection valid
		 */

		if(nA != 0 && nB != 0){
			std::vector<ValidPair> localValidPairs; //all key points pair for pos_pairs[k]

			for(int i = 0; i< nA;++i){
				int         maxJ     = -1;
				float       maxScore = -1;
				bool        found    = false;
				vector<int> used_B;

				for(int j = 0; j < nB;++j){
                    if(find(used_B.begin(),used_B.end(),j) != used_B.end())
                        continue;
					std::pair<float,float> distance(candB[j].point.x - candA[i].point.x,candB[j].point.y - candA[i].point.y);

					float norm = std::sqrt(distance.first*distance.first + distance.second*distance.second);

					if(!norm){
						continue;
					}

					distance.first /= norm;
					distance.second /= norm;

					//Find p(u)
					std::vector<cv::Point> interpCoords;
					populateInterpPoints(candA[i].point,candB[j].point,interp_samples,interpCoords);
					//Find L(p(u))
					std::vector<std::pair<float,float>> pafInterp;
					for(int l = 0; l < interpCoords.size();++l){
						pafInterp.push_back(
							std::pair<float,float>(
								pafX.at<float>(interpCoords[l].y,interpCoords[l].x),
								pafY.at<float>(interpCoords[l].y,interpCoords[l].x)
							));
					}

					std::vector<float> pafScores;
					float sumOfPafScores = 0;
					int numOverTh = 0;
					for(int l = 0; l< pafInterp.size();++l){
						float score = pafInterp[l].first*distance.first + pafInterp[l].second*distance.second;
						sumOfPafScores += score;
						if(score > paf_score_th){
							++numOverTh;
						}

						pafScores.push_back(score);
					}

					float avgPafScore = sumOfPafScores/((float)pafInterp.size());

					if(((float)numOverTh)/((float)interp_samples) > conf_th){
						if(avgPafScore > maxScore){
							maxJ = j;
							maxScore = avgPafScore;
							found = true;
						}
					}

				}/* j */

				if(found){
                    used_B.push_back(maxJ);
					localValidPairs.push_back(ValidPair(candA[i].id,candB[maxJ].id,maxScore));
				}

			}/* i */

			valid_pairs.push_back(localValidPairs);

		} else {
			invalid_pairs.insert(k);
			valid_pairs.push_back(std::vector<ValidPair>());
		}
	}/* k */
}

auto getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>>& valid_pairs,
							const std::set<int>& invalid_pairs,
                            const vector<pair<int,int>>& map_idx,
                            const vector<pair<int,int>>& pose_pairs,
                            const int points_nr) {
    std::vector<std::vector<int>> person_wise_keypoints;
	for(int k = 0; k < map_idx.size();++k){
		if(invalid_pairs.find(k) != invalid_pairs.end()){
			continue;
		}

		const std::vector<ValidPair>& localValidPairs(valid_pairs[k]);

		int indexA(pose_pairs[k].first);
		int indexB(pose_pairs[k].second);

		for(int i = 0; i< localValidPairs.size();++i){
			bool found = false;
			int personIdx = -1;

			for(int j = 0; !found && j < person_wise_keypoints.size();++j){
				if(indexA < person_wise_keypoints[j].size() &&
				   person_wise_keypoints[j][indexA] == localValidPairs[i].aId){
					personIdx = j;
					found = true;
				}
			}/* j */

			if(found){
				person_wise_keypoints[personIdx].at(indexB) = localValidPairs[i].bId;
			} else {
				std::vector<int> lpkp(std::vector<int>(points_nr,-1));

				lpkp.at(indexA) = localValidPairs[i].aId;
				lpkp.at(indexB) = localValidPairs[i].bId;

				person_wise_keypoints.push_back(lpkp);
			}

		}/* i */
	}/* k */
    return person_wise_keypoints;
}


vector<vector<WKeyPoint>> get_key_points(const cv::Mat& paf_map,const float keypoints_th=0.1)
{
	int keyPointId = 0;
	std::vector<std::vector<WKeyPoint>> detectedKeypoints;
    const auto points_nr = paf_map.size[0];

	auto netOutputParts = splitNetOutputBlobToParts(paf_map);

	for(int i = 0; i < points_nr;++i){
		std::vector<WKeyPoint> keyPoints;

		getWKeyPoints(netOutputParts[i],keypoints_th,keyPoints);

		for(int i = 0; i< keyPoints.size();++i,++keyPointId){
			keyPoints[i].id = keyPointId;
		}

		detectedKeypoints.push_back(keyPoints);
	}
    return detectedKeypoints;
}
vector<vector<pair<float,float>>> openpose_decode_imp(const cv::Mat& conf_map,const cv::Mat& paf_map,
        const vector<pair<int,int>>& map_idx,
        const vector<pair<int,int>>& pose_pairs,
        const float keypoints_th,
        const int interp_samples,
        const float paf_score_th,
        const float conf_th) 
{
    auto detectedKeypoints = get_key_points(conf_map,keypoints_th);

    std::vector<WKeyPoint> key_points_list;

    for(auto& ps:detectedKeypoints)
		key_points_list.insert(key_points_list.end(),ps.begin(),ps.end());

    const auto points_nr = paf_map.size[0];
    std::vector<std::vector<ValidPair>> valid_pairs;
    std::set<int> invalid_pairs;

    getValidPairs(paf_map,detectedKeypoints,map_idx,pose_pairs,valid_pairs,invalid_pairs,
            interp_samples,
            paf_score_th,
            conf_th);

    auto _person_wise_keypoints = getPersonwiseKeypoints(valid_pairs,invalid_pairs,
            map_idx,
            pose_pairs,
            points_nr);
    vector<vector<pair<float,float>>> person_wise_keypoints;
    for(auto ds:_person_wise_keypoints) {
        vector<pair<float,float>> lpoints;
        lpoints.reserve(ds.size());
        for(auto id:ds) {
            if(id>=0) {
                const auto p = key_points_list[id];
                lpoints.emplace_back(p.point.x,p.point.y);
            } else {
                lpoints.emplace_back(-1,-1);
            }
        }
        person_wise_keypoints.emplace_back(std::move(lpoints));
    }
    return person_wise_keypoints;
}
