#include "jde_tracker.h"
#include <set>
#include "mot_matching.h"
#include "track_toolkit.h"

using namespace std;
using namespace MOT;

namespace MOT {
JDETracker::JDETracker(float det_thresh,int frame_rate,int track_buffer)
:det_thresh_(det_thresh)
,frame_rate_(frame_rate)
,track_buffer_(track_buffer)
,buffer_size_(int(frame_rate*track_buffer/30))
{
    kalman_filter_ = make_shared<KalmanFilter>();
}
STrackPtrs_t JDETracker::update(const BBoxes_t& bboxes,const Probs_t& probs, const Embeddings_t& embds)
{
    STrackPtrs_t activated_stracks,refind_stracks,lost_stracks,removed_stracks,detections;
    STrackPtrs_t unconfirmed,tracked_stracks;

    frame_id_ += 1;
    
    for(auto i=0; i<bboxes.rows(); ++i) {
        detections.emplace_back(make_shared<STrack>(bboxes.block<1,4>(i,0).transpose(),
                    probs(i,0),embds.block(i,0,1,embds.cols()).transpose(),buffer_size_));
    }

    //Add newly detected tracklets to tracked_stracks
    for(auto track:tracked_stracks_) {
        if(track->is_activated())
            tracked_stracks.push_back(track);
         else
             unconfirmed.push_back(track);
    }

    //Step 2: First association, with embedding
    auto strack_pool = joint_stracks(tracked_stracks,lost_stracks_);

    STrack::multi_predict(strack_pool);

    auto dists = embedding_distance(strack_pool,detections);

    fuse_motion(kalman_filter_,strack_pool,detections,dists);

    vector<pair<int,int>> matches;
    vector<int> u_track,u_detection;
    int itracked,idet;

    linear_assignment(dists,0.4,&matches,&u_track,&u_detection);

    for(auto& _d:matches) {
        tie(itracked,idet) = _d;
        auto& track = strack_pool[itracked];
        auto& det = detections[idet];

        if(track->state() == STrack::TRACKED) {
            track->update(*det,frame_id_);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(*det,frame_id_,false);
            refind_stracks.push_back(track);
        }
    }

    //Step 3: Second association, with IOU
    inplace_gather(detections,u_detection);
    auto r_tracked_stracks = gather(strack_pool,u_track,[](const STrackPtr_t& t){ return t->state()==STrack::TRACKED;});

    lost_stracks = gather(strack_pool,u_track,[](const STrackPtr_t& t){ return t->state()!=STrack::TRACKED;});

    dists = iou_distance(r_tracked_stracks,detections);
    linear_assignment(dists,0.5,&matches,&u_track,&u_detection);

    for(auto& _d:matches) {
        tie(itracked,idet) = _d;
        auto& track = r_tracked_stracks[itracked];
        auto& det = detections[idet];

        if(track->state() == STrack::TRACKED) {
            track->update(*det,frame_id_);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(*det,frame_id_,false);
            refind_stracks.push_back(track);
        }
    }

    for(auto it:u_track) {
        auto& track = r_tracked_stracks[it];
        if(!(track->state()==STrack::LOST)) {
            track->mark_lost();
            lost_stracks.push_back(track);
        }
    }

    //Deal with unconfirmed tracks, usually tracks with only one beginning frame
    vector<int> u_unconfirmed;

    inplace_gather(detections,u_detection);
    dists = iou_distance(unconfirmed,detections);
    linear_assignment(dists,0.7,&matches,&u_track,&u_detection);
    for(auto& _d:matches) {
        tie(itracked,idet) = _d;
        auto& track = unconfirmed[itracked];
        auto& det = detections[idet];

        track->update(*det,frame_id_);
        activated_stracks.push_back(track);
    }
    for(auto it:u_unconfirmed) {
        auto& track = unconfirmed[it];
        track->mark_removed();
        removed_stracks.push_back(track);
    }

    //Step 4: Init new stracks
    for(auto inew:u_detection) {
        auto& track = detections[inew];
        if(track->score() < det_thresh_) 
            continue;
        track->activate(kalman_filter_,frame_id_);
        activated_stracks.push_back(track);
    }

    //Step 5: Update state
    /*
     * the follow line is not included original python code, since some tracks in lost_stracks_ become activated, so only tracks still 
     * in lost state will be removed
     */
    lost_stracks_ = lost_stracks;
    for(auto& track:lost_stracks_) {
        if(frame_id_-track->end_frame()>max_time_lost_) {
            track->mark_removed();
            removed_stracks.push_back(track);
        }
    }

    tracked_stracks_ = joint_stracks(activated_stracks,refind_stracks);
    removed_stracks_ = joint_stracks(removed_stracks_,removed_stracks);

    return tracked_stracks_;

}
STrackPtrs_t JDETracker::update(const BBoxes_t& bboxes,const Probs_t& probs)
{
    STrackPtrs_t activated_stracks,refind_stracks,lost_stracks,removed_stracks,detections;
    STrackPtrs_t unconfirmed,tracked_stracks;

    frame_id_ += 1;
    
    for(auto i=0; i<bboxes.rows(); ++i) {
        detections.emplace_back(make_shared<STrack>(bboxes.block<1,4>(i,0).transpose(),
                    probs(i,0),Eigen::VectorXf(),buffer_size_));
    }

    //Add newly detected tracklets to tracked_stracks
    for(auto track:tracked_stracks_) {
        if(track->is_activated())
            tracked_stracks.push_back(track);
         else
             unconfirmed.push_back(track);
    }

    //Step 2: predict
    auto strack_pool = joint_stracks(tracked_stracks,lost_stracks_);

    STrack::multi_predict(strack_pool);

    //Step 3: Second association, with IOU
    vector<pair<int,int>> matches;
    vector<int> u_track,u_detection;
    STrackPtrs_t r_tracked_stracks;
    int itracked,idet;

    copy_if(strack_pool.begin(),strack_pool.end(),back_inserter(r_tracked_stracks),[](const STrackPtr_t& t){ return t->state()==STrack::TRACKED;});
    copy_if(strack_pool.begin(),strack_pool.end(),back_inserter(lost_stracks),[](const STrackPtr_t& t){ return t->state()!=STrack::TRACKED;});

    auto dists = iou_distance(r_tracked_stracks,detections);
    linear_assignment(dists,0.5,&matches,&u_track,&u_detection);

    for(auto& _d:matches) {
        tie(itracked,idet) = _d;
        auto& track = r_tracked_stracks[itracked];
        auto& det = detections[idet];

        if(track->state() == STrack::TRACKED) {
            track->update(*det,frame_id_);
            activated_stracks.push_back(track);
        } else {
            track->re_activate(*det,frame_id_,false);
            refind_stracks.push_back(track);
        }
    }

    for(auto it:u_track) {
        auto& track = r_tracked_stracks[it];
        if(!(track->state()==STrack::LOST)) {
            track->mark_lost();
            lost_stracks.push_back(track);
        }
    }

    //Deal with unconfirmed tracks, usually tracks with only one beginning frame
    vector<int> u_unconfirmed;

    inplace_gather(detections,u_detection);
    dists = iou_distance(unconfirmed,detections);
    linear_assignment(dists,0.7,&matches,&u_track,&u_detection);
    for(auto& _d:matches) {
        tie(itracked,idet) = _d;
        auto& track = unconfirmed[itracked];
        auto& det = detections[idet];

        track->update(*det,frame_id_);
        activated_stracks.push_back(track);
    }
    for(auto it:u_unconfirmed) {
        auto& track = unconfirmed[it];
        track->mark_removed();
        removed_stracks.push_back(track);
    }

    //Step 4: Init new stracks
    for(auto inew:u_detection) {
        auto& track = detections[inew];
        if(track->score() < det_thresh_) 
            continue;
        track->activate(kalman_filter_,frame_id_);
        activated_stracks.push_back(track);
    }

    //Step 5: Update state
    /*
     * the follow line is not included original python code, since some tracks in lost_stracks_ become activated, so only tracks still 
     * in lost state will be removed
     */
    lost_stracks_ = lost_stracks;
    for(auto& track:lost_stracks_) {
        if(frame_id_-track->end_frame()>max_time_lost_) {
            track->mark_removed();
            removed_stracks.push_back(track);
        }
    }

    tracked_stracks_ = joint_stracks(activated_stracks,refind_stracks);
    removed_stracks_ = joint_stracks(removed_stracks_,removed_stracks);

    return tracked_stracks_;

}
STrackPtrs_t JDETracker::joint_stracks(STrackPtrs_t& va,STrackPtrs_t& vb)
{
    STrackPtrs_t res;
    set<int> ids;

    res.insert(res.end(),va.begin(),va.end());

    for(auto& v:va) {
        ids.insert(v->track_id());
    }
    copy_if(vb.begin(),vb.end(),back_inserter(res),[&ids](const STrackPtr_t& v) {
        return ids.find(v->track_id())==ids.end();
    });

    return res;
}
}
