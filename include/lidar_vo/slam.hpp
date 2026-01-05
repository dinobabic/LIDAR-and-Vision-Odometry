#pragma once
#include "lidar_vo/utils_slam.hpp"
#include "lidar_vo/bundle_adjustment.hpp"

class SLAM
{
public:
    SLAM() {}

    SLAM(const Eigen::Matrix3f &_K, const Eigen::Matrix<float, 3, 4> &_P0, const Eigen::Matrix<float, 3, 4> &_P1, float _baseline) :
            K(_K),
            P0(_P0),
            P1(_P1),
            baseline(_baseline) 
    {
        K_cv = cv::Mat::zeros(3, 3, CV_32F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                K_cv.at<float>(r, c) = K(r, c);
    }

    void slam(const StereoPair &stereo_pair_prev, const StereoPair &stereo_pair_curr, 
                std::vector<Eigen::Vector3f> &all_points3D,  
                std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& camera_poses)
    {
        // for each image initialize ImageDescription
        ImageDescription imgd_cl, imgd_cr, imgd_pl, imgd_pr;
        imgd_cl.img_id = "cl"; // current left
        imgd_cr.img_id = "cr"; // current right
        imgd_pl.img_id = "pl"; // previous left
        imgd_pr.img_id = "pr"; // previous right

        // detect matches between the current left image and the current right image
        detect_and_match_temporal(stereo_pair_prev.left, stereo_pair_curr.left, imgd_pl, imgd_cl);

        // detect matches between the previous left and the previous right image - only for the features that have been matched in the previous left
        detect_features(stereo_pair_prev.right, imgd_pr);
        match_features_stereo(imgd_pl, imgd_pr);

        // filter first stereo matches and then temporal matches
        filter_stereo_matches(imgd_pl, imgd_pr);
        filter_temporal_matches(imgd_pl, imgd_cl);

        // triangulate stereo matches-
        std::vector<cv::Point2f> points2D;
        std::vector<Eigen::Vector3f> _points3D;

        auto points3D_map = triangulate(imgd_pl, imgd_pr);

        for (int kp_idx_cl = 0; kp_idx_cl < imgd_cl.kps.size(); kp_idx_cl++)
        {
            for (const auto &m_cl : imgd_cl.matches_per_kp[kp_idx_cl])
            {
                int idx_pl = m_cl.kp_idx_other;
                if (points3D_map.count(idx_pl))
                { 
                    _points3D.push_back(points3D_map[idx_pl]);
                    points2D.push_back(imgd_cl.kps[m_cl.kp_idx_this].pt);
                }
            }
        }

        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        cv::Mat inliers_mask;
        PnP(_points3D, points2D, R, t, inliers_mask);

        // store all inlier 3D points as observations
        auto T_wp = camera_poses.back().matrix3x4(); // pose of the prev camera frame in the world
        std::vector<Observation> observations_curr;
        for (int i = 0; i < _points3D.size(); i++)
        {
            if (inliers_mask.at<uchar>(i)) 
            {
                Eigen::Vector3f pt_cam = _points3D[i]; 
                Eigen::Vector4f pt_cam_h(pt_cam(0), pt_cam(1), pt_cam(2), 1.0f);
                Eigen::Vector3f pt_world = T_wp * pt_cam_h;
                
                all_points3D.push_back(pt_world);
                observations_curr.push_back({camera_poses.size()-1, all_points3D.size()-1, {points2D[i].x, points2D[i].y}});
            }
        }

        observations.push_back(observations_curr);

        int window_size = 3;
        if (camera_poses.size() > window_size)
        {
            int start_idx = camera_poses.size() - window_size;
            std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> window_poses(camera_poses.begin() + start_idx, camera_poses.end());

            std::unordered_map<int, int> cam_idx_map;
            for (int i = 0; i < window_size; ++i)
                cam_idx_map[start_idx + i] = i;
                
            std::vector<Eigen::Vector3f> window_points;
            std::vector<Observation> window_observations;
            std::unordered_map<int, int> point_idx_map;
            
            for (int cam_global_idx = start_idx; cam_global_idx < camera_poses.size(); ++cam_global_idx) {
                for (const auto &obs : observations[cam_global_idx]) {
                    if (point_idx_map.find(obs.point3D_idx) == point_idx_map.end()) {
                        point_idx_map[obs.point3D_idx] = window_points.size();
                        window_points.push_back(all_points3D[obs.point3D_idx]);
                    }

                    Observation local_obs;
                    local_obs.camera_id = cam_idx_map[cam_global_idx]; 
                    local_obs.point3D_idx = point_idx_map[obs.point3D_idx];
                    local_obs.p = obs.p;
                    window_observations.push_back(local_obs);
                }
            }

            std::cout << "Last pose before bundle adjustment: \n" << window_poses[2].matrix() << std::endl;
            solve_bundle_adjustment(window_poses, window_points, window_observations, K);
            std::cout << "Last pose after bundle adjustment: \n" << window_poses[2].matrix() << std::endl;
            
            for (int i = start_idx; i < camera_poses.size(); i++)
                camera_poses[i] = window_poses[i-start_idx];
        }

        camera_poses.push_back(camera_poses.back() * Sophus::SE3f(R, t).inverse());

        visualize_matches(stereo_pair_prev.left, stereo_pair_prev.right, imgd_pl, imgd_pr);
        int key = cv::waitKey(30);
        if (key == 27)
        {
            cv::destroyAllWindows();
            exit(0);
        }
    
    }

    void vision_odometry(const StereoPair &stereo_pair_prev, const StereoPair &stereo_pair_curr, 
                        Sophus::SE3f &T, std::vector<Eigen::Vector3f> &points3D)
    {
        // for each image initialize ImageDescription
        ImageDescription imgd_cl, imgd_cr, imgd_pl, imgd_pr;
        imgd_cl.img_id = "cl"; // current left
        imgd_cr.img_id = "cr"; // current right
        imgd_pl.img_id = "pl"; // previous left
        imgd_pr.img_id = "pr"; // previous right

        // detect matches between the current left image and the current right image
        detect_and_match_temporal(stereo_pair_prev.left, stereo_pair_curr.left, imgd_pl, imgd_cl);

        // detect matches between the previous left and the previous right image - only for the features that have been matched in the previous left
        detect_features(stereo_pair_prev.right, imgd_pr);
        match_features_stereo(imgd_pl, imgd_pr);

        // filter first stereo matches and then temporal matches
        filter_stereo_matches(imgd_pl, imgd_pr);
        filter_temporal_matches(imgd_pl, imgd_cl);

        // triangulate stereo matches-
        std::vector<cv::Point2f> points2D;
        auto points3D_map = triangulate(imgd_pl, imgd_pr);

        for (int kp_idx_cl = 0; kp_idx_cl < imgd_cl.kps.size(); kp_idx_cl++)
        {
            for (const auto &m_cl : imgd_cl.matches_per_kp[kp_idx_cl])
            {
                int idx_pl = m_cl.kp_idx_other;
                if (points3D_map.count(idx_pl))
                { 
                    points3D.push_back(points3D_map[idx_pl]);
                    points2D.push_back(imgd_cl.kps[m_cl.kp_idx_this].pt);
                }
            }
        }

        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        cv::Mat inliers_mask;
        PnP(points3D, points2D, R, t, inliers_mask);
        T = T * Sophus::SE3f(R, t).inverse();

        visualize_matches(stereo_pair_prev.left, stereo_pair_prev.right, imgd_pl, imgd_pr);
        int key = cv::waitKey(30);
        if (key == 27)
        {
            cv::destroyAllWindows();
            exit(0);
        }
    }

private:
    Eigen::Matrix3f K;
    cv::Mat K_cv;
    Eigen::Matrix<float, 3, 4> P0, P1;
    float baseline;
    std::vector<std::vector<Observation>> observations;

    std::map<int, Eigen::Vector3f> triangulate(const ImageDescription &imgd_pl, const ImageDescription &imgd_pr)
    {
        std::map<int, Eigen::Vector3f> points3D_map;
        float fx = K(0,0); float fy = K(1, 1); float cx = K(0, 2); float cy = K(1,2);

        for (int kp_idx_pr = 0; kp_idx_pr < imgd_pr.kps.size(); kp_idx_pr++)
        {
            for (const auto &m : imgd_pr.matches_per_kp[kp_idx_pr])
            {
                int idx_pl = m.kp_idx_other;
                
                cv::Point2f pt_pl = imgd_pl.kps[idx_pl].pt;
                cv::Point2f pt_pr = imgd_pr.kps[m.kp_idx_this].pt;

                float disparity = pt_pl.x - pt_pr.x;
                if (disparity <= 0) continue;

                float Z = (fx * baseline) / disparity;
                float X = (pt_pl.x - cx) * Z / fx;
                float Y = (pt_pl.y - cy) * Z / fy;

                points3D_map[idx_pl] = Eigen::Vector3f(X, Y, Z);
            }
        }
        return points3D_map;
    }

    void PnP(const std::vector<Eigen::Vector3f> &points3D, const std::vector<cv::Point2f> &points2D, Eigen::Matrix3f &R, Eigen::Vector3f &t, cv::Mat &inliers_mask)
    {
        // convert points3D to vector of cv::Point3f
        std::vector<cv::Point3f> _points3D;
        for (const auto &p : points3D)
            _points3D.push_back({p[0], p[1], p[2]});

        cv::Mat _R_vec, _R, _t;
        bool success = cv::solvePnPRansac(_points3D, points2D, K_cv, cv::Mat(), _R_vec, _t, inliers_mask);

        int inliers = 0;
        for (int i = 0; i < _points3D.size(); i++)
            if (inliers_mask.at<uchar>(i)) 
                inliers++;

        std::cout << "The number of inliers / the number of matches: " << inliers << " / " << _points3D.size() << std::endl;
        cv::Rodrigues(_R_vec, _R);

        _R.convertTo(_R, CV_32F);
        _t.convertTo(_t, CV_32F);

        for (int r = 0; r < 3; ++r)         
            for (int c = 0; c < 3; ++c)
                R(r, c) = _R.at<float>(r, c);

        t(0) = _t.at<float>(0);
        t(1) = _t.at<float>(1);
        t(2) = _t.at<float>(2);
    }
};