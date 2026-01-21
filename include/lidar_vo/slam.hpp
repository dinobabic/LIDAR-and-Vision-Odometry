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

    void initialize_keyframe(const int camera_id, const StereoPair &stereo_pair, const Sophus::SE3f &T)
    {
        // initialize keyframe - detect matches in the left and right image, match them and triangulate
        std::vector<Eigen::Vector3f> new_points3D;
        match_and_triangulate(stereo_pair, keyframe, K, baseline);
        new_points3D.insert(new_points3D.end(), keyframe.points3D.begin(), keyframe.points3D.end());

        auto img_left = read_rgb(dataset_path_color + image_names[camera_id]);
        std::vector<Eigen::Vector3f> _colors; // color of each 3D point
        project_points(img_left, new_points3D, _colors, K);

        colors.insert(colors.end(), _colors.begin(), _colors.end());
        //points3D.insert(points3D.end(), new_points3D.begin(), new_points3D.end()); 

        keyframe_points_to_global_points.clear();
        for (int i = 0; i < new_points3D.size(); i++)
        {
            keyframe_points_to_global_points[i] = points3D.size();
            points3D.push_back(T * new_points3D[i]);
        }

        // initialize observations
        for (int i = 0; i < keyframe.kps.size(); i++)
        {
            auto pt2d = keyframe.kps[i].pt;
            observations.push_back(Observation(camera_id, keyframe_points_to_global_points[i], {pt2d.x, pt2d.y}));
        }
    }


    void keyframe_motion(const int camera_id, const StereoPair &stereo_pair_prev, const StereoPair &stereo_pair_curr,
                        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> &estimated_poses)
    {
        auto T = estimated_poses.back();

        std::vector<cv::Point2f> matched_points2D;
        std::vector<Eigen::Vector3f> matched_points3D;
        std::vector<int> matched_kf_indices;
        match_against_keyframe(keyframe, stereo_pair_curr.left, matched_points3D, matched_points2D, matched_kf_indices);
        
        if (1.0f*matched_points2D.size() / keyframe.kps.size() < 1.0)
        {
            // make the previous frame the keyframe and recompute 2d-3d correspondances for better ratio
            std::cout << "Initializing new keyframe" << std::endl;
            initialize_keyframe(camera_id-1, stereo_pair_prev, T);

            matched_points2D.clear();
            matched_points3D.clear();
            matched_kf_indices.clear();
            match_against_keyframe(keyframe, stereo_pair_curr.left, matched_points3D, matched_points2D, matched_kf_indices);
        }

        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        cv::Mat inliers_mask;
        PnP(matched_points3D, matched_points2D, R, t, inliers_mask);
        estimated_poses.push_back(T * Sophus::SE3f(R, t).inverse());

        // initialize observations
        for (int i = 0; i < matched_points2D.size(); i++)
        {
            if (inliers_mask.at<uchar>(i))
            {
                int local_idx = matched_kf_indices[i];
                int global_idx = keyframe_points_to_global_points[local_idx];
                auto pt2d = matched_points2D[i];
                
                observations.push_back(Observation(camera_id, global_idx, {pt2d.x, pt2d.y}));
            }
        }

        std::cout << "The number of observations is: " << observations.size() << std::endl;

        // perform bundle adjustment
        return;
        if (camera_id >= 3)
        {
            std::vector<Observation> window_observations;
            std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> window_poses;

            int start_idx = camera_id-3;
            window_poses.insert(window_poses.begin(), estimated_poses.begin() + start_idx, estimated_poses.end());

            for (auto it = observations.rbegin(); it != observations.rend(); ++it)
            {
                if (it->camera_id < start_idx) 
                    break; 

                Observation normalized_obs = *it;
                normalized_obs.camera_id -= start_idx;
                window_observations.push_back(normalized_obs);
            }
            
            std::cout << "Last pose before bundle adjustment: " << window_poses.back().matrix() << std::endl;
            solve_bundle_adjustment(window_poses, points3D, window_observations, K);
            std::cout << "Last pose after bundle adjustment: " << window_poses.back().matrix() << std::endl;

            for (int i = 0; i < window_poses.size(); ++i) 
                estimated_poses[start_idx + i] = window_poses[i];
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

        // filter stereo matches by ensuring positive depth and epipolar constraint
        // thereafter filter temporal matches
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

        // visualize_matches(stereo_pair_prev.left, stereo_pair_prev.right, imgd_pl, imgd_pr);
        // int key = cv::waitKey(30);
        // if (key == 27)s
        // {
        //     cv::destroyAllWindows();
        //     exit(0);
        // }
    }

    void set_image_names(const std::vector<std::string> &_image_names)
    {
        image_names = _image_names;
    }

    void set_dataset_path_color(const std::string &_dataset_path_color)
    {
        dataset_path_color = _dataset_path_color;
    }

    std::vector<Eigen::Vector3f>& get_points3D()
    {
        return this->points3D;
    }

    std::vector<Eigen::Vector3f>& get_colors()
    {
        return this->colors;
    }

private:
    Eigen::Matrix3f K;
    cv::Mat K_cv;
    Eigen::Matrix<float, 3, 4> P0, P1;
    float baseline;

    std::vector<std::string> image_names;
    std::string dataset_path_color;

    std::vector<Eigen::Vector3f> points3D;
    std::vector<Eigen::Vector3f> colors;
    std::vector<Observation> observations;
    Keyframe keyframe;
    std::unordered_map<int, int> keyframe_points_to_global_points;

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