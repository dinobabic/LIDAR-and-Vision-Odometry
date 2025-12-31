#pragma once
#include "lidar_vo/utils_slam.hpp"

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
                Sophus::SE3f &T, std::vector<Eigen::Vector3f> &points3D)
    {
        // initial frame - stereo_pair_prev (left image) becomes the first keyframe
        if (!keyframe_initialized) 
        {
            keyframe_initialized = true;
            keyframe.T_wc = Sophus::SE3f(); // identity transformation

            // for each image initialize ImageDescription
            ImageDescription imgd_cl, imgd_cr, imgd_pl, imgd_pr;
            imgd_cl.img_id = "cl"; // current left
            imgd_cr.img_id = "cr"; // current right
            imgd_pl.img_id = "pl"; // previous left
            imgd_pr.img_id = "pr"; // previous right

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

        for (const auto &m_cl : imgd_cl.matches)
        {
            int idx_pl = m_cl.kp_idx_other;
            if (points3D_map.count(idx_pl))
            { 
                points3D.push_back(points3D_map[idx_pl]);
                points2D.push_back(imgd_cl.pts[m_cl.kp_idx_this]);
            }
        }

        Eigen::Matrix3f R;
        Eigen::Vector3f t;
        PnP(points3D, points2D, R, t);
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

    // variables need for slam
    KeyFrame keyframe;
    bool keyframe_initialized;

    std::map<int, Eigen::Vector3f> triangulate(const ImageDescription &imgd_pl, const ImageDescription &imgd_pr)
    {
        std::map<int, Eigen::Vector3f> points3D_map;
        float fx = K(0,0); float fy = K(1, 1); float cx = K(0, 2); float cy = K(1,2);

        for (const auto &m : imgd_pr.matches)
        {
            int idx_pl = m.kp_idx_other;
            
            cv::Point2f pt_pl = imgd_pl.pts[idx_pl];
            cv::Point2f pt_pr = imgd_pr.pts[m.kp_idx_this];

            float disparity = pt_pl.x - pt_pr.x;
            if (disparity <= 0) continue;

            float Z = (fx * baseline) / disparity;
            float X = (pt_pl.x - cx) * Z / fx;
            float Y = (pt_pl.y - cy) * Z / fy;

            points3D_map[idx_pl] = Eigen::Vector3f(X, Y, Z);
        }
        return points3D_map;
    }

    void PnP(const std::vector<Eigen::Vector3f> &points3D, const std::vector<cv::Point2f> &points2D, Eigen::Matrix3f &R, Eigen::Vector3f &t)
    {
        // convert points3D to vector of cv::Point3f
        std::vector<cv::Point3f> _points3D;
        for (const auto &p : points3D)
            _points3D.push_back({p[0], p[1], p[2]});

        cv::Mat _R_vec, _R, _t;
        cv::solvePnPRansac(_points3D, points2D, K_cv, cv::Mat(), _R_vec, _t);
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