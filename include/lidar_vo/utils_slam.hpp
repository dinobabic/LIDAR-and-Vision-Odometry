#pragma once
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <sophus/se3.hpp>

cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(10000);
cv::BFMatcher bf(cv::NORM_HAMMING, false);

struct StereoPair
{
    cv::Mat left;
    cv::Mat right;
};

struct MatchInfo {
    int kp_idx_this;   // keypoint index in this image
    int kp_idx_other;  // keypoint index in the other image
    std::string other_img_id;  // id of the other image
    int point3D_idx = -1; // idx of 3D point this match triangulates to

    bool operator==(const MatchInfo &other) const {
        return kp_idx_this == other.kp_idx_this &&
            kp_idx_other == other.kp_idx_other &&
            other_img_id == other.other_img_id;
    }
};

struct MatchInfoHash {
    std::size_t operator()(const MatchInfo &m) const {
        return std::hash<int>{}(m.kp_idx_this) ^
               (std::hash<int>{}(m.kp_idx_other) << 1) ^
               (std::hash<std::string>{}(m.other_img_id) << 2);
    }
};

struct ImageDescription
{
    std::string img_id;
    std::vector<cv::Point2f> pts;
    std::vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    std::unordered_set<MatchInfo, MatchInfoHash> matches; 
};

struct KeyFrame
{
    std::vector<Eigen::Vector3f> points3D;
    std::vector<cv::KeyPoint> kps;
    std::vector<cv::Point2f> pts;
    cv::Mat descriptors;
    Sophus::SE3f T_wc; // expresses key frame in world frame
};

struct Observation
{   
    Observation(int _camera_id, int _point_id, int _keypoint_id, const Eigen::Vector2f &_p) : 
        camera_id(_camera_id),
        point_id(_point_id),
        keypoint_id(_keypoint_id),
        p(_p) {}

    int camera_id;
    int point_id;
    int keypoint_id;
    Eigen::Vector2f p;
};

StereoPair load_stereo_pair(const std::string &path, const std::string &image_name)
{
    std::cout << "Image path I am reading is " + path + "/image_0" + " and name is " + image_name << std::endl;
    cv::Mat img_left = cv::imread(path + "/image_0/" + image_name, cv::IMREAD_GRAYSCALE);
    std::cout << "Image path I am reading is " + path + "/image_1" + " and name is " + image_name << std::endl;
    cv::Mat img_right = cv::imread(path + "/image_1/" + image_name, cv::IMREAD_GRAYSCALE);
    return {img_left, img_right};
}

std::vector<std::string> get_filenames(const std::string &path) 
{
    std::vector<std::string> filenames;

    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            filenames.push_back(entry.path().filename().string());
        }
    }

    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

std::vector<Eigen::Matrix4f> read_ground_truth(const std::string &path)
{
    std::ifstream file;
    file.open(path);

    if (!file)
    {
        throw std::runtime_error("Unable to read file: " + path);
    }


    std::string line;
    std::vector<Eigen::Matrix4f> poses;
    float v;

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::vector<float> values;

        while (iss >> v)
            values.push_back(v);

        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                pose(i, j) = values[i * 4 + j];

        poses.push_back(pose);
    }

    return poses;
}

void detect_features(const cv::Mat &img, ImageDescription &imgd)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::Mat(), keypoints, descriptors);

    imgd.kps = keypoints;
    imgd.descriptors = descriptors;

    imgd.pts.clear();
    imgd.pts.reserve(imgd.kps.size());
    for (size_t i = 0; i < imgd.kps.size(); i++)
        imgd.pts.push_back(imgd.kps[i].pt);

    std::cout << "The number of detected features in image " << imgd.img_id << " is: " << imgd.kps.size() << std::endl;
}

void match_features_temporal(ImageDescription &imgd_pl, ImageDescription &imgd_cl)
{
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(imgd_pl.descriptors, imgd_cl.descriptors, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    for (size_t i = 0; i < knn_matches.size(); i++)
        if (knn_matches[i].size() == 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            good_matches.push_back(knn_matches[i][0]);

    imgd_pl.matches.reserve(imgd_pl.matches.size() + good_matches.size());
    imgd_cl.matches.reserve(imgd_cl.matches.size() + good_matches.size());
    for (const auto &match : good_matches)
    {
        imgd_pl.matches.insert({match.queryIdx, match.trainIdx, imgd_cl.img_id});
        imgd_cl.matches.insert({match.trainIdx, match.queryIdx, imgd_pl.img_id});
    }

    std::cout << "The number of matches between image " << imgd_pl.img_id << " and image " << imgd_cl.img_id << " is: " << imgd_pl.matches.size() << std::endl;
}

void match_features_stereo(ImageDescription &imgd_pl, ImageDescription &imgd_pr)
{
    // imgd_pl contains features that have already been matched, for instance with the previous left image
    // only for those features we want to find correspondances in imgd_pr
    static cv::Mat query_descriptors;
    std::vector<int> query_indices;
    query_indices.reserve(imgd_pl.matches.size());

    for (const auto &m : imgd_pl.matches)
        query_indices.push_back(m.kp_idx_this);

    query_descriptors.create(query_indices.size(),
                            imgd_pl.descriptors.cols,
                            imgd_pl.descriptors.type());

    for (size_t i = 0; i < query_indices.size(); ++i)
        imgd_pl.descriptors.row(query_indices[i]).copyTo(query_descriptors.row(i));

    std::vector<std::vector<cv::DMatch>> knn_matches;
    knn_matches.reserve(query_indices.size());
    bf.knnMatch(query_descriptors, imgd_pr.descriptors, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i].size() < 2)
            continue;

        const auto &m1 = knn_matches[i][0];
        const auto &m2 = knn_matches[i][1];

        if (m1.distance < ratio_thresh * m2.distance)
        {
            int query_row = m1.queryIdx;
            int train_kp  = m1.trainIdx;

            MatchInfo match;
            match.kp_idx_this  = train_kp;                     
            match.kp_idx_other = query_indices[query_row];  
            match.other_img_id = imgd_pl.img_id;
            imgd_pr.matches.insert(match);

            match.kp_idx_this = query_indices[query_row];
            match.kp_idx_other = train_kp;
            match.other_img_id = imgd_pr.img_id;
            imgd_pl.matches.insert(match);
        }
    }

    std::cout << "The number of matches between image " << imgd_pl.img_id << " and image " << imgd_pr.img_id << " is: " << imgd_pr.matches.size() << std::endl;
}

void detect_and_match_temporal(const cv::Mat img_pl, const cv::Mat img_cl,
                      ImageDescription &imgd_pl, ImageDescription &imgd_cl)
{
    detect_features(img_pl, imgd_pl);
    detect_features(img_cl, imgd_cl);
    match_features_temporal(imgd_pl, imgd_cl);
} 

void filter_stereo_matches(ImageDescription &imgd_pl, ImageDescription &imgd_pr)
{
    auto filtered_matches_l = imgd_pl.matches;
    auto filtered_matches_r = imgd_pr.matches;

    for (const auto &m_l : imgd_pl.matches)
    { 
        if (m_l.other_img_id == imgd_pr.img_id)
        {
            auto pt_l = imgd_pl.pts[m_l.kp_idx_this];
            auto pt_r = imgd_pr.pts[m_l.kp_idx_other];
            
            // ensure positive disparity and epipolar constraint
            if (pt_l.x - pt_r.x <= 0 || std::abs(pt_l.y - pt_r.y) > 2.0)
            {
                // delete this match
                filtered_matches_l.erase(m_l);
                auto m_r = *imgd_pr.matches.find({m_l.kp_idx_other, m_l.kp_idx_this, imgd_pl.img_id});
                filtered_matches_r.erase(m_r);
            }
        }
    }

    imgd_pl.matches = filtered_matches_l;
    imgd_pr.matches = filtered_matches_r;

    // count the number of remaining matches - imgd_pr has only matches with the left image
    std::cout << "After filtering stereo matches, there are "  << imgd_pr.matches.size() << " matches left between pl and pr." << std::endl;
}

void filter_temporal_matches(ImageDescription &imgd_pl, ImageDescription &imgd_cl)
{
    auto filtered_matches_pl = imgd_pl.matches;
    auto filtered_matches_cl = imgd_cl.matches;

    std::unordered_set<int> stereo_indices;
    for (const auto &m_pl : imgd_pl.matches)
        if (m_pl.other_img_id == "pr")
            stereo_indices.insert(m_pl.kp_idx_this);

    for (const auto &m_cl : imgd_cl.matches)
    {
        if (m_cl.other_img_id != imgd_pl.img_id) continue;
        if (stereo_indices.find(m_cl.kp_idx_other) == stereo_indices.end())
        {
            filtered_matches_cl.erase(m_cl);
            auto m_pl_to_delete = *imgd_pl.matches.find({m_cl.kp_idx_other, m_cl.kp_idx_this, "cl"});
            filtered_matches_pl.erase(m_pl_to_delete);
        }
    }

    imgd_pl.matches = filtered_matches_pl;
    imgd_cl.matches = filtered_matches_cl;

    std::cout << "After filtering temporal matches, there are " << imgd_cl.matches.size() << " left between pl and cl." << std::endl;
}

void get_matches_from_descriptions(const ImageDescription &img_description1, const ImageDescription &img_description2, 
                                    std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2)
{
    for (const auto &m : img_description1.matches) {
        if (m.other_img_id == img_description2.img_id) {
            kps1.push_back(img_description1.kps[m.kp_idx_this]);
            kps2.push_back(img_description2.kps[m.kp_idx_other]);
        }
    }   
}

void visualize_matches(const cv::Mat &img1, const cv::Mat &img2, 
                        const ImageDescription &img_description1, const ImageDescription &img_description2)
{   
    std::vector<cv::KeyPoint> kps1, kps2;
    get_matches_from_descriptions(img_description1, img_description2, kps1, kps2);

    // std::vector<cv::DMatch> dummy_matches;
    // for (size_t i = 0; i < int(kps1.size() * 0.2); i++) 
    //     dummy_matches.emplace_back(i, i, 0.0f);
    

    cv::Mat img_matches;
    cv::hconcat(img1, img2, img_matches);
    // cv::drawMatches(img1, kps1, img2, kps2, dummy_matches, img_matches);
    cv::resize(img_matches, img_matches, cv::Size(), 0.7, 0.7);
    cv::imshow("Matches", img_matches);
}