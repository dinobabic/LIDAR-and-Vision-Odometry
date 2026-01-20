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
    std::vector<cv::KeyPoint> kps;
    cv::Mat descriptors;
    std::vector<std::vector<MatchInfo>> matches_per_kp; // it is a vector of size kps.size() -> at index i, contains all matches for the keypoint kps[i]
};

struct Observation
{   
    Observation() {}
    Observation(int _camera_id, int _point3D_idx, const Eigen::Vector2f &_p) : 
        camera_id(_camera_id),
        point3D_idx(_point3D_idx),
        p(_p) {}

    int camera_id;
    int point3D_idx; 
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
            //filenames.push_back(path + entry.path().filename().string()); // for train vocab
            filenames.push_back(entry.path().filename().string()); // for slam node
        }
    }

    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

// used in train_vocab.cpp
std::vector<cv::Mat> load_imgs(const std::vector<std::string> &image_names)
{
    std::vector<cv::Mat> imgs;
    for (const auto &img_name : image_names)
        imgs.push_back(cv::imread(img_name, cv::IMREAD_GRAYSCALE));
    return imgs;
}

cv::Mat read_rgb(const std::string &path)
{
    return cv::imread(path, cv::IMREAD_COLOR);
}

// used in train_vocab.cpp
std::vector<std::vector<cv::Mat>> get_image_descriptors(const std::vector<cv::Mat> &imgs)
{
    std::vector<std::vector<cv::Mat>> descriptors;
    descriptors.reserve(imgs.size());

    for (const auto &img : imgs) 
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
        
        // descriptor is a matrix of size (N x descriptor_size)
        // DBoW2 wants a vector of descriptors each stored as a separate cv::Mat
        std::vector<cv::Mat> descriptors_img;
        for (size_t i = 0; i < descriptor.rows; i++)
            descriptors_img.push_back(descriptor.row(i));
        descriptors.push_back(descriptors_img);
    }

    return descriptors;
}

std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> read_ground_truth(const std::string &path)
{
    std::ifstream file;
    file.open(path);

    if (!file)
    {
        throw std::runtime_error("Unable to read file: " + path);
    }


    std::string line;
    std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>> poses;
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

        poses.push_back(Sophus::SE3f(pose));
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
    imgd.matches_per_kp.resize(imgd.kps.size());

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

    int matches_count = 0;
    for (const auto &match : good_matches)
    {
        ++matches_count;
        imgd_pl.matches_per_kp[match.queryIdx].push_back({match.queryIdx, match.trainIdx, imgd_cl.img_id});
        imgd_cl.matches_per_kp[match.trainIdx].push_back({match.trainIdx, match.queryIdx, imgd_pl.img_id});
    }

    std::cout << "The number of matches between image " << imgd_pl.img_id << " and image " << imgd_cl.img_id << " is: " << matches_count << std::endl;
}

void match_features_stereo(ImageDescription &imgd_pl, ImageDescription &imgd_pr)
{
    // imgd_pl has already been matched with the current left image and we want to find matches between prev left and right
    // only for those features that have been matched between prev left and curr left
    static cv::Mat query_descriptors;
    std::vector<int> query_indices;
    query_indices.reserve(imgd_pl.kps.size());

    for (int kp_idx = 0; kp_idx < imgd_pl.kps.size(); kp_idx++)
    {
        const auto &matches = imgd_pl.matches_per_kp[kp_idx];
        bool has_temporal_match = false;
        for (const auto &m : matches)
        {
            if (m.other_img_id == "cl") {
                has_temporal_match = true;
                break;
            }
        }

        if (has_temporal_match)
            query_indices.push_back(kp_idx);
    }

    query_descriptors.create(query_indices.size(),
                            imgd_pl.descriptors.cols,
                            imgd_pl.descriptors.type());

    for (size_t i = 0; i < query_indices.size(); ++i)
        imgd_pl.descriptors.row(query_indices[i]).copyTo(query_descriptors.row(i));

    std::vector<std::vector<cv::DMatch>> knn_matches;
    knn_matches.reserve(query_indices.size());
    bf.knnMatch(query_descriptors, imgd_pr.descriptors, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    int matches_count = 0;
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
            imgd_pr.matches_per_kp[train_kp].push_back(match);

            match.kp_idx_this = query_indices[query_row];
            match.kp_idx_other = train_kp;
            match.other_img_id = imgd_pr.img_id;
            imgd_pl.matches_per_kp[query_indices[query_row]].push_back(match);
            matches_count++;
        }
    }

    std::cout << "The number of matches between image " << imgd_pl.img_id << " and image " << imgd_pr.img_id << " is: " << matches_count << std::endl;
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
    std::vector<std::vector<MatchInfo>> filtered_pl(imgd_pl.kps.size());
    std::vector<std::vector<MatchInfo>> filtered_pr(imgd_pr.kps.size());

    int kept = 0;
    for (int kp_idx_pl = 0; kp_idx_pl < imgd_pl.kps.size(); kp_idx_pl++)
    {
        const auto &matches = imgd_pl.matches_per_kp[kp_idx_pl];
        const auto &pt_l = imgd_pl.kps[kp_idx_pl].pt;

        for (const auto &m : matches)
        {
            if (m.other_img_id != imgd_pr.img_id)
            {
                filtered_pl[kp_idx_pl].push_back(m);
                continue;
            }

            int kp_idx_pr = m.kp_idx_other;
            const auto &pt_r = imgd_pr.kps[kp_idx_pr].pt;
            bool valid = (pt_l.x - pt_r.x > 0) && (std::abs(pt_l.y - pt_r.y) <= 2.0);

            if (!valid)
                continue;

            filtered_pl[kp_idx_pl].push_back(m);
            filtered_pr[kp_idx_pr].push_back({kp_idx_pr, kp_idx_pl, imgd_pl.img_id});
            kept++;
        }
    }

    imgd_pl.matches_per_kp = std::move(filtered_pl);
    imgd_pr.matches_per_kp = std::move(filtered_pr);

    std::cout << "After filtering stereo matches, " << kept << " valid stereo matches remain between " << imgd_pl.img_id << " and " << imgd_pr.img_id << std::endl;
}


void filter_temporal_matches(ImageDescription &imgd_pl,
                             ImageDescription &imgd_cl)
{
    std::vector<std::vector<MatchInfo>> filtered_pl(imgd_pl.kps.size());
    std::vector<std::vector<MatchInfo>> filtered_cl(imgd_cl.kps.size());

    std::unordered_set<int> pl_kps_with_stereo;
    for (int kp_idx = 0; kp_idx < imgd_pl.kps.size(); kp_idx++)
    {
        for (const auto &m : imgd_pl.matches_per_kp[kp_idx])
        {
            if (m.other_img_id == "pr")
            {
                filtered_pl[kp_idx].push_back(m);
                pl_kps_with_stereo.insert(kp_idx);
                break;
            }
        }
    }

    int kept = 0;
    for (int kp_idx_cl = 0; kp_idx_cl < imgd_cl.kps.size(); kp_idx_cl++)
    {
        for (const auto &m : imgd_cl.matches_per_kp[kp_idx_cl])
        {
            if (m.other_img_id != imgd_pl.img_id)
                continue;  

            int kp_idx_pl = m.kp_idx_other;
            if (pl_kps_with_stereo.count(kp_idx_pl) == 0)
                continue;

            filtered_cl[kp_idx_cl].push_back(m);
            filtered_pl[kp_idx_pl].push_back({kp_idx_pl, kp_idx_cl, imgd_cl.img_id});
            kept++;
        }
    }

    imgd_pl.matches_per_kp = std::move(filtered_pl);
    imgd_cl.matches_per_kp = std::move(filtered_cl);

    std::cout << "After filtering temporal matches, "  << kept  << " remain between "  << imgd_pl.img_id << " and "  << imgd_cl.img_id << std::endl;
}


void get_matches_from_descriptions(const ImageDescription &imgd1, const ImageDescription &imgd2, 
                                    std::vector<cv::KeyPoint>& kps1, std::vector<cv::KeyPoint>& kps2)
{
    for (int kp_idx = 0; kp_idx < imgd1.kps.size(); kp_idx++) {
        for (const auto &m : imgd1.matches_per_kp[kp_idx])
        {
            if (m.other_img_id == imgd2.img_id) {
                kps1.push_back(imgd1.kps[m.kp_idx_this]);
                kps2.push_back(imgd2.kps[m.kp_idx_other]);
            }
        }
    }   
}

void visualize_matches(const cv::Mat &img1, const cv::Mat &img2, 
                        const ImageDescription &imgd1, const ImageDescription &imgd2)
{   
    std::vector<cv::KeyPoint> kps1, kps2;
    get_matches_from_descriptions(imgd1, imgd2, kps1, kps2);

    // std::vector<cv::DMatch> dummy_matches;
    // for (size_t i = 0; i < int(kps1.size() * 0.2); i++) 
    //     dummy_matches.emplace_back(i, i, 0.0f);
    

    cv::Mat img_matches;
    cv::hconcat(img1, img2, img_matches);
    // cv::drawMatches(img1, kps1, img2, kps2, dummy_matches, img_matches);
    cv::resize(img_matches, img_matches, cv::Size(), 0.7, 0.7);
    cv::imshow("Matches", img_matches);
}

void store_estimated_trajectory(const std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& poses_cam_world) 
{
    std::ofstream out("/home/dino/3dvid/lidar_visual_odometry_ws/src/lidar_vo/trajectories/00_estimate_camera.txt");
    if (!out.is_open()) return;

    for (const auto& T_w_cam : poses_cam_world)
    {
        auto _T_w_cam = T_w_cam.matrix3x4();
        out << std::fixed << std::setprecision(9);
        out << _T_w_cam(0,0) << " " << _T_w_cam(0,1) << " " << _T_w_cam(0,2) << " " << _T_w_cam(0,3) << " "
            << _T_w_cam(1,0) << " " << _T_w_cam(1,1) << " " << _T_w_cam(1,2) << " " << _T_w_cam(1,3) << " "
            << _T_w_cam(2,0) << " " << _T_w_cam(2,1) << " " << _T_w_cam(2,2) << " " << _T_w_cam(2,3)
            << "\n";
    }
    out.close();
}

void project_points(const cv::Mat &img, const std::vector<Eigen::Vector3f> &points3D,
                    std::vector<Eigen::Vector3f> &colors, Eigen::Matrix3f K,
                    const std::optional<Sophus::SE3f> &T_w = std::nullopt)
{
    auto f = K(0,0);
    auto cx = K(0,2);
    auto cy = K(1,2);

    colors.reserve(points3D.size());

    for (auto pt3D : points3D)
    {
        Eigen::Vector3f pt_cam = pt3D;

        // if T_w is provided, transform point into camera frame
        if (T_w.has_value()) 
            pt_cam = T_w->inverse() * pt3D;        

        int x = static_cast<int>((pt_cam[0] / pt_cam[2]) * f + cx);
        int y = static_cast<int>((pt_cam[1] / pt_cam[2]) * f + cy); 
        
        if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
        {
            colors.emplace_back(0.0, 0.0, 0.0); // black point
            continue;
        }

        const cv::Vec3b &bgr = img.at<cv::Vec3b>(y, x);

        float r = bgr[2] / 255.0f;
        float g = bgr[1] / 255.0f;
        float b = bgr[0] / 255.0f;

        colors.emplace_back(r, g, b);
    }
}

