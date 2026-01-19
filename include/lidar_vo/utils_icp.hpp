#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <sophus/se3.hpp>

struct Features {
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors;
};

std::vector<pcl::PointXYZI> read_ponint_cloud(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open file " + path);
    }

    file.seekg(0, std::ios::end);
    size_t num_bytes = file.tellg();  
    file.seekg(0, std::ios::beg); 

    size_t num_points = num_bytes / (4 * sizeof(float));

    std::vector<pcl::PointXYZI> points(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(&points[i].x), sizeof(float));
        file.read(reinterpret_cast<char*>(&points[i].y), sizeof(float));
        file.read(reinterpret_cast<char*>(&points[i].z), sizeof(float));
        file.read(reinterpret_cast<char*>(&points[i].intensity), sizeof(float));
    }

    return points;
}

std::vector<std::string> get_filenames(const std::string &path) 
{
    std::vector<std::string> filenames;

    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            filenames.push_back(path + "/" + entry.path().filename().string());
        }
    }

    std::sort(filenames.begin(), filenames.end());
    return filenames;
}


void downsample_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.5f, 0.5f, 0.5f);
    sor.filter(*cloud);
}

std::vector<Eigen::Matrix4f> read_ground_truth(const std::string &path, const Sophus::SE3f &T_cam0_velo)
{
    Sophus::SE3f T_velo_cam0 = T_cam0_velo.inverse();

    std::ifstream file;
    file.open(path);

    if (!file)
        throw std::runtime_error("Unable to read file: " + path);

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

        Eigen::Matrix4f corrected_pose = (T_velo_cam0 * Sophus::SE3f(pose) * T_cam0_velo).matrix();
        poses.push_back(corrected_pose);
    }

    return poses;
}

void store_estimated_trajectory(
    const std::vector<Eigen::Matrix4f>& poses_lidar_world, 
    const Sophus::SE3f& T_cam0_velo) 
{
    std::vector<Eigen::Matrix4f> poses_cam0_world;
    poses_cam0_world.reserve(poses_lidar_world.size());

    Eigen::Matrix4f T_cv = T_cam0_velo.matrix();
    Eigen::Matrix4f T_vc = T_cam0_velo.inverse().matrix();

    for (const auto& T_w_velo : poses_lidar_world)
    {
        Eigen::Matrix4f T_w_cam0 = T_cv * T_w_velo * T_vc;
        poses_cam0_world.push_back(T_w_cam0);
    }

    std::ofstream out("/home/dino/3dvid/lidar_visual_odometry_ws/src/lidar_vo/estimates/00_estimate_lidar.txt");
    if (!out.is_open()) return;

    for (const auto& T_w_cam0 : poses_cam0_world)
    {
        out << std::fixed << std::setprecision(9);
        out << T_w_cam0(0,0) << " " << T_w_cam0(0,1) << " " << T_w_cam0(0,2) << " " << T_w_cam0(0,3) << " "
            << T_w_cam0(1,0) << " " << T_w_cam0(1,1) << " " << T_w_cam0(1,2) << " " << T_w_cam0(1,3) << " "
            << T_w_cam0(2,0) << " " << T_w_cam0(2,1) << " " << T_w_cam0(2,2) << " " << T_w_cam0(2,3)
            << "\n";
    }
    out.close();
}

Features compute_feature_descriptors(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    // compute intrinsic shape signatures keypoints
    pcl::ISSKeypoint3D<pcl::PointXYZI, pcl::PointXYZI> iss;
    iss.setInputCloud(cloud);
    iss.setSalientRadius(1.0f);
    iss.setNonMaxRadius(1.2f);
    iss.setThreshold21(0.975);
    iss.setThreshold32(0.975);
    iss.setMinNeighbors(5);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZI>());
    iss.compute(*keypoints);

    // compute feature descriptor for each keypoint
    // compute surface normals
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.setRadiusSearch(0.5);
    ne.compute(*normals);

    // compute fpfh feature descriptor for each keypoint
    pcl::FPFHEstimation<pcl::PointXYZI, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keypoints);
    fpfh.setInputNormals(normals);
    fpfh.setSearchSurface(cloud);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(2.5);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh.compute(*descriptors);

    Features out;
    out.keypoints = keypoints;
    out.descriptors = descriptors;
    return out;
}