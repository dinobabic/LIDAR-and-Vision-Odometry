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


struct Features {
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors;
};

std::vector<pcl::PointXYZ> read_ponint_cloud(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open file " + path);
    }

    file.seekg(0, std::ios::end);
    size_t num_bytes = file.tellg();  
    file.seekg(0, std::ios::beg); 

    size_t num_points = num_bytes / (4 * sizeof(float));

    std::vector<pcl::PointXYZI> points_with_intensity(num_points);
    std::vector<pcl::PointXYZ> points(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        file.read(reinterpret_cast<char*>(&points_with_intensity[i].x), sizeof(float));
        file.read(reinterpret_cast<char*>(&points_with_intensity[i].y), sizeof(float));
        file.read(reinterpret_cast<char*>(&points_with_intensity[i].z), sizeof(float));
        file.read(reinterpret_cast<char*>(&points_with_intensity[i].intensity), sizeof(float));
    }

    for (auto p : points_with_intensity)
        points.push_back(pcl::PointXYZ(p.x, p.y, p.z));

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


void downsample_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.2f, 0.2f, 0.2f);
    sor.filter(*cloud);
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

Features compute_feature_descriptors(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    // compute intrinsic shape signatures keypoints
    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss;
    iss.setInputCloud(cloud);
    iss.setSalientRadius(0.5f);
    iss.setNonMaxRadius(0.5f);
    iss.setThreshold21(0.975);
    iss.setThreshold32(0.975);
    iss.setMinNeighbors(5);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>());
    iss.compute(*keypoints);

    // compute feature descriptor for each keypoint
    // compute surface normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.setRadiusSearch(0.5);
    ne.compute(*normals);

    // compute fpfh feature descriptor for each keypoint
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(keypoints);
    fpfh.setInputNormals(normals);
    fpfh.setSearchSurface(cloud);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(0.8);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh.compute(*descriptors);

    Features out;
    out.keypoints = keypoints;
    out.descriptors = descriptors;
    return out;
}
