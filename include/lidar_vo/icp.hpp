#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include <vector>
#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <limits>
#include <cmath>
#include <random>
#include <algorithm>

#include "lidar_vo/utils_icp.hpp"

class ICP 
{
private:

    
public: 

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudify(const std::vector<pcl::PointXYZI> &points)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        
        for (const auto &p : points)
            point_cloud->points.push_back(p);

        point_cloud->width = points.size();
        point_cloud->height = 1;
        point_cloud->is_dense = true;

        return point_cloud;
    }

    Eigen::Matrix4f ransac(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> &closest_points, 
                            int iters,
                            float threshold)
    {
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> best_inliers;
        std::random_device rd;
        std::mt19937 g(rd());
        
        int N = closest_points.size();

        // instead of working with all closest_points, work with a smaller sample
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), g);
        int M = std::min(1000, N);
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> sampled_closest_points(M);
        for (int i = 0; i < M; i++)
            sampled_closest_points[i] = closest_points[idx[i]];

        Eigen::MatrixXf pts1(3, M), pts2(3, M);
        for (int i = 0; i < M; ++i) {
            pts1.col(i) = sampled_closest_points[i].first;
            pts2.col(i) = sampled_closest_points[i].second;
        }

        for (int i = 0; i < iters; i++)
        {
            // sample correspondances to build initial model
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> sample;
            sample.reserve(10);
            std::sample(sampled_closest_points.begin(), sampled_closest_points.end(), std::back_inserter(sample), 3, g);
            
            // compute transformation
            Eigen::Matrix4f T = solve_icp(sample);
            auto R = T.block<3, 3>(0, 0);
            auto t = T.block<3, 1>(0, 3);
            
            Eigen::MatrixXf pts1_transformed = R * pts1;
            pts1_transformed.colwise() += t;
            
            Eigen::ArrayXf dists2 = (pts1_transformed - pts2).colwise().squaredNorm();

            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> current_inliers;
            for (int i = 0; i < M; i++)
            {
                if (dists2[i] < threshold)
                    current_inliers.push_back(sampled_closest_points[i]);
            }

            if (current_inliers.size() > best_inliers.size())
                best_inliers = current_inliers;
        }

        if (best_inliers.empty()) {
            std::cerr << "RANSAC failed: no inliers found." << std::endl;
            return Eigen::Matrix4f::Identity();
        }

        // recompute transformation using all inliers
        return solve_icp(best_inliers);
    }

    Eigen::Matrix4f solve_icp(const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> &closest_points)
    {
        auto N = closest_points.size();

        // compute centroids
        Eigen::Vector3f centroid1 = Eigen::Vector3f::Zero();
        Eigen::Vector3f centroid2 = Eigen::Vector3f::Zero();

        for (auto pair : closest_points)
        {
            centroid1 += pair.first;
            centroid2 += pair.second;
        }

        centroid1 /= N;
        centroid2 /= N;

        // compute denormalized coordinates
        std::vector<Eigen::Vector3f> denorm_coords1, denorm_coords2;
        for (auto pair : closest_points)
        {
            denorm_coords1.push_back(pair.first - centroid1);
            denorm_coords2.push_back(pair.second - centroid2);
        }

        // compute matrix H
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for (size_t i = 0; i < N; i++)
        {
            auto q = denorm_coords1[i];
            auto p = denorm_coords2[i];
            H += p * q.transpose();
        }

        // SVD decomposition of H
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f U = svd.matrixU();
        Eigen::Matrix3f V = svd.matrixV();
        Eigen::Matrix3f R = U*V.transpose();
        
        if (R.determinant() < 0)
        {
            V.col(2) *= -1;
            R = U*V.transpose();
        }

        Eigen::Vector3f t = centroid2 - R*centroid1;
    
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;    

        return T;
    }

    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> find_correspondances_features_based(
            pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_1, 
            pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_2)
    {
        auto features_1 = compute_feature_descriptors(point_cloud_1);
        auto features_2 = compute_feature_descriptors(point_cloud_2);

        pcl::KdTreeFLANN<pcl::FPFHSignature33> matchTree;
        matchTree.setInputCloud(features_2.descriptors);

        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>matched_points;

        for (size_t i = 0; i < features_1.descriptors->size(); ++i) {
            std::vector<int> neighIdx(2);
            std::vector<float> neighDist(2);
            
            if (matchTree.nearestKSearch(features_1.descriptors->at(i), 2, neighIdx, neighDist) >= 2) {
                float dist1 = std::sqrt(neighDist[0]);
                float dist2 = std::sqrt(neighDist[1]);

                if (dist1 < 0.75f*dist2) {
                    const pcl::PointXYZI &p1 = features_1.keypoints->points[i];
                    const pcl::PointXYZI &p2 = features_2.keypoints->points[neighIdx[0]];
                    matched_points.emplace_back(Eigen::Vector3f(p1.x, p1.y, p1.z), 
                                                Eigen::Vector3f(p2.x, p2.y, p2.z));
                }
            }
        }

        std::cout << "Number of matched points: " << matched_points.size() << std::endl;
        return matched_points;
    }

    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> find_correspondances_euclidean(
            pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_1,
            pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_2,
            Eigen::Matrix3f R, Eigen::Vector3f t)
    {  
        // find closest points
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> closest_points;

        for (auto p1 : point_cloud_1->points) 
        {
            float min_distance = std::numeric_limits<float>::infinity();
            Eigen::Vector3f closest_point;
            
            Eigen::Vector3f pt1(p1.x, p1.y, p1.z);
            Eigen::Vector3f pt1_transformed = R*pt1 + t;
            
            for (auto p2 : point_cloud_2->points) 
            {
                float dx = pt1_transformed.x() - p2.x;
                float dy = pt1_transformed.y() - p2.y;
                float dz = pt1_transformed.z() - p2.z;

                float distance = dx*dx + dy*dy + dz*dz;
                if (distance < min_distance)
                {
                    min_distance = distance;
                    closest_point << p2.x, p2.y, p2.z;
                }
            }

            closest_points.push_back(std::make_pair(pt1, closest_point)); 
        }

        return closest_points;
    }

    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> find_correspondances_kdtree(
            pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_1, 
            pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_2,
            Eigen::Matrix3f R, Eigen::Vector3f t) 
    {
        // initialize kd tree from the second point cloud
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
        kdtree.setInputCloud(point_cloud_2);

        // find nearest neighbors
        std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> closest_points;
        float avg_squared_dist = 0.0;
        int points_count = 0;

        for (auto p1 : point_cloud_1->points) 
        {  
            int K = 1;
            std::vector<int> point_idx(K);
            std::vector<float> squared_distance(K);

            // transform point p1
            Eigen::Vector3f pt1(p1.x, p1.y, p1.z);
            Eigen::Vector3f pt1_transformed = R*pt1 + t;

            pcl::PointXYZI p1_transformed(pt1_transformed.x(), pt1_transformed.y(), pt1_transformed.z(), p1.intensity);
            kdtree.nearestKSearch(p1_transformed, K, point_idx, squared_distance);
            
            avg_squared_dist += squared_distance[0];
            points_count++;

            // if (squared_distance[0] < 1.0f)
            // {
                pcl::PointXYZI closest_point = (*point_cloud_2)[point_idx[0]];
                closest_points.push_back(std::make_pair(pt1, Eigen::Vector3f(closest_point.x, closest_point.y, closest_point.z))); 
            //}
        }

        avg_squared_dist /= points_count;
        std::cout << "Average distance of points between is: " << avg_squared_dist << std::endl;
        
        return closest_points;
    }
};