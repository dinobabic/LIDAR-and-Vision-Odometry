#include "lidar_vo/utils_icp.hpp"
#include "lidar_vo/icp.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <vector>
#include <string>
#include <chrono>

#include <pcl/registration/icp.h>

class ICPNode : public rclcpp::Node
{
public:
    ICPNode() : Node("point_cloud_node")
    {
        point_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);
        estimated_traj_pub = this->create_publisher<visualization_msgs::msg::Marker>("estimated_trajectory", 10);
        pcl_estimated_traj_pub = this->create_publisher<visualization_msgs::msg::Marker>("pcl_estimated_trajectory", 10);
        ground_truth_traj_pub = this->create_publisher<visualization_msgs::msg::Marker>("ground_truth_trajectory", 10);
        tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // initial pose is identity
        T_wp = Sophus::SE3f();

        Eigen::Matrix4f T_cam0_velo_eigen; // transformation matrix that transforms from the lidar frame to the camera_0 frame
        T_cam0_velo_eigen << 
            4.276802e-04, -9.999672e-01, -8.084492e-03, -1.198460e-02,
            -7.210627e-03,  8.081198e-03, -9.999413e-01, -5.403985e-02,
            9.999739e-01,  4.859486e-04, -7.206934e-03, -2.921969e-01,
            0, 0, 0, 1;

        Sophus::SE3f T_cam0_velo(T_cam0_velo_eigen);

        // read names of point clouds
        point_clouds_names = get_filenames("/media/dino/T7/data_odometry_lidar/sequences/00/velodyne");

        // read ground truth poses
        ground_truth_poses = read_ground_truth("/media/dino/T7/data_odometry_poses/dataset/poses/00.txt", T_cam0_velo);
        
        point_cloud_next = read_ponint_cloud(point_clouds_names[0]);
        for (int i = 0; i < point_clouds_names.size() && rclcpp::ok(); i++)
        {   
            std::cout << "Iteration: " << i+1 << " / " << point_clouds_names.size() << std::endl;

            point_cloud_prev = point_cloud_next;
            point_cloud_next = read_ponint_cloud(point_clouds_names[i]);
            auto stamp = this->get_clock()->now();

            // cloudify points
            auto point_cloud_1 = icp.cloudify(point_cloud_prev);
            auto point_cloud_2 = icp.cloudify(point_cloud_next);

            // downsample point clouds
            downsample_cloud(point_cloud_1);
            downsample_cloud(point_cloud_2);

            // feature based icp - no need to iterate
            // auto matched_points = icp.find_correspondances_features_based(
            //     point_cloud_1,
            //     point_cloud_2
            // );

            // auto T = icp.ransac(matched_points, 50, 0.25f);
            
            // run pcl ICP implementation
            pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp_pcl;
            icp_pcl.setInputSource(point_cloud_1);
            icp_pcl.setInputTarget(point_cloud_2);
            pcl::PointCloud<pcl::PointXYZI> aligned;
            icp_pcl.align(aligned);
            Eigen::Matrix4f T_pcl = icp_pcl.getFinalTransformation();
            T_wp_pcl *= Sophus::SE3f(T_pcl).inverse();
            pcl_estimated_global_poses.push_back(T_wp_pcl.matrix());

            const int icp_iterations = 3;
            Eigen::Matrix4f T;
            
            // initial estimate for transformation
            Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
            Eigen::Vector3f t = Eigen::Vector3f::Zero();
            if (estimated_relative_poses.size() > 0)
            {
                auto T = estimated_relative_poses.back();
                R = T.block<3, 3>(0, 0);
                t = T.block<3, 1>(0, 3);
            }

            for (int iter = 0; iter < icp_iterations; ++iter)
                {
                // find closest points
                auto closest_points = icp.find_correspondances_kdtree(
                    point_cloud_1,
                    point_cloud_2,
                    R, t
                );

                // estimate the relative transformation
                T = icp.ransac(closest_points, 50, 0.25f);
                //T = icp.solve_icp(closest_points);
                R = T.block<3, 3>(0, 0);
                t = T.block<3, 1>(0, 3);
            }


            Sophus::SE3f T_curr_prev(T); // transformation from the prev to the curr point cloud
            estimated_relative_poses.push_back(T_curr_prev.matrix());
            T_wp = T_wp * T_curr_prev.inverse(); // pose of the current frame in the world frame
            estimated_global_poses.push_back(T_wp.matrix()); 

            std::cout << "PCL estimated transformation matrix: \n" << T_wp_pcl.matrix3x4() << std::endl;
            std::cout << "Estimated transformation matrix: \n" << T_wp.matrix3x4() << std::endl;
            std::cout << "Ground truth transformation matrix: \n" << ground_truth_poses[i] << std::endl;

            publish_transform(T_wp.matrix(), "map", "lidar_frame", stamp);
            publish_trajectory(stamp, estimated_traj_pub, estimated_global_poses, estimated_global_poses.size(), std::vector<float>{1.0, 0.0, 0.0});
            publish_trajectory(stamp, ground_truth_traj_pub, ground_truth_poses, estimated_global_poses.size(), std::vector<float>{0.0, 0.0, 1.0});
            publish_trajectory(stamp, pcl_estimated_traj_pub, pcl_estimated_global_poses, pcl_estimated_global_poses.size(), std::vector<float>{0.0, 1.0, 0.0});
        
            sensor_msgs::msg::PointCloud2 msg = point_cloud_to_message(point_cloud_next);
            msg.header.stamp = stamp;
            point_cloud_pub->publish(msg);
        }

        store_estimated_trajectory(estimated_global_poses, T_cam0_velo);
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr estimated_traj_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pcl_estimated_traj_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ground_truth_traj_pub;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    std::vector<std::string> point_clouds_names;

    std::vector<pcl::PointXYZI> point_cloud_prev;
    std::vector<pcl::PointXYZI> point_cloud_next;

    ICP icp;
    std::vector<Eigen::Matrix4f> estimated_relative_poses;
    std::vector<Eigen::Matrix4f> estimated_global_poses;
    std::vector<Eigen::Matrix4f> pcl_estimated_global_poses;
    Sophus::SE3f T_wp; // total estimated pose of the last processed point cloud in the world frame
    Sophus::SE3f T_wp_pcl; 
    std::vector<Eigen::Matrix4f> ground_truth_poses;

    sensor_msgs::msg::PointCloud2 point_cloud_to_message(const std::vector<pcl::PointXYZI>& point_cloud)
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.height = 1;
        msg.width = point_cloud.size();
        msg.header.frame_id = "lidar_frame"; // points are expressed in lidars local coordinate frame
        msg.header.stamp = this->get_clock()->now();

        sensor_msgs::PointCloud2Modifier modifier(msg);
        modifier.setPointCloud2Fields(
            4,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32
        );

        modifier.resize(point_cloud.size());

        sensor_msgs::PointCloud2Iterator<float> it_x(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> it_y(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> it_z(msg, "z");
        sensor_msgs::PointCloud2Iterator<float> it_i(msg, "intensity");

        for (const auto& p : point_cloud) {
            *it_x = p.x;
            *it_y = p.y;
            *it_z = p.z;
            *it_i = p.intensity;
            ++it_x; ++it_y; ++it_z; ++it_i;
        }

        return msg;
    }

    void publish_transform(const Eigen::Matrix4f &T, const std::string &parent_frame, const std::string &child_frame, rclcpp::Time stamp)
    {
        geometry_msgs::msg::TransformStamped transform_msg;
        transform_msg.header.stamp = stamp;
        transform_msg.header.frame_id = parent_frame;
        transform_msg.child_frame_id = child_frame;

        transform_msg.transform.translation.x = T(0, 3);
        transform_msg.transform.translation.y = T(1, 3);
        transform_msg.transform.translation.z = T(2, 3);

        Eigen::Matrix3f R = T.block<3,3>(0,0);
        Eigen::Quaternionf q(R);
        transform_msg.transform.rotation.x = q.x();
        transform_msg.transform.rotation.y = q.y();
        transform_msg.transform.rotation.z = q.z();
        transform_msg.transform.rotation.w = q.w();

        tf_broadcaster->sendTransform(transform_msg);
    }

    void publish_trajectory(const rclcpp::Time &stamp, 
                            rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher, 
                            const std::vector<Eigen::Matrix4f> &poses,
                            const int N,
                            const std::vector<float> &color)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = stamp;
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.scale.x = 0.1;
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = 1.0;

        for (auto i = 0; i < N; i++)
        {
            auto T = poses[i];
            geometry_msgs::msg::Point p;
            p.x = T(0, 3);
            p.y = T(1, 3);
            p.z = T(2, 3);
            marker.points.push_back(p);
        }

        publisher->publish(marker);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ICPNode>());
    rclcpp::shutdown();
    return 0;
}
