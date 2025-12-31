#include "lidar_vo/utils_slam.hpp"
#include "lidar_vo/slam.hpp"

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

const std::string dataset_path = "/media/dino/T7/data_odometry_gray/dataset/sequences/00";


class SLAMNode : public rclcpp::Node
{
public:
    SLAMNode() : Node("slam_node"), index(0)
    {   
        // initialize ROS2 publishers and transformation broadcaster
        point_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);
        estimated_traj_pub = this->create_publisher<visualization_msgs::msg::Marker>("estimated_trajectory", 10);
        ground_truth_traj_pub = this->create_publisher<visualization_msgs::msg::Marker>("ground_truth_trajectory", 10);
        tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // initialize camera's projection matrices and the calibration matrix
        K << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00;

        auto f = K(0,0); // fx and fy are the same
        auto cx = 6.071928000000e+02;
        auto cy = 1.852157000000e+02;
        auto baseline = std::abs(-3.861448000000e+02/f); 

        P0 << f, 0,  cx, 0,
              0, f,  cy, 0,
              0, 0,  1, 0;

        P1 << f, 0,  cx, -f*baseline,
              0, f,  cy, 0,
              0, 0,  1,  0;

        slam = SLAM(K, P0, P1, baseline);
        total_estimated_pose = Eigen::Matrix4f::Identity();
        T = Sophus::SE3f();

        // read file names and ground truth poses
        image_names = get_filenames(dataset_path + "/image_0");
        ground_truth_poses = read_ground_truth("/media/dino/T7/data_odometry_poses/dataset/poses/00.txt");
        int N = image_names.size();

        // load first stereo pair
        stereo_pair_curr = load_stereo_pair(dataset_path, image_names[0]);
        for (int i = 1; i < N; i++) 
        {
            stereo_pair_prev = stereo_pair_curr;
            stereo_pair_curr = load_stereo_pair(dataset_path, image_names[i]);

            std::vector<Eigen::Vector3f> points3D;
            slam.vision_odometry(stereo_pair_prev, stereo_pair_curr, T, points3D);
            estimated_global_poses.push_back(T.matrix());

            auto stamp = this->get_clock()->now();
            publish_transform(T.matrix(), "map", "camera_frame", stamp);
            publish_trajectory(stamp, estimated_traj_pub, estimated_global_poses, estimated_global_poses.size(), std::vector<float>{1.0, 0.0, 0.0});
            publish_trajectory(stamp, ground_truth_traj_pub, ground_truth_poses, i, std::vector<float>{0.0, 0.0, 1.0});
        }
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr estimated_traj_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ground_truth_traj_pub;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    std::vector<std::string> image_names;
    int index;

    StereoPair stereo_pair_prev;
    StereoPair stereo_pair_curr;

    SLAM slam;
    Eigen::Matrix<float, 3, 4>  P0, P1; // projection matrices of the left and right camera
    Sophus::SE3f T;
    Eigen::Matrix3f K; // calibration matrix of both cameras
    std::vector<Eigen::Matrix4f> estimated_relative_poses;
    std::vector<Eigen::Matrix4f> estimated_global_poses;
    Eigen::Matrix4f total_estimated_pose;
    std::vector<Eigen::Matrix4f> ground_truth_poses;


    sensor_msgs::msg::PointCloud2 point_cloud_to_message(const std::vector<Eigen::Vector3f>& point_cloud)
    {
        sensor_msgs::msg::PointCloud2 msg;
        msg.height = 1;
        msg.width = point_cloud.size();
        msg.header.frame_id = "camera_frame"; // points are expressed in lidars local coordinate frame
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
            *it_x = p[0];
            *it_y = p[1];
            *it_z = p[2];
            *it_i = 1.0;//p.intensity;
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
    rclcpp::spin(std::make_shared<SLAMNode>());
    rclcpp::shutdown();
    return 0;
}
