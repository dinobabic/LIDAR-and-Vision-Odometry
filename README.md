# LIDAR and Vision Odometry

This repository contains code for LIDAR and Stereo Vision based odometry. Pipeline for evaluating alogirthms on the KITTI dataset has been implemented. Both implementations are wrapped in the ROS2 nodes for the visualization purposes in
the Rviz2 visualization tool. 

## Running the code
Create an empty directory and inside of it create directory src. In the src folder clone this repository. In the parent directory execute command colcon build, which will build ROS2 package from the cloned repo. Execute command source install/local_setup.bash.
Now you should be able to execute command ros2 run lidar_vo lidar_node or ros2 run lidar_vo slam_node, which will execute LIDAR and Stereo Vision based odometry respectively. Make sure to update paths to the KITTI dataset, where you store any of the sequences and ground truth data.

## Implementation
For the LIDAR based odometry, I implemented Iterative Closest Point (ICP) algorithm. Pipeline is as follows:
  - read two consecutive point clouds and store them as pcl::PointCloud data structures
  - downsample point clouds for efficiency
  - set initial transformation between two point clouds either as an identity transform or as the previously estimated transform between two previous point clouds
  - iterate:
      - determine correspondances between two point clouds - slow Euclidean distance based implementation or efficient implementation based on KDTree have been implemented
      - execute 50 iterations of RANSAC algorithm
  - concatenate estimated transform to obtain global pose of the vehicle
  - publish estimated trajectory, ground truth trajectory and transformed point cloud

To implement Stereo Vision odometry, I utilized stereo pairs provided in the KITTI dataset. Pose between two consecutive stereo pairs is computed using PnP algorithm. Pipeline is as follows:
  - read two consecutive stereo pairs
  - detect and match features between the left images in stereo pairs
  - match features between the earlier stereo pair (only that have match in the left image in the next stereo pair)
  - filter stereo matches - keep only those that have positive disparity and have same y coordinate (up to threhsold) - images have been rectified (horizontal epipolar lines)
  - drop temporal matches for which we no longer have correspondances in the stereo matches
  - triangulate stereo matches in the eralier stereo pair - this is possible because we know pose of the right camera relative to the left camera
  - we know have 3D-2D correspondances - perform PnP algorithm - result is transformation from the 3D points to the image plane
  - concatenate the inverse of the obtain transformation to obtain pose of the next stereo pair in the global coordinate frame

## TODO
- Sliding window bundle adjustment has been implemented using the g2o library, but it still has to be connected with the existing vision odometry pipeline.
- Implement loop closure detection using bag of words approach.
- When loop closure is detected, apply pose graph optimization algorithm to get more consistent trajectory.

## Visual examples
