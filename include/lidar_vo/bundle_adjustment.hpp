    #include <g2o/core/base_vertex.h>
    #include <g2o/core/base_binary_edge.h>
    #include <g2o/core/block_solver.h>
    #include <g2o/core/optimization_algorithm_levenberg.h>
    #include <g2o/solvers/csparse/linear_solver_csparse.h>
    #include <g2o/core/robust_kernel_impl.h>
    #include <iostream>
    #include <sophus/se3.hpp>

    #include "lidar_vo/utils_slam.hpp"

    // structure to store extrinsic information for each camera
    struct Pose {
        Pose() {}
        Pose(const Sophus::SO3f &_R, const Eigen::Vector3f &_t) : R(_R), t(_t) {}
        Sophus::SO3f R;
        Eigen::Vector3f t;
    };

    // define a g2o vertex to store pose of each camera which needs to be optimized
    // 6 - number of optimization variables per camera
    class VertexPose : public g2o::BaseVertex<6, Pose>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexPose(const Eigen::Matrix3f &K_) : K(K_) {}

        virtual void setToOriginImpl() override
        {
            _estimate = Pose(Sophus::SO3f(), Eigen::Vector3f::Zero());
        }

        virtual void oplusImpl(const double *update) override
        {
            // update pointer points to the beginning of the vector containing update values
            // of each optimization variables
            
            Eigen::Vector3f rotation_update(
                static_cast<float>(update[0]),
                static_cast<float>(update[1]),
                static_cast<float>(update[2])
            );
            Eigen::Vector3f translation_update(
                static_cast<float>(update[3]),
                static_cast<float>(update[4]),
                static_cast<float>(update[5])
            );

            _estimate.R = Sophus::SO3f::exp(rotation_update) * _estimate.R;
            _estimate.t += translation_update;
        }

        Eigen::Vector2f project(const Eigen::Vector3f &point)
        {
            Eigen::Vector3f p_camera = _estimate.R*point + _estimate.t;
            p_camera /= p_camera[2];
            return Eigen::Vector2f(p_camera[0]*K(0,0) + K(0,2),
                                    p_camera[1]*K(1,1) + K(1,2));
        }

        virtual bool read(std::istream &in) {return true;}
        virtual bool write(std::ostream &out) const {return true;}

    private:
        Eigen::Matrix3f K;
    };

    // define a g2o vertex to store each 3D point
    // we optimize three coordinates of each point stored as Eigen::Vector3f
    class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3f>
    {
    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        VertexPoint() {}

        virtual void setToOriginImpl() override
        {
            _estimate = Eigen::Vector3f::Zero();
        }

        virtual void oplusImpl(const double *update) override
        {
            _estimate += Eigen::Vector3f(
                static_cast<float>(update[0]),
                static_cast<float>(update[1]),
                static_cast<float>(update[2])
            );
        }

        virtual bool read(std::istream &in) {return true;}
        virtual bool write(std::ostream &out) const {return true;}
    };

    // each edge is error term - error between true projection of 3D point in a camera and our estimated projection depending
    // on camera's parameters and position of a 3D point, which we both optimize
    // dimension of each error is 2 - error in x and y direction
    class EdgeProjection : 
        public g2o::BaseBinaryEdge<2, Eigen::Vector2f, VertexPose, VertexPoint>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeProjection() {
            _error = Eigen::Vector2d::Zero(); 
        }

        virtual void computeError() override
        {
            auto camera_vertex = (VertexPose*)_vertices[0];
            auto point_vertex = (VertexPoint*)_vertices[1];
            auto proj = camera_vertex->project(point_vertex->estimate());
            _error = (camera_vertex->project(point_vertex->estimate()) - _measurement).cast<double>();
        }

        virtual bool read(std::istream &in) {return true;}
        virtual bool write(std::ostream &out) const {return true;}
    };

    void solve_bundle_adjustment(
        std::vector<Sophus::SE3f, Eigen::aligned_allocator<Sophus::SE3f>>& camera_poses,
        std::vector<Eigen::Vector3f> &points3D,
        std::vector<Observation> observations,
        Eigen::Matrix3f K
    )
    {
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
        );
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        std::vector<VertexPose*> vertex_pose;
        std::vector<VertexPoint*> vertex_points;

        // add alll cameras to the optimization problem
        size_t i = 0;
        for (; i < camera_poses.size(); i++)
        {
            VertexPose *v = new VertexPose(K);
            v->setId(i);
            v->setEstimate(Pose(
                Sophus::SO3f(camera_poses[i].rotationMatrix()),
                camera_poses[i].translation() 
            ));
            
            optimizer.addVertex(v);
            vertex_pose.push_back(v);
        }

        // add all points to the optimization problem
        for (size_t j = 0; j < points3D.size(); j++)
        {
            VertexPoint *v = new VertexPoint();
            v->setId(i+j);
            v->setEstimate(points3D[j]);
            v->setMarginalized(true); // every point needs to be marginlized for efficiency
            optimizer.addVertex(v);
            vertex_points.push_back(v);
        }

        // connect vertices with edges for each observation
        for (i = 0; i < observations.size(); i++)
        {
            auto observation = observations[i];
            EdgeProjection *edge = new EdgeProjection();
            edge->setVertex(0, vertex_pose[observation.camera_id]);
            edge->setVertex(1, vertex_points[observation.point3D_idx]);
            edge->setMeasurement(observation.p);
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(4);

        // update optimized variables
        for (i = 0; i < vertex_pose.size(); i++)
        {
            auto v = vertex_pose[i];
            auto estimate = v->estimate();
            camera_poses[i] = Sophus::SE3f(estimate.R, estimate.t);
        }

        for (i = 0; i < vertex_points.size(); i++)
        {
            auto v = vertex_points[i];
            points3D[i] = v->estimate();
        }
    }





