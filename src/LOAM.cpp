#include <ros/ros.h>
#include <mutex>
#include <memory>
#include <deque>
#include <vector>

// ROS messages
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/uniform_sampling.h>

// Custom headers
#include "LOAM.hpp"

/**
 * @brief Load the prior map from a PCD file if enabled, downsample if `pmap_leaf_size` is valid, and initialize KdTree.
 * @param use_priormap If `false`, do not load the prior map.
 * @param priormap_file Path to the prior map file (.pcd)
 * @param pmap_leaf_size Voxel size for downsampling the prior map (if applicable)
 * @param priormap Variable to store the prior map (PointCloud)
 * @param kdTreeMap KdTree for querying points in the prior map
 * @return true if successfully loaded or if `use_priormap = false`, false if there is an error loading the prior map.
 */
bool loadPriorMap(bool use_priormap, const std::string &priormap_file, double pmap_leaf_size,
                  CloudXYZIPtr &priormap, KdFLANNPtr &kdTreeMap)
{
    if (!use_priormap)
    {
        ROS_WARN("Prior map is disabled. Skipping loading.");
        return false;
    }

    if (priormap_file.empty())
    {
        ROS_WARN("No prior map file provided.");
        return false;
    }

    // Load data from the PCD file into the point cloud
    priormap.reset(new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(priormap_file, *priormap) == -1)
    {
        ROS_ERROR("Could not load prior map from %s", priormap_file.c_str());
        return false;
    }

    ROS_INFO("Loaded prior map with %zu points", priormap->size());

    // Reduce data density if needed
    if (pmap_leaf_size > 0)
    {
        pcl::UniformSampling<pcl::PointXYZI> downsampler;
        downsampler.setRadiusSearch(pmap_leaf_size);
        downsampler.setInputCloud(priormap);

        CloudXYZIPtr downsampledMap(new pcl::PointCloud<pcl::PointXYZI>());
        downsampler.filter(*downsampledMap);
        priormap.swap(downsampledMap);

        ROS_INFO("Downsampled prior map to %zu points with leaf size %.2f",
                 priormap->size(), pmap_leaf_size);
    }
    else
    {
        ROS_WARN("Skipping downsampling because pmap_leaf_size is not set.");
    }

    // Build KdTree for fast point queries
    kdTreeMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kdTreeMap->setInputCloud(priormap);

    ROS_INFO("Built KdTree for prior map.");
    return true;
}

void publishPose(ros::Publisher &odom_pub, const std::shared_ptr<LOAM> &loam_ptr, 
                 double processTime, double initial_time)
{
    // Compute the current pose from the trajectory
    SE3d currentPose = loam_ptr->GetTraj()->pose(processTime);

    // Log the computed pose
    ROS_INFO("Computed current pose:");
    ROS_INFO("  Position: [%.3f, %.3f, %.3f]",
             currentPose.translation().x(), 
             currentPose.translation().y(), 
             currentPose.translation().z());
    ROS_INFO("  Orientation (quat): [%.3f, %.3f, %.3f, %.3f]",
             currentPose.unit_quaternion().x(), 
             currentPose.unit_quaternion().y(),
             currentPose.unit_quaternion().z(), 
             currentPose.unit_quaternion().w());

    // Create and set up the odometry message
    nav_msgs::Odometry odomMsg;
    odomMsg.header.stamp = ros::Time(processTime + initial_time);  // Use absolute timestamp
    odomMsg.header.frame_id = "world";
    odomMsg.child_frame_id = "lidar_0_body";

    odomMsg.pose.pose.position.x = currentPose.translation().x();
    odomMsg.pose.pose.position.y = currentPose.translation().y();
    odomMsg.pose.pose.position.z = currentPose.translation().z();

    odomMsg.pose.pose.orientation.x = currentPose.unit_quaternion().x();
    odomMsg.pose.pose.orientation.y = currentPose.unit_quaternion().y();
    odomMsg.pose.pose.orientation.z = currentPose.unit_quaternion().z();
    odomMsg.pose.pose.orientation.w = currentPose.unit_quaternion().w();

    // Publish odometry
    odom_pub.publish(odomMsg);
}

// ROS node for LOAM processing using a queue-based approach with relative time handling.
class LOAMNode
{
public:
    LOAMNode(ros::NodeHandle &nh)
    {
        nh_ptr = boost::make_shared<ros::NodeHandle>(nh);
        init(nh);
    }

void init(ros::NodeHandle &nh)
{
ROS_INFO("Initializing LOAMNode...");

        // (1) Read parameters from the ROS Parameter Server
        bool use_priormap;
        std::string priormap_file;
        double pmap_leaf_size = 0.0;

        std::string init_pose_str;
        nh.param<std::string>("init_pose", init_pose_str, "0.0 0.0 0.0 0.0 0.0 0.0 1.0");

        std::istringstream ss(init_pose_str);
        double init_x, init_y, init_z, init_qx, init_qy, init_qz, init_qw;
        ss >> init_x >> init_y >> init_z >> init_qx >> init_qy >> init_qz >> init_qw;

        nh.param("use_priormap", use_priormap, true);  // Default to enabling prior map
        nh.getParam("priormap_file", priormap_file);
        nh.getParam("pmap_leaf_size", pmap_leaf_size);
        nh.param("cloud_ds", cloud_downsample_radius, 0.1);  // Default downsampling to 0.1m

        priormap.reset(new pcl::PointCloud<pcl::PointXYZI>());
        kdTreeMap.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
        // (2) Load prior map if enabled
        if (use_priormap && !priormap_file.empty())
        {
            if (pcl::io::loadPCDFile<pcl::PointXYZI>(priormap_file, *priormap) == -1)
            {
                ROS_ERROR("Could not load prior map from %s", priormap_file.c_str());
            }
            else
            {
                ROS_INFO("Loaded prior map with %zu points", priormap->size());

                // Reduce point cloud density if needed
                if (pmap_leaf_size > 0)
                {
                    pcl::UniformSampling<pcl::PointXYZI> downsampler;
                    downsampler.setRadiusSearch(pmap_leaf_size);
                    downsampler.setInputCloud(priormap);

                    CloudXYZIPtr downsampledMap(new pcl::PointCloud<pcl::PointXYZI>());
                    downsampler.filter(*downsampledMap);
                    priormap.swap(downsampledMap);

                    ROS_INFO("Downsampled prior map to %zu points with leaf size %.2f",
                             priormap->size(), pmap_leaf_size);
                }

                // Build KdTree
                kdTreeMap->setInputCloud(priormap);
                ROS_INFO("Built KdTree for prior map.");
            }
        }
        else
        {
            ROS_WARN("Prior map is disabled or file not provided.");
        }

        // (3) Initialize system variables
        initial_time = -1.0;
        double t0 = 0.0;
        int lidarIndex = 0;

        myTf<double> T_init_tf;
        T_init_tf.pos << init_x, init_y, init_z;
        T_init_tf.rot = Eigen::Quaternion<double>(init_qw, init_qx, init_qy, init_qz);

        loam_ptr = std::make_shared<LOAM>(
            nh_ptr,
            nh_mutex,
            T_init_tf.getSE3(),
            t0,
            lidarIndex
        );

        ROS_INFO("LOAM system initialized.");

        // (4) Set up Subscriber (Receive data from LiDAR)
        sub_ = nh_ptr->subscribe("/os_cloud_node/points", 1, &LOAMNode::cloudCallback, this);
        ROS_INFO("Subscribed to /os_cloud_node/points");

        // (5) Set up Publishers
        odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/loam_odom", 10);
        cloudinW_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/cloudinW", 10);
        priormap_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap", 1);

        // Publish priormap
        Util::publishCloud(priormap_pub, *priormap, ros::Time::now(), "world");

        // (6) Create buffer processing timer
        processing_timer = nh_ptr->createTimer(ros::Duration(0.2), &LOAMNode::processBuffer, this);

        ROS_INFO("LOAMNode initialization completed.");
}
    
    // Callback for receiving point cloud messages.
    // Converts each incoming message to CloudXYZIT and pushes it into the queue.
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        double msg_time = cloud_msg->header.stamp.toSec();
        // ROS_INFO("Entered cloudCallback at time: %.3f", msg_time);
        
        // If this is the first message, initialize initial_time and update trajectory start time.
        if (initial_time < 0) {
            initial_time = msg_time;
            // ROS_INFO("Initialized initial_time: %.3f", initial_time);
            loam_ptr->GetTraj()->setStartTime(0.0);
        }
        
        // Convert ROS PointCloud2 message to pcl::PointCloud<PointXYZI>
        pcl::PointCloud<PointXYZI> tempCloud;
        pcl::fromROSMsg(*cloud_msg, tempCloud);
        // ROS_INFO("Converted cloud has %zu points", tempCloud.size());
        
        // Convert to CloudXYZIT and assign the 't' field from the message header.
        // We store time as relative to initial_time.
        CloudXYZITPtr rawCloud(new CloudXYZIT);
        rawCloud->resize(tempCloud.size());
        double rel_time = msg_time - initial_time;
        for (size_t i = 0; i < tempCloud.size(); ++i) {
            rawCloud->points[i].x = tempCloud.points[i].x;
            rawCloud->points[i].y = tempCloud.points[i].y;
            rawCloud->points[i].z = tempCloud.points[i].z;
            rawCloud->points[i].intensity = tempCloud.points[i].intensity;
            rawCloud->points[i].t = rel_time;
        }
        
        // Push the cloud into the queue (thread-safe).
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            cloud_queue.push_back(rawCloud);
            // ROS_INFO("Pushed cloud to queue, queue size: %zu", cloud_queue.size());
        }
    }
    
    // Timer callback to process clouds from the queue one by one.
    void processBuffer(const ros::TimerEvent &)
    {
        CloudXYZITPtr cloudToProcess;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            if (cloud_queue.empty())
                return;
            // Retrieve the first cloud from the queue.
            cloudToProcess = cloud_queue.front();
            cloud_queue.pop_front();
        }
        double processTime = (cloudToProcess->empty()) ? 0.0 : cloudToProcess->points[0].t;
        // ROS_INFO("Processing cloud from queue, relative time: %.3f, size: %zu", processTime, cloudToProcess->size());
        
        // Downsample the cloud using UniformSampling.
        // Use the overload that outputs a temporary point cloud to preserve all fields.
        CloudXYZITPtr downsampledCloud(new CloudXYZIT);
        // ROS_INFO("Downsampling cloud of size: %zu", cloudToProcess->size());
        {
            pcl::UniformSampling<PointXYZIT> downsampler;
            downsampler.setRadiusSearch(cloud_downsample_radius);
            downsampler.setInputCloud(cloudToProcess);
            pcl::PointCloud<PointXYZIT> tempFiltered;
            downsampler.filter(tempFiltered);
            *downsampledCloud = tempFiltered;
        }
        // ROS_INFO("Downsampled cloud has %zu points", downsampledCloud->size());
        if (downsampledCloud->empty()) {
            ROS_WARN("Downsampled cloud is empty, skipping processing.");
            return;
        }
        // ROS_INFO("First point timestamp in downsampled cloud (relative): %.3f", downsampledCloud->points[0].t);
        
        // Ensure the trajectory is extended to cover processTime.
        while (loam_ptr->GetTraj()->getMaxTime() < processTime) {
            loam_ptr->GetTraj()->extendOneKnot(loam_ptr->GetTraj()->getKnot(loam_ptr->GetTraj()->getNumKnots()-1));
            // ROS_INFO("Extended trajectory to cover time: %.3f", loam_ptr->GetTraj()->getMaxTime());
        }
        
        // Deskew the cloud.
        CloudXYZIPtr deskewedCloud(new CloudXYZI);
        // ROS_INFO("Starting deskew processing.");
        loam_ptr->Deskew(loam_ptr->GetTraj(), downsampledCloud, deskewedCloud);
        ROS_INFO("Deskewed cloud has %zu points", deskewedCloud->size());
        
        // Transform the deskewed cloud into world coordinates.
        CloudXYZIPtr cloudInW(new CloudXYZI);
        {
            // Get the timestamp of the last point in the deskewed cloud.
            double t_last = downsampledCloud->points.back().t;
            // Retrieve the pose at t_last from the trajectory.
            SE3d pose = loam_ptr->GetTraj()->pose(t_last);
            // Transform the cloud into the world coordinate frame.
            pcl::transformPointCloud(*deskewedCloud, *cloudInW, pose.translation(), pose.so3().unit_quaternion());
        }
        ROS_INFO("Successfully transformed deskewed cloud into world coordinates.");

        std::vector<LidarCoef> Coef;
        ROS_INFO("Starting Associate processing.");
        if (!priormap ) {
            ROS_ERROR("One or more input pointers are nullptr!");
            return;
        }
        loam_ptr->Associate(loam_ptr->GetTraj(), kdTreeMap, priormap, downsampledCloud, deskewedCloud, cloudInW, Coef);
        ROS_INFO("Associated features: %zu", Coef.size());

        // Initialize pose variables for optimization (initially using the current trajectory pose)
        Eigen::Quaterniond current_q = loam_ptr->GetTraj()->pose(processTime).unit_quaternion();
        Eigen::Vector3d current_t = loam_ptr->GetTraj()->pose(processTime).translation();

        // Store pose parameters into an array [qw, qx, qy, qz, tx, ty, tz]
        double pose_update[7] = { current_q.w(), current_q.x(), current_q.y(), current_q.z(),
                                  current_t.x(), current_t.y(), current_t.z() };

        ceres::Problem problem;

        // Add residual blocks for each feature in Coef
        for (const auto &coef : Coef) {
            // Create a cost function for each residual block using AutoDiff 
            // (Residual size = 3, pose has 7 parameters)
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<LoamResidual, 3, 7>(new LoamResidual(coef));
            problem.AddResidualBlock(cost_function, nullptr, pose_update);
        }

        // Configure solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;

        // Solve the optimization problem
        ceres::Solve(options, &problem, &summary);
        ROS_INFO("Ceres optimization complete: %s", summary.BriefReport().c_str());

        // After optimization, update the pose in the LOAM system.
        // Example: Update trajectory with the optimized pose
        Eigen::Quaterniond optimized_q(pose_update[0], pose_update[1], pose_update[2], pose_update[3]);
        Eigen::Vector3d optimized_t(pose_update[4], pose_update[5], pose_update[6]);
        loam_ptr->UpdatePose(optimized_q, optimized_t);

        Util::publishCloud(cloudinW_pub, *cloudInW, ros::Time::now(), "world");

        publishPose(odom_pub, loam_ptr, processTime, initial_time);

    }
    
private:
    boost::shared_ptr<ros::NodeHandle> nh_ptr;
    std::mutex nh_mutex;
    std::shared_ptr<LOAM> loam_ptr;

    ros::Subscriber sub_;
    ros::Publisher odom_pub;
    ros::Publisher cloudinW_pub;
    ros::Publisher priormap_pub; 

    // Queue for storing incoming point clouds.
    std::deque<CloudXYZITPtr> cloud_queue;
    std::mutex buffer_mutex;
    
    double cloud_downsample_radius;
    ros::Timer processing_timer;
    
    // initial_time is the absolute time (from the first message) used to compute relative time.
    double initial_time;

    CloudXYZIPtr priormap;  // Prior map stored as a point cloud
    KdFLANNPtr kdTreeMap;   // KdTree for fast point queries

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "loam_node");
    ros::NodeHandle nh("~");
    
    LOAMNode node(nh);
    ros::spin();
    return 0;
}
