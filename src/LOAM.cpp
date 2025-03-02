#include "LOAM.hpp"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/uniform_sampling.h>
#include <mutex>
#include <memory>
#include <deque>
#include <vector>

// Ensure that your custom point type CloudXYZIT is registered in LOAM.hpp like so:
// struct PointXYZIT {
//     PCL_ADD_POINT4D;                  // Adds XYZ and padding
//     float intensity;
//     double t;                         // Custom timestamp field
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
// } EIGEN_ALIGN16;
// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
//     (float, x, x)
//     (float, y, y)
//     (float, z, z)
//     (float, intensity, intensity)
//     (double, t, t)
// )

// ROS node for LOAM processing using a queue-based approach with relative time handling.
class LOAMNode
{
public:
    LOAMNode(ros::NodeHandle &nh)
    {
        nh_ptr = boost::make_shared<ros::NodeHandle>(nh);
        
        // Read downsampling parameter from the parameter server.
        nh_ptr->param("cloud_ds", cloud_downsample_radius, 0.1);  // Downsampling radius in meters
        
        // Initialize the cloud queue.
        // Each incoming cloud is converted to CloudXYZIT and pushed into this queue.
        // They will be processed asynchronously via a ROS timer.
        initial_time = -1.0;
        
        // Pose initialization using mytf (from utility.h)
        mytf T_init_tf;
        // t0 will be set when the first cloud is received.
        double t0 = 0.0;
        int lidarIndex = 0;
        
        loam_ptr = std::make_shared<LOAM>(
            nh_ptr,
            nh_mutex,
            T_init_tf.getSE3(),
            t0,
            lidarIndex
        );
        
        // Subscribe to the LiDAR point cloud topic.
        sub_ = nh_ptr->subscribe("/os_cloud_node/points", 1,
                                 &LOAMNode::cloudCallback, this);
        ROS_INFO("LOAMNode: subscribed to /os_cloud_node/points");
        
        // Advertise output topics.
        associate_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loam_associate", 1);
        visualize_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loam_visualize", 1);
        odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/loam_odom", 1);
        
        // Create a timer to process queued clouds every 0.2 seconds.
        processing_timer = nh_ptr->createTimer(ros::Duration(0.2),
                                                &LOAMNode::processBuffer, this);
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
            // Removed direct reference to loam_ptr->T_W_Li0 because it is private.
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
        // ROS_INFO("Deskewed cloud has %zu points", deskewedCloud->size());
        
        // Feature Association.
        KdFLANNPtr kdtreeMap = boost::make_shared<pcl::KdTreeFLANN<PointXYZI>>();
        CloudXYZIPtr priormap(new CloudXYZI); // Placeholder for prior map.
        CloudXYZIPtr cloudInW(new CloudXYZI);
        std::vector<LidarCoef> Coef;
        // ROS_INFO("Starting Associate processing.");
        loam_ptr->Associate(loam_ptr->GetTraj(), kdtreeMap, priormap, downsampledCloud, deskewedCloud, cloudInW, Coef);
        ROS_INFO("Associated features: %zu", Coef.size());
        
        // Convert associated features into a point cloud for visualization.
        CloudXYZIPtr associatedCloud(new CloudXYZI);
        for (const auto &coef : Coef)
        {
            PointXYZI p;
            p.x = coef.finW.x();
            p.y = coef.finW.y();
            p.z = coef.finW.z();
            p.intensity = 1.0;
            associatedCloud->push_back(p);
        }
        
        sensor_msgs::PointCloud2 associateMsg;
        pcl::toROSMsg(*associatedCloud, associateMsg);
        // Convert relative processTime back to absolute time by adding initial_time.
        associateMsg.header.stamp = ros::Time(processTime + initial_time);
        associateMsg.header.frame_id = "world";
        associate_pub.publish(associateMsg);
        
        std::deque<std::vector<LidarCoef>> swCloudCoef;
        swCloudCoef.push_back(Coef);
        
        sensor_msgs::PointCloud2 visualizedMsg;
        loam_ptr->Visualize(processTime, processTime, swCloudCoef, associatedCloud, true);
        visualize_pub.publish(visualizedMsg);
        ROS_INFO("Published visualization result");
        
        // Compute and publish odometry.
        SE3d currentPose = loam_ptr->GetTraj()->pose(processTime);
        ROS_INFO("Computed current pose:");
        ROS_INFO("  Position: [%.3f, %.3f, %.3f]",
                 currentPose.translation().x(), currentPose.translation().y(), currentPose.translation().z());
        ROS_INFO("  Orientation (quat): [%.3f, %.3f, %.3f, %.3f]",
                 currentPose.unit_quaternion().x(), currentPose.unit_quaternion().y(),
                 currentPose.unit_quaternion().z(), currentPose.unit_quaternion().w());
        
        nav_msgs::Odometry odomMsg;
        odomMsg.header.stamp = ros::Time(processTime + initial_time);
        odomMsg.header.frame_id = "world";
        odomMsg.child_frame_id = "lidar_0_body";
        odomMsg.pose.pose.position.x = currentPose.translation().x();
        odomMsg.pose.pose.position.y = currentPose.translation().y();
        odomMsg.pose.pose.position.z = currentPose.translation().z();
        odomMsg.pose.pose.orientation.x = currentPose.unit_quaternion().x();
        odomMsg.pose.pose.orientation.y = currentPose.unit_quaternion().y();
        odomMsg.pose.pose.orientation.z = currentPose.unit_quaternion().z();
        odomMsg.pose.pose.orientation.w = currentPose.unit_quaternion().w();
        odom_pub.publish(odomMsg);
        ROS_INFO("Published odometry message");
    }
    
private:
    boost::shared_ptr<ros::NodeHandle> nh_ptr;
    std::mutex nh_mutex;
    std::shared_ptr<LOAM> loam_ptr;
    ros::Subscriber sub_;
    ros::Publisher associate_pub;
    ros::Publisher visualize_pub;
    ros::Publisher odom_pub;
    
    // Queue for storing incoming point clouds.
    std::deque<CloudXYZITPtr> cloud_queue;
    std::mutex buffer_mutex;
    
    double cloud_downsample_radius;
    ros::Timer processing_timer;
    
    // initial_time is the absolute time (from the first message) used to compute relative time.
    double initial_time;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "loam_node");
    ros::NodeHandle nh("~");
    
    LOAMNode node(nh);
    ros::spin();
    return 0;
}
