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

// Make sure your custom point type CloudXYZIT is registered in LOAM.hpp like so:
// struct PointXYZIT {
//     PCL_ADD_POINT4D;
//     float intensity;
//     double t;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
// } EIGEN_ALIGN16;
// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
//     (float, x, x)
//     (float, y, y)
//     (float, z, z)
//     (float, intensity, intensity)
//     (double, t, t)
// )

// A simple ROS node wrapping LOAM with data stream buffering, processing,
// and publishing odometry. It uses the timestamp provided by the rosbag messages.
class LOAMNode
{
public:
    LOAMNode(ros::NodeHandle &nh)
    {
        // Create a shared NodeHandle pointer.
        nh_ptr = boost::make_shared<ros::NodeHandle>(nh);
        
        // Read parameters for buffering and downsampling.
        nh_ptr->param("buffer_duration", buffer_duration, 1.0); // seconds.
        nh_ptr->param("cloud_ds", cloud_downsample_radius, 0.1);  // meters.
        
        // Initialize the point cloud buffer.
        cloud_buffer.reset(new CloudXYZIT);
        // Khởi tạo buffer_start_time với giá trị không hợp lệ.
        buffer_start_time = -1.0;
        
        // Pose initialization using mytf (from utility.h).
        mytf T_init_tf;
        double t0 = ros::Time::now().toSec();
        int lidarIndex = 0;
        
        // Create the LOAM object (constructor expects lvalue parameters).
        loam_ptr = std::make_shared<LOAM>(
            nh_ptr,
            nh_mutex,
            T_init_tf.getSE3(),
            t0,
            lidarIndex
        );
        
        // Subscribe to the rosbag topic "/os_cloud_node/points".
        sub_ = nh_ptr->subscribe("/os_cloud_node/points", 1, 
                                 &LOAMNode::cloudCallback,
                                 this);
        ROS_INFO("LOAMNode: subscribed to /os_cloud_node/points");
        
        // Advertise topics to publish Associate, Visualize, and Odometry results.
        associate_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loam_associate", 1);
        visualize_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loam_visualize", 1);
        odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/loam_odom", 1);
    }

    // Callback for incoming point clouds.
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        double msg_time = cloud_msg->header.stamp.toSec();
        ROS_INFO("Entered cloudCallback at time: %.3f", msg_time);
        
        // Nếu buffer_start_time chưa được khởi tạo, gán bằng thời gian của thông điệp đầu tiên.
        if (buffer_start_time < 0)
        {
            buffer_start_time = msg_time;
            ROS_INFO("Initialized buffer_start_time: %.3f", buffer_start_time);
        }
        
        // Convert the incoming message to a temporary pcl::PointCloud<PointXYZI>.
        pcl::PointCloud<PointXYZI> tempCloud;
        pcl::fromROSMsg(*cloud_msg, tempCloud);
        ROS_INFO("Converted cloud has %zu points", tempCloud.size());
        
        // Create a CloudXYZIT and copy over x, y, z, intensity,
        // assigning the 't' field from the message header stamp.
        CloudXYZITPtr rawCloud(new CloudXYZIT);
        rawCloud->resize(tempCloud.size());
        for (size_t i = 0; i < tempCloud.size(); ++i) {
            rawCloud->points[i].x = tempCloud.points[i].x;
            rawCloud->points[i].y = tempCloud.points[i].y;
            rawCloud->points[i].z = tempCloud.points[i].z;
            rawCloud->points[i].intensity = tempCloud.points[i].intensity;
            rawCloud->points[i].t = msg_time;
        }
        
        // Accumulate the incoming cloud into the buffer.
        *cloud_buffer += *rawCloud;
        
        double dt = msg_time - buffer_start_time;
        ROS_INFO("Time difference dt: %.3f (buffer_duration: %.3f)", dt, buffer_duration);
        
        // Nếu dt chưa vượt quá buffer_duration, chờ thêm dữ liệu.
        if (dt < buffer_duration)
            return;
        
        ROS_INFO("Processing buffered cloud with %zu points", cloud_buffer->size());
        
        // Downsample the accumulated cloud using uniform sampling.
        CloudXYZITPtr downsampledCloud(new CloudXYZIT);
        {
            pcl::UniformSampling<PointXYZIT> downsampler;
            downsampler.setRadiusSearch(cloud_downsample_radius);
            downsampler.setInputCloud(cloud_buffer);
            downsampler.filter(*downsampledCloud);
        }
        
        // Create an output cloud for deskewed data (converted to PointXYZI type).
        CloudXYZIPtr deskewedCloud(new CloudXYZI);
        loam_ptr->Deskew(loam_ptr->GetTraj(), downsampledCloud, deskewedCloud);
        ROS_INFO("Buffered cloud deskewed: %zu points", deskewedCloud->size());
        
        // --- Associate and Visualize Processing ---
        KdFLANNPtr kdtreeMap = boost::make_shared<pcl::KdTreeFLANN<PointXYZI>>();
        CloudXYZIPtr priormap(new CloudXYZI); // Empty priormap (placeholder).
        CloudXYZIPtr cloudInB(new CloudXYZI);   // Placeholder for transformed cloud.
        CloudXYZIPtr cloudInW(new CloudXYZI);   // Placeholder for world-transformed cloud.
        std::vector<LidarCoef> Coef;
        
        loam_ptr->Associate(loam_ptr->GetTraj(), kdtreeMap, priormap, downsampledCloud, deskewedCloud, cloudInW, Coef);
        ROS_INFO("Associated features: %zu", Coef.size());
        
        CloudXYZIPtr associatedCloud(new CloudXYZI());
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
        associateMsg.header.stamp = cloud_msg->header.stamp;
        associateMsg.header.frame_id = "world";
        associate_pub.publish(associateMsg);
        
        std::deque<std::vector<LidarCoef>> swCloudCoef;
        swCloudCoef.push_back(Coef);
        
        sensor_msgs::PointCloud2 visualizedMsg;
        loam_ptr->Visualize(buffer_start_time, msg_time, swCloudCoef, associatedCloud, true);
        visualize_pub.publish(visualizedMsg);
        ROS_INFO("Published visualization result");
        
        // --- Odometry Publication ---
        SE3d currentPose = loam_ptr->GetTraj()->pose(msg_time);
        ROS_INFO("Computed current pose:");
        ROS_INFO("  Position: [%.3f, %.3f, %.3f]", currentPose.translation().x(), currentPose.translation().y(), currentPose.translation().z());
        ROS_INFO("  Orientation (quat): [%.3f, %.3f, %.3f, %.3f]", 
                 currentPose.unit_quaternion().x(), currentPose.unit_quaternion().y(), 
                 currentPose.unit_quaternion().z(), currentPose.unit_quaternion().w());
        
        nav_msgs::Odometry odomMsg;
        odomMsg.header.stamp = cloud_msg->header.stamp;
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
        
        // Clear the buffer and reset buffer_start_time to current message time.
        cloud_buffer->clear();
        buffer_start_time = msg_time;
    }

private:
    boost::shared_ptr<ros::NodeHandle> nh_ptr;
    std::mutex nh_mutex;
    std::shared_ptr<LOAM> loam_ptr;
    ros::Subscriber sub_;
    ros::Publisher associate_pub;
    ros::Publisher visualize_pub;
    ros::Publisher odom_pub;
    
    CloudXYZITPtr cloud_buffer;
    double buffer_start_time;
    double buffer_duration;
    double cloud_downsample_radius;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "loam_node");
    ros::NodeHandle nh("~");
    
    LOAMNode node(nh);
    ros::spin();
    return 0;
}
