#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <camera_info_manager/camera_info_manager.h>

#include <string>
#include <thread>
#include <memory>
#include <vector>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include "nvcam.hpp"
#include "stitcherglobal.h"

using namespace cv;

static std::shared_ptr<nvCam> cameras[CAMERA_NUM];

// Configuration for 4 cameras
stCamCfg camcfgs[CAMERA_NUM] = {
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 0, "/dev/video0", vendor},
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 1, "/dev/video1", vendor},
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 2, "/dev/video2", vendor},
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 3, "/dev/video3", vendor}
};

// Thread function: Parallel Read & Publish
void read_camera_stream(int cam_index, const std::string& common_camera_info_url) {
    std::string camera_name = "camera_" + std::to_string(cam_index + 1);
    
    // Use private node handle for proper namespacing
    ros::NodeHandle nh_cam("~/" + camera_name); 

    image_transport::ImageTransport it(nh_cam);
    image_transport::CameraPublisher pub = it.advertiseCamera("image_raw", 1);

    std::string url = common_camera_info_url;
    if (nh_cam.hasParam("camera_info_url")) {
        nh_cam.getParam("camera_info_url", url);
    }

    camera_info_manager::CameraInfoManager info_manager(nh_cam, camera_name, url);
    if (info_manager.validateURL(url)) {
        info_manager.loadCameraInfo(url);
    } 

    ROS_INFO("Initializing %s on %s", camera_name.c_str(), camcfgs[cam_index].name);
    
    try {
        cameras[cam_index].reset(new nvCam(camcfgs[cam_index]));
    } catch (...) {
        ROS_ERROR("Failed to initialize %s", camera_name.c_str());
        return;
    } 

    while (ros::ok()) {
        ros::Time capture_timestamp;

        // [Parallelism]: This call blocks until hardware interrupt. 
        // Since all cameras are hardware triggered, all threads wake up here simultaneously.
        // CUDA processing inside read_frame is also parallelized via Streams.
        if (cameras[cam_index]->read_frame(capture_timestamp)) {
            cv::Mat frame = cameras[cam_index]->m_ret;
            if (!frame.empty()) {
                std_msgs::Header header;
                header.frame_id = camera_name;
                // Use the timestamp captured immediately after hardware wake-up
                header.stamp = capture_timestamp; 

                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
                
                sensor_msgs::CameraInfo info = info_manager.getCameraInfo();
                info.header = header;
                
                pub.publish(*msg, info);
            }
        } else {
            ROS_WARN_THROTTLE(5, "Failed to read frame from %s", camera_name.c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gmsl_camera");
    ros::NodeHandle nh("~"); 

    std::string camera_info_url = "";
    if (!nh.getParam("camera_info_url", camera_info_url)) {
        ROS_INFO("Global 'camera_info_url' param not set."); 
    }
    
    ROS_INFO("Starting Parallel GMSL Camera Node (4 Cameras, 1080p, CUDA Streams)...");

    // Launch 4 threads to handle I/O and GPU submission in parallel
    std::vector<std::thread> cam_threads;
    for(int i = 0; i < CAMERA_NUM; ++i) {
        cam_threads.emplace_back(read_camera_stream, i, camera_info_url);
    }

    ros::spin();

    for(auto& t : cam_threads) {
        if(t.joinable()) t.join();
    }

    return 0;
}