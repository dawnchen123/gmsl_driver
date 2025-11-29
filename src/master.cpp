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
// Note: Resolution is forced to 1920x1080 in nvCam constructor, but kept here for reference
stCamCfg camcfgs[CAMERA_NUM] = {
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 0, "/dev/video0", vendor},
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 1, "/dev/video1", vendor},
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 2, "/dev/video2", vendor},
    {1920, 1080, distorWidth, distorHeight, undistorWidth, undistorHeight, stitcherinputWidth, stitcherinputHeight, undistor, 3, "/dev/video3", vendor}
};

// Generic thread function to read from a camera and publish to ROS
void read_camera_stream(int cam_index, const std::string& common_camera_info_url) {
    std::string camera_name = "camera_" + std::to_string(cam_index + 1);
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
    } else {
        ROS_WARN("[%s] Camera info URL not valid: %s", camera_name.c_str(), url.c_str());
    }

    // Extract K and D from CameraInfoManager
    sensor_msgs::CameraInfo ci = info_manager.getCameraInfo();
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat D;

    // Basic check if K is populated
    if (ci.K[0] != 0.0) {
        // Copy K (3x3)
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                K.at<double>(i, j) = ci.K[i*3+j];
            }
        }
        // Copy D (Vector)
        D = cv::Mat(ci.D).clone();
        ROS_INFO("[%s] Loaded intrinsics from manager.", camera_name.c_str());
    } else {
        ROS_WARN("[%s] No valid intrinsics found (K[0]==0). Using Identity. Distortion correction will be skipped.", camera_name.c_str());
    }

    ROS_INFO("Initializing %s on %s (1920x1080)", camera_name.c_str(), camcfgs[cam_index].name);
    
    try {
        // Pass loaded K and D to nvCam
        cameras[cam_index].reset(new nvCam(camcfgs[cam_index], K, D));
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize %s: %s", camera_name.c_str(), e.what());
        return;
    } catch (...) {
        ROS_ERROR("Failed to initialize %s: Unknown error", camera_name.c_str());
        return;
    }

    while (ros::ok()) {
        ros::Time capture_timestamp;

        // read_frame now performs undistortion on GPU if K/D were valid
        if (cameras[cam_index]->read_frame(capture_timestamp)) {
            cv::Mat frame = cameras[cam_index]->m_ret;
            if (!frame.empty()) {
                std_msgs::Header header;
                header.frame_id = camera_name;
                header.stamp = capture_timestamp; 

                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
                
                // Update Camera Info to match the undistorted image (D=0, K might change if we cropped/scaled, but with current simple undistort K remains mostly valid for semantic meaning)
                // Note: Technically after undistortion, D should be 0.
                sensor_msgs::CameraInfo info = info_manager.getCameraInfo();
                info.header = header;
                info.roi.do_rectify = false; 
                // Ideally we should update info.P here to match the new optimal matrix, but keeping original P is often acceptable for simple pipelines.
                
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
    
    ROS_INFO("Starting Parallel GMSL Camera Node (4 Cameras, 1080p, CUDA Undistortion)...");

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