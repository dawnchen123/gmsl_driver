/*
 * Optimized nvCam implementation with CUDA acceleration and Dynamic Distortion Correction
 */
#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <signal.h>
#include <poll.h>

#include <queue>
#include <mutex>
#include <condition_variable>

#include "NvUtils.h"
#include "NvCudaProc.h"
#include "nvbuf_utils.h"

#include "camera_v4l2-cuda.h"
#include "spdlog/spdlog.h"
#include "stitcherglobal.h"
#include "helper_timer.h"

using namespace std;

// Global statics (Legacy support)
static std::mutex m_mtx[8];
static std::condition_variable con[8];
static std::mutex changeszmtx;

// V4L2 Helpers
static bool camera_initialize(camcontext_t * ctx) {
    struct v4l2_format fmt;
    ctx->cam_fd = open(ctx->dev_name, O_RDWR);
    if (ctx->cam_fd == -1)
        return false;

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = ctx->cam_w;
    fmt.fmt.pix.height = ctx->cam_h;
    fmt.fmt.pix.pixelformat = ctx->cam_pixfmt;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(ctx->cam_fd, VIDIOC_S_FMT, &fmt) < 0) return false;

    struct v4l2_streamparm streamparm;
    memset (&streamparm, 0x00, sizeof (struct v4l2_streamparm));
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl (ctx->cam_fd, VIDIOC_G_PARM, &streamparm);

    return true;
}

static bool request_camera_buff(camcontext_t *ctx) {
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof(rb));
    rb.count = V4L2_BUFFERS_NUM;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_DMABUF;
    if (ioctl(ctx->cam_fd, VIDIOC_REQBUFS, &rb) < 0) return false;

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof buf);
        buf.index = index;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;
        if (ioctl(ctx->cam_fd, VIDIOC_QUERYBUF, &buf) < 0) return false;
        buf.m.fd = (unsigned long)ctx->g_buff[index].dmabuff_fd;
        if (ioctl(ctx->cam_fd, VIDIOC_QBUF, &buf) < 0) return false;
    }
    return true;
}

static NvBufferColorFormat get_nvbuff_color_fmt(unsigned int v4l2_pixfmt) {
    if (v4l2_pixfmt == V4L2_PIX_FMT_UYVY) return NvBufferColorFormat_UYVY;
    if (v4l2_pixfmt == V4L2_PIX_FMT_YUYV) return NvBufferColorFormat_YUYV;
    return NvBufferColorFormat_Invalid;
}

static bool prepare_buffers(camcontext_t * ctx) {
    NvBufferCreateParams input_params = {0};
    ctx->g_buff = (nv_buffer *)malloc(V4L2_BUFFERS_NUM * sizeof(nv_buffer));
    
    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = ctx->cam_w;
    input_params.height = ctx->cam_h;
    input_params.layout = NvBufferLayout_Pitch;
    input_params.colorFormat = get_nvbuff_color_fmt(ctx->cam_pixfmt);
    input_params.nvbuf_tag = NvBufferTag_CAMERA;

    for (unsigned int index = 0; index < V4L2_BUFFERS_NUM; index++) {
        int fd;
        if (-1 == NvBufferCreateEx(&fd, &input_params)) return false;
        ctx->g_buff[index].dmabuff_fd = fd;
        
        if (ctx->capture_dmabuf) {
            if (-1 == NvBufferMemMap(ctx->g_buff[index].dmabuff_fd, 0, NvBufferMem_Read_Write,
                        (void**)&ctx->g_buff[index].start))
                return false;
        }
    }
    return request_camera_buff(ctx);
}

static bool start_stream(camcontext_t * ctx) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMON, &type) < 0) return false;
    return true;
}

static bool stop_stream(camcontext_t * ctx) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->cam_fd, VIDIOC_STREAMOFF, &type)) return false;
    return true;
}

class nvCam
{
public:
    // Constructor updated to accept Camera Matrix (K) and Distortion Coeffs (D)
    nvCam(stCamCfg &camcfg, const cv::Mat& K, const cv::Mat& D):
        // [Forced Modification]: Hardcode 1920x1080 resolution
        m_camSrcWidth(1920),
        m_camSrcHeight(1080),
        m_retWidth(1920), // Keep output at 1080p
        m_retHeight(1080),
        m_id(camcfg.id)
    {
        // V4L2 Context Setup
        memset(&ctx, 0, sizeof(camcontext_t));
        strcpy(ctx.dev_name, camcfg.name);
        ctx.cam_pixfmt = V4L2_PIX_FMT_YUYV;
        ctx.cam_w = m_camSrcWidth;
        ctx.cam_h = m_camSrcHeight;
        ctx.capture_dmabuf = true;

        if (!camera_initialize(&ctx)) spdlog::critical("Cam {} Init Failed", m_id);
        if (!prepare_buffers(&ctx)) spdlog::critical("Cam {} Buffer Prep Failed", m_id);
        if (!start_stream(&ctx)) spdlog::critical("Cam {} Stream Start Failed", m_id);

        // NvBuffer for Color Conversion (YUYV -> ARGB)
        NvBufferCreateParams bufparams = {0};
        retNvbuf = (nv_buffer *)malloc(sizeof(nv_buffer));
        bufparams.payloadType = NvBufferPayload_SurfArray;
        bufparams.width = m_camSrcWidth;
        bufparams.height = m_camSrcHeight;
        bufparams.layout = NvBufferLayout_Pitch;
        bufparams.colorFormat = NvBufferColorFormat_ARGB32;
        bufparams.nvbuf_tag = NvBufferTag_NONE;
        NvBufferCreateEx(&retNvbuf->dmabuff_fd, &bufparams);

        // Initialize Transform Params
        memset(&transParams, 0, sizeof(transParams));
        transParams.transform_flag = NVBUFFER_TRANSFORM_FILTER;
        transParams.transform_filter = NvBufferTransform_Filter_Smart;

        // Initialize CPU Mats
        m_argb = cv::Mat(m_camSrcHeight, m_camSrcWidth, CV_8UC4);
        m_ret = cv::Mat(m_retHeight, m_retWidth, CV_8UC3);

        // Initialize GPU Mats
        m_gpuargb = cv::cuda::GpuMat(m_camSrcHeight, m_camSrcWidth, CV_8UC4);
        m_gpuret = cv::cuda::GpuMat(m_retHeight, m_retWidth, CV_8UC3); // Final result
        m_gpuRGB = cv::cuda::GpuMat(m_camSrcHeight, m_camSrcWidth, CV_8UC3); // Intermediate RGB
        
        // ---------------------------------------------------------
        // Efficient Distortion Correction Setup
        // ---------------------------------------------------------
        if(!K.empty() && !D.empty()) {
            m_undistor = true;
            cv::Size image_size(m_camSrcWidth, m_camSrcHeight);
            cv::Mat R = cv::Mat::eye(3,3,CV_32F);
            cv::Mat mapx_cpu, mapy_cpu;
            
            // getOptimalNewCameraMatrix with alpha=0 crops the image to remove black borders
            // This ensures the output image contains only valid pixels from the camera
            cv::Mat optMatrix = cv::getOptimalNewCameraMatrix(K, D, image_size, 0, image_size, 0);
            
            // Compute the remap maps once (CPU) and upload to GPU
            cv::initUndistortRectifyMap(K, D, R, optMatrix, image_size, CV_32FC1, mapx_cpu, mapy_cpu);
            
            gpuMapx.upload(mapx_cpu);
            gpuMapy.upload(mapy_cpu);
            spdlog::info("Camera {} Distortion Maps Generated and Uploaded to GPU", m_id);
        } else {
            m_undistor = false;
            spdlog::warn("Camera {} Missing Intrinsics (K/D empty). Undistortion Disabled.", m_id);
        }
    }

    ~nvCam() {
        stop_stream(&ctx);
        if (ctx.cam_fd > 0) close(ctx.cam_fd);
        if (ctx.g_buff) {
            for (unsigned i = 0; i < V4L2_BUFFERS_NUM; i++) 
                if (ctx.g_buff[i].dmabuff_fd) NvBufferDestroy(ctx.g_buff[i].dmabuff_fd);
            free(ctx.g_buff);
        }
        NvBufferDestroy(retNvbuf->dmabuff_fd);
    }

    // Reads frame, undistorts on GPU, and returns CPU Mat
    bool read_frame(ros::Time& capture_time)
    {
        struct v4l2_buffer v4l2_buf;
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_buf.memory = V4L2_MEMORY_DMABUF;

        // [BLOCKING] Wait for hardware frame
        if (ioctl(ctx.cam_fd, VIDIOC_DQBUF, &v4l2_buf) < 0) {
            spdlog::error("Cam {} DQBUF failed", m_id);
            return false;
        }
        
        capture_time = ros::Time::now();

        // [Hardware VIC] Convert YUYV to ARGB (Zero copy relative to CPU)
        if (-1 == NvBufferTransform(ctx.g_buff[v4l2_buf.index].dmabuff_fd, retNvbuf->dmabuff_fd, &transParams)) {
            return false;
        }

        // [Bottleneck] Map/Copy NvBuffer to CPU accessible memory
        // This is currently necessary to get data into a standard cv::Mat/GpuMat pipeline 
        // without complex EGL interop. NvBuffer2Raw is robust.
        NvBuffer2Raw(retNvbuf->dmabuff_fd, 0, m_camSrcWidth, m_camSrcHeight, m_argb.data);

        // Return buffer to driver ASAP
        if (ioctl(ctx.cam_fd, VIDIOC_QBUF, &v4l2_buf)) {
            spdlog::error("QBUF failed");
        }

        // [GPU Acceleration]
        try {
            // 1. Upload to GPU (ARGB)
            m_gpuargb.upload(m_argb, m_stream); 

            if(m_undistor) {
                // 2. Convert ARGB -> RGB (Async)
                // Remap requires RGB or specific types, and final output should be RGB for ROS
                cv::cuda::cvtColor(m_gpuargb, m_gpuRGB, cv::COLOR_RGBA2RGB, 0, m_stream);

                // 3. Remap (Undistort) (Async)
                // This is the most efficient way to undistort on GPU using precomputed maps
                cv::cuda::remap(m_gpuRGB, m_gpuret, gpuMapx, gpuMapy, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), m_stream);
                
            } else {
                // Just convert to RGB if no undistortion
                cv::cuda::cvtColor(m_gpuargb, m_gpuret, cv::COLOR_RGBA2RGB, 0, m_stream);
            }

            // 4. Download to CPU (Blocking wait for stream)
            m_gpuret.download(m_ret, m_stream);
            m_stream.waitForCompletion(); 

        } catch (cv::Exception& e) {
            spdlog::error("CUDA Error: {}", e.what());
            return false;
        }

        return true;
    }

public:
    camcontext_t ctx;
    int m_camSrcWidth, m_camSrcHeight;
    int m_retWidth, m_retHeight;
    int m_id;
    bool m_undistor;

    nv_buffer *retNvbuf;
    NvBufferTransformParams transParams;

    cv::Mat m_argb, m_ret;
    
    // CUDA resources
    cv::cuda::Stream m_stream;
    cv::cuda::GpuMat m_gpuargb;
    cv::cuda::GpuMat m_gpuRGB; 
    cv::cuda::GpuMat m_gpuret;
    cv::cuda::GpuMat gpuMapx, gpuMapy;
};