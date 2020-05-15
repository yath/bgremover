#ifndef VIDEO_WRITER_H
#define VIDEO_WRITER_H

#include <linux/videodev2.h>

#include <opencv2/core.hpp>

class VideoWriter {
    const int fd_, width_, height_, bpp_;

   public:
    VideoWriter(const char* device_name, int width, int height, int pixelformat);
    void writeFrame(const cv::Mat& frame);
};
#endif  // VIDEO_WRITER_H
