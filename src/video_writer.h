#ifndef VIDEO_WRITER_H
#define VIDEO_WRITER_H

#include <opencv2/core.hpp>

#include <linux/videodev2.h>

class VideoWriter {
    const int fd_, width_, height_, bpp_;

public:
    VideoWriter(const char *device_name, int width, int height, int pixelformat=V4L2_PIX_FMT_BGR24);
    void writeFrame(const cv::Mat& frame);
};
#endif // VIDEO_WRITER_H
