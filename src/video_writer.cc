#include "video_writer.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>
#include <ostream>

#include "glog/logging.h"

static std::ostream& operator<<(std::ostream& os, const struct v4l2_capability& cap) {
    std::ios_base::fmtflags f(os.flags());
    os << "struct v4l2_capability { .driver = \"" << (const char*)cap.driver << "\", .card = \""
       << (const char*)cap.card << "\", capabilities = " << std::hex << std::showbase
       << cap.capabilities << ", .device_caps = " << cap.device_caps << "}";
    os.flags(f);
    return os;
}

static std::string fourcc(int v) {
    std::stringstream ss;
    ss << "v4l2_fourcc(";
    for (int i = 0; i < 32; i += 8) {
        if (i > 0) ss << ", ";
        ss << "'" << (char)((v >> i) & 0xff) << "'";
    }
    ss << ")";
    return ss.str();
}

static std::ostream& operator<<(std::ostream& os, const struct v4l2_format& fmt) {
    if (fmt.type == V4L2_BUF_TYPE_VIDEO_CAPTURE || fmt.type == V4L2_BUF_TYPE_VIDEO_OUTPUT) {
        std::ios_base::fmtflags f(os.flags());
        os << "struct v4l2_format { .type = " << fmt.type << ", "
           << ".fmt.pix = struct v4l2_pix_format { .width = " << fmt.fmt.pix.width
           << ", .height = " << fmt.fmt.pix.height
           << ", .pixelformat = " << fourcc(fmt.fmt.pix.pixelformat)
           << ", .field = " << fmt.fmt.pix.field << ", .bytesperline = " << fmt.fmt.pix.bytesperline
           << ", .sizeimage = " << fmt.fmt.pix.sizeimage
           << ", .colorspace = " << fmt.fmt.pix.colorspace << ", .priv = " << fmt.fmt.pix.priv
           << ", .flags = " << std::hex << std::showbase << fmt.fmt.pix.flags << "} }";
        os.flags(f);
    } else {
        os << "struct v4l2_format { .type = " << fmt.type << ", ??? }";
    }
    return os;
}

constexpr int bytesPerPixel(int format) {
    switch (format) {
        case V4L2_PIX_FMT_YUYV:
            return 2;
        case V4L2_PIX_FMT_RGB24:
        case V4L2_PIX_FMT_BGR24:
            return 3;
        case V4L2_PIX_FMT_RGB32:
        case V4L2_PIX_FMT_BGR32:
            return 4;
        default:
            CHECK(0) << "Unknown format " << format;
            return -1;
    }
}

VideoWriter::VideoWriter(const char* device_name, int width, int height, int pixelformat)
    : width_(width),
      height_(height),
      fd_(open(device_name, O_WRONLY)),
      bpp_(bytesPerPixel(pixelformat)) {
    CHECK(bpp_ > 0) << "Can't determine bytes per pixel for format " << pixelformat;
    PCHECK(fd_ >= 0) << "Can't open " << device_name;

    struct v4l2_capability cap;
    PCHECK(ioctl(fd_, VIDIOC_QUERYCAP, &cap) != -1);
    LOG(INFO) << "Capabilities of " << device_name << ": " << cap;

    struct v4l2_format fmt;
    if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
        LOG(WARNING) << "Can't query video format: " << strerror(errno);
    } else {
        LOG(INFO) << "Current video format: " << fmt;
    }

    memset(&fmt, 0, sizeof(fmt));
    // https://www.kernel.org/doc/html/v4.14/media/uapi/v4l/vidioc-g-fmt.html#c.v4l2_format
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    // https://www.kernel.org/doc/html/v4.14/media/uapi/v4l/pixfmt.html
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    fmt.fmt.pix.bytesperline = 0;
    fmt.fmt.pix.sizeimage = 0;
    fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
    PCHECK(ioctl(fd_, VIDIOC_S_FMT, &fmt) != 1) << "Can't set video format " << fmt;
    LOG(INFO) << "Set video format: " << fmt;
    CHECK_EQ(fmt.fmt.pix.bytesperline, bpp_ * width);
    CHECK_EQ(fmt.fmt.pix.sizeimage, bpp_ * width * height);
}

void VideoWriter::writeFrame(const cv::Mat& frame) {
    CHECK(frame.isContinuous());
    CHECK_EQ(frame.cols, width_);
    CHECK_EQ(frame.rows, height_);
    CHECK_EQ(frame.elemSize(), bpp_);
    int total = width_ * height_ * bpp_;
    int ret = write(fd_, frame.data, total);
    PCHECK(ret > 0) << "Can't write " << total << " bytes to v4l loopback";

    if (ret < total)
        LOG(WARNING) << "write() truncated (wrote " << ret << ", want " << total << " bytes)";
}
