#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "tensorflow/lite/c/c_api.h"
#ifdef WITH_GL
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

DEFINE_string(model_filename, "deeplabv3_257_mv_gpu.tflite", "Model filename");

// This is an int, because cv::VideoCapture(int) gives a higher resolution than
// cv::VideoCapture(const std::string&) (640x480, maybe b/c of an implicit gstreamer
// pipeline?)
DEFINE_int32(input_device_number, 0, "Input device number (/dev/videoX)");

DEFINE_string(output_device_path, "/dev/video2", "Output device");
DEFINE_string(image_filename, "bliss.jpg", "Image filename");

const char *label_names[] = {
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "board",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tv",
};


class BackgroundRemover {
private:
    TfLiteModel *model_;
    TfLiteInterpreterOptions *options_;
    TfLiteInterpreter *interpreter_;

    TfLiteTensor *input_;
    const TfLiteTensor *output_;
    int width_, height_, nlabels_;

#ifdef WITH_GL
    TfLiteDelegate *gpu_delegate_;
#endif

    std::string tensor_shape(const TfLiteTensor *t) {
        std::stringstream ret;
        ret << "[";
        for (int i = 0; i < TfLiteTensorNumDims(t); i++) {
            if (i > 0)
                ret << ", ";
            ret << TfLiteTensorDim(t, i);
        }
        ret << "]";
        return ret.str();
    }

public:
    BackgroundRemover(const char *model_filename, int num_threads=4) {
        static_assert(sizeof(float) == 4);

        model_ = CHECK_NOTNULL(TfLiteModelCreateFromFile(model_filename));

        options_ = CHECK_NOTNULL(TfLiteInterpreterOptionsCreate());

        TfLiteInterpreterOptionsSetNumThreads(options_, num_threads);
        TfLiteInterpreterOptionsSetErrorReporter(options_,
                (void(*)(void *, const char *, va_list))std::vfprintf, (void*)stderr);
#ifdef WITH_GL
        auto delegate_opts = TfLiteGpuDelegateOptionsV2Default();
        delegate_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        delegate_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        gpu_delegate_ = CHECK_NOTNULL(TfLiteGpuDelegateV2Create(&delegate_opts));
        TfLiteInterpreterOptionsAddDelegate(options_, gpu_delegate_);
#endif

        interpreter_ = CHECK_NOTNULL(TfLiteInterpreterCreate(model_, options_));
        TfLiteInterpreterAllocateTensors(interpreter_);

        input_ = CHECK_NOTNULL(TfLiteInterpreterGetInputTensor(interpreter_, 0));
        LOG(INFO) << "Input tensor: " << tensor_shape(input_);
        CHECK_EQ(TfLiteTensorType(input_), kTfLiteFloat32) << "input tensor must be float32";
        CHECK_EQ(TfLiteTensorNumDims(input_), 4) << "input tensor must have 4 dimensions";
        CHECK_EQ(TfLiteTensorDim(input_, 0), 1) << "input tensor batch size must be 1";
        width_ = TfLiteTensorDim(input_, 1);
        height_ = TfLiteTensorDim(input_, 2);
        CHECK_EQ(TfLiteTensorDim(input_, 3), 3) << "input tensor must have 3 channels";

        output_ = CHECK_NOTNULL(TfLiteInterpreterGetOutputTensor(interpreter_, 0));
        LOG(INFO) << "Output tensor: " << tensor_shape(output_);
        CHECK_EQ(TfLiteTensorType(output_), kTfLiteFloat32) << "output tensor must be float32";
        CHECK_EQ(TfLiteTensorNumDims(output_), 4) << "output tensor must have 4 dimensions";
        CHECK_EQ(TfLiteTensorDim(output_, 1), width_) << "output tensor width doesn't match input tensor width";
        CHECK_EQ(TfLiteTensorDim(output_, 2), height_) << "output tensor height doesn't match input tensor height";
        nlabels_ = TfLiteTensorDim(output_, 3);
        DCHECK_EQ(nlabels_, sizeof(label_names)/sizeof(label_names[0]));

        LOG(INFO) << "Initialized tflite with " << width_ << "x" << height_ << "px input for model " << model_filename;
    }

    void maskBackground(cv::Mat& frame /* rgb */, const cv::Mat& maskImage) {
        cv::Mat small;
        cv::resize(frame, small, cv::Size(width_, height_));

        cv::Mat input_float;
        small.convertTo(input_float, CV_32FC3, 1./255, -.5);

        CHECK_EQ(input_float.elemSize(), sizeof(float)*3 /* channels */);

        TfLiteTensorCopyFromBuffer(input_, (const void*)input_float.ptr<float>(),
                width_*height_*sizeof(float)*3);

        auto start = std::chrono::steady_clock::now();
        TfLiteInterpreterInvoke(interpreter_);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end-start;
        LOG(INFO) << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms";

        CHECK_EQ(TfLiteTensorByteSize(output_), width_*height_*nlabels_*sizeof(float));
        float *output = (float*)TfLiteTensorData(output_);

        cv::Mat mask = cv::Mat::zeros(cv::Size(width_, height_), CV_8U);

        int label_count[nlabels_] = {0};

        for (int y = 0; y < height_; y++) {
            float *row = &output[y * width_ * nlabels_];
            for (int x = 0; x < width_; x++) {
                float *col = &row[x * nlabels_];
                std::vector<size_t> labels(nlabels_);
                std::iota(labels.begin(), labels.end(), 0);
                std::stable_sort(labels.begin(), labels.end(),
                        [&](size_t a, size_t b) { return col[a] > col[b]; });

                if (labels[0] != 15) {
                    mask.at<unsigned char>(cv::Point(x, y)) = 1;
                }
            }
        }

        cv::resize(mask, mask, cv::Size(frame.cols, frame.rows));
        LOG(INFO) << "frame type: " << frame.type() << ", size: " << frame.size;
        LOG(INFO) << "maskImage type: " << maskImage.type() << ", size: " << maskImage.size;
        for (int x = 0; x < frame.cols; x++)
            for (int y = 0; y < frame.rows; y++)
                if (mask.at<unsigned char>(cv::Point(x, y)))
                    frame.at<cv::Vec3b>(cv::Point(x, y)) = maskImage.at<cv::Vec3b>(cv::Point(x, y));
        //frame.setTo(maskImage, mask);
        //cv::bitwise_and(frame, mask, frame);
    }

    ~BackgroundRemover() {
        TfLiteInterpreterDelete(interpreter_);
#ifdef WITH_GL
        TfLiteGpuDelegateV2Delete(gpu_delegate_);
#endif
        TfLiteInterpreterOptionsDelete(options_);
        TfLiteModelDelete(model_);
    }

};

std::ostream& operator<<(std::ostream& os, const struct v4l2_capability& cap) {
    std::ios_base::fmtflags f(os.flags());
    os << "struct v4l2_capability { .driver = \"" << (const char *)cap.driver << "\", .card = \""
        << (const char *)cap.card << "\", capabilities = " << std::hex << std::showbase
        << cap.capabilities << ", .device_caps = " << cap.device_caps << "}";
    os.flags(f);
    return os;
}

std::string fourcc(int v) {
    std::stringstream ss;
    ss << "v4l2_fourcc(";
    for (int i = 0; i < 32; i += 8) {
        if (i > 0)
            ss << ", ";
        ss << "'" << (char)((v >> i)&0xff) << "'";
    }
    ss << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const struct v4l2_format& fmt) {
    if (fmt.type == V4L2_BUF_TYPE_VIDEO_CAPTURE || fmt.type == V4L2_BUF_TYPE_VIDEO_OUTPUT) {
        std::ios_base::fmtflags f(os.flags());
        os << "struct v4l2_format { .type = " << fmt.type << ", "
            << ".fmt.pix = struct v4l2_pix_format { .width = " << fmt.fmt.pix.width
            << ", .height = " << fmt.fmt.pix.height
            << ", .pixelformat = " << fourcc(fmt.fmt.pix.pixelformat)
            << ", .field = " << fmt.fmt.pix.field << ", .bytesperline = " << fmt.fmt.pix.bytesperline
            << ", .sizeimage = " << fmt.fmt.pix.sizeimage << ", .colorspace = " << fmt.fmt.pix.colorspace
            << ", .priv = " << fmt.fmt.pix.priv << ", .flags = " << std::hex << std::showbase
            << fmt.fmt.pix.flags << "} }";
        os.flags(f);
    } else {
        os << "struct v4l2_format { .type = " << fmt.type << ", ??? }";
    }
    return os;
}

class VideoWriter {
    const int fd_, width_, height_, bpp_;

    int bytesPerPixel(int format) const {
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

public:
    VideoWriter(const char *device_name, int width, int height, int pixelformat=V4L2_PIX_FMT_BGR24)
    : width_(width)
    , height_(height)
    , fd_(open(device_name, O_WRONLY))
    , bpp_(bytesPerPixel(pixelformat))
    {
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
        CHECK_EQ(fmt.fmt.pix.bytesperline, bpp_*width);
        CHECK_EQ(fmt.fmt.pix.sizeimage, bpp_*width*height);
    }

    void writeFrame(const cv::Mat& frame) {
        CHECK(frame.isContinuous());
        CHECK_EQ(frame.cols, width_);
        CHECK_EQ(frame.rows, height_);
        CHECK_EQ(frame.elemSize(), bpp_);
        int total = width_*height_*bpp_;
        int ret = write(fd_, frame.data, total);
        PCHECK(ret > 0) << "Can't write " << total << " bytes to v4l loopback";

        if (ret < total)
            LOG(WARNING) << "write() truncated (wrote " << ret << ", want " << total << " bytes)";
        else
            LOG(INFO) << "Wrote a " << total << " bytes frame";
    }
};

int main(int argc, char **argv) {
    FLAGS_v = 1;
    FLAGS_logtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    BackgroundRemover bgr(FLAGS_model_filename.c_str());

    auto frameFormat = V4L2_PIX_FMT_RGB24;
    cv::VideoCapture cap(FLAGS_input_device_number);

    cv::Mat frame;
    cap >> frame; // Capture a frame to determine output WxH
    VideoWriter wri(FLAGS_output_device_path.c_str(), frame.cols, frame.rows, frameFormat);

    cv::Mat maskImage;
    if (FLAGS_image_filename.size()) {
        maskImage = cv::imread(FLAGS_image_filename);
        cv::resize(maskImage, maskImage, cv::Size(frame.cols, frame.rows));
        cv::cvtColor(maskImage, maskImage, cv::COLOR_BGR2RGB);
        LOG(INFO) << "mask image: " << maskImage;
    } else {
        maskImage = cv::Mat(cv::Size(frame.cols, frame.rows), CV_8UC3, cv::Scalar(255, 255, 255));
    }

    bool doMask = true;
    while (1) {
        cap >> frame;
        if (frame.empty()) {
            LOG(ERROR) << "Empty frame received";
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        if (doMask)
            bgr.maskBackground(frame, maskImage);

        wri.writeFrame(frame);

        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        cv::imshow("frame", frame);

        auto key = cv::waitKey(1);
        switch (key) {
            case ' ':
                doMask = !doMask;
                LOG(INFO) << (doMask ? "enabled" : "disabled") << " mask";
                break;

            case 'q':
                goto out;
        }
    }
out:

    return 0;
}
