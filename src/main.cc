#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "background_remover.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "video_writer.h"

DEFINE_string(model_filename, "deeplabv3_257_mv_gpu.tflite", "Model filename");

// This is an int, because cv::VideoCapture(int) gives a higher resolution than
// cv::VideoCapture(const std::string&) (640x480, maybe b/c of an implicit
// gstreamer pipeline?)
DEFINE_int32(input_device_number, 0, "Input device number (/dev/videoX)");

DEFINE_string(output_device_path, "/dev/video2", "Output device");
DEFINE_string(image_filename, "bliss.jpg", "Image filename");

int main(int argc, char **argv) {
    FLAGS_v = 1;
    FLAGS_logtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    BackgroundRemover bgr(FLAGS_model_filename.c_str());

    auto frameFormat = V4L2_PIX_FMT_RGB24;
    cv::VideoCapture cap(FLAGS_input_device_number);

    cv::Mat frame;
    cap >> frame;  // Capture a frame to determine output WxH
    VideoWriter wri(FLAGS_output_device_path.c_str(), frame.cols, frame.rows, frameFormat);

    cv::Mat mask = cv::Mat(cv::Size(frame.cols, frame.rows), CV_8UC3, cv::Scalar(255, 255, 255));
    if (FLAGS_image_filename.size()) {
        cv::Mat img = cv::imread(FLAGS_image_filename);
        if (!img.empty()) {
            cv::resize(img, img, cv::Size(frame.cols, frame.rows));
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            mask = img;
            LOG(INFO) << "Using mask image " << FLAGS_image_filename;
        } else {
            LOG(WARNING) << "Can't load mask from image " << FLAGS_image_filename;
        }
    }

    bool doMask = true;
    while (1) {
        cap >> frame;
        if (frame.empty()) {
            LOG(ERROR) << "Empty frame received";
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        if (doMask) bgr.maskBackground(frame, mask);

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
