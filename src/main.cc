#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "background_remover.h"
#include "background_selector.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "timing.h"
#include "video_writer.h"

DEFINE_string(model_filename, "deeplabv3_257_mv_gpu.tflite", "Model filename");
DEFINE_string(model_type, "deeplabv3", "Model type [deeplabv3|bodypix_resnet|bodypix_mobilenet]");

// This is an int, because cv::VideoCapture(int) gives a higher resolution than
// cv::VideoCapture(const std::string&) (640x480, maybe b/c of an implicit
// gstreamer pipeline?). XXX: Fix this.
DEFINE_int32(input_device_number, 0, "Input device number (/dev/videoX)");

DEFINE_string(output_device_path, "/dev/video2", "Output device");

DEFINE_string(image_dir, "./backgrounds/", "Directory to background images");
DEFINE_string(color_list, "ff0000,00ff00,0000ff",
              "Comma-separated list of background RRGGBB hex values");

int main(int argc, char **argv) {
    FLAGS_v = 1;
    FLAGS_logtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    BackgroundRemover bgr(FLAGS_model_filename, FLAGS_model_type);
    cv::VideoCapture cap(FLAGS_input_device_number);

    cv::Mat frame;
    cap >> frame;  // Capture a frame to determine output WxH
    CHECK(!frame.empty()) << "Empty frame captured from video input " << FLAGS_input_device_number;

    BackgroundSelector bgs(FLAGS_image_dir, FLAGS_color_list, frame.cols, frame.rows);

    VideoWriter wri(FLAGS_output_device_path.c_str(), frame.cols, frame.rows, V4L2_PIX_FMT_RGB24);

    Timing timing;
    auto timing_last_printed = Timing::now();

    bool doMask = true;
    while (1) {
        cap >> frame;
        if (frame.empty()) {
            LOG(ERROR) << "Empty frame received";
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        if (doMask) bgr.maskBackground(frame, bgs.getBackground(), timing);

        wri.writeFrame(frame);

        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        cv::imshow("frame", frame);

        auto key = cv::waitKey(1);
        switch (key) {
            case ' ':
                doMask = !doMask;
                LOG(INFO) << (doMask ? "enabled" : "disabled") << " mask";
                break;

            case 'C':
                bgs.selectPrevColor();
                break;

            case 'c':
                bgs.selectNextColor();
                break;

            case 'I':
                bgs.selectPrevImage();
                break;

            case 'i':
                bgs.selectNextImage();
                break;

            case 'q':
                goto out;
        }

        if (std::chrono::duration_cast<std::chrono::seconds>(Timing::now() - timing_last_printed)
                .count() >= 1) {
            LOG(INFO) << "fps: " << timing.nframes << ", timing: " << timing;
            timing.reset();
            timing_last_printed = Timing::now();
        }
    }
out:

    return 0;
}
