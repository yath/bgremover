#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>

#include "background_remover.h"
#include "background_selector.h"
#include "debug.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "timing.h"
#include "video_writer.h"

int debug_flags;

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

DEFINE_string(debug_flags, "show_output_frame",
              "Comma-separated list of debug flags (see debug.h)");

static int parseDebugFlags(const std::string& s) {
    int ret = 0;

    std::stringstream ss(s);
    for (std::string flag; std::getline(ss, flag, ',');) {
        if (flag == "show_output_frame")
            ret |= DebugFlagShowOutputFrame;
        else if (flag == "show_model_input_frame")
            ret |= DebugFlagShowModelInputFrame;
        else if (flag == "show_model_output")
            ret |= DebugFlagShowModelOutput;
        else
            CHECK(0) << "Unknown debug flag " << flag;
    }

    return ret;
}

int main(int argc, char** argv) {
    FLAGS_v = 1;
    FLAGS_logtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

    debug_flags = parseDebugFlags(FLAGS_debug_flags);


    BackgroundRemover bgr(FLAGS_model_filename, FLAGS_model_type);
    cv::VideoCapture cap(FLAGS_input_device_number);

    cv::Mat frozen_frame;

    cv::Mat frame;
    cap >> frame;  // Capture a frame to determine output WxH
    CHECK(!frame.empty()) << "Empty frame captured from video input " << FLAGS_input_device_number;

    BackgroundSelector bgs(FLAGS_image_dir, FLAGS_color_list, frame.cols, frame.rows);

    VideoWriter wri(FLAGS_output_device_path.c_str(), frame.cols, frame.rows, V4L2_PIX_FMT_RGB24);

    Timing timing;
    auto timing_last_printed = Timing::now();

    bool do_mask = true;
    bool do_blur_mask = true;
    bool do_blend_layers = false;
    bool is_frozen = false;
    while (1) {
        timing.nframes++;
        // The frame data is in BGR format.
        if (is_frozen)
            frozen_frame.copyTo(frame);
        else
            cap >> frame;

        if (frame.empty()) {
            LOG(ERROR) << "Empty frame received";
            break;
        }
	// Convert to RGB for deducing the mask and writing to the virtual camera.
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        if (do_mask && !is_frozen)
            bgr.maskBackground(frame, bgs.getBackground(), do_blur_mask, do_blend_layers, timing);

        // Frame is written as RGB to the virtual camera.
        wri.writeFrame(frame);

        // Swapping back to BGR is required for the preview as well as the freeze frame function.
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        if (debug_flags & DebugFlagShowOutputFrame) {
            cv::imshow("frame", frame);
        }

        auto key = cv::waitKey(is_frozen ? 1000 : 1);
        switch (key) {
            case ' ':
                do_mask = !do_mask;
                LOG(INFO) << (do_mask ? "enabled" : "disabled") << " mask";
                break;

            case 's':
                do_blur_mask = !do_blur_mask;
                LOG(INFO) << (do_blur_mask ? "enabled" : "disabled") << " mask blurring";
                break;

            case 'b':
                do_blend_layers = !do_blend_layers;
                LOG(INFO) << (do_blur_mask ? "enabled" : "disabled") << " alpha blending layers";
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

            case 'f':
                is_frozen = !is_frozen;
                LOG(INFO) << (is_frozen ? "enabled" : "disabled") << " frame freezing";
                if (is_frozen) {
                  // Store the current frame (BGR).
                  frame.copyTo(frozen_frame);
                }
                break;

            case 'q':
                goto out;
        }

        if (std::chrono::duration_cast<std::chrono::seconds>(Timing::now() - timing_last_printed)
                .count() >= 1) {
            LOG(INFO) << "fps: " << timing.nframes
                      << (is_frozen ? " (frozen)" : "")
                      << ", timing: " << timing;
            timing.reset();
            timing_last_printed = Timing::now();
        }
    }
out:

    return 0;
}
