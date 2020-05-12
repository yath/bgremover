#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

extern "C" {
void eglCreateSyncKHR() { }
void eglClientWaitSyncKHR() { }
void eglWaitSyncKHR() { }
void eglDestroySyncKHR() { }
}

#define WITH_GL 1

DEFINE_string(model_filename, "deeplabv3_257_mv_gpu.tflite", "model filename");

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
        CHECK_EQ(TfLiteTensorNumDims(input_), 4) << "input tensor must be [1, width, height, 3]";
        CHECK_EQ(TfLiteTensorDim(input_, 0), 1) << "input tensor batch size must be 1";
        width_ = TfLiteTensorDim(input_, 1);
        height_ = TfLiteTensorDim(input_, 2);
        CHECK_EQ(TfLiteTensorDim(input_, 3), 3) << "input tensor must have 3 channels";

        output_ = CHECK_NOTNULL(TfLiteInterpreterGetOutputTensor(interpreter_, 0));
        LOG(INFO) << "Output tensor: " << tensor_shape(output_);
        CHECK_EQ(TfLiteTensorType(output_), kTfLiteFloat32) << "output tensor must be float32";
        CHECK_EQ(TfLiteTensorNumDims(output_), 4) << "output tensor must be [1, width, height, labels]";
        CHECK_EQ(TfLiteTensorDim(output_, 1), width_) << "output tensor width doesn't match input tensor width";
        CHECK_EQ(TfLiteTensorDim(output_, 2), height_) << "output tensor height doesn't match input tensor height";
        nlabels_ = TfLiteTensorDim(output_, 3);
        CHECK_EQ(nlabels_, sizeof(label_names)/sizeof(label_names[0]));

        LOG(INFO) << "Initialized tflite with " << width_ << "x" << height_ << "px input for model " << model_filename;
    }

    void maskBackground(cv::Mat& frame) {
        cv::Mat small, input_int;
        cv::resize(frame, small, cv::Size(width_, height_));
        cv::cvtColor(small, input_int, cv::COLOR_BGR2RGB);

        cv::Mat input_float;
        input_int.convertTo(input_float, CV_32FC3, 1./255, -.5);
#if 0
        LOG(INFO) << "input_float.size: " << input_float.size << ", type: " << input_float.type();
        LOG(INFO) << "input_float[0]: " << input_float;
        exit(0);
#endif

        CHECK_EQ(input_float.elemSize(), sizeof(float)*3);

        TfLiteTensorCopyFromBuffer(input_, (const void*)input_float.ptr<float>(),
                width_*height_*sizeof(float)*3);

        auto start = std::chrono::steady_clock::now();
        TfLiteInterpreterInvoke(interpreter_);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end-start;
        LOG(INFO) << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms";

        CHECK_EQ(TfLiteTensorByteSize(output_), width_*height_*nlabels_*sizeof(float));
        float *output = (float*)TfLiteTensorData(output_);

        int label_count[nlabels_] = {0};

        for (int y = 0; y < height_; y++) {
            float *row = &output[y * width_ * nlabels_];
            for (int x = 0; x < width_; x++) {
                float *col = &row[x * nlabels_];
                std::vector<size_t> labels(nlabels_);
                std::iota(labels.begin(), labels.end(), 0);
                std::stable_sort(labels.begin(), labels.end(),
                        [&](size_t a, size_t b) { return col[a] > col[b]; });

#if 0
                LOG(INFO) << "label[0]: " << labels[0] << " (" << label_names[labels[0]] << ")";
                std::stringstream s;
                s << "labels for (" << x << "," << y << "):";
                for (int l = 0; l < nlabels_; l++)
                    s << " " << label_names[labels[l]];
                LOG(INFO) << s.str();
#endif

                if (labels[0] != 15)
                    small.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255, 0, 0);
                label_count[labels[0]]++;
            }
        }
#if 0
        LOG(INFO) << "Label count for frame:";
        for (int i = 0; i < nlabels_; i++) {
            if (label_count[i])
                LOG(INFO) << label_names[i] << ": " << label_count[i];
        }
#endif
        cv::imshow("foo", small);

#ifdef DEBUG_LABELS
        for (int x = 0; x < width_; x++) {
            for (int y = 0; y < height_; y++) {
                std::stringstream s;
                s << "labels for (" << x << "," << y << "):";

                float *labels = &output[x*y*nlabels_];
                for (int l = 0; l < nlabels_; l++)
                    s << " " << labels[l];

                LOG(INFO) << s.str();
            }
        }
#endif
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

int main(int argc, char **argv) {
    FLAGS_v = 1;
    FLAGS_logtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, false);
    google::InitGoogleLogging(argv[0]);

    BackgroundRemover bgr(FLAGS_model_filename.c_str());

    cv::VideoCapture cap(0);
    cv::Mat frame;
    while (1) {
        cap >> frame;
        if (frame.empty())
            break;
        bgr.maskBackground(frame);
        if (cv::waitKey(5) == 'q')
            break;
    }

    return 0;
}
