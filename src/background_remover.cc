#include "background_remover.h"

#include "glog/logging.h"

#ifdef WITH_GL
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

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

static std::string tensor_shape(const TfLiteTensor *t) {
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

BackgroundRemover::BackgroundRemover(const char *model_filename, int num_threads) {
    static_assert(sizeof(float) == 4, "floats must be 32 bits");

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

void BackgroundRemover::maskBackground(cv::Mat& frame /* rgb */, const cv::Mat& maskImage /* rgb */) {
    CHECK_EQ(frame.size, maskImage.size);
    cv::Mat small;
    cv::resize(frame, small, cv::Size(width_, height_), interpolation_method);

    cv::Mat input_float;
    small.convertTo(input_float, CV_32FC3, 1./255, -.5);

    CHECK_EQ(input_float.elemSize(), sizeof(float)*3 /* channels */);

    TfLiteTensorCopyFromBuffer(input_, (const void*)input_float.ptr<float>(),
            width_*height_*sizeof(float)*3);

    auto start = std::chrono::steady_clock::now();
    TfLiteInterpreterInvoke(interpreter_);
    auto end = std::chrono::steady_clock::now();
    auto diffMs = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    LOG(INFO) << "Inference time: " << diffMs << "ms";

    CHECK_EQ(TfLiteTensorByteSize(output_), width_*height_*nlabels_*sizeof(float));
    float *output = (float*)TfLiteTensorData(output_);

    cv::Mat mask = cv::Mat::zeros(cv::Size(width_, height_), CV_8U);

    for (int y = 0; y < height_; y++) {
        float *row = &output[y * width_ * nlabels_];
        for (int x = 0; x < width_; x++) {
            float *col = &row[x * nlabels_];
            // XXX: Only need the max element index.
            std::vector<size_t> labels(nlabels_);
            std::iota(labels.begin(), labels.end(), 0);
            std::stable_sort(labels.begin(), labels.end(),
                    [&](size_t a, size_t b) { return col[a] > col[b]; });

            if (labels[0] != 15) {
                mask.at<unsigned char>(cv::Point(x, y)) = 1;
            }
        }
    }

    cv::resize(mask, mask, cv::Size(frame.cols, frame.rows), interpolation_method);
    // XXX: Fix this.
    for (int x = 0; x < frame.cols; x++)
        for (int y = 0; y < frame.rows; y++)
            if (mask.at<unsigned char>(cv::Point(x, y)))
                frame.at<cv::Vec3b>(cv::Point(x, y)) = maskImage.at<cv::Vec3b>(cv::Point(x, y));
    //frame.setTo(maskImage, mask);
    //cv::bitwise_and(frame, mask, frame);
}

BackgroundRemover::~BackgroundRemover() {
    TfLiteInterpreterDelete(interpreter_);
#ifdef WITH_GL
    TfLiteGpuDelegateV2Delete(gpu_delegate_);
#endif
    TfLiteInterpreterOptionsDelete(options_);
    TfLiteModelDelete(model_);
}
