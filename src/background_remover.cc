#include "background_remover.h"

#include "debug.h"
#include "glog/logging.h"
#include "opencv2/highgui.hpp"

#ifdef WITH_GL
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

const char *deeplabv3_label_names[] = {
    "background", "aeroplane", "bicycle",     "bird",  "board",       "bottle", "bus",
    "car",        "cat",       "chair",       "cow",   "diningtable", "dog",    "horse",
    "motorbike",  "person",    "pottedplant", "sheep", "sofa",        "train",  "tv",
};

constexpr int deeplabv3_label_count =
    sizeof(deeplabv3_label_names) / sizeof(deeplabv3_label_names[0]);
typedef float DeeplabV3Labels[deeplabv3_label_count];

static std::string tensor_shape(const TfLiteTensor *t) {
    std::stringstream ret;
    ret << "[";
    for (int i = 0; i < TfLiteTensorNumDims(t); i++) {
        if (i > 0) ret << ", ";
        ret << TfLiteTensorDim(t, i);
    }
    ret << "]";
    return ret.str();
}

BackgroundRemover::ModelType BackgroundRemover::parseModelType(const std::string &model_type) {
    if (model_type == "deeplabv3")
        return BackgroundRemover::ModelType::DeeplabV3;
    else if (model_type == "bodypix_resnet")
        return BackgroundRemover::ModelType::BodypixResnet;
    else if (model_type == "bodypix_mobilenet")
        return BackgroundRemover::ModelType::BodypixMobilenet;
    else
        return BackgroundRemover::ModelType::Undefined;
}

BackgroundRemover::BackgroundRemover(const std::string &model_filename,
                                     const std::string &model_type, int num_threads)
    : model_type_(parseModelType(model_type)) {
    static_assert(sizeof(float) == 4, "floats must be 32 bits");

    CHECK(model_type_ != ModelType::Undefined) << "Invalid model type " << model_type;

    model_ = CHECK_NOTNULL(TfLiteModelCreateFromFile(model_filename.c_str()));

    options_ = CHECK_NOTNULL(TfLiteInterpreterOptionsCreate());

    TfLiteInterpreterOptionsSetNumThreads(options_, num_threads);
    TfLiteInterpreterOptionsSetErrorReporter(
        options_,
        [](void * /* unused */, const char *fmt, va_list args) {
            std::vector<char> buf(vsnprintf(nullptr, 0, fmt, args) + 1);
            std::vsnprintf(buf.data(), buf.size(), fmt, args);
            LOG(ERROR) << "Tensorflow: " << buf.data();
        },
        nullptr);
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
    int outw = TfLiteTensorDim(output_, 1);
    CHECK_EQ(width_ % outw, 0) << "output tensor width is not a multiple of input tensor width";
    stride_ = width_ / outw;
    int outh = TfLiteTensorDim(output_, 2);
    CHECK_EQ(height_ % outh, 0) << "output tensor height is not a multiple of input tensor height";
    CHECK_EQ(height_ / outh, stride_) << "vertical stride doesn't match horizontal stride";

    if (model_type_ == ModelType::DeeplabV3) {
        CHECK_EQ(stride_, 1);
        CHECK_EQ(TfLiteTensorDim(output_, 3), deeplabv3_label_count);
    } else if (model_type_ == ModelType::BodypixResnet) {
        CHECK(stride_ == 16 || stride_ == 32);
        CHECK_EQ(TfLiteTensorDim(output_, 3), 1);
    } else if (model_type_ == ModelType::BodypixMobilenet) {
        CHECK(stride_ == 8 || stride_ == 16);
        CHECK_EQ(TfLiteTensorDim(output_, 3), 1);
    }

    LOG(INFO) << "Initialized tflite with " << width_ << "x" << height_
              << "px input and stride=" << stride_ << " for model " << model_filename;
}

static float minVec3f(const cv::Vec3f &v) { return std::min({v[0], v[1], v[2]}); }

static float maxVec3f(const cv::Vec3f &v) { return std::max({v[0], v[1], v[2]}); }

static void checkValuesInRange(const cv::Mat &mat, float min, float max) {
#ifndef NDEBUG
    const auto [matmin, matmax] = std::minmax_element(
        std::execution::seq /* XXX */, mat.begin<cv::Vec3f>(), mat.end<cv::Vec3f>(),
        [](const cv::Vec3f &a, const cv::Vec3f &b) { return minVec3f(a) < minVec3f(b); });
    CHECK_GE(minVec3f(*matmin), min);
    CHECK_LE(maxVec3f(*matmax), max);
#endif
}

cv::Mat BackgroundRemover::makeInputTensor(const cv::Mat &img) {
    cv::Mat ret;
    switch (model_type_) {
        case ModelType::DeeplabV3:
        case ModelType::BodypixMobilenet:
            img.convertTo(ret, CV_32FC3, 1. / 255, -.5);
            checkValuesInRange(ret, -.5, .5);
            break;

        case ModelType::BodypixResnet:
            img.convertTo(ret, CV_32FC3);
            // https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/resnet.ts#L22
            std::for_each(std::execution::seq /* XXX */, ret.begin<cv::Vec3f>(), ret.end<cv::Vec3f>(),
                          [](cv::Vec3f &v) { v += cv::Vec3f(-123.15, -115.90, -103.06); });
            checkValuesInRange(ret, -127., 255.);  // ?
            break;

        default:
            CHECK(0);
    }

    return ret;
}

// constexpr float logit(float x) { return std::log(x / (1. - x)); }
static inline float expit(float x) { return 1.f / (1.f + std::exp(-x)); }

cv::Mat BackgroundRemover::getMaskFromOutput() {
    constexpr int person_label = 15;  // XXX
    constexpr float threshold = .7;   // XXX

    int maskw = width_ / stride_;
    int maskh = height_ / stride_;

    cv::Mat ret = cv::Mat::zeros(cv::Size(maskw, maskh), CV_32FC3);

    size_t size = TfLiteTensorByteSize(output_);
    void *data = TfLiteTensorData(output_);

    if (model_type_ == ModelType::DeeplabV3) {
        CHECK_EQ(size, maskw * maskh * sizeof(DeeplabV3Labels));
        DeeplabV3Labels *labels = (DeeplabV3Labels *)data;
        std::for_each(std::execution::seq /* XXX */, labels, labels + maskw * maskh,
                      [&](DeeplabV3Labels l) {
                          float *max = std::max_element(l, l + deeplabv3_label_count);
                          int label = static_cast<int>(max - l);
                          if (label != person_label) {
                              int pixel = static_cast<int>((DeeplabV3Labels *)l - labels);
                              ret.at<cv::Vec3f>(cv::Point(pixel % maskw, pixel / maskw)) = {1.0, 1.0, 1.0};
                          }
                      });
    } else {
        CHECK_EQ(size, maskw * maskh * sizeof(float));
        float *prob = (float *)data;
        std::for_each(std::execution::seq /* XXX */, prob, prob + maskw * maskh, [&](float &p) {
            if (expit(p) < threshold) {  // p > -logit(threshold) && p < logit(threshold)?
                int pixel = static_cast<int>(&p - prob);
                ret.at<cv::Vec3f>(cv::Point(pixel % maskw, pixel / maskw)) = {1.0, 1.0, 1.0};
            }
        });
    }

    return ret;
}

struct Padding {
    int l, r, t, b;
};

static void padMat(cv::Mat &m, const Padding &pad) {
    const auto type = m.type();
    if (pad.l || pad.r) {
        const cv::Mat padl = cv::Mat::zeros(m.rows, pad.l, type);
        const cv::Mat padr = cv::Mat::zeros(m.rows, pad.r, type);
        cv::hconcat(std::array<cv::Mat, 3>({padl, m, padr}), m);
    }

    if (pad.t || pad.b) {
        const cv::Mat padt = cv::Mat::zeros(pad.t, m.cols, type);
        const cv::Mat padb = cv::Mat::zeros(pad.b, m.cols, type);
        cv::vconcat(std::array<cv::Mat, 3>({padt, m, padb}), m);
    }
}

static void unpadMat(cv::Mat &m, const Padding &pad) {
    cv::Rect rect = cv::Rect(0, 0, m.cols, m.rows);
    rect.x += pad.l;
    rect.width -= (pad.l + pad.r);
    rect.y += pad.t;
    rect.height -= (pad.t + pad.b);
    m = m(rect);
}

// stolen from tfjs-models/body-pix/src/util.ts
static cv::Mat resizeAndPadTo(const cv::Mat &frame, int targetw, int targeth, Padding &pad) {
    const int imgw = frame.cols;
    const int imgh = frame.rows;

    const float image_aspect = (float)imgw / (float)imgh;
    const float target_aspect = (float)targetw / (float)targeth;

    int resizew, resizeh;
    if (image_aspect > target_aspect) {
        resizew = targetw;
        resizeh = std::ceil((float)resizew / image_aspect);

        const int padh = targeth - resizeh;
        pad.l = pad.r = 0;
        pad.t = std::floor((float)padh / 2.);
        pad.b = targeth - (resizeh + pad.t);
    } else {
        resizeh = targeth;
        resizew = std::ceil((float)resizeh * image_aspect);

        const int padw = targetw - resizew;
        pad.l = std::floor((float)padw / 2.);
        pad.r = targetw - (resizew + pad.l);
        pad.t = pad.b = 0;
    }

    cv::Mat ret;
    cv::resize(frame, ret, cv::Size(resizew, resizeh), cv::INTER_LINEAR);
    padMat(ret, pad);
    return ret;
}

void BackgroundRemover::maskBackground(cv::Mat &frame /* rgb */,
                                       const cv::Mat &maskImage /* rgb */,
                                       bool do_blur_mask,
                                       Timing &t) {
    auto start = Timing::now();

    CHECK_EQ(frame.size, maskImage.size);
    Padding pad;
    cv::Mat small = resizeAndPadTo(frame, width_, height_, pad);

    if (debug_flags & DebugFlagShowModelInputFrame) {
        cv::Mat bgr;
        cv::cvtColor(small, bgr, cv::COLOR_RGB2BGR);
        cv::imshow("model_input", bgr);
    }

    cv::Mat input_float = makeInputTensor(small);
    CHECK_EQ(input_float.elemSize(), sizeof(float) * 3 /* channels */);

    TfLiteTensorCopyFromBuffer(input_, (const void *)input_float.ptr<float>(),
                               width_ * height_ * sizeof(float) * 3);

    auto startInference = Timing::now();
    t.prepare_input += (startInference - start);

    TfLiteInterpreterInvoke(interpreter_);

    auto startMask = Timing::now();
    t.inference += (startMask - startInference);

    cv::Mat mask = getMaskFromOutput();
    unpadMat(mask, pad);
    cv::resize(mask, mask, cv::Size(frame.cols, frame.rows), interpolation_method);

    if (do_blur_mask) {
      // Use a reasonable size depending on the resolution.
      int blur_width = frame.cols / 64;
      cv::blur(mask, mask, cv::Size(blur_width, blur_width));
    }

    // Create background and foreground layers weightened by the mask.
    cv::Mat foreground;
    frame.convertTo(foreground, CV_32FC3, 1.0/255.0);
    cv::multiply(cv::Scalar::all(1.0) - mask, foreground, foreground);
    cv::Mat background;
    cv::multiply(mask, maskImage, background);
    // Assemble both layers again and convert back.
    cv::add(foreground, background, frame);
    frame.convertTo(frame, CV_8UC3, 255);

    auto end = Timing::now();
    t.mask += (end - startMask);
    t.total += (end - start);
}

BackgroundRemover::~BackgroundRemover() {
    TfLiteInterpreterDelete(interpreter_);
#ifdef WITH_GL
    TfLiteGpuDelegateV2Delete(gpu_delegate_);
#endif
    TfLiteInterpreterOptionsDelete(options_);
    TfLiteModelDelete(model_);
}
