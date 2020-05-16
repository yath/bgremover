#ifndef BACKGROUND_REMOVER_H
#define BACKGROUND_REMOVER_H

#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/c/c_api.h"
#include "timing.h"

class BackgroundRemover {
    enum class ModelType {
        Undefined,
        DeeplabV3,
        BodypixResnet,
        BodypixMobilenet,
    };

    constexpr static int interpolation_method = cv::INTER_LINEAR;

    const ModelType model_type_;
    TfLiteModel *model_;
    TfLiteInterpreterOptions *options_;
    TfLiteInterpreter *interpreter_;

    TfLiteTensor *input_;
    const TfLiteTensor *output_;
    int width_, height_, stride_;

#ifdef WITH_GL
    TfLiteDelegate *gpu_delegate_;
#endif

    static ModelType parseModelType(const std::string &model_type);
    cv::Mat makeInputTensor(const cv::Mat &img);
    cv::Mat getMaskFromOutput();

   public:
    BackgroundRemover(const std::string &model_filename, const std::string &model_type,
                      int num_threads = 4);
    ~BackgroundRemover();

    void maskBackground(cv::Mat &frame /* rgb */, const cv::Mat &maskImage /* rgb */, bool do_blur_mask, Timing &t);
};
#endif  // BACKGROUND_REMOVER_H
