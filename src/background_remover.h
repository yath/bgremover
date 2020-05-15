#ifndef BACKGROUND_REMOVER_H
#define BACKGROUND_REMOVER_H

#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/c/c_api.h"

class BackgroundRemover {
   private:
    constexpr static int interpolation_method = cv::INTER_LINEAR;

    TfLiteModel *model_;
    TfLiteInterpreterOptions *options_;
    TfLiteInterpreter *interpreter_;

    TfLiteTensor *input_;
    const TfLiteTensor *output_;
    int width_, height_, nlabels_;

#ifdef WITH_GL
    TfLiteDelegate *gpu_delegate_;
#endif

   public:
    BackgroundRemover(const char *model_filename, int num_threads = 4);
    ~BackgroundRemover();

    void maskBackground(cv::Mat &frame /* rgb */, const cv::Mat &maskImage /* rgb */);
};
#endif  // BACKGROUND_REMOVER_H
