#include <stdio.h>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "tensorflow/lite/c/c_api.h"

DEFINE_string(model_filename, "foo.tflite", "model filename");

int main(int argc, char **argv) {
    FLAGS_v = 1;
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    TfLiteModel* model = TfLiteModelCreateFromFile(FLAGS_model_filename.c_str());
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 2);
    TfLiteInterpreterOptionsSetErrorReporter(options, (void (*)(void*, const char*, va_list))vfprintf, (void*)stderr);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);

    TfLiteInterpreterInvoke(interpreter);
    LOG(INFO) << "huhu";
    return 0;
}
