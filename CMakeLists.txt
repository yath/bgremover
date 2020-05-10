cmake_minimum_required(VERSION 3.5)

project(bgremover)

include(ExternalProject)
ExternalProject_Add(TFLite
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow/
    GIT_TAG v2.2.0
    GIT_SHALLOW true
    CONFIGURE_COMMAND ""
    BUILD_COMMAND bazel --output_user_root=${CMAKE_CURRENT_BINARY_DIR}/bazel-temp build -c opt 
                   //tensorflow/lite:tensorflowlite 
                   //tensorflow/lite/c:c_api
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE true
)

ExternalProject_Get_property(TFLite SOURCE_DIR)
set(TFLite_INCLUDES ${SOURCE_DIR})
set(TFLite_LIBS 
    ${SOURCE_DIR}/bazel-bin/tensorflow/lite/c/libc_api.so
    ${SOURCE_DIR}/bazel-bin/tensorflow/lite/libtensorflowlite.so
)

find_package(OpenCV REQUIRED)

add_subdirectory(glog)
add_subdirectory(gflags)

###### default target #######

add_executable(bgr main.cc)

add_dependencies(bgr TFLite)
target_include_directories(bgr PRIVATE ${TFLite_INCLUDES})

target_link_libraries(bgr
    ${TFLite_LIBS}
    ${OpenCV_LIBS}
    glog::glog
)
