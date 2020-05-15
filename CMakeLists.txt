cmake_minimum_required(VERSION 3.5)

project(bgremover)

## OpenGL

set(OpenGL_GL_PREFERENCE LEGACY)
include(FindOpenGL)

if(OPENGL_FOUND AND OpenGL_EGL_FOUND)
    set(WITH_GL ON CACHE BOOL "Whether to use OpenGL")
endif()

## Tensorflow

include(ExternalProject)

set(TF_BAZEL_TARGETS
    //tensorflow/lite:tensorflowlite
    //tensorflow/lite/c:c_api
)
if (WITH_GL)
    list(APPEND TF_BAZEL_TARGETS //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so)
endif()

ExternalProject_Add(TFLite
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow/
    GIT_TAG v2.2.0
    GIT_SHALLOW true
    CONFIGURE_COMMAND ""
    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/tf-v2.2.0-nostatus.patch &&
                  patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/tf-v2.2.0-noabsl.patch
    BUILD_COMMAND bazel --output_user_root=${CMAKE_CURRENT_BINARY_DIR}/bazel-temp build -c opt ${TF_BAZEL_TARGETS}
    BUILD_IN_SOURCE true
    INSTALL_COMMAND ""
)

ExternalProject_Get_property(TFLite SOURCE_DIR)
set(TFLite_INCLUDES ${SOURCE_DIR})
set(TFLite_LIBS 
    ${SOURCE_DIR}/bazel-bin/tensorflow/lite/c/libc_api.so
    ${SOURCE_DIR}/bazel-bin/tensorflow/lite/libtensorflowlite.so
)
if (WITH_GL)
    list(APPEND TFLite_LIBS ${SOURCE_DIR}/bazel-bin/tensorflow/lite/delegates/gpu/libtensorflowlite_gpu_delegate.so)
endif()

## OpenCV, glog, gflags, ...

find_package(OpenCV REQUIRED)

# the order matters because of the flags namespace. determined by trial and error.
add_subdirectory(glog EXCLUDE_FROM_ALL)
add_subdirectory(gflags EXCLUDE_FROM_ALL)


## Default target
add_executable(bgr
    src/main.cc

    src/background_remover.cc
    src/background_remover.h

    src/video_writer.cc
    src/video_writer.h
)

add_dependencies(bgr TFLite)
target_include_directories(bgr PRIVATE ${TFLite_INCLUDES})

target_link_libraries(bgr
    ${TFLite_LIBS}
    ${OpenCV_LIBS}
    glog::glog
)

if(WITH_GL)
    target_link_libraries(bgr
        OpenGL::GL
        OpenGL::EGL
    )
    target_sources(bgr PRIVATE egl_stubs.cc)
    add_definitions(-DWITH_GL=1)
else()
    add_definitions(-UWITH_GL)
endif()
