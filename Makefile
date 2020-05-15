.PHONY: all builddir tidy clean

all: builddir deeplabv3_257_mv_gpu.tflite
	$(MAKE) -C build

deeplabv3_257_mv_gpu.tflite:
	curl https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite > $@

builddir:
	mkdir -p build
	(cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..)

tidy: builddir
	clang-tidy -p build $(wildcard src/*.cc)

clean:
	rm -rf build
