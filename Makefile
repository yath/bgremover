SRCS := $(wildcard src/*.cc src/*.h)

.PHONY: all
all: builddir deeplabv3_257_mv_gpu.tflite
	$(MAKE) -C build

deeplabv3_257_mv_gpu.tflite:
	curl https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite > $@

.PHONY: builddir
builddir:
	mkdir -p build
	(cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..)

.PHONY: tidy
tidy: builddir
	clang-tidy -p build $(SRCS)

.PHONY: format
format:
	clang-format -style='{BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 100}' -i $(SRCS)

.PHONY: clean
clean:
	rm -rf build
