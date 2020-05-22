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
	clang-tidy -p build $(SRCS) -checks=bugprone-*,clang-analyzer-*,misc-*,performance-*,portability-*

.PHONY: format
format: .clang-format
	clang-format -style=file -i $(SRCS)

.clang-format:
	clang-format -style='{BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 100}' --dump-config > $@

.PHONY: clean
clean:
	rm -rf build
