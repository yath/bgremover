.PHONY: all builddir tidy clean

all: builddir
	$(MAKE) -C build

builddir:
	mkdir -p build
	(cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..)

tidy: builddir
	clang-tidy -p build $(wildcard *.cc)

clean:
	rm -rf build
