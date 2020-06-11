# What’s that?

A Video4Linux “proxy” masking a webcam’s background, using [TensorFlow and
DeepLab](https://www.tensorflow.org/lite/models/segmentation/overview). This is just a fun side
project ~~distract~~aiding the current the Covid-19 WFH situation—it does not even have a proper
name. If you’re looking for a polished product, there’s [Snap
Camera](https://snapcamera.snapchat.com/), [XSplit VCam](https://www.xsplit.com/vcam) and others.

That said, I’m happy for pull requests. Please be patient, I suck at responding to anything.

# Installation (Debian)

```
# Add Bazel repository, if necessary.
if ! apt-cache show bazel >/dev/null; then
  curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
  echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
fi

# Install build dependencies.
sudo apt install bazel bazel-2.0.0 cmake make libgflags-dev libopencv-dev libgl-dev libegl-dev

# Clone and build repository.
git clone --recursive https://github.com/yath/bgremover
cd bgremover
make -j4

# Install v4l2loopback DKMS module.
sudo apt install v4l2loopback-dkms
```

# Usage

Load the v4l2loopback module: `sudo modprobe v4l2loopback max_buffers=5 exclusive_caps=1`.
`exclusive_caps` prevents the loopback device from being opened for recording for a second time,
which is necessary for Chrome. `max_buffers=5` is just a reasonable looking amount and can probably
be changed.

The `make` invocation from the previous section should have produced an executable (relative to the
project root directory) `build/bgr` and downloaded a `deeplabv3_257_mv_gpu.tflite`. The binary
expects the `.tflite` file to be present in the current working directory, so running `build/bgr`
from the project root direct should just work. If you’d like custom backgrounds, place them into
`$PWD/backgrounds/*.{jpg,png,…}`.

`bgr` defaults to capturing from `/dev/video0` and writing to `/dev/video2`. Use
`--input_device_number` and `--output_device_path` to change them. `--help` has a full list of
flags.

The masking behaviour can be changed at runtime with several keys (sent to the preview window):

* `<space>` disables (or re-enables) masking completely.

* `s` toggles smoothing of the mask.

* `b` toggles alpha-blending of the mask.

* `c` and `C` cycle through the colors set by `--color_list`. This could be used for emulating a
  greenscreen and feeding it to OBS, but I haven’t actually tried that.

* `i` and `I` cycle through the background images that were available at program startup.

* `f` freezes the current frame.

* `q` quits the program.

# Bugs

* DeepLab’s input pixels should be `[-1,1]`, but `bgr` uses `[-0.5,0.5]` because it gives better results.

* `libgflags-dev` shouldn’t be needed for building, but without it `gflags` and `glog` are using
  different namespaces and I haven’t figured out how to summon CMake.

* There’s preliminary support for BodyPix models, but I haven’t had the time to get it working.

* Much more. Please send a pull request.

# License

[Apache 2.0](LICENSE) due to the dependencies used.
