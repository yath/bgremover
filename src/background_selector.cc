#include "background_selector.h"

#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "glog/logging.h"

constexpr uint8_t nthByte(unsigned long byte, int n) { return ((byte >> (n * 8)) & 0xff); }

static std::vector<cv::Vec3b> parseColorList(const std::string& color_list) {
    std::vector<cv::Vec3b> ret;

    std::stringstream colors(color_list);
    for (std::string color; std::getline(colors, color, ',');) {
        static_assert(sizeof(unsigned long) >= 3, "unsigned long can't hold 24 bits");
        auto rgb = std::stoul(color, nullptr, 16);
        ret.push_back(cv::Vec3b(nthByte(rgb, 2), nthByte(rgb, 1), nthByte(rgb, 0)));
    }

    return ret;
}

std::ostream& operator<<(std::ostream& os, const BackgroundSelector::Image& i) {
    return os << "Image(\"" << i.filename << "\", " << i.mat.cols << "x" << i.mat.rows << "px)";
}

void BackgroundSelector::loadImages() {
    for (auto& p : std::filesystem::directory_iterator(image_dir_)) {
        const auto& path = p.path();

        if (!p.is_regular_file()) {
            LOG(WARNING) << path << " is not a regular file, skipping";
            continue;
        }

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            LOG(WARNING) << "Can't read " << path << " as image, skipping";
            continue;
        }
        cv::resize(img, img, cv::Size(width_, height_));
        // Convert to floating point numbers to simplify manipulation later.
        img.convertTo(img, CV_32FC3, 1.0/255.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        auto i = Image{path.filename(), img};
        LOG(INFO) << "Loaded " << i;
        images_.push_back(i);
    }

    std::sort(images_.begin(), images_.end(),
              [](const BackgroundSelector::Image& a, const BackgroundSelector::Image& b) {
                  return a.filename < b.filename;
              });
}

BackgroundSelector::BackgroundSelector(const std::string& image_dir, const std::string& color_list,
                                       int width, int height)
    : curr_image_(0),
      curr_color_(0),
      curr_mode_(Mode::Undefined),
      colors_(parseColorList(color_list)),
      image_dir_(image_dir),
      width_(width),
      height_(height) {
    loadImages();

    CHECK(changeMode(Mode::Image) || changeMode(Mode::Color)) << "No background images or colors";
    changed();
}

bool BackgroundSelector::changeMode(Mode m) {
    if (curr_mode_ == m) return false;

    if (m == Mode::Image && !images_.size()) {
        LOG(ERROR) << "No images loaded";
        return false;
    } else if (m == Mode::Color && !colors_.size()) {
        LOG(ERROR) << "No colors loaded";
        return false;
    }

    curr_mode_ = m;

    CHECK(curr_mode_ == Mode::Image || curr_mode_ == Mode::Color);
    LOG(INFO) << "Mode changed to " << (curr_mode_ == Mode::Image ? "Image" : "Color");
    return true;
}

static cv::Mat makeSolidBackground(const cv::Vec3b& color, int width, int height) {
    return cv::Mat(cv::Size(width, height), CV_32FC3, color);
}

void BackgroundSelector::changed() {
    if (curr_mode_ == Mode::Image) {
        LOG(INFO) << "Current background image: " << images_[curr_image_].filename;
        curr_background_ = images_[curr_image_].mat;
    } else if (curr_mode_ == Mode::Color) {
        LOG(INFO) << "Current color: " << colors_[curr_color_];
        curr_background_ = makeSolidBackground(colors_[curr_color_], width_, height_);
    } else {
        CHECK(0) << "Unknown mode " << static_cast<int>(curr_mode_);
    }
}

void BackgroundSelector::selectPrevColor() {
    if (!changeMode(Mode::Color)) {
        if (curr_color_ == 0)
            curr_color_ = colors_.size() - 1;
        else
            curr_color_--;
    }

    changed();
}

void BackgroundSelector::selectNextColor() {
    if (!changeMode(Mode::Color)) {
        if (curr_color_ == colors_.size() - 1)
            curr_color_ = 0;
        else
            curr_color_++;
    }

    changed();
}

void BackgroundSelector::selectPrevImage() {
    if (!changeMode(Mode::Image)) {
        if (curr_image_ == 0)
            curr_image_ = images_.size() - 1;
        else
            curr_image_--;
    }

    changed();
}

void BackgroundSelector::selectNextImage() {
    if (!changeMode(Mode::Image)) {
        if (curr_image_ == images_.size() - 1)
            curr_image_ = 0;
        else
            curr_image_++;
    }

    changed();
}

cv::Mat BackgroundSelector::getBackground() const { return curr_background_; }
