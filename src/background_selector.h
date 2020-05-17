#ifndef BACKGROUND_SELECTOR_H
#define BACKGROUND_SELECTOR_H

#include <opencv2/highgui.hpp>
#include <string>
#include <utility>
#include <vector>

class BackgroundSelector {
    enum class Mode {
        Undefined,
        Image,
        Color,
    };

    struct Image {
        std::string filename;
        cv::Mat mat;
    };

    const std::string image_dir_;
    const int width_, height_;

    std::vector<Image> images_;
    std::vector<cv::Vec3b> colors_;

    unsigned int curr_image_, curr_color_;
    Mode curr_mode_;
    cv::Mat curr_background_;

    void loadImages();
    bool changeMode(Mode m);
    void changed();

    friend std::ostream& operator<<(std::ostream& os, const Image& i);

   public:
    BackgroundSelector(const std::string& image_dir, const std::string& color_list, int width,
                       int height);
    void selectPrevColor();
    void selectNextColor();
    void selectPrevImage();
    void selectNextImage();
    cv::Mat getBackground() const;
};

#endif  // BACKGROUND_SELECTOR_H
