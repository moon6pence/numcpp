#ifndef __NUMCPP_IMAGE_H__
#define __NUMCPP_IMAGE_H__

#include "array.h"

#include <string.h> // memcpy

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

namespace numcpp {

inline array_t<uint8_t, 2> imread(const std::string &file_path)
{
	using namespace cv;

	Mat cv_image = cv::imread(file_path);
	if (cv_image.empty()) return empty<uint8_t, 2>();

	// Convert to grayscale
	Mat cv_image_grayscale;
	cvtColor(cv_image, cv_image_grayscale, CV_BGR2GRAY);

	auto result = array<uint8_t>(cv_image_grayscale.cols, cv_image_grayscale.rows);
	// TODO: Check cv_image.step
	memcpy(result, cv_image_grayscale.data, cv_image.total() * sizeof(uint8_t));

	return std::move(result);
}

inline void imwrite(const array_t<uint8_t, 2> &image, const std::string &file_path)
{
	using namespace cv;

	Mat cv_image(image.height(), image.width(), CV_8U, const_cast<uint8_t *>(image.raw_pointer()));
	cv::imwrite(file_path, cv_image);
}

inline void imshow(const array_t<uint8_t, 2> &image)
{
	using namespace cv;

	Mat cv_image(image.height(), image.width(), CV_8U, const_cast<uint8_t *>(image.raw_pointer()));

	cv::imshow("numcpp::imshow", cv_image);
	waitKey(0);
}

} // namespace numcpp

#endif // __NUMCPP_IMAGE_H__