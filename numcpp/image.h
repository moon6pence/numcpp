#ifndef __NUMCPP_IMAGE_H__
#define __NUMCPP_IMAGE_H__

#include "array.h"

#include <string.h> // memcpy

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

namespace numcpp {

template <typename T>
inline cv::Mat to_cv_mat(array_t<T, 2> &array)
{
	// static_assert(false, "This type does not support in OpenCV");
	return cv::Mat();
}

template <typename T>
inline const cv::Mat to_cv_mat(const array_t<T, 2> &array)
{
	// static_assert(false, "This type does not support in OpenCV");
	return cv::Mat();
}

template <>
inline cv::Mat to_cv_mat(array_t<uint8_t, 2> &array)
{
	return cv::Mat(array.height(), array.width(), CV_8U, array.raw_pointer());
}

template <>
inline const cv::Mat to_cv_mat(const array_t<uint8_t, 2> &array)
{
	return cv::Mat(array.height(), array.width(), CV_8U, const_cast<uint8_t *>(array.raw_pointer()));
}

template <>
inline cv::Mat to_cv_mat(array_t<float, 2> &array)
{
	return cv::Mat(array.height(), array.width(), CV_32F, array.raw_pointer());
}

template <>
inline const cv::Mat to_cv_mat(const array_t<float, 2> &array)
{
	return cv::Mat(array.height(), array.width(), CV_32F, const_cast<float *>(array.raw_pointer()));
}

inline void cv_mat_deleter(cv::Mat *cv_mat)
{
	delete cv_mat;
}

/** Allocate array from cv::Mat */
inline array_t<uint8_t, 2> from_cv_mat(cv::Mat &cv_mat)
{
	// allocate cv::Mat (add reference)
	cv::Mat *ref = new cv::Mat(cv_mat);

	// address
	std::shared_ptr<void> address(ref, cv_mat_deleter);

	// origin
	uint8_t *origin = reinterpret_cast<uint8_t *>(ref->data);

	// shape
	int *new_shape = new int[2];
	new_shape[0] = cv_mat.cols;
	new_shape[1] = cv_mat.rows;

	return array_t<uint8_t, 2>(address, origin, new_shape);
}

inline array_t<uint8_t, 2> imread(const std::string &file_path)
{
	using namespace cv;

	Mat cv_image = cv::imread(file_path);
	if (cv_image.empty()) return empty<uint8_t, 2>();

	// Convert to grayscale
	Mat cv_image_grayscale;
	cvtColor(cv_image, cv_image_grayscale, CV_BGR2GRAY);

	return from_cv_mat(cv_image_grayscale);
}

inline void imwrite(const array_t<uint8_t, 2> &image, const std::string &file_path)
{
	cv::imwrite(file_path, to_cv_mat(image));
}

inline void imshow(const array_t<uint8_t, 2> &image)
{
	cv::imshow("numcpp::imshow", to_cv_mat(image));
	cv::waitKey(0);
}

} // namespace numcpp

#endif // __NUMCPP_IMAGE_H__