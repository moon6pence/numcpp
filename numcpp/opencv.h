#ifndef NUMCPP_OPENCV_H_
#define NUMCPP_OPENCV_H_

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

namespace numcpp {

template <typename T>
cv::Mat to_cv_mat(array_t<T> &array)
{
	return cv::Mat(
		array.size(0), array.size(1), 
		cv::DataType<T>::type, array.raw_ptr());
}

template <typename T>
array_t<T> from_cv_mat(const cv::Mat &cv_mat)
{
	// TODO: Do not copy
	array_t<T> result(cv_mat.rows, cv_mat.cols);
	memcpy(result.raw_ptr(), cv_mat.data, result.size() * sizeof(T));
	return result;
}

array_t<uint8_t> imread(const std::string &file_path)
{
	cv::Mat cv_image = cv::imread(file_path);

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	return from_cv_mat<uint8_t>(cv_grayscale);
}

} // namespace numcpp

#endif // NUMCPP_OPENCV_H_