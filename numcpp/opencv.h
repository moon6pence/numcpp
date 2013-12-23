#ifndef NUMCPP_OPENCV_H_
#define NUMCPP_OPENCV_H_

#include <opencv2/core/core.hpp>
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
	memcpy(cv_mat.data, result.raw_ptr(), result.size() * sizeof(T));
	return result;
}

} // namespace numcpp

#endif // NUMCPP_OPENCV_H_