#ifndef NUMCPP_OPENCV_H_
#define NUMCPP_OPENCV_H_

#include "../config.h"

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

#ifdef USE_CUDA
#include "cuda.h"
#include <opencv2/gpu/gpu.hpp>
#endif

namespace np {

template <typename T>
cv::Mat to_cv_mat(array_t<T> &array)
{
	return cv::Mat(
		array.size(0), array.size(1), 
		cv::DataType<T>::type, array.raw_ptr());
}

template <typename T>
const cv::Mat to_cv_mat(const array_t<T> &array)
{
	return cv::Mat(
		array.size(0), array.size(1), 
		cv::DataType<T>::type, const_cast<uint8_t *>(array.raw_ptr()));
}

inline base_array_t from_cv_mat(const cv::Mat &cv_mat)
{
	base_array_t result(cv_mat.elemSize());
	result.setSize<heap_allocator>(cv_mat.rows, cv_mat.cols);

	// TODO: Do not copy
	memcpy(result.raw_ptr<void>(), cv_mat.data, result.byteSize());

	return std::move(result);
}

template <typename T>
array_t<T> from_cv_mat(const cv::Mat &cv_mat)
{
	return array_t<T>(from_cv_mat(cv_mat));
}

template <typename T>
void from_cv_mat(array_t<T> &dst, const cv::Mat &src)
{
	// TODO: check type of src
	dst.setSize(src.rows, src.cols);

	// TODO: Do not copy
	memcpy(dst, src.data, dst.byteSize());
}

#ifdef USE_CUDA

template <typename T>
inline cv::gpu::GpuMat to_cv_gpu_mat(device_array_t<T> &array_d)
{
	return cv::gpu::GpuMat(array_d.size(0), array_d.size(1), cv::DataType<T>::type, array_d.raw_ptr());
}

template <typename T>
inline const cv::gpu::GpuMat to_cv_gpu_mat(const device_array_t<T> &array_d)
{
	return cv::gpu::GpuMat(array_d.size(0), array_d.size(1), cv::DataType<T>::type, const_cast<T *>(array_d.raw_ptr()));
}

#endif // USE_CUDA

inline array_t<uint8_t> imread(const std::string &filename)
{
	cv::Mat cv_image = cv::imread(filename);

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	return from_cv_mat<uint8_t>(cv_grayscale);
}

inline void imread(array_t<uint8_t> &dst, const std::string &filename)
{
	cv::Mat cv_image = cv::imread(filename);

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	from_cv_mat(dst, cv_grayscale);
}

inline bool imwrite(const array_t<uint8_t> &image, const std::string &filename)
{
	return cv::imwrite(filename, to_cv_mat(image));
}

inline void imshow(const array_t<uint8_t> &image)
{
	cv::imshow("image", to_cv_mat(image));
	cv::waitKey(0);
}

} // namespace np

#endif // NUMCPP_OPENCV_H_