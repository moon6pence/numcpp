#ifndef NUMCPP_OPENCV_H_
#define NUMCPP_OPENCV_H_

#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
#include "cuda.h"
#include <opencv2/gpu/gpu.hpp>
#endif

namespace np {

template <typename T>
cv::Mat to_cv_mat(Array<T> &array)
{
	return cv::Mat(
		array.size(1), array.size(0), 
		cv::DataType<T>::type, 
		array.raw_ptr(), 
		array.stride(1));
}

template <typename T>
const cv::Mat to_cv_mat(const Array<T> &array)
{
	return cv::Mat(
		array.size(1), array.size(0), 
		cv::DataType<T>::type, 
		const_cast<T *>(array.raw_ptr()), 
		array.stride(1));
}

template <>
inline cv::Mat to_cv_mat(Array<cv::Vec3b> &array)
{
	return cv::Mat(
		array.size(1), array.size(0), 
		CV_8UC3, 
		array.raw_ptr(), 
		array.stride(1));
}

template <>
inline const cv::Mat to_cv_mat(const Array<cv::Vec3b> &array)
{
	return cv::Mat(
		array.size(1), array.size(0), 
		CV_8UC3, 
		const_cast<cv::Vec3b *>(array.raw_ptr()), 
		array.stride(1));
}

inline BaseArray from_cv_mat(const cv::Mat &cv_mat)
{
	BaseArray result((int)cv_mat.elemSize(), make_vector(cv_mat.cols, cv_mat.rows));

	// TODO: Do not copy
	memcpy(result.raw_ptr<void>(), cv_mat.data, result.byteSize());

	return std::move(result);
}

template <typename T>
Array<T> from_cv_mat(const cv::Mat &cv_mat)
{
	return Array<T>(from_cv_mat(cv_mat));
}

template <typename T>
void from_cv_mat(Array<T> &dst, const cv::Mat &src)
{
	// TODO: check type of src
	if (dst.size() != make_vector(src.cols, src.rows))
		dst = Array<T>(src.cols, src.rows);

	// TODO: Do not copy
	memcpy(dst, src.data, dst.byteSize());
}

#ifdef USE_CUDA

template <typename T>
inline cv::gpu::GpuMat to_cv_gpu_mat(GpuArray<T> &array_d)
{
	return cv::gpu::GpuMat(array_d.size(0), array_d.size(1), cv::DataType<T>::type, array_d.raw_ptr());
}

template <typename T>
inline const cv::gpu::GpuMat to_cv_gpu_mat(const GpuArray<T> &array_d)
{
	return cv::gpu::GpuMat(array_d.size(0), array_d.size(1), cv::DataType<T>::type, const_cast<T *>(array_d.raw_ptr()));
}

#endif // USE_CUDA

inline Array<uint8_t> imread(const std::string &filename)
{
	cv::Mat cv_image = cv::imread(filename);

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	return from_cv_mat<uint8_t>(cv_grayscale);
}

inline bool imread(Array<uint8_t> &dst, const std::string &filename)
{
	cv::Mat cv_image = cv::imread(filename);
	if (cv_image.empty()) 
		return false;

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	from_cv_mat(dst, cv_grayscale);
	return true;
}

inline bool imwrite(const Array<uint8_t> &image, const std::string &filename)
{
	return cv::imwrite(filename, to_cv_mat(image));
}

inline bool imwrite(const Array<cv::Vec3b> &image, const std::string &filename)
{
	return cv::imwrite(filename, to_cv_mat(image));
}

inline void imshow(const Array<uint8_t> &image)
{
	cv::imshow("image", to_cv_mat(image));
	cv::waitKey(0);
}

inline void imshow(const Array<cv::Vec3b> &image)
{
	cv::imshow("image", to_cv_mat(image));
	cv::waitKey(0);
}

} // namespace np

#endif // NUMCPP_OPENCV_H_