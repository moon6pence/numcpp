#ifndef __NUMCPP_IMAGE_H__
#define __NUMCPP_IMAGE_H__

#include <string.h> // memcpy

#ifdef USE_MAGICK
// algorithm header must be included first due to gcc STL bug
// reference: http://stackoverflow.com/questions/19043109/gcc-4-8-1-combining-c-code-with-c11-code
#include <algorithm>
#include <Magick++.h>
#endif // USE_MAGICK

#ifdef USE_OPENCV
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#endif // USE_OPENCV

#include "array_allocate.h"

namespace numcpp {

#ifdef USE_MAGICK

array_t<uint8_t, 2> imread(const std::string &file_path)
{
	using namespace Magick;

	try
	{
		Image image;	
		image.read(file_path);

		Blob blob;
		image.write(&blob, "r");

		auto result = array<uint8_t>(image.size().width(), image.size().height());
		memcpy(result, blob.data(), blob.length());
		return std::move(result);
	}
	catch (Magick::Exception &error)
	{
		puts(error.what());
		return empty<uint8_t, 2>();
	}
}

void imwrite(const array_t<uint8_t, 2> &array, const std::string &file_path)
{
	using namespace Magick;

	try
	{
		Blob blob(array.raw_pointer(), array.size() * sizeof(uint8_t));

		Image image;
		image.size(Geometry(array.width(), array.height()));
		image.magick("r");
		image.read(blob);
		image.write(file_path);
	}
	catch (Magick::Exception &error)
	{
		puts(error.what());
	}
}

#endif // USE_MAGICK

#ifdef USE_OPENCV

array_t<uint8_t, 2> imread(const std::string &file_path)
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

void imshow(const array_t<uint8_t, 2> &image)
{
	using namespace cv;

	Mat cv_image(image.height(), image.width(), CV_8U, const_cast<uint8_t *>(image.raw_pointer()));

	cv::imshow("numcpp::imshow", cv_image);
	waitKey(0);
}

void imwrite(const array_t<uint8_t, 2> &image, const std::string &file_path)
{
	using namespace cv;

	Mat cv_image(image.height(), image.width(), CV_8U, const_cast<uint8_t *>(image.raw_pointer()));
	cv::imwrite(file_path, cv_image);
}

#endif // USE_OPENCV

} // namespace numcpp

#endif // __NUMCPP_IMAGE_H__