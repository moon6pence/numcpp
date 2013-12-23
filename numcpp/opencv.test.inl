#include <numcpp/opencv.h>

namespace {

TEST(OpenCV, HelloOpenCV)
{
	cv::Mat image = cv::imread("Lena.bmp");
	ASSERT_EQ(image.cols, 512);
	ASSERT_EQ(image.rows, 512);

	// cv::imshow("Hello OpenCV", image);
	// cv::waitKey(0);
}

} // anonymous namespace