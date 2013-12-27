#include <numcpp/opencv.h>
#include <numcpp/stl.h>

namespace {

using namespace numcpp;

TEST(OpenCV, HelloOpenCV)
{
	cv::Mat image = cv::imread("Lena.bmp");
	ASSERT_EQ(image.cols, 512);
	ASSERT_EQ(image.rows, 512);

	// cv::imshow("Hello OpenCV", image);
	// cv::waitKey(0);
}

TEST(OpenCV, ToCvMat)
{
	array_t<uint8_t> mat(5, 5);
	fill(mat, (uint8_t)128);
	mat(3, 2) = 255;
	mat(1, 4) = 0;

	cv::Mat cv_mat = to_cv_mat(mat);
	ASSERT_EQ(cv_mat.rows, mat.size(0));
	ASSERT_EQ(cv_mat.cols, mat.size(1));
	for (int y = 0; y < mat.size(0); y++)
		for (int x = 0; x < mat.size(1); x++)
			ASSERT_EQ(mat(y, x), cv_mat.at<uint8_t>(y, x));

	array_t<float> mat2(5, 5);
	fill(mat2, 0.5f);
	mat(3, 2) = 1.0f;
	mat(1, 4) = 0.0f;

	cv::Mat cv_mat2 = to_cv_mat(mat2);
	ASSERT_EQ(cv_mat2.rows, mat2.size(0));
	ASSERT_EQ(cv_mat2.cols, mat2.size(1));
	for (int y = 0; y < mat2.size(0); y++)
		for (int x = 0; x < mat2.size(1); x++)
			ASSERT_EQ(mat2(y, x), cv_mat2.at<float>(y, x));
}

TEST(OpenCV, FromCvMat)
{
	cv::Mat cv_mat(5, 5, CV_8U, cvScalar(128));
	cv_mat.at<uint8_t>(3, 2) = 123;
	cv_mat.at<uint8_t>(1, 4) = 7;

	auto mat = from_cv_mat<uint8_t>(cv_mat);
	ASSERT_EQ(cv_mat.rows, mat.size(0));
	ASSERT_EQ(cv_mat.cols, mat.size(1));
	for (int y = 0; y < mat.size(0); y++)
		for (int x = 0; x < mat.size(1); x++)
			ASSERT_EQ(mat(y, x), cv_mat.at<uint8_t>(y, x));
	ASSERT_EQ(mat(3, 2), 123);
	ASSERT_EQ(mat(1, 4), 7);

	cv::Mat cv_mat2(5, 5, CV_32F, cvScalar(0.5f));
	cv_mat2.at<float>(3, 2) = 1.0f;
	cv_mat2.at<float>(1, 4) = 0.0f;

	auto mat2 = from_cv_mat<float>(cv_mat2);
	ASSERT_EQ(cv_mat2.rows, mat2.size(0));
	ASSERT_EQ(cv_mat2.cols, mat2.size(1));
	for (int y = 0; y < mat2.size(0); y++)
		for (int x = 0; x < mat2.size(1); x++)
			ASSERT_EQ(mat2(y, x), cv_mat2.at<float>(y, x));
}

TEST(OpenCV, ImRead)
{
	cv::Mat cv_image = cv::imread("Lena.bmp");
	ASSERT_EQ(cv_image.rows, 512);
	ASSERT_EQ(cv_image.cols, 512);

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	auto image = imread("Lena.bmp");
	ASSERT_EQ(image.size(0), cv_image.rows);
	ASSERT_EQ(image.size(1), cv_image.cols);

	EXPECT_EQ(image(43, 412), cv_grayscale.at<uint8_t>(43, 412));
	EXPECT_EQ(image(360, 240), cv_grayscale.at<uint8_t>(360, 240));
	EXPECT_EQ(image(0, 0), cv_grayscale.at<uint8_t>(0, 0));
	EXPECT_EQ(image(511, 511), cv_grayscale.at<uint8_t>(511, 511));
}

TEST(OpenCV, ImWrite)
{
	auto image = imread("Lena.bmp");
	ASSERT_TRUE(imwrite(image, "result_imwrite.bmp"));
}

} // anonymous namespace