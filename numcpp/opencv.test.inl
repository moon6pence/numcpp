#include <numcpp/opencv.h>
#include <numcpp/stl.h>

namespace {

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

} // anonymous namespace