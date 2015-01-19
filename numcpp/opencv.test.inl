#include <numcpp/opencv.h>

namespace {

using namespace np;

TEST(OpenCV, HelloOpenCV)
{
	cv::Mat image = cv::imread("Lena.bmp");
	EXPECT_EQ(image.cols, 512);
	EXPECT_EQ(image.rows, 512);

	// cv::imshow("Hello OpenCV", image);
	// cv::waitKey(0);
}

TEST(OpenCV, ToCvMat)
{
	// uint_t
	Array<uint8_t, 2> mat(5, 8);
	for (int i = 0; i < mat.length(); i++)
		mat(i) = 128;
	mat(2, 3) = 255;
	mat(4, 1) = 0;

	cv::Mat cv_mat = to_cv_mat(mat);
	ASSERT_EQ(cv_mat.cols, mat.size(0));
	ASSERT_EQ(cv_mat.rows, mat.size(1));
	EXPECT_EQ(cv_mat.at<uint8_t>(3, 2), 255);
	EXPECT_EQ(cv_mat.at<uint8_t>(1, 4), 0);
	for (int y = 0; y < mat.size(0); y++)
		for (int x = 0; x < mat.size(1); x++)
			ASSERT_EQ(mat(y, x), cv_mat.at<uint8_t>(x, y));

	// float
	Array<float, 2> mat2(5, 8);
	for (int i = 0; i < mat2.length(); i++)
		mat2(i) = 0.5f;
	//fill(mat2, 0.5f);
	mat2(2, 3) = 1.0f;
	mat2(4, 1) = 0.0f;

	cv::Mat cv_mat2 = to_cv_mat(mat2);
	ASSERT_EQ(cv_mat2.cols, mat2.size(0));
	ASSERT_EQ(cv_mat2.rows, mat2.size(1));
	EXPECT_EQ(cv_mat2.at<float>(3, 2), 1.0f);
	EXPECT_EQ(cv_mat2.at<float>(1, 4), 0.0f);
	for (int y = 0; y < mat2.size(0); y++)
		for (int x = 0; x < mat2.size(1); x++)
			ASSERT_EQ(mat2(y, x), cv_mat2.at<float>(x, y));
}

TEST(OpenCV, FromCvMatVoid)
{
	cv::Mat cv_mat(5, 8, CV_8U, cvScalar(128));
	cv_mat.at<uint8_t>(3, 2) = 123;
	cv_mat.at<uint8_t>(1, 4) = 7;

	// uint8_t
	Array<uint8_t, 2> mat;
	from_cv_mat(mat, cv_mat);
	ASSERT_EQ(cv_mat.cols, mat.size(0));
	ASSERT_EQ(cv_mat.rows, mat.size(1));
	for (int y = 0; y < mat.size(1); y++)
		for (int x = 0; x < mat.size(0); x++)
			EXPECT_EQ(mat(x, y), cv_mat.at<uint8_t>(y, x));
	EXPECT_EQ(mat(2, 3), 123);
	EXPECT_EQ(mat(4, 1), 7);

	// wrong type of array
	Array<double, 2> mat_error;
	from_cv_mat(mat_error, cv_mat);
	EXPECT_TRUE(empty(mat_error));

	// float
	cv::Mat cv_mat2(5, 8, CV_32F, cvScalar(0.5f));
	cv_mat2.at<float>(3, 2) = 1.0f;
	cv_mat2.at<float>(1, 4) = 0.0f;

	Array<float, 2> mat2;
	from_cv_mat(mat2, cv_mat2);
	ASSERT_EQ(cv_mat2.cols, mat2.size(0));
	ASSERT_EQ(cv_mat2.rows, mat2.size(1));
	for (int y = 0; y < mat2.size(1); y++)
		for (int x = 0; x < mat2.size(0); x++)
			EXPECT_EQ(mat2(x, y), cv_mat2.at<float>(y, x));
}

TEST(OpenCV, ImRead)
{
	cv::Mat cv_image = cv::imread("Lena.bmp");
	ASSERT_EQ(cv_image.rows, 512);
	ASSERT_EQ(cv_image.cols, 512);

	cv::Mat cv_grayscale;
	cv::cvtColor(cv_image, cv_grayscale, CV_BGR2GRAY);

	Array<uint8_t, 2> image;
	bool result = imread(image, "Lena.bmp");
	ASSERT_TRUE(result);
	ASSERT_EQ(cv_image.rows, image.size(0));
	ASSERT_EQ(cv_image.cols, image.size(1));

	EXPECT_EQ(image(412, 43), cv_grayscale.at<uint8_t>(43, 412));
	EXPECT_EQ(image(240, 360), cv_grayscale.at<uint8_t>(360, 240));
	EXPECT_EQ(image(0, 0), cv_grayscale.at<uint8_t>(0, 0));
	EXPECT_EQ(image(511, 511), cv_grayscale.at<uint8_t>(511, 511));
}

TEST(OpenCV, ImWrite)
{
	Array<uint8_t, 2> image;
	bool result = imread(image, "Lena.bmp");
	ASSERT_TRUE(result);
	// imshow(image);
	result = imwrite(image, "result_imwrite.bmp");
	ASSERT_TRUE(result);
}

#if 0

TEST(OpenCV, FromCvMatRuntime)
{
	cv::Mat cv_mat(5, 5, CV_8U, cvScalar(128));
	cv_mat.at<uint8_t>(3, 2) = 123;
	cv_mat.at<uint8_t>(1, 4) = 7;

	BaseArray a1 = from_cv_mat(cv_mat);
	ASSERT_FALSE(empty(a1));
	ASSERT_EQ(cv_mat.rows, a1.size(0));
	ASSERT_EQ(cv_mat.cols, a1.size(1));

	Array<uint8_t> image(std::move(a1));
	EXPECT_EQ(123, image(2, 3));
	EXPECT_EQ(7, image(4, 1));
	for (int y = 0; y < image.size(0); y++)
		for (int x = 0; x < image.size(1); x++)
			EXPECT_EQ(cv_mat.at<uint8_t>(y, x), image(x, y));
}

TEST(OpenCV, ColorImage)
{
	using namespace std;

	cv::Mat cv_image = cv::imread("Lena.bmp");
	//cout << cv_image.type() << endl;
	//cout << CV_8UC3 << endl;
	ASSERT_EQ(cv_image.type(), CV_8UC3);

	np::BaseArray image = np::from_cv_mat(cv_image);
	//cout << image.itemSize() << endl;
	ASSERT_EQ(image.itemSize(), 3);

	np::Array<cv::Vec3b> image3b(std::move(image));
	cv::Vec3b point = image3b.at(0, 0);
	//cout << point << endl;
	ASSERT_EQ(point, cv::Vec3b(125, 137, 226));

	// np::imshow(image3b);
	ASSERT_TRUE(imwrite(image3b, "result_imwrite.bmp"));
}

#endif // if 0

} // anonymous namespace