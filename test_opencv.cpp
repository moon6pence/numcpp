#include <iostream>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main()
{
	cout << "Hello World!" << endl;

	// Testcase: OpenCV
	{
		using namespace cv;

		Mat image = imread("Lena.bmp");
		if (image.empty())
		{
			cout << "Cannot find image file." << endl;
			return -1;
		}
	
		imshow("test_opencv", image);
		waitKey(0);
	}

	return 0;
}