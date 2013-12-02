#include "circularize.h"
#include <iostream>

int main(int argc, char *argv[])
{
	using namespace std;
	using namespace numcpp;

	auto image = imread("example/input.bmp");
	if (image.empty())
	{
		cout << "Cannot read image file!" << endl;
		return -1;
	}

	const int DIAMETER = 1024;
	auto result_image = numcpp::array<uint8_t>(DIAMETER, DIAMETER);

	Circularize circularize;
	circularize(result_image, image, DIAMETER);

	imwrite(result_image, "example/output.bmp");
	imshow(result_image);

	return 0;
}