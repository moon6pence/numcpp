#include "circularize.h"
#include <iostream>

int main(int argc, char *argv[])
{
	using namespace std;
	using namespace numcpp;

	auto image = imread("Lena.bmp");
	if (image.empty())
	{
		cout << "Cannot read image file!" << endl;
		return -1;
	}

	const int DIAMETER = 1024;
	array_t<uint8_t> result_image(DIAMETER, DIAMETER);

	Circularize circularize;
	circularize(result_image, image, DIAMETER);

	imwrite(result_image, "result_circularize.bmp");
	imshow(result_image);

	return 0;
}