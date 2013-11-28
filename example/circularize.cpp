#include "circularize.h"

int main(int argc, char *argv[])
{
	using namespace numcpp;

	auto image = imread("example/input.bmp");

	const int DIAMETER = 1024;
	auto result_image = array<uint8_t>(DIAMETER, DIAMETER);

	Circularize circularize;
	circularize(result_image, image, DIAMETER);

	imwrite(result_image, "example/output.bmp");
	imshow(result_image);
}