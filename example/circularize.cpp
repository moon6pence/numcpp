#include <numcpp.h>
#include <math.h>

int main(int argc, char *argv[])
{
	using namespace numcpp;

	auto image = imread("example/input.bmp");
	const int width = image.width(), height = image.height();;
	printf("width = %d, height = %d\n", width, height);

	const int DIAMETER = 1024;
	auto X = array<float>(DIAMETER, DIAMETER);
	auto Y = array<float>(DIAMETER, DIAMETER);
	meshgrid(X, Y, colon(0.f, DIAMETER - 1.f), colon(0.f, DIAMETER - 1.f));

	const int X0 = DIAMETER / 2, Y0 = DIAMETER / 2;
	map(X, [X0](int _x) { return _x - X0; });
	map(Y, [Y0](int _y) { return _y - Y0; });

	// theta
	auto theta = array<float>(X.width(), X.height());
	map(theta, Y, X, [](float _y, float _x) { return atan2(_y, _x); });

	// X map: interpolate (-pi, pi) -> (0, width - 1)
	const float PI = 3.1415927f;
	auto x_map = array<float>(theta.width(), theta.height());
	map(x_map, theta, [PI, width](float _theta)
	{
		return (_theta + PI) * (width-1) / (2*PI);
	});

	// rho
	auto rho = array<float>(X.width(), X.height());
	map(rho, X, Y, [](float _x, float _y)
	{
		return sqrt(_x * _x + _y * _y);
	});

	// Y map: interpolate (0, RADIUS) -> (1, height)
	auto y_map = array<float>(rho.width(), rho.height());
	map(y_map, rho, [height, DIAMETER](float _rho)
	{
		return _rho * (height-1) / (DIAMETER/2);
	});

	auto result_image = array<uint8_t>(DIAMETER, DIAMETER);

	ippiRemap_8u_C1R(
		image, ippiSize(image), stepBytes(image), ippiRect(image), 
		x_map, stepBytes(x_map), 
		y_map, stepBytes(y_map), 
		result_image, stepBytes(result_image), ippiSize(result_image), 
		IPPI_INTER_LINEAR);

	imwrite(result_image, "example/output.bmp");
	imshow(result_image);
}