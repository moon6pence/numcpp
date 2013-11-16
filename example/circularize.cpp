#include <numcpp/array.h>
#include <math.h>

namespace np = numcpp;

int main(int argc, char *argv[])
{
	int width = 1024, height = 1024;

	const int DIAMETER = 5;
	auto X = np::array<float>(DIAMETER, DIAMETER);
	auto Y = np::array<float>(DIAMETER, DIAMETER);
	meshgrid(X, Y, np::colon(0.f, DIAMETER - 1.f), np::colon(0.f, DIAMETER - 1.f));

	const int X0 = DIAMETER / 2, Y0 = DIAMETER / 2;
	np::map(X, [X0](int _x) { return _x - X0; });
	np::map(Y, [Y0](int _y) { return _y - Y0; });

	// theta
	auto theta = np::array<float>(X.width(), X.height());
	np::map(theta, Y, X, atan2);

	// X map: interpolate (-pi, pi) -> (0, width - 1)
	const float PI = 3.1415927f;
	auto x_map = np::array<float>(theta.width(), theta.height());
	np::map(x_map, theta, [PI, width](float _theta)
	{
		return (_theta + PI) * (width-1) / (2*PI);
	});

	// rho
	auto rho = np::array<float>(X.width(), X.height());
	np::map(rho, X, Y, [](float _x, float _y)
	{
		return sqrt(_x * _x + _y * _y);
	});

	// Y map: interpolate (0, RADIUS) -> (1, height)
	auto y_map = np::array<float>(rho.width(), rho.height());
	np::map(y_map, rho, [height, DIAMETER](float _rho)
	{
		return _rho * (height-1) / (DIAMETER/2);
	});

	np::print(X);
	np::print(Y);
	np::print(theta);
	np::print(rho);
	np::print(x_map);
	np::print(y_map);
}