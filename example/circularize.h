#include <numcpp.h>

namespace numcpp {

struct Circularize
{
	array_t<float, 2> x_map, y_map;

	void operator() (array_t<uint8_t, 2> &dst, const array_t<uint8_t, 2> &src, const int DIAMETER)
	{
		if (x_map.empty() || y_map.empty())
			buildCircularizeMap(src.width(), src.height(), DIAMETER);

		cv::remap(to_cv_mat(src), to_cv_mat(dst), to_cv_mat(x_map), to_cv_mat(y_map), CV_INTER_LINEAR);
	}

	void buildCircularizeMap(const int WIDTH, const int HEIGHT, const int DIAMETER)
	{
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
		x_map = array<float>(theta.width(), theta.height());
		map(x_map, theta, [PI, WIDTH](float _theta)
		{
			return (_theta + PI) * (WIDTH-1) / (2*PI);
		});

		// rho
		auto rho = array<float>(X.width(), X.height());
		map(rho, X, Y, [](float _x, float _y)
		{
			return sqrt(_x * _x + _y * _y);
		});

		// Y map: interpolate (0, RADIUS) -> (1, height)
		y_map = array<float>(rho.width(), rho.height());
		map(y_map, rho, [HEIGHT, DIAMETER](float _rho)
		{
			return _rho * (HEIGHT-1) / (DIAMETER/2);
		});
	}
};

} // namespace numcpp