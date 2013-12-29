#include <numcpp/numcpp.h>
#include <numcpp/opencv.h>

namespace numcpp {

struct Circularize
{
	array_t<float> x_map, y_map;

	void operator() (array_t<uint8_t> &dst, const array_t<uint8_t> &src, const int DIAMETER)
	{
		if (x_map.empty() || y_map.empty())
			buildCircularizeMap(src.size(0), src.size(1), DIAMETER);

		cv::remap(to_cv_mat(src), to_cv_mat(dst), to_cv_mat(x_map), to_cv_mat(y_map), CV_INTER_LINEAR);
	}

	void buildCircularizeMap(const int HEIGHT, const int WIDTH, const int DIAMETER)
	{
		array_t<float> X(DIAMETER, DIAMETER);
		array_t<float> Y(DIAMETER, DIAMETER);
		meshgrid(X, Y, colon(0.f, DIAMETER - 1.f), colon(0.f, DIAMETER - 1.f));

		const int X0 = DIAMETER / 2, Y0 = DIAMETER / 2;
		transform(X, X, [X0](int _x) { return _x - X0; });
		transform(Y, Y, [Y0](int _y) { return _y - Y0; });

		// theta
		array_t<float> theta(X.size(0), X.size(1));
		transform(theta, Y, X, [](float _y, float _x) { return atan2(_y, _x); });

		// X map: interpolate (-pi, pi) -> (0, width - 1)
		const float PI = 3.1415927f;
		x_map = array_t<float>(theta.size(0), theta.size(1));
		transform(x_map, theta, [PI, WIDTH](float _theta)
		{
			return (_theta + PI) * (WIDTH-1) / (2*PI);
		});

		// rho
		array_t<float> rho(X.size(0), X.size(1));
		transform(rho, X, Y, [](float _x, float _y)
		{
			return sqrt(_x * _x + _y * _y);
		});

		// Y map: interpolate (0, RADIUS) -> (1, height)
		y_map = array_t<float>(rho.size(0), rho.size(1));
		transform(y_map, rho, [HEIGHT, DIAMETER](float _rho)
		{
			return _rho * (HEIGHT-1) / (DIAMETER/2);
		});
	}
};

} // namespace numcpp