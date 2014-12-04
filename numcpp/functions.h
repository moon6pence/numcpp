#ifndef NUMCPP_FUNCTIONS_H_
#define NUMCPP_FUNCTIONS_H_

#include "array.h"

#include <fstream>
#include <vector>

namespace np {

/** One-dimensional array from j to k, such as {j, j+1, ..., k} */
template <typename T>
Array<T> colon(T j, T k)
{
	if (k < j) 
		return Array<T>();

	Array<T> result((int)(k - j + 1));
	for (int index = 0; index < result.size(0); index++)
		result(index) = j + index;

	return std::move(result);
}

/** One-dimensional array from j to k step i, such as {j, j+i, j+2i, ... } */
template <typename T>
Array<T> colon(T j, T i, T k)
{
	if (i == 0 || (i > 0 && j > k) || (i < 0 && j < k))
		return Array<T>();

	Array<T> result((int)((k - j) / i) + 1);
	for (int index = 0; index < result.size(0); index++)
		result(index) = j + index * i;

	return std::move(result);
}

/** Generate linearly spaced vectors */
template <typename T>
Array<T> linspace(T a, T b, int n)
{
	Array<T> result(n);
	for (int index = 0; index < result.size(0); index++)
		result(index) = a + (b - a) * index / (n - 1);

	return std::move(result);
}

template <typename T>
void meshgrid(
	Array<T> &X, Array<T> &Y, 
	const Array<T> &xgv, const Array<T> &ygv)
{
	auto expected_size = make_vector(xgv.length(), ygv.length());

	if (X.size() != expected_size) X = Array<T>(expected_size);
	if (Y.size() != expected_size) Y = Array<T>(expected_size);

	for (int y = 0; y < ygv.length(); y++)
		for (int x = 0; x < xgv.length(); x++)
		{
			X(x, y) = xgv(x);
			Y(x, y) = ygv(y);
		}
}

template <typename T>
Array<T> fromfile(const std::string &filename)
{
	using namespace std;

	std::ifstream file(filename);
	if (!file.is_open())
		return Array<T>();

	vector<T> buffer;
	while (!file.eof())
	{
		T value;
		file >> value;

		if (!file.fail())
			buffer.push_back(value);
	}

	Array<T> result((int)buffer.size());
	std::copy(begin(buffer), end(buffer), begin(result));
	return std::move(result);
}

} // namespace np

#endif // NUMCPP_FUNCTIONS_H_