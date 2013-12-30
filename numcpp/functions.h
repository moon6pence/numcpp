#ifndef NUMCPP_FUNCTIONS_H_
#define NUMCPP_FUNCTIONS_H_

#include "array.h"

#include <fstream>
#include <vector>

namespace numcpp {

/** One-dimensional array from j to k, such as {j, j+1, ..., k} */
template <typename T>
array_t<T> colon(T j, T k)
{
	if (k < j) 
		return array_t<T>();

	array_t<T> result((int)(k - j + 1));
	for (int index = 0; index < result.size(0); index++)
		result(index) = j + index;

	return std::move(result);
}

/** One-dimensional array from j to k step i, such as {j, j+i, j+2i, ... } */
template <typename T>
array_t<T> colon(T j, T i, T k)
{
	if (i == 0 || (i > 0 && j > k) || (i < 0 && j < k))
		return array_t<T>();

	array_t<T> result((int)((k - j) / i) + 1);
	for (int index = 0; index < result.size(0); index++)
		result(index) = j + index * i;

	return std::move(result);
}

template <typename T>
void meshgrid(
	array_t<T> &X, array_t<T> &Y, 
	const array_t<T> &xgv, const array_t<T> &ygv)
{
	X.setSize(ygv.size(0), xgv.size(0));
	Y.setSize(ygv.size(0), xgv.size(0));

	for (int y = 0; y < ygv.size(0); y++)
		for (int x = 0; x < xgv.size(0); x++)
		{
			X(y, x) = xgv(x);
			Y(y, x) = ygv(y);
		}
}

template <typename T>
array_t<T> fromfile(const std::string &filename)
{
	using namespace std;

	std::ifstream file(filename);
	if (!file.is_open())
		return array_t<T>();

	vector<T> buffer;
	while (!file.eof())
	{
		T value;
		file >> value;

		if (!file.fail())
			buffer.push_back(value);
	}

	array_t<T> result(buffer.size());
	std::copy(begin(buffer), end(buffer), begin(result));
	return std::move(result);
}

} // namespace numcpp

#endif // NUMCPP_FUNCTIONS_H_