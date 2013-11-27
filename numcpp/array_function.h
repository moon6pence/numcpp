#ifndef __NUMCPP_ARRAY_FUNCTION_H__
#define __NUMCPP_ARRAY_FUNCTION_H__

#include "array.h"

#include <algorithm>
#include <iostream>
#include <fstream> // fromfile

namespace numcpp {

template <typename T, int Dim>
T *begin(array_t<T, Dim> &array)
{
	return array.raw_pointer();
}

template <typename T, int Dim>
T *end(array_t<T, Dim> &array)
{
	return array.raw_pointer() + array.size();
}

template <typename T, int Dim>
const T *begin(const array_t<T, Dim> &array)
{
	return array.raw_pointer();
}

template <typename T, int Dim>
const T *end(const array_t<T, Dim> &array)
{
	return array.raw_pointer() + array.size();
}

/** Fill array with same value */
template <typename T, int Dim>
void fill(array_t<T, Dim> &dst, const T &value)
{
	std::fill(begin(dst), end(dst), value);
}

/** Print all elements in array */
template <typename T, int Dim>
void print(const array_t<T, Dim> &array)
{
	// TODO: print multi dimensional array
	std::for_each(begin(array), end(array), 
		[](const T _element)
		{
			std::cout << _element << " ";
		});

	std::cout << std::endl;
}

/** Apply unary function to each element in array */
template <typename T, typename U, int Dim, typename UnaryFunction>
void map(array_t<T, Dim> &dst, const array_t<U, Dim> &src, UnaryFunction function)
{
	std::transform(begin(src), end(src), begin(dst), function);
}

/** Apply unary function to each element in array (in-place version) */
template <typename T, int Dim, typename UnaryFunction>
void map(array_t<T, Dim> &srcDst, UnaryFunction function)
{
	std::transform(begin(srcDst), end(srcDst), begin(srcDst), function);
}

/** Apply binary function to each element in array */
template <typename T, typename U, typename V, int Dim, typename BinaryFunction>
void map(
	array_t<T, Dim> &dst, 
	const array_t<U, Dim> &src1, const array_t<V, Dim> &src2, 
	BinaryFunction function)
{
	std::transform(begin(src1), end(src1), begin(src2), begin(dst), function);
}

/* Read array from text file */
template <typename T>
array_t<T, 1> fromfile(const std::string &file_name)
{
	using namespace std;

	std::fstream file(file_name, std::ios::in);
	if (!file.is_open()) return empty<T, 1>();

	vector<T> buffer;
	while (!file.eof())
	{
		T value;
		file >> value;

		if (!file.fail())
			buffer.push_back(value);
	}

	auto result = array<T>(buffer.size());
	std::copy(begin(buffer), end(buffer), begin(result));
	return std::move(result);
}

} // namespace numcpp

#endif // __NUMCPP_ARRAY_FUNCTION_H__