#ifndef NUMCPP_ARRAY_FUNCTION_H_
#define NUMCPP_ARRAY_FUNCTION_H_

#include "array.h"

#include <algorithm>
#include <functional> // plus
#include <numeric> // accumulate
#include <iostream>
#include <stdint.h>

namespace np {

template <typename T>
T *begin(Array<T> &array)
{
	return array.raw_ptr();
}

template <typename T>
const T *begin(const Array<T> &array)
{
	return array.raw_ptr();
}

template <typename T>
T *end(Array<T> &array)
{
	// TODO: this is wrong for array with custom stride
	return array.raw_ptr() + array.length();
}

template <typename T>
const T *end(const Array<T> &array)
{
	// TODO: this is wrong for array with custom stride
	return array.raw_ptr() + array.length();
}

// std::for_each
template <typename T, class Function>
void for_each(const Array<T> &array, Function fn)
{
	std::for_each(begin(array), end(array), fn);
}

template <typename T>
void print(const Array<T> &array)
{
	using namespace std;

	for_each(array, [](const T& value) { cout << value << " "; });
	cout << endl;
}

template <>
inline void print(const Array<uint8_t> &array)
{
	using namespace std;

	for_each(array, [](const uint8_t& value) { cout << (int)value << " "; });
	cout << endl;
}

// std::fill
template <typename T>
void fill(Array<T> &dst, const T& value)
{
	std::fill(begin(dst), end(dst), value);	
}

// std::transform
template <typename T, typename U, class UnaryFunction>
void transform(Array<T> &dst, const Array<U> &src, UnaryFunction fn)
{
	if (dst.size() != src.size()) 
		dst = similar<T>(src);

	std::transform(begin(src), end(src), begin(dst), fn);
}

template <typename T, class UnaryFunction>
void transform(Array<T> &srcDst, UnaryFunction fn)
{
	std::transform(begin(srcDst), end(srcDst), begin(srcDst), fn);
}

template <typename T, typename U, typename V, class BinaryFunction>
void transform(
	Array<T> &dst, 
	const Array<U> &src1, 
	const Array<V> &src2, 
	BinaryFunction fn)
{
	// assert(src1.ndims() == src2.ndims())
	// assert(src1.shape() == src2.shape())

	if (dst.size() != src1.size())
		dst = similar<T>(src1);

	std::transform(begin(src1), end(src1), begin(src2), begin(dst), fn);
}

// std::accumulate
template <typename T, class BinaryFunction>
T accumulate(const Array<T> &array, T init, BinaryFunction binary_op)
{
	return std::accumulate(begin(array), end(array), init, binary_op);
}

template <typename T>
T sum(const Array<T> &array)
{
	return accumulate(array, T(), std::plus<T>());
}

template <typename T>
T mean(const Array<T> &array)
{
	return sum(array) / array.length();
}

} // namespace np

#endif // NUMCPP_ARRAY_FUNCTION_H_