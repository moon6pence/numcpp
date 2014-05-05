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
T *begin(array_t<T> &array)
{
	return array.raw_ptr();
}

template <typename T>
const T *begin(const array_t<T> &array)
{
	return array.raw_ptr();
}

template <typename T>
T *end(array_t<T> &array)
{
	// TODO: this is wrong for array with custom stride
	return array.raw_ptr() + array.length();
}

template <typename T>
const T *end(const array_t<T> &array)
{
	// TODO: this is wrong for array with custom stride
	return array.raw_ptr() + array.length();
}

// std::for_each
template <typename T, class Function>
void for_each(const array_t<T> &array, Function fn)
{
	std::for_each(begin(array), end(array), fn);
}

template <typename T>
void print(const array_t<T> &array)
{
	using namespace std;

	for_each(array, [](const T& value) { cout << value << " "; });
	cout << endl;
}

template <>
inline void print(const array_t<uint8_t> &array)
{
	using namespace std;

	for_each(array, [](const uint8_t& value) { cout << (int)value << " "; });
	cout << endl;
}

// std::fill
template <typename T>
void fill(array_t<T> &dst, const T& value)
{
	std::fill(begin(dst), end(dst), value);	
}

// std::transform
template <typename T, typename U, class UnaryFunction>
void transform(array_t<T> &dst, const array_t<U> &src, UnaryFunction fn)
{
	// TODO: similar()
	if (dst.size() != src.size())
		dst = array_t<T>(src);

	std::transform(begin(src), end(src), begin(dst), fn);
}

template <typename T, class UnaryFunction>
void transform(array_t<T> &srcDst, UnaryFunction fn)
{
	std::transform(begin(srcDst), end(srcDst), begin(srcDst), fn);
}

template <typename T, typename U, typename V, class BinaryFunction>
void transform(
	array_t<T> &dst, 
	const array_t<U> &src1, 
	const array_t<V> &src2, 
	BinaryFunction fn)
{
	// assert(src1.ndims() == src2.ndims())
	// assert(src1.shape() == src2.shape())

	// TODO: similar()
	if (dst.size() != src1.size())
		dst = array_t<T>(src1.size());

	std::transform(begin(src1), end(src1), begin(src2), begin(dst), fn);
}

// std::accumulate
template <typename T, class BinaryFunction>
T accumulate(const array_t<T> &array, T init, BinaryFunction binary_op)
{
	return std::accumulate(begin(array), end(array), init, binary_op);
}

template <typename T>
T sum(const array_t<T> &array)
{
	return accumulate(array, T(), std::plus<T>());
}

template <typename T>
T mean(const array_t<T> &array)
{
	return sum(array) / array.length();
}

} // namespace np

#endif // NUMCPP_ARRAY_FUNCTION_H_