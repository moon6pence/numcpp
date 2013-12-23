#ifndef NUMCPP_ARRAY_FUNCTION_H_
#define NUMCPP_ARRAY_FUNCTION_H_

#include "array.h"

#include <algorithm>
#include <iostream>

namespace numcpp {

template <typename T>
T *begin(array_t<T> &array)
{
	return array.raw_ptr();
}

template <typename T>
T *end(array_t<T> &array)
{
	return array.raw_ptr() + array.size();
}

template <typename T>
const T *begin(const array_t<T> &array)
{
	return array.raw_ptr();
}

template <typename T>
const T *end(const array_t<T> &array)
{
	return array.raw_ptr() + array.size();
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
void print(const array_t<uint8_t> &array)
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
template <typename T, class Function>
void transform(array_t<T> &dst, const array_t<T> &src, Function fn)
{
	std::transform(begin(src), end(src), begin(dst), fn);
}

} // namespace numcpp

#endif // NUMCPP_ARRAY_FUNCTION_H_