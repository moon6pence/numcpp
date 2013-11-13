#ifndef __ARRAY_FUNCTION_H__
#define __ARRAY_FUNCTION_H__

#include <algorithm>
#include <iostream>

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

template <typename T, int Dim>
void fill(array_t<T, Dim> &dst, const T &value)
{
	std::fill(begin(dst), end(dst), value);
}

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

} // namespace numcpp

#endif // __ARRAY_FUNCTION_H__