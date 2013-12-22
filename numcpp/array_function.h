#ifndef NUMCPP_ARRAY_FUNCTION_H_
#define NUMCPP_ARRAY_FUNCTION_H_

#include "array.h"
#include <algorithm>

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
void fill(array_t<T> &dst, const T& value)
{
	std::fill(begin(dst), end(dst), value);	
}

} // namespace numcpp

#endif // NUMCPP_ARRAY_FUNCTION_H_