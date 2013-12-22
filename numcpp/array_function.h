#ifndef NUMCPP_ARRAY_FUNCTION_H_
#define NUMCPP_ARRAY_FUNCTION_H_

#include "array.h"

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

#endif // NUMCPP_ARRAY_FUNCTION_H_