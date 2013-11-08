#ifndef __ARRAY_FUNCTION_H__
#define __ARRAY_FUNCTION_H__

#include <algorithm>
#include <iostream>

template <typename T, int Dim>
void print(const array_t<T, Dim> &array)
{
	// TODO: print multi dimensional array
	std::for_each(array.begin(), array.end(), 
		[](const T _element)
		{
			std::cout << _element << " ";
		});

	std::cout << std::endl;
}

#endif // __ARRAY_FUNCTION_H__