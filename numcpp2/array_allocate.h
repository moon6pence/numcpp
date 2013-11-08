#ifndef __ARRAY_FUNCTIONS_H__
#define __ARRAY_FUNCTIONS_H__

template <typename T>
void array_deleter(T const *p)
{
	delete[] p;
}

template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> array(Shape... shape)
{
	int size = product(shape...);

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	size_type *new_shape = new int[sizeof...(Shape)];
	copy(new_shape, shape...);

	return array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

#include <algorithm>

template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> zeros(Shape... shape)
{
	array_t<T, sizeof...(Shape)> result = array<T, Shape...>(shape...);
	std::fill(result.begin(), result.end(), T());
	return result;
}

template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> ones(Shape... shape)
{
	array_t<T, sizeof...(Shape)> result = array<T, Shape...>(shape...);
	std::fill(result.begin(), result.end(), T() + 1);
	return result;
}

#endif // __ARRAY_FUNCTIONS_H__