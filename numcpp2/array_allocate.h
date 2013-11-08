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
	int size = multiply_all(shape...);

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

#include <string.h> // memset

template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> zeros(Shape... shape)
{
	array_t<T, sizeof...(Shape)> result = array<T, Shape...>(shape...);
	memset(result, 0, result.size() * sizeof(T));
	return result;
}

#endif // __ARRAY_FUNCTIONS_H__