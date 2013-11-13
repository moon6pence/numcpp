#ifndef __ARRAY_FUNCTIONS_H__
#define __ARRAY_FUNCTIONS_H__

#include "array_function.h"

namespace numcpp {

template <typename T>
void empty_deleter(T const *p)
{
}

/** Allocate empty array */
template <typename T, int Dim>
array_t<T, Dim> empty()
{
	// data buffer is nullptr
	T *buffer = nullptr;

	// address
	std::shared_ptr<void> address(buffer, empty_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[Dim];
	TMP<Dim>::fill(new_shape, 0);	

	return array_t<T, Dim>(address, origin, new_shape);
}

template <typename T>
void array_deleter(T const *p)
{
	delete[] p;
}

/** Allocate array with given shape */
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
	int *new_shape = new int[sizeof...(Shape)];
	copy(new_shape, shape...);

	return array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

/** Array filled with zero */
template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> zeros(Shape... shape)
{
	array_t<T, sizeof...(Shape)> result = array<T, Shape...>(shape...);
	fill(result, T());
	return result;
}

/** Array filled with one */
template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> ones(Shape... shape)
{
	array_t<T, sizeof...(Shape)> result = array<T, Shape...>(shape...);
	fill(result, T() + 1);
	return result;
}

} // namespace numcpp

#endif // __ARRAY_FUNCTIONS_H__