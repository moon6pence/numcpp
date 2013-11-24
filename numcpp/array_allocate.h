#ifndef __NUMCPP_ARRAY_FUNCTIONS_H__
#define __NUMCPP_ARRAY_FUNCTIONS_H__

#include "array.h"
#include "array_function.h"

#ifdef INITIALIZER_LIST
#include <initializer_list>
#endif

#include <assert.h>

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
	TMP_N<Dim>::fill(new_shape, 0);	

	return array_t<T, Dim>(address, origin, new_shape);
}

template <typename T>
void array_deleter(T const *p)
{
	delete[] p;
}

#ifdef VARIADIC_TEMPLATE

/** Allocate array with given shape */
template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> array(Shape... shape)
{
	int size = TMP_V::product(shape...);

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[sizeof...(Shape)];
	TMP_V::copy(new_shape, shape...);

	return array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

#else // ifndef VARIADIC_TEMPLATE

/** Allocate 1 dimension array */
template <typename T>
array_t<T, 1> array(int shape0)
{
	int size = shape0;

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[1];
	new_shape[0] = shape0;

	return array_t<T, 1>(address, origin, new_shape);
}

/** Allocate 2 dimension array */
template <typename T>
array_t<T, 2> array(int shape0, int shape1)
{
	int size = shape0 * shape1;

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[2];
	new_shape[0] = shape0;
	new_shape[1] = shape1;

	return array_t<T, 2>(address, origin, new_shape);
}

/** Allocate 3 dimension array */
template <typename T>
array_t<T, 3> array(int shape0, int shape1, int shape2)
{
	int size = shape0 * shape1 * shape2;

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[3];
	new_shape[0] = shape0;
	new_shape[1] = shape1;
	new_shape[2] = shape2;

	return array_t<T, 3>(address, origin, new_shape);
}

#endif // VARIADIC_TEMPLATE

#ifdef INITIALIZER_LIST

/** Allocate array assigned with initializer_list */
template <typename T>
array_t<T, 1> array(std::initializer_list<T> values)
{
	auto result = array<T>(values.size());
	std::copy(begin(values), end(values), begin(result));

	return std::move(result);
}

#endif // INITIALIZER_LIST

/** One-dimensional array from j to k, such as {j, j+1, ..., k} */
template <typename T>
array_t<T, 1> colon(T j, T k)
{
	if (k < j) 
		return empty<T, 1>();

	auto result = array<T>((int)(k - j + 1));
	for (int index = 0; index < result.length(); index++)
		result(index) = j + index;

	return std::move(result);
}

/** One-dimensional array from j to k step i, such as {j, j+i, j+2i, ... } */
template <typename T>
array_t<T, 1> colon(T j, T i, T k)
{
	if (i == 0 || (i > 0 && j > k) || (i < 0 && j < k))
		return empty<T, 1>();	

	auto result = array<T>((int)((k - j) / i) + 1);
	for (int index = 0; index < result.length(); index++)
		result(index) = j + index * i;

	return std::move(result);
}

// TODO: return two array at once
template <typename T> 
void meshgrid(
	array_t<T, 2> &X, array_t<T, 2> &Y, 
	const array_t<T, 1> &xgv, const array_t<T, 1> &ygv)
{
	assert(xgv.length() == X.width() && xgv.length() == Y.width());
	assert(ygv.length() == Y.height() && ygv.length() == Y.height());

	for (int y = 0; y < ygv.length(); y++)
		for (int x = 0; x < xgv.length(); x++)
		{
			X(x, y) = xgv(x);
			Y(x, y) = ygv(y);
		}
}

} // namespace numcpp

#endif // __NUMCPP_ARRAY_FUNCTIONS_H__