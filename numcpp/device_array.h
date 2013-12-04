#ifndef __DEVICE_ARRAY_H__
#define __DEVICE_ARRAY_H__

#include "array.h"
#include <cuda_runtime.h>

namespace numcpp {

template <typename T, int Dim>
struct device_array_t : private array_t<T, Dim>
{
public:
	device_array_t() : array_t<T, Dim>() { }

	device_array_t(std::shared_ptr<void> address, T *origin, int *shape) :
		array_t<T, Dim>(address, origin, shape) { }

	~device_array_t()
	{
		// TODO: deallocate shape
	}

private:
	device_array_t(device_array_t &) { }
	const device_array_t &operator=(const device_array_t &) { return *this; }

public:
	// Move constructor
	device_array_t(device_array_t &&other) : array_t<T, Dim>(std::move(other)) { }

	// Move assign
	const device_array_t &operator=(device_array_t &&other)
	{
		return array_t<T, Dim>::operator=(other);
	}

	// inherits array_t (private)
	int size() const { return array_t<T, Dim>::size(); }
	int size(int dim) const { return array_t<T, Dim>::size(); }
	T *raw_pointer() { return array_t<T, Dim>::raw_pointer(); }
	const T *raw_pointer() const { return array_t<T, Dim>::raw_pointer(); }

	bool empty() const { return array_t<T, Dim>::empty(); }
	int length() const { return array_t<T, Dim>::length(); }
	int height() const { return array_t<T, Dim>::height(); }
	int width() const { return array_t<T, Dim>::width(); }
	operator T *() { return array_t<T, Dim>::operator T *(); }
	operator const T *() const { return array_t<T, Dim>::operator const T*(); }
};

template <typename T>
void device_array_deleter(T *p)
{
	cudaFree(p);
}

#ifdef VARIADIC_TEMPLATE

/** Allocate device array */
template <typename T, typename... Shape>
device_array_t<T, sizeof...(Shape)> device_array(Shape... shape)
{
	int size = TMP_V::product(shape...);

	// allocate buffer
	T *buffer = nullptr;
	cudaMalloc((void **)&buffer, size * sizeof(T));

	// address
	std::shared_ptr<void> address(buffer, device_array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[sizeof...(Shape)];
	TMP_V::copy(new_shape, shape...);

	return device_array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

#else // ifndef VARIADIC_TEMPLATE

/** Allocate device array */
template <typename T>
array_t<T, 1> device_array(int shape0)
{
	int size = shape0;

	// allocate buffer
	T *buffer = nullptr;
	cudaMalloc((void **)&buffer, size * sizeof(T));

	// address
	std::shared_ptr<void> address(buffer, device_array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[1];
	new_shape[0] = shape0;

	return device_array_t<T, 1>(address, origin, new_shape);
}

template <typename T>
array_t<T, 2> device_array(int shape0, int shape1)
{
	int size = shape0 * shape1;

	// allocate buffer
	T *buffer = nullptr;
	cudaMalloc((void **)&buffer, size * sizeof(T));

	// address
	std::shared_ptr<void> address(buffer, device_array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[2];
	new_shape[0] = shape0;
	new_shape[1] = shape1;

	return device_array_t<T, 2>(address, origin, new_shape);
}

template <typename T>
array_t<T, 3> device_array(int shape0, int shape1, int shape2)
{
	int size = shape0 * shape1 * shape2;

	// allocate buffer
	T *buffer = nullptr;
	cudaMalloc((void **)&buffer, size * sizeof(T));

	// address
	std::shared_ptr<void> address(buffer, device_array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[3];
	new_shape[0] = shape0;
	new_shape[1] = shape1;
	new_shape[2] = shape2;

	return device_array_t<T, 3>(address, origin, new_shape);
}

#endif // VARIADIC_TEMPLATE

template <typename T, int Dim>
void host_to_device(device_array_t<T, Dim> &dst_d, const array_t<T, Dim> &src)
{
	cudaMemcpy(dst_d, src, dst_d.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T, int Dim>
void device_to_host(array_t<T, Dim> &dst, const device_array_t<T, Dim> &src_d)
{
	cudaMemcpy(dst, src_d, dst.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

} // namespace numcpp

#endif // __DEVICE_ARRAY_H__