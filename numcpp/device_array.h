#ifndef NUMCPP_DEVICE_ARRAY_H_
#define NUMCPP_DEVICE_ARRAY_H_

#include "base_array.h"
#include <cuda_runtime.h>

namespace numcpp {

template <typename T>
T *device_array_allocator(int size)
{
	T *ptr = nullptr;
	cudaMalloc((void **)&ptr, size * sizeof(T));
	return ptr;
}

template <typename T>
void device_array_deallocator(T *ptr)
{
	cudaFree(ptr);
}

template <typename T>
struct device_array_t : public base_array_t<T>
{
public:
	device_array_t()
	{
		base_array_t<T>::init();
	}

	device_array_t(int size0)
	{
		int size = size0;

		int *shape = new int[1];
		shape[0] = size0;

		auto ptr = std::shared_ptr<T>(
			device_array_allocator<T>(size), device_array_deallocator<T>);

		base_array_t<T>::init(1, size, shape, ptr);
	}

	device_array_t(int size0, int size1)
	{
		int size = size0 * size1;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		auto ptr = std::shared_ptr<T>(
			device_array_allocator<T>(size), device_array_deallocator<T>);	

		base_array_t<T>::init(2, size, shape, ptr);
	}

	device_array_t(int size0, int size1, int size2)
	{
		int size = size0 * size1 * size2;

		int *shape = new int[3];
		shape[0] = size0;
		shape[1] = size1;
		shape[2] = size2;

		auto ptr = std::shared_ptr<T>(
			device_array_allocator<T>(size), device_array_deallocator<T>);	

		base_array_t<T>::init(3, size, shape, ptr);
	}

	~device_array_t()
	{
		base_array_t<T>::free();
	}

private:
	// disable copy constructor, assign
	device_array_t(device_array_t &) { }
	const device_array_t &operator=(const device_array_t &) { return *this; }

public:
	// move constructor
	device_array_t(device_array_t &&other)
	{
		base_array_t<T>::init(
			other._ndims, other._size, other._shape, std::move(other._ptr));

		other.init();
	}

	// move assign
	const device_array_t &operator=(device_array_t &&other)
	{
		base_array_t<T>::free();

		base_array_t<T>::init(
			other._ndims, other._size, other._shape, std::move(other._ptr));

		other.init();

		return *this;
	}
};

} // namespace numcpp

#endif // NUMCPP_DEVICE_ARRAY_H_