#ifndef NUMCPP_CUDA_H_
#define NUMCPP_CUDA_H_

#include "base_array.h"
#include "array.h"

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
		setSize(size0);
	}

	device_array_t(int size0, int size1)
	{
		setSize(size0, size1);
	}

	device_array_t(int size0, int size1, int size2)
	{
		setSize(size0, size1, size2);
	}

	~device_array_t()
	{
		base_array_t<T>::free();
	}

	void setSize(int size0)
	{
		if (this->ndims() == 1 && 
			this->size(0) == size0) return;

		base_array_t<T>::free();

		int size = size0;

		int *shape = new int[1];
		shape[0] = size0;

		auto ptr = std::shared_ptr<void>(
			device_array_allocator<T>(size), device_array_deallocator<T>);	

		this->init(1, size, shape, ptr);
	}

	void setSize(int size0, int size1)
	{
		if (this->ndims() == 2 && 
			this->size(0) == size0 && 
			this->size(1) == size1) return;

		base_array_t<T>::free();

		int size = size0 * size1;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		auto ptr = std::shared_ptr<void>(
			device_array_allocator<T>(size), device_array_deallocator<T>);	

		this->init(2, size, shape, ptr);
	}

	void setSize(int size0, int size1, int size2)
	{
		if (this->ndims() == 3 && 
			this->size(0) == size0 && 
			this->size(1) == size1 && 
			this->size(2) == size2) return;

		base_array_t<T>::free();

		int size = size0 * size1 * size2;

		int *shape = new int[3];
		shape[0] = size0;
		shape[1] = size1;
		shape[2] = size2;

		auto ptr = std::shared_ptr<void>(
			device_array_allocator<T>(size), device_array_deallocator<T>);	

		this->init(3, size, shape, ptr);
	}

private:
	// disable copy constructor, assign
	device_array_t(device_array_t &) { }
	const device_array_t &operator=(const device_array_t &) { return *this; }

public:
	// inherits move constructor
	device_array_t(device_array_t &&other) : base_array_t<T>(std::move(other))
	{
	}

	// inherits move assign
	const device_array_t &operator=(device_array_t &&other)
	{
		base_array_t<T>::operator=(std::move(other));
		return *this;
	}

	// Convert from host array
	explicit device_array_t(const array_t<T> &array_h)
	{
		int ndims = array_h.ndims();

		int size = array_h.size();

		int *shape = new int[ndims];
		for (int i = 0; i < ndims; i++)
			shape[i] = array_h.size(i);

		auto ptr = std::shared_ptr<void>(
			device_array_allocator<T>(size), device_array_deallocator<T>);	

		base_array_t<T>::init(ndims, size, shape, ptr);
		host_to_device(*this, array_h);
	}
};

template <typename T>
void host_to_device(device_array_t<T> &dst_d, const array_t<T> &src)
{
	cudaMemcpy(dst_d, src, dst_d.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void device_to_host(array_t<T> &dst, const device_array_t<T> &src_d)
{
	cudaMemcpy(dst, src_d, dst.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

} // namespace numcpp

#endif // NUMCPP_CUDA_H_