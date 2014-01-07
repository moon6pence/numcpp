#ifndef NUMCPP_CUDA_H_
#define NUMCPP_CUDA_H_

#include "base_array.h"
#include "array.h"

#include <cuda_runtime.h>

namespace numcpp {

struct device_allocator
{
	static void *allocate(int size)
	{
		void *ptr = nullptr;
		cudaMalloc((void **)&ptr, size);
		return ptr;
	}

	static void free(void *ptr)
	{
		cudaFree(ptr);
	}
};

template <typename T>
struct device_array_t : public base_array_t
{
public:
	device_array_t()
	{
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

private:
	// disable copy constructor, assign
	device_array_t(device_array_t &) { }
	const device_array_t &operator=(const device_array_t &) { return *this; }

public:
	// inherits move constructor
	device_array_t(device_array_t &&other) : base_array_t(std::move(other))
	{
	}

	// inherits move assign
	const device_array_t &operator=(device_array_t &&other)
	{
		base_array_t::operator=(std::move(other));
		return *this;
	}

	// Convert from host array
	explicit device_array_t(const array_t<T> &array_h)
	{
		host_to_device(*this, array_h);
	}

	void setSize(int size0)
	{
		base_array_t::setSize<device_allocator>(sizeof(T), size0);
	}

	void setSize(int size0, int size1)
	{
		base_array_t::setSize<device_allocator>(sizeof(T), size0, size1);
	}

	void setSize(int size0, int size1, int size2)
	{
		base_array_t::setSize<device_allocator>(sizeof(T), size0, size1, size2);
	}

	void setSize(int ndims, int *shape)
	{
		base_array_t::setSize<device_allocator>(sizeof(T), ndims, shape);
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return base_array_t::raw_ptr<T>();
	}

	const T *raw_ptr() const
	{
		return base_array_t::raw_ptr<T>();
	}

	operator T * ()
	{
		return base_array_t::raw_ptr<T>();
	}

	operator const T * () const
	{
		return base_array_t::raw_ptr<T>();
	}
};


template <typename T>
void host_to_device(device_array_t<T> &dst_d, const array_t<T> &src)
{
	dst_d.setSize(src.ndims(), src.shape());

	cudaMemcpy(dst_d, src, dst_d.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void device_to_host(array_t<T> &dst, const device_array_t<T> &src_d)
{
	dst.setSize(src_d.ndims(), src_d.shape());

	cudaMemcpy(dst, src_d, dst.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

} // namespace numcpp

#endif // NUMCPP_CUDA_H_