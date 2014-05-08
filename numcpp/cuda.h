#ifndef NUMCPP_CUDA_H_
#define NUMCPP_CUDA_H_

#include "base_array.h"
#include "array.h"

#include <cuda_runtime.h>

namespace np {

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
	device_array_t() : base_array_t(sizeof(T))
	{
	}

	explicit device_array_t(int size0) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<device_allocator>(tuple(size0));
	}

	device_array_t(int size0, int size1) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<device_allocator>(tuple(size0, size1));
	}

	device_array_t(int size0, int size1, int size2) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<device_allocator>(tuple(size0, size1, size2));
	}

	explicit device_array_t(const tuple &size) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<device_allocator>(size);
	}

private:
	// delete copy constructor, assign
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
	explicit device_array_t(const array_t<T> &array_h) : base_array_t(sizeof(T))
	{
		to_device(*this, array_h);
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
void to_device(device_array_t<T> &dst_d, const array_t<T> &src)
{
	if (dst_d.size() != src.size())
		dst_d = device_array_t<T>(src.size());

	cudaMemcpy(dst_d, src, dst_d.byteSize(), cudaMemcpyHostToDevice);
}

template <typename T>
void to_device(device_array_t<T> &dst_d, const array_t<T> &src, cudaStream_t stream)
{
	if (dst_d.size() != src.size())
		dst_d = device_array_t<T>(src.size());

	cudaMemcpyAsync(dst_d, src, dst_d.byteSize(), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void to_host(array_t<T> &dst, const device_array_t<T> &src_d)
{
	if (dst.size() != src_d.size())
		dst = array_t<T>(src_d.size());

	cudaMemcpy(dst, src_d, dst.byteSize(), cudaMemcpyDeviceToHost);
}

template <typename T>
void to_host(array_t<T> &dst, const device_array_t<T> &src_d, cudaStream_t stream)
{
	if (dst.size() != src_d.size())
		dst = array_t<T>(src_d.size());

	cudaMemcpyAsync(dst, src_d, dst.byteSize(), cudaMemcpyDeviceToHost, stream);
}

} // namespace np

#endif // NUMCPP_CUDA_H_