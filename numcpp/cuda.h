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
struct GpuArray : public BaseArray
{
public:
	GpuArray() : BaseArray(sizeof(T))
	{
	}

	explicit GpuArray(int size0) : 
		BaseArray(sizeof(T), tuple(size0), device_allocator::allocate, device_allocator::free)
	{
	}

	GpuArray(int size0, int size1) : 
		BaseArray(sizeof(T), tuple(size0, size1), device_allocator::allocate, device_allocator::free)
	{
	}

	GpuArray(int size0, int size1, int size2) : 
		BaseArray(sizeof(T), tuple(size0, size1, size2), device_allocator::allocate, device_allocator::free)
	{
	}

	explicit GpuArray(const tuple &size) : 
		BaseArray(sizeof(T), size, device_allocator::allocate, device_allocator::free)
	{
	}

private:
	// delete copy constructor, assign
	GpuArray(GpuArray &) { }
	const GpuArray &operator=(const GpuArray &) { return *this; }

public:
	// inherits move constructor
	GpuArray(GpuArray &&other) : BaseArray(std::move(other))
	{
	}

	// inherits move assign
	const GpuArray &operator=(GpuArray &&other)
	{
		BaseArray::operator=(std::move(other));
		return *this;
	}

	// Convert from host array
	explicit GpuArray(const Array<T> &array_h) : BaseArray(sizeof(T))
	{
		to_device(*this, array_h);
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return BaseArray::raw_ptr<T>();
	}

	const T *raw_ptr() const
	{
		return BaseArray::raw_ptr<T>();
	}

	operator T * ()
	{
		return BaseArray::raw_ptr<T>();
	}

	operator const T * () const
	{
		return BaseArray::raw_ptr<T>();
	}
};

template <typename T>
void to_device(GpuArray<T> &dst_d, const Array<T> &src)
{
	if (dst_d.size() != src.size())
		dst_d = GpuArray<T>(src.size());

	cudaMemcpy(dst_d, src, dst_d.byteSize(), cudaMemcpyHostToDevice);
}

template <typename T>
void to_device(GpuArray<T> &dst_d, const Array<T> &src, cudaStream_t stream)
{
	if (dst_d.size() != src.size())
		dst_d = GpuArray<T>(src.size());

	cudaMemcpyAsync(dst_d, src, dst_d.byteSize(), cudaMemcpyHostToDevice, stream);
}

template <typename T>
void to_host(Array<T> &dst, const GpuArray<T> &src_d)
{
	if (dst.size() != src_d.size())
		dst = Array<T>(src_d.size());

	cudaMemcpy(dst, src_d, dst.byteSize(), cudaMemcpyDeviceToHost);
}

template <typename T>
void to_host(Array<T> &dst, const GpuArray<T> &src_d, cudaStream_t stream)
{
	if (dst.size() != src_d.size())
		dst = Array<T>(src_d.size());

	cudaMemcpyAsync(dst, src_d, dst.byteSize(), cudaMemcpyDeviceToHost, stream);
}

} // namespace np

#endif // NUMCPP_CUDA_H_