#ifndef NUMCPP_CUDA_H_
#define NUMCPP_CUDA_H_

#include "base_array.h"
#include "array.h"

#include <cuda_runtime.h>

namespace np {

template <typename T>
struct GpuArray : public BaseArray
{
public:
	GpuArray() : BaseArray(sizeof(T))
	{
	}

private:
	// External constructors
	template <typename T>
	friend GpuArray<T> gpuArray(const tuple &size);

protected:
	GpuArray(const tuple &size, 
				   std::unique_ptr<int[]> stride, 
				   std::shared_ptr<void> address, 
				   void *origin) : 
		BaseArray(sizeof(T), size, std::move(stride), std::move(address), origin)
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

// TODO: Use 'this constructor' forwarding (with VS2013)
template <typename T>
GpuArray<T> gpuArray(const tuple &size)
{
	const int itemSize = sizeof(T);
	const int ndims = size.length();

	int *stride = new int[ndims];
	stride[0] = itemSize;
	for (int i = 1; i < ndims; i++)
		stride[i] = stride[i - 1] * size[i - 1];

	void *ptr = device_allocator::allocate(size.product() * itemSize);

	return GpuArray<T>(
		size, 
		std::unique_ptr<int[]>(stride), 
		std::shared_ptr<void>(ptr, device_allocator::free), 
		ptr);
}

template <typename T>
GpuArray<T> gpuArray(int size0)
{
	return gpuArray<T>(tuple(size0));
}

template <typename T>
GpuArray<T> gpuArray(int size0, int size1)
{
	return gpuArray<T>(tuple(size0, size1));
}

template <typename T>
void to_device(GpuArray<T> &dst_d, const Array<T> &src)
{
	if (dst_d.size() != src.size())
		dst_d = gpuArray<T>(src.size());

	cudaMemcpy(dst_d, src, dst_d.byteSize(), cudaMemcpyHostToDevice);
}

template <typename T>
void to_device(GpuArray<T> &dst_d, const Array<T> &src, cudaStream_t stream)
{
	if (dst_d.size() != src.size())
		dst_d = gpuArray<T>(src.size());

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