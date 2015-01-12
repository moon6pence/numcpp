#ifndef NUMCPP_GPU_ARRAY_H_
#define NUMCPP_GPU_ARRAY_H_

#include "array.h"
#include <cuda_runtime.h>

#define CUDA_CALL(func) \
	{ \
		cudaError_t error = (func); \
		if (error != cudaSuccess) std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
	}

namespace np {

template <typename T>
struct cuda_allocator
{
	static std::shared_ptr<T> allocate(int size)
	{
		T *ptr = nullptr;
		CUDA_CALL(cudaMalloc<T>(&ptr, size * sizeof(T)));

		return std::shared_ptr<T>(ptr, free);
	}

	static void free(T *ptr)
	{
		CUDA_CALL(cudaFree(ptr));
	}
};

template <typename T, int Dim = 1>
using GpuArray = Array<T, Dim, cuda_allocator<T>>;

template <typename T, int Dim>
void to_device(GpuArray<T, Dim> &dst_d, const Array<T, Dim> &src)
{
	// FIXME: dst_d.setSize()
	CUDA_CALL(cudaMemcpy(dst_d.raw_ptr(), src.raw_ptr(), byteSize(src), cudaMemcpyHostToDevice));
}

template <typename T, int Dim>
void to_host(Array<T, Dim> &dst, const GpuArray<T, Dim> &src_d)
{
	// FIXME: dst.setSize()
	CUDA_CALL(cudaMemcpy(dst.raw_ptr(), src_d.raw_ptr(), byteSize(src_d), cudaMemcpyDeviceToHost));
}

} // namespace np

#endif // NUMCPP_GPU_ARRAY_H_