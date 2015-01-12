#ifndef NUMCPP_GPU_ARRAY_H_
#define NUMCPP_GPU_ARRAY_H_

#include "array.h"
#include <cuda_runtime.h>

namespace np {

template <typename T>
struct cuda_allocator
{
	static std::shared_ptr<T> allocate(int size)
	{
		T *ptr = nullptr;
		cudaMalloc<T>(&ptr, size);

		return std::shared_ptr<T>(ptr, free);
	}

	static void free(T *ptr)
	{
		cudaFree(ptr);
	}
};

template <typename T, int Dim = 1>
using GpuArray = Array<T, Dim, cuda_allocator<T>>;

} // namespace np

#endif // NUMCPP_GPU_ARRAY_H_