#ifndef __DEVICE_ARRAY_H__
#define __DEVICE_ARRAY_H__

#include "array.h"
#include <cuda_runtime.h>

namespace numcpp {

template <typename T>
void device_array_deleter(T *p)
{
	cudaFree(p);
}

/** Allocate device array */
template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> device_array(Shape... shape)
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

	return array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

} // namespace numcpp

#endif // __DEVICE_ARRAY_H__