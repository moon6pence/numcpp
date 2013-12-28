#ifndef NUMCPP_DEVICE_ARRAY_H_
#define NUMCPP_DEVICE_ARRAY_H_

#include <cuda_runtime.h>
#include <memory>

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
struct device_array_t
{
private:
	int _ndims;
	int _size;
	int *_shape;
	std::shared_ptr<T> _ptr;

public:
	device_array_t() : 
		_ndims(1), _size(0), _shape(nullptr)
	{
		_shape = new int[1];
		_shape[0] = 0;
	}

	device_array_t(int size0) : 
		_ndims(1), _size(size0), _shape(nullptr)
	{
		_shape = new int[1];
		_shape[0] = size0;

		_ptr = std::shared_ptr<T>(
			device_array_allocator<T>(_size), device_array_deallocator<T>);
	}

	device_array_t(int size0, int size1) : 
		_ndims(2), _size(size0 * size1), _shape(nullptr)
	{
		_shape = new int[2];
		_shape[0] = size0;
		_shape[1] = size1;

		_ptr = std::shared_ptr<T>(
			device_array_allocator<T>(_size), device_array_deallocator<T>);
	}

	device_array_t(int size0, int size1, int size2) : 
		_ndims(3), _size(size0 * size1 * size2), _shape(nullptr)
	{
		_shape = new int[3];
		_shape[0] = size0;
		_shape[1] = size1;
		_shape[2] = size2;

		_ptr = std::shared_ptr<T>(
			device_array_allocator<T>(_size), device_array_deallocator<T>);
	}

	~device_array_t()
	{
		if (_shape) { delete[] _shape; _shape = nullptr; }
		_ptr = nullptr;
	}

private:
	// disable copy constructor, assign
	device_array_t(device_array_t &) { }
	const device_array_t &operator=(const device_array_t &) { return *this; }

public:
	// move constructor
	device_array_t(device_array_t &&other) :
		_ndims(other._ndims), 
		_size(other._size), 
		_shape(other._shape), 
		_ptr(std::move(other._ptr))
	{
		other._ndims = 1;
		other._size = 0;
		other._shape = new int[1];
		other._shape[0] = 0;
		other._ptr = nullptr;
	}

	// move assign
	const device_array_t &operator=(device_array_t &&other)
	{
		if (_shape) { delete _shape; _shape = nullptr; }
		_ptr = nullptr;

		_ndims = other._ndims;
		_size = other._size;
		_shape = other._shape;
		_ptr = std::move(other._ptr);

		other._ndims = 1;
		other._size = 0;
		other._shape = new int[1];
		other._shape[0] = 0;
		other._ptr = nullptr;

		return *this;
	}

	bool empty() const
	{
		return _size == 0;
	}

	int ndims() const
	{
		return _ndims;
	}

	int size() const
	{
		return _size;
	}

	int size(int dim) const
	{
		return _shape[dim];
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return _ptr.get();
	}

	const T *raw_ptr() const
	{
		return _ptr.get();
	}

	operator T * ()
	{
		return raw_ptr();
	}

	operator const T * () const
	{
		return raw_ptr();
	}

	// at(index0, index...) : access array elements

	T& at(int index0)
	{
		return raw_ptr()[index0];
	}

	T& at(int index0, int index1)
	{
		return raw_ptr()[index1 + _shape[1] * index0];
	}

	T& at(int index0, int index1, int index2)
	{
		return raw_ptr()[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	const T& at(int index0) const
	{
		return raw_ptr()[index0];
	}

	const T& at(int index0, int index1) const
	{
		return raw_ptr()[index1 + _shape[1] * index0];
	}

	const T& at(int index0, int index1, int index2) const
	{
		return raw_ptr()[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	T& operator() (int index0)
	{
		return at(index0);
	}

	T& operator() (int index0, int index1)
	{
		return at(index0, index1);
	}

	T& operator() (int index0, int index1, int index2)
	{
		return at(index0, index1, index2);
	}

	const T& operator() (int index0) const
	{
		return at(index0);
	}

	const T& operator() (int index0, int index1) const
	{
		return at(index0, index1);
	}

	const T& operator() (int index0, int index1, int index2) const
	{
		return at(index0, index1, index2);
	}
};

} // namespace numcpp

#endif // NUMCPP_DEVICE_ARRAY_H_