#ifndef NUMCPP_TARRAY_H_
#define NUMCPP_TARRAY_H_

#include "allocator.h"
#include "array_functions.h"

#include <array>

namespace np {

template <int N> 
inline int product(const std::array<int, N> &array)
{
	// TODO: do not use runtime loop
	int result = 1;

	for (int i = 0; i < N; i++)
		result *= array[i];

	return result;
}

template <int N>
inline std::array<int, N> zero_array()
{
	std::array<int, N> result;

	for (int i = 0; i < N; i++)
		result[i] = 0;

	return std::move(result);
}

// TODO: variadic template
inline std::array<int, 1> make_array(int size0)
{
	std::array<int, 1> result;
	result[0] = size0;
	return result;
}

inline std::array<int, 2> make_array(int size0, int size1)
{
	std::array<int, 2> result;
	result[0] = size0;
	result[1] = size1;
	return result;
}

inline std::array<int, 3> make_array(int size0, int size1, int size2)
{
	std::array<int, 3> result;
	result[0] = size0;
	result[1] = size1;
	result[2] = size2;
	return result;
}

template <int N>
inline std::array<int, N> make_stride(const std::array<int, N> &size)
{
	// TODO do not use runtime loop
	std::array<int, N> stride;

	stride[0] = 1;
	for (int i = 1; i < N; i++)
		stride[i] = stride[i - 1] * size[i - 1];

	return std::move(stride);
}

template <typename T, int Dim = 1>
struct TArray
{
public:
	typedef T value_type;
	typedef std::array<int, Dim> size_type;
	typedef std::array<int, Dim> stride_type;

private:
	int _length;
	size_type _size;
	stride_type _stride;
	std::shared_ptr<value_type> _address;
	value_type *_origin;

public:
	TArray() : 
		_length(0), 
		_size(zero_array<Dim>()), 
		_stride(make_stride(zero_array<Dim>())), 
		_address(), 
		_origin(nullptr)
	{
	}

	TArray(const size_type &size) : 
		_length(product(size)), 
		_size(size), 
		_stride(make_stride(size)), 
		_address(heap_allocator<value_type>::allocate(product(size))), 
		_origin(nullptr)
	{
		_origin = _address.get();
	}

	TArray(int size0) : 
		_length(size0), 
		_size(make_array(size0)), 
		_stride(make_stride(make_array(size0))), 
		_address(heap_allocator<value_type>::allocate(size0)), 
		_origin(nullptr)
	{
		static_assert(Dim == 1, "This function is only for Array<T, 1>");
		_origin = _address.get();
	}

	TArray(int size0, int size1) : 
		_length(size0 * size1), 
		_size(make_array(size0, size1)), 
		_stride(make_stride(make_array(size0, size1))), 
		_address(heap_allocator<value_type>::allocate(size0 * size1)), 
		_origin(nullptr)
	{
		static_assert(Dim == 2, "This function is only for Array<T, 2>");
		_origin = _address.get();
	}

	// copy constructor (shallow copy)
	explicit TArray(const TArray &other) :
		_length(other._length), 
		_size(other._size), 
		_stride(other._stride), 
		_address(other._address), // add refrence count here
		_origin(other._origin)
	{
	}

	// copy assign (shallow copy)
	const TArray &operator=(const TArray &other) 
	{ 
		_length = other._length;
		_size = other._size;
		_stride = other._stride;
		_address = other._address; // add refrence count here
		_origin = other._origin;

		return *this; 
	}

	// move constructor
	TArray(TArray &&other) :
		_length(other._length), 
		_size(std::move(other._size)), 
		_stride(std::move(other._stride)), 
		_address(std::move(other._address)), 
		_origin(other._origin)
	{
		other._length = 0;
		other._origin = nullptr;
	}

	// move assign
	const TArray &operator=(TArray &&other)
	{
		_length = other._length;
		_size = std::move(other._size);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._length = 0;
		other._origin = nullptr;

		return *this;
	}

	int itemSize() const
	{
		return sizeof(value_type);
	}

	int length() const
	{
		return _length;
	}

	int ndims() const
	{
		return Dim;
	}

	const size_type &size() const
	{
		return _size;
	}

	template <int N>
	int size() const
	{
		return _size[N];
	}

	const stride_type &strides() const
	{
		return _stride;
	}
	
	template <int N>
	int stride() const
	{
		return _stride[N];
	}

	value_type *raw_ptr()
	{
		return _origin;
	}

	const value_type *raw_ptr() const
	{
		return _origin;
	}

	// Accessing elements

	T &at(int index0) 
	{ 
		return _origin[index0 * stride<0>()]; 
	}

	const T &at(int index0) const 
	{ 
		return _origin[index0 * stride<0>()]; 
	}

	T &operator()(int index0) { return at(index0); }
	const T &operator()(int index0) const { return at(index0); }
	T &operator[](int index0) { return at(index0); }
	const T &operator[](int index0) const { return at(index0); }

	T &at(int index0, int index1)
	{
		static_assert(Dim >= 2, "Array dimension bounds error");
		return _origin[index0 * stride<0>() + index1 * stride<1>()];
	}

	const T &at(int index0, int index1) const
	{
		static_assert(Dim >= 2, "Array dimension bounds error");
		return _origin[index0 * stride<0>() + index1 * stride<1>()];
	}

	T &operator()(int index0, int index1) { return at(index0, index1); }
	const T &operator()(int index0, int index1) const { return at(index0, index1); }
};

} // namespace np

#endif // NUMCPP_TARRAY_H_