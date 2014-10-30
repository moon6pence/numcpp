#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

#include "allocator.h"
#include <vector>

namespace np {

inline int product(const std::vector<int> &size)
{
	int result = 1;

	for (int size0 : size)
		result *= size0;

	return result;
}

inline std::vector<int> make_vector(int size0)
{
	std::vector<int> result(1);
	result[0] = size0;
	return std::move(result);
}

inline std::vector<int> make_vector(int size0, int size1)
{
	std::vector<int> result(2);
	result[0] = size0;
	result[1] = size1;
	return std::move(result);
}

inline std::vector<int> make_vector(int size0, int size1, int size2)
{
	std::vector<int> result(3);
	result[0] = size0;
	result[1] = size1;
	result[2] = size2;
	return std::move(result);
}

inline std::vector<std::ptrdiff_t> make_stride(const std::vector<int> &size)
{
	std::vector<std::ptrdiff_t> stride(size.size());
	stride[0] = 1;
	for (int i = 1; i < stride.size(); i++)
		stride[i] = stride[i - 1] * size[i - 1];

	return std::move(stride);
}

struct BaseArray
{
public:
	typedef std::vector<int> size_type;
	typedef std::vector<std::ptrdiff_t> stride_type;

private:
	const int _itemSize;
	int _length;
	size_type _size;
	stride_type _stride;
	std::shared_ptr<void> _address;
	void *_origin;

public:
	BaseArray() : 
		_itemSize(1), _length(0), _origin(nullptr)
	{
	}

	explicit BaseArray(int itemSize) : 
		_itemSize(itemSize), _length(0), _origin(nullptr) 
	{ 
	}

	BaseArray(int itemSize, const size_type &size) :
		_itemSize(itemSize), 
		_length(product(size)), 
		_size(size), 
		_stride(make_stride(size)), 
		_address(heap_allocator<char>::allocate(product(size) * itemSize)), // in byte size
		_origin(nullptr)
	{
		_origin = _address.get();
	}

	// TODO: deprecate this
	BaseArray(int itemSize, const size_type &size, void *(*allocate)(int), void (*free)(void *)) :
		_itemSize(itemSize), 
		_length(product(size)), 
		_size(size), 
		_stride(make_stride(size)), 
		_address(), 
		_origin(nullptr)
	{
		void *ptr = allocate(product(size) * itemSize);

		_address = std::shared_ptr<void>(ptr, free);
		_origin = ptr;
	}

	// copy constructor: shallow copy
	explicit BaseArray(const BaseArray &other) :
		_itemSize(other._itemSize), 
		_length(other._length), 
		_size(other._size), 
		_stride(other._stride), 
		_address(other._address), 
		_origin(other._origin)
	{
	}

	// copy assign
	const BaseArray &operator=(const BaseArray &other) 
	{ 
		(int &)_itemSize = other._itemSize;
		_length = other._length;
		_size = other._size;
		_stride = other._stride;
		_address = other._address; // add refrence count here
		_origin = other._origin;

		return *this; 
	}

	// move constructor
	BaseArray(BaseArray &&other) : 
		_itemSize(other._itemSize), 
		_length(other._length), 
		_size(std::move(other._size)), 
		_stride(std::move(other._stride)), 
		_address(other._address), 
		_origin(other._origin)
	{
		other._origin = nullptr;
		other._length = 0;
	}

	// move assign
	const BaseArray &operator=(BaseArray &&other)
	{
		(int &)_itemSize = other._itemSize;
		_length = other._length;
		_size = std::move(other._size);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._origin = nullptr;
		other._length = 0;

		return *this;
	}

	int itemSize() const
	{
		return _itemSize;
	}

	int length() const
	{
		return _length;
	}

	int ndims() const
	{
		return (int)_size.size();
	}

	const size_type &size() const
	{
		return _size;
	}

	int size(int dim) const
	{
		return _size[dim];
	}

	std::ptrdiff_t stride(int dim) const
	{
		return _stride[dim];
	}

	void *raw_ptr()
	{
		return _origin;
	}

	const void *raw_ptr() const
	{
		return _origin;
	}

	// FIXME: remove this
	template <typename T>
	friend struct Array;
};

template <class Array>
int byteSize(const Array &array)
{
	return array.length() * array.itemSize();
}

template <class Array>
bool empty(const Array &array)
{
	return array.raw_ptr() == nullptr || array.length() == 0;
}

} // namespace np

#endif // NUMCPP_ABSTRACT_ARRAY_H_