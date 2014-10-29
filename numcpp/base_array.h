#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

#include "allocator.h"
#include "tuple.h"

#include <memory>
#include <assert.h>

namespace np {

inline tuple build_stride(int itemSize, const tuple &size)
{
	// FIXME: similar()?
	tuple stride(size);
	stride[0] = itemSize;
	for (int i = 1; i < stride.length(); i++)
		stride[i] = stride[i - 1] * size[i - 1];

	return std::move(stride);
}

struct BaseArray
{
public:
	const int _itemSize;
	tuple _size, _stride;
	std::shared_ptr<void> _address;
	void *_origin;

public:
	BaseArray() : 
		_itemSize(1), _origin(nullptr)
	{
	}

	explicit BaseArray(int itemSize) : 
		_itemSize(itemSize), _origin(nullptr) 
	{ 
	}

	BaseArray(int itemSize, const tuple &size) :
		_itemSize(itemSize), 
		_size(size), 
		_stride(build_stride(itemSize, size)), 
		_address(heap_allocator<char>::allocate(_size.product() * itemSize)), // in byte size
		_origin(nullptr)
	{
		_origin = _address.get();
	}

	// TODO: deprecate this
	BaseArray(int itemSize, const tuple &size, void *(*allocate)(int), void (*free)(void *)) :
		_itemSize(itemSize), 
		_size(size), 
		_stride(build_stride(itemSize, size))
	{
		void *ptr = allocate(size.product() * _itemSize);

		_address = std::shared_ptr<void>(ptr, free);
		_origin = ptr;
	}

	// copy constructor: shallow copy
	explicit BaseArray(const BaseArray &other) :
		_itemSize(other._itemSize), 
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
		_size = other._size;
		_stride = other._stride;
		_address = other._address; // add refrence count here
		_origin = other._origin;

		return *this; 
	}

	// move constructor
	BaseArray(BaseArray &&other) : 
		_itemSize(other._itemSize), 
		_size(std::move(other._size)), 
		_stride(std::move(other._stride)), 
		_address(other._address), 
		_origin(other._origin)
	{
		other._origin = nullptr;
	}

	// move assign
	const BaseArray &operator=(BaseArray &&other)
	{
		(int &)_itemSize = other._itemSize;
		_size = std::move(other._size);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._origin = nullptr;

		return *this;
	}

	int itemSize() const
	{
		return _itemSize;
	}

	int ndims() const
	{
		return _size.length();
	}

	const tuple &size() const
	{
		return _size;
	}

	int size(int dim) const
	{
		return _size[dim];
	}

	int length() const
	{
		return size().product();
	}

	int byteSize() const
	{
		return length() * itemSize();
	}

	int stride(int dim) const
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

	bool empty() const
	{
		return raw_ptr() == nullptr;
	}




	// # deprecate the functions below

	BaseArray slice(int from, int to)
	{
		assert(from <= to);	

		BaseArray result(itemSize());

		int *shape = new int[1];
		shape[0] = to - from;
		result._size = tuple(1, shape);

		result._stride = this->_stride;

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from);

		return result;
	}

	BaseArray slice(int from0, int from1, int to0, int to1)
	{
		assert(from0 <= to0);	
		assert(from1 <= to1);	

		BaseArray result(itemSize());

		int *shape = new int[2];
		shape[0] = to0 - from0;
		shape[1] = to1 - from1;
		result._size = tuple(2, shape);

		result._stride = this->_stride;

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from0, from1);

		return result;
	}

	template <typename T>
	T *raw_ptr()
	{
		return static_cast<T *>(raw_ptr());
	}

	template <typename T>
	const T *raw_ptr() const
	{
		return static_cast<const T *>(raw_ptr());
	}

	void *ptr_at(int index0)
	{
		return raw_ptr<char>() + index0 * stride(0);
	}

	void *ptr_at(int index0, int index1)
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1);
	}

	const void *ptr_at(int index0) const
	{
		return raw_ptr<char>() + index0 * stride(0);
	}

	const void *ptr_at(int index0, int index1) const
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1);
	}

	template <typename T>
	T& at(int index0)
	{
		return *static_cast<T *>(ptr_at(index0));
	}

	template <typename T>
	T& at(int index0, int index1)
	{
		return *static_cast<T *>(ptr_at(index0, index1));
	}

	template <typename T>
	const T& at(int index0) const
	{
		return *static_cast<const T *>(ptr_at(index0));
	}

	template <typename T>
	const T& at(int index0, int index1) const
	{
		return *static_cast<const T *>(ptr_at(index0, index1));
	}
};

} // namespace np

#endif // NUMCPP_ABSTRACT_ARRAY_H_