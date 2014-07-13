#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

#include "tuple.h"

#include <memory>
#include <assert.h>

namespace np {

struct heap_allocator
{
	static void *allocate(int size)
	{
		return new char[size];
	}

	static void free(void *ptr)
	{
		delete[] reinterpret_cast<char *>(ptr);
	}
};

struct BaseArray
{
//private:
public:
	const int _itemSize;
	tuple _size;
	std::unique_ptr<int[]> _stride;
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
		_size(size)
	{
		_stride.reset(new int[ndims()]);
		_stride[0] = itemSize;
		for (int i = 1; i < ndims(); i++)
			_stride[i] = _stride[i - 1] * size[i - 1];

		void *ptr = heap_allocator::allocate(size.product() * _itemSize);

		_address = std::shared_ptr<void>(ptr, heap_allocator::free);
		_origin = ptr;
	}

private:
	// External constructors
	friend BaseArray ref(const BaseArray &array);

	template <class Allocator>
	friend BaseArray baseArrayWithAllocator(int itemSize, const tuple &size);

protected:
	BaseArray(int itemSize, 
			  const tuple &size, 
			  std::unique_ptr<int[]> stride, 
			  std::shared_ptr<void> address, 
			  void *origin) : 
		_itemSize(itemSize), 
		_size(size), 
		_stride(std::move(stride)), 
		_address(address), 
		_origin(origin)
	{
	}

private:
	// delete copy constructor, assign
	BaseArray(const BaseArray &other) : _itemSize(1) { }
	const BaseArray &operator=(const BaseArray &other) { return *this; }

public:
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
		_size = std::move(other._size);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._origin = nullptr;

		return *this;
	}

	BaseArray slice(int from, int to)
	{
		assert(from <= to);	

		BaseArray result(itemSize());

		int *shape = new int[1];
		shape[0] = to - from;
		result._size = tuple(1, shape);

		int *stride = new int[1];
		stride[0] = this->stride(0);
		result._stride = std::unique_ptr<int[]>(stride);

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

		int *stride = new int[2];
		stride[0] = this->stride(0);
		stride[1] = this->stride(1);
		result._stride = std::unique_ptr<int[]>(stride);

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from0, from1);

		return result;
	}

	// ## Access to premitive properties

	int itemSize() const
	{
		return _itemSize;
	}

	const tuple &size() const
	{
		return _size;
	}

	int size(int dim) const
	{
		return _size[dim];
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

	// ## Derived property functions

	int ndims() const
	{
		return _size.length();
	}

	bool empty() const
	{
		return raw_ptr() == nullptr || length() == 0;
	}

	int length() const
	{
		return size().product();
	}

	int byteSize() const
	{
		return length() * itemSize();
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

template <class Allocator>
inline BaseArray baseArrayWithAllocator(int itemSize, const tuple &size)
{
	const int ndims = size.length();

	int *stride = new int[ndims];
	stride[0] = itemSize;
	for (int i = 1; i < ndims; i++)
		stride[i] = stride[i - 1] * size[i - 1];

	void *ptr = Allocator::allocate(size.product() * itemSize);

	return BaseArray(
		itemSize, 
		size, 
		std::unique_ptr<int[]>(stride), 
		std::shared_ptr<void>(ptr, Allocator::free), 
		ptr);
}

inline BaseArray ref(const BaseArray &array)
{
	int *stride = new int[array.ndims()];
	for (int i = 0; i < array.ndims(); i++)
		stride[i] = array.stride(i);

	return BaseArray(
		array._itemSize, 
		array._size, 
		std::unique_ptr<int[]>(stride), 
		array._address, 
		array._origin);
}

} // namespace np

#endif // NUMCPP_ABSTRACT_ARRAY_H_