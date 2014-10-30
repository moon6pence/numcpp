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

inline std::vector<std::ptrdiff_t> make_stride(int itemSize, const std::vector<int> &size)
{
	std::vector<std::ptrdiff_t> stride(size.size());
	stride[0] = itemSize;
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
	size_type _size;
	stride_type _stride;
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

	BaseArray(int itemSize, const size_type &size) :
		_itemSize(itemSize), 
		_size(size), 
		_stride(make_stride(itemSize, size)), 
		_address(heap_allocator<char>::allocate(product(size) * itemSize)), // in byte size
		_origin(nullptr)
	{
		_origin = _address.get();
	}

	// TODO: deprecate this
	BaseArray(int itemSize, const size_type &size, void *(*allocate)(int), void (*free)(void *)) :
		_itemSize(itemSize), 
		_size(size), 
		_stride(make_stride(itemSize, size))
	{
		void *ptr = allocate(product(size) * itemSize);

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

	int length() const
	{
		return product(size());
	}

	int byteSize() const
	{
		return length() * itemSize();
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

	bool empty() const
	{
		return raw_ptr() == nullptr;
	}




	// # deprecate the functions below

	BaseArray slice(int from, int to)
	{
		//assert(from <= to);	

		BaseArray result(itemSize());

		result._size = make_vector(to - from);
		result._stride = this->_stride;

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from);

		return result;
	}

	BaseArray slice(int from0, int from1, int to0, int to1)
	{
		//assert(from0 <= to0);	
		//assert(from1 <= to1);	

		BaseArray result(itemSize());

		result._size = make_vector(to0 - from0, to1 - from1);
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