#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

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

struct base_array_t
{
private:
	const int _itemSize;
	int _ndims;
	std::unique_ptr<int[]> _shape;
	std::unique_ptr<int[]> _stride;
	std::shared_ptr<void> _address;
	void *_origin;

public:
	base_array_t() : 
		_itemSize(1), _ndims(0), _origin(nullptr)
	{
	}

	base_array_t(int itemSize) : 
		_itemSize(itemSize), _ndims(0), _origin(nullptr) 
	{ 
	}

	// copy constructor
	base_array_t(const base_array_t &other) : 
		_itemSize(other._itemSize)
	{
		_ndims = other._ndims;

		_shape.reset(new int[_ndims]);
		std::copy(other._shape.get(), other._shape.get() + _ndims, _shape.get());

		_stride.reset(new int[_ndims]);
		std::copy(other._stride.get(), other._stride.get() + _ndims, _stride.get());

		_address = other._address; // add reference

		_origin = other._origin;
	}

	// copy assign
	const base_array_t &operator=(const base_array_t &other)
	{
		// TODO: assign _itemSize

		_ndims = other._ndims;

		_shape.reset(new int[_ndims]);
		std::copy(other._shape.get(), other._shape.get() + _ndims, _shape.get());

		_stride.reset(new int[_ndims]);
		std::copy(other._stride.get(), other._stride.get() + _ndims, _stride.get());

		_address = other._address; // add reference

		_origin = other._origin;

		return *this;
	}

	// move constructor
	base_array_t(base_array_t &&other) : 
		_itemSize(other._itemSize)
	{
		_ndims = other._ndims;
		_shape = std::move(other._shape);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._ndims = 0;
		// other._shape = nullptr; // already moved
		// other._stride = nullptr; // already moved
		// other._address = nullptr; // already moved
		other._origin = nullptr;
	}

	// move assign
	const base_array_t &operator=(base_array_t &&other)
	{
		_ndims = other._ndims;
		_shape = std::move(other._shape);
		_stride = std::move(other._stride);
		_address = std::move(other._address);
		_origin = other._origin;

		other._ndims = 0;
		// other._shape = nullptr; // already moved
		// other._stride = nullptr; // already moved
		// other._address = nullptr; // already moved
		other._origin = nullptr;

		return *this;
	}

private:
	void init(
		int ndims, int size, int *shape, 
		std::shared_ptr<void> address, void *origin)
	{
		_ndims = ndims;
		_shape = std::unique_ptr<int[]>(shape);
		_address = address;
		_origin = origin;

		// Initialize stride from shape
		int *stride = new int[_ndims];

		stride[0] = itemSize();
		for (int i = 1; i < _ndims; i++)
			stride[i] = stride[i - 1] * shape[i - 1];

		_stride = std::unique_ptr<int[]>(stride);
	}

	template <class Allocator>
	void init(int ndims, int size, int *shape)
	{
		void *ptr = Allocator::allocate(size * _itemSize);
		auto address = std::shared_ptr<void>(ptr, Allocator::free);

		init(ndims, size, shape, address, ptr);
	}

public:
	template <class Allocator>
	void setSize(int size0)
	{
		if (this->ndims() == 1 && 
			this->size(0) == size0) return;

		int *shape = new int[1];
		shape[0] = size0;

		init<Allocator>(1, size0, shape);
	}

	template <class Allocator>
	void setSize(int size0, int size1)
	{
		if (this->ndims() == 2 && 
			this->size(0) == size0 && 
			this->size(1) == size1) return;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		init<Allocator>(2, size0 * size1, shape);
	}

	template <class Allocator>
	void setSize(int size0, int size1, int size2)
	{
		if (this->ndims() == 3 && 
			this->size(0) == size0 && 
			this->size(1) == size1 && 
			this->size(2) == size2) return;

		int *shape = new int[3];
		shape[0] = size0;
		shape[1] = size1;
		shape[2] = size2;

		init<Allocator>(3, size0 * size1 * size2, shape);
	}

	template <class Allocator>
	void setSize(int ndims, int *shape)
	{
		if (this->ndims() == ndims)
		{
			for (int i = 0; i < ndims; i++)
				if (shape[i] != this->size(i))
					goto allocate;

			return;
		}

allocate:
		int size = 1;
		for (int i = 0; i < ndims; i++)
			size *= shape[i];

		int *new_shape = new int[ndims];
		for (int i = 0; i < ndims; i++)
			new_shape[i] = shape[i];

		init<Allocator>(ndims, size, new_shape);
	}

	base_array_t slice(int from, int to)
	{
		assert(from <= to);	

		base_array_t result(itemSize());

		result._ndims = this->_ndims;

		int *shape = new int[1];
		shape[0] = to - from;
		result._shape = std::unique_ptr<int[]>(shape);

		int *stride = new int[1];
		stride[0] = this->stride(0);
		result._stride = std::unique_ptr<int[]>(stride);

		// add reference count here
		result._address = this->_address;

		// new origin with offset
		result._origin = this->ptr_at(from);

		return result;
	}

	base_array_t slice(int from0, int from1, int to0, int to1)
	{
		assert(from0 <= to0);	
		assert(from1 <= to1);	

		base_array_t result(itemSize());

		result._ndims = this->_ndims;

		int *shape = new int[2];
		shape[0] = to0 - from0;
		shape[1] = to1 - from1;
		result._shape = std::unique_ptr<int[]>(shape);

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

	int ndims() const
	{
		return _ndims;
	}

	int *shape() const
	{
		return _shape.get();
	}

	int shape(int dim) const
	{
		return _shape[dim];
	}

	int size(int dim) const
	{
		return _shape[dim];
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

	bool empty() const
	{
		return raw_ptr() == nullptr || size() == 0;
	}

	int size() const
	{
		int result = 1;
		for (int i = 0; i < ndims(); i++)
			result *= size(i);

		return result;
	}

	int byteSize() const
	{
		return size() * itemSize();
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

	void *ptr_at(int index0, int index1, int index2)
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1) + index2 * stride(2);
	}

	const void *ptr_at(int index0) const
	{
		return raw_ptr<char>() + index0 * stride(0);
	}

	const void *ptr_at(int index0, int index1) const
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1);
	}

	const void *ptr_at(int index0, int index1, int index2) const
	{
		return raw_ptr<char>() + index0 * stride(0) + index1 * stride(1) + index2 * stride(2);
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
	T& at(int index0, int index1, int index2)
	{
		return *static_cast<T *>(ptr_at(index0, index1, index2));
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

	template <typename T>
	const T& at(int index0, int index1, int index2) const
	{
		return *static_cast<const T *>(ptr_at(index0, index1, index2));
	}
};

} // namespace np

#endif // NUMCPP_ABSTRACT_ARRAY_H_