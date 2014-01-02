#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

#include <memory>

namespace numcpp {

template <typename T, class Allocator>
struct base_array_t
{
protected:
	int _ndims;
	int _size;
	int *_shape;
	std::shared_ptr<void> _address;
	T *_origin;

	base_array_t() : _ndims(0), _size(0), _shape(nullptr), _origin(nullptr) 
	{ 
	}

private:
	void init()
	{
		int *shape = new int[1];
		shape[0] = 0;

		init(1, 0, shape, nullptr);
	}

	void init(int ndims, int size, int *shape, std::shared_ptr<void> address)
	{
		init(ndims, size, shape, address, (T *)address.get());
	}

	void init(
		int ndims, int size, int *shape, 
		std::shared_ptr<void> address, T *origin)
	{
		_ndims = ndims;
		_size = size;
		_shape = shape;
		_address = address;
		_origin = origin;
	}

protected:
	void free()
	{
		if (_shape) { delete[] _shape; _shape = nullptr; }
		_address = nullptr;
	}

	// move constructor
	base_array_t(base_array_t &&other)
	{
		this->init(other._ndims, other._size, 
			other._shape, std::move(other._address));

		other.init();
	}

	// move assign
	const base_array_t &operator=(base_array_t &&other)
	{
		free();

		this->init(other._ndims, other._size, 
			other._shape, std::move(other._address));

		other.init();

		return *this;
	}

public:
	void setEmpty()
	{
		init();
	}

	void setSize(int size0)
	{
		if (this->ndims() == 1 && 
			this->size(0) == size0) return;

		this->free();

		int size = size0;

		int *shape = new int[1];
		shape[0] = size0;

		auto ptr = std::shared_ptr<void>(
			Allocator::allocate(size), Allocator::free);

		init(1, size, shape, ptr);
	}

	void setSize(int size0, int size1)
	{
		if (this->ndims() == 2 && 
			this->size(0) == size0 && 
			this->size(1) == size1) return;

		this->free();

		int size = size0 * size1;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		auto ptr = std::shared_ptr<void>(
			Allocator::allocate(size), Allocator::free);

		init(2, size, shape, ptr);
	}

	void setSize(int size0, int size1, int size2)
	{
		if (this->ndims() == 3 && 
			this->size(0) == size0 && 
			this->size(1) == size1 && 
			this->size(2) == size2) return;

		this->free();

		int size = size0 * size1 * size2;

		int *shape = new int[3];
		shape[0] = size0;
		shape[1] = size1;
		shape[2] = size2;

		auto ptr = std::shared_ptr<void>(
			Allocator::allocate(size), Allocator::free);

		init(3, size, shape, ptr);
	}

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
		this->free();

		int size = 1;
		for (int i = 0; i < ndims; i++)
			size *= shape[i];

		int *new_shape = new int[ndims];
		for (int i = 0; i < ndims; i++)
			new_shape[i] = shape[i];

		auto ptr = std::shared_ptr<void>(
			Allocator::allocate(size), Allocator::free);

		this->init(ndims, size, new_shape, ptr);
	}

public:
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

	int *shape() const
	{
		return _shape;
	}

	int width() const
	{
		return size(ndims() - 1);
	}

	int height() const
	{
		return size(ndims() - 2);
	}

	int depth() const
	{
		return size(ndims() - 3);
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return _origin;
	}

	const T *raw_ptr() const
	{
		return _origin;
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
		return _origin[index0];
	}

	T& at(int index0, int index1)
	{
		return _origin[index1 + _shape[1] * index0];
	}

	T& at(int index0, int index1, int index2)
	{
		return _origin[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	const T& at(int index0) const
	{
		return _origin[index0];
	}

	const T& at(int index0, int index1) const
	{
		return _origin[index1 + _shape[1] * index0];
	}

	const T& at(int index0, int index1, int index2) const
	{
		return _origin[index2 + _shape[2] * (index1 + _shape[1] * index0)];
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

#endif // NUMCPP_ABSTRACT_ARRAY_H_