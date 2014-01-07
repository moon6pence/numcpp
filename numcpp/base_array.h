#ifndef NUMCPP_BASE_ARRAY_H_
#define NUMCPP_BASE_ARRAY_H_

#include <memory>

namespace numcpp {

struct base_array_t
{
protected:
	int _ndims;
	int _size;
	std::unique_ptr<int[]> _shape;
	std::shared_ptr<void> _address;
	void *_origin;

	base_array_t() : _ndims(0), _size(0), _origin(nullptr) 
	{ 
	}

protected:
	// move constructor
	base_array_t(base_array_t &&other)
	{
		_ndims = other._ndims;
		_size = other._size;
		_shape = std::move(other._shape);
		_address = std::move(other._address);
		_origin = other._origin;

		other._ndims = 0;
		other._size = 0;
		// other._shape = nullptr; // already moved
		// other._address = nullptr; // already moved
		other._origin = nullptr;
	}

	// move assign
	const base_array_t &operator=(base_array_t &&other)
	{
		_ndims = other._ndims;
		_size = other._size;
		_shape = std::move(other._shape);
		_address = std::move(other._address);
		_origin = other._origin;

		other._ndims = 0;
		other._size = 0;
		// other._shape = nullptr; // already moved
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
		_size = size;
		_shape = std::unique_ptr<int[]>(shape);
		_address = address;
		_origin = origin;
	}

	template <typename T, class Allocator>
	void init(int ndims, int size, int *shape)
	{
		void *ptr = Allocator::allocate(size * sizeof(T));
		auto address = std::shared_ptr<void>(ptr, Allocator::free);

		init(ndims, size, shape, address, ptr);
	}

public:
	template <typename T, class Allocator>
	void setSize(int size0)
	{
		if (this->ndims() == 1 && 
			this->size(0) == size0) return;

		int *shape = new int[1];
		shape[0] = size0;

		init<T, Allocator>(1, size0, shape);
	}

	template <typename T, class Allocator>
	void setSize(int size0, int size1)
	{
		if (this->ndims() == 2 && 
			this->size(0) == size0 && 
			this->size(1) == size1) return;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		init<T, Allocator>(2, size0 * size1, shape);
	}

	template <typename T, class Allocator>
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

		init<T, Allocator>(3, size0 * size1 * size2, shape);
	}

	template <typename T, class Allocator>
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

		init<T, Allocator>(ndims, size, new_shape);
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
		return _shape.get();
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

	template <typename T>
	T *raw_ptr()
	{
		return reinterpret_cast<T *>(_origin);
	}

	template <typename T>
	const T *raw_ptr() const
	{
		return reinterpret_cast<T *>(_origin);
	}

	template <typename T>
	operator T * ()
	{
		return raw_ptr<T>();
	}

	template <typename T>
	operator const T * () const
	{
		return raw_ptr<T>();
	}

	// at(index0, index...) : access array elements

	template <typename T>
	T& at(int index0)
	{
		return raw_ptr<T>()[index0];
	}

	template <typename T>
	T& at(int index0, int index1)
	{
		return raw_ptr<T>()[index1 + _shape[1] * index0];
	}

	template <typename T>
	T& at(int index0, int index1, int index2)
	{
		return raw_ptr<T>()[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	template <typename T>
	const T& at(int index0) const
	{
		return raw_ptr<T>()[index0];
	}

	template <typename T>
	const T& at(int index0, int index1) const
	{
		return raw_ptr<T>()[index1 + _shape[1] * index0];
	}

	template <typename T>
	const T& at(int index0, int index1, int index2) const
	{
		return raw_ptr<T>()[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	template <typename T>
	T& operator() (int index0)
	{
		return at<T>(index0);
	}

	template <typename T>
	T& operator() (int index0, int index1)
	{
		return at<T>(index0, index1);
	}

	template <typename T>
	T& operator() (int index0, int index1, int index2)
	{
		return at<T>(index0, index1, index2);
	}

	template <typename T>
	const T& operator() (int index0) const
	{
		return at<T>(index0);
	}

	template <typename T>
	const T& operator() (int index0, int index1) const
	{
		return at<T>(index0, index1);
	}

	template <typename T>
	const T& operator() (int index0, int index1, int index2) const
	{
		return at<T>(index0, index1, index2);
	}
};

} // namespace numcpp

#endif // NUMCPP_ABSTRACT_ARRAY_H_