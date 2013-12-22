#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

namespace numcpp {

template <typename T>
struct array_t
{
	array_t() : 
		_ndims(1), _size(0), _shape(nullptr), _ptr(nullptr)
	{
		_shape = new int[1];
		_shape[0] = 0;
	}

	array_t(int size0) : 
		_ndims(1), _size(size0), _shape(nullptr), _ptr(nullptr)
	{
		_shape = new int[1];
		_shape[0] = size0;

		_ptr = new T[_size];
	}

	array_t(int size0, int size1) : 
		_ndims(2), _size(size0 * size1), _shape(nullptr), _ptr(nullptr)
	{
		_shape = new int[2];
		_shape[0] = size0;
		_shape[1] = size1;

		_ptr = new T[_size];	
	}

	array_t(int size0, int size1, int size2) : 
		_ndims(3), _size(size0 * size1 * size2), _shape(nullptr), _ptr(nullptr)
	{
		_shape = new int[3];
		_shape[0] = size0;
		_shape[1] = size1;
		_shape[2] = size2;

		_ptr = new T[_size];	
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
		return _ptr;
	}

	const T *raw_ptr() const
	{
		return _ptr;
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
		return _ptr[index0];
	}

	T& at(int index0, int index1)
	{
		return _ptr[index1 + _shape[1] * index0];
	}

	T& at(int index0, int index1, int index2)
	{
		return _ptr[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	const T& at(int index0) const
	{
		return _ptr[index0];
	}

	const T& at(int index0, int index1) const
	{
		return _ptr[index1 + _shape[1] * index0];
	}

	const T& at(int index0, int index1, int index2) const
	{
		return _ptr[index2 + _shape[2] * (index1 + _shape[1] * index0)];
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

private:
	int _ndims;
	int _size;
	int *_shape;
	T *_ptr;
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_