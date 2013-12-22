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

	bool empty()
	{
		return _size == 0;
	}

	int ndims()
	{
		return _ndims;
	}

	int size()
	{
		return _size;
	}

	int shape(int dim)
	{
		return _shape[dim];
	}

	T *raw_ptr()
	{
		return _ptr;
	}

	operator T * ()
	{
		return raw_ptr();
	}

	T& at(int index0)
	{
		return _ptr[index0];
	}

	T& operator() (int index0)
	{
		return at(index0);
	}

	T& at(int index0, int index1)
	{
		return _ptr[0];
	}

private:
	int _ndims;
	int _size;
	int *_shape;
	T *_ptr;
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_