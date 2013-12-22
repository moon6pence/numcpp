#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

namespace numcpp {

template <typename T>
struct array_t
{
	array_t() : 
		_ndims(1), _size(0), _ptr(nullptr)
	{
	}

	array_t(int size) : 
		_ndims(1), _size(size), _ptr(nullptr)
	{
		_ptr = new T[_size];
	}

	array_t(int size0, int size1) : 
		_ndims(2), _size(size0 * size1), _ptr(nullptr)
	{
		_ptr = new T[_size];	
	}

	array_t(int size0, int size1, int size2) : 
		_ndims(3), _size(size0 * size1 * size2), _ptr(nullptr)
	{
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

	T *raw_ptr()
	{
		return _ptr;
	}

	operator T * ()
	{
		return raw_ptr();
	}

	T& at(int index)
	{
		return _ptr[index];
	}

	T& operator() (int index)
	{
		return at(index);
	}

private:
	int _ndims;
	int _size;
	T *_ptr;
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_