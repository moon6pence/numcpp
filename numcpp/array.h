#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

namespace numcpp {

template <typename T>
struct array_t
{
	array_t() : _size(0), _ptr(nullptr)
	{
	}

	array_t(int size) : _size(size), _ptr(nullptr)
	{
		_ptr = new T[_size];
	}

	bool empty()
	{
		return _size == 0;
	}

	int ndims()
	{
		return 1;
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
	int _size;
	T *_ptr;
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_