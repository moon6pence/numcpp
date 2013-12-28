#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include <memory>

namespace numcpp {

template <typename T>
struct abstract_array_t
{
protected:
	int _ndims;
	int _size;
	int *_shape;
	std::shared_ptr<T> _ptr;

	void init()
	{
		_ndims = 1;
		_size = 0;
		_shape = new int[1];
		_shape[0] = 0;
		_ptr = nullptr;	
	}

	void init(int ndims, int size, int *shape, std::shared_ptr<T> ptr)
	{
		_ndims = ndims;
		_size = size;
		_shape = shape;
		_ptr = ptr;
	}

	void free()
	{
		if (_shape) { delete[] _shape; _shape = nullptr; }
		_ptr = nullptr;
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

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return _ptr.get();
	}

	const T *raw_ptr() const
	{
		return _ptr.get();
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
		return raw_ptr()[index0];
	}

	T& at(int index0, int index1)
	{
		return raw_ptr()[index1 + _shape[1] * index0];
	}

	T& at(int index0, int index1, int index2)
	{
		return raw_ptr()[index2 + _shape[2] * (index1 + _shape[1] * index0)];
	}

	const T& at(int index0) const
	{
		return raw_ptr()[index0];
	}

	const T& at(int index0, int index1) const
	{
		return raw_ptr()[index1 + _shape[1] * index0];
	}

	const T& at(int index0, int index1, int index2) const
	{
		return raw_ptr()[index2 + _shape[2] * (index1 + _shape[1] * index0)];
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

template <typename T>
void array_deleter(T *ptr)
{
	delete[] ptr;
}

template <typename T>
struct array_t : public abstract_array_t<T>
{
public:
	array_t()
	{
		abstract_array_t<T>::init();
	}

	array_t(int size0)
	{
		int size = size0;

		int *shape = new int[1];
		shape[0] = size0;

		auto ptr = std::shared_ptr<T>(new T[size], array_deleter<T>);

		abstract_array_t<T>::init(1, size, shape, ptr);
	}

	array_t(int size0, int size1)
	{
		int size = size0 * size1;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		auto ptr = std::shared_ptr<T>(new T[size], array_deleter<T>);	

		abstract_array_t<T>::init(2, size, shape, ptr);
	}

	array_t(int size0, int size1, int size2)
	{
		int size = size0 * size1 * size2;

		int *shape = new int[3];
		shape[0] = size0;
		shape[1] = size1;
		shape[2] = size2;

		auto ptr = std::shared_ptr<T>(new T[size], array_deleter<T>);	

		abstract_array_t<T>::init(3, size, shape, ptr);
	}

	~array_t()
	{
		abstract_array_t<T>::free();
	}

private:
	// disable copy constructor, assign
	array_t(array_t &) { }
	const array_t &operator=(const array_t &) { return *this; }

public:
	// move constructor
	array_t(array_t &&other)
	{
		abstract_array_t<T>::init(
			other._ndims, other._size, other._shape, std::move(other._ptr));

		other.init();
	}

	// move assign
	const array_t &operator=(array_t &&other)
	{
		abstract_array_t<T>::free();

		abstract_array_t<T>::init(
			other._ndims, other._size, other._shape, std::move(other._ptr));

		other.init();

		return *this;
	}
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_