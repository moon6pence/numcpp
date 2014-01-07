#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace numcpp {

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

template <typename T>
struct array_t : public base_array_t<heap_allocator>
{
public:
	typedef base_array_t<heap_allocator> parent_t;

	array_t()
	{
	}

	array_t(int size0)
	{
		setSize(size0);
	}

	array_t(int size0, int size1)
	{
		setSize(size0, size1);
	}

	array_t(int size0, int size1, int size2)
	{
		setSize(size0, size1, size2);
	}

private:
	// disable copy constructor, assign
	array_t(array_t &) { }
	const array_t &operator=(const array_t &) { return *this; }

public:
	// inherits move constructor
	array_t(array_t &&other) : parent_t(std::move(other))
	{
	}

	// inherits move assign
	const array_t &operator=(array_t &&other)
	{
		parent_t::operator=(std::move(other));
		return *this;
	}

	void setSize(int size0)
	{
		parent_t::setSize<T>(size0);
	}

	void setSize(int size0, int size1)
	{
		parent_t::setSize<T>(size0, size1);
	}

	void setSize(int size0, int size1, int size2)
	{
		parent_t::setSize<T>(size0, size1, size2);
	}

	void setSize(int ndims, int *shape)
	{
		parent_t::setSize<T>(ndims, shape);
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return parent_t::raw_ptr<T>();
	}

	const T *raw_ptr() const
	{
		return parent_t::raw_ptr<T>();
	}

	operator T * ()
	{
		return parent_t::raw_ptr<T>();
	}

	operator const T * () const
	{
		return parent_t::raw_ptr<T>();
	}

	// at(index0, index...) : access array elements

	T& at(int index0)
	{
		return parent_t::at<T>(index0);
	}

	T& at(int index0, int index1)
	{
		return parent_t::at<T>(index0, index1);
	}

	T& at(int index0, int index1, int index2)
	{
		return parent_t::at<T>(index0, index1, index2);
	}

	const T& at(int index0) const
	{
		return parent_t::at<T>(index0);
	}

	const T& at(int index0, int index1) const
	{
		return parent_t::at<T>(index0, index1);
	}

	const T& at(int index0, int index1, int index2) const
	{
		return parent_t::at<T>(index0, index1, index2);
	}

	T& operator() (int index0)
	{
		return parent_t::at<T>(index0);
	}

	T& operator() (int index0, int index1)
	{
		return parent_t::at<T>(index0, index1);
	}

	T& operator() (int index0, int index1, int index2)
	{
		return parent_t::at<T>(index0, index1, index2);
	}

	const T& operator() (int index0) const
	{
		return parent_t::at<T>(index0);
	}

	const T& operator() (int index0, int index1) const
	{
		return parent_t::at<T>(index0, index1);
	}

	const T& operator() (int index0, int index1, int index2) const
	{
		return parent_t::at<T>(index0, index1, index2);
	}
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_