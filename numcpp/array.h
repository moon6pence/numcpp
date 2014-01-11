#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace numcpp {

template <typename T>
struct array_t : public base_array_t
{
public:
	typedef base_array_t parent_t;

	array_t() : base_array_t(sizeof(T))
	{
	}

	array_t(int size0) : base_array_t(sizeof(T))
	{
		setSize(size0);
	}

	array_t(int size0, int size1) : base_array_t(sizeof(T))
	{
		setSize(size0, size1);
	}

	array_t(int size0, int size1, int size2) : base_array_t(sizeof(T))
	{
		setSize(size0, size1, size2);
	}

private:
	// disable copy constructor, assign
	array_t(array_t &) { }
	const array_t &operator=(const array_t &) { return *this; }

public:
	// inherits move constructor
	array_t(array_t &&other) : base_array_t(std::move(other))
	{
	}

	// inherits move assign
	const array_t &operator=(array_t &&other)
	{
		base_array_t::operator=(std::move(other));
		return *this;
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return base_array_t::raw_ptr<T>();
	}

	const T *raw_ptr() const
	{
		return base_array_t::raw_ptr<T>();
	}

	operator T * ()
	{
		return base_array_t::raw_ptr<T>();
	}

	operator const T * () const
	{
		return base_array_t::raw_ptr<T>();
	}

	// at(index0, index...) : access array elements

	T& at(int index0)
	{
		return base_array_t::at<T>(index0);
	}

	T& at(int index0, int index1)
	{
		return base_array_t::at<T>(index0, index1);
	}

	T& at(int index0, int index1, int index2)
	{
		return base_array_t::at<T>(index0, index1, index2);
	}

	const T& at(int index0) const
	{
		return base_array_t::at<T>(index0);
	}

	const T& at(int index0, int index1) const
	{
		return base_array_t::at<T>(index0, index1);
	}

	const T& at(int index0, int index1, int index2) const
	{
		return base_array_t::at<T>(index0, index1, index2);
	}

	T& operator() (int index0)
	{
		return base_array_t::at<T>(index0);
	}

	T& operator() (int index0, int index1)
	{
		return base_array_t::at<T>(index0, index1);
	}

	T& operator() (int index0, int index1, int index2)
	{
		return base_array_t::at<T>(index0, index1, index2);
	}

	const T& operator() (int index0) const
	{
		return base_array_t::at<T>(index0);
	}

	const T& operator() (int index0, int index1) const
	{
		return base_array_t::at<T>(index0, index1);
	}

	const T& operator() (int index0, int index1, int index2) const
	{
		return base_array_t::at<T>(index0, index1, index2);
	}
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_