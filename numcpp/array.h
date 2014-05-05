#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace np {

template <typename T>
struct array_t : public base_array_t
{
public:
	typedef T element_type;

	array_t() : base_array_t(sizeof(T))
	{
	}

	explicit array_t(int size0) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<heap_allocator>(tuple(size0));
	}

	array_t(int size0, int size1) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<heap_allocator>(tuple(size0, size1));
	}

	array_t(int size0, int size1, int size2) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<heap_allocator>(tuple(size0, size1, size2));
	}

	explicit array_t(const tuple &size) : base_array_t(sizeof(T))
	{
		base_array_t::setSize<heap_allocator>(size);
	}

public:
	// inherits copy constructor
	explicit array_t(const base_array_t &other) : base_array_t(other)
	{
		assert(other.itemSize() == sizeof(T));
	}

	// inherits copy assign
	const array_t &operator=(const base_array_t &other)
	{
		assert(other.itemSize() == sizeof(T));

		base_array_t::operator=(other);
		return *this;
	}

	// move constructor (inherited)
	array_t(array_t &&other) : base_array_t(std::move(other))
	{
	}

	// move constructor (for base_array_t)
	array_t(base_array_t &&other) : base_array_t(std::move(other))
	{
	}

	// move assign (inherited)
	const array_t &operator=(array_t &&other)
	{
		base_array_t::operator=(std::move(other));
		return *this;
	}

	// move assign (for base_array_t)
	const array_t &operator=(base_array_t &&other)
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

	array_t<T> slice(int from, int to)
	{
		return array_t<T>(base_array_t::slice(from, to));
	}

	array_t<T> slice(int from0, int from1, int to0, int to1)
	{
		return array_t<T>(base_array_t::slice(from0, from1, to0, to1));
	}

	template <class LazyArray>
	const array_t<T> &operator=(LazyArray lazy_array)
	{
		if (this->size() != lazy_array.size())
			(*this) = array_t<T>(lazy_array.size());

		for (int i = 0; i < lazy_array.length(); i++)
			this->at(i) = lazy_array.at(i);

		return *this;
	}
};

} // namespace np

#endif // NUMCPP_ARRAY_H_