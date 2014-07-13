#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace np {

template <typename T>
struct array_t : public BaseArray
{
public:
	typedef T element_type;

	array_t() : BaseArray(sizeof(T))
	{
	}

	explicit array_t(int size0) : BaseArray(sizeof(T), tuple(size0))
	{
	}

	array_t(int size0, int size1) : BaseArray(sizeof(T), tuple(size0, size1))
	{
	}

	explicit array_t(const tuple &size) : BaseArray(sizeof(T), size)
	{
	}

private:
	// delete copy constructor, assign
	explicit array_t(const array_t &other)	{ }
	const array_t &operator=(const BaseArray &other) { return *this; }

public:
	// move constructor (inherited)
	array_t(array_t &&other) : BaseArray(std::move(other))
	{
	}

	// move constructor (for BaseArray)
	array_t(BaseArray &&other) : BaseArray(std::move(other))
	{
		assert(other.itemSize() == sizeof(T));
	}

	// move assign (inherited)
	const array_t &operator=(array_t &&other)
	{
		BaseArray::operator=(std::move(other));
		return *this;
	}

	// move assign (for BaseArray)
	const array_t &operator=(BaseArray &&other)
	{
		assert(other.itemSize() == sizeof(T));

		BaseArray::operator=(std::move(other));
		return *this;
	}

	// raw_ptr(): access raw pointer

	T *raw_ptr()
	{
		return BaseArray::raw_ptr<T>();
	}

	const T *raw_ptr() const
	{
		return BaseArray::raw_ptr<T>();
	}

	operator T * ()
	{
		return BaseArray::raw_ptr<T>();
	}

	operator const T * () const
	{
		return BaseArray::raw_ptr<T>();
	}

	// at(index0, index...) : access array elements

	T& at(int index0)
	{
		return BaseArray::at<T>(index0);
	}

	T& at(int index0, int index1)
	{
		return BaseArray::at<T>(index0, index1);
	}

	const T& at(int index0) const
	{
		return BaseArray::at<T>(index0);
	}

	const T& at(int index0, int index1) const
	{
		return BaseArray::at<T>(index0, index1);
	}

	T& operator() (int index0)
	{
		return BaseArray::at<T>(index0);
	}

	T& operator() (int index0, int index1)
	{
		return BaseArray::at<T>(index0, index1);
	}

	const T& operator() (int index0) const
	{
		return BaseArray::at<T>(index0);
	}

	const T& operator() (int index0, int index1) const
	{
		return BaseArray::at<T>(index0, index1);
	}

	array_t<T> slice(int from, int to)
	{
		return array_t<T>(BaseArray::slice(from, to));
	}

	array_t<T> slice(int from0, int from1, int to0, int to1)
	{
		return array_t<T>(BaseArray::slice(from0, from1, to0, to1));
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

template <typename T, typename U>
array_t<T> similar(const array_t<U> &other)
{
	return array_t<T>(other.size());
}

} // namespace np

#endif // NUMCPP_ARRAY_H_