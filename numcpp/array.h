#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace np {

template <typename T>
struct Array : public BaseArray
{
public:
	typedef T element_type;

	Array() : BaseArray(sizeof(T))
	{
	}

	explicit Array(int size0) : BaseArray(sizeof(T), make_vector(size0))
	{
	}

	Array(int size0, int size1) : BaseArray(sizeof(T), make_vector(size0, size1))
	{
	}

	Array(int size0, int size1, int size2) : BaseArray(sizeof(T), make_vector(size0, size1, size2))
	{
	}

	explicit Array(const std::vector<int> &size) : BaseArray(sizeof(T), size)
	{
	}

private:
	// delete copy constructor, assign
	explicit Array(const Array &other)	{ }
	const Array &operator=(const BaseArray &other) { return *this; }

public:
	// move constructor (inherited)
	Array(Array &&other) : BaseArray(std::move(other))
	{
	}

	// move constructor (for BaseArray)
	Array(BaseArray &&other) : BaseArray(std::move(other))
	{
		//assert(other.itemSize() == sizeof(T));
	}

	// move assign (inherited)
	const Array &operator=(Array &&other)
	{
		BaseArray::operator=(std::move(other));
		return *this;
	}

	// move assign (for BaseArray)
	const Array &operator=(BaseArray &&other)
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

	Array<T> slice(int from, int to)
	{
		return Array<T>(BaseArray::slice(from, to));
	}

	Array<T> slice(int from0, int from1, int to0, int to1)
	{
		return Array<T>(BaseArray::slice(from0, from1, to0, to1));
	}

	template <class LazyArray>
	const Array<T> &operator=(LazyArray lazy_array)
	{
		if (this->size() != lazy_array.size())
			(*this) = Array<T>(lazy_array.size());

		for (int i = 0; i < lazy_array.length(); i++)
			this->at(i) = lazy_array.at(i);

		return *this;
	}

	// for backward compatibility
	int byteSize() const
	{
		return length() * itemSize();
	}

	bool empty() const
	{
		return raw_ptr() == nullptr || length() == 0;
	}
};

template <typename T, typename U>
Array<T> similar(const Array<U> &other)
{
	return Array<T>(other.size());
}

} // namespace np

#endif // NUMCPP_ARRAY_H_