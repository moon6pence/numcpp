#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace numcpp {

template <typename T>
struct heap_allocator
{
	static T *allocate(int size)
	{
		return new T[size];
	}

	static void free(T *ptr)
	{
		delete[] ptr;
	}
};

template <typename T>
struct array_t : public base_array_t<T, heap_allocator<T>>
{
public:
	array_t()
	{
		this->setEmpty();
	}

	array_t(int size0)
	{
		this->setSize(size0);
	}

	array_t(int size0, int size1)
	{
		this->setSize(size0, size1);
	}

	array_t(int size0, int size1, int size2)
	{
		this->setSize(size0, size1, size2);
	}

	~array_t()
	{
		this->free();
	}

private:
	// disable copy constructor, assign
	array_t(array_t &) { }
	const array_t &operator=(const array_t &) { return *this; }

public:
	// inherits move constructor
	array_t(array_t &&other) : 
		base_array_t<T, heap_allocator<T>>(std::move(other))
	{
	}

	// inherits move assign
	const array_t &operator=(array_t &&other)
	{
		base_array_t<T, heap_allocator<T>>::operator=(std::move(other));
		return *this;
	}
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_