#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

#include "base_array.h"

namespace numcpp {

template <typename T>
void array_deleter(T *ptr)
{
	delete[] ptr;
}

template <typename T>
struct array_t : public base_array_t<T>
{
public:
	array_t()
	{
		this->init();
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

	~array_t()
	{
		base_array_t<T>::free();
	}

	void setSize(int size0)
	{
		if (this->ndims() == 1 && 
			this->size(0) == size0) return;

		base_array_t<T>::free();

		int size = size0;

		int *shape = new int[1];
		shape[0] = size0;

		auto ptr = std::shared_ptr<void>(new T[size], array_deleter<T>);

		this->init(1, size, shape, ptr);
	}

	void setSize(int size0, int size1)
	{
		if (this->ndims() == 2 && 
			this->size(0) == size0 && 
			this->size(1) == size1) return;

		base_array_t<T>::free();

		int size = size0 * size1;

		int *shape = new int[2];
		shape[0] = size0;
		shape[1] = size1;

		auto ptr = std::shared_ptr<void>(new T[size], array_deleter<T>);	

		this->init(2, size, shape, ptr);
	}

	void setSize(int size0, int size1, int size2)
	{
		if (this->ndims() == 3 && 
			this->size(0) == size0 && 
			this->size(1) == size1 && 
			this->size(2) == size2) return;

		base_array_t<T>::free();

		int size = size0 * size1 * size2;

		int *shape = new int[3];
		shape[0] = size0;
		shape[1] = size1;
		shape[2] = size2;

		auto ptr = std::shared_ptr<void>(new T[size], array_deleter<T>);	

		this->init(3, size, shape, ptr);
	}

private:
	// disable copy constructor, assign
	array_t(array_t &) { }
	const array_t &operator=(const array_t &) { return *this; }

public:
	// inherits move constructor
	array_t(array_t &&other) : base_array_t<T>(std::move(other))
	{
	}

	// inherits move assign
	const array_t &operator=(array_t &&other)
	{
		base_array_t<T>::operator=(std::move(other));
		return *this;
	}
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_