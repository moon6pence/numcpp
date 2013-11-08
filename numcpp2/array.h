#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <memory> // shared_ptr, unique_ptr
#include "tmp.h" // template metaprogramming to unroll small loops

template <typename T, int Dim = 1>
struct array_t
{
public:
	typedef T *iterator;
	typedef const T *const_iterator;

private:
	std::shared_ptr<void> _address;
	T *_origin;
	int *_shape;

public:
	array_t(std::shared_ptr<void> address, T *origin, int *shape) :
		_address(address), 
		_origin(origin), 
		_shape(shape) // this pointer will be deleted in destructor
	{
	}

	~array_t()
	{
		if (_shape) { delete[] _shape; _shape = nullptr; }
	}

	/** Check if array is empty */
	bool empty() const
	{
		return _address == nullptr || _origin == nullptr || _shape == nullptr;
	}

	/** Returns count of all element in array */
	int size() const
	{
		// (without template metaprogramming)
		// int size = 1;
		// for (int i = 0; i < Dim; i++)
		// 	size *= _shape[i];
		int size = TMP<Dim>::product(_shape);	

		return size;	
	}

	/** Returns raw pointer of array */
	T *raw_pointer() 
	{ 
		return _origin; 
	}

	/** Returns raw pointer of array (const) */
	const T *raw_pointer() const
	{
		return _origin;
	}

	/** Auto conversion to raw pointer */
	// 1. Should I use this? 
	// 2. explicit? 
	operator T *()
	{
		return raw_pointer();
	}

	/** Auto conversion to raw pointer (const) */
	operator const T *() const
	{
		return raw_pointer();
	}

	/** begin iterator */
	iterator begin()
	{
		return _origin;
	}

	/** end iterator */
	iterator end()
	{
		return _origin + size();
	}

	/** begin const_iterator */
	const_iterator begin() const
	{
		return _origin;
	}

	/** end const_interator */
	const_iterator end() const
	{
		return _origin + size();
	}
};

#include "array_allocate.h"
#include "array_function.h"

#endif // __ARRAY_H__