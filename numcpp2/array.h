#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <memory> // shared_ptr, unique_ptr
#include "tmp.h" // template metaprogramming to unroll small loops

typedef int size_type;

template <typename T, int Dim = 1>
struct array_t
{
	typedef T value_type;

private:
	// TODO: make member variables const
	std::shared_ptr<void> _address;
	value_type *_origin;
	size_type *_shape;

public:
	array_t(std::shared_ptr<void> address, value_type *origin, size_type *shape) :
		_address(address), 
		_origin(origin), 
		_shape(shape) // this pointer will be deleted in destructor
	{
	}

	~array_t()
	{
		delete[] _shape;
	}

	/** Returns count of all element in array */
	int size() const
	{
		// (without template metaprogramming)
		// int size = 1;
		// for (int i = 0; i < Dim; i++)
		// 	size *= _shape[i];
		int size = TMP<Dim>::multiply_all(_shape);	

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
};

#include "array_allocate.h"

#endif // __ARRAY_H__