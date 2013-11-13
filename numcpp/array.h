#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <memory> // shared_ptr, unique_ptr
#include "tmp.h" // template metaprogramming to unroll small loops

namespace numcpp {

template <typename T, int Dim = 1>
struct array_t
{
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

	/** Returns size in certain dimension */
	int size(int dim) const
	{
		return _shape[dim];
	}

	/** Returns length of 1 dimension array(=vector) */
	int length() const
	{
		static_assert(Dim == 1, "This function is only for array_t<T, 1>");
		return size(0);
	}

	/** Returns height of 2 dimension array(=matrix, image) */
	int height() const
	{
		static_assert(Dim == 2, "This function is only for array_t<T, 2>");
		return size(0);
	}

	/** Returns width of 2 dimension array(=matrix, image) */
	int width() const
	{
		static_assert(Dim == 2, "This function is only for array_t<T, 2>");
		return size(1);
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

} // namespace numcpp

#include "array_allocate.h"
#include "array_function.h"

#endif // __ARRAY_H__