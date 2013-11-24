#ifndef __NUMCPP_ARRAY_H__
#define __NUMCPP_ARRAY_H__

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

private:
	array_t(array_t &) { }

public:
	array_t(array_t &&other) :
		_address(std::move(other._address)), 
		_origin(other._origin), 
		_shape(other._shape)
	{
		other._shape = nullptr;
	}

	// ## Part 1. Concepts of array_t, requirements on array types

	/** Returns count of all element in array */
	int size() const
	{
		return TMP_N<Dim>::product(_shape);	
	}

	/** Returns size in certain dimension */
	int size(int dim) const
	{
		return _shape[dim];
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

#ifdef VARIADIC_TEMPLATE
	template <typename... Index>
	T &at(Index... index)
	{
		return _origin[TMP_V::offset(_shape, index...)];	
	}

	template <typename... Index>
	const T &at(Index... index) const
	{
		return _origin[TMP_V::offset(_shape, index...)];	
	}
#else // ifndef VARIADIC_TEMPLATE
	T &at(int index0)
	{
		return _origin[index0];	
	}

	T &at(int index0, int index1)
	{
		return _origin[index0 + _shape[0] * index1];	
	}

	T &at(int index0, int index1, int index2)
	{
		return _origin[index0 + _shape[0] * (index1 + _shape[1] * index2)];	
	}

	const T &at(int index0) const
	{
		return _origin[index0];	
	}

	const T &at(int index0, int index1) const
	{
		return _origin[index0 + _shape[0] * index1];	
	}

	const T &at(int index0, int index1, int index2) const
	{
		return _origin[index0 + _shape[0] * (index1 + _shape[1] * index2)];	
	}
#endif // VARIADIC_TEMPLATE

	// ## Part 2. Syntatic sugars

	/** Check if array is empty */
	bool empty() const
	{
		return size() == 0;
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

#ifdef VARIADIC_TEMPLATE
	template <typename... Index>
	T &operator() (Index... index)
	{
		return at(index...);
	}
	
	template <typename... Index>
	const T &operator() (Index... index) const
	{
		return at(index...);
	}
#else
	T &operator() (int index0) { return at(index0); }
	T &operator() (int index0, int index1) { return at(index0, index1); }
	T &operator() (int index0, int index1, int index2) { return at(index0, index1, index2); }

	const T &operator() (int index0) const { return at(index0); }
	const T &operator() (int index0, int index1) const { return at(index0, index1); }
	const T &operator() (int index0, int index1, int index2) const { return at(index0, index1, index2); }
#endif
};

} // namespace numcpp

#endif // __NUMCPP_ARRAY_H__