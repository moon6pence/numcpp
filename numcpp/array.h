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
	array_t() : 
		_address(nullptr), 
		_origin(nullptr), 
		_shape(new int[Dim])
	{
		TMP_N<Dim>::fill(_shape, 0);	
	}

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
	const array_t &operator=(const array_t &) { return *this; }

public:
	// Move constructor
	array_t(array_t &&other) :
		_address(std::move(other._address)), 
		_origin(other._origin), 
		_shape(other._shape)
	{
		other._shape = nullptr;
	}

	// Move assign
	const array_t &operator=(array_t &&other)
	{
		// Deallocate this
		if (_shape) { delete[] _shape; _shape = nullptr; }

		// Move from other
		_address = std::move(other._address);
		_origin = other._origin;
		_shape = other._shape;

		// Clear other
		other._shape = nullptr;

		return *this;
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

	/** Returns array contains size in each dimension */
	int *shape() const
	{
		return _shape;
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

	/** Returns width of 2 dimension array(=matrix, image) */
	int width() const
	{
		static_assert(Dim == 2, "This function is only for array_t<T, 2>");
		return size(0);
	}

	/** Returns height of 2 dimension array(=matrix, image) */
	int height() const
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

/** Allocate empty array */
template <typename T, int Dim>
array_t<T, Dim> empty()
{
	return array_t<T, Dim>();
}

template <typename T>
void array_deleter(T const *p)
{
	delete[] p;
}

#ifdef VARIADIC_TEMPLATE

/** Allocate array with given shape */
template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> array(Shape... shape)
{
	int size = TMP_V::product(shape...);

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[sizeof...(Shape)];
	TMP_V::copy(new_shape, shape...);

	return array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

#else // ifndef VARIADIC_TEMPLATE

/** Allocate 1 dimension array */
template <typename T>
array_t<T, 1> array(int shape0)
{
	int size = shape0;

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[1];
	new_shape[0] = shape0;

	return array_t<T, 1>(address, origin, new_shape);
}

/** Allocate 2 dimension array */
template <typename T>
array_t<T, 2> array(int shape0, int shape1)
{
	int size = shape0 * shape1;

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[2];
	new_shape[0] = shape0;
	new_shape[1] = shape1;

	return array_t<T, 2>(address, origin, new_shape);
}

/** Allocate 3 dimension array */
template <typename T>
array_t<T, 3> array(int shape0, int shape1, int shape2)
{
	int size = shape0 * shape1 * shape2;

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	int *new_shape = new int[3];
	new_shape[0] = shape0;
	new_shape[1] = shape1;
	new_shape[2] = shape2;

	return array_t<T, 3>(address, origin, new_shape);
}

#endif // VARIADIC_TEMPLATE

} // namespace numcpp

#endif // __NUMCPP_ARRAY_H__