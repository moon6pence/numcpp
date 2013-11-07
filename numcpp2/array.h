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
	int size()
	{
		// (without template metaprogramming)
		// int size = 1;
		// for (int i = 0; i < Dim; i++)
		// 	size *= _shape[i];
		int size = TMP<Dim>::multiply_all(_shape);	

		return size;	
	}
};

template <typename T>
void array_deleter(T const *p)
{
	delete[] p;
}

template <typename T, int Dim = 1>
array_t<T, Dim> array(int *shape)
{
	// (without template metaprogramming)
	// int size = 1;
	// for (int i = 0; i < Dim; i++) 
	//	size *= shape[i];
	int size = TMP<Dim>::multiply_all(shape);

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	size_type *new_shape = new int[Dim];

	// (without template metaprogramming)
	// for (int i = 0; i < Dim; i++)
	//	new_shape[i] = shape[i];
	TMP<Dim>::copy(new_shape, shape);

	return array_t<T, Dim>(address, origin, new_shape);
}

template <typename T, typename... Shape>
array_t<T, sizeof...(Shape)> array(Shape... shape)
{
	int size = multiply_all(shape...);

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	size_type *new_shape = new int[sizeof...(Shape)];
	copy(new_shape, shape...);

	return array_t<T, sizeof...(Shape)>(address, origin, new_shape);
}

#endif // __ARRAY_H__