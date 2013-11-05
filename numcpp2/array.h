#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <memory> // shared_ptr

typedef int size_type;
typedef size_type* shape_type;

template <typename T, int Dim = 1>
struct array_t
{
	typedef T value_type;

private:
	// TODO: make member variables const
	std::shared_ptr<void> _address;
	value_type *_origin;
	std::unique_ptr<size_type> _shape;

public:
	array_t(std::shared_ptr<void> address, value_type *origin, shape_type shape) :
		_address(address), 
		_origin(origin), 
		_shape(shape)
	{
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
	int size = 1;
	for (int i = 0; i < Dim; i++)
		size *= shape[i];

	// allocate buffer
	T *buffer = new T[size];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	size_type *new_shape = new int[Dim];
	for (int i = 0; i < Dim; i++)
		new_shape[i] = shape[i];

	return array_t<T, Dim>(address, origin, new_shape);
}

template <typename T>
array_t<T, 1> array(int length)
{
	// allocate buffer
	T *buffer = new T[length];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	size_type *shape = new int[1];
	shape[0] = length;

	return array_t<T, 1>(address, origin, shape);
}

template <typename T>
array_t<T, 2> array(int height, int width)
{
	// allocate buffer
	T *buffer = new T[height * width];

	// address
	std::shared_ptr<void> address(buffer, array_deleter<T>);

	// origin
	T *origin = buffer;

	// shape
	size_type *shape = new int[2];
	shape[0] = height;
	shape[1] = width;

	return array_t<T, 2>(address, origin, shape);
}

#endif // __ARRAY_H__