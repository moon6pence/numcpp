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

#endif // __ARRAY_H__