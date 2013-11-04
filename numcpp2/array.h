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
	std::shared_ptr<void> _address;
	value_type *_origin;
	shape_type _shape;

public:
	array_t(std::shared_ptr<void> address, shape_type shape) :
		array_t(address, address, shape)
	{
	}

	array_t(std::shared_ptr<void> address, value_type *origin, shape_type shape) :
		_address(address), 
		_origin(origin), 
		_shape(shape)
	{
	}
};

#endif // __ARRAY_H__