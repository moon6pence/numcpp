#ifndef NUMCPP_TARRAY_H_
#define NUMCPP_TARRAY_H_

#include <array>

namespace np {

template <typename T, int Dim = 1>
struct TArray
{
};

template <typename T>
struct default_allocator
{
	static std::shared_ptr<T> allocate(int size)
	{
		return std::shared_ptr<T>(new T[size], free);
	}

	static void free(T *ptr)
	{
		delete[] ptr;
	}
};

template <typename T>
struct TArray<T, 1>
{
public:
	typedef T value_type;

private:
	int _size0;
	std::shared_ptr<value_type> _address;
	value_type *_origin;

public:
	TArray() : 
		_size0(0), 
		_address(), 
		_origin(nullptr)
	{
	}

	TArray(int size0) : 
		_size0(size0), 
		_address(default_allocator<value_type>::allocate(size0)), 
		_origin(nullptr)
	{
		_origin = _address.get();
	}

	// copy constructor: shallow copy
	explicit TArray(const TArray &other) :
		_size0(other._size), 
		_address(other._address), 
		_origin(other._origin)
	{
	}

	// move constructor
	TArray(TArray &&other) :
		_size0(other._size0), 
		_address(std::move(other._address)), 
		_origin(other._origin)
	{
		other._origin = nullptr;
	}

	int itemSize() const
	{
		return sizeof(value_type);
	}

	int ndims() const
	{
		return 1;
	}

	std::array<int, 1> size() const
	{
		std::array<int, 1> size;
		size[0] = size0;
		return std::move(size);
	}

	template <int N>
	int size() const
	{
		ArrayIndexOutOfBounds;	
	}

	template <>
	int size<0>() const
	{
		return _size0;
	}

	int length() const
	{
		return _size0;
	}

	int byteSize() const
	{
		return length() * itemSize();
	}
	
	template <int N>
	int stride() const
	{
		ArrayIndexOutOfBounds;	
	}

	template <>
	int stride<0>() const
	{
		return sizeof(value_type);
	}

	value_type *raw_ptr()
	{
		return _origin;
	}

	const value_type *raw_ptr() const
	{
		return _origin;
	}

	bool empty() const
	{
		return raw_ptr() == nullptr;
	}
};

} // namespace np

#endif // NUMCPP_TARRAY_H_