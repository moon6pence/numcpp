#ifndef NUMCPP_TARRAY_H_
#define NUMCPP_TARRAY_H_

#include <array>

namespace np {

template <typename T, int Dim = 1>
struct TArray
{
private:
	void *_origin;

public:
	typedef T value_type;

	TArray() : _origin(nullptr)
	{
	}

	TArray(int size0) : _origin(nullptr)
	{
	}

	// ## Access to premitive properties

	int itemSize() const
	{
		return sizeof(value_type);
	}

	//const tuple &size() const
	//{
	//	return _size;
	//}

	//int size(int dim) const
	//{
	//	return _size[dim];
	//}

	//int stride(int dim) const
	//{
	//	return _stride[dim];
	//}

	void *raw_ptr()
	{
		return _origin;
	}

	const void *raw_ptr() const
	{
		return _origin;
	}

	// ## Derived property functions

	int ndims() const
	{
		return Dim;
	}

	bool empty() const
	{
		return true;
		//return raw_ptr() == nullptr || length() == 0;
	}

	//int length() const
	//{
	//	return size().product();
	//}

	//int byteSize() const
	//{
	//	return length() * itemSize();
	//}

	//template <typename T>
	//T *raw_ptr()
	//{
	//	return static_cast<T *>(raw_ptr());
	//}

	//template <typename T>
	//const T *raw_ptr() const
	//{
	//	return static_cast<const T *>(raw_ptr());
	//}
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

	// ## Access to premitive properties

	int itemSize() const
	{
		return sizeof(value_type);
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

	// ## Derived property functions

	int ndims() const
	{
		return 1;
	}

	bool empty() const
	{
		return raw_ptr() == nullptr;
	}

	int length() const
	{
		return _size0;
	}

	int byteSize() const
	{
		return length() * itemSize();
	}
};

} // namespace np

#endif // NUMCPP_TARRAY_H_