#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

namespace numcpp {

template <typename T>
struct array_t
{
	array_t() : _size(0)
	{
	}

	array_t(int size) : _size(size)
	{
	}

	bool empty()
	{
		return _size == 0;
	}

	int size()
	{
		return _size;
	}

private:
	int _size;
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_