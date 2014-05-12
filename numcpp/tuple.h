#ifndef NUMCPP_TUPLE_H_
#define NUMCPP_TUPLE_H_

#include <memory>

namespace np {

// tuple to store array size, immutable
struct tuple
{
    tuple() : _length(0)
    { 
    }

	explicit tuple(int size0) :
		_length(1), 
		_ptr(new int[1])
	{
		_ptr[0] = size0;
	}

	tuple(int size0, int size1) :
		_length(2), 
		_ptr(new int[2])
	{
		_ptr[0] = size0;
		_ptr[1] = size1;
	}

	tuple(int size, int *ptr) : 
        _length(size), 
        _ptr(new int[size])
    {
        for (int i = 0; i < size; i++)
            _ptr[i] = ptr[i];
	}

    // copy constructor
    tuple(const tuple &other) : 
        _length(other.length()), 
        _ptr(new int[other.length()])
    {
        for (int i = 0; i < length(); i++)
            _ptr[i] = other[i];
    }

    // copy assign
    const tuple &operator=(const tuple &other)
    {
        _length = other.length();
        _ptr.reset(new int[length()]);

        for (int i = 0; i < length(); i++)
            _ptr[i] = other[i];

        return *this;
    }

    // move constructor
    tuple(tuple &&other) :
        _length(other.length()), 
        _ptr(std::move(other._ptr))
    {
        other._length = 0;
    }

    // move assign
    const tuple &operator=(tuple &&other)
    {
        _length = other.length();
        _ptr = std::move(other._ptr);

        other._length = 0;

        return *this;
    }

    int length() const
    {
        return _length;
    }

    // TODO: Remove this function
    int *ptr() const
    {
        return _ptr.get();
    }

	int at(int index) const
	{
		return _ptr[index];
	}

    int operator[](int index) const
    {
        return at(index);
    }

	// # Utility functions

	int product() const
	{
		int result = 1;
		for (int i = 0; i < length(); i++)
			result *= at(i);

		return result;
	}

	bool equals(const tuple &other) const
	{
		if (length() != other.length())
			return false;

		for (int i = 0; i < length(); i++)
			if (at(i) != other.at(i))
				return false;

		return true;
	}

	bool operator==(const tuple &other) const
	{
		return equals(other);
	}

	bool operator!=(const tuple &other) const
	{
		return !equals(other);
	}

private:
    int _length;
    std::unique_ptr<int[]> _ptr;
};

} // namespace np

#endif // NUMCPP_TUPLE_H_