#ifndef NUMCPP_TUPLE_H_
#define NUMCPP_TUPLE_H_

#include <memory>

namespace np {

// tuple to store array size, immutable
struct tuple
{
    tuple() : _size(0)
    { 
    }

    tuple(int size, int *ptr) : 
        _size(size), 
        _ptr(new int[size])
    {
        for (int i = 0; i < size; i++)
            _ptr[i] = ptr[i];
    }

    // copy constructor
    tuple(const tuple &other) : 
        _size(other.size()), 
        _ptr(new int[other.size()])
    {
        for (int i = 0; i < size(); i++)
            _ptr[i] = other[i];
    }

    // copy assign
    const tuple &operator=(const tuple &other)
    {
        _size = other.size();
        _ptr.reset(new int[size()]);

        for (int i = 0; i < size(); i++)
            _ptr[i] = other[i];

        return *this;
    }

    // move constructor
    tuple(tuple &&other) :
        _size(other.size()), 
        _ptr(std::move(other._ptr))
    {
        other._size = 0;
    }

    // move assign
    const tuple &operator=(tuple &&other)
    {
        _size = other.size();
        _ptr = std::move(other._ptr);

        other._size = 0;

        return *this;
    }

    int size() const
    {
        return _size;
    }

    // TODO: Remove this function
    int *ptr() const
    {
        return _ptr.get();
    }

    int operator[](int index) const
    {
        return _ptr[index];
    }

private:
    int _size;
    std::unique_ptr<int[]> _ptr;
};

} // namespace np

#endif // NUMCPP_TUPLE_H_