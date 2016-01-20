#ifndef ARRAY_COLLECTION_H_
#define ARRAY_COLLECTION_H_

#include <numcpp/array.h>
#include <vector>

// ArrayCollection interface
template <typename T, int Dim = 1>
struct ArrayCollection
{
	virtual int size() = 0;
    virtual np::Array<T, Dim> at(int index) = 0;
};

// wrapper for std::vector<np::Array<T, Dim>> to use as ArrayCollection<T, Dim>
template <typename T, int Dim>
class ArrayVector : public ArrayCollection<T, Dim>
{
public:
    std::vector<np::Array<T, Dim>> arrays;

    // implements ArrayCollection
    int size() override 
    { 
        return (int)arrays.size(); 
    }

    np::Array<T, Dim> at(int index) override
    {
        return np::Array<T, Dim>(arrays.at(index));
    }
};

#endif // ARRAY_COLLECTION_H_