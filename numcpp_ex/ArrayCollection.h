#ifndef ARRAY_COLLECTION_H_
#define ARRAY_COLLECTION_H_

#include <numcpp/array.h>

// ArrayCollection interface
template <typename T, int Dim = 1>
struct ArrayCollection
{
	virtual int size() = 0;
    virtual np::Array<T, Dim> at(int index) = 0;
};

#endif // ARRAY_COLLECTION_H_