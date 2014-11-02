#ifndef NUMCPP_ALLOCATOR_H_
#define NUMCPP_ALLOCATOR_H_

#include <memory>

namespace np {

template <typename T>
struct heap_allocator
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

} // namespace np

#endif // NUMCPP_ALLOCATOR_H_