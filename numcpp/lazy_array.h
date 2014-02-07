#ifndef LAZY_ARRAY_H_
#define LAZY_ARRAY_H_

#include "array.h"

template <class Array1, class Array2>
struct lazy_add_array
{
	lazy_add_array(Array1 &a1, Array2 &a2) : a1(a1), a2(a2)
	{
		assert(a1.size() == a2.size());
	}

	int size() const
	{
		return a1.size();
	}

	const decltype(Array1().at(0) + Array2().at(0)) at(int index0) const
	{
		return a1.at(index0) + a2.at(index0);
	}

private:
	const Array1 &a1;
	const Array2 &a2;
};

// template <typename T>
// array_t<T> add(const array_t<T> &a1, const array_t<T> &a2)
// {
// 	// TODO: assert shape
// 	assert(a1.size() == a2.size());

// 	array_t<T> result(a1.ndims(), a1.shape());
// 	transform(result, a1, a2, [](T _a1, T _a2) -> T { return _a1 + _a2; });
// 	return result;
// }

// template <typename T>
// void assign(array_t<T> &dst, const array_t<T> &src)
// {
// 	dst.setSize(src.ndims(), src.shape());

// 	auto _src = begin(src);
// 	for (auto _dst = begin(dst); _dst != end(dst); ++_dst, ++_src)
// 		*_dst = *_src;
// }

template <class Array1, class Array2>
lazy_add_array<Array1, Array2> 
	add(Array1 &a1, Array2 &a2)
{
	return lazy_add_array<Array1, Array2>(a1, a2);
}

template <typename T, class Array1, class Array2>
void assign(array_t<T> &dst, const lazy_add_array<Array1, Array2> &lazy_array)
{
	dst.setSize(lazy_array.size());

	for (int i = 0; i < lazy_array.size(); i++)
		dst.at(i) = lazy_array.at(i);
}

#endif // LAZY_ARRAY_H_