#ifndef LAZY_ARRAY_H_
#define LAZY_ARRAY_H_

#include "array.h"

template <typename T>
array_t<T> add(const array_t<T> &a1, const array_t<T> &a2)
{
	// TODO: assert shape
	assert(a1.size() == a2.size());

	array_t<T> result(a1.ndims(), a1.shape());
	transform(result, a1, a2, [](T _a1, T _a2) -> T { return _a1 + _a2; });
	return result;
}

template <typename T>
void assign(array_t<T> &dst, const array_t<T> &src)
{
	dst.setSize(src.ndims(), src.shape());

	auto _src = begin(src);
	for (auto _dst = begin(dst); _dst != end(dst); ++_dst, ++_src)
		*_dst = *_src;
}

#endif // LAZY_ARRAY_H_