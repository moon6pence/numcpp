#ifndef __TMP_H__
#define __TMP_H__

// ## TMP.h : template metaprogramming to unroll small compile-time loops

template <int N>
struct TMP
{
	static int multiply_all(int *array)
	{
		return array[0] * TMP<N - 1>::multiply_all(array + 1);
	}

	static int copy(int *dst, const int *src)
	{
		dst[0] = src[0];
		TMP<N - 1>::copy(dst + 1, src + 1);
	}
};

template <>
struct TMP<1>
{
	static int multiply_all(int *array)
	{
		return array[0];
	}

	static int copy(int *dst, const int *src)
	{
		dst[0] = src[0];
	}
};

#endif // __TMP_H__