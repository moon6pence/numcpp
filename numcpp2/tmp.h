#ifndef __TMP_H__
#define __TMP_H__

// # TMP.h : template metaprogramming to unroll small compile-time loops

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

// ## Variadic templates

template <typename Arg1>
int multiply_all(Arg1 arg1) 
{ 
	return arg1;
}

template <typename Arg1, typename... Args>
int multiply_all(Arg1 arg1, Args... args)
{
	return arg1 * multiply_all(args...);
}

template <typename T, typename Arg1>
void copy(T *dst, Arg1 arg1)
{
	dst[0] = arg1;	
}

template <typename T, typename Arg1, typename... Args>
void copy(T *dst, Arg1 arg1, Args... args)
{
	dst[0] = arg1;
	copy(dst + 1, args...);	
}

#endif // __TMP_H__