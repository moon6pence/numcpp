#ifndef __TMP_H__
#define __TMP_H__

// # TMP.h : template metaprogramming to unroll small compile-time loops

template <int N>
struct TMP
{
	static int product(int *array)
	{
		return array[0] * TMP<N - 1>::product(array + 1);
	}

	static int copy(int *dst, const int *src)
	{
		dst[0] = src[0];
		TMP<N - 1>::copy(dst + 1, src + 1);
	}

	static int fill(int *dst, int value)
	{
		dst[0] = value;
		TMP<N - 1>::copy(dst + 1, value);
	}
};

template <>
struct TMP<1>
{
	static int product(int *array)
	{
		return array[0];
	}

	static int copy(int *dst, const int *src)
	{
		dst[0] = src[0];
	}

	static int fill(int *dst, int value)
	{
		dst[0] = value;
	}
};

// ## Variadic templates

template <typename Arg1>
int product(Arg1 arg1) 
{ 
	return arg1;
}

template <typename Arg1, typename... Args>
int product(Arg1 arg1, Args... args)
{
	return arg1 * product(args...);
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