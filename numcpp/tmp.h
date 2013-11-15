#ifndef __NUMCPP_TMP_H__
#define __NUMCPP_TMP_H__

// # TMP.h : template metaprogramming to unroll small compile-time loops

namespace numcpp {

// ## Recursive templates with N

template <int N>
struct TMP_N
{
	/** Product all elements in array */
	static int product(int *array)
	{
		return array[0] * TMP_N<N - 1>::product(array + 1);
	}

	/** Copy elements from src array to dst array */
	static int copy(int *dst, const int *src)
	{
		dst[0] = src[0];
		TMP_N<N - 1>::copy(dst + 1, src + 1);
	}

	/** Fill array elements with value */
	static int fill(int *dst, int value)
	{
		dst[0] = value;
		TMP_N<N - 1>::fill(dst + 1, value);
	}
};

template <>
struct TMP_N<1>
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

// ## Recursive templates with variadic arguments

namespace TMP_V
{
	/** product all arguments */
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

	/** Copy all arguments to dst array */
	template <typename Arg1>
	void copy(int *dst, Arg1 arg1)
	{
		dst[0] = arg1;	
	}

	template <typename Arg1, typename... Args>
	void copy(int *dst, Arg1 arg1, Args... args)
	{
		dst[0] = arg1;
		copy(dst + 1, args...);	
	}

	/** Calculate offset of array element with shape and givin indexes */
	template <typename Arg1>
	int offset(int *shape, Arg1 arg1)
	{
		return arg1;
	}

	template <typename Arg1, typename... Args>
	int offset(int *shape, Arg1 arg1, Args... args)
	{
		return arg1 + shape[0] * offset(shape + 1, args...);
	}
};

} // namespace numcpp

#endif // __NUMCPP_TMP_H__