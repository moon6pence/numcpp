#ifndef LAZY_ARRAY_H_
#define LAZY_ARRAY_H_

#include "array.h"
#include <cmath>

namespace np {

template <class Array1, class Array2, typename T, T Function(T, T)>
struct lazy_array_with_binary_function
{
	typedef T element_type;

	lazy_array_with_binary_function(const Array1 &a1, const Array2 &a2) : 
		a1(a1), a2(a2)
	{
		assert(a1.size() == a2.size());
	}

	int ndims() const
	{
		return a1.ndims();
	}

	int *shape() const
	{
		return a1.shape();
	}

	int size() const
	{
		return a1.size();
	}

	T at(int index0) const
	{
		return Function(a1.at(index0), a2.at(index0));
	}

private:
	const Array1 &a1;
	const Array2 &a2;
};

// template specialization: second parameter is constant value
template <class Array1, typename T, T Function(T, T)>
struct lazy_array_with_binary_function<Array1, T, T, Function>
{
	typedef T element_type;

	lazy_array_with_binary_function(const Array1 &a1, T value) : 
		a1(a1), value(value)
	{
	}

	int ndims() const
	{
		return a1.ndims();
	}

	int *shape() const
	{
		return a1.shape();
	}

	int size() const
	{
		return a1.size();
	}

	T at(int index0) const
	{
		return Function(a1.at(index0), value);
	}

private:
	const Array1 &a1;
	T value;
};

template <class Array1, typename T, T Function(T)>
struct lazy_array_with_unary_function
{
	typedef T element_type;

	lazy_array_with_unary_function(const Array1 &a1) : a1(a1)
	{
	}

	int ndims() const
	{
		return a1.ndims();
	}

	int *shape() const
	{
		return a1.shape();
	}
	
	int size() const
	{
		return a1.size();
	}

	T at(int index0) const
	{
		return Function(a1.at(index0));
	}

private:
	const Array1 &a1;
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

// template <class Array1, class Array2>
// lazy_array_with_binary_function<
// 	Array1, Array2, typename Array1::element_type, _add> 
// 	add(const Array1 &a1, const Array2 &a2)
// {
// 	return lazy_array_with_binary_function<
// 		Array1, Array2, typename Array1::element_type, _add>(a1, a2);
// }

namespace element_func
{
	template <typename T> T add(T a, T b) { return a + b; }
	template <typename T> T subtract(T a, T b) { return a - b; }
	template <typename T> T multiply(T a, T b) { return a * b; }
	template <typename T> T atan2(T a, T b) { return ::atan2(a, b); }
}

#define MAP_FUNC2_OP(OPERATOR, ELEMENT_FUNC)\
	template <class Array1, class Array2>\
	lazy_array_with_binary_function\
		<Array1, Array2, typename Array1::element_type, ELEMENT_FUNC>\
		operator OPERATOR (const Array1 &a1, const Array2 &a2)\
	{\
		return lazy_array_with_binary_function\
			<Array1, Array2, typename Array1::element_type, ELEMENT_FUNC>\
			(a1, a2);\
	}

MAP_FUNC2_OP(+, element_func::add)

#define MAP_FUNC2(ARRAY_FUNC, ELEMENT_FUNC)\
	template <class Array1, class Array2>\
	lazy_array_with_binary_function\
		<Array1, Array2, typename Array1::element_type, ELEMENT_FUNC>\
		ARRAY_FUNC(const Array1 &a1, const Array2 &a2)\
	{\
		return lazy_array_with_binary_function\
			<Array1, Array2, typename Array1::element_type, ELEMENT_FUNC>\
			(a1, a2);\
	}

MAP_FUNC2(add, element_func::add)
MAP_FUNC2(subtract, element_func::subtract)
MAP_FUNC2(multiply, element_func::multiply)
MAP_FUNC2(atan2, element_func::atan2)

// template <class Array1>
// lazy_array_with_unary_function<
// 	Array1, typename Array1::element_type, _minus> 
// 	minus(const Array1 &a1)
// {
// 	return lazy_array_with_unary_function<
// 		Array1, typename Array1::element_type, _minus>(a1);
// }

#define MAP_FUNC1(ARRAY_FUNC, ELEMENT_FUNC)\
	template <class Array1>\
	lazy_array_with_unary_function<\
		Array1, typename Array1::element_type, ELEMENT_FUNC> \
		ARRAY_FUNC(const Array1 &a1)\
	{\
		return lazy_array_with_unary_function<\
			Array1, typename Array1::element_type, ELEMENT_FUNC>(a1);\
	}

namespace element_func
{
	template <typename T> T minus(T a) { return -a; }
	template <typename T> T cos(T a) { return ::cos(a); }
	template <typename T> T sin(T a) { return ::sin(a); }
	template <typename T> T sqrt(T a) { return ::sqrt(a); }
}

MAP_FUNC1(minus, element_func::minus)
MAP_FUNC1(cos, element_func::cos)
MAP_FUNC1(sin, element_func::sin)
MAP_FUNC1(sqrt, element_func::sqrt)

template <typename T, class Array1>
struct lazy_array_cast
{
	typedef T element_type;

	lazy_array_cast(const Array1 &a1) : a1(a1)
	{
	}

	int ndims() const
	{
		return a1.ndims();
	}

	int *shape() const
	{
		return a1.shape();
	}
	
	int size() const
	{
		return a1.size();
	}

	T at(int index0) const
	{
		return static_cast<T>(a1.at(index0));
	}

private:
	const Array1 &a1;
};

template <typename T, class Array1>
lazy_array_cast<T, Array1> array_cast(const Array1 &a1)
{
	return lazy_array_cast<T, Array1>(a1);
}

// Wake up lazy array
template <typename T, class LazyArray>
void assign(array_t<T> &dst, const LazyArray &lazy_array)
{
	// dst.setSize(lazy_array.size());
	dst.setSize(lazy_array.ndims(), lazy_array.shape());

	for (int i = 0; i < lazy_array.size(); i++)
		dst.at(i) = lazy_array.at(i);
}

} // namespace np

#endif // LAZY_ARRAY_H_