#include "lazy_array.h"

TEST(LazyArray, AddArrays)
{
	array_t<int> a1(5), a2(5), result;
	a1(0) = 1; a1(1) = 2; a1(2) = 3; a1(3) = 4; a1(4) = 5;
	a2(0) = 3; a2(1) = 3; a2(2) = 3; a2(3) = 3; a2(4) = 3;

	result = add(a1, a2);

	ASSERT_EQ(5, result.size(0));
	for (int i = 0; i < result.size(0); i++)
		EXPECT_EQ(a1(i) + a2(i), result(i));
}

TEST(LazyArray, AddConstantToArray)
{
	array_t<int> a1(5), result;
	a1(0) = 1; a1(1) = 2; a1(2) = 3; a1(3) = 4; a1(4) = 5;

	result = add(a1, 3);

	ASSERT_EQ(5, result.size(0));
	for (int i = 0; i < result.size(0); i++)
		EXPECT_EQ(a1(i) + 3, result(i));
}

TEST(LazyArray, InvertArraySign)
{
	array_t<int> a1(5), result;
	a1(0) = 1; a1(1) = 2; a1(2) = 3; a1(3) = 4; a1(4) = 5;

	result = minus(a1);

	ASSERT_EQ(5, result.size(0));
	for (int i = 0; i < result.size(0); i++)
		EXPECT_EQ(-a1(i), result(i));
}

TEST(LazyArray, ArrayExpression)
{
	array_t<int> a1(5), a2(5), result;
	a1(0) = 1; a1(1) = 2; a1(2) = 3; a1(3) = 4; a1(4) = 5;
	a2(0) = 3; a2(1) = 3; a2(2) = 3; a2(3) = 3; a2(4) = 3;

	result = add(a1, minus(a2));

	ASSERT_EQ(5, result.size(0));
	for (int i = 0; i < result.size(0); i++)
		EXPECT_EQ(a1(i) - a2(i), result(i));
}

TEST(LazyArray, ArrayCast)
{
	array_t<float> a1(5);
	array_t<int> result;
	a1(0) = 1.5; a1(1) = 2.5; a1(2) = 3.5; a1(3) = 4.5; a1(4) = 5.5;

	result = array_cast<int>(a1);

	ASSERT_EQ(5, result.size(0));
	for (int i = 0; i < result.size(0); i++)
		EXPECT_EQ((int)a1(i), result(i));
}

template <typename T>
array_t<T> add_without_lazy(const array_t<T> &a1, const array_t<T> &a2)
{
	// TODO: assert shape
	assert(a1.size() == a2.size());

	array_t<T> result(a1.ndims(), a1.shape());
	transform(result, a1, a2, [](T _a1, T _a2) -> T { return _a1 + _a2; });
	return std::move(result);
}

TEST(LazyArray, Performance)
{
	// const int N = 10000000;
	const int N = 1000;

	array_t<int> a1(N);
	fill(a1, 1);

	puts("Start without lazy array");
	for (int i = 0; i < 10; i++)
	{
		auto result = add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, 
			add_without_lazy(a1, a1)))))))));

		ASSERT_EQ(10, result(0));
		ASSERT_EQ(10, result(N - 1));
	}
	puts("End without lazy array");

	puts("Start with lazy array");
	for (int i = 0; i < 10; i++)
	{
		array_t<int> result;
		result = add(a1, 
			add(a1, 
			add(a1, 
			add(a1, 
			add(a1, 
			add(a1, 
			add(a1, 
			add(a1, 
			add(a1, a1)))))))));

		ASSERT_EQ(10, result(0));
		ASSERT_EQ(10, result(N - 1));
	}
	puts("End with lazy array");
}