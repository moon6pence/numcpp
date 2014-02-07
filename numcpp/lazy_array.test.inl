#include "lazy_array.h"

TEST(LazyArray, AddArrays)
{
	array_t<int> a1(5), a2(5), result;

	a1(0) = 1; a1(1) = 2; a1(2) = 3; a1(3) = 4; a1(4) = 5;
	a2(0) = 3; a2(1) = 3; a2(2) = 3; a2(3) = 3; a2(4) = 3;

	assign(result, add<int>(a1, a2));

	ASSERT_EQ(5, result.size(0));

	for (int i = 0; i < result.size(0); i++)
		EXPECT_EQ(a1(i) + a2(i), result(i));
}
