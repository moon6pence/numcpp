#include <stdio.h>
#include <gtest/gtest.h>

TEST(Main, HelloGoogleTest)
{
	EXPECT_EQ(1 + 1, 2);
}

#include <numcpp/array.h>

namespace np = numcpp;

TEST(ArrayType, DeclareEmptyArray)
{
	np::array_t<int> a1;
	np::array_t<int, 2> a2;
	np::array_t<int, 3> a3;
}

int main(int argc, char **argv)
{
	puts("Hello Tests!");

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}