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

	ASSERT_TRUE(a1.empty());
}

TEST(ArrayType, DeclareArrayWithSize)
{
	np::array_t<int> a1(5);

	ASSERT_FALSE(a1.empty());
}

int main(int argc, char **argv)
{
	puts("Hello Tests!");

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}