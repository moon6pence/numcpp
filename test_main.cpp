#include <stdio.h>
#include <gtest/gtest.h>

TEST(Main, HelloGoogleTest)
{
	EXPECT_EQ(1 + 1, 2);
}

#include <numcpp/array.h>

using namespace numcpp;

TEST(ArrayType, DeclareEmptyArray)
{
	array_t<int> a1;

	ASSERT_TRUE(a1.empty());
	EXPECT_EQ(a1.size(), 0);
	EXPECT_EQ(a1.raw_ptr(), nullptr);
}

TEST(ArrayType, DeclareArrayWithSize)
{
	array_t<int> a1(5);

	ASSERT_FALSE(a1.empty());
	EXPECT_EQ(a1.size(), 5);
	EXPECT_NE(a1.raw_ptr(), nullptr);
}

TEST(ArrayType, AccessElements)
{
	array_t<int> a1(5);

	a1.at(0) = 2;
	a1.at(1) = 3;
	// a1.at(2) = 5;
	// a1.at(3) = 1;
	a1(2) = 5;
	a1(3) = 1;
	a1.at(4) = 7;

	EXPECT_EQ(a1.at(0), 2);
	EXPECT_EQ(a1.at(1), 3);
	EXPECT_EQ(a1.at(2), 5);
	EXPECT_EQ(a1.at(3), 1);
	EXPECT_EQ(a1.at(4), 7);

	int *ptr = a1.raw_ptr();
	ptr[2] = 8;

	EXPECT_EQ(a1(2), 8);

	int *ptr2 = a1;
	ptr[3] = 9;

	EXPECT_EQ(a1(3), 9);
	EXPECT_EQ(ptr, ptr2);
}

int main(int argc, char **argv)
{
	puts("Hello Tests!");

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}