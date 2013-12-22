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

	EXPECT_TRUE(a1.empty());
	EXPECT_EQ(a1.ndims(), 1);
	EXPECT_EQ(a1.size(0), 0);
	EXPECT_EQ(a1.size(), 0);
	EXPECT_EQ(a1.raw_ptr(), nullptr);
}

TEST(ArrayType, DeclareArrayWithSize)
{
	array_t<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(a1.ndims(), 1);
	EXPECT_EQ(a1.size(0), 5);
	EXPECT_EQ(a1.size(), 5);
	EXPECT_NE(a1.raw_ptr(), nullptr);

	array_t<int> a2(3, 2);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(a2.ndims(), 2);
	EXPECT_EQ(a2.size(0), 3);
	EXPECT_EQ(a2.size(1), 2);
	EXPECT_EQ(a2.size(), 3 * 2);
	EXPECT_NE(a2.raw_ptr(), nullptr);

	array_t<int> a3(2, 3, 4);

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(a3.ndims(), 3);
	EXPECT_EQ(a3.size(0), 2);
	EXPECT_EQ(a3.size(1), 3);
	EXPECT_EQ(a3.size(2), 4);
	EXPECT_EQ(a3.size(), 2 * 3 * 4);
	EXPECT_NE(a3.raw_ptr(), nullptr);
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
	// EXPECT_EQ(a1.at(3), 1);
	// EXPECT_EQ(a1.at(4), 7);
	EXPECT_EQ(a1(3), 1);
	EXPECT_EQ(a1(4), 7);

	int *ptr = a1.raw_ptr();
	ptr[2] = 8;

	EXPECT_EQ(a1(2), 8);

	int *ptr2 = a1;
	ptr[3] = 9;

	EXPECT_EQ(a1(3), 9);
	EXPECT_EQ(ptr, ptr2);

	int data2[6] = 
	{ 
		7, 2, 3, 
		4, 1, 8 
	};

	array_t<int> a2(2, 3);
	for (int i = 0; i < 6; i++)
		a2(i) = data2[i];

	EXPECT_EQ(a2(0), 7);
	EXPECT_EQ(a2(2), 3);
	EXPECT_EQ(a2(5), 8);

	EXPECT_EQ(a2.at(0, 0), 7);
	// EXPECT_EQ(a2.at(1, 1), 1);
	EXPECT_EQ(a2(1, 1), 1);
	EXPECT_EQ(a2.at(1, 2), 8);

	a2.at(0, 2) = 5;
	EXPECT_EQ(a2.at(0, 2), 5);

	int data3[24] = 
	{
		32, 19, 22, 10, 
		81, 42, 71, 86, 
		44, 66, 77, 88, 

		98, 76, 54, 32, 
		15, 16, 17, 18, 
		21, 22, 23, 24
	};	

	array_t<int> a3(2, 3, 4);
	for (int i = 0; i < 24; i++)
		a3(i) = data3[i];

	EXPECT_EQ(a3.at(0), 32);
	EXPECT_EQ(a3(15), 32);
	EXPECT_EQ(a3(23), 24);

	EXPECT_EQ(a3.at(0, 0, 0), 32);
	// EXPECT_EQ(a3.at(0, 1, 2), 71);
	EXPECT_EQ(a3(0, 1, 2), 71);
	EXPECT_EQ(a3.at(1, 2, 3), 24);

	a3.at(0, 2, 3) = 99;
	EXPECT_EQ(a3.at(0, 2, 3), 99);
}

int main(int argc, char **argv)
{
	puts("Hello Tests!");

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}