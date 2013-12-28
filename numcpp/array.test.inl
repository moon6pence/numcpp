#include <numcpp/array.h>
#include "test_arrayfixture.h"

namespace {

using namespace numcpp;

TEST(ArrayType, DeclareEmptyArray)
{
	array_t<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(a0.ndims(), 1);
	EXPECT_EQ(a0.size(0), 0);
	EXPECT_EQ(a0.size(), 0);
	EXPECT_EQ(a0.raw_ptr(), nullptr);
}

TEST(ArrayType, DeclareArrayWithSize)
{
	array_t<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(a1.ndims(), 1);
	EXPECT_EQ(a1.size(0), 5);
	EXPECT_EQ(a1.size(), 5);
	EXPECT_NE(a1.raw_ptr(), nullptr);

	array_t<int> a2(2, 3);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(a2.ndims(), 2);
	EXPECT_EQ(a2.size(0), 2);
	EXPECT_EQ(a2.size(1), 3);
	EXPECT_EQ(a2.size(), 2 * 3);
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

TEST_F(ArrayFixture, AccessElements)
{
	// 1d array
	EXPECT_EQ(a1.at(0), 2);
	EXPECT_EQ(a1.at(1), 3);
	EXPECT_EQ(a1.at(2), 5);
	EXPECT_EQ(a1(3), 1);
	EXPECT_EQ(a1(4), 7);

	int *ptr = a1.raw_ptr();
	ptr[2] = 8;

	EXPECT_EQ(a1(2), 8);

	int *ptr2 = a1;
	ptr[3] = 9;

	EXPECT_EQ(a1(3), 9);
	EXPECT_EQ(ptr, ptr2);

	// 2d array
	EXPECT_EQ(a2(0), 7);
	EXPECT_EQ(a2(2), 3);
	EXPECT_EQ(a2(5), 8);

	EXPECT_EQ(a2.at(0, 0), 7);
	EXPECT_EQ(a2(1, 1), 1);
	EXPECT_EQ(a2.at(1, 2), 8);

	a2.at(0, 2) = 5;
	EXPECT_EQ(a2.at(0, 2), 5);

	// 3d array
	EXPECT_EQ(a3.at(0), 32);
	EXPECT_EQ(a3(15), 32);
	EXPECT_EQ(a3(23), 24);

	EXPECT_EQ(a3.at(0, 0, 0), 32);
	EXPECT_EQ(a3(0, 1, 2), 71);
	EXPECT_EQ(a3.at(1, 2, 3), 24);

	a3.at(0, 2, 3) = 99;
	EXPECT_EQ(a3.at(0, 2, 3), 99);
}

TEST_F(ArrayFixture, MoveSemantics)
{
	// move constructor
	auto moved = std::move(a1);

	EXPECT_TRUE(a1.empty());
	EXPECT_EQ(a1.ndims(), 1);
	EXPECT_EQ(a1.size(0), 0);
	EXPECT_EQ(a1.size(), 0);
	EXPECT_EQ(a1.raw_ptr(), nullptr);

	EXPECT_FALSE(moved.empty());
	EXPECT_EQ(moved.ndims(), 1);
	EXPECT_EQ(moved.size(0), 5);
	EXPECT_EQ(moved.size(), 5);
	EXPECT_NE(moved.raw_ptr(), nullptr);

	// move assign
	array_t<int> moved2;
	moved2 = std::move(a2);

	EXPECT_TRUE(a2.empty());
	EXPECT_EQ(a2.ndims(), 1);
	EXPECT_EQ(a2.size(0), 0);
	EXPECT_EQ(a2.size(), 0);
	EXPECT_EQ(a2.raw_ptr(), nullptr);

	EXPECT_FALSE(moved2.empty());
	EXPECT_EQ(moved2.ndims(), 2);
	EXPECT_EQ(moved2.size(0), 2);
	EXPECT_EQ(moved2.size(1), 3);
	EXPECT_EQ(moved2.size(), 2 * 3);
	EXPECT_NE(moved2.raw_ptr(), nullptr);
}

} // anonymous namespace