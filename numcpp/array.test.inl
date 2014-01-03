#include <numcpp/array.h>
#include "test_arrayfixture.h"

namespace {

using namespace numcpp;

TEST(ArrayType, DeclareEmptyArray)
{
	array_t<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(0, a0.ndims());
	EXPECT_EQ(0, a0.size());
	EXPECT_EQ(nullptr, a0.raw_ptr());
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

TEST(ArrayType, SetSize)
{
	array_t<int> a1;
	a1.setSize(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(a1.ndims(), 1);
	EXPECT_EQ(a1.size(0), 5);
	EXPECT_EQ(a1.size(), 5);
	EXPECT_NE(a1.raw_ptr(), nullptr);

	array_t<int> a1_same(5);
	int *ptr = a1_same.raw_ptr();

	// Do not allocate again
	a1_same.setSize(5);
	EXPECT_EQ(ptr, a1_same.raw_ptr());

	array_t<int> a2;
	a2.setSize(2, 3);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(a2.ndims(), 2);
	EXPECT_EQ(a2.size(0), 2);
	EXPECT_EQ(a2.size(1), 3);
	EXPECT_EQ(a2.size(), 2 * 3);
	EXPECT_NE(a2.raw_ptr(), nullptr);

	array_t<int> a3;
	a3.setSize(2, 3, 4);

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(a3.ndims(), 3);
	EXPECT_EQ(a3.size(0), 2);
	EXPECT_EQ(a3.size(1), 3);
	EXPECT_EQ(a3.size(2), 4);
	EXPECT_EQ(a3.size(), 2 * 3 * 4);
	EXPECT_NE(a3.raw_ptr(), nullptr);

	array_t<int> a4;
	int shape[4] = { 2, 2, 2, 2 };
	a4.setSize(4, shape);

	EXPECT_FALSE(a4.empty());
	EXPECT_EQ(4, a4.ndims());
	EXPECT_EQ(2, a4.size(0));
	EXPECT_EQ(2, a4.size(1));
	EXPECT_EQ(2, a4.size(2));
	EXPECT_EQ(2, a4.size(3));
	EXPECT_EQ(2 * 2 * 2 * 2, a4.size());
	EXPECT_NE(nullptr, a4.raw_ptr());
}

TEST_F(ArrayFixture, MoveSemantics)
{
	// move constructor
	auto moved = std::move(a1);

	EXPECT_TRUE(a1.empty());
	EXPECT_EQ(0, a1.ndims());
	EXPECT_EQ(0, a1.size());
	EXPECT_EQ(nullptr, a1.raw_ptr());

	EXPECT_FALSE(moved.empty());
	EXPECT_EQ(1, moved.ndims());
	EXPECT_EQ(5, moved.size(0));
	EXPECT_EQ(5, moved.size());
	EXPECT_NE(nullptr, moved.raw_ptr());

	// move assign
	array_t<int> moved2;
	moved2 = std::move(a2);

	EXPECT_TRUE(a2.empty());
	EXPECT_EQ(0, a2.ndims());
	EXPECT_EQ(0, a2.size());
	EXPECT_EQ(nullptr, a2.raw_ptr());

	EXPECT_FALSE(moved2.empty());
	EXPECT_EQ(2, moved2.ndims());
	EXPECT_EQ(2, moved2.size(0));
	EXPECT_EQ(3, moved2.size(1));
	EXPECT_EQ(2 * 3, moved2.size());
	EXPECT_NE(nullptr, moved2.raw_ptr());
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

} // anonymous namespace