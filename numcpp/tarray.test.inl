#include <numcpp/tarray.h>

namespace {

using namespace np;

TEST(TArray, DeclareEmptyArray)
{
	TArray<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(sizeof(int), a0.itemSize());
	EXPECT_EQ(1, a0.ndims());
	EXPECT_EQ(nullptr, a0.raw_ptr());
}

TEST(TArray, DeclareArrayWithSize)
{
	TArray<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(sizeof(int), a1.itemSize());
	ASSERT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

//	Array<float> a2(3, 2);
//
//	EXPECT_FALSE(a2.empty());
//	EXPECT_EQ(sizeof(float), a2.itemSize());
//	ASSERT_EQ(2, a2.ndims());
//	EXPECT_EQ(3, a2.size(0));
//	EXPECT_EQ(2, a2.size(1));
//	EXPECT_NE(nullptr, a2.raw_ptr());
//
//	int shape[3] = { 4, 3, 2 };
//	Array<double> a3(tuple(3, shape));
//
//	EXPECT_FALSE(a3.empty());
//	EXPECT_EQ(sizeof(double), a3.itemSize());
//	ASSERT_EQ(3, a3.ndims());
//	EXPECT_EQ(4, a3.size(0));
//	EXPECT_EQ(3, a3.size(1));
//	EXPECT_EQ(2, a3.size(2));
//	EXPECT_NE(nullptr, a3.raw_ptr());
}

//TEST(Array, SetSize)
//{
//	Array<int> a1;
//	a1 = Array<int>(5);
//
//	EXPECT_FALSE(a1.empty());
//	EXPECT_EQ(sizeof(int), a1.itemSize());
//	ASSERT_EQ(1, a1.ndims());
//	EXPECT_EQ(5, a1.size(0));
//	EXPECT_NE(nullptr, a1.raw_ptr());
//
//	Array<int> a1_same(5);
//	int *ptr = a1_same.raw_ptr();
//
//	// Do not allocate again
//	if (a1_same.size() != tuple(5))
//		a1_same = Array<int>(5);
//	EXPECT_EQ(ptr, a1_same.raw_ptr());
//
//	Array<int> a2;
//	a2 = Array<int>(3, 2);
//
//	EXPECT_FALSE(a2.empty());
//	EXPECT_EQ(sizeof(float), a2.itemSize());
//	ASSERT_EQ(2, a2.ndims());
//	EXPECT_EQ(3, a2.size(0));
//	EXPECT_EQ(2, a2.size(1));
//	EXPECT_NE(nullptr, a2.raw_ptr());
//
//	Array<double> a3;
//	int shape[3] = { 4, 3, 2 };
//	a3 = Array<int>(tuple(3, shape));
//
//	EXPECT_FALSE(a3.empty());
//	EXPECT_EQ(sizeof(double), a3.itemSize());
//	ASSERT_EQ(3, a3.ndims());
//	EXPECT_EQ(4, a3.size(0));
//	EXPECT_EQ(3, a3.size(1));
//	EXPECT_EQ(2, a3.size(2));
//	EXPECT_NE(nullptr, a3.raw_ptr());
//}
//
//TEST(Array, MoveSemantics)
//{
//	// move constructor
//	Array<int> a1(5);
//	auto moved = std::move(a1);
//
//	EXPECT_TRUE(a1.empty());
//	EXPECT_EQ(0, a1.ndims());
//	EXPECT_EQ(nullptr, a1.raw_ptr());
//
//	EXPECT_FALSE(moved.empty());
//	ASSERT_EQ(1, moved.ndims());
//	EXPECT_EQ(5, moved.size(0));
//	EXPECT_NE(nullptr, moved.raw_ptr());
//
//	// move assign
//	Array<int> a2(3, 2), moved2;
//	moved2 = std::move(a2);
//
//	EXPECT_TRUE(a2.empty());
//	EXPECT_EQ(0, a2.ndims());
//	EXPECT_EQ(nullptr, a2.raw_ptr());
//
//	EXPECT_FALSE(moved2.empty());
//	ASSERT_EQ(2, moved2.ndims());
//	EXPECT_EQ(3, moved2.size(0));
//	EXPECT_EQ(2, moved2.size(1));
//	EXPECT_NE(nullptr, moved2.raw_ptr());
//}
//
//TEST(Array, MoveFromBaseArray)
//{
//	// move constructor
//	{
//		BaseArray base1(sizeof(int), tuple(5));
//
//		EXPECT_FALSE(base1.empty());
//		EXPECT_EQ(1, base1.ndims());
//		EXPECT_NE(nullptr, base1.raw_ptr());
//
//		Array<int> a1(std::move(base1));
//
//		EXPECT_TRUE(base1.empty());
//		EXPECT_EQ(0, base1.ndims());
//		EXPECT_EQ(nullptr, base1.raw_ptr());
//
//		EXPECT_FALSE(a1.empty());
//		ASSERT_EQ(1, a1.ndims());
//		EXPECT_EQ(5, a1.size(0));
//		EXPECT_NE(nullptr, a1.raw_ptr());
//	}
//
//	// move assign
//	{
//		BaseArray base2(sizeof(float), tuple(3, 2));
//
//		Array<int> a2(5);
//		a2 = std::move(base2);
//
//		EXPECT_TRUE(base2.empty());
//		EXPECT_EQ(0, base2.ndims());
//		EXPECT_EQ(nullptr, base2.raw_ptr());
//
//		EXPECT_FALSE(a2.empty());
//		ASSERT_EQ(2, a2.ndims());
//		EXPECT_EQ(3, a2.size(0));
//		EXPECT_EQ(2, a2.size(1));
//		EXPECT_NE(nullptr, a2.raw_ptr());
//	}
//}

TEST(TArray, DerivedFunctions)
{
	TArray<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(0, a0.length());
	EXPECT_EQ(0 * sizeof(int), a0.byteSize());

	TArray<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(5, a1.length());
	EXPECT_EQ(5 * sizeof(int), a1.byteSize());
	EXPECT_EQ(sizeof(int), a1.stride(0));

//	Array<float> a2(3, 2);
//
//	EXPECT_FALSE(a2.empty());
//	EXPECT_EQ(3 * 2, a2.length());
//	EXPECT_EQ(3 * 2 * sizeof(float), a2.byteSize());
//	EXPECT_EQ(sizeof(float), a2.stride(0));
//	EXPECT_EQ(sizeof(float) * 3, a2.stride(1));
//
//	int shape[3] = { 4, 3, 2 };
//	Array<char> a3(tuple(3, shape));
//
//	EXPECT_FALSE(a3.empty());
//	EXPECT_EQ(4 * 3 * 2, a3.length());
//	EXPECT_EQ(4 * 3 * 2 * sizeof(char), a3.byteSize());
//	EXPECT_EQ(sizeof(char), a3.stride(0));
//	EXPECT_EQ(sizeof(char) * 4, a3.stride(1));
//	EXPECT_EQ(sizeof(char) * 4 * 3, a3.stride(2));
}

//TEST(Array, AccessElements)
//{
//	// 1d array
//	np::Array<int> a1(5);
//
//	const int data1[5] = 
//	{ 
//		2, 3, 5, 1, 7 
//	};
//
//	memcpy(a1.raw_ptr(), data1, 5 * sizeof(int));
//
//	EXPECT_EQ(2, a1.at(0));
//	EXPECT_EQ(3, a1.at(1));
//	EXPECT_EQ(5, a1.at(2));
//	EXPECT_EQ(1, a1(3));
//	EXPECT_EQ(7, a1(4));
//
//	int *ptr = a1.raw_ptr();
//	ptr[2] = 8;
//
//	EXPECT_EQ(8, a1(2));
//
//	int *ptr2 = a1;
//	ptr[3] = 9;
//
//	EXPECT_EQ(9, a1(3));
//	EXPECT_EQ(ptr, ptr2);
//
//	// 2d array
//	np::Array<int> a2(3, 2);
//
//	const int data2[6] = 
//	{ 
//		7, 2, 3, 
//		4, 1, 8 
//	};
//
//	memcpy(a2.raw_ptr(), data2, 6 * sizeof(int));
//
//	EXPECT_EQ(7, a2(0));
//	EXPECT_EQ(3, a2(2));
//	EXPECT_EQ(8, a2(5));
//
//	EXPECT_EQ(7, a2.at(0, 0));
//	EXPECT_EQ(1, a2(1, 1));
//	EXPECT_EQ(8, a2.at(2, 1));
//
//	a2.at(2, 0) = 5;
//	EXPECT_EQ(5, a2.at(2, 0));
//}
//
//TEST(Array, Slice)
//{
//	// 1d array
//	np::Array<int> a1(5);
//
//	const int data1[5] = 
//	{ 
//		2, 3, 5, 1, 7 
//	};
//
//	memcpy(a1.raw_ptr(), data1, 5 * sizeof(int));
//
//	auto slice1 = a1.slice(1, 4);
//
//	EXPECT_FALSE(slice1.empty());
//	EXPECT_EQ(3, slice1.size(0));
//
//	EXPECT_EQ(3, slice1(0));
//	EXPECT_EQ(5, slice1(1));
//	EXPECT_EQ(1, slice1(2));
//
//	// 2d array
//	np::Array<int> a2(3, 2);
//
//	const int data2[6] = 
//	{ 
//		7, 2, 3, 
//		4, 1, 8 
//	};
//
//	memcpy(a2.raw_ptr(), data2, 6 * sizeof(int));
//
//	auto slice2 = a2.slice(1, 0, 3, 2);
//
//	EXPECT_EQ(2, slice2.size(0));
//	EXPECT_EQ(2, slice2.size(1));
//	EXPECT_EQ(a2.stride(0), slice2.stride(0));
//	EXPECT_EQ(a2.stride(1), slice2.stride(1));
//
//	EXPECT_EQ(2, slice2(0, 0));
//	EXPECT_EQ(3, slice2(1, 0));
//	EXPECT_EQ(1, slice2(0, 1));
//	EXPECT_EQ(8, slice2(1, 1));
//}

} // anonymous namespace