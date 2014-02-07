#include <numcpp/array.h>

namespace {

using namespace np;

TEST(Array, DeclareEmptyArray)
{
	array_t<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(sizeof(int), a0.itemSize());
	EXPECT_EQ(0, a0.ndims());
	EXPECT_EQ(nullptr, a0.raw_ptr());
}

TEST(Array, DeclareArrayWithSize)
{
	array_t<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(sizeof(int), a1.itemSize());
	ASSERT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

	array_t<float> a2(3, 2);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(sizeof(float), a2.itemSize());
	ASSERT_EQ(2, a2.ndims());
	EXPECT_EQ(3, a2.size(0));
	EXPECT_EQ(2, a2.size(1));
	EXPECT_NE(nullptr, a2.raw_ptr());

	array_t<double> a3(4, 3, 2);

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(sizeof(double), a3.itemSize());
	ASSERT_EQ(3, a3.ndims());
	EXPECT_EQ(4, a3.size(0));
	EXPECT_EQ(3, a3.size(1));
	EXPECT_EQ(2, a3.size(2));
	EXPECT_NE(nullptr, a3.raw_ptr());

	int shape[4] = { 2, 2, 2, 2 };
	array_t<int> a4(4, shape);
	
	EXPECT_FALSE(a4.empty());
	ASSERT_EQ(4, a4.ndims());
	EXPECT_EQ(2, a4.size(0));
	EXPECT_EQ(2, a4.size(1));
	EXPECT_EQ(2, a4.size(2));
	EXPECT_EQ(2, a4.size(3));
	EXPECT_NE(nullptr, a4.raw_ptr());
}

TEST(Array, SetSize)
{
	array_t<int> a1;
	a1.setSize(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(sizeof(int), a1.itemSize());
	ASSERT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

	array_t<int> a1_same(5);
	int *ptr = a1_same.raw_ptr();

	// Do not allocate again
	a1_same.setSize(5);
	EXPECT_EQ(ptr, a1_same.raw_ptr());

	array_t<int> a2;
	a2.setSize(3, 2);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(sizeof(float), a2.itemSize());
	ASSERT_EQ(2, a2.ndims());
	EXPECT_EQ(3, a2.size(0));
	EXPECT_EQ(2, a2.size(1));
	EXPECT_NE(nullptr, a2.raw_ptr());

	array_t<double> a3;
	a3.setSize(4, 3, 2);

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(sizeof(double), a3.itemSize());
	ASSERT_EQ(3, a3.ndims());
	EXPECT_EQ(4, a3.size(0));
	EXPECT_EQ(3, a3.size(1));
	EXPECT_EQ(2, a3.size(2));
	EXPECT_NE(nullptr, a3.raw_ptr());

	array_t<int> a4;
	int shape[4] = { 2, 2, 2, 2 };
	a4.setSize(4, shape);

	EXPECT_FALSE(a4.empty());
	ASSERT_EQ(4, a4.ndims());
	EXPECT_EQ(2, a4.size(0));
	EXPECT_EQ(2, a4.size(1));
	EXPECT_EQ(2, a4.size(2));
	EXPECT_EQ(2, a4.size(3));
	EXPECT_NE(nullptr, a4.raw_ptr());
}

TEST(Array, MoveSemantics)
{
	// move constructor
	array_t<int> a1(5);
	auto moved = std::move(a1);

	EXPECT_TRUE(a1.empty());
	EXPECT_EQ(0, a1.ndims());
	EXPECT_EQ(nullptr, a1.raw_ptr());

	EXPECT_FALSE(moved.empty());
	ASSERT_EQ(1, moved.ndims());
	EXPECT_EQ(5, moved.size(0));
	EXPECT_NE(nullptr, moved.raw_ptr());

	// move assign
	array_t<int> a2(3, 2), moved2;
	moved2 = std::move(a2);

	EXPECT_TRUE(a2.empty());
	EXPECT_EQ(0, a2.ndims());
	EXPECT_EQ(nullptr, a2.raw_ptr());

	EXPECT_FALSE(moved2.empty());
	ASSERT_EQ(2, moved2.ndims());
	EXPECT_EQ(3, moved2.size(0));
	EXPECT_EQ(2, moved2.size(1));
	EXPECT_NE(nullptr, moved2.raw_ptr());
}

TEST(Array, MoveFromBaseArray)
{
	base_array_t base_a1(sizeof(int));
	base_a1.setSize<heap_allocator>(5);

	array_t<int> a1(std::move(base_a1));

	EXPECT_TRUE(base_a1.empty());
	EXPECT_EQ(0, base_a1.ndims());
	EXPECT_EQ(nullptr, base_a1.raw_ptr());

	EXPECT_FALSE(a1.empty());
	ASSERT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());
}

TEST(Array, DerivedFunctions)
{
	array_t<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(1, a0.size()); // Be careful, not 0
	EXPECT_EQ(1 * sizeof(int), a0.byteSize());

	array_t<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(5, a1.size());
	EXPECT_EQ(5 * sizeof(int), a1.byteSize());
	EXPECT_EQ(sizeof(int), a1.stride(0));

	array_t<float> a2(3, 2);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(3 * 2, a2.size());
	EXPECT_EQ(3 * 2 * sizeof(float), a2.byteSize());
	EXPECT_EQ(sizeof(float), a2.stride(0));
	EXPECT_EQ(sizeof(float) * 3, a2.stride(1));

	array_t<char> a3(4, 3, 2);

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(4 * 3 * 2, a3.size());
	EXPECT_EQ(4 * 3 * 2 * sizeof(char), a3.byteSize());
	EXPECT_EQ(sizeof(char), a3.stride(0));
	EXPECT_EQ(sizeof(char) * 4, a3.stride(1));
	EXPECT_EQ(sizeof(char) * 4 * 3, a3.stride(2));
}

TEST(Array, AccessElements)
{
	// 1d array
	np::array_t<int> a1(5);

	const int data1[5] = 
	{ 
		2, 3, 5, 1, 7 
	};

	memcpy(a1.raw_ptr(), data1, 5 * sizeof(int));

	EXPECT_EQ(2, a1.at(0));
	EXPECT_EQ(3, a1.at(1));
	EXPECT_EQ(5, a1.at(2));
	EXPECT_EQ(1, a1(3));
	EXPECT_EQ(7, a1(4));

	int *ptr = a1.raw_ptr();
	ptr[2] = 8;

	EXPECT_EQ(8, a1(2));

	int *ptr2 = a1;
	ptr[3] = 9;

	EXPECT_EQ(9, a1(3));
	EXPECT_EQ(ptr, ptr2);

	// 2d array
	np::array_t<int> a2(3, 2);

	const int data2[6] = 
	{ 
		7, 2, 3, 
		4, 1, 8 
	};

	memcpy(a2.raw_ptr(), data2, 6 * sizeof(int));

	EXPECT_EQ(7, a2(0));
	EXPECT_EQ(3, a2(2));
	EXPECT_EQ(8, a2(5));

	EXPECT_EQ(7, a2.at(0, 0));
	EXPECT_EQ(1, a2(1, 1));
	EXPECT_EQ(8, a2.at(2, 1));

	a2.at(2, 0) = 5;
	EXPECT_EQ(5, a2.at(2, 0));

	// 3d array
	np::array_t<int> a3(4, 3, 2);

	const int data3[24] = 
	{
		32, 19, 22, 10, 
		81, 42, 71, 86, 
		44, 66, 77, 88, 

		98, 76, 54, 32, 
		15, 16, 17, 18, 
		21, 22, 23, 24
	};	

	memcpy(a3.raw_ptr(), data3, 24 * sizeof(int));

	EXPECT_EQ(32, a3.at(0));
	EXPECT_EQ(32, a3(15));
	EXPECT_EQ(24, a3(23));

	EXPECT_EQ(32, a3.at(0, 0, 0));
	EXPECT_EQ(71, a3(2, 1, 0));
	EXPECT_EQ(24, a3.at(3, 2, 1));

	a3.at(3, 2, 0) = 99;
	EXPECT_EQ(99, a3.at(3, 2, 0));
}

TEST(Array, Slice)
{
	// 1d array
	np::array_t<int> a1(5);

	const int data1[5] = 
	{ 
		2, 3, 5, 1, 7 
	};

	memcpy(a1.raw_ptr(), data1, 5 * sizeof(int));

	auto slice1 = a1.slice(1, 4);

	EXPECT_FALSE(slice1.empty());
	EXPECT_EQ(3, slice1.size(0));

	EXPECT_EQ(3, slice1(0));
	EXPECT_EQ(5, slice1(1));
	EXPECT_EQ(1, slice1(2));

	// 2d array
	np::array_t<int> a2(3, 2);

	const int data2[6] = 
	{ 
		7, 2, 3, 
		4, 1, 8 
	};

	memcpy(a2.raw_ptr(), data2, 6 * sizeof(int));

	auto slice2 = a2.slice(1, 0, 3, 2);

	EXPECT_EQ(2, slice2.size(0));
	EXPECT_EQ(2, slice2.size(1));
	EXPECT_EQ(a2.stride(0), slice2.stride(0));
	EXPECT_EQ(a2.stride(1), slice2.stride(1));

	EXPECT_EQ(2, slice2(0, 0));
	EXPECT_EQ(3, slice2(1, 0));
	EXPECT_EQ(1, slice2(0, 1));
	EXPECT_EQ(8, slice2(1, 1));
}

} // anonymous namespace