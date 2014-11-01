#include <numcpp/base_array.h>

namespace {

using namespace np;

TEST(BaseArray, DeclareEmptyArray)
{
	BaseArray a0;

	EXPECT_EQ(1, a0.itemSize());
	EXPECT_EQ(0, a0.length());
	EXPECT_EQ(0, a0.ndims());
	EXPECT_EQ(BaseArray::size_type(), a0.size());
	EXPECT_EQ(BaseArray::stride_type(), a0.strides());
	EXPECT_EQ(nullptr, a0.raw_ptr());

	EXPECT_TRUE(empty(a0));
	EXPECT_EQ(0, byteSize(a0));
}

TEST(BaseArray, DeclareArrayWithSize)
{
	BaseArray a1(sizeof(int), make_vector(5));

	EXPECT_EQ(sizeof(int), a1.itemSize());
	EXPECT_EQ(5, a1.length());
	EXPECT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_EQ(1, a1.stride(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

	EXPECT_FALSE(empty(a1));
	EXPECT_EQ(5 * sizeof(int), byteSize(a1));

	BaseArray a2(sizeof(float), make_vector(2, 3));

	EXPECT_EQ(sizeof(float), a2.itemSize());
	EXPECT_EQ(2 * 3, a2.length());
	EXPECT_EQ(2, a2.ndims());
	EXPECT_EQ(2, a2.size(0));
	EXPECT_EQ(3, a2.size(1));
	EXPECT_EQ(1, a2.stride(0));
	EXPECT_EQ(2, a2.stride(1));
	EXPECT_NE(nullptr, a2.raw_ptr());

	EXPECT_FALSE(empty(a2));
	EXPECT_EQ(2 * 3 * sizeof(int), byteSize(a2));

	BaseArray a3(sizeof(double), make_vector(2, 3, 4));

	EXPECT_EQ(sizeof(double), a3.itemSize());
	EXPECT_EQ(2 * 3 * 4, a3.length());
	EXPECT_EQ(3, a3.ndims());
	EXPECT_EQ(2, a3.size(0));
	EXPECT_EQ(3, a3.size(1));
	EXPECT_EQ(4, a3.size(2));
	EXPECT_EQ(1, a3.stride(0));
	EXPECT_EQ(2, a3.stride(1));
	EXPECT_EQ(6, a3.stride(2));
	EXPECT_NE(nullptr, a3.raw_ptr());

	EXPECT_FALSE(empty(a3));
	EXPECT_EQ(2 * 3 * 4 * sizeof(double), byteSize(a3));
}

} // anonymous namespace