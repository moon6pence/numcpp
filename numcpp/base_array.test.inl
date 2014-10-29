#include <numcpp/base_array.h>

namespace {

using namespace np;

TEST(BaseArray, DeclareEmptyArray)
{
	BaseArray a0(sizeof(int));

	EXPECT_EQ(sizeof(int), a0.itemSize());
	EXPECT_EQ(0, a0.ndims());
	EXPECT_EQ(nullptr, a0.raw_ptr());
}

TEST(BaseArray, SetSize)
{
	BaseArray a1(sizeof(int), make_vector(5));

	EXPECT_EQ(sizeof(int), a1.itemSize());
	EXPECT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

	// Do not allocate again
	void *ptr = a1.raw_ptr();

	if (a1.size() != make_vector(5))
		a1 = BaseArray(sizeof(int), make_vector(5));
	EXPECT_EQ(ptr, a1.raw_ptr());

	BaseArray a2(sizeof(float), make_vector(2, 3));

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(sizeof(float), a2.itemSize());
	EXPECT_EQ(2, a2.ndims());
	EXPECT_EQ(2, a2.size(0));
	EXPECT_EQ(3, a2.size(1));
	EXPECT_NE(a2.raw_ptr(), nullptr);

	BaseArray a3(sizeof(double), make_vector(2, 3, 4));

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(sizeof(double), a3.itemSize());
	EXPECT_EQ(3, a3.ndims());
	EXPECT_EQ(2, a3.size(0));
	EXPECT_EQ(3, a3.size(1));
	EXPECT_EQ(4, a3.size(2));
	EXPECT_NE(nullptr, a3.raw_ptr());
}

} // anonymous namespace