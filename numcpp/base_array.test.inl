#include <numcpp/base_array.h>

namespace {

using namespace np;

TEST(BaseArray, DeclareEmptyArray)
{
	base_array_t a0(sizeof(int));

	EXPECT_EQ(sizeof(int), a0.itemSize());
	EXPECT_EQ(0, a0.ndims());
	EXPECT_EQ(nullptr, a0.raw_ptr());
}

TEST(BaseArray, SetSize)
{
	base_array_t a1(sizeof(int), tuple(5));

	EXPECT_EQ(sizeof(int), a1.itemSize());
	EXPECT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

	// Do not allocate again
	void *ptr = a1.raw_ptr();

	if (a1.size() != tuple(5))
		a1 = base_array_t(sizeof(int), tuple(5));
	EXPECT_EQ(ptr, a1.raw_ptr());

	base_array_t a2(sizeof(float), tuple(2, 3));

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(sizeof(float), a2.itemSize());
	EXPECT_EQ(2, a2.ndims());
	EXPECT_EQ(2, a2.size(0));
	EXPECT_EQ(3, a2.size(1));
	EXPECT_NE(a2.raw_ptr(), nullptr);

	base_array_t a3(sizeof(double), tuple(2, 3, 4));

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(sizeof(double), a3.itemSize());
	EXPECT_EQ(3, a3.ndims());
	EXPECT_EQ(2, a3.size(0));
	EXPECT_EQ(3, a3.size(1));
	EXPECT_EQ(4, a3.size(2));
	EXPECT_NE(nullptr, a3.raw_ptr());

	int shape[4] = { 2, 2, 2, 2 };
	base_array_t a4(sizeof(char), tuple(4, shape));

	EXPECT_FALSE(a4.empty());
	EXPECT_EQ(sizeof(char), a4.itemSize());
	EXPECT_EQ(4, a4.ndims());
	EXPECT_EQ(2, a4.size(0));
	EXPECT_EQ(2, a4.size(1));
	EXPECT_EQ(2, a4.size(2));
	EXPECT_EQ(2, a4.size(3));
	EXPECT_NE(nullptr, a4.raw_ptr());
}

} // anonymous namespace