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

class ArrayTypeF : public ::testing::Test
{
public:
	ArrayTypeF() : a1(5), a2(2, 3), a3(2, 3, 4)
	{
	}

protected:
	virtual void SetUp()
	{
		memcpy(a1.raw_ptr(), data1, 5 * sizeof(int));
		memcpy(a2.raw_ptr(), data2, 6 * sizeof(int));
		memcpy(a3.raw_ptr(), data3, 24 * sizeof(int));
	}

	virtual void TearDown() {}

	array_t<int> a1, a2, a3;

	const int data1[5] = { 2, 3, 5, 1, 7 };

	const int data2[6] = 
	{ 
		7, 2, 3, 
		4, 1, 8 
	};

	const int data3[24] = 
	{
		32, 19, 22, 10, 
		81, 42, 71, 86, 
		44, 66, 77, 88, 

		98, 76, 54, 32, 
		15, 16, 17, 18, 
		21, 22, 23, 24
	};	
};

TEST_F(ArrayTypeF, AccessElements)
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

#include <numcpp/array_function.h>

TEST_F(ArrayTypeF, DenseIterator)
{
	const int *p1 = data1;
	for (auto i = begin(a1); i != end(a1); ++i)
		EXPECT_EQ(*i, *p1++);

	const int *p2 = data2;
	for (auto i = begin(a2); i != end(a2); ++i)
		EXPECT_EQ(*i, *p2++);

	const int *p3 = data3;
	for (auto i = begin(a3); i != end(a3); ++i)
		EXPECT_EQ(*i, *p3++);
}

TEST_F(ArrayTypeF, ForEach)
{
	using namespace std;

	for_each(a1, [](int value) { cout << value << " "; });
	cout << endl;
}

TEST_F(ArrayTypeF, Fill)
{
	fill(a1, 927);
	for (auto i = begin(a1); i != end(a1); ++i)
		EXPECT_EQ(*i, 927);

	fill(a2, 264);
	for (auto i = begin(a2); i != end(a2); ++i)
		EXPECT_EQ(*i, 264);

	fill(a3, 4979);
	for (auto i = begin(a3); i != end(a3); ++i)
		EXPECT_EQ(*i, 4979);
}

int main(int argc, char **argv)
{
	puts("Hello Tests!");

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}