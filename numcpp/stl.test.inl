#include <numcpp/stl.h>
#include "test_arrayfixture.h"

namespace {

typedef ArrayFixture NumcppSTL;

using namespace numcpp;

TEST_F(NumcppSTL, DenseIterator)
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

TEST_F(NumcppSTL, ForEach)
{
	using namespace std;

	for_each(a1, [](int value) { cout << value << " "; });
	cout << endl;

	print(a1);
	print(a2);
	print(a3);
}

TEST_F(NumcppSTL, Fill)
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

TEST_F(NumcppSTL, Transform)
{
	array_t<int> result1(a1.size(0));
	transform(result1, a1, [](int _a1) { return _a1 + 1; });

	const int *p1 = data1;
	for (auto i = begin(result1); i != end(result1); ++i)
		EXPECT_EQ(*i, *(p1++) + 1);
}

} // anonymous namespace