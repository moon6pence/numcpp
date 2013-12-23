#include <numcpp/functions.h>

namespace {

using namespace numcpp;

TEST(Functions, Colon)
{
	auto a1 = colon(1, 10);
	int value1 = 1;
	for (int i = 0; i < 10; i++, value1++)
		EXPECT_EQ(a1(i), value1);

	auto a2 = colon(1.5, 10.0);
	double value2 = 1.5;
	for (int i = 0; i < 9; i++, value2 += 1)
		EXPECT_EQ(a2(i), value2);

	auto a3 = colon(1, 2, 10);
	int value3 = 1;
	for (int i = 0; i < 5; i++, value3 += 2)
		EXPECT_EQ(a3(i), value3);

	auto a4 = colon(1.0, 1.5, 10.0);
	EXPECT_EQ(a4.size(0), 7);
	double value4 = 1.0;
	for (int i = 0; i < 7; i++, value4 += 1.5)
		EXPECT_EQ(a4(i), value4);

	auto a5 = colon(1.0, 1.5, 9.9);
	EXPECT_EQ(a5.size(0), 6);
	double value5 = 1.0;
	for (int i = 0; i < 6; i++, value5 += 1.5)
		EXPECT_EQ(a5(i), value5);
}

TEST(Functions, MeshGrid)
{
	array_t<int> xgv = colon(1, 5), ygv = colon(2, 6);
	array_t<int> X(5, 5), Y(5, 5);

	meshgrid(X, Y, xgv, ygv);

	for (int y = 0; y < X.size(0); y++)
		for (int x = 0; x < X.size(1); x++)
		{
			ASSERT_EQ(X(x, y), xgv(x));
			ASSERT_EQ(Y(x, y), ygv(y));
		}
}

} // anonymous namespace