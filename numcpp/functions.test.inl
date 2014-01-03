#include <numcpp/functions.h>
#include <fstream>

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

TEST(Functions, Linspace)
{
	auto a1 = linspace(1, 10, 10);
	ASSERT_EQ(10, a1.size(0));

	auto a1_expect = colon(1, 10);
	for (int i = 0; i < a1.size(0); i++)
		EXPECT_EQ(a1_expect(i), a1(i));

	auto a2 = linspace(1, 10, 5);
	ASSERT_EQ(5, a2.size(0));
	for (int i = 0; i < a2.size(0); i++)
		EXPECT_EQ(1 + (10 - 1) * i / (5 - 1), a2(i));

	auto a3 = linspace(1.0, 10.0, 6 + 1);
	ASSERT_EQ(7, a3.size(0));
	for (int i = 0; i < a3.size(0); i++)
		EXPECT_EQ(1.0 + 1.5 * i, a3(i));
}

TEST(Functions, MeshGrid)
{
	array_t<int> xgv = colon(1, 5), ygv = colon(2, 10);
	array_t<int> X(5, 5), Y(5, 5);

	meshgrid(X, Y, xgv, ygv);

	for (int y = 0; y < X.size(0); y++)
		for (int x = 0; x < X.size(1); x++)
		{
			ASSERT_EQ(X(y, x), xgv(x));
			ASSERT_EQ(Y(y, x), ygv(y));
		}
}

TEST(Functions, FromFile)
{
	using namespace std;

	const char *FILENAME_NOTEXISTS = "NOTEXISTS_190283912831";
	const char *FILENAME_INT = "fromfile_int.txt";

	auto a0 = fromfile<int>(FILENAME_NOTEXISTS);
	EXPECT_TRUE(a0.empty());

	// Check files exists
	ifstream file1(FILENAME_INT);
	ASSERT_TRUE(file1.good());

	auto a1 = fromfile<int>(FILENAME_INT);
	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));

	for (auto i = begin(a1); i != end(a1); ++i)
	{
		int value;
		file1 >> value;

		EXPECT_EQ(value, *i);
	}

	file1.close();
}

} // anonymous namespace