#ifndef NUMCPP_TEST_ARRAYFIXTURE_H_
#define NUMCPP_TEST_ARRAYFIXTURE_H_

#include <gtest/gtest.h>
#include <numcpp/array.h>

class ArrayFixture : public ::testing::Test
{
public:
	ArrayFixture() : a1(5), a2(2, 3), a3(2, 3, 4)
	{
	}

protected:
	virtual void SetUp()
	{
		const int data1[5] = 
		{ 
			2, 3, 5, 1, 7 
		};

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

		memcpy(a1.raw_ptr(), data1, 5 * sizeof(int));
		memcpy(a2.raw_ptr(), data2, 6 * sizeof(int));
		memcpy(a3.raw_ptr(), data3, 24 * sizeof(int));

		memcpy(this->data1, data1, 5 * sizeof(int));
		memcpy(this->data2, data2, 6 * sizeof(int));
		memcpy(this->data3, data3, 24 * sizeof(int));
	}

	virtual void TearDown() {}

	np::array_t<int> a1, a2, a3;
	int data1[5], data2[6], data3[24];
};

#endif // NUMCPP_TEST_ARRAYFIXTURE_H_