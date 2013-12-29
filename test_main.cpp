#include <stdio.h>
#include <gtest/gtest.h>

TEST(Main, HelloGoogleTest)
{
	EXPECT_EQ(1 + 1, 2);
}

#include "config.h"

#include "numcpp/array.test.inl"
#include "numcpp/stl.test.inl"
#include "numcpp/functions.test.inl"

#ifdef USE_OPENCV
#include "numcpp/opencv.test.inl"
#endif

#ifdef USE_CUDA
#include "numcpp/cuda.test.inl"
#endif

int main(int argc, char **argv)
{
	puts("Hello Tests!");

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}