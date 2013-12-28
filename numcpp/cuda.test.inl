#include <numcpp/cuda.h>
#include <iostream>

namespace {

TEST(CUDA, HelloCUDA)
{
	using namespace std;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cout << "Device count: " << deviceCount << endl;

	ASSERT_TRUE(deviceCount > 0);
}

} // anonymous namespace