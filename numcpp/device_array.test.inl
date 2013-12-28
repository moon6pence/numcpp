#include <numcpp/device_array.h>
#include <iostream>

// function declared in cu file
void vecAdd(const int *A, const int *B, int *C, int N);

namespace {

TEST(CUDA, HelloCUDA)
{
	using namespace std;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cout << "Device count: " << deviceCount << endl;

	ASSERT_TRUE(deviceCount > 0);
}

TEST(CUDA, RunKernel)
{
	using namespace std;

	const int N = 5;

	// Data on the host memory
	int a[N] = { 1, 2, 3, 4, 5 }, b[N] = { 3, 3, 3, 3, 3 }, c[N];

	// Print A
	for (int i = 0; i < N; i++)
		cout << a[i] << " ";
	cout << endl;

	// Print B
	for (int i = 0; i < N; i++)
		cout << b[i] << " ";
	cout << endl;

	// Data on the device memory
	int *a_d, *b_d, *c_d;

	// Allocate the device memory
	cudaMalloc((void **)&a_d, N * sizeof(int));
	cudaMalloc((void **)&b_d, N * sizeof(int));
	cudaMalloc((void **)&c_d, N * sizeof(int));
	ASSERT_NE(a_d, nullptr);
	ASSERT_NE(b_d, nullptr);
	ASSERT_NE(c_d, nullptr);

	// Copy from host to device
	cudaMemcpy(a_d, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// Run kernel
	vecAdd(a_d, b_d, c_d, N);

	// Blocks until the device has completed all preceding requested tasks
	cudaThreadSynchronize();

	// Copy from device to host
	cudaMemcpy(c, c_d, N * sizeof(int), cudaMemcpyDeviceToHost);

	// Print C
	for (int i = 0; i < N; i++)
		cout << c[i] << " ";
	cout << endl;

	// Verify result
	for (int i = 0; i < N; i++)
		EXPECT_EQ(c[i], a[i] + b[i]);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

TEST(CUDA, DeclareEmptyDeviceArray)
{
	device_array_t<int> a0;

	EXPECT_TRUE(a0.empty());
	EXPECT_EQ(a0.ndims(), 1);
	EXPECT_EQ(a0.size(0), 0);
	EXPECT_EQ(a0.size(), 0);
	EXPECT_EQ(a0.raw_ptr(), nullptr);
}

TEST(CUDA, DeclareDeviceArrayWithSize)
{
	device_array_t<int> a1(5);

	EXPECT_FALSE(a1.empty());
	EXPECT_EQ(a1.ndims(), 1);
	EXPECT_EQ(a1.size(0), 5);
	EXPECT_EQ(a1.size(), 5);
	EXPECT_NE(a1.raw_ptr(), nullptr);

	device_array_t<int> a2(2, 3);

	EXPECT_FALSE(a2.empty());
	EXPECT_EQ(a2.ndims(), 2);
	EXPECT_EQ(a2.size(0), 2);
	EXPECT_EQ(a2.size(1), 3);
	EXPECT_EQ(a2.size(), 2 * 3);
	EXPECT_NE(a2.raw_ptr(), nullptr);

	device_array_t<int> a3(2, 3, 4);

	EXPECT_FALSE(a3.empty());
	EXPECT_EQ(a3.ndims(), 3);
	EXPECT_EQ(a3.size(0), 2);
	EXPECT_EQ(a3.size(1), 3);
	EXPECT_EQ(a3.size(2), 4);
	EXPECT_EQ(a3.size(), 2 * 3 * 4);
	EXPECT_NE(a3.raw_ptr(), nullptr);

	cudaPointerAttributes attr;
	cudaPointerGetAttributes(&attr, a1.raw_ptr());
	EXPECT_EQ(attr.memoryType, cudaMemoryTypeDevice);
}

} // anonymous namespace