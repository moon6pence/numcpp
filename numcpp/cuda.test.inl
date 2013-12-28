#include <numcpp/cuda.h>
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

} // anonymous namespace