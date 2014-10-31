#include <numcpp/cuda.h>

// function declared in cu file
void vecAdd(const int *A, const int *B, int *C, int N);

namespace {

using namespace np;

TEST(CUDA, HelloCUDA)
{
	using namespace std;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	ASSERT_TRUE(deviceCount > 0);
}

TEST(CUDA, RunKernel)
{
	using namespace std;

	const int N = 5;

	// Data on the host memory
	int a[N] = { 1, 2, 3, 4, 5 }, b[N] = { 3, 3, 3, 3, 3 }, c[N];

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

	// Verify result
	for (int i = 0; i < N; i++)
		EXPECT_EQ(c[i], a[i] + b[i]);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

TEST(CUDA, DeclareEmptyDeviceArray)
{
	GpuArray<int> a0;

	EXPECT_TRUE(empty(a0));
	EXPECT_EQ(sizeof(int), a0.itemSize());
	EXPECT_EQ(0, a0.ndims());
	EXPECT_EQ(nullptr, a0.raw_ptr());
}

TEST(CUDA, DeclareDeviceArrayWithSize)
{
	GpuArray<int> a1(5);

	EXPECT_FALSE(empty(a1));
	EXPECT_EQ(sizeof(int), a1.itemSize());
	EXPECT_EQ(1, a1.ndims());
	EXPECT_EQ(5, a1.size(0));
	EXPECT_NE(nullptr, a1.raw_ptr());

	GpuArray<float> a2(2, 3);

	EXPECT_FALSE(empty(a2));
	EXPECT_EQ(sizeof(float), a2.itemSize());
	EXPECT_EQ(2, a2.ndims());
	EXPECT_EQ(2, a2.size(0));
	EXPECT_EQ(3, a2.size(1));
	EXPECT_NE(nullptr, a2.raw_ptr());

	int shape[3] = { 2, 3, 4 };
	GpuArray<double> a3(make_vector(2, 3, 4));

	EXPECT_FALSE(empty(a3));
	EXPECT_EQ(sizeof(double), a3.itemSize());
	EXPECT_EQ(3, a3.ndims());
	EXPECT_EQ(2, a3.size(0));
	EXPECT_EQ(3, a3.size(1));
	EXPECT_EQ(4, a3.size(2));
	EXPECT_NE(nullptr, a3.raw_ptr());
	
	// This test doesn't work in windows
	// cudaPointerAttributes attr;
	// cudaPointerGetAttributes(&attr, a1.raw_ptr());
	// EXPECT_EQ(attr.memoryType, cudaMemoryTypeDevice);
}

typedef ArrayFixture CUDA_F;

TEST_F(CUDA_F, HostToDevice)
{
	GpuArray<int> a1_d(5);
	to_device(a1_d, a1);

	Array<int> a1_h(5);
	to_host(a1_h, a1_d);

	int *ptr1 = data1;
	for (auto i = begin(a1_h); i != end(a1_h); ++i, ++ptr1)
		EXPECT_EQ(*i, *ptr1);

	GpuArray<int> a2_d(2, 3);
	to_device(a2_d, a2);

	// Asynchronous copy
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	to_device(a2_d, a2, stream);
	to_host(a2, a2_d, stream);

	cudaStreamSynchronize(stream);
}

TEST_F(CUDA_F, ConstructorWithHostArray)
{
	GpuArray<int> a1_d(a1);

	EXPECT_FALSE(empty(a1_d));
	EXPECT_EQ(a1_d.ndims(), 1);
	EXPECT_EQ(a1_d.size(0), 5);
	EXPECT_NE(a1_d.raw_ptr(), nullptr);

	Array<int> a1_h(5);
	to_host(a1_h, a1_d);

	int *ptr1 = data1;
	for (auto i = begin(a1_h); i != end(a1_h); ++i, ++ptr1)
		EXPECT_EQ(*i, *ptr1);
}

TEST(CUDA, RunKernel2)
{
	using namespace std;

	const int N = 5;

	// Data on the host memory
	Array<int> a(N), b(N), c(N);

	// TODO: initialize in better way
	a(0) = 1; a(1) = 2; a(2) = 3; a(3) = 4; a(4) = 5;
	b(0) = 3; b(1) = 3; b(2) = 3; b(3) = 3; b(4) = 3;

	// Data on the device memory
	GpuArray<int> a_d(a), b_d(b);
	GpuArray<int> c_d(N);

	// Run kernel
	vecAdd(a_d, b_d, c_d, N);

	// Blocks until the device has completed all preceding requested tasks
	cudaThreadSynchronize();

	// Copy from device to host
	to_host(c, c_d);

	// Verify result
	for (int i = 0; i < N; i++)
		EXPECT_EQ(c(i), a(i) + b(i));
}

} // anonymous namespace