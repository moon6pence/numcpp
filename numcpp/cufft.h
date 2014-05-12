#ifndef NUMCPP_CUFFT_H_
#define NUMCPP_CUFFT_H_

#include "cuda.h"
#include <cufft.h>

#define CUFFT_ERROR(cufft_function_call) \
{\
	cufftResult __result;\
	__result = (cufft_function_call);\
	if (__result != CUFFT_SUCCESS)\
	{\
		fprintf(stderr, "CUFFT Error: %d\n", __result);\
		exit(-1);\
	}\
}

namespace np {

// TODO: Can I do this in constant time complexity?
// Return least k, such as x <= k, k = 2^n
inline int getLeastPower2Over(int x)
{
	int k = 1;

	while (!(x <= k))
		k = k << 1;

	return k;
}

struct CuFFT_R2C
{
	CuFFT_R2C() : plan(0)
	{
	}

	~CuFFT_R2C()
	{
		if (plan) { cufftDestroy(plan); plan = 0; }
	}

	void operator() (device_array_t<float2> &dst, device_array_t<float> &src, cudaStream_t stream = 0)
	{
		// Lazy initialization
		if (plan == 0)
		{
			CUFFT_ERROR(cufftPlan1d(&plan, src.size(0), CUFFT_R2C, src.size(1)));

			if (stream) 
				CUFFT_ERROR(cufftSetStream(plan, stream));
		}

		const tuple expected(getLeastPower2Over(src.size(0)), src.size(1));
		if (dst.size() != expected) 
			dst = DeviceArray<float2>(expected);

		// Execute CuFFT
		CUFFT_ERROR(cufftExecR2C(plan, src, dst));
	}

private:
	cufftHandle plan;
};

struct CuFFT_C2C
{
	CuFFT_C2C() : plan(0)
	{
	}

	~CuFFT_C2C()
	{
		if (plan) { cufftDestroy(plan); plan = 0; }
	}

	void forward(device_array_t<float2> &dst, device_array_t<float2> &src, cudaStream_t stream = 0)
	{
		operator() (dst, src, CUFFT_FORWARD, stream);
	}

	void inverse(device_array_t<float2> &dst, device_array_t<float2> &src, cudaStream_t stream = 0)
	{
		operator() (dst, src, CUFFT_INVERSE, stream);
	}

	void operator() (device_array_t<float2> &dst, device_array_t<float2> &src, int direction, cudaStream_t stream = 0)
	{
		// Lazy initialization
		if (plan == 0)
		{
			CUFFT_ERROR(cufftPlan1d(&plan, src.size(0), CUFFT_C2C, src.size(1)));

			if (stream) 
				CUFFT_ERROR(cufftSetStream(plan, stream));
		}

		const tuple expected(getLeastPower2Over(src.size(0)), src.size(1));
		if (dst.size() != expected) 
			dst = DeviceArray<float2>(expected);

		// Execute CuFFT
		CUFFT_ERROR(cufftExecC2C(plan, src, dst, direction));
	}

private:
	cufftHandle plan;
	int cufft_direction;
};

} // namespace np

#endif // NUMCPP_CUFFT_H_