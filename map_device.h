#ifndef MAP_DEVICE_H_
#define MAP_DEVICE_H_

#include <numcpp/gpu_array.h>

template <typename Input, typename Output, class GetFunc, class KernelFunc, class PutFunc>
void map_device(int from, int to, GetFunc &get, KernelFunc &kernel, PutFunc &put)
{
	np::Array<Input, 2> input;
	np::GpuArray<Input, 2> input_d;
	np::GpuArray<Output, 2> output_d;
	np::Array<Output, 2> output;

	for (int index = from; index < to; index++)
	{
		get(index, input);
		np::to_device(input_d, input);
		kernel(output_d, input_d);
		np::to_host(output, output_d);
		put(index, std::move(output));
	}
}

template <typename Input, typename Output, class GetFunc, class KernelFunc, class PutFunc>
void map_device2(int from, int to, GetFunc &get, KernelFunc &kernel, PutFunc &put)
{
	cudaStream_t stream_h2d, stream_kernel, stream_d2h;

	cudaStreamCreate(&stream_h2d);
	cudaStreamCreate(&stream_kernel);
	cudaStreamCreate(&stream_d2h);

	np::Array<Input, 2> input;
	np::GpuArray<Input, 2> input_d[2];
	np::GpuArray<Output, 2> output_d[2];
	np::Array<Output, 2> output;

	// index = -2
	get(from + 0, input);
	np::to_device(input_d[0], input, stream_h2d);

	cudaStreamSynchronize(stream_h2d);

	// index = -1
	get(from + 1, input);
	np::to_device(input_d[1], input, stream_h2d);

	kernel(output_d[0], input_d[0], stream_kernel);

	cudaStreamSynchronize(stream_h2d);
	cudaStreamSynchronize(stream_kernel);

	for (int index = from; index < to; index++)
	{
		// index + 2
		if (index + 2 < to)
			get(index + 2, input);

		const int copy_phase = (index - from) % 2, kernel_phase = (index - from + 1) % 2;

		// index + 2
		np::to_device(input_d[copy_phase], input, stream_h2d);

		// index + 1
		kernel(output_d[kernel_phase], input_d[kernel_phase], stream_kernel);

		// index
		np::to_host(output, output_d[copy_phase], stream_d2h);

		cudaStreamSynchronize(stream_h2d);
		cudaStreamSynchronize(stream_kernel);
		cudaStreamSynchronize(stream_d2h);

		put(index, std::move(output));
	}

	cudaStreamDestroy(stream_h2d);
	cudaStreamDestroy(stream_kernel);
	cudaStreamDestroy(stream_d2h);
}

template <typename Input, typename Output, class GetFunc, class KernelFunc, class PutFunc>
void map_device3(int from, int to, GetFunc &get, KernelFunc &kernel, PutFunc &put)
{
	cudaStream_t stream_h2d, stream_kernel;

	cudaStreamCreate(&stream_h2d);
	cudaStreamCreate(&stream_kernel);

	np::Array<Input, 2> input[2];
	np::GpuArray<Input, 2> input_d[2];
	np::GpuArray<Output, 2> output_d[2];
	np::Array<Output, 2> output[2];

	// index = -4
	get(from + 0, input[0]);

	// index = -3
	get(from + 1, input[1]);
	np::to_device(input_d[0], input[0], stream_h2d);

	cudaStreamSynchronize(stream_h2d);

	// index = -2
	get(from + 2, input[0]);
	np::to_device(input_d[1], input[1], stream_h2d);
	kernel(output_d[0], input_d[0], stream_kernel); 

	cudaStreamSynchronize(stream_h2d);
	cudaStreamSynchronize(stream_kernel);

	// index = -1
	np::to_device(input_d[0], input[0], stream_h2d);
	kernel(output_d[1], input_d[1], stream_kernel); 
	np::to_host(output[0], output_d[0], stream_h2d);
	get(from + 3, input[1]);

	cudaStreamSynchronize(stream_h2d);
	cudaStreamSynchronize(stream_kernel);

	cudaEvent_t clock;
	cudaEventCreate(&clock);

	for (int index = from; index < to; index++)
	{
		// Actual running sequence
		// if (index + 4 < to) get(index + 4, input[0]);
		// np::to_device(input_d[1], input[1], stream_h2d);
		// kernel(output_d[0], input_d[0], stream_kernel);
		// np::to_host(output[1], output_d[1], stream_d2h);
		// put(index, output[0]);

		cudaEventRecord(clock);

		const int phase0 = (index - from) % 2, phase1 = (index - from + 1) % 2;

		kernel(output_d[phase0], input_d[phase0], stream_kernel);
		np::to_device(input_d[phase1], input[phase1], stream_h2d);
		np::to_host(output[phase1], output_d[phase1], stream_h2d);
		if (index + 4 < to) get(index + 4, input[phase0]);
		put(index, std::move(output[phase0]));

		cudaStreamSynchronize(stream_h2d);
		cudaStreamSynchronize(stream_kernel);
	}
}

#endif // MAP_DEVICE_H_