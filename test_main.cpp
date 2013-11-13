#include <numcpp/array.h>
#include <stdio.h>

int main()
{
	namespace np = numcpp;

	// Test: Program is just working
	puts("Hello World!");

	// Test: allocate 1d array
	{
		auto test = np::array<int>(1024);
		printf("1d array size = %d\n", test.size());
	}

	// Test: allocate 2d array
	{
		auto test = np::array<int>(3, 2);
		printf("2d array size = %d\n", test.size());
	}

	// Test: allocate n-dimensional
	{
		auto test = np::array<int>(2, 3, 4);
		printf("nd array size = %d\n", test.size());
	}

	// Test: zeros, ones, fill, print
	{
		auto test0 = np::zeros<int>(2, 3, 4);
		auto test1 = np::ones<int>(2, 3, 4);

		np::print(test0);
		np::print(test1);

		np::fill(test1, 11);
		np::print(test1);
	}

	// Test: empty array
	{
		auto empty_array = np::empty<int, 1>();

		if (empty_array.empty()) puts("It's empty!");
		printf("empty_array.size() = %d\n", empty_array.size());

		printf("print(empty_array) : "); 
		print(empty_array);
	}

	puts("Bye!");

	return 0;
}
