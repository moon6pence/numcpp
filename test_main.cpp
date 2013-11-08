#include <numcpp2/array.h>
#include <stdio.h>

int main()
{
	// Test: Program is just working
	puts("Hello World!");

	// Test: allocate 1d array
	{
		auto test = array<int>(1024);
		printf("1d array size = %d\n", test.size());
	}

	// Test: allocate 2d array
	{
		auto test = array<int>(3, 2);
		printf("2d array size = %d\n", test.size());
	}

	// Test: allocate n-dimensonal array
	{
		int shape[3] = {2, 3, 4};
		auto test = array<int, 3>(shape);
		printf("nd array size = %d\n", test.size());
	}

	// Test: allocate n-dimensional array with variadic tempalte
	{
		auto test = array<int>(2, 3, 4);
		printf("nd array size = %d\n", test.size());
	}

	// Test: zeros
	{
		int shape[3] = {2, 3, 4};
		auto test1 = zeros<int, 3>(shape), test2 = zeros<int>(4, 5, 6);

		printf("test1.size() = %d\n", test1.size());
		printf("test2.size() = %d\n", test2.size());
		printf("test1[0] = %d\n", test1[0]);
		printf("test1[1] = %d\n", test2[0]);
	}

	puts("Bye!");

	return 0;
}
