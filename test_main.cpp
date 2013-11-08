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

	// Test: allocate n-dimensional
	{
		auto test = array<int>(2, 3, 4);
		printf("nd array size = %d\n", test.size());
	}

	// Test: zeros
	{
		auto test = zeros<int>(4, 5, 6);

		printf("test.size() = %d\n", test.size());
		printf("test[0] = %d\n", test[0]);
	}

	puts("Bye!");

	return 0;
}
