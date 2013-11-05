#include <numcpp2/array.h>
#include <stdio.h>

int main()
{
	// Test: Program is just working
	printf("Hello World!\n");

	// Test: allocate 1d array
	auto test1 = array<int>(1024);

	// Test: allocate 2d array
	auto test2 = array<int>(3, 2);

	return 0;
}
