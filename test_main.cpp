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

	// Test: allocate n-dimensonal array
	int shape[3] = {2, 3, 4};
	auto test3 = array<int, 3>(shape);

	printf("Bye!");

	return 0;
}
