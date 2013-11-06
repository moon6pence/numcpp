#include <numcpp2/array.h>
#include <stdio.h>

int main()
{
	// Test: Program is just working
	puts("Hello World!");

	// Test: allocate 1d array
	auto test1 = array<int>(1024);
	printf("1d array size = %d\n", test1.size());

	// Test: allocate 2d array
	auto test2 = array<int>(3, 2);
	printf("2d array size = %d\n", test2.size());

	// Test: allocate n-dimensonal array
	int shape[3] = {2, 3, 4};
	auto test3 = array<int, 3>(shape);
	printf("nd array size = %d\n", test3.size());

	puts("Bye!");

	return 0;
}
