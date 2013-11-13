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

	// Test: at(x, y, ...) or operator(x, y, ...)
	{
		auto test1 = np::array<int>(5);
		test1(0) = 1;
		test1(1) = 2;
		test1(2) = 3;
		test1(3) = 4;
		test1(4) = 5;
		np::print(test1);

		auto test2 = np::array<int>(2, 3);
		test2(0, 0) = 1;
		test2(0, 1) = 2;
		test2(0, 2) = 3;
		test2(1, 0) = 4;
		test2(1, 1) = 5;
		test2(1, 2) = 6;
		np::print(test2);

		auto test3 = np::array<int>(2, 2, 2);
		test3(0, 0, 0) = 1;
		test3(0, 0, 1) = 2;
		test3(0, 1, 0) = 3;
		test3(0, 1, 1) = 4;
		test3(1, 0, 0) = 5;
		test3(1, 0, 1) = 6;
		test3(1, 1, 0) = 7;
		test3(1, 1, 1) = 8;
		np::print(test3);
	}

	puts("Bye!");

	return 0;
}
