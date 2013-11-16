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
		auto empty1 = np::empty<int, 1>();

		if (empty1.empty()) puts("It's empty!");
		printf("empty1.size() = %d\n", empty1.size());

		printf("print(empty1) : "); 
		np::print(empty1);

		auto empty2 = np::empty<int, 2>();
		printf("empty2.size() = %d\n", empty2.size());
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

		auto test2 = np::array<int>(3, 2);
		test2(0, 0) = 1;
		test2(1, 0) = 2;
		test2(2, 0) = 3;
		test2(0, 1) = 4;
		test2(1, 1) = 5;
		test2(2, 1) = 6;
		np::print(test2);

		auto test3 = np::array<int>(2, 2, 2);
		test3(0, 0, 0) = 1;
		test3(1, 0, 0) = 2;
		test3(0, 1, 0) = 3;
		test3(1, 1, 0) = 4;
		test3(0, 0, 1) = 5;
		test3(1, 0, 1) = 6;
		test3(0, 1, 1) = 7;
		test3(1, 1, 1) = 8;
		np::print(test3);
	}

	// Test: map high-order function
	{
		auto test1 = np::colon(1, 5), test2 = np::array<int>(5);	

		np::map(test2, test1, [](int i) { return i + 1; });
		np::print(test1);
		np::print(test2);
	}

	// Test: colon function
	{
		np::print(np::colon(1, 10));	
		np::print(np::colon(1.5, 10.0));	

		np::print(np::colon(1, 2, 10));	
		np::print(np::colon(1.0, 1.5, 10.0));	
		np::print(np::colon(1.0, 1.5, 9.9));	
	}

	// Test: array with initializer_list
	{
		auto test1 = np::array({3, 2, 1, 5, 4});
		np::print(test1);	
	}

	// Test: meshgrid
	{
		auto xgv = np::colon(0, 5), ygv = np::colon(0, 5);
		auto X = np::array<int>(6, 6), Y = np::array<int>(6, 6);

		np::meshgrid(X, Y, xgv, ygv);
		np::print(X);
		np::print(Y);
	}

	puts("Bye!");

	return 0;
}
