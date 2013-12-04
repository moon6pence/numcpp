#include <numcpp.h>

#include <ipp.h>
#include <iostream>

int main()
{
	// Test IPP
	{
		using namespace std;

		const int N = 5;

		Ipp32f *a = ippsMalloc_32f(N), *b = ippsMalloc_32f(N), *c = ippsMalloc_32f(N);

		// a = 1
		ippsSet_32f(1.f, a, N);
		for (int i = 0; i < N; i++)
			cout << a[i] << " ";
		cout << endl;

		// b = 2
		ippsSet_32f(2.f, b, N);
		for (int i = 0; i < N; i++)
			cout << b[i] << " ";
		cout << endl;

		// c = a + b
		ippsAdd_32f(a, b, c, N);
		for (int i = 0; i < N; i++)
			cout << c[i] << " ";
		cout << endl;

		ippsFree(a);
		ippsFree(b);
		ippsFree(c);
	}

	// numcpp2
	{
		using namespace numcpp;

		auto a = array<float>(5), b = array<float>(5), c = array<float>(5);
		
		// a = 1
		fill(a, 1.f);
		print(a);

		// b = 2
		fill(b, 2.f);
		print(b);

		// c = a + b
		ippsAdd_32f(a, b, c, c.length());
		print(c);

		// or use map function
		map(c, a, b, [](float _a, float _b) { return _a + _b; });
		print(c);
	}

	return 0;
}