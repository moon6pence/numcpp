#include <ipp.h>
#include <stdio.h>

int main()
{
	IppiSize roi = {5, 4};
	Ipp8u x[8 * 4] = {0};

	for (int i = 0; i < 32; i++)
		printf("%u ", x[i]);
	puts("");

	ippiSet_8u_C1R(1, x, 8, roi);

	for (int i = 0; i < 32; i++)
		printf("%u ", x[i]);
	puts("");

	return 0;
}