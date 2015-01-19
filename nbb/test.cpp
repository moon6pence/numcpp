// do nothing yet but check json library linking

#include <nbb/QuickJSON.h>

int main(int argc, char **argv)
{
	nbb::Context context;
	nbb::writeJson(context, "test.json");

	return 0;
}