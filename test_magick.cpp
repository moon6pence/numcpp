#include <stdio.h>

// algorithm header must be included first due to gcc STL bug
// reference: http://stackoverflow.com/questions/19043109/gcc-4-8-1-combining-c-code-with-c11-code
#include <algorithm>
#include <Magick++.h>

int main(int argc, char *argv[])
{
	// Test: Magick++
	{
		Magick::InitializeMagick(argv[0]);

		Magick::Image image;	

		try
		{
			image.read("Lena.bmp");
			printf("width: %d height: %d\n", 
				image.size().width(), image.size().height());
		}
		catch (Magick::Exception &error)
		{
			puts(error.what());
		}
	}
}
