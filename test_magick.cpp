#include <stdio.h>
#include <string.h>

// algorithm header must be included first due to gcc STL bug
// reference: http://stackoverflow.com/questions/19043109/gcc-4-8-1-combining-c-code-with-c11-code
#include <algorithm>
#include <Magick++.h>

#include <numcpp/array.h>

namespace numcpp {

array_t<uint8_t, 2> imread(const std::string &file_path)
{
	using namespace Magick;

	try
	{
		Image image;	
		image.read(file_path);

		Blob blob;
		image.write(&blob, "r");

		auto result = array<uint8_t>(image.size().width(), image.size().height());
		memcpy(result, blob.data(), blob.length());
		return result;
	}
	catch (Magick::Exception &error)
	{
		puts(error.what());
		return empty<uint8_t, 2>();
	}
}

} // namespace numcpp

int main(int argc, char *argv[])
{
	Magick::InitializeMagick(argv[0]);

	// Test: Magick++
	{
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

	namespace np = numcpp;

	// Test: imread
	{
		auto image = np::imread("Lena.bmp");
		printf("image size = (%d, %d)\n", image.width(), image.height());
	}
}
