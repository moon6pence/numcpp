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

void imwrite(const array_t<uint8_t, 2> &array, const std::string &file_path)
{
	using namespace Magick;

	try
	{
		Blob blob(array.raw_pointer(), array.size() * sizeof(uint8_t));

		Image image;
		image.size(Geometry(array.width(), array.height()));
		image.magick("r");
		image.read(blob);
		image.write(file_path);
	}
	catch (Magick::Exception &error)
	{
		puts(error.what());
	}
}

} // namespace numcpp

int main(int argc, char *argv[])
{
	Magick::InitializeMagick(argv[0]);

	// Test: Magick++
	{
		using namespace Magick;

		try
		{
			Image image;
			image.read("Lena.bmp");
			printf("width: %d height: %d\n", 
				image.size().width(), image.size().height());

			Blob blob;
			image.write(&blob, "RGB");
			printf("blob.length() = %lu\n", blob.length());
		}
		catch (Magick::Exception &error)
		{
			puts(error.what());
		}
	}

	namespace np = numcpp;

	// Test: imread, imwrite
	{
		auto image = np::imread("Lena.bmp");
		printf("image size = (%d, %d)\n", image.width(), image.height());

		// Invert image
		for (int y = 0; y < image.height(); y++)
			for (int x = 0; x < image.width(); x++)
				image(x, y) = 255 - image(x, y);

		np::imwrite(image, "result.bmp");
	}
}
