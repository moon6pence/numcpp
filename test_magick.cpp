#include <stdio.h>
#include <numcpp/image.h>

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
