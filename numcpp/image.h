#ifndef __NUMCPP_IMAGE_H__
#define __NUMCPP_IMAGE_H__

#include <string.h> // memcpy

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

#endif // __NUMCPP_IMAGE_H__