#ifndef NUMCPP_IPPI_H_
#define NUMCPP_IPPI_H_

// Utility function for ippi libraries (Intel Performance Premitives, Image)
#include <ippi.h>

namespace np {

template <typename T>
int stepBytes(const array_t<T> &image)
{
	return image.size(0) * sizeof(T);
}

template <typename T>
inline IppiSize ippiSize(const array_t<T> &image)
{
	IppiSize result = { image.size(0), image.size(1) };
	return result;
}

inline IppiRect ippiRect(int width, int height)
{
	IppiRect result = { 0, 0, width, height };
	return result;
}

template <typename T>
inline IppiRect ippiRect(const array_t<T> &image)
{
	return ippiRect(image.width(), image.height());
}

} // namespace np

#endif // NUMCPP_IPPI_H_