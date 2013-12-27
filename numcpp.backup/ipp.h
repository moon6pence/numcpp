#ifndef __NUMCPP_IPP_H__
#define __NUMCPP_IPP_H__

#include <ipp.h>

namespace numcpp {

template <typename T>
int stepBytes(const array_t<T, 2> &image)
{
	return image.width() * sizeof(T);
}

template <typename T>
IppiSize ippiSize(const array_t<T, 2> &image)
{
	IppiSize result = { image.width(), image.height() };
	return result;
}

template <typename T>
IppiRect ippiRect(const array_t<T, 2> &image)
{
	IppiRect result = { 0, 0, image.width(), image.height() };
	return result;
}

} // namespace numcpp

#endif // __NUMCPP_IPP_H__