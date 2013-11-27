#ifndef __NUMCPP_H__
#define __NUMCPP_H__

#include "config.h"

#include "numcpp/array.h"
#include "numcpp/array_function.h"

#ifdef USE_OPENCV
#include "numcpp/image.h"
#endif

#ifdef USE_IPP
#include "numcpp/ipp.h"
#endif

#endif // __NUMCPP_H__