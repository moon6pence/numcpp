#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("Hello World!", "[hello]") 
{
    REQUIRE((1 + 1) == 2);
}

#include "config.h"
#include "numcpp/base_array.test.inl"
//#include "numcpp/array.test.inl"

//#ifdef USE_OPENCV
//#include "numcpp/opencv.test.inl"
//#endif

//#ifdef USE_CUDA
//#include "numcpp/gpu_array.test.inl"
//#endif
