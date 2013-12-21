#ifndef NUMCPP_ARRAY_H_
#define NUMCPP_ARRAY_H_

namespace numcpp {

template <typename T, int Dim = 1>
struct array_t
{
	bool empty()
	{
		return true;
	}
};

} // namespace numcpp

#endif // NUMCPP_ARRAY_H_