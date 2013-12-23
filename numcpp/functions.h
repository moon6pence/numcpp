#ifndef NUMCPP_FUNCTIONS_H_
#define NUMCPP_FUNCTIONS_H_

/** One-dimensional array from j to k, such as {j, j+1, ..., k} */
template <typename T>
array_t<T> colon(T j, T k)
{
	if (k < j) 
		return array_t<T>();

	array_t<T> result((int)(k - j + 1));
	for (int index = 0; index < result.size(0); index++)
		result(index) = j + index;

	return result;
}

/** One-dimensional array from j to k step i, such as {j, j+i, j+2i, ... } */
template <typename T>
array_t<T> colon(T j, T i, T k)
{
	if (i == 0 || (i > 0 && j > k) || (i < 0 && j < k))
		return array_t<T>();

	array_t<T> result((int)((k - j) / i) + 1);
	for (int index = 0; index < result.size(0); index++)
		result(index) = j + index * i;

	return result;
}

#endif // NUMCPP_FUNCTIONS_H_