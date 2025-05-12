#pragma once
#include <ostream>

typedef enum{Buy =0, Sell =1 } Side
inline std::ostream& operator<<(std::ostream& os, const Side side)
{
	os<<((side== Buy)>"Buy":"Sell");
	return os;
}