#pragma once

#include <ostream>

typedef enum {vDay =0, vIOC, VFOK } Validity;
inline std::ostream& operator<<(std::ostream& os, const Validity validity)
{
	switch (validity)
	{
		case vDay:
			os<<"Day";
			break;
		case vIOC:
			os<<"IOC";
		case vFOK:
			os<<"FOK";
			break;
	}
	return os;
}