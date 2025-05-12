#include "DebugConsole.hxx"

#if defined(_MSC_BUILD)
#include <Windows.h>

void writeToDebugConsole(std::ostringstream& oss)
{
	::OutputDebugStringA(oss.str().c_str());
}

#else
void writeToDebugConsole(std::ostringstream& oss)
{

}
#endif