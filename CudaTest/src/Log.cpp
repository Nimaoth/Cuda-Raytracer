#include "Log.h"

#include <iostream>

void Log(const char* msg)
{
	std::cout << msg << std::endl;
}

void LogError(const char * msg)
{
	std::cout << "[ERROR] " << msg << std::endl;
}
