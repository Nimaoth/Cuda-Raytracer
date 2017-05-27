#include "CudaUtils.h"

#include <iostream>
#include <sstream>

void cudaAssert(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cout << file << "(" << line << "): " << cudaGetErrorString(err) << std::endl;
		//__debugbreak();
	}
}

void cudaTry(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::stringstream ss;
		ss << file << "(" << line << "): " << cudaGetErrorString(err);
		throw std::exception(ss.str().c_str());
	}
}