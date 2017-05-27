#pragma once

#include <cuda_runtime.h>
#include <exception>

template <typename T>
__device__ float map(T value, T srcMin, T srcMax, T dstMin, T dstMax) {
	float l = (value - srcMin) / (srcMax - srcMin);
	return (1 - l) * dstMin + l * dstMax;
}

template <typename T>
__device__ float mapAndClamp(T value, T srcMin, T srcMax, T dstMin, T dstMax) {
	return clamp(map(value, srcMin, srcMax, dstMin, dstMax), dstMin, dstMax);
}


template <typename T>
__device__ float clamp(T value, T min, T max) {
	if (value < min)
		return min;
	if (value > max)
		return max;
	return value;
}

template <typename T>
__device__ float min(T a, T b) {
	return a < b ? a : b;
}

template <typename T>
__device__ float max(T a, T b) {
	return a > b ? a : b;
}

#define CUDA_CALL(fun) { cudaError_t err = (fun); cudaAssert(err, __FILE__, __LINE__); }
void cudaAssert(cudaError_t err, const char* file, int line);

#define CUDA_TRY(fun) { cudaError_t err = (fun); cudaTry(err, __FILE__, __LINE__); }
void cudaTry(cudaError_t err, const char* file, int line);

template <typename T>
class DeviceObject {
	T* m_object;

public:
	DeviceObject(const T& init) {
		CUDA_TRY(cudaMalloc(&m_object, sizeof(T)));
		CUDA_TRY(cudaMemcpy(m_object, &init, sizeof(T), cudaMemcpyHostToDevice));
	}

	~DeviceObject() {
		CUDA_CALL(cudaFree(m_object));
	}

	T* GetRaw() const {
		return m_object;
	}
};

template <typename T>
class DeviceArray {
	T* m_deviceMemory;
	size_t m_size;

public:
	explicit DeviceArray(size_t size) : m_size(size) {
		CUDA_TRY(cudaMalloc(&m_deviceMemory, size * sizeof(T)));
	}

	~DeviceArray() {
		CUDA_CALL(cudaFree(m_deviceMemory));
	}

	T* GetRaw() const {
		return m_deviceMemory;
	}

	size_t GetSize() const {
		return m_size;
	}

	void Set(size_t index, const T& data) {
		checkIndex(index);
		CUDA_TRY(cudaMemcpy(m_deviceMemory + index, &data, sizeof(T), cudaMemcpyHostToDevice));
	}

	T Get(size_t index) const {
		checkIndex(index);
		T t;
		CUDA_TRY(cudaMemcpy(&t, m_deviceMemory + index, sizeof(T), cudaMemcpyDeviceToHost));
		return t;
	}

private:
	void checkIndex(size_t index) const {
		if (index >= m_size)
			throw std::exception("Index out of range");
	}
};