#pragma once

#include <cuda_runtime.h>

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	__host__ __device__ Vec3() {}
	__host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

__device__ Vec3 vec3Cross(const Vec3& a, const Vec3& b);

__device__ float vec3Dot(const Vec3& a, const Vec3& b);
__device__ void vec3Normalize(Vec3& v);
__device__ Vec3 vec3Normalized(const Vec3& v);
__device__ float vec3Mag(const Vec3& v);
__device__ float vec3MagSq(const Vec3& v);
__device__ Vec3 vec3Reflect(const Vec3& v, const Vec3& normal);
__device__ Vec3 vec3Project(const Vec3& v, const Vec3& normal);

__device__ Vec3 operator *(const Vec3& a, const Vec3& b);
__device__ Vec3 operator *(const Vec3& v, float f);
__device__ Vec3& operator *=(Vec3& v, float f);
__device__ Vec3 operator +(const Vec3& a, const Vec3& b);
__device__ Vec3& operator +=(Vec3& a, const Vec3& b);
__device__ Vec3 operator -(const Vec3& a, const Vec3& b);
__device__ Vec3& operator -=(Vec3& a, const Vec3& b);
__device__ Vec3 operator -(const Vec3& a);