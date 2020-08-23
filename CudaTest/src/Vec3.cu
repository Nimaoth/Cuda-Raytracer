#include "Vec3.cuh"

#include <cmath>

__device__ Vec3 vec3Cross(const Vec3& a, const Vec3& b)
{
	return Vec3
	(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__device__ float vec3Dot(const Vec3& a, const Vec3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3 vec3Normalized(const Vec3& v)
{
	float mag = vec3Mag(v);
	if (mag != 0)
	{
		mag = 1.0f / mag;
		return Vec3
		{
			v.x * mag,
			v.y * mag,
			v.z * mag
		};
	}
	else
	{
		return Vec3{ 0, 0, 0 };
	}
}

__device__ void vec3Normalize(Vec3& v)
{
	float mag = vec3Mag(v);
	if (mag != 0)
	{
		mag = 1.0f / mag;
		v.x *= mag;
		v.y *= mag;
		v.z *= mag;
	}
}

__device__ float vec3Mag(const Vec3& v)
{
	return sqrtf(vec3MagSq(v));
}

__device__ float vec3MagSq(const Vec3& v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ Vec3 vec3Reflect(const Vec3& v, const Vec3& normal)
{
	return vec3Project(v, normal) * 2 - v;
}

__device__ Vec3 vec3Project(const Vec3& v, const Vec3& normal)
{
	return normal * vec3Dot(v, normal);
}

__device__ Vec3 operator*(const Vec3& a, const Vec3& b)
{
	return Vec3
	{
		a.x * b.x,
		a.y * b.y,
		a.z * b.z
	};
}

__device__ Vec3 operator*(const Vec3& v, float f)
{
	return Vec3
	{
		v.x * f,
		v.y * f,
		v.z * f
	};
}

__device__ Vec3& operator*=(Vec3& v, float f)
{
	v.x *= f;
	v.y *= f;
	v.z *= f;
	return v;
}

__device__ Vec3 operator+(const Vec3& a, const Vec3& b)
{
	return Vec3
	{
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	};
}

__device__ Vec3 operator+(const Vec3& a, float f)
{
	return Vec3(a.x + f, a.y + f, a.z + f);
}

__device__ Vec3 & operator+=(Vec3 & a, const Vec3 & b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__device__ Vec3 operator-(const Vec3& a, const Vec3& b)
{
	return Vec3
	{
		a.x - b.x,
		a.y - b.y,
		a.z - b.z
	};
}

__device__ Vec3 & operator-=(Vec3 & a, const Vec3 & b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

__device__ Vec3 operator-(const Vec3& v)
{
	return Vec3
	{
		-v.x,
		-v.y,
		-v.z
	};
}
