#pragma once

#include "Vec3.cuh"

struct Object
{
	enum Type
	{
		SPHERE
	} type;

	int material = -1;

	union
	{
		struct
		{
			Vec3 position;
			float radius;
		} sphere;
	};

	Object() {}

	static Object Sphere(const Vec3& position, float radius, int material = -1);
};


struct Light
{
	enum Type
	{
		POINT,
		DIRECTIONAL
	} type;

	Vec3 color;
	float intensity;

	union
	{
		struct
		{
			Vec3 position;
		} point;
		struct
		{
			Vec3 direction;
		} directional;
	};

	Light() {}
	
	static Light Point(const Vec3& position, const Vec3& color, float intensity = 1.0f);
};

struct Material
{
	enum Type
	{
		PHONG,
		GLASS,
		GLOSSY
	} type;

	Vec3 color;

	union 
	{
		struct
		{
			float ambient;
			float diffuse;
			float specular;
			float specularExp;
		} phong;
	};

	static Material Phong(const Vec3& color, float ambieng, float diffuse, float specular, float specularExp);
	static Material Glossy(const Vec3& color);
};