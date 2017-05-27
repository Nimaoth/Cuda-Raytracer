#include "Raytracer.cuh"
#include "CudaUtils.h"
#include "Vec3.cuh"

#include <helper_math.h>

#define MAX_RECURSION_DEPTH 2

struct HitInfo
{
	Vec3 position;
	Vec3 normal;
	float distanceSq;
	int index;
	Object* object;
};

struct Ray
{
	Vec3 origin;
	Vec3 direction;
};

// function declarations
__device__ Vec3 calculateColor(RenderData* data, Ray out, HitInfo* hit);
__device__ Ray getCameraRay(RenderData* data, int x, int y);
__device__ bool isHitSphere(Ray ray, Object* obj, HitInfo* hitInfo);
__device__ bool isHit(Ray ray, Object* obj, HitInfo* hitInfo);
__device__ bool traceRay(RenderData* data, Ray ray, HitInfo* hit, int mask = -1);
__device__ void calculateGlossy(int depth, RenderData* data, Ray out, HitInfo* hit, Material* material, Vec3* color);
__device__ void calculatePhong(int depth, RenderData* data, Ray out, HitInfo* hit, Material* material, Vec3* color);
__device__ Vec3 calculateColor(int depth, RenderData* data, Ray out, HitInfo* hit);

// function definitions
__device__ Ray getCameraRay(RenderData* data, int x, int y)
{
	Vec3 ray = data->cameraDirection;
	Vec3 right = vec3Cross(ray, { 0.0f, 1.0f, 0.0f });
	Vec3 up = vec3Cross(right, ray);

	ray *= data->inverseTanHalfFov;

	float dx = map<float>(x, 0, data->width, -1, 1) * data->aspectRatio;
	float dy = map<float>(y, 0, data->height, 1, -1);

	ray += right * dx;
	ray += up * dy;

	vec3Normalize(ray);

	return Ray{ data->cameraPosition, ray };
}

__device__ bool isHitSphere(Ray ray, Object* obj, HitInfo* hitInfo)
{
	Vec3 op = ray.origin - obj->sphere.position;
	float a = vec3MagSq(ray.direction);
	float b = 2 * vec3Dot(ray.direction, op);
	float c = vec3MagSq(op) - obj->sphere.radius * obj->sphere.radius;

	float dis = b * b - 4 * a * c;

	float t = 0;

	if (dis < 0)
		return false;
	if (dis == 0)
		t = -b / (2 * a);
	else
	{
		float s = sqrtf(dis);
		float a2 = 2 * a;
		
		t = (-b - s) / a2;

		if (t <= 0)
			t = (-b + s) / a2;
	}

	if (t <= 0)
		return false;

	hitInfo->position = ray.origin + ray.direction * t;
	hitInfo->distanceSq = vec3MagSq(hitInfo->position - ray.origin);
	hitInfo->normal = vec3Normalized(hitInfo->position - obj->sphere.position);

	return true;
}

__device__ bool isHit(Ray ray, Object* obj, HitInfo* hitInfo)
{
	switch (obj->type) {
	case Object::SPHERE:
		return isHitSphere(ray, obj, hitInfo);
	default:
		return false;
	}
}

__device__ bool traceRay(RenderData* data, Ray ray, HitInfo* hit, int mask)
{
	HitInfo temp;
	int index = -1;
	float distanceSquared = 0.0f;
	for (int i = 0; i < data->objectsCount; i++)
	{
		if (i == mask)
			continue;

		Object* obj = &data->objects[i];
		if (isHit(ray, obj, &temp))
		{
			if (index == -1 || temp.distanceSq < distanceSquared)
			{
				index = i;
				distanceSquared = temp.distanceSq;
				*hit = temp;
				hit->index = i;
				hit->object = obj;
			}
		}
	}

	return index != -1;
}

__device__ void calculateGlossy(int depth, RenderData* data, Ray out, HitInfo* hit, Material* material, Vec3* color)
{
	Vec3 reflected = -vec3Reflect(out.direction, hit->normal);
	Ray reflectedRay{ hit->position, reflected };

	Vec3 reflectedColor = data->backgroundColor;
	HitInfo info;
	if (traceRay(data, reflectedRay, &info, hit->index))
	{
		reflectedColor = calculateColor(depth + 1, data, reflectedRay, &info);
	}

	*color += reflectedColor * material->color;
}

__device__ void calculatePhong(int depth, RenderData* data, Ray out, HitInfo* hit, Material* material, Vec3* color)
{
	// ambient part
	*color += material->color * material->phong.ambient;

	Vec3 diffuse;
	Vec3 specular;
	for (size_t i = 0; i < data->lightCount; i++)
	{
		const Light& light = data->lights[i];
		Vec3 toLight = light.point.position - hit->position;
		vec3Normalize(toLight);

		// calculate shadow
		Ray shadowRay;
		shadowRay.origin = hit->position;
		shadowRay.direction = toLight;
		HitInfo shadowHit;
		if (!traceRay(data, shadowRay, &shadowHit, hit->index))
		{
			// diffuse part
			float dot = vec3Dot(hit->normal, toLight);
			if (dot > 0)
			{
				diffuse += (light.color * material->color) * (light.intensity * dot);
			}

			// specular part
			Vec3 reflected = vec3Reflect(out.direction, hit->normal);
			dot = -vec3Dot(reflected, toLight);
			if (dot > 0)
			{
				dot = powf(dot, material->phong.specularExp);

				specular += Vec3(1, 1, 1) * dot;
			}
		}
	}

	*color += diffuse * material->phong.diffuse;
	*color += specular * material->phong.specular;
}

__device__ Vec3 calculateColor(int depth, RenderData* data, Ray out, HitInfo* hit)
{
	if (depth > MAX_RECURSION_DEPTH)
		return data->backgroundColor;

	Material* material = &data->materials[hit->object->material];

	// final color
	Vec3 color;

	switch (material->type)
	{
	case Material::PHONG:
		calculatePhong(depth, data, out, hit, material, &color);
		break;
	case Material::GLOSSY:
		calculateGlossy(depth, data, out, hit, material, &color);
		break;
	};

	return color;
}

__global__ void raytrace(RenderData* data)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= data->width || y >= data->height)
		return;

	int index = x + y * data->width;

	Ray ray = getCameraRay(data, x, y);
	//ray.direction = { 0, 0, -1 };

	HitInfo hit;
	if (traceRay(data, ray, &hit))
	{
		Vec3 color = calculateColor(0, data, ray, &hit);

		Pixel& pixel = data->pixels[index];
		pixel.r = color.x;
		pixel.g = color.y;
		pixel.b = color.z;
	}
	else
	{

		Pixel& pixel = data->pixels[index];
		pixel.r = data->backgroundColor.x;
		pixel.g = data->backgroundColor.y;
		pixel.b = data->backgroundColor.z;
	}
}