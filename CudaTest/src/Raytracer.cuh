#pragma once

#include <cuda_runtime.h>
#include "Objects.h"

struct Pixel {
	float r;
	float g;
	float b;
};

struct RenderData {
	unsigned long long seed;

	unsigned int width;
	unsigned int height;

	Pixel* pixels;

	// camera
	Vec3 cameraPosition;
	Vec3 cameraDirection;
	float aspectRatio;
	float fov;
	float inverseTanHalfFov;

	Vec3 backgroundColor;

	// objects
	Object* objects;
	size_t objectsCount;

	// materials
	Material* materials;
	size_t materialCount;

	// lights
	Light* lights;
	size_t lightCount;
};

__global__ void raytrace(RenderData* data);