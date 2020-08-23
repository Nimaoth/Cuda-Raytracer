#include "Raytracer.cuh"
#include "CudaUtils.h"
#include "Bitmap.h"
#include "Log.h"

#include <curand_kernel.h>

#include "ConfigLoader.h"

#define PI 3.14159265359

#define DEG_TO_RAD (PI / 180.0)

typedef unsigned char byte;

__device__ float generateRandomNumber(curandState* globalState, int ind) {
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void convertImage(RenderData* data, byte* pixels)
{
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= data->width || y >= data->height)
		return;

	size_t index = x + y * data->width;
	size_t offset = index * 3;

	Vec3& pixel = data->pixels[index];
	pixels[offset + 0] = (byte)mapAndClamp<float>(pixel.z, 0, 1, 0, 255);
	pixels[offset + 1] = (byte)mapAndClamp<float>(pixel.y, 0, 1, 0, 255);
	pixels[offset + 2] = (byte)mapAndClamp<float>(pixel.x, 0, 1, 0, 255);
}

unsigned int getBlockCount(unsigned int total, unsigned int threads)
{
	if (total % threads)
	{
		return total / threads + 1;
	}
	return total / threads;
}

void asdf(int argc, char** argv)
{
	RenderData renderData;
	renderData.width = 128;
	renderData.height = 128;
	renderData.seed = 12345;
	renderData.backgroundColor = { .1f, .1f, .1f };

	// camera
	renderData.cameraPosition.x = 0;
	renderData.cameraPosition.y = 0;
	renderData.cameraPosition.z = 5;
	renderData.cameraDirection.x = 0;
	renderData.cameraDirection.y = 0;
	renderData.cameraDirection.z = -1;
	renderData.fov = (float)PI * 0.5f;

	// parse cmdline arguments
	std::vector<Object> objects;
	std::vector<Material> materials;
	std::vector<Light> lights;
	std::string outFile = "image.bmp";
	std::string configFile = "";
	if (argc >= 2)
		configFile = argv[1];
	if (configFile.length() > 0)
	{
		loadConfiguration(configFile, renderData, objects, materials, lights, outFile);
	}

	renderData.inverseTanHalfFov = 1.0f / tanf(renderData.fov * 0.5f);
	renderData.aspectRatio = (float)renderData.width / (float)renderData.height;

	// pixels array
	DeviceArray<Vec3> d_pixels(renderData.width * renderData.height);
	renderData.pixels = d_pixels.GetRaw();

	// objects
	DeviceArray<Object> d_objects(objects.size());
	for (size_t i = 0; i < objects.size(); i++)
		d_objects.Set(i, objects[i]);
	renderData.objects = d_objects.GetRaw();
	renderData.objectsCount = d_objects.GetSize();

	// materials
	DeviceArray<Material> d_materials(materials.size());
	for (size_t i = 0; i < materials.size(); i++)
		d_materials.Set(i, materials[i]);
	renderData.materials = d_materials.GetRaw();
	renderData.materialCount = d_materials.GetSize();

	// lights
	DeviceArray<Light> d_lights(lights.size());
	for (size_t i = 0; i < lights.size(); i++)
		d_lights.Set(i, lights[i]);
	renderData.lights = d_lights.GetRaw();
	renderData.lightCount = d_lights.GetSize();

	dim3 threads{ 16, 16 };
	dim3 blocks{ getBlockCount(renderData.width, threads.x), getBlockCount(renderData.height, threads.y) };

	// move renderData to gpu memory
	DeviceObject<RenderData> d_renderData = renderData;

	// raytrace
	Log("Raytracing...");
	raytrace<<<blocks, threads>>>(d_renderData.GetRaw());
	cudaDeviceSynchronize();

	// create bitmap and convert from float[] to byte[]
	Log("Converting image data...");
	DeviceArray<byte> d_bytePixelData(renderData.width * renderData.height * 3);
	convertImage<<<blocks, threads>>>(d_renderData.GetRaw(), d_bytePixelData.GetRaw());
	cudaDeviceSynchronize();

	// save image
	Log("Saving image...");
	BitMap bitmap(renderData.width, renderData.height);
	CUDA_CALL(cudaMemcpy(bitmap.GetRaw(), d_bytePixelData.GetRaw(), d_bytePixelData.GetSize(), cudaMemcpyDeviceToHost));
	TRY(bitmap.Save(outFile));
}

int main(int argc, char** argv)
{
	try
	{
		size_t defaultStackSize;
		CUDA_TRY(cudaDeviceGetLimit(&defaultStackSize, cudaLimitStackSize));

		size_t newStackSize = defaultStackSize << 3;

		CUDA_TRY(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));
		
		//CUDA_TRY(cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 100));
		asdf(argc, argv);
	}
	catch (std::exception& e)
	{
		LogError(e.what());
	}
}