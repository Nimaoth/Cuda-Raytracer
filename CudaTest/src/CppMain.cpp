#include "ConfigLoader.h"


int main()
{
	RenderData renderData;
	std::vector<Object> objects;
	std::vector<Light> lights;
	std::string outFile;
	loadConfiguration("config.xml", renderData, objects, lights, outFile);
}