#include "ConfigLoader.h"
#include <rapidxml.hpp>
#include <fstream>
#include <sstream>
#include <functional>

#define PI 3.14159265359
#define DEG_TO_RAD (PI / 180.0)

std::istream& operator>>(std::istream& in, Vec3& vec)
{
	if (in)
		in >> vec.x;
	if (in)
		in >> vec.y;
	if (in)
		in >> vec.z;
	return in;
}

void forEach(rapidxml::xml_node<>* t, std::function<void(const std::string& name, const std::string& value)> fun)
{
	while (t) {
		std::string name(t->name(), t->name_size());
		std::string value(t->value(), t->value_size());
		fun(name, value);
		t = t->next_sibling();
	}
}

void forEach(rapidxml::xml_attribute<>* t, std::function<void(const std::string& name, const std::string& value)> fun)
{
	while (t) {
		std::string name(t->name(), t->name_size());
		std::string value(t->value(), t->value_size());
		fun(name, value);
		t = t->next_attribute();
	}
}

void loadConfiguration(const std::string& filename, RenderData& data, std::vector<Object>& objects, std::vector<Material>& materials, std::vector<Light>& lights, std::string& outFile)
{
	std::ifstream file(filename);
	if (!file.is_open())
		return;

	std::vector<char> content;
	file.seekg(0, std::ios::end);
	size_t size = (size_t)file.tellg();
	content.resize(size_t(size * 1.5) + 1);
	file.seekg(0);
	file.read(content.data(), size);
	content[size + 1] = 0;


	rapidxml::xml_document<> doc;
	doc.parse<0>(content.data());

	auto root = doc.first_node();

	auto renderNode = root->first_node("render");
	if (renderNode)
	{
		forEach(renderNode->first_node(), [&](auto& name, auto& value)
		{
			std::stringstream valueStream;
			valueStream << value;

			if (name == "width")
				valueStream >> data.width;
			else if (name == "height")
				valueStream >> data.height;
			else if (name == "seed")
				valueStream >> data.seed;
			else if (name == "fov")
			{
				valueStream >> data.fov;
				data.fov *= (float)DEG_TO_RAD;
			}
			else if (name == "camPos")
				valueStream >> data.cameraPosition;
			else if (name == "camDir")
				valueStream >> data.cameraDirection;
			else if (name == "out")
				outFile = value;
		});
	}

	auto objectsNode = root->first_node("objects");
	if (objectsNode)
	{
		auto objNode = objectsNode->first_node();
		while (objNode)
		{

			if (strcmp("sphere", objNode->name()) == 0) {
				Vec3 position;
				float radius = 1.0f;
				int material = -1;

				forEach(objNode->first_node(), [&](auto& name, auto& value)
				{
					std::stringstream valueStream;
					valueStream << value;

					if (name == "position")
						valueStream >> position;
					else if (name == "radius")
						valueStream >> radius;
					else if (name == "material")
						valueStream >> material;
				});

				objects.push_back(Object::Sphere(position, radius, material));
			}

			objNode = objNode->next_sibling();
		}
	}

	auto materialsNode = root->first_node("materials");
	if (materialsNode)
	{
		auto material = materialsNode->first_node();
		while (material)
		{

			if (strcmp("phong", material->name()) == 0)
			{
				Vec3 color;
				float ambient;
				float diffuse;
				float specular;
				float specularExp;

				forEach(material->first_node(), [&](auto& name, auto& value)
				{
					std::stringstream valueStream;
					valueStream << value;

					if (name == "color")
						valueStream >> color;
					else if (name == "ambient")
						valueStream >> ambient;
					else if (name == "diffuse")
						valueStream >> diffuse;
					else if (name == "specular")
						valueStream >> specular;
					else if (name == "specularExp")
						valueStream >> specularExp;
				});

				materials.push_back(Material::Phong(color, ambient, diffuse, specular, specularExp));
			}
			else if (strcmp("glossy", material->name()) == 0)
			{
				Vec3 color;

				forEach(material->first_node(), [&](auto& name, auto& value)
				{
					std::stringstream valueStream;
					valueStream << value;

					if (name == "color")
						valueStream >> color;
				});

				materials.push_back(Material::Glossy(color));
			}

			material = material->next_sibling();
		}
	}

	auto lightsNode = root->first_node("lights");
	if (lightsNode)
	{
		auto light = lightsNode->first_node();
		while (light) {

			if (strcmp("point", light->name()) == 0) {
				Vec3 position;
				Vec3 color;
				float intensity = 1.0f;

				forEach(light->first_node(), [&](auto& name, auto& value)
				{
					std::stringstream valueStream;
					valueStream << value;

					if (name == "position")
						valueStream >> position;
					else if (name == "color")
						valueStream >> color;
					else if (name == "intensity")
						valueStream >> intensity;
				});
				
				lights.push_back(Light::Point(position, color, intensity));
			}

			light = light->next_sibling();
		}
	}
}
