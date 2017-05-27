#include "Objects.h"

Object Object::Sphere(const Vec3& position, float radius, int material)
{
	Object o;
	o.type = Object::SPHERE;
	o.sphere.position = position;
	o.sphere.radius = radius;
	o.material = material;
	return o;
}

Light Light::Point(const Vec3& position, const Vec3& color, float intensity)
{
	Light l;
	l.type = Light::POINT;
	l.point.position = position;
	l.color = color;
	l.intensity = intensity;
	return l;
}

Material Material::Phong(const Vec3& color, float ambient, float diffuse, float specular, float specularExp)
{
	Material m;
	m.type = Material::PHONG;
	m.color = color;
	m.phong.ambient = ambient;
	m.phong.diffuse = diffuse;
	m.phong.specular = specular;
	m.phong.specularExp = specularExp;
	return m;
}

Material Material::Glossy(const Vec3 & color)
{
	Material m;
	m.type = Material::GLOSSY;
	m.color = color;
	return m;
}
