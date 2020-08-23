#pragma once

#include "Raytracer.cuh"
#include <vector>
#include <string>
#include "Log.h"

#define TRY(x) { try { x; } catch (std::exception& e) { LogError(e.what()); }}

void loadConfiguration(const std::string& filename, RenderData& data, std::vector<Object>& objects, std::vector<Material>& materials, std::vector<Light>& lights, std::string& outFile);