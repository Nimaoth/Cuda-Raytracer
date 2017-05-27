#include "Bitmap.h"

#include <fstream>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

BitMap::BitMap(uint32_t width, uint32_t height)
	: m_width(width), m_height(height)
{
	m_data = new uint8_t[3 * width * height];
	memset(m_data, 0, 3 * width * height);
}

BitMap::~BitMap()
{
	delete[] m_data;
}

void BitMap::Set(uint8_t* pixels, Format format)
{
	for (uint32_t i = 0; i < 3 * m_width * m_height; i += 3) {
		switch (format)
		{
		case BitMap::Format::RGB:
			m_data[i + 0] = pixels[i + 2];
			m_data[i + 1] = pixels[i + 1];
			m_data[i + 2] = pixels[i + 0];
			break;
		case BitMap::Format::BGR:
			m_data[i + 0] = pixels[i + 0];
			m_data[i + 1] = pixels[i + 1];
			m_data[i + 2] = pixels[i + 2];
			break;
		default:
			break;
		}
	}
}

void BitMap::Set(float* pixels, Format format)
{
	for (uint32_t i = 0; i < 3 * m_width * m_height; i += 3) {
		switch (format)
		{
		case BitMap::Format::RGB:
			m_data[i + 0] = (uint8_t) (pixels[i + 2] * 256);
			m_data[i + 1] = (uint8_t) (pixels[i + 1] * 256);
			m_data[i + 2] = (uint8_t) (pixels[i + 0] * 256);
			break;
		case BitMap::Format::BGR:
			m_data[i + 0] = (uint8_t) (pixels[i + 0] * 256);
			m_data[i + 1] = (uint8_t) (pixels[i + 1] * 256);
			m_data[i + 2] = (uint8_t) (pixels[i + 2] * 256);
			break;
		default:
			break;
		}
	}
}

void BitMap::Save(const std::string & filename) const
{
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open())
		throw std::exception(("Failed to open file for writing: " + filename).c_str());

	uint32_t filesize = 54 + 3 * m_width * m_height;
	uint32_t padding = (4 - (3 * m_width) % 4) % 4;

	char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
	char pad[3] = { 0, 0, 0 };

	bmpfileheader[2] = (char)(filesize);
	bmpfileheader[3] = (char)(filesize >> 8);
	bmpfileheader[4] = (char)(filesize >> 16);
	bmpfileheader[5] = (char)(filesize >> 24);

	bmpinfoheader[4]  = (char)(m_width);
	bmpinfoheader[5]  = (char)(m_width >> 8);
	bmpinfoheader[6]  = (char)(m_width >> 16);
	bmpinfoheader[7]  = (char)(m_width >> 24);
	bmpinfoheader[8]  = (char)(m_height);
	bmpinfoheader[9]  = (char)(m_height >> 8);
	bmpinfoheader[10] = (char)(m_height >> 16);
	bmpinfoheader[11] = (char)(m_height >> 24);

	file.write(bmpfileheader, sizeof(bmpfileheader));
	file.write(bmpinfoheader, sizeof(bmpinfoheader));

	for (int i = m_height - 1; i >= 0; i--) {
		
		uint32_t offset = 3 * m_width * i;
		file.write(reinterpret_cast<const char*>(m_data) + offset, 3 * m_width);
		
		// write padding
		file.write(pad, padding);
	}

	file.close();
}
