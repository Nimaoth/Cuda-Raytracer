#pragma once

#include <cstdint>
#include <string>

class BitMap
{
private:
	struct Color {
		uint8_t r, g, b;

		Color(uint8_t grey) : r(grey), g(grey), b(grey) {}
		Color(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
	};

	enum class Format {
		RGB,
		BGR
	};

	uint8_t* m_data = nullptr;

	uint32_t m_width;
	uint32_t m_height;

public:
	BitMap(uint32_t width, uint32_t height);
	BitMap(const BitMap&) = delete;
	BitMap(BitMap&&) = delete;
	BitMap& operator =(const BitMap&) = delete;
	BitMap& operator =(BitMap&&) = delete;

	~BitMap();

	inline uint8_t* GetRaw() const { return m_data; }
	inline size_t GetRawSize() const { return 3 * m_width * m_height; }


	inline uint32_t GetSize() const { return m_width * m_height; }
	inline uint32_t GetWidth() const { return m_width; }
	inline uint32_t GetHeight() const { return m_height; }

	inline void Set(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b)
	{
		m_data[(x + y * m_width) * 3 + 2] = r;
		m_data[(x + y * m_width) * 3 + 1] = g;
		m_data[(x + y * m_width) * 3 + 0] = b;
	}

	inline void Set(uint32_t x, uint32_t y, const Color& color)
	{
		m_data[(x + y * m_width) * 3 + 2] = color.r;
		m_data[(x + y * m_width) * 3 + 1] = color.g;
		m_data[(x + y * m_width) * 3 + 0] = color.b;
	}

	inline void Set(uint32_t x, uint32_t y, uint8_t grey)
	{
		m_data[(x + y * m_width) * 3 + 2] = grey;
		m_data[(x + y * m_width) * 3 + 1] = grey;
		m_data[(x + y * m_width) * 3 + 0] = grey;
	}

	void Set(uint8_t* pixels, Format format);
	void Set(float* pixels, Format format);

	inline Color Get(uint32_t x, uint32_t y)
	{
		return {
			m_data[(x + y * m_width) * 3 + 2],
			m_data[(x + y * m_width) * 3 + 1],
			m_data[(x + y * m_width) * 3 + 0]
		};
	}

	inline void Get(uint32_t x, uint32_t y, uint8_t& r, uint8_t& g, uint8_t& b)
	{
		r = m_data[(x + y * m_width) * 3 + 2];
		g = m_data[(x + y * m_width) * 3 + 1];
		b = m_data[(x + y * m_width) * 3 + 0];
	}

	inline uint32_t GetR(uint32_t x, uint32_t y) const { return m_data[(x + y * m_width) * 3 + 2]; };
	inline uint32_t GetG(uint32_t x, uint32_t y) const { return m_data[(x + y * m_width) * 3 + 1]; };
	inline uint32_t GetB(uint32_t x, uint32_t y) const { return m_data[(x + y * m_width) * 3 + 0]; };

	void Save(const std::string& filename) const;
};