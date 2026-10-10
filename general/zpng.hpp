// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
#pragma once

#include "../config/config.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MFEM_USE_ZLIB
#include <zlib.h>
#else
#include "error.hpp"
#endif

namespace mfem::png
{

/// 8-bit RGB image. Row 0 is the top row, pixels are tightly packed.
struct image
{
   unsigned width = 0;
   unsigned height = 0;
   std::vector<unsigned char> rgb;
};

/// Thrown when a PNG cannot be read or written.
class png_error : public std::runtime_error
{
public:
   using std::runtime_error::runtime_error;
};

/// Read a PNG file and expand it to 8-bit RGB.
[[nodiscard]] inline image read(const std::string &path);

/// Read a PNG from memory and expand it to 8-bit RGB.
[[nodiscard]] inline image read(const unsigned char *bytes, std::size_t size);

/// Write an 8-bit RGB buffer as a PNG file.
inline void write(const std::string &path, unsigned width, unsigned height,
                  const unsigned char *rgb, std::size_t nbytes);

/// Encode an 8-bit RGB buffer as a PNG byte stream.
[[nodiscard]] inline std::vector<unsigned char>
encode(unsigned width, unsigned height, const unsigned char *rgb,
       std::size_t nbytes);

namespace detail
{

constexpr unsigned char kPngSign[8] = {137, 80, 78, 71, 13, 10, 26, 10};

[[noreturn]] inline void Fail(const std::string &why)
{
   throw png_error(why);
}

inline std::uint32_t GetBe32(const unsigned char *p)
{
   return (static_cast<std::uint32_t>(p[0]) << 24) |
          (static_cast<std::uint32_t>(p[1]) << 16) |
          (static_cast<std::uint32_t>(p[2]) << 8) |
          static_cast<std::uint32_t>(p[3]);
}

inline void StoreBe32(unsigned char *p, std::uint32_t v)
{
   p[0] = static_cast<unsigned char>((v >> 24) & 0xffu);
   p[1] = static_cast<unsigned char>((v >> 16) & 0xffu);
   p[2] = static_cast<unsigned char>((v >> 8) & 0xffu);
   p[3] = static_cast<unsigned char>(v & 0xffu);
}

inline void PutBe32(std::vector<unsigned char> &o, std::uint32_t v)
{
   unsigned char b[4];
   StoreBe32(b, v);
   o.insert(o.end(), b, b + 4);
}

inline bool Mul(std::size_t a, std::size_t b, std::size_t &out)
{
   if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
   {
      return false;
   }
   out = a * b;
   return true;
}

inline int Samples(int color_type)
{
   switch (color_type)
   {
      case 0: return 1; // gray
      case 2: return 3; // RGB
      case 3: return 1; // palette index
      case 4: return 2; // gray+alpha
      case 6: return 4; // RGBA
      default: return 0;
   }
}

inline unsigned char Paeth(int a, int b, int c)
{
   const int p = a + b - c;
   const int pa = p > a ? p - a : a - p;
   const int pb = p > b ? p - b : b - p;
   const int pc = p > c ? p - c : c - p;
   if (pa <= pb && pa <= pc) { return static_cast<unsigned char>(a); }
   if (pb <= pc) { return static_cast<unsigned char>(b); }
   return static_cast<unsigned char>(c);
}

/// Inverse PNG filters.
inline void Unfilter(const unsigned char *raw, unsigned char *recon,
                     unsigned width, unsigned height, std::size_t stride,
                     unsigned bpp)
{
   const std::size_t row_n = static_cast<std::size_t>(width) * bpp;
   const unsigned char *prev = nullptr;
   for (unsigned y = 0; y < height; ++y)
   {
      const unsigned char *row = raw + static_cast<std::size_t>(y) * stride;
      const int ftype = row[0];
      if (ftype > 4) { Fail("unknown PNG filter"); }
      unsigned char *out = recon + static_cast<std::size_t>(y) * row_n;
      const unsigned char *filt = row + 1;
      for (std::size_t x = 0; x < row_n; ++x)
      {
         const int a = (x >= bpp) ? out[x - bpp] : 0;
         const int b = prev ? prev[x] : 0;
         const int c = (prev && x >= bpp) ? prev[x - bpp] : 0;
         unsigned char v = filt[x];
         switch (ftype)
         {
            case 0: break; // None
            case 1: v = static_cast<unsigned char>(v + a); break; // Sub
            case 2: v = static_cast<unsigned char>(v + b); break; // Up
            case 3: // Average
               v = static_cast<unsigned char>(
                      v + static_cast<unsigned char>((a + b) / 2));
               break;
            case 4: // Paeth
               v = static_cast<unsigned char>(v + Paeth(a, b, c));
               break;
            default: Fail("unknown PNG filter");
         }
         out[x] = v;
      }
      prev = out;
   }
}

inline void ToRgb(const unsigned char *samples, unsigned char *rgb,
                  unsigned width, unsigned height, int color_type,
                  unsigned bpp, const unsigned char *plte, std::size_t plte_n)
{
   const std::size_t npx = static_cast<std::size_t>(width) * height;
   if (color_type == 0 || color_type == 4)
   {
      for (std::size_t i = 0; i < npx; ++i)
      {
         const unsigned char v = samples[i * bpp];
         rgb[i * 3] = v;
         rgb[i * 3 + 1] = v;
         rgb[i * 3 + 2] = v;
      }
   }
   else if (color_type == 2 || color_type == 6)
   {
      for (std::size_t i = 0; i < npx; ++i)
      {
         const std::size_t si = i * bpp;
         rgb[i * 3] = samples[si];
         rgb[i * 3 + 1] = samples[si + 1];
         rgb[i * 3 + 2] = samples[si + 2];
      }
   }
   else
   {
      for (std::size_t i = 0; i < npx; ++i)
      {
         const unsigned idx = samples[i * bpp];
         if (static_cast<std::size_t>(idx) * 3u + 2u >= plte_n)
         {
            Fail("palette index out of range");
         }
         rgb[i * 3] = plte[idx * 3];
         rgb[i * 3 + 1] = plte[idx * 3 + 1];
         rgb[i * 3 + 2] = plte[idx * 3 + 2];
      }
   }
}

inline std::vector<unsigned char> ReadFile(const std::string &path)
{
   std::ifstream in(path.c_str(), std::ios::binary);
   if (!in) { Fail("cannot open"); }
   in.seekg(0, std::ios::end);
   if (!in) { Fail("seek failed"); }
   const std::streamoff end = in.tellg();
   if (end < 8) { Fail("too small"); }
   in.seekg(0, std::ios::beg);
   if (!in) { Fail("seek failed"); }
   std::vector<unsigned char> buf(static_cast<std::size_t>(end));
   in.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(end));
   if (in.gcount() != end) { Fail("short read"); }
   return buf;
}

#ifdef MFEM_USE_ZLIB

inline std::uint32_t Crc(const unsigned char *data, std::size_t n)
{
   uLong c = crc32(0L, Z_NULL, 0);
   while (n > 0)
   {
      const auto cap =
         static_cast<std::size_t>(std::numeric_limits<uInt>::max());
      const uInt take = static_cast<uInt>(n > cap ? cap : n);
      c = crc32(c, data, take);
      data += take;
      n -= take;
   }
   return static_cast<std::uint32_t>(c & 0xffffffffu);
}

inline void Chunk(std::vector<unsigned char> &o, const char type[4],
                  const unsigned char *data, std::size_t n)
{
   PutBe32(o, static_cast<std::uint32_t>(n));
   const std::size_t type_off = o.size();
   o.insert(o.end(), type, type + 4);
   if (n && data) { o.insert(o.end(), data, data + n); }
   const std::uint32_t crc = Crc(o.data() + type_off, 4 + n);
   PutBe32(o, crc);
}

inline std::vector<unsigned char> Deflate(const unsigned char *src,
                                          std::size_t n)
{
   if (n > static_cast<std::size_t>(std::numeric_limits<uLong>::max()))
   {
      Fail("zlib deflate failed");
   }
   uLongf bound = compressBound(static_cast<uLong>(n));
   std::vector<unsigned char> out(static_cast<std::size_t>(bound));
   const int rc = compress2(out.data(), &bound, src, static_cast<uLong>(n),
                            Z_DEFAULT_COMPRESSION);
   if (rc != Z_OK || bound == 0) { Fail("zlib deflate failed"); }
   out.resize(static_cast<std::size_t>(bound));
   return out;
}

inline void Inflate(const unsigned char *src, std::size_t nsrc,
                    unsigned char *dst, std::size_t ndst)
{
   if (nsrc > static_cast<std::size_t>(std::numeric_limits<uLong>::max()) ||
       ndst > static_cast<std::size_t>(std::numeric_limits<uLong>::max()))
   {
      Fail("zlib inflate failed");
   }
   auto dest_len = static_cast<uLongf>(ndst);
   const int rc = uncompress(dst, &dest_len, src, static_cast<uLong>(nsrc));
   if (rc == Z_MEM_ERROR) { Fail("zlib out of memory"); }
   if (rc == Z_BUF_ERROR) { Fail("zlib output buffer too small"); }
   if (rc == Z_DATA_ERROR) { Fail("zlib data error"); }
   if (rc != Z_OK) { Fail("zlib inflate failed"); }
   if (static_cast<std::size_t>(dest_len) != ndst)
   {
      Fail("zlib inflate size mismatch");
   }
}

struct parsed
{
   unsigned width = 0;
   unsigned height = 0;
   int color_type = 0;
   std::vector<unsigned char> plte;
   std::vector<unsigned char> idat;
};

inline parsed Parse(const unsigned char *buf, std::size_t size)
{
   if (!buf || size < 8 || std::memcmp(buf, kPngSign, 8) != 0)
   {
      Fail("not a PNG");
   }

   parsed out;
   bool saw_ihdr = false;
   bool saw_iend = false;
   bool saw_idat = false;
   bool idat_closed = false;

   std::size_t off = 8;
   while (off <= size && size - off >= 12)
   {
      const std::uint32_t len = GetBe32(buf + off);
      if (size - off - 12 < len) { Fail("truncated chunk"); }
      const unsigned char *type = buf + off + 4;
      const unsigned char *data = buf + off + 8;
      const std::uint32_t crc_got = GetBe32(buf + off + 8 + len);
      const std::size_t crc_n = static_cast<std::size_t>(len) + 4u;
      const std::uint32_t crc_exp = Crc(type, crc_n);
      if (crc_got != crc_exp) { Fail("bad CRC"); }

      const bool ancillary = (type[0] & 0x20u) != 0;
      if (std::memcmp(type, "IHDR", 4) == 0)
      {
         if (saw_ihdr || off != 8) { Fail("IHDR not first"); }
         if (len != 13) { Fail("bad IHDR length"); }
         saw_ihdr = true;
         const std::uint32_t w = GetBe32(data);
         const std::uint32_t h = GetBe32(data + 4);
         if (w == 0 || h == 0 || w > 0x7fffffffu || h > 0x7fffffffu)
         {
            Fail("invalid dimensions");
         }
         out.width = static_cast<unsigned>(w);
         out.height = static_cast<unsigned>(h);
         const int bit_depth = data[8];
         out.color_type = data[9];
         const int compression = data[10];
         const int filter = data[11];
         const int interlace = data[12];
         if (bit_depth != 8) { Fail("only 8-bit PNG is supported"); }
         if (Samples(out.color_type) == 0) { Fail("unsupported color type"); }
         if (compression != 0) { Fail("bad compression"); }
         if (filter != 0) { Fail("bad filter method"); }
         if (interlace != 0)
         {
            Fail("interlaced (Adam7) PNG is not supported");
         }
      }
      else if (std::memcmp(type, "PLTE", 4) == 0)
      {
         if (!saw_ihdr) { Fail("PLTE before IHDR"); }
         if (len == 0 || (len % 3u) != 0 || len > 768u) { Fail("bad PLTE"); }
         if (saw_idat) { idat_closed = true; }
         out.plte.assign(data, data + len);
      }
      else if (std::memcmp(type, "IDAT", 4) == 0)
      {
         if (!saw_ihdr) { Fail("IDAT before IHDR"); }
         if (idat_closed) { Fail("IDAT chunks are not consecutive"); }
         saw_idat = true;
         out.idat.insert(out.idat.end(), data, data + len);
      }
      else if (std::memcmp(type, "IEND", 4) == 0)
      {
         saw_iend = true;
         break;
      }
      else if (!ancillary)
      {
         Fail("unsupported critical chunk");
      }
      else if (saw_idat)
      {
         idat_closed = true;
      }

      off += 12u + static_cast<std::size_t>(len);
   }

   if (!saw_ihdr) { Fail("missing IHDR"); }
   if (!saw_iend) { Fail("missing IEND"); }
   if (out.idat.empty()) { Fail("missing IDAT"); }
   if (out.color_type == 3 && out.plte.empty())
   {
      Fail("palette PNG missing PLTE");
   }
   return out;
}

inline image Decode(const unsigned char *bytes, std::size_t size)
{
   const parsed src = Parse(bytes, size);
   const unsigned spp = static_cast<unsigned>(Samples(src.color_type));
   std::size_t row_n = 0;
   if (!Mul(src.width, spp, row_n) ||
       row_n == std::numeric_limits<std::size_t>::max())
   {
      Fail("invalid dimensions");
   }
   const std::size_t stride = row_n + 1;
   std::size_t raw_n = 0;
   std::size_t recon_n = 0;
   std::size_t rgb_row = 0;
   std::size_t rgb_n = 0;
   if (!Mul(stride, src.height, raw_n) || !Mul(row_n, src.height, recon_n) ||
       !Mul(src.width, 3, rgb_row) || !Mul(rgb_row, src.height, rgb_n))
   {
      Fail("invalid dimensions");
   }

   std::vector<unsigned char> raw(raw_n);
   std::vector<unsigned char> recon(recon_n);
   Inflate(src.idat.data(), src.idat.size(), raw.data(), raw_n);
   Unfilter(raw.data(), recon.data(), src.width, src.height, stride, spp);

   image img;
   img.width = src.width;
   img.height = src.height;
   img.rgb.resize(rgb_n);
   ToRgb(recon.data(), img.rgb.data(), src.width, src.height, src.color_type,
         spp, src.plte.data(), src.plte.size());
   return img;
}

inline std::vector<unsigned char> Build(unsigned width, unsigned height,
                                        const unsigned char *rgb,
                                        std::size_t nbytes)
{
   if (width == 0 || height == 0 || width > 0x7fffffffu ||
       height > 0x7fffffffu)
   {
      Fail("invalid dimensions");
   }
   std::size_t row3 = 0;
   std::size_t rgb_n = 0;
   if (!Mul(width, 3, row3) || !Mul(row3, height, rgb_n) ||
       row3 == std::numeric_limits<std::size_t>::max())
   {
      Fail("invalid dimensions");
   }
   if (!rgb) { Fail("null image"); }
   if (nbytes < rgb_n) { Fail("truncated image buffer"); }

   const std::size_t stride = row3 + 1;
   std::size_t raw_n = 0;
   if (!Mul(stride, height, raw_n)) { Fail("invalid dimensions"); }

   std::vector<unsigned char> raw(raw_n);
   for (unsigned y = 0; y < height; ++y)
   {
      const std::size_t off = static_cast<std::size_t>(y) * stride;
      raw[off] = 0; // filter None
      std::memcpy(raw.data() + off + 1,
                  rgb + static_cast<std::size_t>(y) * row3, row3);
   }

   const std::vector<unsigned char> z = Deflate(raw.data(), raw_n);
   std::vector<unsigned char> png = {137, 80, 78, 71, 13, 10, 26, 10};
   unsigned char ihdr[13];
   StoreBe32(ihdr, width);
   StoreBe32(ihdr + 4, height);
   ihdr[8] = 8;  // bit depth
   ihdr[9] = 2;  // truecolor
   ihdr[10] = 0;
   ihdr[11] = 0;
   ihdr[12] = 0;
   Chunk(png, "IHDR", ihdr, 13);
   Chunk(png, "IDAT", z.data(), z.size());
   Chunk(png, "IEND", nullptr, 0);
   return png;
}

#endif // MFEM_USE_ZLIB

} // namespace detail

inline image read(const std::string &path)
{
#ifndef MFEM_USE_ZLIB
   MFEM_CONTRACT_VAR(path);
   throw png_error("zlib is required");
#else
   try
   {
      const std::vector<unsigned char> buf = detail::ReadFile(path);
      return read(buf.data(), buf.size());
   }
   catch (const png_error &ex)
   {
      throw png_error(std::string("Failed to load PNG: ") + path + " (" +
                      ex.what() + ")");
   }
#endif
}

inline image read(const unsigned char *bytes, std::size_t size)
{
#ifndef MFEM_USE_ZLIB
   MFEM_CONTRACT_VAR(bytes);
   MFEM_CONTRACT_VAR(size);
   throw png_error("zlib is required");
#else
   return detail::Decode(bytes, size);
#endif
}

inline void write(const std::string &path, unsigned width, unsigned height,
                  const unsigned char *rgb, std::size_t nbytes)
{
#ifndef MFEM_USE_ZLIB
   MFEM_CONTRACT_VAR(path);
   MFEM_CONTRACT_VAR(width);
   MFEM_CONTRACT_VAR(height);
   MFEM_CONTRACT_VAR(rgb);
   MFEM_CONTRACT_VAR(nbytes);
   throw png_error("zlib is required");
#else
   const std::vector<unsigned char> png = encode(width, height, rgb, nbytes);
   std::ofstream out(path.c_str(), std::ios::binary);
   if (!out) { throw png_error("cannot open"); }
   out.write(reinterpret_cast<const char *>(png.data()),
             static_cast<std::streamsize>(png.size()));
   if (!out) { throw png_error("short write"); }
#endif
}

inline std::vector<unsigned char>
encode(unsigned width, unsigned height, const unsigned char *rgb,
       std::size_t nbytes)
{
#ifndef MFEM_USE_ZLIB
   MFEM_CONTRACT_VAR(width);
   MFEM_CONTRACT_VAR(height);
   MFEM_CONTRACT_VAR(rgb);
   MFEM_CONTRACT_VAR(nbytes);
   throw png_error("zlib is required");
#else
   return detail::Build(width, height, rgb, nbytes);
#endif
}

} // namespace mfem::png
