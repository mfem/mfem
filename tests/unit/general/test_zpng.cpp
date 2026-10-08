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

#include "mfem.hpp"
#include "general/zpng.hpp"

#include "catch.hpp"

#include <cstring>
#include <string>

using namespace mfem;

#ifdef MFEM_USE_ZLIB
#include <zlib.h>
#include <cstdio>
#include <vector>

namespace
{

void ExpectRgb(const png::image &img, unsigned width, unsigned height,
               const unsigned char *rgb, std::size_t n)
{
   REQUIRE(img.width == width);
   REQUIRE(img.height == height);
   REQUIRE(img.rgb.size() == n);
   REQUIRE(std::memcmp(img.rgb.data(), rgb, n) == 0);
}

void PutBe32(std::vector<unsigned char> &o, std::uint32_t v)
{
   o.push_back(static_cast<unsigned char>((v >> 24) & 0xffu));
   o.push_back(static_cast<unsigned char>((v >> 16) & 0xffu));
   o.push_back(static_cast<unsigned char>((v >> 8) & 0xffu));
   o.push_back(static_cast<unsigned char>(v & 0xffu));
}

std::uint32_t Crc(const unsigned char *data, std::size_t n)
{
   uLong c = crc32(0L, Z_NULL, 0);
   while (n > 0)
   {
      const uInt take = static_cast<uInt>(
                           n > static_cast<std::size_t>(0xffffffffu)
                           ? 0xffffffffu : n);
      c = crc32(c, data, take);
      data += take;
      n -= take;
   }
   return static_cast<std::uint32_t>(c & 0xffffffffu);
}

void Chunk(std::vector<unsigned char> &o, const char type[4],
           const unsigned char *data, std::size_t n)
{
   PutBe32(o, static_cast<std::uint32_t>(n));
   const std::size_t type_off = o.size();
   o.insert(o.end(), type, type + 4);
   if (n && data) { o.insert(o.end(), data, data + n); }
   PutBe32(o, Crc(o.data() + type_off, 4 + n));
}

std::vector<unsigned char> Deflate(const unsigned char *src, std::size_t n)
{
   uLongf bound = compressBound(static_cast<uLong>(n));
   std::vector<unsigned char> out(static_cast<std::size_t>(bound));
   const int rc = compress2(out.data(), &bound, src, static_cast<uLong>(n),
                            Z_DEFAULT_COMPRESSION);
   REQUIRE(rc == Z_OK);
   out.resize(static_cast<std::size_t>(bound));
   return out;
}

unsigned char Paeth(int a, int b, int c)
{
   const int p = a + b - c;
   const int pa = p > a ? p - a : a - p;
   const int pb = p > b ? p - b : b - p;
   const int pc = p > c ? p - c : c - p;
   if (pa <= pb && pa <= pc) { return static_cast<unsigned char>(a); }
   if (pb <= pc) { return static_cast<unsigned char>(b); }
   return static_cast<unsigned char>(c);
}

std::vector<unsigned char> Filtered(unsigned width, unsigned height,
                                    unsigned bpp, const unsigned char *pix,
                                    int ftype)
{
   const std::size_t row_n = static_cast<std::size_t>(width) * bpp;
   const std::size_t stride = row_n + 1;
   std::vector<unsigned char> raw(stride * height);
   for (unsigned y = 0; y < height; ++y)
   {
      const unsigned char *cur =
         pix + static_cast<std::size_t>(y) * row_n;
      const unsigned char *prev =
         y ? pix + static_cast<std::size_t>(y - 1) * row_n : nullptr;
      unsigned char *dst =
         raw.data() + static_cast<std::size_t>(y) * stride;
      dst[0] = static_cast<unsigned char>(ftype);
      for (std::size_t x = 0; x < row_n; ++x)
      {
         const int a = x >= bpp ? cur[x - bpp] : 0;
         const int b = prev ? prev[x] : 0;
         const int c = (prev && x >= bpp) ? prev[x - bpp] : 0;
         int pred = 0;
         if (ftype == 1) { pred = a; }
         else if (ftype == 2) { pred = b; }
         else if (ftype == 3) { pred = (a + b) / 2; }
         else if (ftype == 4) { pred = Paeth(a, b, c); }
         dst[x + 1] = static_cast<unsigned char>(cur[x] - pred);
      }
   }
   return raw;
}

struct Built
{
   bool text = false;
   bool split = false;
   bool bad_crc = false;
   bool bad_sig = false;
};

std::vector<unsigned char>
MakePng(unsigned width, unsigned height, int bit_depth, int color_type,
        int interlace, const std::vector<unsigned char> &raw,
        const std::vector<unsigned char> &plte, Built opt)
{
   std::vector<unsigned char> png = {137, 80, 78, 71, 13, 10, 26, 10};
   unsigned char ihdr[13];
   ihdr[0] = static_cast<unsigned char>((width >> 24) & 0xffu);
   ihdr[1] = static_cast<unsigned char>((width >> 16) & 0xffu);
   ihdr[2] = static_cast<unsigned char>((width >> 8) & 0xffu);
   ihdr[3] = static_cast<unsigned char>(width & 0xffu);
   ihdr[4] = static_cast<unsigned char>((height >> 24) & 0xffu);
   ihdr[5] = static_cast<unsigned char>((height >> 16) & 0xffu);
   ihdr[6] = static_cast<unsigned char>((height >> 8) & 0xffu);
   ihdr[7] = static_cast<unsigned char>(height & 0xffu);
   ihdr[8] = static_cast<unsigned char>(bit_depth);
   ihdr[9] = static_cast<unsigned char>(color_type);
   ihdr[10] = 0;
   ihdr[11] = 0;
   ihdr[12] = static_cast<unsigned char>(interlace);
   Chunk(png, "IHDR", ihdr, 13);
   if (!plte.empty()) { Chunk(png, "PLTE", plte.data(), plte.size()); }
   if (opt.text)
   {
      const unsigned char text[] = {'C', 'o', 'm', 'm', 'e', 'n', 't', 0,
                                    'h', 'i'
                                   };
      Chunk(png, "tEXt", text, sizeof(text));
   }
   const std::vector<unsigned char> z =
      raw.empty() ? Deflate(ihdr, 1) : Deflate(raw.data(), raw.size());
   if (!opt.split || z.size() < 2)
   {
      Chunk(png, "IDAT", z.data(), z.size());
   }
   else
   {
      const std::size_t mid = z.size() / 2;
      Chunk(png, "IDAT", z.data(), mid);
      Chunk(png, "IDAT", z.data() + mid, z.size() - mid);
   }
   Chunk(png, "IEND", nullptr, 0);
   if (opt.bad_crc) { png[32] ^= 0xffu; }
   if (opt.bad_sig) { png[0] ^= 0xffu; }
   return png;
}

} // namespace

TEST_CASE("PNG round trip", "[png]")
{
   const unsigned char rgb[12] =
   {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12
   };
   const std::vector<unsigned char> png = png::encode(2, 2, rgb, 12);
   const png::image img = png::read(png.data(), png.size());
   ExpectRgb(img, 2, 2, rgb, 12);
}

TEST_CASE("PNG file round trip", "[png]")
{
   const unsigned char rgb[12] =
   {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12
   };
   const char *path = "zpng_roundtrip.png";
   struct Cleanup
   {
      const char *p;
      ~Cleanup() { std::remove(p); }
   } cleanup {path};

   png::write(path, 2, 2, rgb, sizeof(rgb));
   const png::image img = png::read(std::string(path));
   ExpectRgb(img, 2, 2, rgb, sizeof(rgb));
}

TEST_CASE("PNG color types expand to RGB", "[png]")
{
   // Shared picture: two gray pixels. Alpha bytes must not leak into RGB.
   const unsigned char expect[6] = {8, 8, 8, 16, 16, 16};
   const unsigned char gray[2] = {8, 16};
   const unsigned char ga[4] = {8, 0, 16, 255};
   const unsigned char rgba[8] = {8, 8, 8, 0, 16, 16, 16, 255};
   const unsigned char index[2] = {0, 1};
   const unsigned char plte[6] = {8, 8, 8, 16, 16, 16};

   SECTION("gray")
   {
      const auto raw = Filtered(2, 1, 1, gray, 0);
      const auto png = MakePng(2, 1, 8, 0, 0, raw, {}, {});
      ExpectRgb(png::read(png.data(), png.size()), 2, 1, expect, 6);
   }
   SECTION("palette")
   {
      const auto raw = Filtered(2, 1, 1, index, 0);
      const auto png = MakePng(2, 1, 8, 3, 0, raw, {plte, plte + 6}, {});
      ExpectRgb(png::read(png.data(), png.size()), 2, 1, expect, 6);
   }
   SECTION("gray alpha")
   {
      const auto raw = Filtered(2, 1, 2, ga, 0);
      const auto png = MakePng(2, 1, 8, 4, 0, raw, {}, {});
      ExpectRgb(png::read(png.data(), png.size()), 2, 1, expect, 6);
   }
   SECTION("rgba")
   {
      const auto raw = Filtered(2, 1, 4, rgba, 0);
      const auto png = MakePng(2, 1, 8, 6, 0, raw, {}, {});
      ExpectRgb(png::read(png.data(), png.size()), 2, 1, expect, 6);
   }
}

TEST_CASE("PNG filters", "[png]")
{
   const unsigned char rgb[12] =
   {
      1, 2, 3, 40, 50, 60,
      7, 80, 9, 10, 200, 12
   };
   const int filters[] = {1, 2, 3, 4};
   for (int ftype : filters)
   {
      const auto raw = Filtered(2, 2, 3, rgb, ftype);
      const auto png = MakePng(2, 2, 8, 2, 0, raw, {}, {});
      ExpectRgb(png::read(png.data(), png.size()), 2, 2, rgb, 12);
   }
}

TEST_CASE("PNG ancillary chunk and split IDAT", "[png]")
{
   const unsigned char rgb[3] = {9, 8, 7};
   const auto raw = Filtered(1, 1, 3, rgb, 0);
   Built opt;
   opt.text = true;
   opt.split = true;
   const auto png = MakePng(1, 1, 8, 2, 0, raw, {}, opt);
   ExpectRgb(png::read(png.data(), png.size()), 1, 1, rgb, 3);
}

TEST_CASE("PNG rejects unsupported files", "[png]")
{
   const unsigned char pix[3] = {1, 2, 3};
   const auto raw = Filtered(1, 1, 3, pix, 0);

   SECTION("bad signature")
   {
      Built opt;
      opt.bad_sig = true;
      const auto png = MakePng(1, 1, 8, 2, 0, raw, {}, opt);
      REQUIRE_THROWS_WITH(png::read(png.data(), png.size()),
                          Catch::Matchers::Contains("not a PNG"));
   }
   SECTION("bad CRC")
   {
      Built opt;
      opt.bad_crc = true;
      const auto png = MakePng(1, 1, 8, 2, 0, raw, {}, opt);
      REQUIRE_THROWS_WITH(png::read(png.data(), png.size()),
                          Catch::Matchers::Contains("bad CRC"));
   }
   SECTION("bit depth 16")
   {
      const auto png = MakePng(1, 1, 16, 2, 0, raw, {}, {});
      REQUIRE_THROWS_WITH(png::read(png.data(), png.size()),
                          Catch::Matchers::Contains(
                             "only 8-bit PNG is supported"));
   }
   SECTION("Adam7")
   {
      const auto png = MakePng(1, 1, 8, 2, 1, raw, {}, {});
      REQUIRE_THROWS_WITH(
         png::read(png.data(), png.size()),
         Catch::Matchers::Contains(
            "interlaced (Adam7) PNG is not supported"));
   }
}

#else

TEST_CASE("PNG requires zlib", "[png]")
{
   REQUIRE_THROWS_WITH(png::read(std::string("x")),
                       Catch::Matchers::Contains("zlib is required"));
}

#endif
