// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_NVVP_HPP
#define MFEM_NVVP_HPP

#include <string>
#include <cassert>

#include "globals.hpp"
#include "backends.hpp"

#if defined(MFEM_USE_CUDA) //||1
#include <cuda.h>
#include <nvToolsExt.h>
#include <cudaProfiler.h>
#include <cuda_runtime.h>
#else // MFEM_USE_CUDA
typedef struct
{
   uint16_t version;
   uint16_t size;
   uint32_t category;
   int32_t colorType;
   uint32_t color;
   int32_t payloadType;
   int32_t reserved0;
   int32_t messageType;
   struct { const char* ascii; } message;
} nvtxEventAttributes_t;
#define NVTX_VERSION 2
#define NVTX_COLOR_ARGB 1
#define NVTX_MESSAGE_TYPE_ASCII 1
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE \
    ((uint16_t)(sizeof(nvtxEventAttributes_t)))
inline int nvtxRangePushEx(const nvtxEventAttributes_t*) { return 0; }
inline int nvtxRangePop(void) { return 0; }
#endif // MFEM_USE_CUDA

namespace mfem
{

// en.wikipedia.org/wiki/Web_colors#Hex_triplet
typedef enum
{
   Black, NavyBlue, DarkBlue, MediumBlue, Blue, DarkGreen, WebGreen, Teal,
   DarkCyan, DeepSkyBlue, DarkTurquoise, MediumSpringGreen, Green, Lime,
   SpringGreen, Aqua, Cyan, MidnightBlue, DodgerBlue, LightSeaGreen,
   ForestGreen, SeaGreen, DarkSlateGray, LimeGreen, MediumSeaGreen,
   Turquoise, RoyalBlue, SteelBlue, DarkSlateBlue, MediumTurquoise, Indigo,
   DarkOliveGreen, CadetBlue, Cornflower, RebeccaPurple, MediumAquamarine,
   DimGray, SlateBlue, OliveDrab, SlateGray, LightSlateGray,
   MediumSlateBlue, LawnGreen, WebMaroon, WebPurple, Chartreuse,
   Aquamarine, Olive, WebGray, SkyBlue, LightSkyBlue, BlueViolet, DarkRed,
   DarkMagenta, SaddleBrown, DarkSeaGreen, LightGreen, MediumPurple,
   DarkViolet, PaleGreen, DarkOrchid, YellowGreen, Purple, Sienna, Brown,
   DarkGray, LightBlue, GreenYellow, PaleTurquoise, Maroon,
   LightSteelBlue, PowderBlue, Firebrick, DarkGoldenrod, MediumOrchid,
   RosyBrown, DarkKhaki, Gray, Silver, MediumVioletRed, IndianRed, Peru,
   Chocolate, Tan, LightGray, Thistle, Orchid, Goldenrod, PaleVioletRed,
   Crimson, Gainsboro, Plum, Burlywood, LightCyan, Lavender, DarkSalmon,
   Violet, PaleGoldenrod, LightCoral, Khaki, AliceBlue, Honeydew, Azure,
   SandyBrown, Wheat, Beige, WhiteSmoke, MintCream, GhostWhite, Salmon,
   AntiqueWhite, Linen, LightGoldenrod, OldLace, Red, Fuchsia, Magenta,
   DeepPink, OrangeRed, Tomato, HotPink, Coral, DarkOrange, LightSalmon,
   Orange, LightPink, Pink, Gold, PeachPuff, NavajoWhite, Moccasin,
   Bisque, MistyRose, BlanchedAlmond, PapayaWhip, LavenderBlush, Seashell,
   Cornsilk, LemonChiffon, FloralWhite, Snow, Yellow, LightYellow, Ivory,
   White
} COLOR_NAMES;

static constexpr uint32_t HEX_COLORS[] =
{
   0x000000, 0x000080, 0x00008B, 0x0000CD, 0x0000FF, 0x006400, 0x008000,
   0x008080, 0x008B8B, 0x00BFFF, 0x00CED1, 0x00FA9A, 0x00FF00, 0x00FF00,
   0x00FF7F, 0x00FFFF, 0x00FFFF, 0x191970, 0x1E90FF, 0x20B2AA, 0x228B22,
   0x2E8B57, 0x2F4F4F, 0x32CD32, 0x3CB371, 0x40E0D0, 0x4169E1, 0x4682B4,
   0x483D8B, 0x48D1CC, 0x4B0082, 0x556B2F, 0x5F9EA0, 0x6495ED, 0x663399,
   0x66CDAA, 0x696969, 0x6A5ACD, 0x6B8E23, 0x708090, 0x778899, 0x7B68EE,
   0x7CFC00, 0x7F0000, 0x7F007F, 0x7FFF00, 0x7FFFD4, 0x808000, 0x808080,
   0x87CEEB, 0x87CEFA, 0x8A2BE2, 0x8B0000, 0x8B008B, 0x8B4513, 0x8FBC8F,
   0x90EE90, 0x9370DB, 0x9400D3, 0x98FB98, 0x9932CC, 0x9ACD32, 0xA020F0,
   0xA0522D, 0xA52A2A, 0xA9A9A9, 0xADD8E6, 0xADFF2F, 0xAFEEEE, 0xB03060,
   0xB0C4DE, 0xB0E0E6, 0xB22222, 0xB8860B, 0xBA55D3, 0xBC8F8F, 0xBDB76B,
   0xBEBEBE, 0xC0C0C0, 0xC71585, 0xCD5C5C, 0xCD853F, 0xD2691E, 0xD2B48C,
   0xD3D3D3, 0xD8BFD8, 0xDA70D6, 0xDAA520, 0xDB7093, 0xDC143C, 0xDCDCDC,
   0xDDA0DD, 0xDEB887, 0xE0FFFF, 0xE6E6FA, 0xE9967A, 0xEE82EE, 0xEEE8AA,
   0xF08080, 0xF0E68C, 0xF0F8FF, 0xF0FFF0, 0xF0FFFF, 0xF4A460, 0xF5DEB3,
   0xF5F5DC, 0xF5F5F5, 0xF5FFFA, 0xF8F8FF, 0xFA8072, 0xFAEBD7, 0xFAF0E6,
   0xFAFAD2, 0xFDF5E6, 0xFF0000, 0xFF00FF, 0xFF00FF, 0xFF1493, 0xFF4500,
   0xFF6347, 0xFF69B4, 0xFF7F50, 0xFF8C00, 0xFFA07A, 0xFFA500, 0xFFB6C1,
   0xFFC0CB, 0xFFD700, 0xFFDAB9, 0xFFDEAD, 0xFFE4B5, 0xFFE4C4, 0xFFE4E1,
   0xFFEBCD, 0xFFEFD5, 0xFFF0F5, 0xFFF5EE, 0xFFF8DC, 0xFFFACD, 0xFFFAF0,
   0xFFFAFA, 0xFFFF00, 0xFFFFE0, 0xFFFFF0, 0xFFFFFF
};

static const int NUM_HEX_COLORS = sizeof(HEX_COLORS)/sizeof(uint32_t);

class Nvtx
{
   const bool nvtx = false;
   const bool enforce_kernel_sync = false;
   const char *base, *file;
   const uint32_t color = Black;
   mutable std::string ascii;
   mutable nvtxEventAttributes_t event;

public:
   Nvtx() { }

   Nvtx(bool enforce_kernel_sync,
        const char *FILE, const int LINE, const char *FUNC, uint32_t COLOR):
      nvtx(true),
      enforce_kernel_sync(enforce_kernel_sync),
      base(Strrnchr(FILE,'/', 2)),
      file(base ? base + 1 : FILE),
      color(COLOR),
      ascii(file),
      event({})
   {
      event.version = NVTX_VERSION;
      event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      event.colorType = NVTX_COLOR_ARGB;
      event.color = HEX_COLORS[color % NUM_HEX_COLORS];
      event.messageType = NVTX_MESSAGE_TYPE_ASCII;

      ascii += ":";
      ascii += std::to_string(LINE);
      ascii += ":[";
      ascii += FUNC;
      ascii += "] ";
   }

   Nvtx(const char *title, int color = Wheat, bool enforce_kernel_sync = true):
      nvtx(true),
      enforce_kernel_sync(enforce_kernel_sync),
      color(color),
      ascii(title),
      event({})
   {
      event.version = NVTX_VERSION;
      event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      event.colorType = NVTX_COLOR_ARGB;
      event.color = HEX_COLORS[color % NUM_HEX_COLORS];
      event.messageType = NVTX_MESSAGE_TYPE_ASCII;
      event.message.ascii = ascii.c_str();
      nvtxRangePushEx(&event); // push
   }

   ~Nvtx()
   {
      if (!nvtx) { return; }
      if (enforce_kernel_sync)
      {
         nvtxEventAttributes_t eks = {};
         eks.version = NVTX_VERSION;
         eks.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
         eks.category = 0; // user value
         eks.colorType = NVTX_COLOR_ARGB;
         eks.messageType = NVTX_MESSAGE_TYPE_ASCII;
         eks.message.ascii = "Sync";
         eks.color = HEX_COLORS[Yellow];
         nvtxRangePushEx(&eks);
         MFEM_STREAM_SYNC;
         nvtxRangePop();
      }
      nvtxRangePop(); // pop
   }

   // used through MFEM_NVTX
   inline void operator()() const noexcept
   {
      event.message.ascii = ascii.c_str();
      nvtxRangePushEx(&event); // push
   }

   template<typename T>
   inline void operator()(const T &arg) const noexcept
   {
      if (!nvtx) { return; }
      operator<<(arg);
      event.message.ascii = ascii.c_str();
      nvtxRangePushEx(&event); // push
   }

   template<typename T, typename... Args>
   inline void operator()(const char *fmt, const T &arg,
                          Args... args) const noexcept
   {
      if (!nvtx) { return; }

      for (; *fmt != '\0'; fmt++ )
      {
         if (*fmt == '%')
         {
            fmt++;
            const char c = *fmt;
            if (c == 'p') { operator<<(arg); }
            if (c == 's' || c == 'd' || c == 'f') { operator<<(arg); }
            if (c == 'x' || c == 'X')
            {
               mfem::out << std::hex;
               if (c == 'X') { mfem::out << std::uppercase; }
               operator<<(arg);
               mfem::out << std::nouppercase << std::dec;
            }
            if (c == '.')
            {
               fmt++;
               const char c = *fmt;
               char num[8] = { 0 };
               for (int k = 0; *fmt != '\0'; fmt++, k++)
               {
                  if (*fmt == 'e' || *fmt == 'f') { break; }
                  if (*fmt < 0x30 || *fmt > 0x39) { break; }
                  num[k] = *fmt;
               }
               const int fx = std::atoi(num);
               if (c == 'e') { mfem::out << std::scientific; }
               if (c == 'f') { mfem::out << std::fixed; }
               mfem::out << std::setprecision(fx);
               operator<<(arg);
               mfem::out << std::setprecision(6);
            }
            return operator()(fmt + 1, args...);
         }
         operator<<(*fmt);
      }
      // should never be here
      assert(false);
   }

   template <typename T>
   inline void operator<<(const T &arg) const noexcept
   {
      if (!nvtx) { return; }
      ascii += arg;
   }

   inline void operator<<(const int &arg) const noexcept
   {
      if (!nvtx) { return; }
      ascii += std::to_string(arg);
   }


public:
   static const Nvtx Set(const char *FILE, const int LINE, const char *FUNC,
                         uint32_t COLOR)
   {
      static bool env_nvtx = false;
      static bool env_eks = false;
      static bool ini_nvtx = false;
      if (!ini_nvtx)
      {
         env_nvtx = getenv("MFEM_NVTX") != nullptr;
         env_eks = getenv("MFEM_EKS") != nullptr;
         ini_nvtx = true;
      }
      return env_nvtx ? Nvtx(env_eks, FILE, LINE, FUNC, COLOR) : Nvtx();
   }

private:
   inline const char *Strrnchr(const char *s, const unsigned char c, int n)
   {
      size_t len = strlen(s);
      char *p = const_cast<char*>(s) + len - 1;
      for (; n; n--,p--,len--)
      {
         for (; len; p--,len--)
            if (*p == c) { break; }
         if (!len) { return nullptr; }
         if (n == 1) { return p; }
      }
      return nullptr;
   }
};

#ifndef MFEM_NVTX_COLOR
#define MFEM_NVTX_COLOR SeaGreen
#endif

// Helpers for generating unique variable names
#define NVTX_PRIVATE_NAME(name) NVTX_PRIVATE_CONCAT(name, __LINE__)
#define NVTX_PRIVATE_CONCAT(a, b) NVTX_PRIVATE_CONCAT2(a, b)
#define NVTX_PRIVATE_CONCAT2(a, b) a##b

// temporary object which is only alive in the expression
// __PRETTY_FUNCTION__
#define NVTX(...) \
    mfem::Nvtx NVTX_PRIVATE_NAME(nvtx) = \
    mfem::Nvtx::Set(__FILE__,__LINE__,__FUNCTION__,MFEM_NVTX_COLOR);\
    NVTX_PRIVATE_NAME(nvtx).operator()(__VA_ARGS__)

#define MFEM_NVTX NVTX()

} // namespace mfem

#endif // MFEM_NVVP_HPP
