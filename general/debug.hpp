#pragma once
#include <cstring>
// https://fmt.dev/11.0/api/
#include </opt/homebrew/include/fmt/format.h>

#include <iomanip>
#include <iostream>
#include <utility>

namespace debug
{

///////////////////////////////////////////////////////////////////////////////
// https://en.wikipedia.org/wiki/Web_colors#Extended_colors
// http://www.calmar.ws/vim/256-xterm-24bit-rgb-color-chart.html
// clang-format off
enum color_names
{
   kBlack = 0, kNavyBlue, kDarkBlue, kMediumBlue, kBlue, kDarkGreen, kWebGreen, kTeal,
   kDarkCyan, kDeepSkyBlue, kDarkTurquoise, kMediumSpringGreen, kGreen, kLime,
   kSpringGreen, kAqua, kCyan, kMidnightBlue, kDodgerBlue, kLightSeaGreen, kForestGreen,
   kSeaGreen, kDarkSlateGray, kLimeGreen, kMediumSeaGreen, kTurquoise, kRoyalBlue,
   kSteelBlue, kDarkSlateBlue, kMediumTurquoise, kIndigo, kDarkOliveGreen, kCadetBlue,
   kCornflower, kRebeccaPurple, kMediumAquamarine, kDimGray, kSlateBlue, kOliveDrab,
   kSlateGray, kLightSlateGray, kMediumSlateBlue, kLawnGreen, kWebMaroon, kWebPurple,
   kChartreuse, kAquamarine, kOlive, kWebGray, kSkyBlue, kLightSkyBlue, kBlueViolet,
   kDarkRed, kDarkMagenta, kSaddleBrown, kDarkSeaGreen, kLightGreen, kMediumPurple,
   kDarkViolet, kPaleGreen, kDarkOrchid, kYellowGreen, kPurple, kSienna, kBrown,
   kDarkGray, kLightBlue, kGreenYellow, kPaleTurquoise, kMaroon, kLightSteelBlue,
   kPowderBlue, kFirebrick, kDarkGoldenrod, kMediumOrchid, kRosyBrown, kDarkKhaki,
   kGray, kSilver, kMediumVioletRed, kIndianRed, kPeru, kChocolate, kTan, kLightGray,
   kThistle, kOrchid, kGoldenrod, kPaleVioletRed, kCrimson, kGainsboro, kPlum, kBurlywood,
   kLightCyan, kLavender, kDarkSalmon, kViolet, kPaleGoldenrod, kLightCoral, kKhaki,
   kAliceBlue, kHoneydew, kAzure, kSandyBrown, kWheat, kBeige, kWhiteSmoke, kMintCream,
   kGhostWhite, kSalmon, kAntiqueWhite, kLinen, kLightGoldenrod, kOldLace, kRed,
   kFuchsia, kMagenta, kDeepPink, kOrangeRed, kTomato, kHotPink, kCoral, kDarkOrange,
   kLightSalmon, kOrange, kLightPink, kPink, kGold, kPeachPuff, kNavajoWhite, kMoccasin,
   kBisque, kMistyRose, kBlanchedAlmond, kPapayaWhip, kLavenderBlush, kSeashell,
   kCornsilk, kLemonChiffon, kFloralWhite, kSnow, kYellow, kLightYellow, kIvory, kWhite,
   kNvidia
};
// clang-format on

static constexpr int kNumHexColors = 146;
static constexpr std::array<uint32_t, kNumHexColors> kHexColors =
{
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
      0xFFFAFA, 0xFFFF00, 0xFFFFE0, 0xFFFFF0, 0xFFFFFF, 0x76B900
   }
};

///////////////////////////////////////////////////////////////////////////////
constexpr size_t static_strlen(const char *str)
{
   return *str == '\0' ? 0 : static_strlen(str + 1) + 1;
}

constexpr uint8_t static_checksum8(const char *bfr)
{
   unsigned int chk = 0;
   size_t len = static_strlen(bfr);
   for (; len; len--, bfr++) { chk += static_cast<unsigned int>(*bfr); }
   return static_cast<uint8_t>(chk);
}

constexpr char *static_strrnchr(const char *str, const char c, int n)
{
   size_t len = static_strlen(str);
   char *p = const_cast<char *>(str) + len - 1;
   for (; n; n--, p--, len--)
   {
      for (; len; p--, len--)
         if (*p == c) { break; }
      if (!len) { return nullptr; }
      if (n == 1) { return p; }
   }
   return nullptr;
}

inline uint32_t static_color(const uint8_t COLOR) { return kHexColors[COLOR]; }

///////////////////////////////////////////////////////////////////////////////
struct Debug
{
   const bool debug = false;

   inline Debug() = default;

   inline Debug(const char *FILE, const int LINE, const char *FUNC,
                uint8_t COLOR)
      : debug(true)
   {
      const char *base = static_strrnchr(FILE, '/', 2);
      const char *file = base ? base + 1 : FILE;
      const uint32_t rgb = static_color(COLOR);
      const uint8_t r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
      std::cout << "\033[38;2;";
      std::cout << std::to_string(r) << ";";
      std::cout << std::to_string(g) << ";";
      std::cout << std::to_string(b) << "m";
      std::cout << 0 << std::setw(64) << file << ":";
      std::cout << "\033[2m" << std::setw(4) << std::left << LINE << "\033[22m: ";
      if (FUNC) { std::cout << "[" << FUNC << "] "; }
      std::cout << std::right << "\033[1m";
   }

   inline ~Debug()
   {
      if (debug) { std::cout << "\033[m\n" << std::flush; }
   }

   template <typename T>
   inline void operator<<(const T &arg) const noexcept
   {
      if (debug) { std::cout << arg; }
   }

   template <typename T>
   inline void operator()(const T &arg) const noexcept
   {
      if (debug) { this->operator<<(arg); }
   }

   template <typename... Args>
   inline void operator()(const char *fmt, Args &&...args) const noexcept
   {
      if (debug) { std::cout << fmt::format(fmt, std::forward<Args>(args)...); }
   }

   inline void operator()() const noexcept {}

   static Debug Set(const char *FILE, const int LINE, const char *FUNC,
                    uint8_t COLOR)
   {
      static bool env_dbg = false;
      if (static bool ini = false; !std::exchange(ini, true))
      {
         env_dbg = (getenv("MFEM_DEBUG") != nullptr);
      }
      const bool debug = env_dbg;
      return debug ? Debug(FILE, LINE, FUNC, COLOR) : Debug();
   }
};

// Helpers to generate unique variable names
#define DBG_FLF __FILE__, __LINE__, __FUNCTION__
#define DBG_PRIVATE_NAME(prefix) DBG_PRIVATE_CONCAT(prefix, __LINE__)
#define DBG_PRIVATE_CONCAT(a, b) DBG_PRIVATE_CONCAT2(a, b)
#define DBG_PRIVATE_CONCAT2(a, b) a##b

#ifndef DBG_COLOR
#define DBG_COLOR ::debug::kBlack
#endif

// Debug console traces, unnamed
#define dbg(...) ::debug::Debug::Set(DBG_FLF, DBG_COLOR).operator()(__VA_ARGS__)

}  // namespace debug
