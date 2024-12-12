#pragma once

#include <fmt/format.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <string>

#ifdef MFEM_USE_CALIPER
#include <caliper/cali.h>
#endif

#ifdef MFEM_USE_CUDA
#include <cudaProfiler.h>
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#else
struct nvtxEventAttributes_t
{
   int version;
   int size;
   int category;
   int colorType;
   uint32_t color;
   int payloadType;
   uint64_t payload;
   int messageType;
   struct
   {
      std::string ascii;
   } message;
};
#define NVTX_VERSION 1
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE 256
#define NVTX_COLOR_ARGB 0
#define NVTX_MESSAGE_TYPE_ASCII 0
#define nvtxRangePushEx(...)
#define nvtxRangePop(...)
#define cudaStreamSynchronize(...)
#endif

//// ///////////////////////////////////////////////////////////////////////////
template <class T>
std::enable_if_t<!std::numeric_limits<T>::is_integer, bool>
AlmostEq(T x, T y, T tolerance = 10.0*std::numeric_limits<T>::epsilon())
{
   const T neg = std::abs(x - y);
   constexpr T min = std::numeric_limits<T>::min();
   constexpr T eps = std::numeric_limits<T>::epsilon();
   const T min_abs = std::min(std::abs(x), std::abs(y));
   if (std::abs(min_abs) == 0.0) { return neg < eps; }
   return (neg / (1.0 + std::max(min, min_abs))) < tolerance;
}

namespace gpu::nvtx
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
      {
         if (*p == c) { break; }
      }
      if (!len) { return nullptr; }
      if (n == 1) { return p; }
   }
   return nullptr;
}

inline uint32_t static_color(const uint8_t COLOR, const int RANK,
                             const char *FILE)
{
   constexpr auto kMpiColorShift = 1;
   const auto rank_shift = kMpiColorShift * RANK;
   if (COLOR > 0) { return kHexColors[COLOR + rank_shift]; }
   const auto file_color = static_checksum8(FILE);
   return kHexColors[(file_color + rank_shift) % kNumHexColors];
}

///////////////////////////////////////////////////////////////////////////////
// Helpers to generate unique variable names
#define NVTX_FLF __FILE__, __LINE__, __FUNCTION__
#define NVTX_PRIVATE_NAME(prefix) NVTX_PRIVATE_CONCAT(prefix, __LINE__)
#define NVTX_PRIVATE_CONCAT(a, b) NVTX_PRIVATE_CONCAT2(a, b)
#define NVTX_PRIVATE_CONCAT2(a, b) a##b

#ifndef NVTX_COLOR
#define NVTX_COLOR ::gpu::nvtx::kBlack
#endif

///////////////////////////////////////////////////////////////////////////////
struct Debug
{
   const bool debug = false, end = true;

   inline Debug() = default;

   inline Debug(const int RANK, const char *FILE, const int LINE,
                const char *FUNC, uint8_t COLOR, bool ini = true,
                bool END = true): debug(true), end(END)
   {
      const char *base = static_strrnchr(FILE, '/', 2);
      const char *file = base ? base + 1 : FILE;
      const uint32_t rgb = static_color(COLOR, RANK, FILE);
      const uint8_t r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF,
                    b = rgb & 0xFF;
      std::cout << "\033[38;2;";
      std::cout << std::to_string(r) << ";";
      std::cout << std::to_string(g) << ";";
      std::cout << std::to_string(b) << "m";
      if (ini)
      {
         std::cout << RANK << std::setw(64) << file << ":";
         std::cout << "\033[2m" << std::setw(4) << std::left << LINE
                   << "\033[22m: ";
         if (FUNC) { std::cout << "[" << FUNC << "] "; }
      }
      std::cout << std::right << "\033[1m";
   }

   inline ~Debug()
   {
      if (debug) { std::cout << "\033[m" << (end ? "\n" : "") << std::flush; }
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
                    uint8_t COLOR, bool INI = true, bool END = true)
   {
      static int mpi_rank = 0, dbg_mpi_rank = 0;
      static bool env_mpi = false, env_dbg = false;
      if (static bool ini = false; !std::exchange(ini, true))
      {
         env_dbg = (::getenv("MFEM_DEBUG") != nullptr);
         env_mpi = ::getenv("MFEM_DEBUG_MPI") != nullptr;
         // int mpi_flag = 0;
         // MPI_Initialized(&mpi_flag);
         // if (mpi_flag) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); }
         dbg_mpi_rank = atoi(env_mpi ? ::getenv("MFEM_DEBUG_MPI") : "0");
      }
      const bool debug = (env_dbg && (!env_mpi || (dbg_mpi_rank == mpi_rank)));
      return debug ? Debug(mpi_rank, FILE, LINE, FUNC, COLOR, INI, END)
             : Debug();
   }
};

// Debug console traces, unnamed
#define NVTX_DEBUG(...)                                                  \
   ::gpu::nvtx::Debug::Set(NVTX_FLF, NVTX_COLOR).operator()(__VA_ARGS__)

#define NVTX_DEBUG_NO_INI(...)                                \
   ::gpu::nvtx::Debug::Set(NVTX_FLF, NVTX_COLOR, false, true) \
      .operator()(__VA_ARGS__)

#define NVTX_DEBUG_APPEND(...)                                 \
   ::gpu::nvtx::Debug::Set(NVTX_FLF, NVTX_COLOR, false, false) \
      .operator()(__VA_ARGS__)

#define NVTX_DEBUG_NO_END(...)                                \
   ::gpu::nvtx::Debug::Set(NVTX_FLF, NVTX_COLOR, true, false) \
      .operator()(__VA_ARGS__)

///////////////////////////////////////////////////////////////////////////////
struct Nvtx
{
   const bool nvtx = false, enforce_kernel_sync = false;
   const char *base, *file;
   const uint32_t color = kBlack;
   mutable std::string ascii;
   mutable nvtxEventAttributes_t event;
   mutable bool pushed = false;

   inline Nvtx() = default;

   Nvtx(bool enforce_kernel_sync, const char *FILE, const int LINE,
        const char *FUNC, uint8_t COLOR):
      nvtx(true), enforce_kernel_sync(enforce_kernel_sync),
      base(static_strrnchr(FILE, '/', 2)), file(base ? base + 1 : FILE),
      color(COLOR), ascii(file), event({})
   {
      event.version = NVTX_VERSION;
      event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      event.colorType = NVTX_COLOR_ARGB;
      event.color = static_color(COLOR, 0, FILE);
      event.messageType = NVTX_MESSAGE_TYPE_ASCII;

      ascii += ":";
      ascii += std::to_string(LINE);
      ascii += ":[";
      ascii += FUNC;
      ascii += "] ";

      pushed = false;
   }

   explicit Nvtx(const char *title, uint8_t color = kWheat,
                 bool enforce_kernel_sync = true):
      nvtx(true), enforce_kernel_sync(enforce_kernel_sync), color(color),
      ascii(title), event({})
   {
      event.version = NVTX_VERSION;
      event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      event.colorType = NVTX_COLOR_ARGB;
      event.color = static_color(color, 0, "");
      event.messageType = NVTX_MESSAGE_TYPE_ASCII;
      event.message.ascii = ascii.c_str();
      nvtxRangePushEx(&event);
      pushed = true;
   }

   inline void operator()() const
   {
      if (!nvtx) { return; }
      event.message.ascii = ascii.c_str();
      assert(!pushed);
      nvtxRangePushEx(&event);
      pushed = true;
   }

   template <typename T>
   inline void operator()(const T &arg) const
   {
      if (!nvtx) { return; }
      this->operator<<(arg);
      event.message.ascii = ascii.c_str();
      assert(!pushed);
      nvtxRangePushEx(&event);
      pushed = true;
   }

   template <typename... Args>
   inline void operator()(fmt::format_string<Args...> fmt,
                          Args &&...args) const
   {
      if (!nvtx) { return; }
      ascii += fmt::format(fmt, std::forward<Args>(args)...);
      event.message.ascii = ascii.c_str();
      assert(!pushed);
      nvtxRangePushEx(&event);
      pushed = true;
   }

   template <typename T>
   inline void operator<<(const T &arg) const
   {
      if (nvtx) { ascii += arg; }
   }

   inline ~Nvtx()
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
         eks.message.ascii = "!"; // enforce kernel synchronization
         eks.color = kHexColors[kYellow];
         nvtxRangePushEx(&eks);
         cudaStreamSynchronize(nullptr);
         nvtxRangePop(/*eks*/);
      }
      assert(pushed);
      nvtxRangePop(/*event*/);
   }

   using nvtx_ptr = std::unique_ptr<Nvtx>;
   using nvtx_stack_t = std::stack<nvtx_ptr>;

   static nvtx_ptr Set(const char *FILE, const int LINE, const char *FUNC,
                       uint8_t COLOR)
   {
      static bool nvtx = false, eks = false;
      if (static bool ini = false; !std::exchange(ini, true))
      {
         eks = ::getenv("MFEM_EKS") != nullptr;
         nvtx = ::getenv("MFEM_NVTX") != nullptr;
         Nvtx force_first_eks("Init EKS", kYellow, true);
      }
      return nvtx_ptr(nvtx ? new Nvtx(eks, FILE, LINE, FUNC, COLOR)
                      : new Nvtx());
   }

   static nvtx_stack_t &Stack()
   {
      auto nvtx_events = []() -> nvtx_stack_t &
      {
         static nvtx_stack_t events;
         return events;
      };
      static std::once_flag ready;
      // one touch to guarantee the object is ready
      std::call_once(ready, [&] { nvtx_events(); });
      return nvtx_events();
   }
};

// Temporary object only alive for the current statement
#define NVTX_(COLOR, ...)                                       \
   NVTX_DEBUG(__VA_ARGS__);                                     \
   std::unique_ptr<::gpu::nvtx::Nvtx> NVTX_PRIVATE_NAME(nvtx) = \
      ::gpu::nvtx::Nvtx::Set(NVTX_FLF, COLOR);                  \
   NVTX_PRIVATE_NAME(nvtx)->operator()(__VA_ARGS__)

// Temporary object only alive for the current statement
#define NVTX(...) NVTX_(NVTX_COLOR, __VA_ARGS__)

// Begin(with color)/End NVTX event traces
#define NVTX_BEGIN_(COLOR, ...)                                              \
   NVTX_DEBUG(__VA_ARGS__);                                                  \
   ::gpu::nvtx::Nvtx::Stack().push(::gpu::nvtx::Nvtx::Set(NVTX_FLF, COLOR)); \
   ::gpu::nvtx::Nvtx::Stack().top()->operator()(__VA_ARGS__)

// Begin/End NVTX event traces
#define NVTX_BEGIN(...) NVTX_BEGIN_(NVTX_COLOR, __VA_ARGS__);

#define NVTX_END(...)                        \
   ::gpu::nvtx::Nvtx::Stack().top().reset(); \
   ::gpu::nvtx::Nvtx::Stack().pop()

#ifdef USE_CALIPER
// CALIPER & NVTX marks
#define NVTX_MARK_FUNCTION                                            \
      NVTX();                                                            \
      std::unique_ptr<cali::Function> __cali_ann##__func__;              \
      __cali_ann##__func__ = std::make_unique<cali::Function>(__func__);

#define NVTX_MARK(...)                                                   \
      NVTX(__VA_ARGS__);                                                    \
      std::unique_ptr<cali::Function> __cali_ann##__func__;                 \
      __cali_ann##__func__ = std::make_unique<cali::Function>(__VA_ARGS__);

#define NVTX_MARK_FUNCTION_NAME(STR_NAME)                                \
      NVTX(STR_NAME);                                                       \
      std::unique_ptr<cali::Function> __cali_ann##__func__;                 \
      if (g_caliper) {                                                      \
         __cali_ann##__func__ = std::make_unique<cali::Function>(STR_NAME); \
      }

#define NVTX_MARK_BEGIN(...)     \
      CALI_MARK_BEGIN(__VA_ARGS__); \
      NVTX_BEGIN(__VA_ARGS__);

#define NVTX_MARK_END(...)     \
      NVTX_END(__VA_ARGS__);      \
      CALI_MARK_END(__VA_ARGS__);
#else
#define NVTX_MARK_FUNCTION NVTX()
#define NVTX_MARK(...) NVTX(__VA_ARGS__)
#define NVTX_MARK_FUNCTION_NAME(...) NVTX(__VA_ARGS__)
#define NVTX_MARK_BEGIN(...) NVTX_BEGIN(__VA_ARGS__)
#define NVTX_MARK_END(...) NVTX_END(__VA_ARGS__)
#endif

} // namespace gpu::nvtx

// Debug console traces, unnamed
#if 1
#define dbg(...) NVTX_DEBUG(__VA_ARGS__)
#define dbl(...) NVTX_DEBUG_NO_END(__VA_ARGS__)
#define dba(...) NVTX_DEBUG_APPEND(__VA_ARGS__)
#define dbc(...) NVTX_DEBUG_NO_INI(__VA_ARGS__)
#else
#define dbg(...)
#define dbl(...) (void)0
#define dba(...)
#define dbc(...)
#endif

inline bool ClearScreen()
{
   dbg("\x1B[2J\x1B[3J\x1B[H");
   return true;
}
