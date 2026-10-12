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

#include "sinc.hpp"
#include "../../general/zpng.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

using RealArray2D = mfem::Array2D<mfem::real_t>;

namespace holo
{

inline std::string OptionsError(real_t wavelengthNm, real_t zProp,
                                real_t focal, real_t stallPct, int order,
                                int maxIter, int cgIter, real_t gemMaxGib)
{
   const auto finitePositive = [](real_t v)
   {
      return std::isfinite(v) && v > 0.0;
   };
   std::ostringstream msg;
   if (!finitePositive(wavelengthNm))
   {
      msg << "Invalid -w/--wavelength " << wavelengthNm
          << ": must be finite and positive";
   }
   else if (!(wavelengthNm * 1e-9 > 0.0))
   {
      msg << "Invalid -w/--wavelength " << wavelengthNm
          << ": too small to convert to metres";
   }
   else if (!finitePositive(zProp))
   {
      msg << "Invalid -z/--z " << zProp << ": must be finite and positive";
   }
   else if (!finitePositive(focal))
   {
      msg << "Invalid -f/--focal " << focal << ": must be finite and positive";
   }
   else if (!(std::isfinite(stallPct) && stallPct >= real_t{0}))
   {
      msg << "Invalid -stall/--loss-stall " << stallPct
          << ": must be finite and >= 0 (0 disables the stall stop)";
   }
   else if (order < 0)
   {
      msg << "Invalid -p/--order " << order << ": must be >= 0";
   }
   else if (maxIter < 0)
   {
      msg << "Invalid -mi/--max-iterations " << maxIter << ": must be >= 0";
   }
   else if (cgIter < 1)
   {
      msg << "Invalid -cg/--cg-iterations " << cgIter << ": must be >= 1";
   }
   else if (!finitePositive(gemMaxGib))
   {
      msg << "Invalid -gmem/--gem-max-gib " << gemMaxGib
          << ": must be finite and positive";
   }
   return msg.str();
}

// Empty when at least one Newton step was accepted and the final loss is below
// the first loss, otherwise the -check error message
inline std::string LossCheckError(int steps, real_t firstLoss, real_t finalLoss)
{
   std::ostringstream msg;
   if (steps < 1)
   {
      msg << "Loss check failed: no Newton step was accepted";
   }
   else if (!(finalLoss < firstLoss))
   {
      msg << "Loss check failed: final loss " << finalLoss
          << " is not below the first loss " << firstLoss << " after "
          << steps << " accepted Newton steps";
   }
   return msg.str();
}

inline bool FftFits(int width, int height)
{
   const auto w = static_cast<std::size_t>(width);
   const auto h = static_cast<std::size_t>(height);
   const std::size_t embedX = ToeplitzEmbedSize(w);
   const std::size_t embedY = ToeplitzEmbedSize(h);
   return embedX > 0 && embedY > 0 && CheckedIntProduct(embedX, h) >= 0 &&
          CheckedIntProduct(embedY, w) >= 0;
}

inline std::string ImageError(int width, int height, int order, bool useFft)
{
   std::ostringstream msg;
   if (order < 0)
   {
      msg << "Invalid -p/--order " << order << ": must be >= 0";
      return msg.str();
   }
   const long long q1d = static_cast<long long>(order) + 1;
   const bool badWidth = width % q1d != 0, badHeight = height % q1d != 0;
   if (badWidth || badHeight)
   {
      msg << "Invalid -p/--order " << order << ": image";
      if (badWidth) { msg << " width " << width; }
      if (badWidth && badHeight) { msg << " and"; }
      if (badHeight) { msg << " height " << height; }
      msg << " not divisible by order+1 = " << q1d;
      return msg.str();
   }
   const auto w = static_cast<std::size_t>(width);
   const auto h = static_cast<std::size_t>(height);
   const int pixels = CheckedIntProduct(w, h);
   const bool fftFits = FftFits(width, height);
   if (pixels < 0 ||
       CheckedIntProduct(3, static_cast<std::size_t>(pixels)) < 0)
   {
      msg << "Image " << width << "x" << height
          << " is too large: 3*width*height exceeds the int size limit";
   }
   else if (!useFft)
   {
      const std::size_t n = std::max(w, h);
      if (CheckedIntProduct(n, n) < 0)
      {
         msg << "Image " << width << "x" << height << " is too large for -gem: "
             << "the " << n << "x" << n << " dense Fresnel matrix exceeds "
             << "the int size limit" << (fftFits ? ", use -fft" : "");
      }
   }
   else if (!fftFits)
   {
      msg << "Image " << width << "x" << height << " is too large for -fft: "
          << "the circulant embedding exceeds the int size limit";
   }
   return msg.str();
}

// Wavenumber, Fresnel weight and lens phase terms of Phase, fresnel_first and
// Lens for the sample grid, rejected when not finite or zero
inline std::string OpticsError(real_t wavelengthNm, real_t zProp, real_t focal,
                               real_t extentX, real_t extentY,
                               int sampleNx, int sampleNy)
{
   const auto good = [](real_t v) { return std::isfinite(v) && v > 0; };
   std::ostringstream msg;
   msg << "Invalid -w/--wavelength " << wavelengthNm;
   const real_t wavelength = wavelengthNm * 1e-9;
   const real_t k = 2.0 * M_PI / wavelength;
   if (!good(k))
   {
      msg << ": the wavenumber 2*pi/wavelength is " << k;
      return msg.str();
   }
   msg << " with -z/--z " << zProp;
   if (!good(k * zProp))
   {
      msg << ": the phase k*z is " << k * zProp;
      return msg.str();
   }
   const real_t extent[2] = {extentX, extentY};
   const int samples[2] = {sampleNx, sampleNy};
   for (int axis = 0; axis < 2; ++axis)
   {
      const real_t len = extent[axis];
      const real_t d = len / static_cast<real_t>(samples[axis]);
      const real_t s2zk = std::sqrt(2.0 * zProp / k);
      const real_t sk2z = std::sqrt(k / (2.0 * zProp));
      const real_t uHw = M_PI * s2zk / (2.0 * d);
      const real_t amp = (d / M_PI) * sk2z * std::sqrt(M_PI / 2.0);
      const real_t slidePhase = len * len * k / (2.0 * zProp);
      const real_t fresnelArg = (uHw + sk2z * len) * std::sqrt(2.0 / M_PI);
      const real_t fresnelPhase = 0.5 * M_PI * fresnelArg * fresnelArg;
      const char *bad = nullptr;
      if (!good(s2zk) || !good(sk2z)) { bad = "sqrt(2z/k)"; }
      else if (!good(uHw)) { bad = "half bandwidth"; }
      else if (!good(amp)) { bad = "amplitude"; }
      else if (!good(slidePhase)) { bad = "slide phase"; }
      else if (!good(fresnelPhase)) { bad = "Fresnel integral phase"; }
      if (bad)
      {
         msg << ": the Fresnel weight " << bad << " for " << samples[axis]
             << " samples is not finite and positive";
         return msg.str();
      }
   }
   msg.str("");
   msg << "Invalid -w/--wavelength " << wavelengthNm << " with -f/--focal "
       << focal;
   const real_t lensScale = M_PI / (wavelength * focal);
   const real_t lensPhase =
      lensScale * 0.25 * (extentX * extentX + extentY * extentY);
   if (!good(lensScale))
   {
      msg << ": the lens phase pi/(wavelength*f) is " << lensScale;
      return msg.str();
   }
   if (!good(lensPhase))
   {
      msg << ": the lens phase at the image corner is " << lensPhase;
      return msg.str();
   }
   return std::string();
}

// Bytes of the GemFresnel matrices Hx, Hy, their adjoints and its work vector
inline std::size_t GemBytes(int width, int height)
{
   const auto w = static_cast<std::uint64_t>(std::max(width, 0));
   const auto h = static_cast<std::uint64_t>(std::max(height, 0));
   const std::uint64_t maxCount =
      std::numeric_limits<std::size_t>::max() / sizeof(complex_t);
   const std::uint64_t squares = w * w + h * h, area = w * h;
   if (area > maxCount || squares > (maxCount - area) / 2)
   {
      return std::numeric_limits<std::size_t>::max();
   }
   return static_cast<std::size_t>(2 * squares + area) * sizeof(complex_t);
}

// Physical memory in bytes, 0 when unknown
inline std::size_t HostMemoryBytes()
{
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
   const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
   if (pages > 0 && page > 0)
   {
      const auto p = static_cast<std::uint64_t>(pages);
      const auto s = static_cast<std::uint64_t>(page);
      if (p <= std::numeric_limits<std::size_t>::max() / s)
      {
         return static_cast<std::size_t>(p * s);
      }
      return std::numeric_limits<std::size_t>::max();
   }
#endif
   return 0;
}

// Free device memory in bytes, 0 when not on a CUDA or HIP device
inline std::size_t DeviceFreeBytes()
{
   if (!mfem::Device::Allows(mfem::Backend::CUDA_MASK |
                             mfem::Backend::HIP_MASK))
   {
      return 0;
   }
   std::size_t free = 0, total = 0;
   mfem::Device::DeviceMem(&free, &total);
   return free;
}

// -gem memory: the GemBytes of a width x height image against gemMaxGib, the
// host memory shared by hostCopies processes and the free device memory shared
// by deviceCopies processes
inline std::string GemMemoryError(int width, int height, real_t gemMaxGib,
                                  int hostCopies = 1, int deviceCopies = 1)
{
   constexpr real_t gib = 1024.0 * 1024.0 * 1024.0;
   const std::size_t bytes = GemBytes(width, height);
   const real_t need = static_cast<real_t>(bytes) / gib;
   const real_t host = static_cast<real_t>(HostMemoryBytes()) / gib;
   const std::size_t device = DeviceFreeBytes();
   const real_t hostNeed = need * static_cast<real_t>(hostCopies);
   const auto copies = static_cast<std::size_t>(std::max(deviceCopies, 1));
   const std::size_t deviceBytes =
      bytes > std::numeric_limits<std::size_t>::max() / copies ?
      std::numeric_limits<std::size_t>::max() : bytes * copies;
   std::ostringstream msg;
   msg.precision(3);
   msg << "Image " << width << "x" << height << " needs " << need
       << " GiB of dense Fresnel matrices for -gem";
   if (need > gemMaxGib)
   {
      msg << ", more than -gmem/--gem-max-gib " << gemMaxGib;
   }
   else if (host > 0 && hostNeed > host)
   {
      if (hostCopies > 1)
      {
         msg << " per process, " << hostNeed << " GiB for the " << hostCopies
             << " processes on the node";
      }
      msg << ", more than the " << host << " GiB of host memory";
   }
   else if (device > 0 && deviceBytes > device)
   {
      if (copies > 1)
      {
         msg << " per process, " << need * static_cast<real_t>(copies)
             << " GiB for the " << copies << " processes sharing the device";
      }
      msg << ", more than the " << static_cast<real_t>(device) / gib
          << " GiB of free device memory";
   }
   else { return std::string(); }
   if (FftFits(width, height)) { msg << ", use -fft"; }
   return msg.str();
}

inline MFEM_HOST_DEVICE int RasterIndex(int ex, int ey, int ix, int iy,
                                        int nx, int q1d)
{
   const int col = ex * q1d + ix, row = ey * q1d + iy;
   return col + row * (nx * q1d);
}

inline MFEM_HOST_DEVICE int RasterIndex(int dof, int nx, int q1d)
{
   const int q2 = q1d * q1d;
   const int e = dof / q2, q = dof - e * q2;
   const int ix = q % q1d, iy = q / q1d;
   const int ex = e % nx, ey = e / nx;
   return RasterIndex(ex, ey, ix, iy, nx, q1d);
}

inline void Pack(int q1d, int n, int nx,
                 const Vector &re_s, const Vector &im_s, Vector &E)
{
   const real_t *re = re_s.Read(), *im = im_s.Read();
   real_t *e = E.Write();
   if (q1d == 1)
   {
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         e[2 * i] = re[i];
         e[2 * i + 1] = im[i];
      });
      return;
   }
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int k = RasterIndex(i, nx, q1d);
      e[2 * k] = re[i];
      e[2 * k + 1] = im[i];
   });
}

inline void Unpack(int q1d, int n, int nx, const Vector &E,
                   Vector &re_s, Vector &im_s)
{
   const real_t *e = E.Read();
   real_t *re = re_s.Write(), *im = im_s.Write();
   if (q1d == 1)
   {
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         re[i] = e[2 * i];
         im[i] = e[2 * i + 1];
      });
      return;
   }
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const int k = RasterIndex(i, nx, q1d);
      re[i] = e[2 * k];
      im[i] = e[2 * k + 1];
   });
}

inline void SetInterleavedSize(Vector &v, int nx, int ny)
{
   v.UseDevice(true);
   v.SetSize(2 * nx * ny);
}

inline int InterleavedSize(const Vector &v)
{
   return v.Size() / 2;
}

class RowSum : public mfem::InnerProductOperator
{
   const int rowLength;
   mutable Vector rowSums;

protected:
   // Row sums of all ranks in global row order
   virtual const real_t *AllRows(const Vector &local, int &rows) const
   {
      rows = local.Size();
      return local.HostRead();
   }

public:
   explicit RowSum(int length) : rowLength(length)
   {
      MFEM_VERIFY(rowLength > 0, "RowSum: row length must be positive");
   }

   real_t Sum(const Vector &a, const Vector *b = nullptr) const
   {
      MFEM_VERIFY(a.Size() % rowLength == 0, "RowSum: size " << a.Size()
                  << " is not a multiple of the row length " << rowLength);
      MFEM_VERIFY(b == nullptr || b->Size() == a.Size(), "RowSum: size");
      const int rows = a.Size() / rowLength, len = rowLength;
      rowSums.UseDevice(true);
      rowSums.SetSize(rows);
      const real_t *pa = a.Read();
      const real_t *pb = b ? b->Read() : nullptr;
      real_t *rs = rowSums.Write();
      mfem::forall(rows, [=] MFEM_HOST_DEVICE (int r)
      {
         real_t acc = 0.0;
         for (int j = r * len; j < (r + 1) * len; ++j)
         {
            acc += pb ? pa[j] * pb[j] : pa[j];
         }
         rs[r] = acc;
      });
      int allRows = 0;
      const real_t *all = AllRows(rowSums, allRows);
      real_t sum = 0.0;
      for (int r = 0; r < allRows; ++r) { sum += all[r]; }
      return sum;
   }

   real_t Eval(const Vector &x, const Vector &y) override { return Sum(x, &y); }
};

inline real_t MatchTarget(Vector &E, const Vector &target,
                          Vector &dot_v, Vector &energy_v,
                          const RowSum &rowSum)
{
   const int n = InterleavedSize(E);
   const real_t *e = E.Read();
   const real_t *t = target.Read();
   real_t *d = dot_v.Write();
   real_t *en = energy_v.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t mag = std::hypot(e[2 * i], e[2 * i + 1]);
      d[i] = t[i] * mag;
      en[i] = mag * mag;
   });
   const real_t dot = rowSum.Sum(dot_v);
   const real_t energy = rowSum.Sum(energy_v);
   const real_t scale = (energy > 1e-30) ? (dot / energy) : 0.0;
   E *= scale;
   return scale;
}

class Lens
{
   Vector forward, backward;

public:
   void operator()(int nx, int ny,
                   real_t extent_x, real_t extent_y,
                   real_t wavelength, real_t focal,
                   int firstRow = 0, int rows = -1)
   {
      if (rows < 0) { rows = ny - firstRow; }
      MFEM_VERIFY(firstRow >= 0 && firstRow + rows <= ny, "Lens rows");
      SetInterleavedSize(forward, nx, rows);
      SetInterleavedSize(backward, nx, rows);
      const real_t dx = extent_x / static_cast<real_t>(nx);
      const real_t dy = extent_y / static_cast<real_t>(ny);
      const real_t scale = -M_PI / (wavelength * focal);
      const int n = nx * rows, row0 = firstRow;
      real_t *f = forward.Write();
      real_t *b = backward.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int col = idx % nx;
         const int row = row0 + idx / nx;
         const real_t x = (static_cast<real_t>(col) + 0.5) * dx - 0.5 * extent_x;
         const real_t y = (static_cast<real_t>(row) + 0.5) * dy - 0.5 * extent_y;
         const real_t ph = scale * (x * x + y * y);
         const real_t c = std::cos(ph);
         const real_t s = std::sin(ph);
         f[2 * idx] = c;
         f[2 * idx + 1] = s;
         b[2 * idx] = c;
         b[2 * idx + 1] = -s;
      });
   }

   void Apply(Vector &E, bool backward_pass = false) const
   {
      const int n = InterleavedSize(E);
      MFEM_VERIFY(E.Size() == forward.Size(), "Lens size");
      const real_t *L = (backward_pass ? backward : forward).Read();
      real_t *e = E.ReadWrite();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t a = e[2 * i], b = e[2 * i + 1];
         const real_t c = L[2 * i], d = L[2 * i + 1];
         e[2 * i] = a * c - b * d;
         e[2 * i + 1] = a * d + b * c;
      });
   }
};

// ────────────────────────────────────────────────────────────────────────────
struct GrayImage
{
   int width = 0, height = 0;
   std::vector<float> pixels;
};

namespace detail
{

inline float Luma(float r, float g, float b)
{
   return 0.299f * r + 0.587f * g + 0.114f * b;
}

// HSV→RGB with saturation and value fixed at 1. Hue is in [0, 1).
inline MFEM_HOST_DEVICE void HsvToRgb(real_t h, real_t &r, real_t &g, real_t &b)
{
   const real_t x_val = 1.0 - std::fabs(std::fmod(h * 6.0, 2.0) - 1.0);
   switch (static_cast<int>(h * 6.0) % 6)
   {
      case 0: r = 1.0; g = x_val; b = 0.0; break;
      case 1: r = x_val; g = 1.0; b = 0.0; break;
      case 2: r = 0.0; g = 1.0; b = x_val; break;
      case 3: r = 0.0; g = x_val; b = 1.0; break;
      case 4: r = x_val; g = 0.0; b = 1.0; break;
      case 5: r = 1.0; g = 0.0; b = x_val; break;
      default: r = g = b = 0.0; break;
   }
}

// CIE 1931 2° CMFs, 380 nm + i nm, ×10^4. Second differences of X, Y, Z.
// One hex digit is a value in [-7, 7] (digit minus 7).
// Digit f then two hex digits is the int8 value.
// Sum twice from zero to recover each sample.
constexpr char kCieD2[] =
{
   "f0e7f41ff37fc478986777977a87a87977a77987777877a97c88c86f0997f0998e86c8"
   "8966777578aa6f09a7f10b8f16d7f1bc7f1bd7f19c7f19c8f18c7f17b6f19f088f20d8"
   "f22f096f28f098f2bf097f31f0c7f35f0c8f3cf0c8f3ff0d7f41f0a9f31c6f1c88e48f"
   "f207fe1ff77fdcff88fd9ff67fd7ff77fd6ff68fd3ff57fd2ff67fd3ff68fd0ff67fd2"
   "ff68fd1ff57fd2ff77fd5ff88fd7ff67fd9ff87fdc07fde08fe417fe517feb38fed28f"
   "f257ff448058367669247ff637ff339fed27fea29fe937fea38fec37feb38feb38fe51"
   "7fdd09fd7ff78fcdff78fcb19fd328fe359fef883a9f0899f09baf0898da9d99caaca8"
   "f09a8f09a9f0ba8f0dc8f0fa9f10c9f12cbf12caf14cbf14cbf12acf11bbf11acf0fbd"
   "f0f9bf0eacf0dbcf0dabf0c9df0cbef0a9f08f0caf08f0aaf0af0aaf09e9eebdabbaab"
   "7ba8cb8bb9ccaccadbad9ccabd6cd7dd3cd4dc0dcff8ccff7dbff7cbff8bc1ac1ab39a"
   "18b299198179087ff88a28709a27929818928a28828928919937818938928838927a39"
   "837a38847a388489279487279378388277278387267287477378385377276475376275"
   "48536538447547336348337247236247258046157258056257078167057ff876ff768f"
   "f567ff566ff577ff668ff767ff886ff878ff866ff788ff666ff488ff576ff468ff696f"
   "f768ff796ff888ff877087ff897ff877ff887087ff767ff887ff777ff867ff88616819"
   "74774976977a7a97ab6cb8ca7ba79a7ba7997897a97997a97997b979a7b87ba7a97c97"
   "ba7c97c97ca7d97c97ba7c87b97b97c97a97c87b97a97c87b97a87b97c87a97b87c97b"
   "97b87a87c97987a87a87987977787877877887877987877a8778797798787788787787"
   "7987877787977777777877787767887777877777787867787777867687877777777877"
   "6778777777777877678776878676878677877777777677878776777678876678776878"
   "7767787767787776768787767787777767787777777777777777777777777767787777"
   "7777777777777"
};
constexpr int kCieN = 389;

inline int CieHex(char c)
{
   return c >= 'a' ? c - 'a' + 10 : c - '0';
}

inline int CieDelta2(const char *&p)
{
   const int code = CieHex(*p++);
   if (code != 15) { return code - 7; }
   int v = (CieHex(p[0]) << 4) | CieHex(p[1]);
   p += 2;
   if (v >= 128) { v -= 256; }
   return v;
}

inline void CieXyzAt(int index, real_t &x, real_t &y, real_t &z)
{
   if (index < 0 || index >= kCieN)
   {
      x = y = z = 0;
      return;
   }
   const char *p = kCieD2;
   int X = 0, Y = 0, Z = 0;
   int dx = 0, dy = 0, dz = 0;
   for (int i = 0; i <= index; ++i)
   {
      dx += CieDelta2(p);
      dy += CieDelta2(p);
      dz += CieDelta2(p);
      X += dx;
      Y += dy;
      Z += dz;
   }
   x = static_cast<real_t>(X) * 1e-4;
   y = static_cast<real_t>(Y) * 1e-4;
   z = static_cast<real_t>(Z) * 1e-4;
}

inline MFEM_HOST_DEVICE real_t SrgbGamma(real_t value)
{
   if (value <= 0.00304) { return 12.92 * value; }
   return 1.055 * std::pow(value, 1.0 / 2.4) - 0.055;
}

// Unit-intensity linear sRGB for a monochromatic wavelength in [380, 780] nm,
// black outside. The table stops at 768 nm, past which the CMFs are below its
// 1e-4 resolution, so 769 to 780 nm use the last (deep red) sample.
inline void MonoLinearRgb(real_t wavelength_nm, real_t &r, real_t &g, real_t &b)
{
   constexpr real_t factor = (779.0 - 380.0) / 400.0 * 0.003975 * 683.002;
   constexpr real_t lo{380}, hi{780};
   real_t x = 0.0, y = 0.0, z = 0.0;
   if (wavelength_nm >= lo && wavelength_nm <= hi)
   {
      const int i = std::min(static_cast<int>(wavelength_nm - lo), kCieN - 1);
      CieXyzAt(i, x, y, z);
      x *= factor;
      y *= factor;
      z *= factor;
   }
   r = std::max(0.0,  3.2406 * x - 1.5372 * y - 0.4986 * z);
   g = std::max(0.0, -0.9689 * x + 1.8758 * y + 0.0415 * z);
   b = std::max(0.0,  0.0557 * x - 0.2040 * y + 1.0570 * z);
}

} // namespace detail

inline GrayImage LoadPngGray(const std::string &path)
{
   const mfem::png::image rgb = mfem::png::read(path);
   const auto kIntMax =
      static_cast<unsigned>(std::numeric_limits<int>::max());
   if (rgb.width > kIntMax || rgb.height > kIntMax)
   {
      throw std::runtime_error("Failed to load PNG: " + path +
                               " (dimensions exceed int)");
   }

   GrayImage img;
   img.width = static_cast<int>(rgb.width);
   img.height = static_cast<int>(rgb.height);
   img.pixels.resize(static_cast<std::size_t>(img.width) *
                     static_cast<std::size_t>(img.height));
   for (int y = 0; y < img.height; ++y)
   {
      for (int x = 0; x < img.width; ++x)
      {
         const std::size_t pix =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(img.width) +
            static_cast<std::size_t>(x);
         const std::size_t i = pix * 3u;
         img.pixels[pix] = detail::Luma(rgb.rgb[i] / 255.f,
                                        rgb.rgb[i + 1] / 255.f,
                                        rgb.rgb[i + 2] / 255.f);
      }
   }
   return img;
}

inline bool WritePngRgb(const std::string &path, int width, int height,
                        const unsigned char *rgb, std::size_t nbytes) try
{
   if (width <= 0 || height <= 0) { return false; }
   mfem::png::write(path, static_cast<unsigned>(width),
                    static_cast<unsigned>(height), rgb, nbytes);
   return true;
}
catch (const mfem::png::png_error &) { return false; }

inline bool WritePngRgb(const std::string &path, int width, int height,
                        const std::vector<unsigned char> &rgb)
{
   return WritePngRgb(path, width, height, rgb.data(), rgb.size());
}

inline bool WritePhasePng(const std::string &path, const RealArray2D &phase)
{
   const int height = phase.NumRows(), width = phase.NumCols();
   if (height <= 0 || width <= 0) { return false; }

   const int N = height * width;
   mfem::Array<unsigned char> rgb(N * 3);

   Vector phase_v(N);
   phase_v.UseDevice(true);
   {
      const real_t *src = phase(0);
      real_t *h = phase_v.HostWrite();
      for (int i = 0; i < N; ++i) { h[i] = src[i]; }
   }
   const real_t *ph = phase_v.Read();
   unsigned char *out = rgb.Write();
   constexpr real_t twoPi = 2.0 * kPi;

   mfem::forall(N, [=] MFEM_HOST_DEVICE (int idx)
   {
      real_t h = (ph[idx] + kPi) / twoPi;
      h = h - std::floor(h);
      real_t r, g, b;
      detail::HsvToRgb(h, r, g, b);
      const int o = idx * 3;
      out[o + 0] = static_cast<unsigned char>(std::round(255.0 * r));
      out[o + 1] = static_cast<unsigned char>(std::round(255.0 * g));
      out[o + 2] = static_cast<unsigned char>(std::round(255.0 * b));
   });

   const unsigned char *host_rgb = rgb.HostRead();
   return WritePngRgb(path, width, height, host_rgb,
                      static_cast<std::size_t>(N) * 3u);
}

inline RealArray2D GrayToReal2D(const GrayImage &img)
{
   const int height = img.height;
   const int width = img.width;
   const int N = height * width;
   RealArray2D out(height, width);
   if (N <= 0) { return out; }

   real_t *oh = out(0);
   for (int i = 0; i < N; ++i)
   {
      oh[i] = static_cast<real_t>(img.pixels[static_cast<std::size_t>(i)]);
   }
   return out;
}

inline real_t PercentileWhite(std::vector<real_t> values)
{
   values.erase(std::remove_if(values.begin(), values.end(),
   [](real_t v) { return !std::isfinite(v); }),
   values.end());
   if (values.empty()) { return 0.0; }
   real_t mx = 0.0;
   for (real_t v : values) { mx = std::max(mx, v); }
   const std::size_t n = values.size();
   constexpr real_t kDisplayWhitePercentile = 99.5;
   const auto idx = static_cast<std::size_t>(
                       kDisplayWhitePercentile / 100.0 *
                       static_cast<real_t>(n - 1));
   std::nth_element(values.begin(),
                    values.begin() + static_cast<std::ptrdiff_t>(idx),
                    values.end());
   const real_t ref = values[idx];
   return (ref > 0.0) ? ref : mx;
}

inline bool EncodeMonoWavelengthRgb(const RealArray2D &intensity,
                                    real_t wavelength_nm,
                                    std::vector<unsigned char> &rgb_out,
                                    int &width, int &height)
{
   const int ny = intensity.NumRows();
   const int nx = intensity.NumCols();
   width = nx, height = ny;
   if (nx <= 0 || ny <= 0) { return false; }

   real_t rl0, gl0, bl0;
   detail::MonoLinearRgb(wavelength_nm, rl0, gl0, bl0);

   const int N = ny * nx;

   Vector intensity_v;
   intensity_v.UseDevice(true);
   intensity_v.SetSize(N);
   real_t ref = 0.0;
   {
      const real_t *ih = intensity(0);
      real_t *h = intensity_v.HostWrite();
      std::vector<real_t> samples(static_cast<std::size_t>(N));
      for (int i = 0; i < N; ++i)
      {
         h[i] = ih[i];
         samples[static_cast<std::size_t>(i)] = h[i];
      }
      ref = PercentileWhite(std::move(samples));
   }
   const real_t inv_max = (ref > 0.0) ? (1.0 / ref) : 1.0;

   mfem::Array<unsigned char> rgb(N * 3);
   const real_t *I = intensity_v.Read();
   unsigned char *out = rgb.Write();
   constexpr real_t eps = 1e-8;

   mfem::forall(N, [=] MFEM_HOST_DEVICE (int idx)
   {
      const real_t s = I[idx] * inv_max;
      real_t r = detail::SrgbGamma(s * rl0);
      real_t g = detail::SrgbGamma(s * gl0);
      real_t b = detail::SrgbGamma(s * bl0);
      real_t peak = r;
      if (g > peak) { peak = g; }
      if (b > peak) { peak = b; }
      peak += eps;
      if (peak > 1.0)
      {
         const real_t scale = 1.0 / peak;
         r *= scale;
         g *= scale;
         b *= scale;
      }
      if (r < 0.0) { r = 0.0; }
      else if (r > 1.0) { r = 1.0; }
      if (g < 0.0) { g = 0.0; }
      else if (g > 1.0) { g = 1.0; }
      if (b < 0.0) { b = 0.0; }
      else if (b > 1.0) { b = 1.0; }
      const int o = idx * 3;
      out[o]     = static_cast<unsigned char>(std::round(r * 255.0));
      out[o + 1] = static_cast<unsigned char>(std::round(g * 255.0));
      out[o + 2] = static_cast<unsigned char>(std::round(b * 255.0));
   });

   const unsigned char *host = rgb.HostRead();
   rgb_out.assign(host, host + static_cast<std::size_t>(N) * 3u);
   return true;
}

} // namespace holo
