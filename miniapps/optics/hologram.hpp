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
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace holo
{

struct HologramConfig
{
   const char *in_file = MFEM_SOURCE_DIR "/doc/logo-small.png";
   const char *holo_file = "hologram.png";
   const char *mask_file = "mask.png";
   const char *device_config = "cpu";
   real_t wavelength_nm = 532.0;
   real_t z_prop = 0.80;
   real_t focal = 0.80;
   int nwt_max_iter = 50;
   int cgs_max_iter = 20;
   int order = 0;
   real_t stall_pct = 1.0;
   real_t gemMaxGib = 16.0;
   bool checkLoss = false;
#if defined(HOLO_USE_CPU_FFT)
   bool use_fft = true;
#else
   bool use_fft = false;
#endif
   bool chose_propagator = false;

   void AddTo(mfem::OptionsParser &args);
   void NotePropagator(int argc, char *argv[]);
};

inline void HologramConfig::AddTo(mfem::OptionsParser &args)
{
   args.AddOption(&in_file, "-i", "--input", "Target PNG.");
   args.AddOption(&holo_file, "-o", "--output", "Hologram sRGB PNG.");
   args.AddOption(&mask_file, "-m", "--mask", "Phase-mask PNG.");
   args.AddOption(&wavelength_nm, "-w", "--wavelength",
                  "Wavelength in nanometres (default 532). Any positive "
                  "value is accepted. The hologram PNG is black outside "
                  "380 to 780 nm.");
   args.AddOption(&nwt_max_iter, "-mi", "--max-iterations",
                  "Maximum Newton iterations (default 50).");
   args.AddOption(&cgs_max_iter, "-cg", "--cg-iterations",
                  "Maximum CG iterations per Newton step (default 20).");
   args.AddOption(&order, "-p", "--order",
                  "Pixel block degree. One block covers (order+1)^2 pixels "
                  "(default 0).");
   args.AddOption(&z_prop, "-z", "--z",
                  "Propagation distance in metres (default 0.80).");
   args.AddOption(&focal, "-f", "--focal",
                  "Lens focal length in metres (default 0.80).");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_fft, "-fft", "--fft", "-gem", "--gem",
                  "Toeplitz FFT or the dense product (-gem). The default is "
                  "-fft when the library for the device is linked (FFTW on "
                  "cpu, cuFFT or hipFFT on CUDA or HIP).");
   args.AddOption(&stall_pct, "-stall", "--loss-stall",
                  "Newton stall relative loss drop in percent, 0 disables "
                  "the stall stop (default 1).");
   args.AddOption(&gemMaxGib, "-gmem", "--gem-max-gib",
                  "Largest memory in GiB for the dense -gem Fresnel matrices "
                  "and the work image (default 16).");
   args.AddOption(&checkLoss, "-check", "--check", "-no-check", "--no-check",
                  "Exit with an error unless a Newton step was accepted and "
                  "the final loss is below the first loss (default off).");
}

inline void HologramConfig::NotePropagator(int argc, char *argv[])
{
   for (int i = 1; i < argc; ++i)
   {
      const char *arg = argv[i];
      if (std::strcmp(arg, "-fft") == 0 || std::strcmp(arg, "--fft") == 0 ||
          std::strcmp(arg, "-gem") == 0 || std::strcmp(arg, "--gem") == 0)
      {
         chose_propagator = true;
      }
   }
}

inline void MaskExtent(int width, int height,
                       real_t &extent_x, real_t &extent_y)
{
   extent_x = 19.0e-3;
   extent_y = extent_x * (static_cast<real_t>(height) /
                          static_cast<real_t>(width));
}

inline int VerifyPixelOrder(int pixels_x, int pixels_y, int order)
{
   MFEM_VERIFY(pixels_x >= 1 && pixels_y >= 1,
               "Pixels: image size must be positive");
   MFEM_VERIFY(order >= 0, "Pixels: order must be >= 0");
   const int q1d = order + 1;
   MFEM_VERIFY(pixels_x % q1d == 0 && pixels_y % q1d == 0,
               "Pixels: pixel count must be divisible by order+1");
   return order;
}

inline mfem::Mesh CenteredQuadMesh(int nx, int ny,
                                   real_t extent_x, real_t extent_y)
{
   mfem::Mesh serial =
      mfem::Mesh::MakeCartesian2D(
         nx, ny, mfem::Element::QUADRILATERAL,
         false, extent_x, extent_y, false);
   serial.Transform([&](const Vector &x, Vector &p)
   {
      p(0) = x(0) - 0.5 * extent_x;
      p(1) = x(1) - 0.5 * extent_y;
   });
   return serial;
}

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
   return embedX > 0 && embedY > 0 &&
          CheckedIntProduct(embedX, h) >= 0 &&
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
       << " GiB for the dense Fresnel matrices and the work image";
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

// ----------------------------------------------------------------------------
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
         const real_t x = (static_cast<real_t>(col) + 0.5_r) * dx
                          - 0.5_r * extent_x;
         const real_t y = (static_cast<real_t>(row) + 0.5_r) * dy
                          - 0.5_r * extent_y;
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

template <class Prop>
inline void ReplayIntensity(const Prop &prop, const RealArray2D &phase,
                            int firstRow, int rows,
                            real_t extent_x, real_t extent_y,
                            real_t wavelength, real_t focal,
                            Vector &intensity)
{
   const int nx = phase.NumCols(), ny = phase.NumRows(), n = nx * rows;
   Vector phase_v(n), field(2 * n);
   phase_v.UseDevice(true);
   field.UseDevice(true);
   intensity.UseDevice(true);
   {
      const real_t *src = phase.GetRow(firstRow);
      real_t *h = phase_v.HostWrite();
      for (int k = 0; k < n; ++k) { h[k] = src[k]; }
   }
   const real_t *ph = phase_v.Read();
   real_t *p = field.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int idx)
   {
      p[2 * idx] = std::cos(ph[idx]);
      p[2 * idx + 1] = std::sin(ph[idx]);
   });
   Lens lens;
   lens(nx, ny, extent_x, extent_y, wavelength, focal, firstRow, rows);
   lens.Apply(field);
   prop.Propagate(field);
   const auto *f = field.Read();
   auto *Id = intensity.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int k)
   {
      const real_t re = f[2 * k], im = f[2 * k + 1];
      Id[k] = re * re + im * im;
   });
}

// ----------------------------------------------------------------------------
template <class Mesh, class CGF>
struct FieldsImpl
{
   CGF u, du, s, g;
   Vector dot_v, energy_v, loss_q, phase_v, glob_re, glob_im;
   Vector E, E_trial, gE, dE, seed;

   explicit FieldsImpl(const Mesh &grid)
      : u(&grid.Space()),
        du(&grid.Space()),
        s(&grid.Space()),
        g(&grid.Space())
   {
      const int nloc = grid.LocalNumSamples();
      const int n = grid.NumSamples();
      auto set_size = [](Vector &v, int m)
      {
         v.SetSize(m);
         v.UseDevice(true);
      };
      set_size(dot_v, nloc);
      set_size(energy_v, nloc);
      set_size(loss_q, nloc);
      // phase_v holds the full mask image on every rank
      set_size(phase_v, n);
      const int scratch = grid.PhaseScratch();
      if (scratch > 0)
      {
         set_size(glob_re, scratch);
         set_size(glob_im, scratch);
      }

      const int nx = grid.SampleNx();
      const int ny = grid.LocalSampleRows();
      SetInterleavedSize(E, nx, ny);
      SetInterleavedSize(E_trial, nx, ny);
      SetInterleavedSize(gE, nx, ny);
      SetInterleavedSize(dE, nx, ny);
      SetInterleavedSize(seed, nx, ny);

      for (Vector *v : {&dot_v, &energy_v, &loss_q, &phase_v})
      {
         v->Write();
      }
      if (scratch > 0)
      {
         glob_re.Write();
         glob_im.Write();
      }
      for (Vector *v : {&E, &E_trial, &gE, &dE, &seed})
      {
         v->Write();
      }
   }
};

template <class Mesh, class CGF, class FieldSet>
struct RasterFieldImpl
{
   void operator()(const Mesh &grid,
                   const Vector &re_s, const Vector &im_s,
                   RealArray2D &image, FieldSet &data) const
   {
      const int n = grid.NumSamples();
      const Vector *re_v = &re_s;
      const Vector *im_v = &im_s;
      if (grid.NeedsPhaseGather())
      {
         grid.GatherDofs(re_s, data.glob_re);
         grid.GatherDofs(im_s, data.glob_im);
         re_v = &data.glob_re;
         im_v = &data.glob_im;
      }
      const int nx = grid.Nx();
      const int q1d = grid.Q1D();
      const real_t *er = re_v->Read();
      const real_t *ei = im_v->Read();
      real_t *p = data.phase_v.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const int k = RasterIndex(i, nx, q1d);
         p[k] = std::atan2(ei[i], er[i]);
      });
      image.SetSize(grid.SampleNy(), grid.SampleNx());
      const real_t *ph = data.phase_v.HostRead();
      real_t *dst = image(0);
      for (int i = 0; i < n; ++i) { dst[i] = ph[i]; }
   }

   static void FromSamples(const Mesh &grid, const Vector &re_s,
                           const Vector &im_s, Vector &E)
   {
      SetInterleavedSize(E, grid.SampleNx(), grid.LocalSampleRows());
      Pack(grid.Q1D(), grid.LocalNumSamples(), grid.Nx(), re_s, im_s, E);
   }

   static void ToSamples(const Mesh &grid, const Vector &E,
                         Vector &re_s, Vector &im_s)
   {
      Unpack(grid.Q1D(), grid.LocalNumSamples(), grid.Nx(), E, re_s, im_s);
   }

   static void ToSamples(const Mesh &grid, const Vector &E, CGF &field)
   {
      ToSamples(grid, E, field.real(), field.imag());
      field.SyncAlias();
   }
};

// ----------------------------------------------------------------------------
template <class Mesh, class GF, class CGF, class FieldSet>
class PhaseMaskImpl
{
   Mesh &grid;
   const GF *src, *tgt;
   CGF lin;
   static constexpr real_t kLossEps = 1e-24;

public:
   explicit PhaseMaskImpl(Mesh &grid)
      : grid(grid), src(nullptr), tgt(nullptr), lin(&grid.Space()) {}

   void SetAmplitudes(const GF &a_src_in, const GF &a_tgt_in)
   {
      src = &a_src_in;
      tgt = &a_tgt_in;
   }

   // Fold the phase into [-π, π)
   void Wrap(Vector &phi) const
   {
      const int n = phi.Size();
      real_t *p = phi.ReadWrite();
      constexpr real_t two_pi = 2.0 * M_PI;
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         p[i] = p[i] - two_pi * std::floor((p[i] + M_PI) / two_pi);
      });
   }

   // Compute A exp(i φ)
   void Modulate(const Vector &phi, CGF &field) const
   {
      const int n = grid.LocalNumSamples();
      const int q2 = grid.Q1D() * grid.Q1D();
      const real_t *p = phi.Read(), *a = src->Read();
      real_t *re = field.real().Write();
      real_t *im = field.imag().Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t ai = a[i / q2];
         re[i] = ai * std::cos(p[i]);
         im[i] = ai * std::sin(p[i]);
      });
      field.SyncAlias();
   }

   void Linearize(const Vector &phi) { Modulate(phi, lin); }

   void Derivative(const Vector &dphi, CGF &dfield) const
   {
      const int n = grid.LocalNumSamples();
      const real_t *er = lin.real().Read();
      const real_t *ei = lin.imag().Read();
      const real_t *dp = dphi.Read();
      real_t *dr = dfield.real().Write();
      real_t *di = dfield.imag().Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         dr[i] = -ei[i] * dp[i];
         di[i] = er[i] * dp[i];
      });
      dfield.SyncAlias();
   }

   // Map a field variation back to the phase
   void Adjoint(const CGF &dfield, Vector &g_phi) const
   {
      const int n = grid.LocalNumSamples();
      const real_t *er = lin.real().Read();
      const real_t *ei = lin.imag().Read();
      const real_t *gr = dfield.real().Read();
      const real_t *gi = dfield.imag().Read();
      real_t *ww = g_phi.Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         ww[i] = er[i] * gi[i] - ei[i] * gr[i];
      });
   }

   // Mean of (|U| - A_target)² and its gradient with respect to the field
   real_t MeanAmplitudeError(const CGF &field, CGF &grad,
                             FieldSet &data) const
   {
      const int n = field.real().Size();
      const real_t inv_n = 1.0 / static_cast<real_t>(grid.NumSamples());
      const real_t *er = field.real().Read();
      const real_t *ei = field.imag().Read();
      const real_t *at = tgt->Read();
      real_t *lq = data.loss_q.Write();
      real_t *gr = grad.real().Write();
      real_t *gi = grad.imag().Write();
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t mag =
            std::sqrt(er[i] * er[i] + ei[i] * ei[i] + kLossEps);
         const real_t d = mag - at[i];
         lq[i] = d * d;
         const real_t c = (2_r * d / mag) * inv_n;
         gr[i] = c * er[i];
         gi[i] = c * ei[i];
      });
      grad.SyncAlias();
      real_t sum = grid.Reducer().Sum(data.loss_q);
      return sum * inv_n;
   }
};

// ----------------------------------------------------------------------------
template <class Mesh, class Mask, class FieldSet, class Raster>
class GaussNewtonOpImpl : public mfem::Operator
{
   Mask &mask;
   const Mesh &grid;
   const mfem::Operator &prop;
   const Lens &lens;
   const Vector &E_scaled;
   FieldSet &data;
   const Vector *phi = nullptr;
   static constexpr real_t kAbsEps = 1e-14;
   static constexpr real_t kLmLambda0 = 1e-2;
   real_t scale = 0.0, lambda = kLmLambda0;

public:
   GaussNewtonOpImpl(Mask &mask, const Mesh &grid,
                     const mfem::Operator &prop, const Lens &lens,
                     const Vector &E_scaled, FieldSet &data)
      : mfem::Operator(grid.Space().GetVSize()),
        mask(mask), grid(grid), prop(prop),
        lens(lens), E_scaled(E_scaled), data(data) {}

   void SetState(const Vector &phi_in, real_t scale_in)
   {
      phi = &phi_in;
      scale = scale_in;
      mask.Linearize(*phi);
   }

   void SetLambda(real_t lam = kLmLambda0) { lambda = lam; }

   real_t Lambda() const { return lambda; }

   void Mult(const Vector &dphi, Vector &out) const override
   {
      mask.Derivative(dphi, data.du);
      Assemble(out);
      if (lambda != 0.0)
      {
         const real_t inv_n =
            1.0 / static_cast<real_t>(grid.NumSamples());
         out.Add(lambda * inv_n, dphi);
      }
   }

   void Assemble(Vector &out) const
   {
      Raster::FromSamples(grid, data.du.real(), data.du.imag(),
                          data.dE);
      lens.Apply(data.dE, false);
      prop.Mult(data.dE, data.dE);

      const int n = InterleavedSize(data.dE);
      const real_t *Es = E_scaled.Read();
      const real_t *dE = data.dE.Read();
      real_t *seed = data.seed.Write();
      const real_t scale2 = scale * scale;
      mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
      {
         const real_t a = Es[2 * i], b = Es[2 * i + 1];
         const real_t dr = dE[2 * i], di = dE[2 * i + 1];
         const real_t mag = std::hypot(a, b);
         if (mag > kAbsEps)
         {
            const real_t d_abs = (a * dr + b * di) / mag;
            const real_t c = (scale2 * d_abs) / mag;
            seed[2 * i] = c * a;
            seed[2 * i + 1] = c * b;
         }
         else
         {
            seed[2 * i] = 0.0;
            seed[2 * i + 1] = 0.0;
         }
      });
      prop.MultTranspose(data.seed, data.seed);
      lens.Apply(data.seed, true);
      Raster::ToSamples(grid, data.seed, data.s);
      mask.Adjoint(data.s, out);
      out *= 2.0 / static_cast<real_t>(grid.NumSamples());
   }
};

// ----------------------------------------------------------------------------
template <class Mesh, class Mask, class FieldSet, class Raster>
class ObjectiveOpImpl : public mfem::Operator
{
   Mask &mask;
   Mesh &grid;
   const mfem::Operator &prop;
   Lens &lens;
   Vector &E;
   Vector &target;
   GaussNewtonOpImpl<Mesh, Mask, FieldSet, Raster> &op;
   FieldSet &data;

public:
   ObjectiveOpImpl(Mask &mask, Mesh &grid,
                   const mfem::Operator &prop, Lens &lens,
                   Vector &E, Vector &target,
                   GaussNewtonOpImpl<Mesh, Mask, FieldSet, Raster> &op,
                   FieldSet &data)
      : mfem::Operator(grid.Space().GetVSize()),
        mask(mask), grid(grid), prop(prop),
        lens(lens), E(E), target(target), op(op),
        data(data) {}

   real_t ForwardLoss(const Vector &phi, Vector &E_out) const
   {
      mask.Modulate(phi, data.u);
      Raster::FromSamples(grid, data.u.real(), data.u.imag(), E_out);
      lens.Apply(E_out, false);
      prop.Mult(E_out, E_out);
      last_scale = holo::MatchTarget(E_out, target, data.dot_v, data.energy_v,
                                     grid.Reducer());
      Raster::ToSamples(grid, E_out, data.s);
      last_J = mask.MeanAmplitudeError(data.s, data.g, data);
      return last_J;
   }

   void Mult(const Vector &phi, Vector &F) const override
   {
      F.SetSize(Width());
      ForwardLoss(phi, E);
      data.g.real() *= last_scale;
      data.g.imag() *= last_scale;
      data.g.SyncAlias();
      Raster::FromSamples(grid, data.g.real(), data.g.imag(), data.gE);
      prop.MultTranspose(data.gE, data.gE);
      lens.Apply(data.gE, true);
      Raster::ToSamples(grid, data.gE, data.s);
      op.SetState(phi, last_scale);
      mask.Adjoint(data.s, F);
   }

   mfem::Operator &GetGradient(const Vector &phi) const override
   {
      op.SetState(phi, last_scale);
      return op;
   }

   real_t LastLoss() const { return last_J; }

   mutable real_t last_scale = 0.0;
   mutable real_t last_J = 0.0;
};

// ----------------------------------------------------------------------------
template <class Mesh, class Mask, class FieldSet, class Raster>
class GaussNewtonCgImpl : public mfem::Solver
{
   mfem::CGSolver &cg;
   GaussNewtonOpImpl<Mesh, Mask, FieldSet, Raster> &op;

public:
   GaussNewtonCgImpl(mfem::CGSolver &cg,
                     GaussNewtonOpImpl<Mesh, Mask, FieldSet, Raster> &op)
      : mfem::Solver(op.Height()), cg(cg), op(op) {}

   void SetOperator(const mfem::Operator &op_in) override
   {
      MFEM_VERIFY(&op_in == &op, "phase CG expects the Gauss-Newton operator");
      cg.SetOperator(op_in);
   }

   void Mult(const Vector &r, Vector &c) const override
   {
      cg.Mult(r, c);
   }

   int Iterations() const { return cg.GetNumIterations(); }
};

// ----------------------------------------------------------------------------
template <class Mesh, class Mask, class FieldSet, class Raster>
class PhaseNewtonImpl : public mfem::NewtonSolver
{
   mutable real_t J_log_prev = -1.0;

public:
   PhaseNewtonImpl() : mfem::NewtonSolver() {}

#ifdef MFEM_USE_MPI
   explicit PhaseNewtonImpl(MPI_Comm comm) : mfem::NewtonSolver(comm) {}
#endif

   ObjectiveOpImpl<Mesh, Mask, FieldSet, Raster> *op = nullptr;
   Mask *phase = nullptr;
   GaussNewtonOpImpl<Mesh, Mask, FieldSet, Raster> *mask_op = nullptr;
   GaussNewtonCgImpl<Mesh, Mask, FieldSet, Raster> *step = nullptr;
   Vector *trial_field = nullptr;
   Vector *trial = nullptr;
   real_t alpha0 = 1.0;
   int max_outer = 0;
   mutable int gn_ran = 0;
   mutable bool started = false;
   mutable bool line_search_failed = false;
   bool on_root = true;
   // Loss at the first Newton step and at the current iterate
   mutable real_t firstLoss = -1.0, finalLoss = -1.0;

   real_t ComputeScalingFactor(const Vector &x, const Vector &) const override
   {
      const real_t J0 = op->LastLoss();
      char drop[16] = "    n/a", loss[16];
      if (J_log_prev > 0.0)
      {
         const real_t rel = 100.0 * (J_log_prev - J0) / J_log_prev;
         std::snprintf(drop, sizeof(drop), "%6.2f%%", rel);
      }
      std::snprintf(loss, sizeof(loss), "%6.2e", J0);
      if (on_root)
      {
         mfem::out << "[hologram] Newton "
                   << (gn_ran + 1) << "/" << max_outer
                   << " (" << step->Iterations() << ")"
                   << " " << loss << " " << drop << "\n";
      }
      J_log_prev = J0;
      if (firstLoss < 0.0) { firstLoss = J0; }
      finalLoss = J0;
      const real_t scale_at_x = op->last_scale;

      auto try_decrease = [&]() -> real_t
      {
         real_t alpha = alpha0;
         constexpr int kMaxBacktracks = 8;
         for (int bt = 0; bt < kMaxBacktracks; ++bt)
         {
            *trial = x;
            trial->Add(-alpha, c);
            phase->Wrap(*trial);
            const real_t J_new = op->ForwardLoss(*trial, *trial_field);
            if (J_new < J0)
            {
               finalLoss = J_new;
               mask_op->SetLambda(std::max(mask_op->Lambda() * 0.5, 1e-8));
               return alpha;
            }
            alpha *= 0.5;
         }
         return 0.0;
      };

      real_t a = try_decrease();
      if (a > 0.0) { return a; }
      mask_op->SetLambda(std::min(mask_op->Lambda() * 10.0, 1e4));
      mask_op->SetState(x, scale_at_x);
      step->SetOperator(*mask_op);
      step->Mult(r, c);
      a = try_decrease();
      if (a <= 0.0) { line_search_failed = true; }
      return a;
   }

   void ProcessNewState(const Vector &x) const override
   {
      phase->Wrap(const_cast<Vector &>(x));
      if (!started) { started = true; return; }
      ++gn_ran;
   }
};

// ----------------------------------------------------------------------------
template <class Mesh, class Mask, class FieldSet, class Raster>
class StallControllerImpl : public mfem::IterativeSolverController
{
   real_t J_prev = -1.0;
   int stall = 0;

public:
   ObjectiveOpImpl<Mesh, Mask, FieldSet, Raster> *op = nullptr;
   real_t rel_tol = 0.0;
   mutable const char *stop_reason = nullptr;

   void Reset() override
   {
      IterativeSolverController::Reset();
      J_prev = -1.0;
      stall = 0;
      stop_reason = nullptr;
   }

   void MonitorResidual(int it, real_t, const Vector &, bool final) override
   {
      if (final || op == nullptr || rel_tol <= 0.0) { return; }
      const real_t J = op->LastLoss();
      if (J_prev < 0.0) { J_prev = J; return; }
      const real_t rel =
         (J_prev - J) / std::max(J_prev, static_cast<real_t>(1e-30));
      stall = (rel < rel_tol) ? stall + 1 : 0;
      J_prev = J;
      constexpr int kMinNewtonIters = 4, kLossStallNeed = 3;
      if (it >= kMinNewtonIters && stall >= kLossStallNeed)
      {
         converged = true;
         stop_reason = "loss_stall";
      }
   }
};

// ----------------------------------------------------------------------------
template <class Mesh, class GF, class CGF, class Prop>
class PhaseRetrievalImpl
{
   using FieldSet = FieldsImpl<Mesh, CGF>;
   using Raster   = RasterFieldImpl<Mesh, CGF, FieldSet>;
   using Mask     = PhaseMaskImpl<Mesh, GF, CGF, FieldSet>;
   using GnOp     = GaussNewtonOpImpl<Mesh, Mask, FieldSet, Raster>;
   using ObjOp    = ObjectiveOpImpl<Mesh, Mask, FieldSet, Raster>;
   using GnCG     = GaussNewtonCgImpl<Mesh, Mask, FieldSet, Raster>;

public:
   using Newton = PhaseNewtonImpl<Mesh, Mask, FieldSet, Raster>;

   PhaseRetrievalImpl(RealArray2D target, int order, real_t wavelength,
                      real_t extent_x, real_t extent_y,
                      real_t z, real_t focal, bool fft = false)
      : mesh(target.NumCols(), target.NumRows(), order, extent_x, extent_y),
        target(std::move(target)),
        wavelength(wavelength), extent_x(extent_x), extent_y(extent_y),
        z(z), focal(focal), fft(fft)
   {
      mask_image.SetSize(mesh.SampleNy(), mesh.SampleNx());
      mask_image = 0.0;
   }

   template <class MakeProp, class MakeCG, class MakeNewton>
   void Solve(int max_iter, real_t stall_rel, real_t learning_rate,
              int cg_max_iter, bool root,
              MakeProp make_prop, MakeCG make_cg, MakeNewton make_newton)
   {
      GF phi(&mesh.Space());
      phi = 0.0;
      GF a_src(&mesh.ElementSpace());
      a_src = 1.0;
      const int localSamples = mesh.LocalSampleRows() * mesh.SampleNx();
      Vector target_image(localSamples);
      target_image.UseDevice(true);
      {
         const real_t *s = target(mesh.FirstSampleRow());
         real_t *d = target_image.HostWrite();
         for (int i = 0; i < localSamples; ++i) { d[i] = s[i]; }
      }
      GF a_tgt(&mesh.Space());
      {
         const int n = mesh.LocalNumSamples();
         const int nx = mesh.Nx();
         const int q1d = mesh.Q1D();
         const int sample_nx = mesh.SampleNx();
         const int row0 = mesh.FirstSampleRow();
         const int g0 = mesh.GlobalDof0();
         const real_t *s = target_image.Read();
         real_t *d = a_tgt.Write();
         mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
         {
            const int raster = RasterIndex(g0 + i, nx, q1d);
            d[i] = s[raster - row0 * sample_nx];
         });
      }
      Mask mask(mesh);
      mask.SetAmplitudes(a_src, a_tgt);

      propagator = make_prop();

      Lens lens;
      lens(mesh.SampleNx(), mesh.SampleNy(),
           extent_x, extent_y, wavelength, focal,
           mesh.FirstSampleRow(), mesh.LocalSampleRows());

      FieldSet fields(mesh);

      GnOp mask_op(mask, mesh, *propagator, lens, fields.E, fields);
      ObjOp op(mask, mesh, *propagator, lens, fields.E,
               target_image, mask_op, fields);
      op.ForwardLoss(phi, fields.E);

      auto cg = make_cg();
      cg.SetInnerProduct(&mesh.Reducer());
      cg.SetRelTol(1e-3);
      cg.SetAbsTol(1e-12);
      cg.SetMaxIter(std::max(1, cg_max_iter));
      cg.SetPrintLevel(-1);

      auto newton = make_newton();
      newton.on_root = root;
      newton.iterative_mode = true;
      newton.SetRelTol(0.0);
      newton.SetAbsTol(1e-8);
      newton.SetMaxIter(std::max(0, max_iter));
      newton.SetPrintLevel(-1);
      newton.SetInnerProduct(&mesh.Reducer());
      GnCG step(cg, mask_op);
      newton.SetOperator(op);
      newton.SetSolver(step);
      newton.op = &op;
      newton.phase = &mask;
      newton.mask_op = &mask_op;
      newton.step = &step;
      newton.trial_field = &fields.E_trial;
      GF phi_trial(&mesh.Space());
      newton.trial = &phi_trial;
      newton.alpha0 = (learning_rate > 0.0) ? learning_rate : 1.0;
      newton.max_outer = std::max(0, max_iter);
      StallControllerImpl<Mesh, Mask, FieldSet, Raster> stall;
      stall.op = &op;
      stall.rel_tol = stall_rel;
      newton.SetController(stall);

      if (root)
      {
         mfem::out << "[hologram] solve elements=" << mesh.Nx()
                   << "x" << mesh.Ny()
                   << "  order=" << mesh.Order()
                   << "  samples=" << mesh.SampleNx() << "x" << mesh.SampleNy()
                   << "  forward=" << (fft ? "fft" : "gem")
                   << "  max_iter=" << max_iter
                   << "  cg_max_iter=" << std::max(1, cg_max_iter) << "\n";
      }

      Vector b(phi.Size());
      b.UseDevice(true);
      b = 0.0;

      mask_op.SetState(phi, op.last_scale);
      mask_op.SetLambda();
      step.SetOperator(mask_op);

      newton.Mult(b, phi);
      mask.Wrap(phi);
      mask.Modulate(phi, fields.u);

      Raster aperture;
      aperture(mesh, fields.u.real(), fields.u.imag(),
               mask_image, fields);

      acceptedSteps = newton.gn_ran;
      firstLoss = newton.firstLoss;
      finalLoss = newton.finalLoss;

      const char *why = "max_iter";
      if (newton.line_search_failed) { why = "line_search"; }
      else if (stall.stop_reason) { why = stall.stop_reason; }
      else if (newton.GetConverged()) { why = "abs_tol"; }
      if (root)
      {
         mfem::out << "[hologram] Newton stop reason=" << why << "\n";
      }
   }

   const RealArray2D &MaskImage() const { return mask_image; }
   int AcceptedSteps() const { return acceptedSteps; }
   real_t FirstLoss() const { return firstLoss; }
   real_t FinalLoss() const { return finalLoss; }
   const Prop &Propagator() const { return *propagator; }
   int SampleNx() const { return mesh.SampleNx(); }
   int SampleNy() const { return mesh.SampleNy(); }
   const Mesh &Grid() const { return mesh; }

private:
   Mesh mesh;
   RealArray2D target;
   const real_t wavelength, extent_x, extent_y, z, focal;
   const bool fft;
   RealArray2D mask_image;
   int acceptedSteps = 0;
   real_t firstLoss = 0.0, finalLoss = 0.0;
   std::unique_ptr<Prop> propagator;
};


// ----------------------------------------------------------------------------
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
   const real_t x_val = 1_r - std::fabs(std::fmod(h * 6_r, 2_r) - 1_r);
   switch (static_cast<int>(h * 6_r) % 6)
   {
      case 0: r = 1_r; g = x_val; b = 0_r; break;
      case 1: r = x_val; g = 1_r; b = 0_r; break;
      case 2: r = 0_r; g = 1_r; b = x_val; break;
      case 3: r = 0_r; g = x_val; b = 1_r; break;
      case 4: r = x_val; g = 0_r; b = 1_r; break;
      case 5: r = 1_r; g = 0_r; b = x_val; break;
      default: r = g = b = 0_r; break;
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
   if (value <= 0.00304_r) { return 12.92_r * value; }
   return 1.055_r * std::pow(value, 1_r / 2.4_r) - 0.055_r;
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
   constexpr real_t twoPi = 2_r * kPi;

   mfem::forall(N, [=] MFEM_HOST_DEVICE (int idx)
   {
      real_t h = (ph[idx] + kPi) / twoPi;
      h = h - std::floor(h);
      real_t r, g, b;
      detail::HsvToRgb(h, r, g, b);
      const int o = idx * 3;
      out[o + 0] = static_cast<unsigned char>(std::round(255_r * r));
      out[o + 1] = static_cast<unsigned char>(std::round(255_r * g));
      out[o + 2] = static_cast<unsigned char>(std::round(255_r * b));
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
      if (peak > 1_r)
      {
         const real_t scale = 1_r / peak;
         r *= scale;
         g *= scale;
         b *= scale;
      }
      if (r < 0_r) { r = 0_r; }
      else if (r > 1_r) { r = 1_r; }
      if (g < 0_r) { g = 0_r; }
      else if (g > 1_r) { g = 1_r; }
      if (b < 0_r) { b = 0_r; }
      else if (b > 1_r) { b = 1_r; }
      const int o = idx * 3;
      out[o]     = static_cast<unsigned char>(std::round(r * 255_r));
      out[o + 1] = static_cast<unsigned char>(std::round(g * 255_r));
      out[o + 2] = static_cast<unsigned char>(std::round(b * 255_r));
   });

   const unsigned char *host = rgb.HostRead();
   rgb_out.assign(host, host + static_cast<std::size_t>(N) * 3u);
   return true;
}

} // namespace holo
