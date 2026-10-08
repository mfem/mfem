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

#include "mfem.hpp"
#include "general/device.hpp"
#include "general/forall.hpp"
#include "linalg/complex.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#if defined(HOLO_USE_CPU_FFT)
#include <fftw3.h>
#endif

#if defined(HOLO_USE_GPU_FFT)
#if defined(MFEM_USE_CUDA)
#include <cufft.h>
#elif defined(MFEM_USE_HIP)
#if defined(__has_include)
#if __has_include(<hipfft/hipfft.h>)
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif
#else
#include <hipfft.h>
#endif
#else
#error "HOLO_USE_GPU_FFT requires MFEM_USE_CUDA or MFEM_USE_HIP"
#endif
#endif

namespace holo
{

using mfem::real_t;
using mfem::Vector;
using mfem::operator""_r;

using complex_t = mfem::complex_t;

using ComplexArray1D = mfem::Array<complex_t>;

class ComplexArray2D
{
   int rows = 0, cols = 0;
   mfem::Array<complex_t> data;

public:
   ComplexArray2D() = default;
   ComplexArray2D(int rows, int cols) { SetSize(rows, cols); }

   void SetSize(int num_rows, int num_cols)
   {
      rows = num_rows;
      cols = num_cols;
      data.UseDevice(true);
      data.SetSize(rows * cols);
   }

   int NumRows() const { return rows; }
   int NumCols() const { return cols; }

   complex_t *Write() { return data.Write(); }
   const complex_t *Read() const { return data.Read(); }
};

#if defined(HOLO_USE_CPU_FFT)
namespace
{
#if defined(MFEM_USE_SINGLE)
using fftw_cpx = fftwf_complex;
using fftw_pln = fftwf_plan;
#define holo_fftw_plan_dft_1d fftwf_plan_dft_1d
#define holo_fftw_plan_many_dft fftwf_plan_many_dft
#define holo_fftw_execute fftwf_execute
#define holo_fftw_destroy_plan fftwf_destroy_plan
#define holo_fftw_malloc fftwf_malloc
#define holo_fftw_free fftwf_free
#else
using fftw_cpx = fftw_complex;
using fftw_pln = fftw_plan;
#define holo_fftw_plan_dft_1d fftw_plan_dft_1d
#define holo_fftw_plan_many_dft fftw_plan_many_dft
#define holo_fftw_execute fftw_execute
#define holo_fftw_destroy_plan fftw_destroy_plan
#define holo_fftw_malloc fftw_malloc
#define holo_fftw_free fftw_free
#endif
static_assert(sizeof(complex_t) == sizeof(fftw_cpx));
} // namespace
#endif

// Fresnel propagation on an complex pixel image.
class FresnelOp : public mfem::Operator
{
protected:
   int nx = 0, ny = 0;
   real_t w = 0, z = 0;

   void SetImage(real_t w, real_t z, int nx, int ny);
   complex_t *Load(const Vector &x, Vector &y) const;

public:
   FresnelOp() = default;
   ~FresnelOp() override = default;
   FresnelOp(const FresnelOp &) = delete;
   FresnelOp &operator=(const FresnelOp &) = delete;

   virtual void Assemble(real_t w, real_t x,
                         real_t y, real_t z,
                         int nx, int ny) = 0;

   void Propagate(Vector &field) const;
};

inline std::unique_ptr<FresnelOp> MakeFresnelOp(bool use_fft);

namespace
{

// Fresnel weights ────────────────────────────────────────────────────────────
constexpr real_t kPi = M_PI;
constexpr real_t kHalf = 0.5_r;
constexpr real_t kHalfPi = kPi * kHalf;

MFEM_HOST_DEVICE inline void fresnel_series(real_t ax, real_t &C,
                                            real_t &S)
{
   // C(x) = Σ (-1)^n (π/2)^{2n} x^{4n+1} / ((2n)! (4n+1))
   // S(x) = Σ (-1)^n (π/2)^{2n+1} x^{4n+3} / ((2n+1)! (4n+3))
   const real_t x2 = ax * ax;
   const real_t factor = -(kHalfPi * kHalfPi) * (x2 * x2);
   real_t term_c = C = ax, term_s = S = kHalfPi * ax * x2 / 3_r;
   for (int n = 0; n < 60; ++n)
   {
      const int n2 = 2 * n, n4 = 4 * n;
      term_c *=
         factor * (static_cast<real_t>(n4 + 1) /
                   (static_cast<real_t>((n2 + 1) * (n2 + 2)) *
                    static_cast<real_t>(n4 + 5)));
      C += term_c;
      term_s *=
         factor * (static_cast<real_t>(n4 + 3) /
                   (static_cast<real_t>((n2 + 2) * (n2 + 3)) *
                    static_cast<real_t>(n4 + 7)));
      S += term_s;
      if (std::fabs(term_c) < 1e-18 * (1.0 + std::fabs(C)) &&
          std::fabs(term_s) < 1e-18 * (1.0 + std::fabs(S)))
      {
         break;
      }
   }
}

MFEM_HOST_DEVICE inline void fresnel_asymp_fg(real_t ax, real_t &C, real_t &S)
{
   // Auxiliary f,g for x > 0:
   //   C = 1/2 + f sin(θ) - g cos(θ),  S = 1/2 - f cos(θ) - g sin(θ),
   //   θ = π x² / 2
   // Evaluate f+ig via the continued fraction (resummed asymptotic)
   //   f+ig = 1/(π x) * 1 / (1 + 1*z/(1 + 2*z/(1 + 3*z/...))),
   //   z = -i / (π x²)
   const complex_t z{0_r, -1_r / (kPi * ax * ax)};
   constexpr real_t tiny = 1e-30_r, eps = 1e-15_r;
   complex_t f_cf{1_r, 0_r};
   complex_t C_lt = f_cf;
   complex_t D_lt{0_r, 0_r};
   for (int n = 1; n <= 200; ++n)
   {
      const complex_t az = static_cast<real_t>(n) * z;
      D_lt = 1_r + az * D_lt;
      if (abs(D_lt) < tiny) { D_lt = complex_t{tiny, 0_r}; }
      D_lt = 1_r / D_lt;
      C_lt = 1_r + az / C_lt;
      if (abs(C_lt) < tiny) { C_lt = complex_t{tiny, 0_r}; }
      const complex_t delta = C_lt * D_lt;
      f_cf *= delta;
      if (abs(delta - complex_t{1_r, 0_r}) < eps) { break; }
   }
   const complex_t fg = (1_r / f_cf) / (kPi * ax);
   const real_t f = fg.real(), g = fg.imag();
   const real_t theta = kHalfPi * ax * ax;
   const real_t sin_t = std::sin(theta);
   const real_t cos_t = std::cos(theta);
   C = kHalf + f * sin_t - g * cos_t;
   S = kHalf - f * cos_t - g * sin_t;
}

} // namespace

// C(x) and S(x) together (odd: C(-x)=-C(x), S(-x)=-S(x))
MFEM_HOST_DEVICE inline void fresnel_CS(real_t x, real_t &C, real_t &S)
{
   if (x == 0_r) { C = 0_r, S = 0_r; return; }
   constexpr real_t kFresnelSwitch = 1.6_r;
   const real_t ax = std::fabs(x);
   if (ax < kFresnelSwitch) { fresnel_series(ax, C, S); }
   else { fresnel_asymp_fg(ax, C, S); }
   if (x < 0_r) { C = -C, S = -S; }
}

MFEM_HOST_DEVICE inline void fresnel_phi(real_t slide, real_t u_hw,
                                         real_t sk_2z, real_t sfac,
                                         real_t amp, real_t z, real_t k,
                                         real_t &out_re, real_t &out_im)
{
   const real_t u_shift = sk_2z * slide;
   real_t C_1, S_1, C_2, S_2;
   fresnel_CS((-u_hw - u_shift) * sfac, C_1, S_1);
   fresnel_CS((+u_hw - u_shift) * sfac, C_2, S_2);
   const real_t theta = slide * slide * k / (2_r * z);
   const real_t er = std::cos(theta);
   const real_t ei = std::sin(theta);
   const real_t br = C_2 - C_1;
   const real_t bi = -(S_2 - S_1);
   out_re = amp * (er * br - ei * bi);
   out_im = amp * (er * bi + ei * br);
}

inline ComplexArray1D fresnel_first(real_t L, std::size_t N,
                                    real_t z, real_t k, bool row)
{
   const int n = static_cast<int>(N);
   const real_t d = L / static_cast<real_t>(N);
   const auto Nn = static_cast<real_t>(N);
   const real_t x0 = -0.5_r * L;
   const real_t W = 1_r / (2_r * d);
   const real_t s2z_k = std::sqrt(2.0 * z / k);
   const real_t sk_2z = std::sqrt(k / (2_r * z));
   const real_t u_hw = kPi * s2z_k * W;
   const real_t sfac = std::sqrt(2_r / kPi);
   const real_t amp = (d / kPi) * sk_2z * std::sqrt(kPi / 2_r);
   ComplexArray1D out(n);
   auto *o = out.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t xi = L * (static_cast<real_t>(i) / Nn - 0.5_r);
      const real_t slide = row ? (x0 - xi) : (xi - x0);
      real_t rr, ii;
      fresnel_phi(slide, u_hw, sk_2z, sfac, amp, z, k, rr, ii);
      o[i] = complex_t{rr, ii};
   });
   return out;
}

inline ComplexArray2D conj_transpose(const ComplexArray2D &A)
{
   const int nr = A.NumRows();
   const int nc = A.NumCols();
   const int N = nr * nc;
   ComplexArray2D T(nc, nr);
   const auto *a = A.Read();
   auto *t = T.Write();
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx / nc;
      const int j = idx - i * nc;
      const complex_t z = a[idx];
      t[j * nr + i] = complex_t{z.real(), -z.imag()};
   });
   return T;
}

struct ToeplitzGenerators
{
   ComplexArray1D fr_x, fc_x, fr_y, fc_y;
};

inline ToeplitzGenerators
FresnelWeights(real_t extent_x, real_t extent_y, int nx, int ny,
               real_t z, real_t k)
{
   const auto Nx = static_cast<std::size_t>(nx);
   const auto Ny = static_cast<std::size_t>(ny);
   const ComplexArray1D wx = fresnel_first(extent_x, Nx, z, k, true);
   const ComplexArray1D wy = fresnel_first(extent_y, Ny, z, k, true);
   return { wx, wx, wy, wy };
}

// Toeplitz tools ─────────────────────────────────────────────────────────────
inline ComplexArray2D toeplitz_to_dense(const ComplexArray1D &first_row,
                                        const ComplexArray1D &first_col)
{
   MFEM_VERIFY(first_row.Size() == first_col.Size(),
               "toeplitz_to_dense: row/col size mismatch");
   const int n = first_row.Size();
   ComplexArray2D H(n, n);
   const complex_t *row = first_row.Read();
   const complex_t *col = first_col.Read();
   complex_t *h = H.Write();
   mfem::forall(n * n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx / n;
      const int j = idx - i * n;
      const int d = i - j;
      h[idx] = (d >= 0) ? col[d] : row[-d];
   });
   return H;
}

inline void dense_left_multiply(const ComplexArray2D &H,
                                const complex_t *b, complex_t *c, int k)
{
   const int n = H.NumRows();
   MFEM_VERIFY(H.NumCols() == n, "dense_left_multiply shape");
   const complex_t *h = H.Read();
   mfem::forall(n * k, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx / k;
      const int j = idx - i * k;
      complex_t s{0_r, 0_r};
      for (int t = 0; t < n; ++t)
      {
         s = s + h[i * n + t] * b[t * k + j];
      }
      c[idx] = s;
   });
}

inline void dense_right_multiply(const complex_t *b, const ComplexArray2D &H,
                                 complex_t *c, int m)
{
   const int n = H.NumRows();
   MFEM_VERIFY(H.NumCols() == n, "dense_right_multiply shape");
   const complex_t *h = H.Read();
   mfem::forall(m * n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx / n;
      const int j = idx - i * n;
      complex_t s{0_r, 0_r};
      for (int t = 0; t < n; ++t)
      {
         s = s + b[i * n + t] * h[t * n + j];
      }
      c[idx] = s;
   });
}

inline std::size_t ToeplitzEmbedSize(std::size_t n)
{
   const auto NextPow2 = [](std::size_t x) -> std::size_t
   {
      if (x <= 1) { return 1; }
      --x;
      (x |= x >> 1, x |= x >> 2, x |= x >> 4);
      (x |= x >> 8, x |= x >> 16, x |= x >> 32);
      return x + 1;
   };
   return NextPow2(std::max(2 * n, static_cast<std::size_t>(2)));
}

inline void ToeplitzEmbed(const ComplexArray1D &first_row,
                          const ComplexArray1D &first_col,
                          ComplexArray1D &circ)
{
   const int n = first_row.Size();
   MFEM_VERIFY(first_col.Size() == n, "ToeplitzEmbed row/col size");
   const int m = static_cast<int>(ToeplitzEmbedSize(static_cast<std::size_t>(n)));
   circ.UseDevice(true);
   circ.SetSize(m);
   const complex_t *row = first_row.Read();
   const complex_t *col = first_col.Read();
   complex_t *c = circ.Write();
   mfem::forall(m, [=] MFEM_HOST_DEVICE (int i)
   {
      if (i == 0) { c[i] = row[0]; }
      else if (i < n) { c[i] = row[i]; }
      else if (i > m - n) { c[i] = col[m - i]; }
      else { c[i] = complex_t{0.0, 0.0}; }
   });
}

inline void FillAdjointWeights(const ComplexArray1D &fr,
                               const ComplexArray1D &fc,
                               ComplexArray1D &adj_fr,
                               ComplexArray1D &adj_fc)
{
   const int n = fr.Size();
   MFEM_VERIFY(fc.Size() == n, "FillAdjointWeights size");
   adj_fr.UseDevice(true), adj_fc.UseDevice(true);
   adj_fr.SetSize(n), adj_fc.SetSize(n);
   const auto *frp = fr.Read(), *fcp = fc.Read();
   auto *afr = adj_fr.Write(), *afc = adj_fc.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      afr[i] = conj(fcp[i]);
      afc[i] = conj(frp[i]);
   });
}

// ────────────────────────────────────────────────────────────────────────────
/** Circulant embedding of the Toeplitz Fresnel weights. */
class ToeplitzFft
{
public:
   virtual ~ToeplitzFft() = default;
   virtual void Prepare(const ComplexArray1D &fr_x, const ComplexArray1D &fc_x,
                        const ComplexArray1D &fr_y, const ComplexArray1D &fc_y,
                        int nx, int ny) = 0;
   virtual void AxisY(complex_t *E, bool adjoint) const = 0;
   virtual void AxisX(complex_t *E, bool adjoint) const = 0;
};

// FFT implementations ────────────────────────────────────────────────────────
#if defined(HOLO_USE_CPU_FFT)
class CpuFFT
{
   struct Impl
   {
      int n = 0, m = 0, batch = 0;
      bool along_rows = false;
      complex_t *buf = nullptr;
      fftw_pln fwd = nullptr, inv = nullptr;

      ~Impl()
      {
         if (fwd) { holo_fftw_destroy_plan(fwd); }
         if (inv) { holo_fftw_destroy_plan(inv); }
         if (buf) { holo_fftw_free(buf); }
      }
   };
   std::unique_ptr<Impl> impl;

public:
   inline CpuFFT() = default;
   inline ~CpuFFT() = default;
   inline CpuFFT(CpuFFT &&) noexcept = default;
   inline CpuFFT &operator=(CpuFFT &&) noexcept = default;

   CpuFFT(const CpuFFT &) = delete;
   CpuFFT &operator=(const CpuFFT &) = delete;

   static inline void Forward(complex_t *data, int m)
   {
      auto plan = holo_fftw_plan_dft_1d(
                     m, reinterpret_cast<fftw_cpx *>(data),
                     reinterpret_cast<fftw_cpx *>(data),
                     FFTW_FORWARD, FFTW_ESTIMATE);
      MFEM_VERIFY(plan, "FFTW could not plan a spectrum of length " << m);
      holo_fftw_execute(plan);
      holo_fftw_destroy_plan(plan);
   }

   inline void Plan(int n_in, int m_in, int batch_in, bool along_rows_in)
   {
      impl = std::make_unique<Impl>();
      impl->n = n_in;
      impl->m = m_in;
      impl->batch = batch_in;
      impl->along_rows = along_rows_in;
      impl->buf = static_cast<complex_t *>(
                     holo_fftw_malloc(sizeof(complex_t) * static_cast<std::size_t>(m_in) *
                                      static_cast<std::size_t>(batch_in)));
      MFEM_VERIFY(impl->buf, "FFTW buffer allocation failed");
      impl->fwd = holo_fftw_plan_many_dft(
                     1, &impl->m, impl->batch,
                     reinterpret_cast<fftw_cpx *>(impl->buf), nullptr, 1, impl->m,
                     reinterpret_cast<fftw_cpx *>(impl->buf), nullptr, 1, impl->m,
                     FFTW_FORWARD, FFTW_ESTIMATE);
      impl->inv = holo_fftw_plan_many_dft(
                     1, &impl->m, impl->batch,
                     reinterpret_cast<fftw_cpx *>(impl->buf), nullptr, 1, impl->m,
                     reinterpret_cast<fftw_cpx *>(impl->buf), nullptr, 1, impl->m,
                     FFTW_BACKWARD, FFTW_ESTIMATE);
      MFEM_VERIFY(impl->fwd && impl->inv,
                  "FFTW could not plan the batched Fresnel FFT");
   }

   inline void Mult(complex_t *E, const complex_t *freq) const
   {
      MFEM_VERIFY(impl && impl->buf, "CpuFFT::Apply before Plan");
      const int n = impl->n, m = impl->m;
      const int batch = impl->batch;
      const bool rows = impl->along_rows;
      const int nx = rows ? n : batch;
      const complex_t *e = E;
      complex_t *buf = impl->buf;
      for (int line = 0; line < batch; ++line)
      {
         complex_t *b = buf + static_cast<std::size_t>(line) * m;
         for (int i = 0; i < m; ++i) { b[i] = complex_t{0.0, 0.0}; }
         for (int i = 0; i < n; ++i)
         {
            b[i] = rows ? e[line * nx + i] : e[i * nx + line];
         }
      }
      holo_fftw_execute(impl->fwd);
      for (int line = 0; line < batch; ++line)
      {
         complex_t *b = buf + static_cast<std::size_t>(line) * m;
         for (int i = 0; i < m; ++i) { b[i] *= freq[i]; }
      }
      holo_fftw_execute(impl->inv);
      const real_t scale = 1.0 / static_cast<real_t>(m);
      complex_t *o = E;
      for (int line = 0; line < batch; ++line)
      {
         const complex_t *b = buf + static_cast<std::size_t>(line) * m;
         for (int i = 0; i < n; ++i)
         {
            const complex_t z = b[i] * scale;
            if (rows) { o[line * nx + i] = z; }
            else { o[i * nx + line] = z; }
         }
      }
   }
};

class CpuToeplitzFft : public ToeplitzFft
{
   CpuFFT axis_y, axis_x;
   std::vector<complex_t> y_fwd, y_adj, x_fwd, x_adj;

   static void Spectrum(const ComplexArray1D &row, const ComplexArray1D &col,
                        std::vector<complex_t> &host)
   {
      ComplexArray1D circ;
      ToeplitzEmbed(row, col, circ);
      host.resize(static_cast<std::size_t>(circ.Size()));
      const complex_t *src = circ.HostRead();
      for (int i = 0; i < circ.Size(); ++i) { host[static_cast<std::size_t>(i)] = src[i]; }
      CpuFFT::Forward(host.data(), circ.Size());
   }

public:
   void Prepare(const ComplexArray1D &fr_x, const ComplexArray1D &fc_x,
                const ComplexArray1D &fr_y, const ComplexArray1D &fc_y,
                int nx, int ny) override
   {
      ComplexArray1D adj_fr_x, adj_fc_x, adj_fr_y, adj_fc_y;
      FillAdjointWeights(fr_x, fc_x, adj_fr_x, adj_fc_x);
      FillAdjointWeights(fr_y, fc_y, adj_fr_y, adj_fc_y);
      Spectrum(fr_y, fc_y, y_fwd);
      Spectrum(adj_fr_y, adj_fc_y, y_adj);
      Spectrum(fc_x, fr_x, x_fwd);
      Spectrum(adj_fc_x, adj_fr_x, x_adj);
      axis_y.Plan(ny, static_cast<int>(y_fwd.size()), nx, false);
      axis_x.Plan(nx, static_cast<int>(x_fwd.size()), ny, true);
   }

   void AxisY(complex_t *E, bool adjoint) const override
   {
      axis_y.Mult(E, (adjoint ? y_adj : y_fwd).data());
   }

   void AxisX(complex_t *E, bool adjoint) const override
   {
      axis_x.Mult(E, (adjoint ? x_adj : x_fwd).data());
   }
};

inline std::unique_ptr<ToeplitzFft> MakeCpuToeplitzFft()
{
   return std::make_unique<CpuToeplitzFft>();
}

#undef holo_fftw_plan_dft_1d
#undef holo_fftw_plan_many_dft
#undef holo_fftw_execute
#undef holo_fftw_destroy_plan
#undef holo_fftw_malloc
#undef holo_fftw_free
#endif // HOLO_USE_CPU_FFT

#if defined(HOLO_USE_GPU_FFT)
namespace
{

#if defined(MFEM_USE_CUDA)
#if defined(MFEM_USE_SINGLE)
using gpu_cpx = cufftComplex;
constexpr cufftType gpu_type = CUFFT_C2C;
inline cufftResult gpu_exec(cufftHandle plan, gpu_cpx *z, int dir)
{
   return cufftExecC2C(plan, z, z, dir);
}
#else
using gpu_cpx = cufftDoubleComplex;
constexpr cufftType gpu_type = CUFFT_Z2Z;
inline cufftResult gpu_exec(cufftHandle plan, gpu_cpx *z, int dir)
{
   return cufftExecZ2Z(plan, z, z, dir);
}
#endif // MFEM_USE_SINGLE
void Check(cufftResult result, const char *what)
{
   MFEM_VERIFY(result == CUFFT_SUCCESS,
               "GPU FFT error in " << what << " (code "
               << static_cast<int>(result) << ")");
}
#elif defined(MFEM_USE_HIP)
#if defined(MFEM_USE_SINGLE)
using gpu_cpx = hipfftComplex;
constexpr hipfftType gpu_type = HIPFFT_C2C;
inline hipfftResult gpu_exec(hipfftHandle plan, gpu_cpx *z, int dir)
{
   return hipfftExecC2C(plan, z, z, dir);
}
#else
using gpu_cpx = hipfftDoubleComplex;
constexpr hipfftType gpu_type = HIPFFT_Z2Z;
inline hipfftResult gpu_exec(hipfftHandle plan, gpu_cpx *z, int dir)
{
   return hipfftExecZ2Z(plan, z, z, dir);
}
#endif // MFEM_USE_SINGLE
void Check(hipfftResult result, const char *what)
{
   MFEM_VERIFY(result == HIPFFT_SUCCESS,
               "GPU FFT error in " << what << " (code "
               << static_cast<int>(result) << ")");
}
#endif

void EmbedCols(const complex_t *e, complex_t *b, int n, int m, int batch)
{
   mfem::forall(batch * m, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int line = idx / m;
      const int i = idx - line * m;
      b[idx] = (i < n) ? e[i * batch + line] : complex_t{0.0, 0.0};
   });
}

void EmbedRows(const complex_t *e, complex_t *b, int n, int m, int batch)
{
   mfem::forall(batch * m, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int line = idx / m;
      const int i = idx - line * m;
      b[idx] = (i < n) ? e[line * n + i] : complex_t{0.0, 0.0};
   });
}

void Pointwise(complex_t *b, const complex_t *spectrum, int m, int batch)
{
   mfem::forall(batch * m, [=] MFEM_HOST_DEVICE (int idx)
   {
      b[idx] *= spectrum[idx - (idx / m) * m];
   });
}

void ExtractCols(const complex_t *b, complex_t *e,
                 int n, int m, int batch, real_t scale)
{
   mfem::forall(n * batch, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int i = idx / batch;
      const int line = idx - i * batch;
      e[idx] = b[line * m + i] * scale;
   });
}

void ExtractRows(const complex_t *b, complex_t *e,
                 int n, int m, int batch, real_t scale)
{
   mfem::forall(batch * n, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int line = idx / n;
      const int i = idx - line * n;
      e[idx] = b[line * m + i] * scale;
   });
}

} // namespace

class GpuFFT
{
   struct Impl;
   std::unique_ptr<Impl> impl;
public:
   inline GpuFFT() = default;
   inline ~GpuFFT() = default;
   inline GpuFFT(GpuFFT &&) noexcept = default;
   inline GpuFFT &operator=(GpuFFT &&) noexcept = default;

   GpuFFT(const GpuFFT &) = delete;
   GpuFFT &operator=(const GpuFFT &) = delete;
   static void Forward(complex_t *data, int m);
   void Plan(int n, int m, int batch, bool along_rows);
   void Mult(complex_t *E, const complex_t *spectrum) const;
};

struct GpuFFT::Impl
{
#if defined(MFEM_USE_CUDA)
   cufftHandle handle {};
#elif defined(MFEM_USE_HIP)
   hipfftHandle handle {};
#endif
   void *workspace = nullptr;
   int n = 0, m = 0, batch = 0;
   bool along_rows = false;
   mutable ComplexArray1D embed;

   ~Impl() { Release(); }

   void Release()
   {
#if defined(MFEM_USE_CUDA)
      if (handle) { cufftDestroy(handle); }
      if (workspace) { MFEM_GPU_CHECK(cudaFree(workspace)); }
#elif defined(MFEM_USE_HIP)
      if (handle) { hipfftDestroy(handle); }
      if (workspace) { MFEM_GPU_CHECK(hipFree(workspace)); }
#endif
      handle = {};
      workspace = nullptr;
   }

   void Create(int n_in, int m_in, int batch_in, bool along_rows_in)
   {
      n = n_in;
      m = m_in;
      batch = batch_in;
      along_rows = along_rows_in;
      int nfft = m;
      std::size_t bytes = 0;
#if defined(MFEM_USE_CUDA)
      Check(cufftCreate(&handle), "Fresnel create");
      Check(cufftSetAutoAllocation(handle, 0), "Fresnel workspace");
      Check(cufftMakePlanMany(handle, 1, &nfft, nullptr, 1, m,
                              nullptr, 1, m, gpu_type, batch, &bytes),
            "Fresnel plan");
      if (bytes > 0)
      {
         MFEM_GPU_CHECK(cudaMalloc(&workspace, bytes));
         Check(cufftSetWorkArea(handle, workspace), "Fresnel work area");
      }
#elif defined(MFEM_USE_HIP)
      Check(hipfftCreate(&handle), "Fresnel create");
      Check(hipfftSetAutoAllocation(handle, 0), "Fresnel workspace");
      Check(hipfftMakePlanMany(handle, 1, &nfft, nullptr, 1, m,
                               nullptr, 1, m, gpu_type, batch, &bytes),
            "Fresnel plan");
      if (bytes > 0)
      {
         MFEM_GPU_CHECK(hipMalloc(&workspace, bytes));
         Check(hipfftSetWorkArea(handle, workspace), "Fresnel work area");
      }
#endif
      embed.UseDevice(true);
      embed.SetSize(batch * m);
      embed.Write();
   }

   void Execute(complex_t *data, bool inverse) const
   {
#if defined(MFEM_USE_CUDA)
      auto *z = reinterpret_cast<gpu_cpx *>(data);
      Check(gpu_exec(handle, z, inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
            inverse ? "Fresnel inverse" : "Fresnel forward");
#elif defined(MFEM_USE_HIP)
      auto *z = reinterpret_cast<gpu_cpx *>(data);
      Check(gpu_exec(handle, z, inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD),
            inverse ? "Fresnel inverse" : "Fresnel forward");
#endif
   }

   static void ForwardInPlace(complex_t *data, int length)
   {
      int nfft = length;
#if defined(MFEM_USE_CUDA)
      cufftHandle plan {};
      Check(cufftPlanMany(&plan, 1, &nfft, nullptr, 1, length,
                          nullptr, 1, length, gpu_type, 1),
            "spectrum plan");
      auto *z = reinterpret_cast<gpu_cpx *>(data);
      Check(gpu_exec(plan, z, CUFFT_FORWARD), "spectrum exec");
      cufftDestroy(plan);
#elif defined(MFEM_USE_HIP)
      hipfftHandle plan {};
      Check(hipfftPlanMany(&plan, 1, &nfft, nullptr, 1, length,
                           nullptr, 1, length, gpu_type, 1),
            "spectrum plan");
      auto *z = reinterpret_cast<gpu_cpx *>(data);
      Check(gpu_exec(plan, z, HIPFFT_FORWARD), "spectrum exec");
      hipfftDestroy(plan);
#endif
   }
};

inline void GpuFFT::Forward(complex_t *data, int m)
{
   Impl::ForwardInPlace(data, m);
}

inline void GpuFFT::Plan(int n, int m, int batch, bool along_rows)
{
   impl = std::make_unique<Impl>();
   impl->Create(n, m, batch, along_rows);
}

inline void GpuFFT::Mult(complex_t *E, const complex_t *spectrum) const
{
   MFEM_VERIFY(impl, "GpuFFT::Apply before Plan");
   const int n = impl->n;
   const int m = impl->m;
   const int batch = impl->batch;
   complex_t *b = impl->embed.Write();
   if (impl->along_rows) { EmbedRows(E, b, n, m, batch); }
   else { EmbedCols(E, b, n, m, batch); }
   impl->Execute(impl->embed.ReadWrite(), false);
   Pointwise(impl->embed.ReadWrite(), spectrum, m, batch);
   impl->Execute(impl->embed.ReadWrite(), true);
   const real_t scale = 1.0 / static_cast<real_t>(m);
   if (impl->along_rows) { ExtractRows(impl->embed.Read(), E, n, m, batch, scale); }
   else { ExtractCols(impl->embed.Read(), E, n, m, batch, scale); }
}

class GpuToeplitzFft : public ToeplitzFft
{
   GpuFFT axis_y, axis_x;
   ComplexArray1D spec_y_fwd, spec_y_adj, spec_x_fwd, spec_x_adj;

   static void Spectrum(const ComplexArray1D &row, const ComplexArray1D &col,
                        ComplexArray1D &spec)
   {
      ToeplitzEmbed(row, col, spec);
      GpuFFT::Forward(spec.ReadWrite(), spec.Size());
   }

public:
   void Prepare(const ComplexArray1D &fr_x, const ComplexArray1D &fc_x,
                const ComplexArray1D &fr_y, const ComplexArray1D &fc_y,
                int nx, int ny) override
   {
      ComplexArray1D adj_fr_x, adj_fc_x, adj_fr_y, adj_fc_y;
      FillAdjointWeights(fr_x, fc_x, adj_fr_x, adj_fc_x);
      FillAdjointWeights(fr_y, fc_y, adj_fr_y, adj_fc_y);
      Spectrum(fr_y, fc_y, spec_y_fwd);
      Spectrum(adj_fr_y, adj_fc_y, spec_y_adj);
      Spectrum(fc_x, fr_x, spec_x_fwd);
      Spectrum(adj_fc_x, adj_fr_x, spec_x_adj);
      axis_y.Plan(ny, spec_y_fwd.Size(), nx, false);
      axis_x.Plan(nx, spec_x_fwd.Size(), ny, true);
   }

   void AxisY(complex_t *E, bool adjoint) const override
   {
      axis_y.Mult(E, (adjoint ? spec_y_adj : spec_y_fwd).Read());
   }

   void AxisX(complex_t *E, bool adjoint) const override
   {
      axis_x.Mult(E, (adjoint ? spec_x_adj : spec_x_fwd).Read());
   }
};

inline std::unique_ptr<ToeplitzFft> MakeGpuToeplitzFft()
{
   return std::make_unique<GpuToeplitzFft>();
}

#endif // HOLO_USE_GPU_FFT

inline void AbortMissingFft()
{
   const char *lib = "FFTW";
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK)) { lib = "cuFFT"; }
   else if (mfem::Device::Allows(mfem::Backend::HIP_MASK)) { lib = "hipFFT"; }
   MFEM_ABORT("hologram -fft needs " << lib
              << ", which was not found when hologram was built");
}

inline bool FftLinkedForDevice()
{
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK))
   {
#if defined(HOLO_USE_GPU_FFT)
      return true;
#else
      return false;
#endif
   }
#if defined(HOLO_USE_CPU_FFT)
   return true;
#else
   return false;
#endif
}

inline std::unique_ptr<ToeplitzFft> MakeToeplitzFft()
{
   if (mfem::Device::Allows(mfem::Backend::CUDA_MASK | mfem::Backend::HIP_MASK))
   {
#if defined(HOLO_USE_GPU_FFT)
      return MakeGpuToeplitzFft();
#else
      return nullptr;
#endif
   }
#if defined(HOLO_USE_CPU_FFT)
   return MakeCpuToeplitzFft();
#else
   return nullptr;
#endif
}

// ────────────────────────────────────────────────────────────────────────────
inline void FresnelOp::SetImage(real_t wavelength_in, real_t z_in,
                                int nx_in, int ny_in)
{
   w = wavelength_in;
   z = z_in;
   nx = nx_in;
   ny = ny_in;
   height = width = 2 * nx * ny;
}

inline complex_t *FresnelOp::Load(const Vector &x, Vector &y) const
{
   const int N = nx * ny;
   MFEM_VERIFY(x.Size() == 2 * N, "FresnelOp input size");
   y.UseDevice(true);
   if (y.Size() != 2 * N) { y.SetSize(2 * N); }
   if (&x != &y) { y = x; }
   return reinterpret_cast<complex_t *>(y.ReadWrite());
}

inline void FresnelOp::Propagate(Vector &field) const
{
   const int N = nx * ny;
   MFEM_VERIFY(field.Size() == 2 * N, "FresnelOp replay size");
   const real_t k = 2.0 * M_PI / w;
   const complex_t phase = exp(complex_t{0.0, k * z});
   auto *Z = reinterpret_cast<complex_t *>(field.ReadWrite());
   mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { Z[i] *= phase; });
   Mult(field, field);
}

// ────────────────────────────────────────────────────────────────────────────
class GemFresnel : public FresnelOp
{
   ComplexArray2D Hx, Hy, Hx_adj, Hy_adj;
   mutable Vector t;

   void Mult(const Vector &x, Vector &y, bool adjoint) const
   {
      complex_t *E = Load(x, y);
      auto *T = reinterpret_cast<complex_t *>(t.ReadWrite());
      dense_left_multiply(adjoint ? Hy_adj : Hy, E, T, nx);
      dense_right_multiply(T, adjoint ? Hx_adj : Hx, E, ny);
   }

public:
   void Assemble(real_t wavelength_in, real_t extent_x, real_t extent_y,
                 real_t z_in, int nx_in, int ny_in) override
   {
      SetImage(wavelength_in, z_in, nx_in, ny_in);
      const real_t k = 2.0 * M_PI / w;
      const ToeplitzGenerators g =
         FresnelWeights(extent_x, extent_y, nx, ny, z, k);
      t.SetSize(2 * nx * ny);
      t.UseDevice(true);
      t.Write();
      Hx = toeplitz_to_dense(g.fr_x, g.fc_x);
      Hy = toeplitz_to_dense(g.fr_y, g.fc_y);
      Hx_adj = conj_transpose(Hx);
      Hy_adj = conj_transpose(Hy);
   }

   void Mult(const Vector &x, Vector &y) const override { Mult(x, y, false); }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y, true); }
};

class FftFresnel : public FresnelOp
{
   std::unique_ptr<ToeplitzFft> fft;

   void Mult(const Vector &x, Vector &y, bool adjoint) const
   {
      complex_t *E = Load(x, y);
      fft->AxisY(E, adjoint);
      fft->AxisX(E, adjoint);
   }

public:
   void Assemble(real_t wavelength_in, real_t extent_x, real_t extent_y,
                 real_t z_in, int nx_in, int ny_in) override
   {
      SetImage(wavelength_in, z_in, nx_in, ny_in);
      const real_t k = 2.0 * M_PI / w;
      const ToeplitzGenerators g =
         FresnelWeights(extent_x, extent_y, nx, ny, z, k);
      if (!fft) { fft = MakeToeplitzFft(); }
      if (!fft) { AbortMissingFft(); }
      fft->Prepare(g.fr_x, g.fc_x, g.fr_y, g.fc_y, nx, ny);
   }

   void Mult(const Vector &x, Vector &y) const override { Mult(x, y, false); }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y, true); }
};

inline std::unique_ptr<FresnelOp> MakeFresnelOp(bool use_fft)
{
   if (use_fft) { return std::make_unique<FftFresnel>(); }
   return std::make_unique<GemFresnel>();
}

} // namespace holo
