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
#include "catch.hpp"
#include "miniapps/optics/sinc.hpp"

#include <cmath>

using namespace mfem;

namespace
{

void FillField(Vector &v, int stride)
{
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = real_t((i * stride) % 11 + 1) / real_t{9};
   }
}

void RequireAdjoint(holo::FresnelOp &op, const Vector &x, const Vector &y)
{
   Vector Ax(x.Size()), AHy(y.Size());
   op.Mult(x, Ax);
   op.MultTranspose(y, AHy);
   Ax.HostRead();
   AHy.HostRead();
   const real_t lhs = InnerProduct(Ax, y);
   const real_t rhs = InnerProduct(x, AHy);
   const real_t scale = std::fmax(std::fabs(lhs), std::fabs(rhs));
   REQUIRE(std::fabs(lhs - rhs) <= scale * real_t{1e-10} + real_t{1e-12});
}

void RequireClose(const Vector &a, const Vector &b)
{
   a.HostRead();
   b.HostRead();
   Vector d(a.Size());
   subtract(a, b, d);
   const real_t err = d.Norml2();
   const real_t scale = std::fmax(a.Norml2(), b.Norml2());
   REQUIRE(err <= scale * real_t{1e-8} + real_t{1e-12});
}

} // namespace

TEST_CASE("Fresnel dense and FFT propagation", "[Complex]")
{
   Device device("cpu");
   const int n = 4;
   const real_t wavelength = real_t{532e-9};
   const real_t extent = real_t{1e-3};
   const real_t z = real_t{0.05};
   auto gem = holo::MakeFresnelOp(false);
   gem->Assemble(wavelength, extent, extent, z, n, n);

   Vector x(2 * n * n), y(2 * n * n);
   FillField(x, 3);
   FillField(y, 5);
   RequireAdjoint(*gem, x, y);

#ifdef HOLO_USE_CPU_FFT
   auto fft = holo::MakeFresnelOp(true);
   fft->Assemble(wavelength, extent, extent, z, n, n);
   RequireAdjoint(*fft, x, y);

   Vector dense(x.Size()), transformed(x.Size());
   gem->Mult(x, dense);
   fft->Mult(x, transformed);
   RequireClose(dense, transformed);
   gem->MultTranspose(y, dense);
   fft->MultTranspose(y, transformed);
   RequireClose(dense, transformed);
#endif
}
