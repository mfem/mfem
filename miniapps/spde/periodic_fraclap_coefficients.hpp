// periodic_fraclap_coefficients.hpp
#pragma once

#include "mfem.hpp"
#include <cmath>

namespace periodic_fraclap
{
using mfem::Vector;
using mfem::real_t;

static constexpr real_t pi =
   static_cast<real_t>(3.141592653589793238462643383279502884L);

// -----------------------------------------------------------------------------
// 2D unit periodic box: Omega = [0,1]^2
//
// u(x,y) = 3
//        + sin(2*pi*x)
//        + 2*cos(2*pi*y)
//        + 0.5*cos(2*pi*(x+y))
//
// (-Delta)^(alpha/2) u =
//        (2*pi)^alpha sin(2*pi*x)
//      + 2*(2*pi)^alpha cos(2*pi*y)
//      + 0.5*(2*pi*sqrt(2))^alpha cos(2*pi*(x+y))
// -----------------------------------------------------------------------------

inline real_t U2D(const Vector &X)
{
   const real_t x = X(0);
   const real_t y = X(1);
   const real_t t = 2.0 * pi;

   return 3.0
        + std::sin(t * x)
        + 2.0 * std::cos(t * y)
        + 0.5 * std::cos(t * (x + y));
}

inline real_t FracLapU2D(const Vector &X, real_t alpha)
{
   const real_t x = X(0);
   const real_t y = X(1);
   const real_t t = 2.0 * pi;

   return std::pow(t, alpha) * std::sin(t * x)
        + 2.0 * std::pow(t, alpha) * std::cos(t * y)
        + 0.5 * std::pow(t * std::sqrt(2.0), alpha)
              * std::cos(t * (x + y));
}

// -----------------------------------------------------------------------------
// 3D unit periodic box: Omega = [0,1]^3
//
// u(x,y,z) = 1
//          + sin(2*pi*x)
//          + cos(2*pi*(y+z))
//          + 2*sin(2*pi*(x+y+z))
//
// (-Delta)^(alpha/2) u =
//        (2*pi)^alpha sin(2*pi*x)
//      + (2*pi*sqrt(2))^alpha cos(2*pi*(y+z))
//      + 2*(2*pi*sqrt(3))^alpha sin(2*pi*(x+y+z))
// -----------------------------------------------------------------------------

inline real_t U3D(const Vector &X)
{
   const real_t x = X(0);
   const real_t y = X(1);
   const real_t z = X(2);
   const real_t t = 2.0 * pi;

   return 1.0
        + std::sin(t * x)
        + std::cos(t * (y + z))
        + 2.0 * std::sin(t * (x + y + z));
}

inline real_t FracLapU3D(const Vector &X, real_t alpha)
{
   const real_t x = X(0);
   const real_t y = X(1);
   const real_t z = X(2);
   const real_t t = 2.0 * pi;

   return std::pow(t, alpha) * std::sin(t * x)
        + std::pow(t * std::sqrt(2.0), alpha)
              * std::cos(t * (y + z))
        + 2.0 * std::pow(t * std::sqrt(3.0), alpha)
              * std::sin(t * (x + y + z));
}

// -----------------------------------------------------------------------------
// Exact solution on Omega = (0,1)^2:
//
// u(x,y) = cos(pi*x)
//        + 2*cos(2*pi*y)
//        + 0.5*cos(pi*x)*cos(2*pi*y)
//
// This satisfies du/dn = 0 on all sides of the unit square.
// -----------------------------------------------------------------------------

inline real_t UExact(const Vector &X)
{
   const real_t x = X(0);
   const real_t y = X(1);

   return std::cos(pi * x)
        + 2.0 * std::cos(2.0 * pi * y)
        + 0.5 * std::cos(pi * x) * std::cos(2.0 * pi * y);
}

// -----------------------------------------------------------------------------
// RHS for the spectral Neumann fractional Laplacian:
//
// f = (-Delta_N)^(alpha/2) u
//
// alpha = 1 gives the half-Laplacian.
// alpha = 2 gives the classical -Delta.
// -----------------------------------------------------------------------------

inline real_t RHS(const Vector &X, real_t alpha)
{
   const real_t x = X(0);
   const real_t y = X(1);

   const real_t mode_10 = std::cos(pi * x);
   const real_t mode_02 = std::cos(2.0 * pi * y);
   const real_t mode_12 = std::cos(pi * x) * std::cos(2.0 * pi * y);

   const real_t lambda_10 = pi * pi;
   const real_t lambda_02 = 4.0 * pi * pi;
   const real_t lambda_12 = 5.0 * pi * pi;

   return std::pow(lambda_10, 0.5 * alpha) * mode_10
        + 2.0 * std::pow(lambda_02, 0.5 * alpha) * mode_02
        + 0.5 * std::pow(lambda_12, 0.5 * alpha) * mode_12;
}



} // namespace periodic_fraclap
