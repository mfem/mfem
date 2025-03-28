// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ELTRANS_BASIS
#define MFEM_ELTRANS_BASIS

#include "../../general/forall.hpp"

#include "../geom.hpp"

// this file contains utilities for computing nodal basis functions and their
// derivatives in device kernels

namespace mfem
{

namespace eltrans
{

/// Various utilities for working with different element geometries
template <int GeomType> struct GeometryUtils;

template <> struct GeometryUtils<Geometry::SEGMENT>
{
   static constexpr MFEM_HOST_DEVICE int Dimension() { return 1; }

   /// @b true if the given point x in ref space is inside the element
   static bool MFEM_HOST_DEVICE inside(real_t x) { return x >= 0 && x <= 1; }

   /// @b Bound the reference coordinate @a x += dx to be inside the segment.
   /// @a dx is updated to be dx = project(x+dx) - x
   /// @return true if x + dx hit a boundary
   static bool MFEM_HOST_DEVICE project(real_t &x, real_t &dx)
   {
      real_t tmp = x;
      x += dx;
      if (x < 0)
      {
         x = 0;
         dx = x - tmp;
         return true;
      }
      if (x > 1)
      {
         x = 1;
         dx = x - tmp;
         return true;
      }
      return false;
   }
};

template <> struct GeometryUtils<Geometry::SQUARE>
{
   static constexpr MFEM_HOST_DEVICE int Dimension() { return 2; }
   /// @b true if the given point (x,y) in ref space is inside the element
   static bool MFEM_HOST_DEVICE inside(real_t x, real_t y)
   {
      return (x >= 0) && (x <= 1) && (y >= 0) && (y <= 1);
   }

   /// @b Bound the reference coordinate @a (x,y) += (dx,dy) to be inside the
   /// square.
   /// @a dx and @a dy are updated to be (dx,dy) = project(x+dx,y+dy) - (x,y)
   /// @return true if (x,y) + (dx,dy) hit a boundary
   static bool MFEM_HOST_DEVICE project(real_t &x, real_t &y, real_t &dx,
                                        real_t &dy)
   {
      bool x_cond = GeometryUtils<Geometry::SEGMENT>::project(x, dx);
      bool y_cond = GeometryUtils<Geometry::SEGMENT>::project(y, dy);
      return x_cond || y_cond;
   }
};

template <> struct GeometryUtils<Geometry::CUBE>
{
   static constexpr MFEM_HOST_DEVICE int Dimension() { return 3; }
   /// @b true if the given point (x,y,z) in ref space is inside the element
   static bool MFEM_HOST_DEVICE inside(real_t x, real_t y, real_t z)
   {
      return (x >= 0) && (x <= 1) && (y >= 0) && (y <= 1) && (z >= 0) && (z <= 1);
   }

   /// @b Bound the reference coordinate @a (x,y,z) += (dx,dy,dz) to be inside
   /// the cube.
   /// @a dx, @a dy, and @ dz are updated to be
   /// (dx,dy,dz) = project(x+dx,y+dy,z+dz) - (x,y,z)
   /// @return true if (x,y,z) + (dx,dy,dz) hit a boundary
   static bool MFEM_HOST_DEVICE project(real_t &x, real_t &y, real_t &z,
                                        real_t &dx, real_t &dy, real_t &dz)
   {
      bool x_cond = GeometryUtils<Geometry::SEGMENT>::project(x, dx);
      bool y_cond = GeometryUtils<Geometry::SEGMENT>::project(y, dy);
      bool z_cond = GeometryUtils<Geometry::SEGMENT>::project(z, dz);
      return x_cond || y_cond || z_cond;
   }
};

/// 1D Lagrange basis from [0, 1]
class Lagrange
{
public:
   /// interpolant node locations, in reference space
   const real_t *z;

   /// number of points
   int pN;

   /// @b Evaluates the @a i'th Lagrange polynomial at @a x
   real_t MFEM_HOST_DEVICE eval(real_t x, int i) const
   {
      real_t u0 = 1;
      real_t den = 1;
      for (int j = 0; j < pN; ++j)
      {
         if (i != j)
         {
            real_t d_j = (x - z[j]);
            u0 = d_j * u0;
            den *= (z[i] - z[j]);
         }
      }
      den = 1 / den;
      return u0 * den;
   }

   /// @b Evaluates the @a i'th Lagrange polynomial and its first derivative at
   /// @a x
   void MFEM_HOST_DEVICE eval_d1(real_t &p, real_t &d1, real_t x, int i) const
   {
      real_t u0 = 1;
      real_t u1 = 0;
      real_t den = 1;
      for (int j = 0; j < pN; ++j)
      {
         if (i != j)
         {
            real_t d_j = (x - z[j]);
            u1 = d_j * u1 + u0;
            u0 = d_j * u0;
            den *= (z[i] - z[j]);
         }
      }
      den = 1 / den;
      p = u0 * den;
      d1 = u1 * den;
   }

   /// @b Evaluates the @a i'th Lagrange polynomial and its first and second
   /// derivatives at @a x
   void MFEM_HOST_DEVICE eval_d2(real_t &p, real_t &d1, real_t &d2, real_t x,
                                 int i) const
   {
      real_t u0 = 1;
      real_t u1 = 0;
      real_t u2 = 0;
      real_t den = 1;
      for (int j = 0; j < pN; ++j)
      {
         if (i != j)
         {
            real_t d_j = (x - z[j]);
            u2 = d_j * u2 + u1;
            u1 = d_j * u1 + u0;
            u0 = d_j * u0;
            den *= (z[i] - z[j]);
         }
      }
      den = 1 / den;
      p = den * u0;
      d1 = den * u1;
      d2 = 2 * den * u2;
   }
};

} // namespace eltrans
} // namespace mfem

#endif
