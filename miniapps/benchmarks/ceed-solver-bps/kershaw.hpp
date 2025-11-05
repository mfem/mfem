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

#ifndef __KERSHAW_HPP__
#define __KERSHAW_HPP__

#include "mfem.hpp"

namespace mfem
{

// 1D transformation at the right boundary.
real_t right(const real_t eps, const real_t x)
{
   return (x <= 0.5) ? (2-eps) * x : 1 + eps*(x-1);
}

// 1D transformation at the left boundary
real_t left(const real_t eps, const real_t x)
{
   return 1-right(eps,1-x);
}

// Transition from a value of "a" for x=0, to a value of "b" for x=1. Smoothness
// is controlled by the parameter "s", taking values 0, 1, or 2.
real_t step(const real_t a, const real_t b, real_t x, int s)
{
   if (x <= 0) { return a; }
   if (x >= 1) { return b; }
   switch (s)
   {
      case 0:
      default:
         return a + (b-a) * (x);
      case 1: return a + (b-a) * (x*x*(3-2*x));
      case 2: return a + (b-a) * (x*x*x*(x*(6*x-15)+10));
   }
}

// 3D version of a generalized Kershaw mesh transformation, see D. Kershaw,
// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
// JCP, 39:375–395, 1981.
//
// The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
// ny, nz divisible by 2.
//
// The eps parameters are in (0, 1]. Uniform mesh is recovered for epsy=epsz=1.
void kershaw(const real_t epsy, const real_t epsz, const int smoothness,
             const real_t x, const real_t y, const real_t z,
             real_t &X, real_t &Y, real_t &Z)
{
   X = x;

   int layer = x*6.0;
   real_t lambda = (x-layer/6.0)*6;

   // The x-range is split in 6 layers going from left-to-left, left-to-right,
   // right-to-left (2 layers), left-to-right and right-to-right yz-faces.
   switch (layer)
   {
      case 0:
         Y = left(epsy, y);
         Z = left(epsz, z);
         break;
      case 1:
      case 4:
         Y = step(left(epsy, y), right(epsy, y), lambda, smoothness);
         Z = step(left(epsz, z), right(epsz, z), lambda, smoothness);
         break;
      case 2:
         Y = step(right(epsy, y), left(epsy, y), lambda/2, smoothness);
         Z = step(right(epsz, z), left(epsz, z), lambda/2, smoothness);
         break;
      case 3:
         Y = step(right(epsy, y), left(epsy, y), (1+lambda)/2, smoothness);
         Z = step(right(epsz, z), left(epsz, z), (1+lambda)/2, smoothness);
         break;
      default:
         Y = right(epsy, y);
         Z = right(epsz, z);
         break;
   }
}

struct KershawTransformation : VectorCoefficient
{
   real_t epsy, epsz;
   int dim, s;
   KershawTransformation(int dim_, real_t epsy_, real_t epsz_, int s_=0)
      : VectorCoefficient(dim_), epsy(epsy_), epsz(epsz_), dim(dim_), s(s_) { }
   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      real_t xyz[3];
      Vector transip(xyz, 3);
      T.Transform(ip, transip);
      if (dim == 1)
      {
         V[0] = xyz[0]; // no transformation in 1D
      }
      else if (dim == 2)
      {
         real_t z=0, zt;
         kershaw(epsy, epsz, s, xyz[0], xyz[1], z, V[0], V[1], zt);
      }
      else // dim == 3
      {
         kershaw(epsy, epsz, s, xyz[0], xyz[1], xyz[2], V[0], V[1], V[2]);
      }
   }
};

ParMesh CreateKershawMesh(int nx, int ny, int nz, real_t epsy, real_t epsz)
{
   const bool sfc_order = true;
   Mesh serial_mesh;
   if (nx > 0 && ny == 0 && nz == 0)
   {
      serial_mesh = Mesh::MakeCartesian1D(nx, 1.0);
   }
   else if (nx > 0 && ny > 0 && nz == 0)
   {
      serial_mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL,
                                          false, 1, 1, sfc_order);
   }
   else if (nx > 0 && ny > 0 && nz > 0)
   {
      serial_mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                          1, 1, 1, sfc_order);
   }
   else
   {
      MFEM_ABORT("Bad grid size");
   }
   KershawTransformation kt(serial_mesh.Dimension(), epsy, epsz);
   serial_mesh.Transform(kt);
   return ParMesh(MPI_COMM_WORLD, serial_mesh);
}

ParMesh CreateKershawMesh(int n, real_t eps)
{
   return CreateKershawMesh(n, n, n, eps, eps);
}

}

#endif
