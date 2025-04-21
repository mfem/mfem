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

#ifndef MFEM_MESH_EXTRAS
#define MFEM_MESH_EXTRAS

#include "mfem.hpp"
#include <sstream>

namespace mfem
{

namespace common
{

class ElementMeshStream : public std::stringstream
{
public:
   ElementMeshStream(Element::Type e);
};

/// Merges vertices which lie at the same location
void MergeMeshNodes(Mesh * mesh, int logging);

/// Convert a set of attribute numbers to a marker array
/** The marker array will be of size max_attr and it will contain only zeroes
    and ones. Ones indicate which attribute numbers are present in the attrs
    array. In the special case when attrs has a single entry equal to -1 the
    marker array will contain all ones. */
void AttrToMarker(int max_attr, const Array<int> &attrs, Array<int> &marker);

/// Transform a mesh according to an arbitrary affine transformation
///    y = A x + b
/// Where A is a spaceDim x spaceDim matrix and b is a vector of size spaceDim.
/// If A is of size zero the transformation will be y = b.
/// If b is of size zero the transformation will be y = A x.
///
/// Note that no error checking related to the determinant of A is performed.
/// If A has a non-positive determinant it is likely to produce an invalid
/// transformed mesh.
class AffineTransformation : public VectorCoefficient
{
private:
   DenseMatrix A;
   Vector b;
   Vector x;

public:
   AffineTransformation(int dim_, const DenseMatrix &A_, const Vector & b_)
      : VectorCoefficient(dim_), A(A_), b(b_), x(dim_)
   {
      MFEM_VERIFY((A.Height() == dim_ && A.Width() == dim_) ||
                  (A.Height() == 0 && A.Width() == 0),
                  "Affine transformation given an invalid matrix");
      MFEM_VERIFY(b.Size() == dim_ || b.Size() == 0,
                  "Affine transformation given an invalid vector");
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   using VectorCoefficient::Eval;
};

/// Generalized Kershaw mesh transformation in 2D and 3D, see D. Kershaw,
/// "Differencing of the diffusion equation in Lagrangian hydrodynamic codes",
/// JCP, 39:375–395, 1981.
/** The input mesh should be Cartesian nx x ny x nz with nx divisible by 6 and
    ny, nz divisible by 2.
    The parameters @a epsy and @a epsz must be in (0, 1].
    Uniform mesh is recovered for epsy=epsz=1.
    The @a smooth parameter controls the transition between different layers. */
// Usage:
// common::KershawTransformation kershawT(pmesh->Dimension(), 0.3, 0.3, 2);
// pmesh->Transform(kershawT);
class KershawTransformation : public VectorCoefficient
{
private:
   int dim;
   real_t epsy, epsz;
   int smooth;

public:
   KershawTransformation(const int dim_, real_t epsy_ = 0.3,
                         real_t epsz_ = 0.3, int smooth_ = 1)
      : VectorCoefficient(dim_), dim(dim_), epsy(epsy_),
        epsz(epsz_), smooth(smooth_)
   {
      MFEM_VERIFY(dim > 1,"Kershaw transformation only works for 2D and 3D"
                  "meshes.");
      MFEM_VERIFY(smooth >= 1 && smooth <= 3,
                  "Kershaw parameter smooth must be in [1, 3]");
      MFEM_VERIFY(epsy > 0 && epsy <=1,
                  "Kershaw parameter epsy must be in (0, 1].");
      if (dim == 3)
      {
         MFEM_VERIFY(epsz > 0 && epsz <=1,
                     "Kershaw parameter epsz must be in (0, 1].");
      }
   }

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

   // Transition from a value of "a" for x=0, to a value of "b" for x=1.
   // Controlled through "smooth" parameter.
   real_t step(const real_t a, const real_t b, real_t x)
   {
      if (x <= 0) { return a; }
      if (x >= 1) { return b; }
      if (smooth == 1) { return a + (b-a) * (x); }
      else if (smooth == 2) { return a + (b-a) * (x*x*(3-2*x)); }
      else { return a + (b-a) * (x*x*x*(x*(6*x-15)+10)); }
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   using VectorCoefficient::Eval;
};

/// Transform a [0,1]^D mesh into a spiral. The parameters are:
/// @a turns - number of turns around the origin,
/// @a width - for D >= 2, the width of the spiral arm,
/// @ gap    - gap between adjacent spiral arms at the end of each turn,
/// @ height - for D = 3, the maximum height of the spiral.
// Usage:
// common::SpiralTransformation spiralT(spaceDim, 2.4, 0.1, 0.05, 1.0);
// pmesh->Transform(spiralT);
class SpiralTransformation : public VectorCoefficient
{
private:
   real_t dim, turns, width, gap, height;

public:
   SpiralTransformation(int dim_, real_t turns_ = 1.0, real_t width_ = 0.1,
                        real_t gap_ = 0.05, real_t height_ = 1.0)
      : VectorCoefficient(dim_), dim(dim_),
        turns(turns_), width(width_), gap(gap_), height(height_)
   {
      MFEM_VERIFY(turns > 0 && width > 0 && gap > 0 && height > 0,
                  "Spiral transformation requires positive parameters: turns, "
                  " width, gap, and height.");
   }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override;

   using VectorCoefficient::Eval;
};


} // namespace common

} // namespace mfem

#endif
