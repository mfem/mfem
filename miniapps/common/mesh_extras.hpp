// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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


} // namespace common

} // namespace mfem

#endif
