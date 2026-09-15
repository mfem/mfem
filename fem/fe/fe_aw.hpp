// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_AW
#define MFEM_FE_AW

#include "fe_cubic_symmetric.hpp"

namespace mfem
{

/** Lowest-order conforming Arnold--Winther triangle:
    symmetric cubic tensors whose divergence is linear. Its 24 DOFs are
    Cartesian tensor components (xx,xy,yy) at each vertex, four traction
    moments per edge (constant nn, nt, then linear nn, nt), and three cell
    moments of the double-Piola pullback. Physical evaluation and transfer
    require affine 2D transformations. */
class ArnoldWintherTriangleFiniteElement : public
   CubicSymmetricTriangleFiniteElement
{
public:
   ArnoldWintherTriangleFiniteElement();
};

} // namespace mfem

#endif
