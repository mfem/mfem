// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#ifndef MFEM_FE_HZ
#define MFEM_FE_HZ

#include "fe_cubic_symmetric.hpp"

namespace mfem
{

/** Lowest-order (cubic) Hu--Zhang triangle, with the full 30-dimensional
    symmetric P3 stress space and P2 vector divergence. The DOFs are Cartesian
    components (xx,xy,yy) at each vertex, four traction moments per edge
    (constant nn, nt, then linear nn, nt), and nine cell moments of the
    double-Piola pullback against (1,x,y) times each symmetric component.

    Vertex tensors and edge tractions are globally continuous. The nine cell
    DOFs are local, including the freedom to jump in tangential-tangential
    stress across edges. Physical evaluation and transfer require affine 2D
    transformations. The Airy image of Argyris is contained in this space.

    See Hu and Zhang, arXiv:1406.7457, for the original triangular family. */
class HuZhangTriangleFiniteElement : public CubicSymmetricTriangleFiniteElement
{
public:
   HuZhangTriangleFiniteElement();
};

} // namespace mfem

#endif
