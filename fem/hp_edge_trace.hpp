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

#ifndef MFEM_HP_EDGE_TRACE
#define MFEM_HP_EDGE_TRACE

#include "fespace.hpp"
#include "eltrans.hpp"
#include "../mesh/mesh.hpp"

#include <algorithm>
#include <map>
#include <vector>

namespace mfem
{
namespace internal
{
inline bool SupportsInternalEdgeConstraints(const FiniteElementSpace &fes)
{
   return fes.GetMesh()->Dimension() == 3 &&
          (dynamic_cast<const H1_FECollection *>(fes.FEColl()) ||
           dynamic_cast<const ND_FECollection *>(fes.FEColl()));
}

using HpConstraintRow = std::map<int, real_t>;

inline void AddSignedHpEntry(HpConstraintRow &row, int dof, real_t value)
{
   if (dof < 0) { dof = -1-dof; value = -value; }
   row[dof] += value;
}

// Low-order interpolation alone does not impose that a high-order face trace
// belongs to the lower-order edge space. Retain its full edge trace equation;
// the existing low-to-high edge dependencies supply the other representation.
inline void AddInternalEdgeCompatibility(
   const FiniteElementSpace &fes, const Array<int> &master_dofs,
   const FiniteElement &master_fe, int master_order, int slave_face,
   const DenseMatrix &pm, std::vector<HpConstraintRow> &constraints)
{
   Array<int> vertices, edges, orientations, dofs;
   fes.GetMesh()->GetFaceVertices(slave_face, vertices);
   fes.GetMesh()->GetFaceEdges(slave_face, edges, orientations);
   MFEM_ASSERT(vertices.Size() == 4 && edges.Size() == 4, "");
   IsoparametricTransformation T;
   T.SetFE(&SegmentFE);
   T.GetPointMat().SetSize(2, 2);
   DenseMatrix transfer;
   for (int i = 0; i < edges.Size(); i++)
   {
      int a = i, b = (i+1) % edges.Size();
      bool boundary = false;
      for (int d = 0; d < 2; d++)
      {
         boundary |= (pm(d, a) == 0.0 && pm(d, b) == 0.0) ||
                     (pm(d, a) == 1.0 && pm(d, b) == 1.0);
      }
      if (boundary || fes.GetEdgeDofs(edges[i], dofs, 0) >= master_order)
      {
         continue;
      }
      if (vertices[a] > vertices[b]) { std::swap(a, b); }
      for (int d = 0; d < 2; d++)
      {
         T.GetPointMat()(d, 0) = pm(d, a);
         T.GetPointMat()(d, 1) = pm(d, b);
      }
      int order = -1;
      for (int variant = 1; ; variant++)
      {
         order = fes.GetEdgeDofs(edges[i], dofs, variant);
         if (order < 0 || order == master_order) { break; }
      }
      MFEM_VERIFY(order == master_order,
                  "Missing internal-edge variant for master face order");
      fes.FEColl()->GetFE(Geometry::SEGMENT, order)->GetTransferMatrix(
         master_fe, T, transfer);
      for (int r = 0; r < dofs.Size(); r++)
      {
         HpConstraintRow row;
         AddSignedHpEntry(row, dofs[r], 1.0);
         for (int c = 0; c < master_dofs.Size(); c++)
         {
            AddSignedHpEntry(row, master_dofs[c], -transfer(r, c));
         }
         constraints.push_back(std::move(row));
      }
   }
}

} // namespace internal
} // namespace mfem

#endif
