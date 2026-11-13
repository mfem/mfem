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

#include "vector.hpp"

namespace mfem
{

namespace internal::mma
{

void SimplexVectorBasis(const FiniteElementSpace &fes,
                        const IntegrationRule &ir,
                        const ElementDofOrdering ordering,
                        const bool with_value, const bool with_grad,
                        Array<real_t> &P)
{
   const FiniteElement &el = *fes.GetTypicalFE();
   const int dim = el.GetDim();
   const int nd = el.GetDof();
   const int nq = ir.GetNPoints();
   const int np = (with_value ? 1 : 0) + (with_grad ? dim : 0);
   MFEM_VERIFY(np > 0, "Empty simplex basis request");

   const DofToQuad &maps = el.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(maps.ndof == nd && maps.nqpt == nq, "");

   // E-vector dof j holds native dof map[j], as in ElementRestriction.
   const int *map = nullptr;
   if (ordering == ElementDofOrdering::LEXICOGRAPHIC && !fes.IsDGSpace())
   {
      const auto *nodal = dynamic_cast<const NodalFiniteElement *>(&el);
      const auto *tri = dynamic_cast<const H1Pos_TriangleElement *>(&el);
      const auto *tet = dynamic_cast<const H1Pos_TetrahedronElement *>(&el);
      MFEM_VERIFY(nodal || tri || tet, "No lexicographic simplex dof map");
      const Array<int> &dof_map = nodal ? nodal->GetLexicographicOrdering() :
                                  tri ? tri->GetDofMap() : tet->GetDofMap();
      MFEM_VERIFY(dof_map.Size() == nd, "Invalid simplex dof map");
      map = dof_map.HostRead();
   }

   const real_t *B = maps.B.HostRead();
   const real_t *G = maps.G.HostRead();
   P.SetSize(nq * np * nd);
   real_t *Ph = P.HostWrite();
   for (int j = 0; j < nd; ++j)
   {
      const int i = map ? map[j] : j;
      for (int q = 0; q < nq; ++q)
      {
         int p = 0;
         if (with_value) { Ph[q + nq * (p++ + np * j)] = B[q + nq * i]; }
         if (with_grad)
         {
            for (int d = 0; d < dim; ++d)
            {
               Ph[q + nq * (p + d + np * j)] = G[q + nq * (d + dim * i)];
            }
         }
      }
   }
}

void SimplexAdjSetup(Mesh &mesh, const IntegrationRule &ir,
                     const Vector &c, const bool inv_det, Vector &d)
{
   const int dim = mesh.Dimension();
   const int NE = mesh.GetNE();
   const int NQ = ir.GetNPoints();
   const MemoryType mt = Device::GetDeviceMemoryType();
   d.SetSize(NQ * dim * dim * NE, mt);
   if (NE == 0) { return; }

   Vector nodes_e;
   const Array<real_t> *Gn = nullptr;
   int ND = 0;
   GetSimplexSetupGeom(mesh, ir, mt, nodes_e, Gn, ND);

   const bool const_c = c.Size() == 1;
   MFEM_VERIFY(const_c || c.Size() == NQ * NE, "Invalid coefficient size");
   const auto W = Reshape(ir.GetWeights().Read(), NQ);
   const auto G = Reshape(Gn->Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = Reshape(c.Read(), const_c ? 1 : NQ, const_c ? 1 : NE);
   auto D = Reshape(d.Write(), NQ, dim * dim, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int q = idx % NQ, e = idx / NQ;
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t det = J11 * J22 - J21 * J12;
         real_t s = W(q) * (const_c ? C(0, 0) : C(q, e));
         if (inv_det) { s /= det; }
         D(q, 0, e) = s * J22;
         D(q, 1, e) = -s * J21;
         D(q, 2, e) = -s * J12;
         D(q, 3, e) = s * J11;
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "Simplex MMA needs dim 2 or 3");
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int q = idx % NQ, e = idx / NQ;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND,
                    J11, J21, J31, J12, J22, J32, J13, J23, J33);
      real_t C11, C12, C13, C21, C22, C23, C31, C32, C33;
      CofactorsJ3(J11, J21, J31, J12, J22, J32, J13, J23, J33,
                  C11, C12, C13, C21, C22, C23, C31, C32, C33);
      const real_t det = J11 * C11 + J12 * C21 + J13 * C31;
      real_t s = W(q) * (const_c ? C(0, 0) : C(q, e));
      if (inv_det) { s /= det; }
      D(q, 0, e) = s * C11;
      D(q, 1, e) = s * C21;
      D(q, 2, e) = s * C31;
      D(q, 3, e) = s * C12;
      D(q, 4, e) = s * C22;
      D(q, 5, e) = s * C32;
      D(q, 6, e) = s * C13;
      D(q, 7, e) = s * C23;
      D(q, 8, e) = s * C33;
   });
}

} // namespace internal::mma

} // namespace mfem
