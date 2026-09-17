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

#include "../../bilininteg.hpp"
#include "mma.hpp"
#include "diffusion.hpp"
#include "form/register.hpp"

namespace mfem
{


namespace internal
{

namespace
{

/** Cofactor matrix of J (transpose of adjugate / used by diffusion PA). */
MFEM_HOST_DEVICE inline void CofactorsJ3(const real_t J11, const real_t J21,
                                         const real_t J31, const real_t J12,
                                         const real_t J22, const real_t J32,
                                         const real_t J13, const real_t J23,
                                         const real_t J33,
                                         real_t &A11, real_t &A12, real_t &A13,
                                         real_t &A21, real_t &A22, real_t &A23,
                                         real_t &A31, real_t &A32, real_t &A33)
{
   A11 = (J22 * J33) - (J23 * J32);
   A12 = (J32 * J13) - (J12 * J33);
   A13 = (J12 * J23) - (J22 * J13);
   A21 = (J31 * J23) - (J21 * J33);
   A22 = (J11 * J33) - (J13 * J31);
   A23 = (J21 * J13) - (J11 * J23);
   A31 = (J21 * J32) - (J31 * J22);
   A32 = (J31 * J12) - (J11 * J32);
   A33 = (J11 * J22) - (J12 * J21);
}

} // namespace

void PADiffusionSetupSimplexFromNodes(const int dim,
                                      const int coeffDim,
                                      const int NE,
                                      const int NQ,
                                      const int ND,
                                      const Array<real_t> &w,
                                      const Array<real_t> &g,
                                      const Vector &nodes_e,
                                      const Vector &c,
                                      Vector &d)
{
   const bool symmetric = (coeffDim != dim * dim);
   const bool const_c = c.Size() == coeffDim;
   const int pa_size = symmetric ? (dim * (dim + 1)) / 2 : dim * dim;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   auto D = Reshape(d.Write(), NQ, pa_size, NE);

   if (dim == 2)
   {
      const auto C = const_c
                     ? Reshape(c.Read(), coeffDim, 1, 1)
                     : Reshape(c.Read(), coeffDim, NQ, NE);
      const auto get_coeff =
         [const_c] MFEM_HOST_DEVICE
         (const decltype(C) &C, int i, int q, int e)
      {
         return const_c ? C(i, 0, 0) : C(i, q, e);
      };

      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ, q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t w_detJ = W(q) / (J11 * J22 - J21 * J12);
         if (coeffDim == 3 || coeffDim == 4)
         {
            const real_t M11 = get_coeff(C, 0, q, e);
            const real_t M12 = get_coeff(C, 1, q, e);
            const real_t M21 = symmetric ? M12 : get_coeff(C, 2, q, e);
            const real_t M22 = symmetric
                               ? get_coeff(C, 2, q, e)
                               : get_coeff(C, 3, q, e);
            const real_t R11 = M11 * J22 - M12 * J12;
            const real_t R21 = M21 * J22 - M22 * J12;
            const real_t R12 = -M11 * J21 + M12 * J11;
            const real_t R22 = -M21 * J21 + M22 * J11;
            D(q, 0, e) = w_detJ * (J22 * R11 - J12 * R21);
            D(q, 1, e) = w_detJ * (-J21 * R11 + J11 * R21);
            D(q, 2, e) = w_detJ * (symmetric
                                   ? (-J21 * R12 + J11 * R22)
                                   : (J22 * R12 - J12 * R22));
            if (!symmetric)
            {
               D(q, 3, e) = w_detJ * (-J21 * R12 + J11 * R22);
            }
         }
         else
         {
            const real_t C1 = get_coeff(C, 0, q, e);
            const real_t C2 = get_coeff(C, coeffDim == 2 ? 1 : 0, q, e);
            D(q, 0, e) = w_detJ * (C2 * J12 * J12 + C1 * J22 * J22);
            D(q, 1, e) = -w_detJ * (C2 * J12 * J11 + C1 * J22 * J21);
            D(q, 2, e) = w_detJ * (C2 * J11 * J11 + C1 * J21 * J21);
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3,
               "PADiffusionSetupSimplexFromNodes only supports dim 2 or 3");
   const auto C = const_c
                  ? Reshape(c.Read(), coeffDim, 1, 1)
                  : Reshape(c.Read(), coeffDim, NQ, NE);
   const auto get_coeff =
      [const_c] MFEM_HOST_DEVICE
      (const decltype(C) &C, int i, int q, int e)
   {
      return const_c ? C(i, 0, 0) : C(i, q, e);
   };

   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ, q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      EvalSimplexJ3(E, G, q, e, ND,
                    J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t w_detJ = W(q) / (J11 * (J22 * J33 - J32 * J23) -
                                    J21 * (J12 * J33 - J32 * J13) +
                                    J31 * (J12 * J23 - J22 * J13));
      real_t A11, A12, A13, A21, A22, A23, A31, A32, A33;
      CofactorsJ3(J11, J21, J31, J12, J22, J32, J13, J23, J33,
                  A11, A12, A13, A21, A22, A23, A31, A32, A33);

      if (coeffDim == 6 || coeffDim == 9)
      {
         const real_t M11 = get_coeff(C, 0, q, e);
         const real_t M12 = get_coeff(C, 1, q, e);
         const real_t M13 = get_coeff(C, 2, q, e);
         const real_t M21 = (!symmetric) ? get_coeff(C, 3, q, e) : M12;
         const real_t M22 = (!symmetric) ? get_coeff(C, 4, q, e)
                            : get_coeff(C, 3, q, e);
         const real_t M23 = (!symmetric) ? get_coeff(C, 5, q, e)
                            : get_coeff(C, 4, q, e);
         const real_t M31 = (!symmetric) ? get_coeff(C, 6, q, e) : M13;
         const real_t M32 = (!symmetric) ? get_coeff(C, 7, q, e) : M23;
         const real_t M33 = (!symmetric) ? get_coeff(C, 8, q, e)
                            : get_coeff(C, 5, q, e);

         const real_t R11 = M11 * A11 + M12 * A12 + M13 * A13;
         const real_t R12 = M11 * A21 + M12 * A22 + M13 * A23;
         const real_t R13 = M11 * A31 + M12 * A32 + M13 * A33;
         const real_t R21 = M21 * A11 + M22 * A12 + M23 * A13;
         const real_t R22 = M21 * A21 + M22 * A22 + M23 * A23;
         const real_t R23 = M21 * A31 + M22 * A32 + M23 * A33;
         const real_t R31 = M31 * A11 + M32 * A12 + M33 * A13;
         const real_t R32 = M31 * A21 + M32 * A22 + M33 * A23;
         const real_t R33 = M31 * A31 + M32 * A32 + M33 * A33;

         D(q, 0, e) = w_detJ * (A11 * R11 + A12 * R21 + A13 * R31);
         const real_t D12 = w_detJ * (A11 * R12 + A12 * R22 + A13 * R32);
         D(q, 1, e) = D12;
         D(q, 2, e) = w_detJ * (A11 * R13 + A12 * R23 + A13 * R33);
         const real_t D22 = w_detJ * (A21 * R12 + A22 * R22 + A23 * R32);
         const real_t D23 = w_detJ * (A21 * R13 + A22 * R23 + A23 * R33);
         const real_t D33 = w_detJ * (A31 * R13 + A32 * R23 + A33 * R33);
         D(q, 4, e) = symmetric ? D23 : D22;
         D(q, 5, e) = symmetric ? D33 : D23;
         if (symmetric) { D(q, 3, e) = D22; }
         else
         {
            D(q, 3, e) = w_detJ * (A21 * R11 + A22 * R21 + A23 * R31);
            D(q, 6, e) = w_detJ * (A31 * R11 + A32 * R21 + A33 * R31);
            D(q, 7, e) = w_detJ * (A31 * R12 + A32 * R22 + A33 * R32);
            D(q, 8, e) = D33;
         }
      }
      else
      {
         const real_t C1 = get_coeff(C, 0, q, e);
         const real_t C2 = get_coeff(C, coeffDim == 3 ? 1 : 0, q, e);
         const real_t C3 = get_coeff(C, coeffDim == 3 ? 2 : 0, q, e);
         D(q, 0, e) = w_detJ * (C1 * A11 * A11 + C2 * A12 * A12 + C3 * A13 * A13);
         D(q, 1, e) = w_detJ * (C1 * A11 * A21 + C2 * A12 * A22 + C3 * A13 * A23);
         D(q, 2, e) = w_detJ * (C1 * A11 * A31 + C2 * A12 * A32 + C3 * A13 * A33);
         D(q, 3, e) = w_detJ * (C1 * A21 * A21 + C2 * A22 * A22 + C3 * A23 * A23);
         D(q, 4, e) = w_detJ * (C1 * A21 * A31 + C2 * A22 * A32 + C3 * A23 * A33);
         D(q, 5, e) = w_detJ * (C1 * A31 * A31 + C2 * A32 * A32 + C3 * A33 * A33);
      }
   });
}

} // namespace internal

void DiffusionIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   fespace = &fes;
   maps = nullptr;
   geom = nullptr;

   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(mesh->SpaceDimension() == dim, "");

   const FiniteElement &el = *fes.GetTypicalFE();
   const Geometry::Type geom_t =
      (dim == 2) ? Geometry::TRIANGLE : Geometry::TETRAHEDRON;
   MFEM_VERIFY(el.GetGeomType() == geom_t, "");
   MFEM_VERIFY(IsSimplexMmaH1Element(el, dim), "");

   const int p = el.GetOrder();
   const int dims = el.GetDim();
   const int symmDims = (dims * (dims + 1)) / 2;

   const IntegrationRule &ir = IntRule ? *IntRule : GetRule(el, el);
   nq = ir.GetNPoints();
   const int dof = el.GetDof();

   dofs1D = p + 1;
   quad1D = 0;
   ne = mesh->GetNE();
   use_simplices_mma = true;

   simplex_mma_G.SetSize(nq * dof * dim, mt);
   {
      real_t *Gh = simplex_mma_G.HostWrite();
      dshape.SetSize(dof, dim);
      for (int q = 0; q < nq; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el.CalcDShape(ip, dshape);
         for (int d = 0; d < dim; d++)
         {
            for (int i = 0; i < dof; i++)
            {
               Gh[q + nq * (i + dof * d)] = dshape(i, d);
            }
         }
      }
   }

   // Assemble geometry directly from restricted mesh nodes
   geom = nullptr;
   Vector nodes_e;
   int nd_n = 0, sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, sdim);
   MFEM_VERIFY(sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq, "");

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::COMPRESSED);
   if (MQ) { coeff.ProjectTranspose(*MQ); }
   else if (VQ) { coeff.Project(*VQ); }
   else if (Q) { coeff.Project(*Q); }
   else { coeff.SetConstant(1.0); }

   const int coeff_dim = coeff.GetVDim();
   symmetric = (coeff_dim != dims * dims);
   const int pa_size = symmetric ? symmDims : dims * dims;
   pa_data.SetSize(pa_size * nq * ne, mt);
   internal::PADiffusionSetupSimplexFromNodes(dim, coeff_dim, ne, nq, nd_n,
                                              ir.GetWeights(), nmaps.G, nodes_e,
                                              coeff, pa_data);
}

void DiffusionIntegrator::RegisterSimplexMmaKernels()
{
   // Shared with VectorDiffusion — see form/register.hpp. Unregistered → Fallback.
   internal::mma::RegisterDiffusionSimplexMmaSpecializations<DiffusionIntegrator>();
}

void DiffusionIntegrator::RegisterTensorsMmaKernels()
{
   // Shared tensor list (p = 3..7) — see form/register.hpp.
   internal::mma::RegisterTensorsMmaSpecializations<DiffusionIntegrator>();
}

DiffusionIntegrator::ApplyTensorsMmaKernelType
DiffusionIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaDiffusionApplyTensors2D; }
   if (dim == 3) { return internal::MmaDiffusionApplyTensors3D; }
   MFEM_ABORT("Tensors MMA diffusion PA is only implemented for dim 2 or 3");
   return nullptr;
}


} // namespace mfem
