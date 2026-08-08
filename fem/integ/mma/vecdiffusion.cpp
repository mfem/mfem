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

namespace
{

/** Pack VectorDiffusion PA like stock: (NQ, dim*dim, ncomp, NE) with
    ncomp = vdim (Q/VQ) or vdim*dim (MQ). Full metric slots; 3D fills 0..5. */
void PAVectorDiffusionSetupSimplex(const int dim,
                                   const int NE,
                                   const int NQ,
                                   const int ND,
                                   const int vdim,
                                   const int coeff_vdim,
                                   const Array<real_t> &w,
                                   const Array<real_t> &g,
                                   const Vector &nodes_e,
                                   const Vector &c,
                                   Vector &d)
{
   const bool vector_coeff = coeff_vdim == vdim;
   const bool matrix_coeff = coeff_vdim == vdim * vdim;
   MFEM_VERIFY(coeff_vdim == 1 || vector_coeff || matrix_coeff, "");
   const int pa_size = dim * dim;
   const int ncomp = vdim * (matrix_coeff ? dim : 1);
   const bool const_c = c.Size() == coeff_vdim;

   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), coeff_vdim, 1, 1)
                  : Reshape(c.Read(), coeff_vdim, NQ, NE);
   auto DE = Reshape(d.Write(), NQ, pa_size, ncomp, NE);

   if (dim == 2)
   {
      const int map[4] = {0, 2, 1, 3};
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ;
         const int q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         internal::EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t w_detJ = W(q) / ((J11 * J22) - (J21 * J12));
         const real_t D0 =  w_detJ * (J12 * J12 + J22 * J22);
         const real_t D1 = -w_detJ * (J12 * J11 + J22 * J21);
         const real_t D2 =  w_detJ * (J11 * J11 + J21 * J21);
         for (int i = 0; i < (matrix_coeff ? coeff_vdim : vdim); ++i)
         {
            const int k = matrix_coeff ? map[i] : (vector_coeff ? i : 0);
            const real_t Cc = const_c ? C(k, 0, 0) : C(k, q, e);
            DE(q, 0, i, e) = D0 * Cc;
            DE(q, 1, i, e) = D1 * Cc;
            DE(q, 2, i, e) = D1 * Cc;
            DE(q, 3, i, e) = D2 * Cc;
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "");
   const int map[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
   mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
   {
      const int e = idx / NQ;
      const int q = idx - NQ * e;
      real_t J11, J21, J31, J12, J22, J32, J13, J23, J33;
      internal::EvalSimplexJ3(E, G, q, e, ND,
                              J11, J21, J31, J12, J22, J32, J13, J23, J33);
      const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
                          J21 * (J12 * J33 - J32 * J13) +
                          J31 * (J12 * J23 - J22 * J13);
      const real_t c_detJ = W(q) / detJ;
      const real_t A11 = (J22 * J33) - (J23 * J32);
      const real_t A12 = (J32 * J13) - (J12 * J33);
      const real_t A13 = (J12 * J23) - (J22 * J13);
      const real_t A21 = (J31 * J23) - (J21 * J33);
      const real_t A22 = (J11 * J33) - (J13 * J31);
      const real_t A23 = (J21 * J13) - (J11 * J23);
      const real_t A31 = (J21 * J32) - (J31 * J22);
      const real_t A32 = (J31 * J12) - (J11 * J32);
      const real_t A33 = (J11 * J22) - (J12 * J21);
      const real_t D11 = c_detJ * (A11*A11 + A12*A12 + A13*A13);
      const real_t D21 = c_detJ * (A11*A21 + A12*A22 + A13*A23);
      const real_t D31 = c_detJ * (A11*A31 + A12*A32 + A13*A33);
      const real_t D22 = c_detJ * (A21*A21 + A22*A22 + A23*A23);
      const real_t D32 = c_detJ * (A21*A31 + A22*A32 + A23*A33);
      const real_t D33 = c_detJ * (A31*A31 + A32*A32 + A33*A33);

      for (int i = 0; i < (matrix_coeff ? coeff_vdim : vdim); ++i)
      {
         const int k = matrix_coeff ? map[i] : (vector_coeff ? i : 0);
         const real_t Ck = const_c ? C(k, 0, 0) : C(k, q, e);
         DE(q, 0, i, e) = D11 * Ck;
         DE(q, 1, i, e) = D21 * Ck;
         DE(q, 2, i, e) = D31 * Ck;
         DE(q, 3, i, e) = D22 * Ck;
         DE(q, 4, i, e) = D32 * Ck;
         DE(q, 5, i, e) = D33 * Ck;
         DE(q, 6, i, e) = 0.0;
         DE(q, 7, i, e) = 0.0;
         DE(q, 8, i, e) = 0.0;
      }
   });
}

} // namespace

void VectorDiffusionIntegrator::AssembleSimplexMmaPA(
   const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   maps = nullptr;
   geom = nullptr;

   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   sdim = mesh->SpaceDimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(sdim == dim, "");

   const FiniteElement &el = *fes.GetTypicalFE();
   const Geometry::Type geom_t =
      (dim == 2) ? Geometry::TRIANGLE : Geometry::TETRAHEDRON;
   MFEM_VERIFY(el.GetGeomType() == geom_t, "");
   MFEM_VERIFY(IsSimplexMmaH1Element(el, dim), "");

   vdim = (vdim == -1) ? dim : vdim;
   MFEM_VERIFY(vdim == fes.GetVDim(), "vdim != fes.GetVDim()");

   const int p = el.GetOrder();

   const IntegrationRule &ir = IntRule ? *IntRule
                               : DiffusionIntegrator::GetRule(el, el);
   nq = ir.GetNPoints();
   const int dof = el.GetDof();

   dofs1D = p + 1;
   quad1D = 0;
   ne = mesh->GetNE();
   use_simplices_mma = true;
   use_tensors_mma = false;

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

   Vector nodes_e;
   int nd_n = 0, nodes_sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, nodes_sdim);
   MFEM_VERIFY(nodes_sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq, "");

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs, CoefficientStorage::FULL);
   if (Q)
   {
      coeff.Project(*Q);
   }
   else if (VQ)
   {
      coeff.Project(*VQ);
      MFEM_VERIFY(VQ->GetVDim() == vdim, "VQ vdim vs. vdim error");
   }
   else if (MQ)
   {
      coeff.ProjectTranspose(*MQ);
      MFEM_VERIFY(MQ->GetVDim() == vdim, "MQ dimension vs. vdim error");
      MFEM_VERIFY(coeff.Size() == (vdim * vdim) * ne * nq, "MQ size error");
   }
   else { coeff.SetConstant(1.0); }

   coeff_vdim = coeff.GetVDim();
   const bool scalar_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == vdim;
   const bool matrix_coeff = coeff_vdim == vdim * vdim;
   MFEM_VERIFY(scalar_coeff + vector_coeff + matrix_coeff == 1, "");

   const int pa_size = dim * dim;
   pa_data.SetSize(nq * pa_size * vdim * (matrix_coeff ? dim : 1) * ne, mt);
   PAVectorDiffusionSetupSimplex(dim, ne, nq, nd_n, vdim, coeff_vdim,
                                 ir.GetWeights(), nmaps.G, nodes_e, coeff,
                                 pa_data);
}

void VectorDiffusionIntegrator::RegisterSimplexMmaKernels()
{
   // Shared with Diffusion — see form/register.hpp.
   internal::mma::RegisterDiffusionSimplexMmaSpecializations<
   VectorDiffusionIntegrator>();
}

VectorDiffusionIntegrator::ApplySimplexMmaKernelType
VectorDiffusionIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaVectorDiffusionApplySimplex2D; }
   if (dim == 3) { return internal::MmaVectorDiffusionApplySimplex3D; }
   MFEM_ABORT("Simplex MMA VectorDiffusion PA is only implemented for dim 2/3");
   return nullptr;
}



void VectorDiffusionIntegrator::RegisterTensorsMmaKernels()
{
   // Shared tensor list (p = 3..7) — see form/register.hpp.
   internal::mma::RegisterTensorsMmaSpecializations<VectorDiffusionIntegrator>();
}

VectorDiffusionIntegrator::ApplyTensorsMmaKernelType
VectorDiffusionIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaVectorDiffusionApplyTensors2D; }
   if (dim == 3) { return internal::MmaVectorDiffusionApplyTensors3D; }
   MFEM_ABORT("Tensors MMA VectorDiffusion PA is only implemented for dim 2 or 3");
   return nullptr;
}


} // namespace mfem
