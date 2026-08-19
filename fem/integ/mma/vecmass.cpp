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
#include "mass.hpp"
#include "form/register.hpp"

namespace mfem
{

namespace
{

/** Pack VectorMass PA (NQ, coeff_vdim, NE) = C * w * detJ — stock layout. */
void PAVectorMassDetJSetupSimplex(const int dim,
                                  const int NE,
                                  const int NQ,
                                  const int ND,
                                  const int coeff_vdim,
                                  const bool by_val,
                                  const Array<real_t> &w,
                                  const Array<real_t> &g,
                                  const Vector &nodes_e,
                                  const Vector &c,
                                  Vector &d)
{
   const bool const_c = c.Size() == coeff_vdim;
   const auto W = Reshape(w.Read(), NQ);
   const auto G = Reshape(g.Read(), NQ, dim, ND);
   const auto E = Reshape(nodes_e.Read(), ND, dim, NE);
   const auto C = const_c ? Reshape(c.Read(), coeff_vdim, 1, 1)
                  : Reshape(c.Read(), coeff_vdim, NQ, NE);
   auto D = Reshape(d.Write(), NQ, coeff_vdim, NE);

   if (dim == 2)
   {
      mfem::forall(NQ * NE, [=] MFEM_HOST_DEVICE (int idx)
      {
         const int e = idx / NQ;
         const int q = idx - NQ * e;
         real_t J11, J21, J12, J22;
         internal::EvalSimplexJ2(E, G, q, e, ND, J11, J21, J12, J22);
         const real_t detJ = J11 * J22 - J21 * J12;
         const real_t w_det = W(q) * (by_val ? detJ : real_t(1) / detJ);
         for (int c = 0; c < coeff_vdim; ++c)
         {
            const real_t Cc = const_c ? C(c, 0, 0) : C(c, q, e);
            D(q, c, e) = Cc * w_det;
         }
      });
      return;
   }

   MFEM_VERIFY(dim == 3, "");
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
      const real_t w_det = W(q) * (by_val ? detJ : real_t(1) / detJ);
      for (int c = 0; c < coeff_vdim; ++c)
      {
         const real_t Cc = const_c ? C(c, 0, 0) : C(c, q, e);
         D(q, c, e) = Cc * w_det;
      }
   });
}

} // namespace

void VectorMassIntegrator::AssembleSimplexMmaPA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "");
   MFEM_VERIFY(mesh->SpaceDimension() == dim, "");

   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation &Trans = *mesh->GetTypicalElementTransformation();
   const Geometry::Type geom_t = (dim == 2) ? Geometry::TRIANGLE
                                 : Geometry::TETRAHEDRON;
   MFEM_VERIFY(el.GetGeomType() == geom_t, "");
   MFEM_VERIFY(IsSimplexMmaH1Element(el, dim), "");

   vdim = (vdim == -1) ? Trans.GetSpaceDim() : vdim;
   MFEM_VERIFY(vdim == fes.GetVDim(), "vdim != fes.GetVDim()");

   const int map_type = el.GetMapType();
   const int p = el.GetOrder();
   dofs1D = p + 1;
   const int ndof = el.GetDof();

   const auto *ir_ptr = IntRule ? IntRule : &MassIntegrator::GetRule(el, el,
                                                                     Trans);
   const IntegrationRule &ir = *ir_ptr;
   nq = ir.GetNPoints();
   quad1D = 0;
   ne = mesh->GetNE();
   use_simplices_mma = true;
   use_tensors_mma = false;
   maps = nullptr;
   geom = nullptr;

   simplex_mma_P.SetSize(nq * ndof, mt);
   {
      real_t *Ph = simplex_mma_P.HostWrite();
      Vector shape_ref(ndof);
      for (int q = 0; q < nq; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         el.CalcShape(ip, shape_ref);
         for (int i = 0; i < ndof; i++)
         {
            Ph[q + nq * i] = shape_ref(i);
         }
      }
   }

   Vector nodes_e;
   int nd_n = 0, sdim = 0;
   internal::GetSimplexMeshNodesE(*mesh, mt, nodes_e, nd_n, sdim);
   MFEM_VERIFY(sdim == dim, "");
   const FiniteElement &nfe = *mesh->GetNodes()->FESpace()->GetTypicalFE();
   const DofToQuad &nmaps = nfe.GetDofToQuad(ir, DofToQuad::FULL);
   MFEM_VERIFY(nmaps.ndof == nd_n && nmaps.nqpt == nq, "");

   QuadratureSpace qs(*mesh, ir);
   CoefficientVector coeff(qs);
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
   const bool const_coeff = coeff_vdim == 1;
   const bool vector_coeff = coeff_vdim == vdim;
   const bool matrix_coeff = coeff_vdim == vdim * vdim;
   MFEM_VERIFY(const_coeff + vector_coeff + matrix_coeff == 1, "");

   pa_data.SetSize(coeff_vdim * nq * ne, mt);

   const bool by_val = map_type == FiniteElement::VALUE;
   PAVectorMassDetJSetupSimplex(dim, ne, nq, nd_n, coeff_vdim, by_val,
                                ir.GetWeights(), nmaps.G, nodes_e, coeff,
                                pa_data);
}

void VectorMassIntegrator::RegisterSimplexMmaKernels()
{
   // Shared with Mass — see form/register.hpp.
   internal::mma::RegisterMassSimplexMmaSpecializations<VectorMassIntegrator>();
}

VectorMassIntegrator::ApplySimplexMmaKernelType
VectorMassIntegrator::ApplySimplexMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaVectorMassApplySimplex2D; }
   if (dim == 3) { return internal::MmaVectorMassApplySimplex3D; }
   MFEM_ABORT("Simplex MMA VectorMass PA is only implemented for dim 2 or 3");
   return nullptr;
}

void VectorMassIntegrator::RegisterTensorsMmaKernels()
{
   // Shared tensor list (p = 3..7) — see form/register.hpp.
   internal::mma::RegisterTensorsMmaSpecializations<VectorMassIntegrator>();
}

VectorMassIntegrator::ApplyTensorsMmaKernelType
VectorMassIntegrator::ApplyTensorsMmaPAKernels::Fallback(int dim, int, int)
{
   if (dim == 2) { return internal::MmaVectorMassApplyTensors2D; }
   if (dim == 3) { return internal::MmaVectorMassApplyTensors3D; }
   MFEM_ABORT("Tensors MMA VectorMass PA is only implemented for dim 2 or 3");
   return nullptr;
}


} // namespace mfem
