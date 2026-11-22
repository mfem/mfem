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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "../qfunction.hpp"
#include "../restriction.hpp"
#include "../ceed/integrators/mass/mass.hpp"
#include "bilininteg_mass_kernels.hpp"
#include "bilininteg_mass_pa_simplices.hpp"

namespace mfem
{

// PA Mass Integrator

void MassIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   dim = mesh->Dimension();
   const FiniteElement &el = *fes.GetTypicalFE();
   ElementTransformation *T0 = mesh->GetTypicalElementTransformation();
   const bool stroud = fes.UsesRaggedTensorBasis();
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0, stroud);
   if (DeviceCanUseCeed())
   {
      delete ceedOp;
      const bool mixed = mesh->GetNumGeometries(mesh->Dimension()) > 1 ||
                         fes.IsVariableOrder();
      if (mixed)
      {
         ceedOp = new ceed::MixedPAMassIntegrator(*this, fes, Q);
      }
      else
      {
         ceedOp = new ceed::PAMassIntegrator(fes, *ir, Q);
      }
      return;
   }
   var_order = fes.IsVariableOrder();
   pa_groups.clear();
   if (var_order)
   {
      MFEM_VERIFY(!stroud,
                  "Variable-order MassIntegrator PA does not support ragged tensor bases.");
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "Variable-order MassIntegrator PA requires vdim == 1.");
      MFEM_VERIFY(dim >= 1 && dim <= 3, "Unsupported mesh dimension.");
      ne = mesh->GetNE();
      const auto *vr = dynamic_cast<const VarOrderElementRestriction*>(
                          fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
      MFEM_VERIFY(vr != nullptr,
                  "Expected VarOrderElementRestriction for variable-order space.");

      pa_groups.resize(vr->GetNumGroups());
      for (int gi = 0; gi < vr->GetNumGroups(); ++gi)
      {
         const Array<int> &elems = vr->GetGroupElements(gi);
         PAGroup &g = pa_groups[gi];
         g.ne = elems.Size();
         g.evec_offset = vr->GetGroupOffset(gi);
         g.evec_size = vr->GetGroupSize(gi);
         if (g.ne == 0) { continue; }

         const FiniteElement *fel = fes.GetFE(elems[0]);
         const IntegrationRule *ir_g =
            IntRule ? IntRule
            : &GetRule(*fel, *fel, *mesh->GetElementTransformation(elems[0]), false);
         g.maps = &fel->GetDofToQuad(*ir_g, DofToQuad::TENSOR);
         g.dofs1D = g.maps->ndof;
         g.quad1D = g.maps->nqpt;
         const int nq_g = ir_g->GetNPoints();
         const int exp_nq = (dim == 3) ? (g.quad1D * g.quad1D * g.quad1D)
                            : ((dim == 2) ? (g.quad1D * g.quad1D) : g.quad1D);
         MFEM_VERIFY(nq_g == exp_nq,
                     "Variable-order MassIntegrator PA requires isotropic tensor-product quadrature.");
         MFEM_VERIFY(g.dofs1D <= DeviceDofQuadLimits::Get().MAX_D1D &&
                     g.quad1D <= DeviceDofQuadLimits::Get().MAX_Q1D,
                     "Variable-order MassIntegrator PA exceeds MAX_D1D / MAX_Q1D.");

         const bool by_val = (fel->GetMapType() == FiniteElement::VALUE);
         const Array<real_t> &W = ir_g->GetWeights();
         g.pa_data.SetSize(nq_g * g.ne, mt);
         auto v = Reshape(g.pa_data.HostWrite(), nq_g, g.ne);
         for (int ge = 0; ge < g.ne; ++ge)
         {
            ElementTransformation *T = mesh->GetElementTransformation(elems[ge]);
            for (int q = 0; q < nq_g; ++q)
            {
               const IntegrationPoint &ip = (*ir_g)[q];
               T->SetIntPoint(&ip);
               const real_t detJ = T->Weight();
               const real_t c = Q ? Q->Eval(*T, ip) : 1.0;
               v(q, ge) = W[q] * c * (by_val ? detJ : 1.0 / detJ);
            }
         }
      }
      return;
   }
   int map_type = el.GetMapType();
   ne = fes.GetMesh()->GetNE();
   nq = ir->GetNPoints();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS, mt);
   if (stroud)
   {
      maps = &el.GetDofToQuad(*ir, DofToQuad::RAGGED_TENSOR);
   }
   else
   {
      maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   }
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   QuadratureSpace qs(*mesh, *ir);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);
   // QuadratureSpace expects ir defined in reference simplex for Bernstein
   // elements with partial assembly
   {
      const int NE = ne;
      const int NQ = nq;
      const bool const_c = coeff.Size() == 1;
      const bool by_val = map_type == FiniteElement::VALUE;
      const auto W = Reshape(ir->GetWeights().Read(), NQ);
      const auto J = Reshape(geom->detJ.Read(), NQ, NE);
      const auto C =
         const_c ? Reshape(coeff.Read(), 1, 1) : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE(int q, int e)
      {
         const real_t detJ = J(q, e);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * coeff * (by_val ? detJ : 1.0 / detJ);
      });
   }
}

void MassIntegrator::AssemblePABoundary(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;

   // Assuming the same element type
   fespace = &fes;
   Mesh *mesh = fes.GetMesh();
   ne = mesh->GetNFbyType(FaceType::Boundary);
   if (ne == 0) { return; }
   const FiniteElement &el = *fes.GetBE(0);
   ElementTransformation *T0 = mesh->GetBdrElementTransformation(0);
   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, *T0);

   int map_type = el.GetMapType();
   dim = el.GetDim(); // Dimension of the boundary element, *not* the mesh
   nq = ir->GetNPoints();
   face_geom = mesh->GetFaceGeometricFactors(*ir, GeometricFactors::DETERMINANTS,
                                             FaceType::Boundary, mt);
   maps = &el.GetDofToQuad(*ir, DofToQuad::TENSOR);
   dofs1D = maps->ndof;
   quad1D = maps->nqpt;
   pa_data.SetSize(ne*nq, mt);

   FaceQuadratureSpace qs(*mesh, *ir, FaceType::Boundary);
   CoefficientVector coeff(Q, qs, CoefficientStorage::COMPRESSED);

   const int NE = ne;
   const int NQ = nq;
   const bool const_c = coeff.Size() == 1;
   const bool by_val = map_type == FiniteElement::VALUE;
   {
      const auto W = Reshape(ir->GetWeights().Read(), NQ);
      const auto J = Reshape(face_geom->detJ.Read(), NQ, NE);
      const auto C = const_c ? Reshape(coeff.Read(), 1, 1)
                     : Reshape(coeff.Read(), NQ, NE);
      auto v = Reshape(pa_data.Write(), NQ, NE);
      mfem::forall(NQ, NE, [=] MFEM_HOST_DEVICE(int q, int e)
      {
         const real_t detJ = J(q, e);
         const real_t coeff = const_c ? C(0, 0) : C(q, e);
         v(q, e) = W(q) * coeff * (by_val ? detJ : 1.0 / detJ);
      });
   }
}

void MassIntegrator::AssembleDiagonalPA(Vector &diag)
{
   if (DeviceCanUseCeed())
   {
      ceedOp->GetDiagonal(diag);
   }
   else if (var_order)
   {
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector dg;
         dg.MakeRef(diag, g.evec_offset, g.evec_size);
         DiagonalPAKernels::Run(dim, g.dofs1D, g.quad1D, g.ne, g.maps->B,
                                g.pa_data, dg, g.dofs1D, g.quad1D);
         dg.SyncAliasMemory(diag);
      }
   }
   else
   {
      MFEM_VERIFY(maps && maps->mode != DofToQuad::RAGGED_TENSOR,
                  "AssembleDiagonalPA requires AssemblePA to be called first,"
                  " and is not implemented for ragged tensor bases");
      DiagonalPAKernels::Run(dim, dofs1D, quad1D, ne, maps->B, pa_data,
                             diag, dofs1D, quad1D);
   }
}

void MassIntegrator::AddMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      ceedOp->AddMult(x, y);
   }
   else if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(dim, g.dofs1D, g.quad1D, g.ne, g.maps->B,
                             g.maps->Bt, g.pa_data, xg, yg, g.dofs1D, g.quad1D);
         yg.SyncAliasMemory(y);
      }
   }
   else
   {
      const int D1D = dofs1D;
      const int Q1D = quad1D;
      const Vector &D = pa_data;
      const Array<real_t> &B = maps->B;
      const Array<real_t> &Bt = maps->Bt;

#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         if (dim == 2)
         {
            return internal::OccaPAMassApply2D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         if (dim == 3)
         {
            return internal::OccaPAMassApply3D(D1D,Q1D,ne,B,Bt,D,x,y);
         }
         MFEM_ABORT("OCCA PA Mass Apply unknown kernel!");
      }
#endif // MFEM_USE_OCCA

      if (fespace->UsesRaggedTensorBasis())
      {
         const auto *rmaps = static_cast<const RaggedDofToQuad*>(maps);

         const Array<real_t> &Ba1 = rmaps->Ba1;
         const Array<real_t> &Ba2 = rmaps->Ba2;
         const Array<real_t> &Ba3 = rmaps->Ba3;
         const Array<real_t> &Ba1t = rmaps->Ba1t;
         const Array<real_t> &Ba2t = rmaps->Ba2t;
         const Array<real_t> &Ba3t = rmaps->Ba3t;
         const Array<int> &lex_map = rmaps->lex_map;
         const Array<int> &forward_map2d = rmaps->forward_map2d_mass;
         const Array<int> &inverse_map2d = rmaps->inverse_map2d_mass;
         const Array<int> &forward_map3d = rmaps->forward_map3d_mass;
         const Array<int> &inverse_map3d = rmaps->inverse_map3d_mass;
         ApplySimplexPAKernels::Run(dim, D1D, Q1D, ne, lex_map, forward_map2d,
                                    inverse_map2d,
                                    forward_map3d, inverse_map3d, Ba1, Ba2, Ba3, Ba1t, Ba2t, Ba3t,
                                    D, x, y, D1D, Q1D);
      }
      else
      {
         ApplyPAKernels::Run(dim, D1D, Q1D, ne, B, Bt, D, x, y, D1D, Q1D);
      }
   }
}

void MassIntegrator::AddAbsMultPA(const Vector &x, Vector &y) const
{
   if (DeviceCanUseCeed())
   {
      MFEM_ABORT("AddAbsMultPA not implemented with CEED!");
      ceedOp->AddMult(x, y);
   }
   else if (var_order)
   {
      x.Read();
      for (const PAGroup &g : pa_groups)
      {
         if (g.ne == 0) { continue; }
         Vector abs_pa_data(g.pa_data);
         abs_pa_data.Abs();
         Array<real_t> absB(g.maps->B);
         Array<real_t> absBt(g.maps->Bt);
         absB.Abs();
         absBt.Abs();
         Vector xg, yg;
         xg.MakeRef(const_cast<Vector&>(x), g.evec_offset, g.evec_size);
         yg.MakeRef(y, g.evec_offset, g.evec_size);
         ApplyPAKernels::Run(dim, g.dofs1D, g.quad1D, g.ne, absB, absBt,
                             abs_pa_data, xg, yg, g.dofs1D, g.quad1D);
         yg.SyncAliasMemory(y);
      }
   }
   else
   {
      MFEM_VERIFY(!fespace->UsesRaggedTensorBasis(),
                  "AbsMultPA not implemented for ragged tensor basis");
      Vector abs_pa_data(pa_data);
      abs_pa_data.Abs();
      Array<real_t> absB(maps->B);
      Array<real_t> absBt(maps->Bt);
      absB.Abs();
      absBt.Abs();

      ApplyPAKernels::Run(dim, dofs1D, quad1D, ne, absB, absBt, abs_pa_data,
                          x, y, dofs1D, quad1D);
   }
}

void MassIntegrator::AddMultTransposePA(const Vector &x, Vector &y) const
{
   // Mass integrator is symmetric
   AddMultPA(x, y);
}

void MassIntegrator::AddAbsMultTransposePA(const Vector &x, Vector &y) const
{
   // Mass integrator is symmetric
   AddAbsMultPA(x, y);
}

} // namespace mfem
