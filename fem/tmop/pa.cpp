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

#include "../tmop.hpp"
#include "../tmop_tools.hpp"
#include "../gridfunc.hpp"
#include "../../linalg/dtensor.hpp"

#include <algorithm>

namespace mfem
{

namespace
{

int AbsDof(int dof)
{
   return (dof >= 0) ? dof : -1 - dof;
}

// Map lexicographic local dofs to the element's native ordering.
void BuildLexToNativeMap(const FiniteElement &fe, Array<int> &lex_to_native)
{
   const Array<int> *ordering = nullptr;
   if (const auto *nfe = dynamic_cast<const NodalFiniteElement *>(&fe))
   {
      if (nfe->GetLexicographicOrdering().Size() > 0)
      {
         ordering = &nfe->GetLexicographicOrdering();
      }
   }
   if (!ordering)
   {
      if (const auto *tfe = dynamic_cast<const TensorBasisElement *>(&fe))
      {
         if (tfe->GetDofMap().Size() > 0) { ordering = &tfe->GetDofMap(); }
      }
   }
   lex_to_native.SetSize(fe.GetDof());
   for (int i = 0; i < lex_to_native.Size(); i++)
   {
      lex_to_native[i] = ordering ? AbsDof((*ordering)[i]) : i;
   }
}

// Convert a parameter-space E-vector to physical coordinates and optionally
// update the nodal Jacobians and Hessians of the analytic map. This is the
// coefficient-like setup for a nonlinear state: AnalyticSurface is a host-side
// runtime-polymorphic interface. The repeated residual and Hessian operations
// on the resulting E-vectors are device kernels. The PA E-vector is
// lexicographic, while AnalyticCompositeSurface works with native element
// ordering, so each element is reordered around ConvertParamToPhys().
void ConvertTangential(const AnalyticCompositeSurface &surfaces,
                       const FiniteElementSpace &fes,
                       const Vector &param, Vector &phys,
                       Vector *map_jac, Vector *map_hess)
{
   const int dim = fes.GetMesh()->Dimension();
   const int ne = fes.GetNE();
   const FiniteElement &fe = *fes.GetTypicalFE();
   const int dof = fe.GetDof();
   MFEM_VERIFY(param.Size() == ne*dof*dim, "invalid E-vector size");

   phys.SetSize(param.Size(), Device::GetMemoryType());
   phys.UseDevice(true);
   const real_t *p = param.HostRead();
   real_t *x = phys.HostWrite();

   real_t *jac = nullptr;
   real_t *hess = nullptr;
   if (map_jac)
   {
      map_jac->SetSize(ne*dof*dim*dim, Device::GetMemoryType());
      map_jac->UseDevice(true);
      jac = map_jac->HostWrite();
      std::fill(jac, jac + map_jac->Size(), 0.0);
   }
   if (map_hess)
   {
      map_hess->SetSize(ne*dof*dim*dim*dim, Device::GetMemoryType());
      map_hess->UseDevice(true);
      hess = map_hess->HostWrite();
      std::fill(hess, hess + map_hess->Size(), 0.0);
   }

   Array<int> vdofs;
   Array<int> lex_to_native;
   BuildLexToNativeMap(fe, lex_to_native);
   Vector param_native(dof*dim), phys_native(dof*dim);
   for (int e = 0; e < ne; e++)
   {
      fes.GetElementVDofs(e, vdofs);
      MFEM_VERIFY(vdofs.Size() == dof*dim, "invalid element vdof count");

      for (int i = 0; i < dof; i++)
      {
         const int ni = lex_to_native[i];
         for (int c = 0; c < dim; c++)
         {
            param_native(ni + c*dof) = p[i + dof*(c + dim*e)];
         }
      }
      surfaces.ConvertParamToPhys(vdofs, param_native, phys_native);
      for (int i = 0; i < dof; i++)
      {
         const int ni = lex_to_native[i];
         for (int c = 0; c < dim; c++)
         {
            x[i + dof*(c + dim*e)] = phys_native(ni + c*dof);
         }

         if (!jac && !hess) { continue; }
         const AnalyticSurface *surface = surfaces.GetSurface(AbsDof(vdofs[ni]));
         const int node = i + dof*e;
         if (!surface)
         {
            if (jac)
            {
               for (int c = 0; c < dim; c++)
               {
                  jac[c + dim*(c + dim*node)] = 1.0;
               }
            }
            continue;
         }

         const int nparam = surface->NumParams();
         MFEM_VERIFY(nparam > 0 && nparam < dim,
                     "invalid analytic-surface parameter dimension");
         double node_param[3] = { 0.0, 0.0, 0.0 };
         for (int a = 0; a < nparam; a++)
         {
            node_param[a] = param_native(ni + a*dof);
         }
         DenseMatrix deriv_1;
         DenseTensor deriv_2;
         surface->Deriv_1(node_param, deriv_1);
         surface->Deriv_2(node_param, deriv_2);
         for (int c = 0; c < dim; c++)
         {
            for (int a = 0; a < nparam; a++)
            {
               if (jac) { jac[c + dim*(a + dim*node)] = deriv_1(c, a); }
               if (hess)
               {
                  for (int b = 0; b < nparam; b++)
                  {
                     hess[c + dim*(a + dim*(b + dim*node))] =
                        deriv_2(c, a, b);
                  }
               }
            }
         }
      }
   }
}

} // namespace

void TMOP_Integrator::AssembleGradPA(const Vector &de,
                                     const FiniteElementSpace &fes)
{
   MFEM_VERIFY(PA.enabled, "PA extension setup has not been done!");
   MFEM_VERIFY(PA.fes == &fes, "");
   // TODO: we need a more robust way to check that the 'fes' used when
   // AssemblePA() was called has not been modified or completely destroyed and
   // a new object created at the same address.

   // Form the Vector of node positions, depending on what's the input.
   Vector xe(de.Size());
   xe.UseDevice(true);
   if (x_0)
   {
      // The input is the displacement.
      add(PA.X0, de, xe);
   }
   else { xe = de; }

   const Vector *state = &xe;
   if (tan_analytic_surf)
   {
      ConvertTangential(*tan_analytic_surf, fes, xe, PA.tan_X,
                        &PA.tan_J, &PA.tan_H);
      state = &PA.tan_X;
   }

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(*state);
      PA.Jtr_debug_grad = true;
   }

   if (PA.dim == 2)
   {
      AssembleGradPA_2D(*state);
      if (lim_coeff) { AssembleGradPA_C0_2D(*state); }
      if (adapt_lim_gf.Size() > 0) { AssembleGradPA_AdaptLim_2D(*state); }

      // The mapping-curvature part of the parameter-space Hessian uses the
      // physical-coordinate residual at the linearization point.
      if (tan_analytic_surf)
      {
         PA.tan_R.SetSize(state->Size(), Device::GetMemoryType());
         PA.tan_R.UseDevice(true);
         PA.tan_R = 0.0;
         AddMultPA_2D(*state, PA.tan_R);
         if (lim_coeff) { AddMultPA_C0_2D(*state, PA.tan_R); }
         if (adapt_lim_gf.Size() > 0)
         {
            AddMultPA_AdaptLim_2D(*state, PA.tan_R);
         }
      }
   }

   if (PA.dim == 3)
   {
      AssembleGradPA_3D(*state);
      if (lim_coeff) { AssembleGradPA_C0_3D(*state); }
      if (adapt_lim_gf.Size() > 0) { AssembleGradPA_AdaptLim_3D(*state); }

      if (tan_analytic_surf)
      {
         PA.tan_R.SetSize(state->Size(), Device::GetMemoryType());
         PA.tan_R.UseDevice(true);
         PA.tan_R = 0.0;
         AddMultPA_3D(*state, PA.tan_R);
         if (lim_coeff) { AddMultPA_C0_3D(*state, PA.tan_R); }
         if (adapt_lim_gf.Size() > 0)
         {
            AddMultPA_AdaptLim_3D(*state, PA.tan_R);
         }
      }
   }
}

void TMOP_Integrator::AssemblePA_Limiting()
{
   const MemoryType mt =
      (pa_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : pa_mt;
   // Return immediately if limiting is not enabled
   if (lim_coeff == nullptr) { return; }
   MFEM_VERIFY(lim_nodes0, "internal error");

   MFEM_VERIFY(PA.enabled, "AssemblePA_Limiting but PA is not enabled!");
   MFEM_VERIFY(lim_func, "No TMOP_LimiterFunction specification!")
   MFEM_VERIFY(
      dynamic_cast<TMOP_QuadraticLimiter *>(lim_func) ||
      dynamic_cast<TMOP_ExponentialLimiter *>(lim_func),
      "Only TMOP_QuadraticLimiter and TMOP_ExponentialLimiter are supported");

   const FiniteElementSpace *fes = PA.fes;
   const int NE = PA.ne;
   if (NE == 0) { return; } // Quick return for empty processors
   const IntegrationRule &ir = *PA.ir;

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   // H0 for lim_coeff, (dim x dim) Q-vector
   PA.H0.UseDevice(true);
   PA.H0.SetSize(PA.dim * PA.dim * PA.nq * NE, mt);

   // lim_coeff -> PA.C0 (Q-vector)
   PA.C0.UseDevice(true);
   if (auto *cQ = dynamic_cast<ConstantCoefficient *>(lim_coeff))
   {
      PA.C0.SetSize(1, Device::GetMemoryType());
      PA.C0.HostWrite();
      PA.C0(0) = cQ->constant;
   }
   else
   {
      PA.C0.SetSize(PA.nq * PA.ne, Device::GetMemoryType());
      auto C0 = Reshape(PA.C0.HostWrite(), PA.nq, PA.ne);
      for (int e = 0; e < NE; ++e)
      {
         ElementTransformation &T = *fes->GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            C0(q, e) = lim_coeff->Eval(T, ir.IntPoint(q));
         }
      }
   }

   // lim_nodes0 -> PA.XL (E-vector)
   MFEM_VERIFY(lim_nodes0->FESpace()->GetVSize() == fes->GetVSize(), "");
   const Operator *n0_R = fes->GetElementRestriction(ordering);
   PA.XL.SetSize(n0_R->Height(), Device::GetMemoryType());
   PA.XL.UseDevice(true);
   n0_R->Mult(*lim_nodes0, PA.XL);

   // Limiting distances: lim_dist -> PA.LD (E-vector)
   // TODO: remove the hack for the case lim_dist == NULL.
   const FiniteElementSpace *limfes = (lim_dist) ? lim_dist->FESpace() : fes;
   const FiniteElement &lim_fe = *limfes->GetTypicalFE();
   PA.maps_lim = &lim_fe.GetDofToQuad(ir, DofToQuad::TENSOR);
   PA.LD.SetSize(NE * lim_fe.GetDof(), Device::GetMemoryType());
   PA.LD.UseDevice(true);
   if (lim_dist)
   {
      const Operator *ld_R = limfes->GetElementRestriction(ordering);
      ld_R->Mult(*lim_dist, PA.LD);
   }
   else { PA.LD = 1.0; }
}

void TargetConstructor::ComputeAllElementTargets(const FiniteElementSpace &fes,
                                                 const IntegrationRule &ir,
                                                 const Vector &xe,
                                                 DenseTensor &Jtr) const
{
   MFEM_VERIFY(Jtr.SizeI() == Jtr.SizeJ() && Jtr.SizeI() > 1, "");
   const int dim = Jtr.SizeI();
   bool done = false;
   if (dim == 2) { done = ComputeAllElementTargets<2>(fes, ir, xe, Jtr); }
   if (dim == 3) { done = ComputeAllElementTargets<3>(fes, ir, xe, Jtr); }

   if (!done) { ComputeAllElementTargets_Fallback(fes, ir, xe, Jtr); }
}

void AnalyticAdaptTC::ComputeAllElementTargets(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const Vector &xe,
                                               DenseTensor &Jtr) const
{
   ComputeAllElementTargets_Fallback(fes, ir, xe, Jtr);
}

// Code paths leading to ComputeElementTargets:
// - GetElementEnergy(elfun) which is done through GetGridFunctionEnergyPA(x)
// - AssembleElementVectorExact(elfun)
// - AssembleElementGradExact(elfun)
// - EnableNormalization(x) -> ComputeNormalizationEnergies(x)
// - (AssembleElementVectorFD(elfun))
// - (AssembleElementGradFD(elfun))
// ============================================================================
//      - TargetConstructor():
//          - IDEAL_SHAPE_UNIT_SIZE: Wideal
//          - IDEAL_SHAPE_EQUAL_SIZE: α * Wideal
//          - IDEAL_SHAPE_GIVEN_SIZE: β * Wideal
//          - GIVEN_SHAPE_AND_SIZE:   β * Wideal
//      - AnalyticAdaptTC(elfun):
//          - GIVEN_FULL: matrix_tspec->Eval(Jtr(elfun))
//      - DiscreteAdaptTC():
//          - IDEAL_SHAPE_GIVEN_SIZE: size^{1.0/dim} * Jtr(i)  (size)
//          - GIVEN_SHAPE_AND_SIZE:   Jtr(i) *= D_rho          (ratio)
//                                    Jtr(i) *= Q_phi          (skew)
//                                    Jtr(i) *= R_theta        (orientation)
void TMOP_Integrator::ComputeAllElementTargets(const Vector &xe) const
{
   PA.Jtr_needs_update = false;
   PA.Jtr_debug_grad = false;
   const FiniteElementSpace *fes = PA.fes;
   if (PA.ne == 0) { return; } // Quick return for empty processors
   const IntegrationRule &ir = *PA.ir;

   const FiniteElement &fe = *fes->GetTypicalFE();
   MFEM_VERIFY(PA.ir == &EnergyIntegrationRule(fe),
               "TMOP_Integrator::ComputeAllElementTargets() called with "
               "different IntegrationRule than the one used in AssemblePA()!");

   // Compute PA.Jtr for all elements
   targetC->ComputeAllElementTargets(*fes, ir, xe, PA.Jtr);
}

void TMOP_Integrator::UpdateCoefficientsPA(const Vector &d_loc)
{
   // All are constant or not specified.
   const int nal = PA.nal;
   const bool alc_is_qvec =
      (nal > 0) ? (PA.ALC.Size() == nal * PA.nq * PA.ne) : false;
   if (PA.MC.Size() == 1 && PA.C0.Size() <= 1 && !alc_is_qvec) { return; }

   Vector param_loc;
   if (periodic)
   {
      GetPeriodicPositions(*x_0, d_loc, *x_0->FESpace(), *PA.fes, param_loc);
   }
   else
   {
      param_loc.SetSize(x_0->Size());
      add(*x_0, d_loc, param_loc);
   }

   Vector x_loc;
   if (tan_analytic_surf)
   {
      tan_analytic_surf->ConvertParamCoordToPhys(param_loc, x_loc);
   }
   else { x_loc = param_loc; }

   // Coefficients are always evaluated on the CPU for now.
   PA.MC.HostWrite();
   PA.C0.HostWrite();
   if (alc_is_qvec) { PA.ALC.HostWrite(); }

   const IntegrationRule &ir = *PA.ir;
   auto T = new IsoparametricTransformation;
   for (int e = 0; e < PA.ne; ++e)
   {
      // Uses the node positions in x_loc.
      PA.fes->GetMesh()->GetElementTransformation(e, x_loc, T);

      if (PA.MC.Size() > 1)
      {
         for (int q = 0; q < PA.nq; ++q)
         {
            PA.MC(q + e * PA.nq) = metric_coeff->Eval(*T, ir.IntPoint(q));
         }
      }

      if (PA.C0.Size() > 1)
      {
         for (int q = 0; q < PA.nq; ++q)
         {
            PA.C0(q + e * PA.nq) = lim_coeff->Eval(*T, ir.IntPoint(q));
         }
      }

      if (alc_is_qvec)
      {
         MFEM_VERIFY(nal == adapt_lim_coeff.Size(), "internal error");
         for (int c = 0; c < nal; c++)
         {
            real_t *ALC_c = PA.ALC.HostWrite() + c * PA.nq * PA.ne;
            for (int q = 0; q < PA.nq; ++q)
            {
               ALC_c[q + e * PA.nq] =
                  adapt_lim_coeff[c]->Eval(*T, ir.IntPoint(q));
            }
         }
      }
   }

   delete T;
}

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt =
      (pa_mt == MemoryType::DEFAULT) ? Device::GetDeviceMemoryType() : pa_mt;
   PA.enabled = true;
   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int ne = PA.ne = mesh->GetNE();
   if (ne == 0) { return; } // Quick return for empty processors
   const int dim = PA.dim = mesh->Dimension();

   MFEM_VERIFY(PA.dim == 2 || PA.dim == 3, "Not yet implemented!");
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "TMOP+PA does not support mixed meshes.");
   MFEM_VERIFY(mesh->HasGeometry(Geometry::SQUARE) ||
               mesh->HasGeometry(Geometry::CUBE),
               "TMOP+PA only supports squares and cubes.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "TMOP+PAP only supports Ordering::byNODES!");

   const FiniteElement &fe = *fes.GetTypicalFE();
   PA.ir = &EnergyIntegrationRule(fe);
   const IntegrationRule &ir = *PA.ir;
   const int nq = PA.nq = ir.GetNPoints();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   PA.maps = &fe.GetDofToQuad(ir, mode);
   // Note - initial mesh. TODO delete this?
   PA.geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   // Initial node positions.
   // PA.X0 is also updated in TMOP_Integrator::SetInitialMeshPos.
   if (x_0 != nullptr)
   {
      const ElementDofOrdering ord = ElementDofOrdering::LEXICOGRAPHIC;
      auto n0_R = x_0->FESpace()->GetElementRestriction(ord);
      PA.X0.UseDevice(true);
      PA.X0.SetSize(n0_R->Height(), Device::GetMemoryType());
      n0_R->Mult(*x_0, PA.X0);
   }

   // Energy vector, scalar Q-vector
   PA.E.UseDevice(true);
   PA.E.SetSize(ne * nq, Device::GetDeviceMemoryType());

   // H for Grad, (dim x dim) Q-vector
   PA.H.UseDevice(true);
   PA.H.SetSize(dim * dim * dim * dim * nq * ne, mt);

   // Scalar Q-vector of '1', used to compute sums via dot product
   PA.O.UseDevice(true);
   PA.O.SetSize(ne * nq, Device::GetDeviceMemoryType());
   PA.O = 1.0;

   if (metric_coeff)
   {
      if (auto cc = dynamic_cast<ConstantCoefficient *>(metric_coeff))
      {
         PA.MC.SetSize(1, Device::GetMemoryType());
         PA.MC.HostWrite();
         PA.MC(0) = cc->constant;
      }
      else
      {
         PA.MC.SetSize(PA.nq * PA.ne, Device::GetMemoryType());
         auto M0 = Reshape(PA.MC.HostWrite(), PA.nq, PA.ne);
         for (int e = 0; e < PA.ne; ++e)
         {
            ElementTransformation &T = *PA.fes->GetElementTransformation(e);
            for (int q = 0; q < ir.GetNPoints(); ++q)
            {
               // Note that this is always on the initial mesh.
               M0(q,e) = metric_coeff->Eval(T, ir.IntPoint(q));
            }
         }
      }
   }
   else
   {
      PA.MC.SetSize(1, Device::GetMemoryType());
      PA.MC.HostWrite();
      PA.MC(0) = 1.0;
   }

   // Setup ref->target Jacobians, PA.Jtr, (dim x dim) Q-vector, DenseTensor
   PA.Jtr.SetSize(dim, dim, PA.ne * PA.nq, mt);
   PA.Jtr_needs_update = true;
   PA.Jtr_debug_grad = false;

   // Limiting: lim_coeff -> PA.C0, lim_nodes0 -> PA.XL, lim_dist -> PA.LD, PA.H0
   if (lim_coeff)    { AssemblePA_Limiting(); }
   // Adaptive limiting: adapt_lim_coeff -> PA.ALC, adapt_lim_gf -> PA.ALF,
   //                    adapt_lim_gf0 -> PA.ALF0, adapt_lim_delta_max -> PA.ALD
   if (adapt_lim_gf.Size() > 0) { AssemblePA_AdaptLim(); }
}

void TMOP_Integrator::AssemblePA_AdaptLim()
{
   const int nal = adapt_lim_coeff.Size();
   MFEM_VERIFY(nal > 0, "internal error");

   MFEM_VERIFY(adapt_lim_gf.Size() == nal && adapt_lim_gf0.Size() == nal,
               "internal error");
   const FiniteElementSpace *alfes = adapt_lim_gf[0]->FESpace();
   MFEM_VERIFY(alfes && alfes->GetVDim() == 1, "internal error");

   MFEM_VERIFY(strcmp(alfes->FEColl()->Name(), PA.fes->FEColl()->Name()) == 0 &&
               alfes->FEColl()->GetOrder() == PA.fes->FEColl()->GetOrder(),
               "The PA code assumes the same FE spaces for mesh and limiting.");

   PA.AL_grads_assembled = false;
   PA.nal = nal;

   // adapt_lim_coeff -> PA.ALC
   // Keep the ConstantCoefficient fast-path: when all coefficients are
   // constant, store one scalar per adaptive-limiting term.
   bool all_const = true;
   for (int c = 0; c < nal; c++)
   {
      if (!dynamic_cast<ConstantCoefficient *>(adapt_lim_coeff[c]))
      {
         all_const = false;
         break;
      }
   }
   PA.ALC.UseDevice(true);
   if (all_const)
   {
      PA.ALC.SetSize(nal, Device::GetMemoryType());
      real_t *ALC_all = PA.ALC.HostWrite();
      for (int c = 0; c < nal; c++)
      {
         auto *cc = dynamic_cast<ConstantCoefficient *>(adapt_lim_coeff[c]);
         MFEM_VERIFY(cc, "internal error");
         ALC_all[c] = cc->constant;
      }
   }
   else
   {
      // If one Coefficient is not constant, we allocate the full size for
      // all Coefficients. Could be optimized in the future.
      PA.ALC.SetSize(nal * PA.nq * PA.ne, Device::GetMemoryType());
      real_t *ALC_all = PA.ALC.HostWrite();
      for (int c = 0; c < nal; c++)
      {
         real_t *ALC_c = ALC_all + c * PA.nq * PA.ne;
         for (int e = 0; e < PA.ne; ++e)
         {
            ElementTransformation &T = *PA.fes->GetElementTransformation(e);
            for (int q = 0; q < PA.ir->GetNPoints(); ++q)
            {
               ALC_c[q + e * PA.nq] =
                  adapt_lim_coeff[c]->Eval(T, PA.ir->IntPoint(q));
            }
         }
      }
   }

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   const FiniteElement *fe_n = PA.fes->GetTypicalFE();
   // GetNodes() for tensor H1 elements with H1_DOF_MAP is stored in NATIVE
   // order (via dof_map), while DofToQuad::TENSOR assumes LEXICOGRAPHIC
   // ordering of the integration points.
   const IntegrationRule &nodes = fe_n->GetNodes();
   const auto *nfe = dynamic_cast<const NodalFiniteElement *>(fe_n);
   const Array<int> *lex = (nfe && nfe->GetLexicographicOrdering().Size() > 0)
                           ? &nfe->GetLexicographicOrdering() : nullptr;
   if (!lex)
   {
      PA.maps_nodes = &fe_n->GetDofToQuad(nodes, DofToQuad::TENSOR);
   }
   else
   {
      IntegrationRule lex_nodes(nodes.GetNPoints());
      MFEM_VERIFY(lex->Size() == nodes.GetNPoints(), "");
      for (int i = 0; i < nodes.GetNPoints(); i++)
      {
         lex_nodes.IntPoint(i) = nodes.IntPoint((*lex)[i]);
      }
      PA.maps_nodes = &fe_n->GetDofToQuad(lex_nodes, DofToQuad::TENSOR);
   }

   // Restrict each adaptive limiting field into separate contiguous E-vectors
   // (one block per adaptive limiting term).
   const Operator *alf_R = alfes->GetElementRestriction(ordering);
   const int ndofs = alfes->GetVSize();

   const int Esize = alf_R->Height();
   PA.ALF.SetSize(nal * Esize, Device::GetMemoryType());
   PA.ALF.UseDevice(true);
   PA.ALFmF0.SetSize(nal * Esize, Device::GetMemoryType());
   PA.ALFmF0.UseDevice(true);

   Vector ALFc, ALF0c;
   for (int c = 0; c < nal; c++)
   {
      ALFc.MakeRef(PA.ALF, c * Esize, Esize);
      ALF0c.MakeRef(PA.ALFmF0, c * Esize, Esize);

      MFEM_VERIFY(adapt_lim_gf[c]->Size() == ndofs, "internal error");
      MFEM_VERIFY(adapt_lim_gf0[c]->Size() == ndofs, "internal error");
      alf_R->Mult(*adapt_lim_gf[c], ALFc);
      alf_R->Mult(*adapt_lim_gf0[c], ALF0c);
   }

   // Build differences in-place: ALFmF0 = ALF - ALF0.
   PA.ALFmF0 *= -1.0;
   PA.ALFmF0 += PA.ALF;

   // Per-field delta_max values.
   MFEM_VERIFY(adapt_lim_delta_max.Size() == nal, "internal error");
   PA.ALD.SetSize(nal);
   PA.ALD.HostWrite();
   for (int c = 0; c < nal; c++) { PA.ALD(c) = adapt_lim_delta_max[c]; }

   // Allocate storage for gradient and Hessian of ALF at quadrature points
   // These will be filled during AssembleGradPA
   const int dim = PA.dim;
   PA.ALFG.UseDevice(true);
   PA.ALFG.SetSize(nal * dim * PA.nq * PA.ne, Device::GetMemoryType());
   PA.ALFH.UseDevice(true);
   PA.ALFH.SetSize(nal * dim * dim * PA.nq * PA.ne, Device::GetMemoryType());

}

void TMOP_Integrator::AssembleGradDiagonalPA(Vector &de) const
{
   // This method must be called after AssembleGradPA().

   MFEM_VERIFY(PA.Jtr_needs_update == false, "");

   if (targetC->UsesPhysicalCoordinates())
   {
      MFEM_VERIFY(PA.Jtr_debug_grad == true,
                  "AssembleGradPA() was not called"
                  " or Jtr was overwritten by another method!");
   }

   MFEM_VERIFY(tan_analytic_surf == nullptr,
               "The diagonal of tangential PA is not implemented.");

   if (PA.dim == 2)
   {
      AssembleDiagonalPA_2D(de);
      if (lim_coeff) { AssembleDiagonalPA_C0_2D(de); }
      if (adapt_lim_gf.Size() > 0) { AssembleDiagonalPA_AdaptLim_2D(de); }
   }

   if (PA.dim == 3)
   {
      AssembleDiagonalPA_3D(de);
      if (lim_coeff) { AssembleDiagonalPA_C0_3D(de); }
      if (adapt_lim_gf.Size() > 0) { AssembleDiagonalPA_AdaptLim_3D(de); }
   }
}

void TMOP_Integrator::AddMultPA(const Vector &de, Vector &ye) const
{
   // This method must be called after AssemblePA().

   // Form the Vector of node positions, depending on what's the input.
   Vector xe(de.Size());
   xe.UseDevice(true);
   if (x_0)
   {
      // The input is the displacement.
      add(PA.X0, de, xe);
   }
   else { xe = de; }

   const Vector *state = &xe;
   if (tan_analytic_surf)
   {
      ConvertTangential(*tan_analytic_surf, *PA.fes, xe, PA.tan_X,
                        &PA.tan_D, nullptr);
      state = &PA.tan_X;
   }

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(*state);
   }

   if (PA.dim == 2)
   {
      Vector *output = &ye;
      if (tan_analytic_surf)
      {
         PA.tan_C.SetSize(state->Size(), Device::GetMemoryType());
         PA.tan_C.UseDevice(true);
         PA.tan_C = 0.0;
         output = &PA.tan_C;
      }
      AddMultPA_2D(*state, *output);
      if (lim_coeff) { AddMultPA_C0_2D(*state, *output); }
      if (adapt_lim_gf.Size() > 0)
      {
         // AddMultPA_AdaptLim_2D uses the precomputed AdaptLim field gradient
         // at quadrature points (PA.ALFG). Ensure it is up-to-date for the
         // current mesh configuration.
         AssembleGradPA_AdaptLim_2D(*state);
         AddMultPA_AdaptLim_2D(*state, *output);
      }
      if (tan_analytic_surf)
      {
         TangentialPullBackPA_2D(PA.tan_D, PA.tan_C, xe);
         ye += xe;
      }
   }

   if (PA.dim == 3)
   {
      Vector *output = &ye;
      if (tan_analytic_surf)
      {
         PA.tan_C.SetSize(state->Size(), Device::GetMemoryType());
         PA.tan_C.UseDevice(true);
         PA.tan_C = 0.0;
         output = &PA.tan_C;
      }
      AddMultPA_3D(*state, *output);
      if (lim_coeff) { AddMultPA_C0_3D(*state, *output); }
      if (adapt_lim_gf.Size() > 0)
      {
         AssembleGradPA_AdaptLim_3D(*state);
         AddMultPA_AdaptLim_3D(*state, *output);
      }
      if (tan_analytic_surf)
      {
         TangentialPullBackPA_3D(PA.tan_D, PA.tan_C, xe);
         ye += xe;
      }
   }
}

void TMOP_Integrator::AddMultGradPA(const Vector &re, Vector &ce) const
{
   // This method must be called after AssembleGradPA().

   MFEM_VERIFY(PA.Jtr_needs_update == false, "");

   if (targetC->UsesPhysicalCoordinates())
   {
      MFEM_VERIFY(PA.Jtr_debug_grad == true,
                  "AssembleGradPA() was not called or "
                  "Jtr was overwritten by another method!");
   }

   if (PA.dim == 2)
   {
      if (tan_analytic_surf)
      {
         // Matrix-free parameter-space action:
         //   H_p v = B^T H_x (B v) + (R_x : d^2 Phi) v.
         // The middle operations retain their tensor-factorized PA kernels;
         // the two tangential maps are pointwise device kernels.
         TangentialPushForwardPA_2D(PA.tan_J, re, PA.tan_D);
         PA.tan_C.SetSize(re.Size(), Device::GetMemoryType());
         PA.tan_C.UseDevice(true);
         PA.tan_C = 0.0;
         AddMultGradPA_2D(PA.tan_D, PA.tan_C);
         if (lim_coeff) { AddMultGradPA_C0_2D(PA.tan_D, PA.tan_C); }
         if (adapt_lim_gf.Size() > 0)
         {
            AddMultGradPA_AdaptLim_2D(PA.tan_D, PA.tan_C);
         }
         TangentialPullBackGradPA_2D(PA.tan_C, re, PA.tan_X);
         ce += PA.tan_X;
      }
      else
      {
         AddMultGradPA_2D(re, ce);
         if (lim_coeff) { AddMultGradPA_C0_2D(re, ce); }
         if (adapt_lim_gf.Size() > 0) { AddMultGradPA_AdaptLim_2D(re, ce); }
      }
   }

   if (PA.dim == 3)
   {
      if (tan_analytic_surf)
      {
         TangentialPushForwardPA_3D(PA.tan_J, re, PA.tan_D);
         PA.tan_C.SetSize(re.Size(), Device::GetMemoryType());
         PA.tan_C.UseDevice(true);
         PA.tan_C = 0.0;
         AddMultGradPA_3D(PA.tan_D, PA.tan_C);
         if (lim_coeff) { AddMultGradPA_C0_3D(PA.tan_D, PA.tan_C); }
         if (adapt_lim_gf.Size() > 0)
         {
            AddMultGradPA_AdaptLim_3D(PA.tan_D, PA.tan_C);
         }
         TangentialPullBackGradPA_3D(PA.tan_C, re, PA.tan_X);
         ce += PA.tan_X;
      }
      else
      {
         AddMultGradPA_3D(re, ce);
         if (lim_coeff) { AddMultGradPA_C0_3D(re, ce); }
         if (adapt_lim_gf.Size() > 0) { AddMultGradPA_AdaptLim_3D(re, ce); }
      }
   }
}

real_t TMOP_Integrator::GetLocalStateEnergyPA(const Vector &de) const
{
   // This method must be called after AssemblePA().

   // Form the Vector of node positions, depending on what's the input.
   Vector xe(de.Size());
   xe.UseDevice(true);
   if (x_0)
   {
      // The input is the displacement.
      add(PA.X0, de, xe);
   }
   else { xe = de; }

   const Vector *state = &xe;
   if (tan_analytic_surf)
   {
      ConvertTangential(*tan_analytic_surf, *PA.fes, xe, PA.tan_X,
                        nullptr, nullptr);
      state = &PA.tan_X;
   }

   real_t energy = 0.0;

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(*state);
   }

   if (PA.dim == 2)
   {
      GetLocalStateEnergyPA_2D(*state, energy);
      if (lim_coeff) { energy += GetLocalStateEnergyPA_C0_2D(*state); }
      if (adapt_lim_gf.Size() > 0)
      { energy += GetLocalStateEnergyPA_AdaptLim_2D(); }
   }

   if (PA.dim == 3)
   {
      GetLocalStateEnergyPA_3D(*state, energy);
      if (lim_coeff) { energy += GetLocalStateEnergyPA_C0_3D(*state); }
      if (adapt_lim_gf.Size() > 0)
      { energy += GetLocalStateEnergyPA_AdaptLim_3D(); }
   }

   return energy;
}

} // namespace mfem
