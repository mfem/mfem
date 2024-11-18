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

#include "../tmop.hpp"
#include "../linearform.hpp"
#include "../pgridfunc.hpp"
#include "../tmop_tools.hpp"
#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../qinterp/grad.hpp"

namespace mfem
{

void TMOP_Integrator::AssembleGradPA(const Vector &xe,
                                     const FiniteElementSpace &fes)
{
   MFEM_VERIFY(PA.enabled, "PA extension setup has not been done!");
   MFEM_VERIFY(PA.fes == &fes, "");
   // TODO: we need a more robust way to check that the 'fes' used when
   // AssemblePA() was called has not been modified or completely destroyed and
   // a new object created at the same address.

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(xe);
      PA.Jtr_debug_grad = true;
   }

   if (PA.dim == 2)
   {
      AssembleGradPA_2D(xe);
      if (lim_coeff) { AssembleGradPA_C0_2D(xe); }
      if (surf_fit_coeff) { AssembleGradPA_Fit_2D(xe); }
   }

   if (PA.dim == 3)
   {
      AssembleGradPA_3D(xe);
      if (lim_coeff) { AssembleGradPA_C0_3D(xe); }
      if (surf_fit_coeff) { AssembleGradPA_Fit_3D(xe); }
   }
}

void TMOP_Integrator::AssemblePA_Limiting()
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   // Return immediately if limiting is not enabled
   if (lim_coeff == nullptr) { return; }
   MFEM_VERIFY(lim_nodes0, "internal error");

   MFEM_VERIFY(PA.enabled, "AssemblePA_Limiting but PA is not enabled!");
   MFEM_VERIFY(lim_func, "No TMOP_LimiterFunction specification!")
   MFEM_VERIFY(dynamic_cast<TMOP_QuadraticLimiter*>(lim_func) ||
               dynamic_cast<TMOP_ExponentialLimiter*>(lim_func),
               "Only TMOP_QuadraticLimiter and TMOP_ExponentialLimiter are supported");

   const FiniteElementSpace *fes = PA.fes;
   const int NE = PA.ne;
   if (NE == 0) { return; }  // Quick return for empty processors
   const IntegrationRule &ir = *PA.ir;

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   // H0 for lim_coeff, (dim x dim) Q-vector
   PA.H0.UseDevice(true);
   PA.H0.SetSize(PA.dim * PA.dim * PA.nq * NE, mt);

   // lim_coeff -> PA.C0 (Q-vector)
   PA.C0.UseDevice(true);
   if (ConstantCoefficient* cQ =
          dynamic_cast<ConstantCoefficient*>(lim_coeff))
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
         ElementTransformation& T = *fes->GetElementTransformation(e);
         for (int q = 0; q < ir.GetNPoints(); ++q)
         {
            C0(q,e) = lim_coeff->Eval(T, ir.IntPoint(q));
         }
      }
   }

   // lim_nodes0 -> PA.X0 (E-vector)
   MFEM_VERIFY(lim_nodes0->FESpace() == fes, "");
   const Operator *n0_R = fes->GetElementRestriction(ordering);
   PA.X0.SetSize(n0_R->Height(), Device::GetMemoryType());
   PA.X0.UseDevice(true);
   n0_R->Mult(*lim_nodes0, PA.X0);

   // Limiting distances: lim_dist -> PA.LD (E-vector)
   // TODO: remove the hack for the case lim_dist == NULL.
   const FiniteElementSpace *limfes = (lim_dist) ? lim_dist->FESpace() : fes;
   const FiniteElement &lim_fe = *limfes->GetFE(0);
   PA.maps_lim = &lim_fe.GetDofToQuad(ir, DofToQuad::TENSOR);
   PA.LD.SetSize(NE*lim_fe.GetDof(), Device::GetMemoryType());
   PA.LD.UseDevice(true);
   if (lim_dist)
   {
      const Operator *ld_R = limfes->GetElementRestriction(ordering);
      ld_R->Mult(*lim_dist, PA.LD);
   }
   else
   {
      PA.LD = 1.0;
   }
}

IntegrationRule PermuteIR(const IntegrationRule *irule,
                          const Array<int> &perm)
{
   const int np = irule->GetNPoints();
   MFEM_VERIFY(np == perm.Size(), "Invalid permutation size");
   IntegrationRule ir(np);
   ir.SetOrder(irule->GetOrder());

   for (int i = 0; i < np; i++)
   {
      IntegrationPoint &ip_new = ir.IntPoint(i);
      const IntegrationPoint &ip_old = irule->IntPoint(perm[i]);
      ip_new.Set(ip_old.x, ip_old.y, ip_old.z, ip_old.weight);
   }

   return ir;
}

void TMOP_Integrator::AssemblePA_Fitting()
{
   // Return immediately if surface fitting is not enabled
   if (surf_fit_coeff == nullptr) { return; }
   MFEM_VERIFY(PA.enabled, "AssemblePA_Fitting but PA is not enabled!");
   MFEM_VERIFY(surf_fit_gf, "No surface fitting function specification!");

   const int NE = PA.ne;
   if (NE == 0) { return; }  // Quick return for empty processors
   const FiniteElementSpace *fes_fit = surf_fit_gf->FESpace();
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   // surf_fit_coeff -> PA.SFC
   ConstantCoefficient* cS = dynamic_cast<ConstantCoefficient*>(surf_fit_coeff);
   MFEM_VERIFY(cS, "PA only supported for constant coefficient currently\n");
   PA.SFC = cS->constant;

   // surf_fit_gf -> PA.SFV (E-vector)
   const Operator *n1_R = fes_fit->GetElementRestriction(ordering);
   PA.SFV.SetSize(n1_R->Height(), Device::GetMemoryType());
   PA.SFV.UseDevice(true);
   n1_R->Mult(*surf_fit_gf, PA.SFV);

   // surf_fit_dof_count -> PA.SFDC (E-vector)
   Vector temp1;
   temp1.SetSize(surf_fit_dof_count.Size());
   for (int i = 0; i < temp1.Size(); i++)
   {
      temp1[i] = surf_fit_dof_count[i];
   }
   PA.SFDC.SetSize(n1_R->Height(), Device::GetMemoryType());
   PA.SFDC.UseDevice(true);
   n1_R->Mult(temp1, PA.SFDC);

   // surf_fit_marker -> PA.SFM
   Vector temp2(surf_fit_dof_count.Size());
   for (int i = 0; i < surf_fit_marker->Size(); i++)
   {
      temp2[i] = (*surf_fit_marker)[i] ? 1.0 : 0.0;
      // if ((*surf_fit_marker)[i] == true)
      // {
      //    temp2[i] = 1.0;
      // }
      // else
      // {
      //    temp2[i] = 0.0;
      // }
   }
   PA.SFM.SetSize(n1_R->Height(), Device::GetMemoryType());
   PA.SFM.UseDevice(true);
   n1_R->Mult(temp2, PA.SFM);

   // Make list of elements that have atleast one dof marked for fitting
   PA.SFEList.SetSize(0);
   for (int el_id = 0; el_id < NE; el_id++)
   {
      Array<int> dofs, vdofs;
      fes_fit->GetElementVDofs(el_id, vdofs);
      int count = 0;
      const FiniteElement &el_s = *fes_fit->GetFE(el_id);
      const int dof_s = el_s.GetDof();
      for (int s = 0; s < dof_s; s++)
      {
         const int scalar_dof_id = fes_fit->VDofToDof(vdofs[s]);
         count += ((*surf_fit_marker)[scalar_dof_id]) ? 1 : 0;
      }
      if (count != 0) { PA.SFEList.Append(el_id);}
   }
   PA.nefit = PA.SFEList.Size();
   int fit_el_dof_count = 0;
   if (PA.nefit > 0)
   {
      Array<int> dofs;
      PA.fes->GetElementVDofs(0, dofs);
      fit_el_dof_count = PA.nefit*dofs.Size();
   }

   if (surf_fit_grad)
   {
      const FiniteElementSpace *fes_grad = surf_fit_grad->FESpace();
      const FiniteElementSpace *fes_hess = surf_fit_hess->FESpace();

      // surf_fit_grad -> PA.SFG
      const Operator *n2_R = fes_grad->GetElementRestriction(ordering);
      PA.SFG.SetSize(n2_R->Height(), Device::GetMemoryType());
      PA.SFG.UseDevice(true);
      n2_R->Mult(*surf_fit_grad, PA.SFG);

      // surf_fit_hess -> PA.SFH
      const Operator *n3_R = fes_hess->GetElementRestriction(ordering);
      PA.SFH.SetSize(n3_R->Height(), Device::GetMemoryType());
      PA.SFH.UseDevice(true);
      n3_R->Mult(*surf_fit_hess, PA.SFH);
   }
   else
   {
      const int dim = fes_fit->GetMesh()->Dimension();
      const FiniteElement &fe = *(fes_fit->GetFE(0));
      const IntegrationRule irnodes = fe.GetNodes();
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>(&fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir = PermuteIR(&irnodes, irordering);
      const DofToQuad maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);
      auto geom = fes_fit->GetMesh()->GetGeometricFactors(ir,
                                                          GeometricFactors::JACOBIANS);
      int nelem = fes_fit->GetMesh()->GetNE();

      constexpr QVectorLayout L = QVectorLayout::byNODES;
      using CGK = QuadratureInterpolator::CollocatedGradKernels;
      const int nd = maps.ndof;

      //Gradient using Collocated Derivatives
      PA.SFG.SetSize(dim*PA.SFV.Size(), Device::GetMemoryType());
      PA.SFG.UseDevice(true);
      CGK::Run(dim, L, true, 1, nd, nelem, maps.G.Read(),
               geom->J.Read(), PA.SFV.Read(), PA.SFG.Write(), dim, 1, nd);

      //Hessian using Collocated Derivatives
      PA.SFH.SetSize(dim*dim*PA.SFV.Size(), Device::GetMemoryType());
      PA.SFH.UseDevice(true);
      CGK::Run(dim, L, true, dim, nd, nelem, maps.G.Read(),
               geom->J.Read(), PA.SFG.Read(), PA.SFH.Write(), dim, dim, nd);
   }

   // "Partial" E-vector of '1' for surface fitting.
   PA.SFO.SetSize(fit_el_dof_count, Device::GetDeviceMemoryType());
   PA.SFO = 1.0;

   // "Partial" E-vector for energy contribution due to surface fitting.
   PA.SFE.UseDevice(true);
   PA.SFE.SetSize(fit_el_dof_count, Device::GetDeviceMemoryType());
   PA.SFE = 0.0;

   // Hessian vector for surface fitting.
   PA.SFH0.UseDevice(true);
   PA.SFH0.SetSize(PA.SFH.Size(), Device::GetDeviceMemoryType());
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
   if (PA.ne == 0) { return; }  // Quick return for empty processors
   const IntegrationRule &ir = *PA.ir;

   // Compute PA.Jtr for all elements
   targetC->ComputeAllElementTargets(*fes, ir, xe, PA.Jtr);
}

void TMOP_Integrator::UpdateCoefficientsPA(const Vector &x_loc)
{
   UpdateSurfaceFittingPA(x_loc);

   // Both are constant or not specified.
   if (PA.MC.Size() == 1 && PA.C0.Size() == 1) { return; }
   // Limiting coefficients are always evaluated on the CPU for now.
   PA.MC.HostWrite();
   PA.C0.HostWrite();

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

   }
   delete T;
}

void TMOP_Integrator::UpdateSurfaceFittingPA(const Vector &x_loc)
{
   // Update surf_fit_gf and its gradients if surface fitting is enabled.
   if (!surf_fit_gf) { return; }
   const FiniteElementSpace *fes_fit = surf_fit_gf->FESpace();
   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;

   const Operator *n1_R_int = fes_fit->GetElementRestriction(ordering);
   n1_R_int->Mult(*surf_fit_gf, PA.SFV);

   if (surf_fit_grad)
   {
      const FiniteElementSpace *fes_grad = surf_fit_grad->FESpace();
      const FiniteElementSpace *fes_hess = surf_fit_hess->FESpace();

      const Operator *n2_R_int = fes_grad->GetElementRestriction(ordering);
      n2_R_int->Mult(*surf_fit_grad, PA.SFG);

      const Operator *n3_R_int = fes_hess->GetElementRestriction(ordering);
      n3_R_int->Mult(*surf_fit_hess, PA.SFH);
   }
   else
   {
      const FiniteElement &fe = *(fes_fit->GetFE(0));
      const IntegrationRule irnodes = fe.GetNodes();
      const NodalFiniteElement *nfe = dynamic_cast<const NodalFiniteElement*>(&fe);
      const Array<int> &irordering = nfe->GetLexicographicOrdering();
      IntegrationRule ir = PermuteIR(&irnodes, irordering);

      int nelem = fes_fit->GetMesh()->GetNE();
      const DofToQuad maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);

      const Operator *R_nodes = PA.fes->GetElementRestriction(ordering);
      Vector xelem;
      xelem.SetSize(R_nodes->Height(), Device::GetMemoryType());
      xelem.UseDevice(true);
      R_nodes->Mult(x_loc, xelem);

      Vector Jacobians;
      Jacobians.SetSize(xelem.Size()*PA.dim, Device::GetMemoryType());
      Jacobians.UseDevice(true);
      constexpr QVectorLayout L = QVectorLayout::byNODES;

      using CGK = QuadratureInterpolator::CollocatedGradKernels;

      const int nd = maps.ndof;
      // Compute Jacobians since mesh might not know about coordinate change
      CGK::Run(PA.dim, L, false, PA.dim, nd, nelem, maps.G.Read(), nullptr,
               xelem.Read(), Jacobians.Write(), PA.dim, PA.dim, nd);

      if (PA.dim == 2)
      {
         constexpr bool grad_phys = true;
         const int sdim = 2; // spatial dimension = 2
         const int vdim = 1; // level-set field is a scalar function

         CGK::Run(sdim, L, grad_phys, vdim, nd, nelem, maps.G.Read(),
                  Jacobians.Read(), PA.SFV.Read(), PA.SFG.Write(),
                  sdim, vdim, nd);

         CGK::Run(sdim, L, grad_phys, 2*vdim, nd, nelem, maps.G.Read(),
                  Jacobians.Read(), PA.SFG.Read(), PA.SFH.Write(),
                  sdim, 2*vdim, nd);
      }
      if (PA.dim == 3)
      {
         constexpr bool grad_phys = true;
         const int sdim = 3; // spatial dimension = 3
         const int vdim = 1; // level-set field is a scalar function

         CGK::Run(sdim, L, grad_phys, vdim, nd, nelem, maps.G.Read(),
                  Jacobians.Read(), PA.SFV.Read(), PA.SFG.Write(),
                  sdim, vdim, nd);

         CGK::Run(sdim, L, grad_phys, 3*vdim, nd, nelem, maps.G.Read(),
                  Jacobians.Read(), PA.SFG.Read(), PA.SFH.Write(),
                  sdim, 3*vdim, nd);
      }
   }

   ConstantCoefficient* cS = dynamic_cast<ConstantCoefficient*>(surf_fit_coeff);
   PA.SFC = cS->constant;
}

void TMOP_Integrator::AssemblePA(const FiniteElementSpace &fes)
{
   const MemoryType mt = (pa_mt == MemoryType::DEFAULT) ?
                         Device::GetDeviceMemoryType() : pa_mt;
   PA.enabled = true;
   PA.fes = &fes;
   Mesh *mesh = fes.GetMesh();
   const int ne = PA.ne = mesh->GetNE();
   if (ne == 0) { return; }  // Quick return for empty processors
   const int dim = PA.dim = mesh->Dimension();
   MFEM_VERIFY(PA.dim == 2 || PA.dim == 3, "Not yet implemented!");
   MFEM_VERIFY(mesh->GetNumGeometries(dim) <= 1,
               "mixed meshes are not supported");
   MFEM_VERIFY(!fes.IsVariableOrder(), "variable orders are not supported");
   const FiniteElement &fe = *fes.GetFE(0);
   PA.ir = &EnergyIntegrationRule(fe);
   const IntegrationRule &ir = *PA.ir;
   MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
               "PA Only supports Ordering::byNODES!");

   const int nq = PA.nq = ir.GetNPoints();
   const DofToQuad::Mode mode = DofToQuad::TENSOR;
   PA.maps = &fe.GetDofToQuad(ir, mode);
   PA.geom = mesh->GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   // Energy vector, scalar Q-vector
   PA.E.UseDevice(true);
   PA.E.SetSize(ne*nq, Device::GetDeviceMemoryType());

   // H for Grad, (dim x dim) Q-vector
   PA.H.UseDevice(true);
   PA.H.SetSize(dim*dim * dim*dim * nq*ne, mt);

   // Scalar Q-vector of '1', used to compute sums via dot product
   PA.O.SetSize(ne*nq, Device::GetDeviceMemoryType());
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
            ElementTransformation& T = *PA.fes->GetElementTransformation(e);
            for (int q = 0; q < ir.GetNPoints(); ++q)
            {
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
   PA.Jtr.SetSize(dim, dim, PA.ne*PA.nq, mt);
   PA.Jtr_needs_update = true;
   PA.Jtr_debug_grad = false;

   // Limiting: lim_coeff -> PA.C0, lim_nodes0 -> PA.X0, lim_dist -> PA.LD, PA.H0
   if (lim_coeff) { AssemblePA_Limiting(); }
   if (surf_fit_gf) { AssemblePA_Fitting(); }
}

void TMOP_Integrator::AssembleGradDiagonalPA(Vector &de) const
{
   // This method must be called after AssembleGradPA().

   MFEM_VERIFY(PA.Jtr_needs_update == false, "");

   if (targetC->UsesPhysicalCoordinates())
   {
      MFEM_VERIFY(PA.Jtr_debug_grad == true, "AssembleGradPA() was not called"
                  " or Jtr was overwritten by another method!");
   }

   if (PA.dim == 2)
   {
      AssembleDiagonalPA_2D(de);
      if (lim_coeff) { AssembleDiagonalPA_C0_2D(de); }
      if (surf_fit_coeff) { AssembleDiagonalPA_Fit_2D(de); }
   }

   if (PA.dim == 3)
   {
      AssembleDiagonalPA_3D(de);
      if (lim_coeff) { AssembleDiagonalPA_C0_3D(de); }
      if (surf_fit_coeff) { AssembleDiagonalPA_Fit_3D(de); }

   }
}

void TMOP_Integrator::AddMultPA(const Vector &xe, Vector &ye) const
{
   // This method must be called after AssemblePA().

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(xe);
   }

   if (PA.dim == 2)
   {
      AddMultPA_2D(xe,ye);
      if (lim_coeff) { AddMultPA_C0_2D(xe,ye); }
      if (surf_fit_coeff) { AddMultPA_Fit_2D(xe,ye);}
   }

   if (PA.dim == 3)
   {
      AddMultPA_3D(xe,ye);
      if (lim_coeff) { AddMultPA_C0_3D(xe,ye); }
      if (surf_fit_coeff) {AddMultPA_Fit_3D(xe,ye);}
   }
}

void TMOP_Integrator::AddMultGradPA(const Vector &re, Vector &ce) const
{
   // This method must be called after AssembleGradPA().

   MFEM_VERIFY(PA.Jtr_needs_update == false, "");

   if (targetC->UsesPhysicalCoordinates())
   {
      MFEM_VERIFY(PA.Jtr_debug_grad == true, "AssembleGradPA() was not called or "
                  "Jtr was overwritten by another method!");
   }

   if (PA.dim == 2)
   {
      AddMultGradPA_2D(re,ce);
      if (lim_coeff) { AddMultGradPA_C0_2D(re,ce); }
      if (surf_fit_coeff) { AddMultGradPA_Fit_2D(re,ce); }
   }

   if (PA.dim == 3)
   {
      AddMultGradPA_3D(re,ce);
      if (lim_coeff) { AddMultGradPA_C0_3D(re,ce); }
      if (surf_fit_coeff) { AddMultGradPA_Fit_3D(re,ce); }
   }
}

real_t TMOP_Integrator::GetLocalStateEnergyPA(const Vector &xe) const
{
   // This method must be called after AssemblePA().

   real_t energy = 0.0;

   if (PA.Jtr_needs_update || targetC->UsesPhysicalCoordinates())
   {
      ComputeAllElementTargets(xe);
   }

   if (PA.dim == 2)
   {
      energy = GetLocalStateEnergyPA_2D(xe);
      if (lim_coeff) { energy += GetLocalStateEnergyPA_C0_2D(xe); }
      if (surf_fit_coeff) { energy += GetLocalStateEnergyPA_Fit_2D(xe); }
   }

   if (PA.dim == 3)
   {
      energy = GetLocalStateEnergyPA_3D(xe);
      if (lim_coeff) { energy += GetLocalStateEnergyPA_C0_3D(xe); }
      if (surf_fit_coeff) { energy += GetLocalStateEnergyPA_Fit_3D(xe); }
   }

   return energy;
}

} // namespace mfem
