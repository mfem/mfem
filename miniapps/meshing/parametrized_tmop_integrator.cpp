// Copyright A(c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "parametrized_tmop_integrator.hpp"
#include <unordered_map>

namespace mfem
{
void ParametrizedTMOP_Integrator::AssembleElementVectorExact(const FiniteElement &el,
                                                             ElementTransformation &T,
                                                             const Vector &elfun,
                                                             Vector &elvect)
{
   const int dof = el.GetDof(), dim = el.GetDim();
   DenseMatrix Amat(dim), work1(dim), work2(dim);
   DenseMatrix Pmat_scale(dof,dim), Pmat_temp(dof,dim), Pmat_check(dof);
   Pmat_scale = 0.0;
   Pmat_temp = 0.0;
   Pmat_check = 0.0;
   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);
   P.SetSize(dim);
   //  PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof*dim);
   PMatO.UseExternalData(elvect.GetData(), dof, dim);
   const IntegrationRule &ir = ActionIntegrationRule(el);
   const int nqp = ir.GetNPoints();
   
   // Convert parametric coordinates to physical to compute Jpt
   Array<int> vdofs;
   analyticalSurface->pfes_mesh.GetElementVDofs(T.ElementNo, vdofs);
   Vector convertedX(elfun);
   analyticalSurface->convertToPhysical(vdofs, elfun, convertedX);
   // Use converted coordinates for PMatI
   PMatI.UseExternalData(convertedX.GetData(), dof, dim);

   elvect = 0.0;
   Vector weights(nqp);
   DenseTensor Jtr(dim, dim, nqp);
   DenseTensor dJtr(dim, dim, dim*nqp);
   targetC->ComputeElementTargets(T.ElementNo, el, ir, convertedX, Jtr);

   // Limited case.
   DenseMatrix pos0;
   Vector shape, p, p0, d_vals, grad;
   shape.SetSize(dof);
   if (lim_coeff)
   {
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      lim_nodes0->FESpace()->GetElementVDofs(T.ElementNo, pos_dofs);
      lim_nodes0->GetSubVector(pos_dofs, pos0V);
      if (lim_dist)
      {
         lim_dist->GetValues(T.ElementNo, ir, d_vals);
      }
      else
      {
         d_vals.SetSize(nqp); d_vals = 1.0;
      }
   }
   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (metric_coeff || lim_coeff || adapt_lim_gf ||
       surf_fit_gf || surf_fit_pos || exact_action)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->mesh = T.mesh;
      Tpr->GetPointMat().Transpose(PMatI); // PointMat = PMatI^T
      if (exact_action)
      {
         targetC->ComputeElementTargetsGradient(ir, elfun, *Tpr, dJtr);
      }
   }

   Vector d_detW_dx(dim);
   Vector d_Winv_dx(dim);

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      metric->SetTargetJacobian(Jtr(q));
      CalcInverse(Jtr(q), Jrt);
      weights(q) = (integ_over_target) ? ip.weight * Jtr(q).Det() : ip.weight;
      double weight_m = weights(q) * metric_normal;

      el.CalcDShape(ip, DSh);
      // change comes here for DSh?
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);

      metric->EvalP(Jpt, P);

      if (metric_coeff) { weight_m *= metric_coeff->Eval(*Tpr, ip); }

      P *= weight_m;
      // AddMultABt(DS, P, PMatO);
      Pmat_temp = 0.0;
      MultABt(DS, P, Pmat_temp);
      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            Pmat_scale = 0.0;
            Pmat_check = 0.0;
            // Pmat_scale = (dx_{Ao}/dt(Bj})
            // Pmat_scale is the derivative of the parametized curve w.r.t t
            analyticalSurface->SetScaleMatrix(elfun, vdofs, i, j, Pmat_scale);
            Pmat_check = 0.0;
            MultABt(Pmat_temp, Pmat_scale, Pmat_check);
            PMatO(i,j) += Pmat_check.Trace();
         }
      }
      if (exact_action)
      {
         el.CalcShape(ip, shape);
         // Derivatives of adaptivity-based targets.
         // First term: w_q d*(Det W)/dx * mu(T)
         // d(Det W)/dx = det(W)*Tr[Winv*dW/dx]
         DenseMatrix dwdx(dim);
         for (int d = 0; d < dim; d++)
         {
            const DenseMatrix &dJtr_q = dJtr(q + d * nqp);
            Mult(Jrt, dJtr_q, dwdx );
            d_detW_dx(d) = dwdx.Trace();
         }
         d_detW_dx *= weight_m*metric->EvalW(Jpt); // *[w_q*det(W)]*mu(T)

         // Second term: w_q det(W) dmu/dx : AdWinv/dx
         // dWinv/dx = -Winv*dW/dx*Winv
         MultAtB(PMatI, DSh, Amat);
         for (int d = 0; d < dim; d++)
         {
            const DenseMatrix &dJtr_q = dJtr(q + d*nqp);
            Mult(Jrt, dJtr_q, work1); // Winv*dw/dx
            Mult(work1, Jrt, work2);  // Winv*dw/dx*Winv
            Mult(Amat, work2, work1); // A*Winv*dw/dx*Winv
            MultAtB(P, work1, work2); // dmu/dT^T*A*Winv*dw/dx*Winv
            d_Winv_dx(d) = work2.Trace(); // Tr[dmu/dT : AWinv*dw/dx*Winv]
         }
         d_Winv_dx *= -weight_m; // Include (-) factor as well

         d_detW_dx += d_Winv_dx;
         AddMultVWt(shape, d_detW_dx, PMatO);
      }

      if (lim_coeff)
      {
         if (!exact_action) { el.CalcShape(ip, shape); }
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         lim_func->Eval_d1(p, p0, d_vals(q), grad);
         grad *= weights(q) * lim_normal * lim_coeff->Eval(*Tpr, ip);
         AddMultVWt(shape, grad, PMatO);
      }
   }

   if (adapt_lim_gf) { AssembleElemVecAdaptLim(el, *Tpr, ir, weights, PMatO); }
   if (surf_fit_gf || surf_fit_pos) { AssembleElemVecSurfFit(el, *Tpr, PMatO); }

   delete Tpr;
}



void ParametrizedTMOP_Integrator::AssembleElementGradExact(const FiniteElement &el,
                                                           ElementTransformation &T,
                                                           const Vector &elfun,
                                                           DenseMatrix &elmat)
{
   const int dof = el.GetDof(), dim = el.GetDim();
   DenseMatrix Pmat_scale(dof*dim), elmat_temp(dof*dim), elmat_temp2(dof*dim), Pmat_temp(dof,dim), Pmat_hessian(dof,dim), Pmat_check(dof);
   elmat.SetSize(dof*dim);
   elmat = 0.0;
   elmat_temp = 0.0;
   elmat_temp2 = 0.0;
   Pmat_scale = 0.0;
   Pmat_temp = 0.0;
   P.SetSize(dim);
   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   Jpt.SetSize(dim);

   const IntegrationRule &ir = GradientIntegrationRule(el);
   const int nqp = ir.GetNPoints();
   // Convert parametric coordinates to physical to compute Jpt
   Array<int> vdofs;
   analyticalSurface->pfes_mesh.GetElementVDofs(T.ElementNo, vdofs);
   Vector convertedX(elfun);
   analyticalSurface->convertToPhysical(vdofs, elfun, convertedX);
   // Use converted coordinates for PMatI
   PMatI.UseExternalData(convertedX.GetData(), dof, dim);
   // Pmat_scale = (dx_{Ao}/dt(Bj}) is stored as a 2D array
   // where A,o are the row indices and B,j are the column indices
   // Pmat_scale is the derivative of the parametized curve w.r.t t
   analyticalSurface->SetScaleMatrixFourthOrder(elfun, vdofs, Pmat_scale);
   Vector weights(nqp);
   DenseTensor Jtr(dim, dim, nqp);
   targetC->ComputeElementTargets(T.ElementNo, el, ir, convertedX, Jtr);

   // Limited case.
   DenseMatrix pos0, hess;
   Vector shape, p, p0, d_vals;
   if (lim_coeff)
   {
      shape.SetSize(dof);
      p.SetSize(dim);
      p0.SetSize(dim);
      pos0.SetSize(dof, dim);
      Vector pos0V(pos0.Data(), dof * dim);
      Array<int> pos_dofs;
      lim_nodes0->FESpace()->GetElementVDofs(T.ElementNo, pos_dofs);
      lim_nodes0->GetSubVector(pos_dofs, pos0V);
      if (lim_dist)
      {
         lim_dist->GetValues(T.ElementNo, ir, d_vals);
      }
      else
      {
	d_vals.SetSize(nqp); d_vals = 1.0;
      }
   }

   // Define ref->physical transformation, when a Coefficient is specified.
   IsoparametricTransformation *Tpr = NULL;
   if (metric_coeff || lim_coeff || adapt_lim_gf || surf_fit_gf || surf_fit_pos)
   {
      Tpr = new IsoparametricTransformation;
      Tpr->SetFE(&el);
      Tpr->ElementNo = T.ElementNo;
      Tpr->ElementType = ElementTransformation::ELEMENT;
      Tpr->Attribute = T.Attribute;
      Tpr->mesh = T.mesh;
      Tpr->GetPointMat().Transpose(PMatI);
   }

   for (int q = 0; q < nqp; q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      const DenseMatrix &Jtr_q = Jtr(q);
      metric->SetTargetJacobian(Jtr_q);
      CalcInverse(Jtr_q, Jrt);
      weights(q) = (integ_over_target) ? ip.weight * Jtr_q.Det() : ip.weight;
      double weight_m = weights(q); // * metric_normal;
      el.CalcDShape(ip, DSh);
      // change comes here for DSh?
      Mult(DSh, Jrt, DS);
      MultAtB(PMatI, DS, Jpt);
      metric->EvalP(Jpt, P);
      if (metric_coeff) { weight_m *= metric_coeff->Eval(*Tpr, ip); }
      P *= weight_m;
      Pmat_temp = 0.0;
      MultABt(DS, P, Pmat_temp);
      for (int i = 0 ; i < dof; i++)
      {
         for (int idim = 0 ; idim < dim; idim++)
         {
            for (int j = 0 ; j < dof; j++)
            {
               for (int jdim = 0 ; jdim < dim; jdim++)
               {
                  Pmat_check = 0.0;
                  Pmat_hessian = 0.0;
                  // Pmat_hessian = d^{2}x_{Ao}/dt{Bj}dt_{Dr}
                  analyticalSurface->SetHessianScaleMatrix(elfun, vdofs, i, jdim, j, jdim, Pmat_hessian);
                  MultABt(Pmat_temp, Pmat_hessian, Pmat_check);
                  elmat(i + idim * dof,j + jdim * dof) += Pmat_check.Trace();
               }
            }
         }
      }
      metric->AssembleH(Jpt, DS, weight_m, elmat_temp);
      MultABt(Pmat_scale, elmat_temp, elmat_temp2);
      AddMultABt(Pmat_scale, elmat_temp2, elmat);
      if (lim_coeff)
      {
         el.CalcShape(ip, shape);
         PMatI.MultTranspose(shape, p);
         pos0.MultTranspose(shape, p0);
         weight_m = weights(q) * lim_normal * lim_coeff->Eval(*Tpr, ip);
         lim_func->Eval_d2(p, p0, d_vals(q), hess);
         for (int i = 0; i < dof; i++)
         {
            const double w_shape_i = weight_m * shape(i);
            for (int j = 0; j < dof; j++)
            {
               const double w = w_shape_i * shape(j);
               for (int d1 = 0; d1 < dim; d1++)
               {
                  for (int d2 = 0; d2 < dim; d2++)
                  {
                     elmat(d1*dof + i, d2*dof + j) += w * hess(d1, d2);
                  }
               }
            }
         }
      }
   }
   if (adapt_lim_gf) { AssembleElemGradAdaptLim(el, *Tpr, ir, weights, elmat);}
   if (surf_fit_gf || surf_fit_pos) { AssembleElemGradSurfFit(el, *Tpr, elmat);}
   delete Tpr;
} // namespace fem

} // namespace mfem
