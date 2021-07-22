// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "sbm_solver.hpp"
#include "mfem.hpp"

namespace mfem
{

double ShiftedFunctionCoefficient::Eval(ElementTransformation & T,
                                        const IntegrationPoint & ip,
                                        const Vector &D)
{
   Vector transip;
   T.Transform(ip, transip);
   for (int i = 0; i < D.Size(); i++)
   {
      transip(i) += D(i);
   }

   return Function(transip);
}

void ShiftedVectorFunctionCoefficient::Eval(Vector &V,
                                            ElementTransformation & T,
                                            const IntegrationPoint & ip,
                                            const Vector &D)
{
   Vector transip;
   T.Transform(ip, transip);
   for (int i = 0; i < D.Size(); i++)
   {
      transip(i) += D(i);
   }

   Function(transip, V);
}

void SBM2DirichletIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, ndof1, ndof2, ndof, ndoftotal;
   double w;
   DenseMatrix temp_elmat;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   ndoftotal = Trans.ElementType == ElementTransformation::BDR_FACE ?
               ndof1 : ndof1 + ndof2;

   elmat.SetSize(ndoftotal);
   elmat = 0.0;

   bool elem1f = true; // flag indicating whether Trans.Elem1No is part of the
   // surrogate domain or not.
   int elem1 = Trans.Elem1No,
       elem2 = Trans.Elem2No,
       marker1 = (*elem_marker)[elem1];

   int marker2;
   if (Trans.Elem2No >= NEproc)
   {
      marker2 = (*elem_marker)[NEproc+par_shared_face_count];
      par_shared_face_count++;
   }
   else if (Trans.ElementType == ElementTransformation::BDR_FACE)
   {
      marker2 = marker1;
   }
   else
   {
      marker2 = (*elem_marker)[elem2];
   }

   if (!include_cut_cell)
   {
      // 1 is inside and 2 is cut or 1 is a boundary element.
      if (marker1 == ShiftedFaceMarker::SBElementType::INSIDE &&
          (cut_marker.Find(marker2) != -1 ||
           Trans.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
      }
      // 1 is cut, 2 is inside
      else if (cut_marker.Find(marker1) != -1 &&
               marker2 == ShiftedFaceMarker::SBElementType::INSIDE)
      {
         if (Trans.Elem2No >= NEproc) { return; }
         elem1f = false;
      }
      else
      {
         return;
      }
   }
   else
   {
      // 1 is cut and 2 is outside or 1 is a boundary element.
      if (cut_marker.Find(marker1) != -1 &&
          (marker2 == ShiftedFaceMarker::SBElementType::OUTSIDE ||
           Trans.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
      }
      // 1 is outside, 2 is cut
      else if (marker1 == ShiftedFaceMarker::SBElementType::OUTSIDE &&
               cut_marker.Find(marker2) != -1)
      {
         if (Trans.Elem2No >= NEproc) { return; }
         elem1f = false;
      }
      else
      {
         return;
      }
   }

   ndof = elem1f ? ndof1 : ndof2;

   temp_elmat.SetSize(ndof);
   temp_elmat = 0.;

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshapephys.SetSize(ndof, dim);
   Vector dshapephysdd(ndof);
   dshapedn.SetSize(ndof);
   Vector wrk = shape;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = elem1f ? 4*el1.GetOrder() : 4*el2.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   Array<DenseMatrix *> dkphi_dxk;
   DenseMatrix grad_phys;
   Vector Factorial;
   Array<DenseMatrix *> grad_phys_dir;

   if (nterms > 0)
   {
      if (elem1f)
      {
         el1.ProjectGrad(el1, *Trans.Elem1, grad_phys);
      }
      else
      {
         el2.ProjectGrad(el2, *Trans.Elem2, grad_phys);
      }

      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrices for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(ndof, ndof);
         grad_phys_dir[i]->CopyRows(grad_phys, i*ndof, (i+1)*ndof-1);
      }

      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(ndof, ndof*dim);

      dkphi_dxk.SetSize(nterms);

      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         dkphi_dxk[i] = new DenseMatrix(ndof, ndof*sz1*dim);
         int loc_col_per_dof = sz1;
         int tot_col_per_dof = loc_col_per_dof*dim;
         for (int k = 0; k < dim; k++)
         {
            grad_work.SetSize(ndof, ndof*sz1);
            // grad_work[k] has derivative in kth direction for each DOF.
            // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
            // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
            if (i == 0)
            {
               Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            }
            else
            {
               Mult(*grad_phys_dir[k], *dkphi_dxk[i-1], grad_work);
            }
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
            for (int j = 0; j < ndof; j++)
            {
               for (int d = 0; d < loc_col_per_dof; d++)
               {
                  Vector col;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[i]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
               }
            }
         }
      }

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }

      Factorial.SetSize(nterms);
      Factorial(0) = 2;
      for (int i = 1; i < nterms; i++)
      {
         Factorial(i) = Factorial(i-1)*(i+2);
      }
   }

   DenseMatrix q_hess_dn(dim, ndof);
   Vector q_hess_dn_work(q_hess_dn.GetData(), ndof*dim);
   Vector q_hess_dot_d(ndof);

   Vector D(vD->GetVDim());
   // Assemble: -< \nabla u.n, w >
   //           -< u + \nabla u.d + h.o.t, \nabla w.n>
   //           -<alpha h^{-1} (u + \nabla u.d + h.o.t), w + \nabla w.d + h.o.t>
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         // Note: this normal accounts for the weight of the surface transformation
         // Jacobian i.e. nor = nhat*det(J)
         CalcOrtho(Trans.Jacobian(), nor);
      }
      vD->Eval(D, Trans, ip);

      double nor_dot_d = nor*D;
      // If we are clipping inside the domain, ntilde and d vector should be
      // aligned.
      if (!include_cut_cell && nor_dot_d < 0) { nor *= -1; }
      if (include_cut_cell && nor_dot_d > 0) { nor *= -1; }

      if (elem1f)
      {
         el1.CalcShape(eip1, shape);
         el1.CalcDShape(eip1, dshape);
         w = ip.weight/Trans.Elem1->Weight();
         CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      }
      else
      {
         el1.CalcShape(eip2, shape);
         el1.CalcDShape(eip2, dshape);
         w = ip.weight/Trans.Elem2->Weight();
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
      }

      ni.Set(w, nor); // alpha_k*nor/det(J)
      adjJ.Mult(ni, nh);
      dshape.Mult(nh, dshapedn); // dphi/dn * Jinv * alpha_k * nor

      // <grad u.n, w> - Term 2
      AddMult_a_VWt(-1., shape, dshapedn, temp_elmat);

      if (elem1f) { el1.CalcPhysDShape(*(Trans.Elem1), dshapephys); }
      else { el1.CalcPhysDShape(*(Trans.Elem2), dshapephys); } // dphi/dx
      dshapephys.Mult(D, dshapephysdd); // dphi/dx.D);

      q_hess_dot_d = 0.;
      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         DenseMatrix T1(dim, ndof*sz1);
         Vector T1_wrk(T1.GetData(), dim*ndof*sz1);
         dkphi_dxk[i]->MultTranspose(shape, T1_wrk);

         DenseMatrix T2;
         Vector T2_wrk;
         for (int j = 0; j < i+1; j++)
         {
            int sz2 = pow(dim, i-j);
            T2.SetSize(dim, ndof*sz2);
            T2_wrk.SetDataAndSize(T2.GetData(), dim*ndof*sz2);
            T1.MultTranspose(D, T2_wrk);
            T1 = T2;
         }
         Vector q_hess_dot_d_work(ndof);
         T1.MultTranspose(D, q_hess_dot_d_work);
         q_hess_dot_d_work *= 1./Factorial(i);
         q_hess_dot_d += q_hess_dot_d_work;
      }

      wrk = shape;
      wrk += dshapephysdd;
      wrk += q_hess_dot_d;
      // <u + grad u.d + h.o.t, grad w.n>  - Term 3
      AddMult_a_VWt(-1., dshapedn, wrk, temp_elmat);

      double hinvdx;
      if (elem1f) { hinvdx = nor*nor/Trans.Elem1->Weight(); }
      else { hinvdx = nor*nor/Trans.Elem2->Weight(); }

      w = ip.weight*alpha*hinvdx;
      // + <alpha * hinv * u + grad u.d + h.o.t, w + grad w.d + h.o.t> - Term 4
      AddMult_a_VVt(w, wrk, temp_elmat);

      int offset = elem1f ? 0 : ndof1;
      elmat.CopyMN(temp_elmat, offset, offset);
   } // p < ir->GetNPoints()

   for (int i = 0; i < dkphi_dxk.Size(); i++)
   {
      delete dkphi_dxk[i];
   }
}


void SBM2DirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("SBM2DirichletLFIntegrator::AssembleRHSElementVect");
}

void SBM2DirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   AssembleRHSElementVect(el, el, Tr, elvect);
}

void SBM2DirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof1, ndof2, ndof, ndoftotal;
   double w;
   Vector temp_elvect;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   ndoftotal = ndof1 + ndof2;
   if (Tr.Elem2No >= NEproc ||
       Tr.ElementType == ElementTransformation::BDR_FACE)
   {
      ndoftotal = ndof1;
   }

   elvect.SetSize(ndoftotal);
   elvect = 0.0;

   bool elem1f = true;
   int elem1 = Tr.Elem1No,
       elem2 = Tr.Elem2No,
       marker1 = (*elem_marker)[elem1];

   int marker2;
   if (Tr.Elem2No >= NEproc)
   {
      marker2 = (*elem_marker)[NEproc+par_shared_face_count];
      par_shared_face_count++;
   }
   else if (Tr.ElementType == ElementTransformation::BDR_FACE)
   {
      marker2 = marker1;
   }
   else
   {
      marker2 = (*elem_marker)[elem2];
   }

   if (!include_cut_cell)
   {
      // 1 is inside and 2 is cut or 1 is a boundary element.
      if ( marker1 == ShiftedFaceMarker::SBElementType::INSIDE &&
           (marker2 == ShiftedFaceMarker::SBElementType::CUT
            + cut_cell_marker_offset ||
            Tr.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
         ndof = ndof1;
      }
      // 1 is cut, 2 is inside
      else if (marker1 == ShiftedFaceMarker::SBElementType::CUT
               + cut_cell_marker_offset &&
               marker2 == ShiftedFaceMarker::SBElementType::INSIDE)
      {
         if (Tr.Elem2No >= NEproc) { return; }
         elem1f = false;
         ndof = ndof2;
      }
      else
      {
         return;
      }
   }
   else
   {
      // 1 is cut and 2 is outside or 1 is a boundary element.
      if (marker1 == ShiftedFaceMarker::SBElementType::CUT
          + cut_cell_marker_offset &&
          (marker2 == ShiftedFaceMarker::SBElementType::OUTSIDE ||
           Tr.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
         ndof = ndof1;
      }
      // 1 is outside, 2 is cut
      else if (marker1 == ShiftedFaceMarker::SBElementType::OUTSIDE &&
               marker2 == ShiftedFaceMarker::SBElementType::CUT
               + cut_cell_marker_offset)
      {
         if (Tr.Elem2No >= NEproc) { return; }
         elem1f = false;
         ndof = ndof2;
      }
      else
      {
         return;
      }
   }

   temp_elvect.SetSize(ndof);
   temp_elvect = 0.0;

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dd.SetSize(ndof);
   dshape_dn.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = elem1f ? 4*el1.GetOrder() : 4*el2.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   Array<DenseMatrix *> dkphi_dxk;
   DenseMatrix grad_phys;
   Vector Factorial;
   Array<DenseMatrix *> grad_phys_dir;

   if (nterms > 0)
   {
      if (elem1f)
      {
         el1.ProjectGrad(el1, *Tr.Elem1, grad_phys);
      }
      else
      {
         el2.ProjectGrad(el2, *Tr.Elem2, grad_phys);
      }

      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrix for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(ndof, ndof);
         grad_phys_dir[i]->CopyRows(grad_phys, i*ndof, (i+1)*ndof-1);
      }


      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(ndof, ndof*dim);

      dkphi_dxk.SetSize(nterms);

      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         dkphi_dxk[i] = new DenseMatrix(ndof, ndof*sz1*dim);
         int loc_col_per_dof = sz1;
         for (int k = 0; k < dim; k++)
         {
            grad_work.SetSize(ndof, ndof*sz1);
            // grad_work[k] has derivative in kth direction for each DOF.
            // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
            // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
            if (i == 0)
            {
               Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            }
            else
            {
               Mult(*grad_phys_dir[k], *dkphi_dxk[i-1], grad_work);
            }
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
            for (int j = 0; j < ndof; j++)
            {
               for (int d = 0; d < loc_col_per_dof; d++)
               {
                  Vector col;
                  int tot_col_per_dof = loc_col_per_dof*dim;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[i]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
               }
            }
         }
      }

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }

      Factorial.SetSize(nterms);
      Factorial(0) = 2;
      for (int i = 1; i < nterms; i++)
      {
         Factorial(i) = Factorial(i-1)*(i+2);
      }
   }

   DenseMatrix q_hess_dn(dim, ndof);
   Vector q_hess_dn_work(q_hess_dn.GetData(), ndof*dim);
   Vector q_hess_dot_d(ndof);

   Vector D(vD->GetVDim());
   Vector wrk = shape;
   // Assemble: -< u_D, \nabla w.n >
   //           -<alpha h^{-1} u_D, w + \nabla w.d + h.o.t>
   for (int p = 0; p < ir->GetNPoints(); p++)
   {

      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      vD->Eval(D, Tr, ip);

      double nor_dot_d = nor*D;
      if (!include_cut_cell && nor_dot_d < 0) { nor *= -1; }
      if (include_cut_cell && nor_dot_d > 0) { nor *= -1; }
      // note here that if we are clipping outside the domain, we will have to
      // flip the sign if nor_dot_d is positive.

      double hinvdx;

      if (elem1f)
      {
         el1.CalcShape(eip1, shape);
         el1.CalcDShape(eip1, dshape);
         hinvdx =nor*nor/Tr.Elem1->Weight();
         w = ip.weight * uD->Eval(Tr, ip, D) / Tr.Elem1->Weight();
         CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      }
      else
      {
         el2.CalcShape(eip2, shape);
         el2.CalcDShape(eip2, dshape);
         hinvdx = nor*nor/Tr.Elem2->Weight();
         w = ip.weight * uD->Eval(Tr, ip, D) / Tr.Elem2->Weight();
         CalcAdjugate(Tr.Elem2->Jacobian(), adjJ);
      }

      ni.Set(w, nor);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      temp_elvect.Add(-1., dshape_dn); // T2

      double jinv;
      if (elem1f)
      {
         w = ip.weight * uD->Eval(Tr, ip, D) * alpha * hinvdx;
         jinv = 1./Tr.Elem1->Weight();
      }
      else
      {
         w = ip.weight * uD->Eval(Tr, ip, D) * alpha * hinvdx;
         jinv = 1./Tr.Elem2->Weight();
      }
      adjJ.Mult(D, nh);
      nh *= jinv;
      dshape.Mult(nh, dshape_dd);

      q_hess_dot_d = 0.;
      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         DenseMatrix T1(dim, ndof*sz1);
         Vector T1_wrk(T1.GetData(), dim*ndof*sz1);
         dkphi_dxk[i]->MultTranspose(shape, T1_wrk);

         DenseMatrix T2;
         Vector T2_wrk;
         for (int j = 0; j < i+1; j++)
         {
            int sz2 = pow(dim, i-j);
            T2.SetSize(dim, ndof*sz2);
            T2_wrk.SetDataAndSize(T2.GetData(), dim*ndof*sz2);
            T1.MultTranspose(D, T2_wrk);
            T1 = T2;
         }
         Vector q_hess_dot_d_work(ndof);
         T1.MultTranspose(D, q_hess_dot_d_work);
         q_hess_dot_d_work *= 1./Factorial(i);
         q_hess_dot_d += q_hess_dot_d_work;
      }

      wrk = shape;
      wrk += dshape_dd; // \grad w .d
      wrk += q_hess_dot_d;
      temp_elvect.Add(w, wrk); // <u, gradw.d>

      int offset = elem1f ? 0 : ndof1;
      for (int i = 0; i < temp_elvect.Size(); i++)
      {
         elvect(i+offset) = temp_elvect(i);
      }
   }

   for (int i = 0; i < dkphi_dxk.Size(); i++)
   {
      delete dkphi_dxk[i];
   }
}


void SBM2NeumannIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, ndof1, ndof2, ndof, ndoftotal;
   double w;
   DenseMatrix temp_elmat;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   ndoftotal = Trans.ElementType == ElementTransformation::BDR_FACE ?
               ndof1 : ndof1 + ndof2;

   elmat.SetSize(ndoftotal);
   elmat = 0.0;

   bool elem1f = true; // flag indicating whether Trans.Elem1No is part of the
   // surrogate domain or not.
   int elem1 = Trans.Elem1No,
       elem2 = Trans.Elem2No,
       marker1 = (*elem_marker)[elem1];

   int marker2;

   if (Trans.Elem2No >= NEproc)
   {
      marker2 = (*elem_marker)[NEproc+par_shared_face_count];
      par_shared_face_count++;
   }
   else if (Trans.ElementType == ElementTransformation::BDR_FACE)
   {
      marker2 = marker1;
   }
   else
   {
      marker2 = (*elem_marker)[elem2];
   }

   if (!include_cut_cell)
   {
      // 1 is inside and 2 is cut or 1 is a boundary element.
      if (marker1 == ShiftedFaceMarker::SBElementType::INSIDE &&
          (cut_marker.Find(marker2) != -1 ||
           Trans.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
      }
      // 1 is cut, 2 is inside
      else if (cut_marker.Find(marker1) != -1 &&
               marker2 == ShiftedFaceMarker::SBElementType::INSIDE)
      {
         if (Trans.Elem2No >= NEproc) { return; }
         elem1f = false;
      }
      else
      {
         return;
      }
   }
   else
   {
      // 1 is cut and 2 is outside or 1 is a boundary element.
      if (cut_marker.Find(marker1) != -1 &&
          (marker2 == ShiftedFaceMarker::SBElementType::OUTSIDE ||
           Trans.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
      }
      // 1 is outside, 2 is cut
      else if (marker1 == ShiftedFaceMarker::SBElementType::OUTSIDE &&
               cut_marker.Find(marker2) != -1)
      {
         if (Trans.Elem2No >= NEproc) { return; }
         elem1f = false;
      }
      else
      {
         return;
      }
   }

   ndof = elem1f ? ndof1 : ndof2;

   temp_elmat.SetSize(ndof);
   temp_elmat = 0.;

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);

   shape.SetSize(ndof1);
   dshape.SetSize(ndof1, dim);
   dshapephys.SetSize(ndof1, dim);
   dshapedn.SetSize(ndof1);
   Vector wrk = shape;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = elem1f ? 4*el1.GetOrder() : 4*el2.GetOrder();
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   MFEM_VERIFY(nterms == 1, " nterms must be 1 for Neumann.\n");

   Array<DenseMatrix *> dkphi_dxk;
   DenseMatrix grad_phys;
   Vector Factorial;
   Array<DenseMatrix *> grad_phys_dir;

   if (nterms > 0)
   {
      if (elem1f)
      {
         el1.ProjectGrad(el1, *Trans.Elem1, grad_phys);
      }
      else
      {
         el2.ProjectGrad(el2, *Trans.Elem2, grad_phys);
      }

      DenseMatrix grad_work;
      grad_phys_dir.SetSize(dim); // NxN matrices for derivative in each direction
      for (int i = 0; i < dim; i++)
      {
         grad_phys_dir[i] = new DenseMatrix(ndof, ndof);
         grad_phys_dir[i]->CopyRows(grad_phys, i*ndof, (i+1)*ndof-1);
      }

      DenseMatrix grad_phys_work = grad_phys;
      grad_phys_work.SetSize(ndof, ndof*dim);

      dkphi_dxk.SetSize(nterms);

      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         dkphi_dxk[i] = new DenseMatrix(ndof, ndof*sz1*dim);
         int loc_col_per_dof = sz1;
         int tot_col_per_dof = loc_col_per_dof*dim;
         for (int k = 0; k < dim; k++)
         {
            grad_work.SetSize(ndof, ndof*sz1);
            // grad_work[k] has derivative in kth direction for each DOF.
            // grad_work[0] has d^2phi/dx^2 and d^2phi/dxdy terms and
            // grad_work[1] has d^2phi/dydx and d^2phi/dy2 terms for each dof
            if (i == 0)
            {
               Mult(*grad_phys_dir[k], grad_phys_work, grad_work);
            }
            else
            {
               Mult(*grad_phys_dir[k], *dkphi_dxk[i-1], grad_work);
            }
            // Now we must place columns for each dof together so that they are
            // in order: d^2phi/dx^2, d^2phi/dxdy, d^2phi/dydx, d^2phi/dy2.
            for (int j = 0; j < ndof; j++)
            {
               for (int d = 0; d < loc_col_per_dof; d++)
               {
                  Vector col;
                  grad_work.GetColumn(j*loc_col_per_dof+d, col);
                  dkphi_dxk[i]->SetCol(j*tot_col_per_dof+k*loc_col_per_dof+d, col);
               }
            }
         }
      }

      for (int i = 0; i < grad_phys_dir.Size(); i++)
      {
         delete grad_phys_dir[i];
      }

      Factorial.SetSize(nterms);
      Factorial(0) = 2;
      for (int i = 1; i < nterms; i++)
      {
         Factorial(i) = Factorial(i-1)*(i+2);
      }
   }


   DenseMatrix q_hess_dn(dim, ndof1);
   Vector q_hess_dn_work(q_hess_dn.GetData(), ndof1*dim);
   Vector q_hess_dot_d(ndof1);

   Vector D(vD->GetVDim());
   Vector N(vN->GetVDim());
   // assemble: -< \nabla u.n, w >
   //           -< u + \nabla u.d + h.o.t, \nabla w.n>
   //           -<alpha h^{-1} (u + \nabla u.d + h.o.t), w + \nabla w.d + h.o.t>
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         // Note: this normal accounts for the weight of the surface transformation
         // Jacobian i.e. nor = nhat*det(J)
         CalcOrtho(Trans.Jacobian(), nor);
      }
      vD->Eval(D, Trans, ip);
      vN->Eval(N, Trans, ip, D);

      double nor_dot_d = nor*D;
      // If we are clipping inside the domain, ntilde and d vector should be
      // aligned.
      if (!include_cut_cell && nor_dot_d < 0) { nor *= -1; }
      if (include_cut_cell && nor_dot_d > 0) { nor *= -1; }

      if (elem1f)
      {
         el1.CalcShape(eip1, shape);
         el1.CalcDShape(eip1, dshape);
         w = ip.weight/Trans.Elem1->Weight();
         CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
      }
      else
      {
         el1.CalcShape(eip2, shape);
         el1.CalcDShape(eip2, dshape);
         w = ip.weight/Trans.Elem2->Weight();
         CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
      }

      ni.Set(w, nor); // alpha_k*nor/det(J)
      adjJ.Mult(ni, nh);
      dshape.Mult(nh, dshapedn); //dphi/dn * Jinv * alpha_k * nor

      // -<w, grad u.n> - Term 2
      AddMult_a_VWt(-1., shape, dshapedn, temp_elmat);

      // -<w, (grad u.nhat)nhat.n
      // Here nhat is the normal vector at true boundary and n is the normal
      // vector at shifted boundary
      ni.Set(w, N);
      adjJ.Mult(ni, nh);
      dshape.Mult(nh, dshapedn);
      dshapedn *= (nor*N);
      AddMult_a_VWt(1., shape, dshapedn, temp_elmat);

      double n_dot_ntilde = (nor*N); //nor and N are pointing in opposite direction
      q_hess_dot_d = 0.;
      for (int i = 0; i < nterms; i++)
      {
         int sz1 = pow(dim, i+1);
         DenseMatrix T1(dim, ndof1*sz1);
         Vector T1_wrk(T1.GetData(), dim*ndof1*sz1);
         dkphi_dxk[i]->MultTranspose(shape, T1_wrk);

         DenseMatrix T2;
         Vector T2_wrk;
         for (int j = 0; j < i+1; j++)
         {
            int sz2 = pow(dim, i-j);
            T2.SetSize(dim, ndof1*sz2);
            T2_wrk.SetDataAndSize(T2.GetData(), dim*ndof1*sz2);
            T1.MultTranspose(D, T2_wrk);
            T1 = T2;
         }
         Vector q_hess_dot_d_work(ndof1);
         T1.MultTranspose(N, q_hess_dot_d_work);
         q_hess_dot_d += q_hess_dot_d_work;
      }

      wrk =  q_hess_dot_d;
      wrk *= ip.weight * n_dot_ntilde;

      AddMult_a_VWt(1., shape, wrk, temp_elmat);
      int offset = elem1f ? 0 : ndof1;
      elmat.CopyMN(temp_elmat, offset, offset);
   } //p < ir->GetNPoints()

   for (int i = 0; i < dkphi_dxk.Size(); i++)
   {
      delete dkphi_dxk[i];
   }
}

void SBM2NeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("SBM2NeumannLFIntegrator::AssembleRHSElementVect");
}

void SBM2NeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   AssembleRHSElementVect(el, el, Tr, elvect);
}

void SBM2NeumannLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Tr, Vector &elvect)
{

   int dim, ndof1, ndof2, ndof, ndoftotal;
   double w;
   Vector temp_elvect;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   ndoftotal = ndof1 + ndof2;
   if (Tr.Elem2No >= NEproc ||
       Tr.ElementType == ElementTransformation::BDR_FACE)
   {
      ndoftotal = ndof1;
   }

   elvect.SetSize(ndoftotal);
   elvect = 0.0;

   bool elem1f = true;
   int elem1 = Tr.Elem1No,
       elem2 = Tr.Elem2No,
       marker1 = (*elem_marker)[elem1];

   int marker2;
   if (Tr.Elem2No >= NEproc)
   {
      marker2 = (*elem_marker)[NEproc+par_shared_face_count];
      par_shared_face_count++;
   }
   else if (Tr.ElementType == ElementTransformation::BDR_FACE)
   {
      marker2 = marker1;
   }
   else
   {
      marker2 = (*elem_marker)[elem2];
   }
   if (!include_cut_cell)
   {
      // 1 is inside and 2 is cut or 1 is a boundary element.
      if ( marker1 == ShiftedFaceMarker::SBElementType::INSIDE &&
           (marker2 == ShiftedFaceMarker::SBElementType::CUT
            + cut_cell_marker_offset ||
            Tr.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
         ndof = ndof1;
      }
      // 1 is cut, 2 is inside
      else if (marker1 == ShiftedFaceMarker::SBElementType::CUT
               + cut_cell_marker_offset &&
               marker2 == ShiftedFaceMarker::SBElementType::INSIDE)
      {
         if (Tr.Elem2No >= NEproc) { return; }
         elem1f = false;
         ndof = ndof2;
      }
      else
      {
         return;
      }
   }
   else
   {
      // 1 is cut and 2 is outside or 1 is a boundary element.
      if (marker1 == ShiftedFaceMarker::SBElementType::CUT
          + cut_cell_marker_offset &&
          (marker2 == ShiftedFaceMarker::SBElementType::OUTSIDE ||
           Tr.ElementType == ElementTransformation::BDR_FACE))
      {
         elem1f = true;
         ndof = ndof1;
      }
      // 1 is outside, 2 is cut
      else if (marker1 == ShiftedFaceMarker::SBElementType::OUTSIDE &&
               marker2 == ShiftedFaceMarker::SBElementType::CUT
               + cut_cell_marker_offset)
      {
         if (Tr.Elem2No >= NEproc) { return; }
         elem1f = false;
         ndof = ndof2;
      }
      else
      {
         return;
      }
   }

   temp_elvect.SetSize(ndof);
   temp_elvect = 0.0;


   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = elem1f ? 4*el1.GetOrder() : 4*el2.GetOrder();
      ir = &IntRules.Get(Tr.GetGeometryType(), order);
   }

   Vector D(vD->GetVDim());
   Vector N(vN->GetVDim());
   Vector wrk = shape;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring element
      Tr.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      const IntegrationPoint &eip2 = Tr.GetElement2IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Jacobian(), nor);
      }
      vD->Eval(D, Tr, ip);
      vN->Eval(N, Tr, ip, D);

      double nor_dot_d = nor*D;
      if (!include_cut_cell && nor_dot_d < 0) { nor *= -1; }
      if (include_cut_cell && nor_dot_d > 0) { nor *= -1; }
      // note here that if we are clipping outside the domain, we will have to
      // flip the sign if nor_dot_d is +ve.

      if (elem1f)
      {
         el1.CalcShape(eip1, shape);
         w = ip.weight * uN->Eval(Tr, ip, D);
      }
      else
      {
         el2.CalcShape(eip2, shape);
         w = ip.weight * uN->Eval(Tr, ip, D);
      }

      double n_dot_ntilde = (nor*N); //nor and N are pointing in opposite direction
      wrk.Set(n_dot_ntilde*w, shape);
      //<w, (nhat.n)t_n)
      temp_elvect.Add(1., wrk);

      int offset = elem1f ? 0 : ndof1;
      for (int i = 0; i < temp_elvect.Size(); i++)
      {
         elvect(i+offset) = temp_elvect(i);
      }
   }
}

}
