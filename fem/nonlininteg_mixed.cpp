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

#include "fem.hpp"

namespace mfem
{

real_t mfem::LinearDiffusionFlux::ComputeDualFlux(
   const Vector &, const DenseMatrix &flux, ElementTransformation &Tr,
   DenseMatrix &dualFlux) const
{
   if (coeff)
   {
      const real_t ikappa = coeff->Eval(Tr, Tr.GetIntPoint());
      dualFlux.Set(ikappa, flux);
      return ikappa;
   }
   else if (vcoeff)
   {
      Vector ikappa(dim);
      vcoeff->Eval(ikappa, Tr, Tr.GetIntPoint());
      dualFlux = flux;
      dualFlux.RightScaling(ikappa);
      return ikappa.Normlinf();
   }
   else if (mcoeff)
   {
      DenseMatrix ikappa(dim);
      mcoeff->Eval(ikappa, Tr, Tr.GetIntPoint());
      MultABt(flux, ikappa, dualFlux);
      return ikappa.MaxMaxNorm();
   }
   return 0.;
}

real_t LinearDiffusionFlux::ComputeFlux(
   const Vector &, ElementTransformation &, DenseMatrix &flux) const
{
   flux = 0.;
   return 0.;
}

void LinearDiffusionFlux::ComputeDualFluxJacobian(
   const Vector &, const DenseMatrix &flux, ElementTransformation &Tr,
   DenseMatrix &J_u, DenseMatrix &J_F) const
{
   J_u.SetSize(dim, 1);
   J_u = 0.;

   if (coeff)
   {
      const real_t ikappa = coeff->Eval(Tr, Tr.GetIntPoint());
      J_F.Diag(ikappa, dim);
   }
   else if (vcoeff)
   {
      Vector ikappa(dim);
      vcoeff->Eval(ikappa, Tr, Tr.GetIntPoint());
      J_F.Diag(ikappa);
   }
   else if (mcoeff)
   {
      mcoeff->Eval(J_F, Tr, Tr.GetIntPoint());
   }
}

real_t mfem::FunctionDiffusionFlux::ComputeDualFlux(
   const Vector &u, const DenseMatrix &flux, ElementTransformation &Tr,
   DenseMatrix &dualFlux) const
{
   Vector x(3);
   Tr.Transform(Tr.GetIntPoint(), x);

   if (func)
   {
      const real_t ikappa = func(x, u(0));
      dualFlux.Set(ikappa, flux);
      return ikappa;
   }
   else if (func_vec)
   {
      Vector ikappa(dim);
      func_vec(x, u(0), ikappa);
      dualFlux = flux;
      dualFlux.RightScaling(ikappa);
      return ikappa.Normlinf();
   }
   else if (func_mat)
   {
      DenseMatrix ikappa(dim);
      func_mat(x, u(0), ikappa);
      MultABt(flux, ikappa, dualFlux);
      return ikappa.MaxMaxNorm();
   }
   return 0.;
}

real_t FunctionDiffusionFlux::ComputeFlux(
   const Vector &, ElementTransformation &, DenseMatrix &flux) const
{
   flux = 0.;
   return 0.;
}

void FunctionDiffusionFlux::ComputeDualFluxJacobian(
   const Vector &u, const DenseMatrix &flux, ElementTransformation &Tr,
   DenseMatrix &J_u, DenseMatrix &J_F) const
{
   Vector x(3);
   Tr.Transform(Tr.GetIntPoint(), x);

   J_u.SetSize(dim, 1);

   if (func)
   {
      const real_t ikappa = func(x, u(0));
      J_F.Diag(ikappa, dim);

      const real_t dikappa = dfunc(x, u(0));
      for (int i = 0; i < dim; i++)
      {
         J_u(i,0) = dikappa * flux(0,i);
      }
   }
   else if (func_vec)
   {
      Vector ikappa(dim);
      func_vec(x, u(0), ikappa);
      J_F.Diag(ikappa);

      Vector dikappa(dim);
      dfunc_vec(x, u(0), dikappa);
      for (int i = 0; i < dim; i++)
      {
         J_u(i,0) = dikappa(i) * flux(0,i);
      }
   }
   else if (func_mat)
   {
      func_mat(x, u(0), J_F);

      DenseMatrix dikappa(dim);
      dfunc_mat(x, u(0), dikappa);
      MultABt(dikappa, flux, J_u);
   }
}

void MixedConductionNLFIntegrator::AssembleElementVector(
   const Array<const FiniteElement*> &el, ElementTransformation &Tr,
   const Array<const Vector*> &elfun, const Array<Vector*> &elvect)
{
   const FiniteElement &fe_u = *el[0];
   const FiniteElement &fe_p = *el[1];
   const int ndof_u = fe_u.GetDof();
   const int ndof_p = fe_p.GetDof();
   const int sdim = Tr.GetSpaceDim();
   const bool scalar_u = (fe_u.GetRangeType() == FiniteElement::SCALAR);

   const Vector &elfun_u = *elfun[0];
   const Vector &elfun_p = *elfun[1];
   Vector &elvect_u = *elvect[0];

   // The number of equations comes from the flux function; the spaces must
   // agree with it. Equation e occupies one contiguous block of each vector,
   // which is the Ordering::byNODES layout of a space of vdim = neq (or of
   // vdim = neq * sdim for a scalar-valued flux space).
   const int neq = fluxFunction.num_equations;
   const int nvdof_u = (scalar_u ? sdim : 1) * ndof_u;

   MFEM_ASSERT(elfun_p.Size() == neq * ndof_p,
               "The potential space must have vdim = " << neq << ".");
   MFEM_ASSERT(elfun_u.Size() == neq * nvdof_u,
               "The flux space must have vdim = " << neq * (scalar_u ? sdim : 1)
               << ".");

   shape_p.SetSize(ndof_p);

   if (elvect[1]) { elvect[1]->SetSize(0); } // not used

   Vector x(sdim), p(neq), ue(sdim), Fe(sdim);
   DenseMatrix mu(neq, sdim), mF(neq, sdim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*fe_u.GetOrder() + Tr.OrderW();//<---
      ir = &IntRules.Get(fe_u.GetGeomType(), order);
   }

   elvect_u.SetSize(neq * nvdof_u);
   elvect_u = 0.0;

   if (scalar_u) { shape_u.SetSize(ndof_u); }
   else { vshape_u.SetSize(ndof_u, sdim); }

   for (int q = 0; q < ir->Size(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      Tr.Transform(ip, x);

      fe_p.CalcShape(ip, shape_p);
      for (int e = 0; e < neq; e++)
      {
         const Vector p_e(elfun_p.GetData() + e*ndof_p, ndof_p);
         p(e) = p_e * shape_p;
      }

      const real_t w = ip.weight * Tr.Weight();

      if (scalar_u)
      {
         fe_u.CalcShape(ip, shape_u);
         for (int e = 0; e < neq; e++)
            for (int d = 0; d < sdim; d++)
            {
               const Vector u_ed(elfun_u.GetData() + (e*sdim + d)*ndof_u,
                                 ndof_u);
               mu(e, d) = u_ed * shape_u;
            }
      }
      else
      {
         fe_u.CalcVShape(Tr, vshape_u);
         for (int e = 0; e < neq; e++)
         {
            const Vector u_e(elfun_u.GetData() + e*ndof_u, ndof_u);
            vshape_u.MultTranspose(u_e, ue);
            for (int d = 0; d < sdim; d++) { mu(e, d) = ue(d); }
         }
      }

      fluxFunction.ComputeDualFlux(p, mu, Tr, mF);
      MFEM_ASSERT(mF.Height() == neq && mF.Width() == sdim,
                  "The dual flux must be num_equations by dim.");

      if (scalar_u)
      {
         for (int e = 0; e < neq; e++)
            for (int d = 0; d < sdim; d++)
               for (int i = 0; i < ndof_u; i++)
               {
                  elvect_u((e*sdim + d)*ndof_u + i) += w * shape_u(i) * mF(e, d);
               }
      }
      else
      {
         for (int e = 0; e < neq; e++)
         {
            for (int d = 0; d < sdim; d++) { Fe(d) = mF(e, d); }
            Vector ev_e(elvect_u.GetData() + e*ndof_u, ndof_u);
            vshape_u.AddMult_a(w, Fe, ev_e);
         }
      }
   }
}

void MixedConductionNLFIntegrator::AssembleFaceVector(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Trans, const Array<const Vector *> &elfun,
   const Array<Vector *> &elvect)
{
   // The face terms are still single-equation. Generalizing them is not the
   // index bookkeeping the element terms were: the HDG stabilization here is
   // built from the inverse of the flux Jacobian contracted with the face
   // normal, and what that should be for a system -- a matrix tau coupling the
   // equations, or one scalar per equation -- is a question about the
   // formulation rather than about the code. Refuse rather than assemble the
   // first equation and silently ignore the rest.
   MFEM_VERIFY(fluxFunction.num_equations == 1,
               "MixedConductionNLFIntegrator face terms are implemented for a "
               "single equation only; the element terms support "
               << fluxFunction.num_equations << ".");

   const FiniteElement &el1_u = *el1[0];
   const FiniteElement &el2_u = *el2[0];
   const FiniteElement &el1_p = *el1[1];
   const FiniteElement &el2_p = *el2[1];
   const int dim = el1_p.GetDim();
   const int ndof1_u = el1_u.GetDof();
   const int ndof2_u = (Trans.Elem2No >= 0)?(el2_u.GetDof()):(0);
   const int ndof1_p = el1_p.GetDof();
   const int ndof2_p = (Trans.Elem2No >= 0)?(el2_p.GetDof()):(0);

   DenseMatrix J_u, J_F;
   DenseMatrixInverse J_Fi;
   Vector nor(dim), nh(dim), ni(dim);

   shape1.SetSize(ndof1_p);
   shape2.SetSize(ndof2_p);

   const Vector elfun1_u(const_cast<Vector&>(*elfun[0]), 0, ndof1_u * dim);
   const Vector elfun2_u(const_cast<Vector&>(*elfun[0]), ndof1_u * dim,
                         ndof2_u * dim);
   DenseMatrix u1(1, dim), u2(1, dim);
   u1 = 0.;
   u2 = 0.;

   const Vector elfun1_p(const_cast<Vector&>(*elfun[1]), 0, ndof1_p);
   const Vector elfun2_p(const_cast<Vector&>(*elfun[1]), ndof1_p, ndof2_p);
   Vector p1(1), p2(1);

   const int ndofs_u = (ndof1_u + ndof2_u) * dim;
   Vector &elvect_u = *elvect[0];
   elvect_u.SetSize(ndofs_u);
   elvect_u = 0.0;

   const int ndofs_p = ndof1_p + ndof2_p;
   Vector &elvect_p = *elvect[1];
   elvect_p.SetSize(ndofs_p);
   elvect_p = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2_p)
      {
         order = 2*std::max(el1_p.GetOrder(), el2_p.GetOrder());
      }
      else
      {
         order = 2*el1_p.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1_p.CalcPhysShape(*Trans.Elem1, shape1);
      real_t w = ip.weight/Trans.Elem1->Weight();
      if (ndof2_p)
      {
         w /= 2;
      }

      p1(0) = shape1 * elfun1_p;

      nh.Set(w, nor);
      fluxFunction.ComputeDualFluxJacobian(p1, u1, Trans, J_u, J_F);
      J_Fi.Factor(J_F);
      J_Fi.Mult(nh, ni);

      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (ndof2_p)
      {
         el2_p.CalcPhysShape(*Trans.Elem2, shape2);
         w = ip.weight/2/Trans.Elem2->Weight();

         p2(0) = shape2 * elfun2_p;

         nh.Set(w, nor);
         fluxFunction.ComputeDualFluxJacobian(p2, u2, Trans, J_u, J_F);
         J_Fi.Factor(J_F);
         J_Fi.Mult(nh, ni);
         wq += ni * nor;
      }

      wq *= 0.5 * beta;

      for (int i = 0; i < ndof1_p; i++)
      {
         elvect_p(i) += wq * shape1(i) * p1(0);
      }
      if (ndof2_p)
      {
         for (int i = 0; i < ndof1_p; i++)
         {
            elvect_p(i) -= wq * shape1(i) * p2(0);
         }
         for (int i = 0; i < ndof2_p; i++)
         {
            elvect_p(ndof1_p + i) -= wq * shape2(i) * p1(0);
            elvect_p(ndof1_p + i) += wq * shape2(i) * p2(0);
         }
      }
   }
}

void MixedConductionNLFIntegrator::AssembleElementGrad(
   const Array<const FiniteElement *> &el, ElementTransformation &Tr,
   const Array<const Vector *> &elfun, const Array2D<DenseMatrix *> &elmats)
{
   const FiniteElement &fe_u = *el[0];
   const FiniteElement &fe_p = *el[1];
   const int ndof_u = fe_u.GetDof();
   const int ndof_p = fe_p.GetDof();
   const int sdim = Tr.GetSpaceDim();
   const bool scalar_u = (fe_u.GetRangeType() == FiniteElement::SCALAR);
   const int nvdof_u = (scalar_u ? sdim : 1) * ndof_u;

   const Vector &elfun_u = *elfun[0];
   const Vector &elfun_p = *elfun[1];

   const int neq = fluxFunction.num_equations;

   MFEM_ASSERT(elfun_p.Size() == neq * ndof_p,
               "The potential space must have vdim = " << neq << ".");
   MFEM_ASSERT(elfun_u.Size() == neq * nvdof_u,
               "The flux space must have vdim = " << neq * (scalar_u ? sdim : 1)
               << ".");

   shape_p.SetSize(ndof_p);

   if (elmats(1,1)) { elmats(1,1)->SetSize(0); } // not used
   if (elmats(0,0))
   {
      elmats(0,0)->SetSize(neq * nvdof_u);
      *elmats(0,0) = 0.0;
   }
   if (elmats(0,1))
   {
      elmats(0,1)->SetSize(neq * nvdof_u, neq * ndof_p);
      *elmats(0,1) = 0.0;
   }
   if (elmats(1,0)) { elmats(1,0)->SetSize(0); } // not used

   // The dual flux is num_equations by dim, so its derivative with respect to
   // the state is (neq*sdim) by neq and with respect to the flux is
   // (neq*sdim) squared, both indexed equation-major: row e*sdim + d. For one
   // equation this is the (sdim,1) and (sdim,sdim) pair it has always been.
   DenseMatrix J_u(neq*sdim, neq), J_F(neq*sdim, neq*sdim);
   Vector x(sdim), p(neq), ue(sdim);
   DenseMatrix mu(neq, sdim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*fe_u.GetOrder() + Tr.OrderW();//<---
      ir = &IntRules.Get(fe_u.GetGeomType(), order);
   }

   if (scalar_u) { shape_u.SetSize(ndof_u); }
   else { vshape_u.SetSize(ndof_u, sdim); }

   // Scratch for the blocked H(div) form.
   DenseMatrix J_Fb(sdim, sdim), vshapeJ_u(sdim, ndof_u), block(ndof_u, ndof_u);
   Vector J_ub(sdim), vshapeJu(ndof_u);
   DenseMatrix blockp(ndof_u, ndof_p);

   for (int q = 0; q < ir->Size(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      Tr.Transform(ip, x);

      fe_p.CalcShape(ip, shape_p);
      for (int e = 0; e < neq; e++)
      {
         const Vector p_e(elfun_p.GetData() + e*ndof_p, ndof_p);
         p(e) = p_e * shape_p;
      }

      if (scalar_u)
      {
         fe_u.CalcShape(ip, shape_u);
         for (int e = 0; e < neq; e++)
            for (int d = 0; d < sdim; d++)
            {
               const Vector u_ed(elfun_u.GetData() + (e*sdim + d)*ndof_u,
                                 ndof_u);
               mu(e, d) = u_ed * shape_u;
            }
      }
      else
      {
         fe_u.CalcVShape(Tr, vshape_u);
         for (int e = 0; e < neq; e++)
         {
            const Vector u_e(elfun_u.GetData() + e*ndof_u, ndof_u);
            vshape_u.MultTranspose(u_e, ue);
            for (int d = 0; d < sdim; d++) { mu(e, d) = ue(d); }
         }
      }

      fluxFunction.ComputeDualFluxJacobian(p, mu, Tr, J_u, J_F);
      MFEM_ASSERT(J_F.Height() == neq*sdim && J_F.Width() == neq*sdim,
                  "J_F must be (num_equations*dim) squared.");
      MFEM_ASSERT(J_u.Height() == neq*sdim && J_u.Width() == neq,
                  "J_u must be (num_equations*dim) by num_equations.");

      const real_t w = ip.weight * Tr.Weight();

      if (scalar_u)
      {
         if (elmats(0,0))
         {
            for (int e_j = 0; e_j < neq; e_j++)
               for (int d_j = 0; d_j < sdim; d_j++)
                  for (int e_i = 0; e_i < neq; e_i++)
                     for (int d_i = 0; d_i < sdim; d_i++)
                     {
                        const real_t a =
                           w * J_F(e_i*sdim + d_i, e_j*sdim + d_j);
                        if (a == 0.0) { continue; }
                        for (int j = 0; j < ndof_u; j++)
                           for (int i = 0; i < ndof_u; i++)
                           {
                              (*elmats(0,0))((e_i*sdim + d_i)*ndof_u + i,
                                             (e_j*sdim + d_j)*ndof_u + j)
                              += a * shape_u(i) * shape_u(j);
                           }
                     }
         }

         if (elmats(0,1))
         {
            for (int e_j = 0; e_j < neq; e_j++)
               for (int e_i = 0; e_i < neq; e_i++)
                  for (int d_i = 0; d_i < sdim; d_i++)
                  {
                     const real_t a = w * J_u(e_i*sdim + d_i, e_j);
                     if (a == 0.0) { continue; }
                     for (int j = 0; j < ndof_p; j++)
                        for (int i = 0; i < ndof_u; i++)
                        {
                           (*elmats(0,1))((e_i*sdim + d_i)*ndof_u + i,
                                          e_j*ndof_p + j)
                           += a * shape_u(i) * shape_p(j);
                        }
                  }
         }
      }
      else
      {
         if (elmats(0,0))
         {
            for (int e_i = 0; e_i < neq; e_i++)
               for (int e_j = 0; e_j < neq; e_j++)
               {
                  J_Fb.CopyMN(J_F, sdim, sdim, e_i*sdim, e_j*sdim);
                  MultABt(J_Fb, vshape_u, vshapeJ_u);
                  block = 0.0;
                  AddMult_a(w, vshape_u, vshapeJ_u, block);
                  elmats(0,0)->AddMatrix(1.0, block, e_i*ndof_u, e_j*ndof_u);
               }
         }

         if (elmats(0,1))
         {
            for (int e_i = 0; e_i < neq; e_i++)
               for (int e_j = 0; e_j < neq; e_j++)
               {
                  for (int d = 0; d < sdim; d++)
                  {
                     J_ub(d) = J_u(e_i*sdim + d, e_j);
                  }
                  vshape_u.Mult(J_ub, vshapeJu);
                  blockp = 0.0;
                  AddMult_a_VWt(w, vshapeJu, shape_p, blockp);
                  elmats(0,1)->AddMatrix(1.0, blockp, e_i*ndof_u, e_j*ndof_p);
               }
         }
      }
   }
}

void MixedConductionNLFIntegrator::AssembleFaceGrad(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Trans, const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   // The face terms are still single-equation. Generalizing them is not the
   // index bookkeeping the element terms were: the HDG stabilization here is
   // built from the inverse of the flux Jacobian contracted with the face
   // normal, and what that should be for a system -- a matrix tau coupling the
   // equations, or one scalar per equation -- is a question about the
   // formulation rather than about the code. Refuse rather than assemble the
   // first equation and silently ignore the rest.
   MFEM_VERIFY(fluxFunction.num_equations == 1,
               "MixedConductionNLFIntegrator face terms are implemented for a "
               "single equation only; the element terms support "
               << fluxFunction.num_equations << ".");

   const FiniteElement &el1_u = *el1[0];
   const FiniteElement &el2_u = *el2[0];
   const FiniteElement &el1_p = *el1[1];
   const FiniteElement &el2_p = *el2[1];
   const int dim = el1_p.GetDim();
   const int ndof1_u = el1_u.GetDof();
   const int ndof2_u = (Trans.Elem2No >= 0)?(el2_u.GetDof()):(0);
   const int ndof1_p = el1_p.GetDof();
   const int ndof2_p = (Trans.Elem2No >= 0)?(el2_p.GetDof()):(0);

   DenseMatrix J_u, J_F;
   DenseMatrixInverse J_Fi;
   Vector nor(dim), nh(dim), ni(dim);

   shape1.SetSize(ndof1_p);
   shape2.SetSize(ndof2_p);

   const Vector elfun1_u(const_cast<Vector&>(*elfun[0]), 0, ndof1_u * dim);
   const Vector elfun2_u(const_cast<Vector&>(*elfun[0]), ndof1_u * dim,
                         ndof2_u * dim);
   DenseMatrix u1(1, dim), u2(1, dim);
   u1 = 0.;
   u2 = 0.;

   const Vector elfun1_p(const_cast<Vector&>(*elfun[1]), 0, ndof1_p);
   const Vector elfun2_p(const_cast<Vector&>(*elfun[1]), ndof1_p, ndof2_p);
   Vector p1(1), p2(1);

   // not used
   if (elmats(0,0)) { elmats(0,0)->SetSize(0); }
   if (elmats(1,0)) { elmats(1,0)->SetSize(0); }
   if (elmats(0,1)) { elmats(0,1)->SetSize(0); }

   const int ndofs_p = ndof1_p + ndof2_p;
   DenseMatrix &elmat_p = *elmats(1,1);
   elmat_p.SetSize(ndofs_p);
   elmat_p = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order;
      if (ndof2_p)
      {
         order = 2*std::max(el1_p.GetOrder(), el2_p.GetOrder());
      }
      else
      {
         order = 2*el1_p.GetOrder();
      }
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      el1_p.CalcPhysShape(*Trans.Elem1, shape1);
      real_t w = ip.weight/Trans.Elem1->Weight();
      if (ndof2_p)
      {
         w /= 2;
      }

      p1(0) = shape1 * elfun1_p;

      nh.Set(w, nor);
      fluxFunction.ComputeDualFluxJacobian(p1, u1, Trans, J_u, J_F);
      J_Fi.Factor(J_F);
      J_Fi.Mult(nh, ni);

      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (ndof2_p)
      {
         el2_p.CalcPhysShape(*Trans.Elem2, shape2);
         w = ip.weight/2/Trans.Elem2->Weight();

         p2(0) = shape2 * elfun2_p;

         nh.Set(w, nor);
         fluxFunction.ComputeDualFluxJacobian(p2, u2, Trans, J_u, J_F);
         J_Fi.Factor(J_F);
         J_Fi.Mult(nh, ni);
         wq += ni * nor;
      }

      wq *= 0.5 * beta;

      // only assemble the lower triangular part
      for (int i = 0; i < ndof1_p; i++)
      {
         const real_t wsi = wq * shape1(i);
         for (int j = 0; j <= i; j++)
         {
            elmat_p(i, j) += wsi * shape1(j);
         }
      }
      if (ndof2_p)
      {
         for (int i = 0; i < ndof2_p; i++)
         {
            const int i2 = ndof1_p + i;
            const real_t wsi = wq * shape2(i);
            for (int j = 0; j < ndof1_p; j++)
            {
               elmat_p(i2, j) -= wsi * shape1(j);
            }
            for (int j = 0; j <= i; j++)
            {
               elmat_p(i2, ndof1_p + j) += wsi * shape2(j);
            }
         }
      }

   }

   // complete the upper triangular part
   for (int i = 0; i < ndofs_p; i++)
      for (int j = 0; j < i; j++)
      {
         elmat_p(j,i) = elmat_p(i,j);
      }
}

void mfem::MixedConductionNLFIntegrator::AssembleHDGFaceVector(
   int type, const FiniteElement &trace_el,
   const Array<const FiniteElement *> &el,
   FaceElementTransformations &Trans,
   const Vector &trfun, const Array<const Vector *> &elfun,
   const Array<Vector *> &elvect)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }
   ElementTransformation *Elem = (type & 1)?(Trans.Elem2):(Trans.Elem1);

   const FiniteElement &el_u = *el[0];
   const FiniteElement &el_p = *el[1];
   const Vector &elfun_p = *elfun[1];

   const int dim = el_p.GetDim();
   const int ndof_tr = trace_el.GetDof();
   const int ndof_u = el_p.GetDof();
   const int ndof_p = el_p.GetDof();

   DenseMatrix J_u, J_F;
   DenseMatrixInverse J_Fi;

   DenseMatrix u(1, dim);
   u = 0.;

   Vector p(1);
   real_t a, b;

   Vector vu(dim), nor(dim), nh(dim), ni(dim);

   shape_tr.SetSize(ndof_tr);
   shape_u.SetSize(ndof_u);
   shape_p.SetSize(ndof_p);

   Vector *elvect_u{}, *elvect_p{}, *elvect_tr{};
   if (type & (HDGFaceType::ELEM | HDGFaceType::TRACE))
   {
      elvect_u = elvect[0];
      if (elvect_u)
      {
         elvect_u->SetSize(0);//not used
      }
      elvect_p = elvect[1];
      if (elvect_p)
      {
         elvect_p->SetSize(ndof_p);
         *elvect_p = 0.;
      }
   }
   if (type & (HDGFaceType::CONSTR | HDGFaceType::FACE))
   {
      elvect_tr = elvect[2];
      if (elvect_tr)
      {
         elvect_tr->SetSize(ndof_tr);
         *elvect_tr = 0.;
      }
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * (el_u.GetOrder(), el_p.GetOrder());
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      trace_el.CalcShape(ip, shape_tr);
      const real_t tr = trfun * shape_tr;

      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }
      if (type & 1) { un *= -1.; }

      el_p.CalcPhysShape(*Trans.Elem1, shape_p);
      real_t w = ip.weight / Elem->Weight();

      p(0) = elfun_p * shape_p;

      nh.Set(w, nor);
      fluxFunction.ComputeDualFluxJacobian(p, u, Trans, J_u, J_F);
      J_Fi.Factor(J_F);
      J_Fi.Mult(nh, ni);

      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      /*if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else*/
      {
         a = 0.0;
         b = beta;
      }

      w = wq * (b+a);
      if (w == 0.) { continue; }

      if (elvect_p)
      {
         real_t wp = 0.;
         if (type & HDGFaceType::ELEM) { wp += p(0); }
         if (type & HDGFaceType::TRACE) { wp -= tr; }

         // assemble the element vector
         elvect_p->Add(w * wp, shape_p);
      }

      if (elvect_tr)
      {
         real_t wtr = 0.;
         if (type & HDGFaceType::CONSTR) { wtr += p(0); }
         if (type & HDGFaceType::FACE) { wtr -= tr; }

         // assemble the trace vector
         elvect_tr->Add(w * wtr, shape_tr);
      }
   }
}

void mfem::MixedConductionNLFIntegrator::AssembleHDGFaceGrad(
   int type, const FiniteElement &trace_el,
   const Array<const FiniteElement *> &el,
   FaceElementTransformations &Trans,
   const Vector &trfun, const Array<const Vector *> &elfun,
   const Array2D<DenseMatrix *> &elmats)
{
   MFEM_VERIFY(trace_el.GetMapType() == FiniteElement::VALUE, "");

   if (Trans.Elem2No < 0) { type &= ~1; }
   ElementTransformation *Elem = (type & 1)?(Trans.Elem2):(Trans.Elem1);

   const FiniteElement &el_u = *el[0];
   const FiniteElement &el_p = *el[1];
   const Vector &elfun_p = *elfun[1];

   const int dim = el_p.GetDim();
   const int ndof_tr = trace_el.GetDof();
   const int ndof_u = el_p.GetDof();
   const int ndof_p = el_p.GetDof();

   DenseMatrix J_u, J_F;
   DenseMatrixInverse J_Fi;

   DenseMatrix u(1, dim);
   u = 0.;

   Vector p(1);
   real_t a, b;

   Vector vu(dim), nor(dim), nh(dim), ni(dim);

   shape_tr.SetSize(ndof_tr);
   shape_u.SetSize(ndof_u);
   shape_p.SetSize(ndof_p);

   DenseMatrix *elmat_A{}, *elmat_D{}, *elmat_E{}, *elmat_G{}, *elmat_H{};
   if (type & (HDGFaceType::ELEM))
   {
      elmat_A = elmats(0,0);
      if (elmat_A)
      {
         elmat_A->SetSize(0);//not used
      }
      elmat_D = elmats(1,1);
      if (elmat_D)
      {
         elmat_D->SetSize(ndof_p);
         *elmat_D = 0.;
      }
   }
   if (type & (HDGFaceType::TRACE))
   {
      elmat_E = elmats(1,2);
      if (elmat_E)
      {
         elmat_E->SetSize(ndof_p, ndof_tr);
         *elmat_E = 0.;
      }
   }
   if (type & (HDGFaceType::CONSTR))
   {
      elmat_G = elmats(2,1);
      if (elmat_G)
      {
         elmat_G->SetSize(ndof_tr, ndof_p);
         *elmat_G = 0.;
      }
   }
   if (type & (HDGFaceType::FACE))
   {
      elmat_H = elmats(2,2);
      if (elmat_H)
      {
         elmat_H->SetSize(ndof_tr);
         *elmat_H = 0.;
      }
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      const int order = 2 * (el_u.GetOrder(), el_p.GetOrder());
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // assemble: alpha < {h^{-1} Q} [u],[v] >
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Access the neighboring element's integration point
      const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Jacobian(), nor);
      }

      trace_el.CalcShape(ip, shape_tr);

      real_t un;
      if (v)
      {
         v->Eval(vu, *Trans.Elem1, eip1);
         un = vu * nor;
      }
      else
      {
         un = 0.0;
      }
      if (type & 1) { un *= -1.; }

      el_p.CalcPhysShape(*Trans.Elem1, shape_p);
      real_t w = ip.weight / Elem->Weight();

      p(0) = elfun_p * shape_p;

      nh.Set(w, nor);
      fluxFunction.ComputeDualFluxJacobian(p, u, Trans, J_u, J_F);
      J_Fi.Factor(J_F);
      J_Fi.Mult(nh, ni);

      real_t wq = ni * nor;
      // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
      // independent of Loc1 and always gives the size of element 1 in
      // direction perpendicular to the face. Indeed, for linear transformation
      //     |nor|=measure(face)/measure(ref. face),
      //   det(J1)=measure(element)/measure(ref. element),
      // and the ratios measure(ref. element)/measure(ref. face) are
      // compatible for all element/face pairs.
      // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
      // for any tetrahedron vol(tet)=(1/3)*height*area(base).
      // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

      if (un != 0.)
      {
         un /= fabs(un);
         a = 0.5 * alpha * un;
         b = beta * fabs(un);
      }
      else
      {
         a = 0.0;
         b = beta;
      }

      w = wq * (b+a);
      if (w == 0.) { continue; }

      if (elmat_D)
      {
         // assemble the element matrix
         AddMult_a_VVt(+w, shape_p, *elmat_D);
      }
      if (elmat_E)
      {
         // assemble the trace matrix
         AddMult_a_VWt(-w, shape_p, shape_tr, *elmat_E);
      }
      if (elmat_G)
      {
         // assemble the constraint matrix
         AddMult_a_VWt(+w, shape_tr, shape_p, *elmat_G);
      }
      if (elmat_H)
      {
         // assemble the face matrix
         AddMult_a_VVt(-w, shape_tr, *elmat_H);
      }
   }
}
}
