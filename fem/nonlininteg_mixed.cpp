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
      dualFlux.LeftScaling(ikappa);
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
      dualFlux.LeftScaling(ikappa);
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
   const int ndof_u = el[0]->GetDof();
   const int ndof_p = el[1]->GetDof();
   const int sdim = Tr.GetSpaceDim();

   const FiniteElement &fe_u = *el[0];
   const FiniteElement &fe_p = *el[1];
   const Vector &elfun_u = *elfun[0];
   const Vector &elfun_p = *elfun[1];
   Vector &elvect_u = *elvect[0];

   shape_p.SetSize(ndof_p);

   if (elvect[1]) { elvect[1]->SetSize(0); } // not used

   Vector x(sdim), u(sdim), F(sdim), p(1);
   DenseMatrix mu(u.GetData(), 1, sdim), mF(F.GetData(), 1, sdim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*fe_u.GetOrder() + Tr.OrderW();//<---
      ir = &IntRules.Get(fe_u.GetGeomType(), order);
   }

   if (fe_u.GetRangeType() == FiniteElement::SCALAR)
   {
      shape_u.SetSize(ndof_u);
      elvect_u.SetSize(ndof_u * sdim);
      elvect_u = 0.0;

      DenseMatrix elfun_u_mat(elfun_u.GetData(), ndof_u, sdim);

      for (int q = 0; q < ir->Size(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, x);

         fe_u.CalcShape(ip, shape_u);
         fe_p.CalcShape(ip, shape_p);

         p(0) = elfun_p * shape_p;
         real_t w = ip.weight * Tr.Weight();

         elfun_u_mat.MultTranspose(shape_u, u);

         fluxFunction.ComputeDualFlux(p, mu, Tr, mF);

         for (int d = 0; d < sdim; d++)
            for (int i = 0; i < ndof_u; i++)
            {
               elvect_u(i+d*ndof_u) += w * shape_u(i) * F(d);
            }
      }
   }
   else
   {
      vshape_u.SetSize(ndof_u, sdim);
      elvect_u.SetSize(ndof_u);
      elvect_u = 0.0;

      for (int q = 0; q < ir->Size(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, x);

         fe_u.CalcVShape(Tr, vshape_u);
         fe_p.CalcShape(ip, shape_p);

         p(0) = elfun_p * shape_p;
         real_t w = ip.weight * Tr.Weight();

         vshape_u.MultTranspose(elfun_u, u);

         fluxFunction.ComputeDualFlux(p, mu, Tr, mF);

         vshape_u.AddMult_a(w, F, elvect_u);
      }
   }
}

void MixedConductionNLFIntegrator::AssembleFaceVector(
   const Array<const FiniteElement *> &el1,
   const Array<const FiniteElement *> &el2,
   FaceElementTransformations &Trans, const Array<const Vector *> &elfun,
   const Array<Vector *> &elvect)
{
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
   const int nvdof_u = ((fe_u.GetRangeType() == FiniteElement::SCALAR)?(sdim):
                        (1)) * ndof_u;

   const Vector &elfun_u = *elfun[0];
   const Vector &elfun_p = *elfun[1];

   shape_p.SetSize(ndof_p);

   if (elmats(1,1)) { elmats(1,1)->SetSize(0); } // not used
   if (elmats(0,0))
   {
      elmats(0,0)->SetSize(nvdof_u);
      *elmats(0,0) = 0.0;
   }
   if (elmats(0,1))
   {
      elmats(0,1)->SetSize(nvdof_u, ndof_p);
      *elmats(0,1) = 0.0;
   }
   if (elmats(1,0)) { elmats(1,0)->SetSize(0); } // not used

   DenseMatrix J_u(sdim, 1), J_F(sdim);
   Vector x(sdim), u(sdim), p(1);
   DenseMatrix mu(u.GetData(), 1, sdim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*fe_u.GetOrder() + Tr.OrderW();//<---
      ir = &IntRules.Get(fe_u.GetGeomType(), order);
   }

   if (fe_u.GetRangeType() == FiniteElement::SCALAR)
   {
      shape_u.SetSize(ndof_u);

      DenseMatrix elfun_u_mat(elfun_u.GetData(), ndof_u, sdim);

      for (int q = 0; q < ir->Size(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, x);

         fe_u.CalcShape(ip, shape_u);
         fe_p.CalcShape(ip, shape_p);

         p(0) = elfun_p * shape_p;
         elfun_u_mat.MultTranspose(shape_u, u);

         fluxFunction.ComputeDualFluxJacobian(p, mu, Tr, J_u, J_F);

         real_t w = ip.weight * Tr.Weight();

         if (elmats(0,0))
         {
            for (int d_j = 0; d_j < sdim; d_j++)
               for (int d_i = 0; d_i < sdim; d_i++)
                  for (int j = 0; j < ndof_u; j++)
                     for (int i = 0; i < ndof_u; i++)
                     {
                        (*elmats(0,0))(i+d_i*ndof_u, j+d_j*ndof_u)
                        += w * J_F(d_i,d_j) * shape_u(i) * shape_u(j);
                     }
         }

         if (elmats(0,1))
         {
            for (int d_i = 0; d_i < sdim; d_i++)
               for (int j = 0; j < ndof_p; j++)
                  for (int i = 0; i < ndof_u; i++)
                  {
                     (*elmats(0,1))(i+d_i*ndof_u, j)
                     += w * J_u(d_i,0) * shape_u(i) * shape_p(j);
                  }
         }
      }
   }
   else
   {
      vshape_u.SetSize(ndof_u, sdim);

      DenseMatrix vshapeJ_u(sdim, ndof_u);
      Vector vshapeJu(ndof_u);
      Vector J_uv(J_u.GetData(), sdim);

      for (int q = 0; q < ir->Size(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Tr.SetIntPoint(&ip);
         Tr.Transform(ip, x);

         fe_u.CalcVShape(Tr, vshape_u);
         fe_p.CalcShape(ip, shape_p);

         p(0) = elfun_p * shape_p;
         vshape_u.MultTranspose(elfun_u, u);

         fluxFunction.ComputeDualFluxJacobian(p, mu, Tr, J_u, J_F);

         real_t w = ip.weight * Tr.Weight();

         if (elmats(0,0))
         {
            MultABt(J_F, vshape_u, vshapeJ_u);
            AddMult_a(w, vshape_u, vshapeJ_u, *elmats(0,0));
         }

         if (elmats(0,1))
         {
            vshape_u.Mult(J_uv, vshapeJu);
            AddMult_a_VWt(w, vshapeJu, shape_p, *elmats(0,1));
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
   constexpr real_t beta = 0.5;

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
}
