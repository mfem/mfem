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
   DenseMatrix &J) const
{
   if (coeff)
   {
      const real_t ikappa = coeff->Eval(Tr, Tr.GetIntPoint());
      J.Diag(ikappa, dim);
   }
   else if (vcoeff)
   {
      Vector ikappa(dim);
      vcoeff->Eval(ikappa, Tr, Tr.GetIntPoint());
      J.Diag(ikappa);
   }
   else if (mcoeff)
   {
      mcoeff->Eval(J, Tr, Tr.GetIntPoint());
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
   Vector &elvect_p = *elvect[1];

   shape_p.SetSize(ndof_p);

   //not used
   elvect_p.SetSize(ndof_p);
   elvect_p = 0.0;

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

void MixedConductionNLFIntegrator::AssembleElementGrad(
   const Array<const FiniteElement *> &el, ElementTransformation &Tr,
   const Array<const Vector *> &elfun, const Array2D<DenseMatrix *> &elmats)
{
   const int ndof_u = el[0]->GetDof();
   const int ndof_p = el[1]->GetDof();
   const int sdim = Tr.GetSpaceDim();

   const FiniteElement &fe_u = *el[0];
   const FiniteElement &fe_p = *el[1];
   const Vector &elfun_u = *elfun[0];
   const Vector &elfun_p = *elfun[1];

   shape_p.SetSize(ndof_p);

   //not used
   elmats(1,1)->SetSize(ndof_p);
   *elmats(1,1) = 0.0;

   DenseMatrix J(sdim);
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
      elmats(0,0)->SetSize(ndof_u * sdim);
      *elmats(0,0) = 0.0;

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

         fluxFunction.ComputeDualFluxJacobian(p, mu, Tr, J);

         for (int d_j = 0; d_j < sdim; d_j++)
            for (int d_i = 0; d_i < sdim; d_i++)
               for (int j = 0; j < ndof_u; j++)
                  for (int i = 0; i < ndof_u; i++)
                  {
                     (*elmats(0,0))(i+d_i*ndof_u, j+d_j*ndof_u)
                     += w * J(d_i,d_j) * shape_u(i) * shape_u(j);
                  }
      }
   }
   else
   {
      vshape_u.SetSize(ndof_u, sdim);
      elmats(0,0)->SetSize(ndof_u);
      *elmats(0,0) = 0.0;

      DenseMatrix vshapeJ_u(sdim, ndof_u);

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

         fluxFunction.ComputeDualFluxJacobian(p, mu, Tr, J);

         MultABt(J, vshape_u, vshapeJ_u);
         AddMult_a(w, vshape_u, vshapeJ_u, *elmats(0,0));
      }
   }
}

}
