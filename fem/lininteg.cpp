// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.


#include <math.h>
#include "fem.hpp"

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}


void DomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0;

   const IntegrationRule *ir;
   if (IntRule)
   {
      ir = IntRule;
   }
   else
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   int intorder = oa * el.GetOrder() + ob;  // <----------
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void VectorDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   double val,cf;

   shape.SetSize(dof);       // vector of size dof

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   int intorder = el.GetOrder() + 1;
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Tr.SetIntPoint (&ip);
      val = Tr.Weight();

      el.CalcShape(ip, shape);
      Q.Eval (Qvec, Tr, ip);

      for (int k = 0; k < vdim; k++)
      {
         cf = val * Qvec(k);

         for (int s = 0; s < dof; s++)
            elvect(dof*k+s) += ip.weight * cf * shape(s);
      }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   int intorder = el.GetOrder() + 1;
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Q.Eval(vec, Tr, ip);
      Tr.SetIntPoint (&ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
            elvect(dof*k+s) += vec(k) * shape(s);
   }
}

void VectorFEDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int dim = el.GetDim();

   vshape.SetSize(dof,dim);
   vec.SetSize(dim);

   elvect.SetSize(dof);
   elvect = 0.0;

   // int intorder = 2*el.GetOrder() - 1; // <-- ok for O(h^{k+1}) conv. in L2
   int intorder = 2*el.GetOrder();
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcVShape(Tr, vshape);

      QF.Eval (vec, Tr, ip);
      vec *= ip.weight * Tr.Weight();

      vshape.AddMult (vec, elvect);
   }
}


void VectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir;

   if (!IntRule)
      ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
   else
      ir = IntRule;

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);
      const DenseMatrix &Jac = Tr.Jacobian();
      if (dim == 2)
      {
         nor(0) =  Jac (1,0);
         nor(1) = -Jac (0,0);
      }
      else if (dim == 3)
      {
         nor(0) = Jac (1,0) * Jac (2,1) - Jac (2,0) * Jac (1,1);
         nor(1) = Jac (2,0) * Jac (0,1) - Jac (0,0) * Jac (2,1);
         nor(2) = Jac (0,0) * Jac (1,1) - Jac (1,0) * Jac (0,1);
      }
      el.CalcShape (ip, shape);
      nor *= Sign * ip.weight * F -> Eval (Tr, ip);
      for (int j = 0; j < dof; j++)
         for (int k = 0; k < dim; k++)
            elvect(dof*k+j) += nor(k) * shape(j);
   }
}


void VectorFEBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   int intorder = 2*el.GetOrder();  // <----------
   const IntegrationRule &ir = IntRules.Get(el.GetGeomType(), intorder);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = ip.weight*F.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, val, shape, elvect);
   }
}
