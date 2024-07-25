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

#include "my_integrators.hpp"
using namespace std;


namespace mfem
{

void MySimpleDiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   real_t w;

   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);


   elmat.SetSize(nd);

   int order = 4; //Todo: set this differently 
   const IntegrationRule *ir = nullptr;

   ir = &IntRules.Get(el.GetGeomType(),order);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (w);
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      AddMult_a_AAt(w, dshapedxt, elmat);
      
   }
}


void MyDiffusionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   real_t w;

   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);


   elmat.SetSize(nd);

   int order = 2*el.GetOrder();  
   const IntegrationRule *ir = nullptr;
   ir = &IntRules.Get(el.GetGeomType(),order);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (w);
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      if (Q)
      {
         w *= Q->Eval(Trans, ip);
      }
      AddMult_a_AAt(w, dshapedxt, elmat);
      
   }
}


void MyVectorDiffusionIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   dim = el.GetDim();
   sdim = Trans.GetSpaceDim();

   // If vdim is not set, set it to the space dimension;
   vdim = (vdim <= 0) ? sdim : vdim;
   const bool square = (dim == sdim);

   if (VQ)
   {
      vcoeff.SetSize(vdim);
   }
   else if (MQ)
   {
      mcoeff.SetSize(vdim);
   }

   dshape.SetSize(dof, dim);
   dshapedxt.SetSize(dof, sdim);

   elmat.SetSize(vdim * dof);
   pelmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el,el);
   }

   elmat = 0.0;

   for (int i = 0; i < ir -> GetNPoints(); i++)
   {

      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      real_t w = Trans.Weight();
      w = ip.weight / (square ? w : w*w*w);
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);

      if (MQ)
      {
         MQ->Eval(mcoeff, Trans, ip);
         for (int ii = 0; ii < vdim; ++ii)
         {
            for (int jj = 0; jj < vdim; ++jj)
            {
               Mult_a_AAt(w*mcoeff(ii,jj), dshapedxt, pelmat);
               elmat.AddMatrix(pelmat, dof*ii, dof*jj);
            }
         }
      }
      else if (VQ)
      {
         VQ->Eval(vcoeff, Trans, ip);
         for (int k = 0; k < vdim; ++k)
         {
            Mult_a_AAt(w*vcoeff(k), dshapedxt, pelmat);
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
      else
      {
         if (Q) { w *= Q->Eval(Trans, ip); }
         Mult_a_AAt(w, dshapedxt, pelmat);
         for (int k = 0; k < vdim; ++k)
         {
            elmat.AddMatrix(pelmat, dof*k, dof*k);
         }
      }
   }
}

} // end mfem namespace