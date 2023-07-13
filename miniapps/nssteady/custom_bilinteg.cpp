// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "custom_bilinteg.hpp"

namespace mfem
{

const IntegrationRule&
VectorConvectionIntegrator::GetRule(const FiniteElement &fe,
                                       ElementTransformation &T)
{
   const int order = 2 * fe.GetOrder() + T.OrderGrad(&fe);
   return IntRules.Get(fe.GetGeomType(), order);
}

void VectorConvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dof = el.GetDof();
   dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, W_ir, pelmat, pelmat_T;
   Vector shape, vec1, vec2, vec3;
#endif
   elmat.SetSize(dim*dof);
   dshape.SetSize(dof,dim);
   adjJ.SetSize(dim);
   shape.SetSize(dof);
   vec1.SetSize(dim);
   vec2.SetSize(dim);
   vec3.SetSize(dof);
   pelmat.SetSize(dof);
   DenseMatrix pelmat_T(dof);


   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir =  &GetRule(el, Trans);
   }

   W->Eval(W_ir, Trans, *ir);

   elmat = 0.0;
   pelmat_T = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      W_ir.GetColumnReference(i, vec1);     // tmp = W

      const double q = alpha ? alpha * ip.weight : ip.weight; // q = alpha*weight   || q = weight
      adjJ.Mult(vec1, vec2);               // element transformation J^{-1} |J|      
      vec2 *= q;

      dshape.Mult(vec2, vec3);           // (w . grad u)           q ( alpha J^{-1} |J| w dPhi )  
      MultVWt(shape, vec3, pelmat);      // (w . grad u,v)         q ( alpha J^{-1} |J| w dPhi Phi^T)

      if( SkewSym )
      {
        pelmat_T.Transpose(pelmat);
      }

      for (int k = 0; k < dim; k++)
      {
        if( SkewSym )
        {
            elmat.AddMatrix(.5, pelmat, dof*k, dof*k);
            elmat.AddMatrix(-.5, pelmat_T, dof*k, dof*k);
        }
        else
        {
            elmat.AddMatrix(pelmat, dof*k, dof*k);
        }
      }
   }
}

}