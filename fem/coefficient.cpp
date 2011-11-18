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

// Implementation of Coefficient class

#include <math.h>
#include <limits>
#include "fem.hpp"

double PWConstCoefficient::Eval(ElementTransformation & T,
                                const IntegrationPoint & ip)
{
   int att = T.Attribute;
   return(constants(att-1));
}

void PWConstCoefficient::Read(istream &in)
{
   int i, n;

   in >> n;
   constants.SetSize(n);
   for (i = 0; i < n; i++)
      in >> constants(i);
}

double FunctionCoefficient::Eval(ElementTransformation & T,
                                 const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   return((*Function)(transip));
}

double GridFunctionCoefficient::Eval (ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   return GridF -> GetValue (T.ElementNo, ip, Component);
}

void VectorCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                             const IntegrationRule &ir)
{
   Vector Mi;
   M.SetSize(vdim, ir.GetNPoints());
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      M.GetColumnReference(i, Mi);
      const IntegrationPoint &ip = ir.IntPoint(i);
      T.SetIntPoint(&ip);
      Eval(Mi, T, ip);
   }
}

void VectorFunctionCoefficient::Eval (Vector &V, ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform (ip, transip);

   V.SetSize (vdim);
   (*Function) (transip, V);
   if (Q)
      V *= Q -> Eval (T, ip);
}

VectorArrayCoefficient::VectorArrayCoefficient (int dim)
   : VectorCoefficient(dim), Coeff(dim)
{
   for (int i = 0; i < dim; i++)
      Coeff[i] = NULL;
}

VectorArrayCoefficient::~VectorArrayCoefficient()
{
   for (int i = 0; i < vdim; i++)
      delete Coeff[i];
}

void VectorArrayCoefficient::Eval (Vector &V, ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   int i;

   V.SetSize(vdim);
   for (i = 0; i < vdim; i++)
      V(i) = Coeff[i] -> Eval (T, ip);
}

VectorGridFunctionCoefficient::VectorGridFunctionCoefficient (
   GridFunction *gf) : VectorCoefficient (gf -> VectorDim())
{
   GridFunc = gf;
}

void VectorGridFunctionCoefficient::Eval (Vector &V, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   GridFunc -> GetVectorValue (T.ElementNo, ip, V);
}

void VectorRestrictedCoefficient::Eval(Vector &V, ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   V.SetSize(vdim);
   if (active_attr[T.Attribute-1])
      c->Eval(V, T, ip);
   else
      V = 0.0;
}

void MatrixFunctionCoefficient::Eval (DenseMatrix &K, ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform (ip, transip);

   K.SetSize (vdim);
   (*Function) (transip, K);
}

MatrixArrayCoefficient::MatrixArrayCoefficient (int dim)
   : MatrixCoefficient (dim)
{
   Coeff.SetSize (vdim*vdim);
}

MatrixArrayCoefficient::~MatrixArrayCoefficient ()
{
   for (int i=0; i< vdim*vdim; i++)
      delete Coeff[i];
}

void MatrixArrayCoefficient::Eval (DenseMatrix &K, ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   int i, j;

   for (i = 0; i < vdim; i++)
      for (j = 0; j < vdim; j++)
         K(i,j) = Coeff[i*vdim+j] -> Eval(T, ip);
}

double ComputeLpNorm(double p, Coefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[])
{
   double norm = 0.0;
   ElementTransformation *tr;

   for (int i = 0; i < mesh.GetNE(); i++)
   {
      tr = mesh.GetElementTransformation(i);
      const IntegrationRule &ir = *irs[mesh.GetElementType(i)];
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         tr->SetIntPoint(&ip);
         double val = fabs(coeff.Eval(*tr, ip));
         if (p < numeric_limits<double>::infinity())
         {
            norm += ip.weight * tr->Weight() * pow(val, p);
         }
         else
         {
            if (norm < val)
               norm = val;
         }
      }
   }

   if (p < numeric_limits<double>::infinity())
   {
      // negative quadrature weights may cause norm to be negative
      if (norm < 0.)
         norm = -pow(-norm, 1. / p);
      else
         norm = pow(norm, 1. / p);
   }

   return norm;
}
