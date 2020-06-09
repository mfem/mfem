// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of Coefficient class

#include "fem.hpp"

#include <cmath>
#include <limits>

namespace mfem
{

using namespace std;

double PWConstCoefficient::Eval(ElementTransformation & T,
                                const IntegrationPoint & ip)
{
   int att = T.Attribute;
   return (constants(att-1));
}

double FunctionCoefficient::Eval(ElementTransformation & T,
                                 const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   if (Function)
   {
      return ((*Function)(transip));
   }
   else
   {
      return (*TDFunction)(transip, GetTime());
   }
}

double GridFunctionCoefficient::Eval (ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   return GridF -> GetValue (T, ip, Component);
}

double TransformedCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   if (Q2)
   {
      return (*Transform2)(Q1->Eval(T, ip, GetTime()),
                           Q2->Eval(T, ip, GetTime()));
   }
   else
   {
      return (*Transform1)(Q1->Eval(T, ip, GetTime()));
   }
}

void DeltaCoefficient::SetDeltaCenter(const Vector& vcenter)
{
   MFEM_VERIFY(vcenter.Size() <= 3,
               "SetDeltaCenter::Maximum number of dim supported is 3")
   for (int i = 0; i < vcenter.Size(); i++) { center[i] = vcenter[i]; }
   sdim = vcenter.Size();
}

void DeltaCoefficient::GetDeltaCenter(Vector& vcenter)
{
   vcenter.SetSize(sdim);
   vcenter = center;
}

double DeltaCoefficient::EvalDelta(ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   double w = Scale();
   return weight ? weight->Eval(T, ip, GetTime())*w : w;
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

void VectorFunctionCoefficient::Eval(Vector &V, ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   V.SetSize(vdim);
   if (Function)
   {
      (*Function)(transip, V);
   }
   else
   {
      (*TDFunction)(transip, GetTime(), V);
   }
   if (Q)
   {
      V *= Q->Eval(T, ip, GetTime());
   }
}

VectorArrayCoefficient::VectorArrayCoefficient (int dim)
   : VectorCoefficient(dim), Coeff(dim), ownCoeff(dim)
{
   for (int i = 0; i < dim; i++)
   {
      Coeff[i] = NULL;
      ownCoeff[i] = true;
   }
}

void VectorArrayCoefficient::Set(int i, Coefficient *c, bool own)
{
   if (ownCoeff[i]) { delete Coeff[i]; }
   Coeff[i] = c;
   ownCoeff[i] = own;
}

VectorArrayCoefficient::~VectorArrayCoefficient()
{
   for (int i = 0; i < vdim; i++)
   {
      if (ownCoeff[i]) { delete Coeff[i]; }
   }
}

void VectorArrayCoefficient::Eval(Vector &V, ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   V.SetSize(vdim);
   for (int i = 0; i < vdim; i++)
   {
      V(i) = this->Eval(i, T, ip);
   }
}

VectorGridFunctionCoefficient::VectorGridFunctionCoefficient (
   const GridFunction *gf)
   : VectorCoefficient ((gf) ? gf -> VectorDim() : 0)
{
   GridFunc = gf;
}

void VectorGridFunctionCoefficient::SetGridFunction(const GridFunction *gf)
{
   GridFunc = gf; vdim = (gf) ? gf -> VectorDim() : 0;
}

void VectorGridFunctionCoefficient::Eval(Vector &V, ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   GridFunc->GetVectorValue(T, ip, V);
}

void VectorGridFunctionCoefficient::Eval(
   DenseMatrix &M, ElementTransformation &T, const IntegrationRule &ir)
{
   GridFunc->GetVectorValues(T, ir, M);
}

GradientGridFunctionCoefficient::GradientGridFunctionCoefficient (
   const GridFunction *gf)
   : VectorCoefficient((gf) ?
                       gf -> FESpace() -> GetMesh() -> SpaceDimension() : 0)
{
   GridFunc = gf;
}

void GradientGridFunctionCoefficient::SetGridFunction(const GridFunction *gf)
{
   GridFunc = gf; vdim = (gf) ?
                         gf -> FESpace() -> GetMesh() -> SpaceDimension() : 0;
}

void GradientGridFunctionCoefficient::Eval(Vector &V, ElementTransformation &T,
                                           const IntegrationPoint &ip)
{
   GridFunc->GetGradient(T, V);
}

void GradientGridFunctionCoefficient::Eval(
   DenseMatrix &M, ElementTransformation &T, const IntegrationRule &ir)
{
   GridFunc->GetGradients(T, ir, M);
}

CurlGridFunctionCoefficient::CurlGridFunctionCoefficient (
   const GridFunction *gf)
   : VectorCoefficient ((gf) ?
                        gf -> FESpace() -> GetMesh() -> SpaceDimension() : 0)
{
   GridFunc = gf;
}

void CurlGridFunctionCoefficient::SetGridFunction(const GridFunction *gf)
{
   GridFunc = gf; vdim = (gf) ?
                         gf -> FESpace() -> GetMesh() -> SpaceDimension() : 0;
}

void CurlGridFunctionCoefficient::Eval(Vector &V, ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   GridFunc->GetCurl(T, V);
}

DivergenceGridFunctionCoefficient::DivergenceGridFunctionCoefficient (
   const GridFunction *gf) : Coefficient()
{
   GridFunc = gf;
}

double DivergenceGridFunctionCoefficient::Eval(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   return GridFunc->GetDivergence(T);
}

void VectorDeltaCoefficient::SetDirection(const Vector &_d)
{
   dir = _d;
   (*this).vdim = dir.Size();
}

void VectorDeltaCoefficient::EvalDelta(
   Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
{
   V = dir;
   d.SetTime(GetTime());
   V *= d.EvalDelta(T, ip);
}

void VectorRestrictedCoefficient::Eval(Vector &V, ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   V.SetSize(vdim);
   if (active_attr[T.Attribute-1])
   {
      c->SetTime(GetTime());
      c->Eval(V, T, ip);
   }
   else
   {
      V = 0.0;
   }
}

void VectorRestrictedCoefficient::Eval(
   DenseMatrix &M, ElementTransformation &T, const IntegrationRule &ir)
{
   if (active_attr[T.Attribute-1])
   {
      c->SetTime(GetTime());
      c->Eval(M, T, ir);
   }
   else
   {
      M.SetSize(vdim, ir.GetNPoints());
      M = 0.0;
   }
}

void MatrixFunctionCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   K.SetSize(height, width);

   if (Function)
   {
      (*Function)(transip, K);
   }
   else if (TDFunction)
   {
      (*TDFunction)(transip, GetTime(), K);
   }
   else
   {
      K = mat;
   }
   if (Q)
   {
      K *= Q->Eval(T, ip, GetTime());
   }
}

MatrixArrayCoefficient::MatrixArrayCoefficient (int dim)
   : MatrixCoefficient (dim)
{
   Coeff.SetSize(height*width);
   ownCoeff.SetSize(height*width);
   for (int i = 0; i < (height*width); i++)
   {
      Coeff[i] = NULL;
      ownCoeff[i] = true;
   }
}

void MatrixArrayCoefficient::Set(int i, int j, Coefficient * c, bool own)
{
   if (ownCoeff[i*width+j]) { delete Coeff[i*width+j]; }
   Coeff[i*width+j] = c;
   ownCoeff[i*width+j] = own;
}

MatrixArrayCoefficient::~MatrixArrayCoefficient ()
{
   for (int i=0; i < height*width; i++)
   {
      if (ownCoeff[i]) { delete Coeff[i]; }
   }
}

void MatrixArrayCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                  const IntegrationPoint &ip)
{
   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j < width; j++)
      {
         K(i,j) = this->Eval(i, j, T, ip);
      }
   }
}

void MatrixRestrictedCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   if (active_attr[T.Attribute-1])
   {
      c->SetTime(GetTime());
      c->Eval(K, T, ip);
   }
   else
   {
      K.SetSize(height, width);
      K = 0.0;
   }
}

InnerProductCoefficient::InnerProductCoefficient(VectorCoefficient &A,
                                                 VectorCoefficient &B)
   : a(&A), b(&B)
{
   MFEM_ASSERT(A.GetVDim() == B.GetVDim(),
               "InnerProductCoefficient:  "
               "Arguments have incompatible dimensions.");
}

double InnerProductCoefficient::Eval(ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   a->Eval(va, T, ip);
   b->Eval(vb, T, ip);
   return va * vb;
}

VectorRotProductCoefficient::VectorRotProductCoefficient(VectorCoefficient &A,
                                                         VectorCoefficient &B)
   : a(&A), b(&B), va(A.GetVDim()), vb(B.GetVDim())
{
   MFEM_ASSERT(A.GetVDim() == 2 && B.GetVDim() == 2,
               "VectorRotProductCoefficient:  "
               "Arguments must have dimension equal to two.");
}

double VectorRotProductCoefficient::Eval(ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   a->Eval(va, T, ip);
   b->Eval(vb, T, ip);
   return va[0] * vb[1] - va[1] * vb[0];
}

DeterminantCoefficient::DeterminantCoefficient(MatrixCoefficient &A)
   : a(&A), ma(A.GetHeight(), A.GetWidth())
{
   MFEM_ASSERT(A.GetHeight() == A.GetWidth(),
               "DeterminantCoefficient:  "
               "Argument must be a square matrix.");
}

double DeterminantCoefficient::Eval(ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   a->Eval(ma, T, ip);
   return ma.Det();
}

VectorSumCoefficient::VectorSumCoefficient(VectorCoefficient &A,
                                           VectorCoefficient &B,
                                           double _alpha, double _beta)
   : VectorCoefficient(A.GetVDim()), a(&A), b(&B), alpha(_alpha), beta(_beta),
     va(A.GetVDim())
{
   MFEM_ASSERT(A.GetVDim() == B.GetVDim(),
               "VectorSumCoefficient:  "
               "Arguments must have the same dimension.");
}

void VectorSumCoefficient::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   b->Eval(V, T, ip);
   if ( beta != 1.0 ) { V *= beta; }
   a->Eval(va, T, ip);
   V.Add(alpha, va);
}

ScalarVectorProductCoefficient::ScalarVectorProductCoefficient(
   Coefficient &A,
   VectorCoefficient &B)
   : VectorCoefficient(B.GetVDim()), a(&A), b(&B)
{}

void ScalarVectorProductCoefficient::Eval(Vector &V, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   double sa = a->Eval(T, ip);
   b->Eval(V, T, ip);
   V *= sa;
}

VectorCrossProductCoefficient::VectorCrossProductCoefficient(
   VectorCoefficient &A,
   VectorCoefficient &B)
   : VectorCoefficient(3), a(&A), b(&B), va(A.GetVDim()), vb(B.GetVDim())
{
   MFEM_ASSERT(A.GetVDim() == 3 && B.GetVDim() == 3,
               "VectorCrossProductCoefficient:  "
               "Arguments must have dimension equal to three.");
}

void VectorCrossProductCoefficient::Eval(Vector &V, ElementTransformation &T,
                                         const IntegrationPoint &ip)
{
   a->Eval(va, T, ip);
   b->Eval(vb, T, ip);
   V.SetSize(3);
   V[0] = va[1] * vb[2] - va[2] * vb[1];
   V[1] = va[2] * vb[0] - va[0] * vb[2];
   V[2] = va[0] * vb[1] - va[1] * vb[0];
}

MatVecCoefficient::MatVecCoefficient(MatrixCoefficient &A,
                                     VectorCoefficient &B)
   : VectorCoefficient(A.GetHeight()), a(&A), b(&B),
     ma(A.GetHeight(), A.GetWidth()), vb(B.GetVDim())
{
   MFEM_ASSERT(A.GetWidth() == B.GetVDim(),
               "MatVecCoefficient:  Arguments have incompatible dimensions.");
}

void MatVecCoefficient::Eval(Vector &V, ElementTransformation &T,
                             const IntegrationPoint &ip)
{
   a->Eval(ma, T, ip);
   b->Eval(vb, T, ip);
   ma.Mult(vb, V);
}

void IdentityMatrixCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   M.SetSize(dim);
   M = 0.0;
   for (int d=0; d<dim; d++) { M(d,d) = 1.0; }
}

MatrixSumCoefficient::MatrixSumCoefficient(MatrixCoefficient &A,
                                           MatrixCoefficient &B,
                                           double _alpha, double _beta)
   : MatrixCoefficient(A.GetHeight(), A.GetWidth()),
     a(&A), b(&B), alpha(_alpha), beta(_beta),
     ma(A.GetHeight(), A.GetWidth())
{
   MFEM_ASSERT(A.GetHeight() == B.GetHeight() && A.GetWidth() == B.GetWidth(),
               "MatrixSumCoefficient:  "
               "Arguments must have the same dimensions.");
}

void MatrixSumCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   b->Eval(M, T, ip);
   if ( beta != 1.0 ) { M *= beta; }
   a->Eval(ma, T, ip);
   M.Add(alpha, ma);
}

ScalarMatrixProductCoefficient::ScalarMatrixProductCoefficient(
   Coefficient &A,
   MatrixCoefficient &B)
   : MatrixCoefficient(B.GetHeight(), B.GetWidth()), a(&A), b(&B)
{}

void ScalarMatrixProductCoefficient::Eval(DenseMatrix &M,
                                          ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   double sa = a->Eval(T, ip);
   b->Eval(M, T, ip);
   M *= sa;
}

TransposeMatrixCoefficient::TransposeMatrixCoefficient(MatrixCoefficient &A)
   : MatrixCoefficient(A.GetWidth(), A.GetHeight()), a(&A)
{}

void TransposeMatrixCoefficient::Eval(DenseMatrix &M,
                                      ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   a->Eval(M, T, ip);
   M.Transpose();
}

InverseMatrixCoefficient::InverseMatrixCoefficient(MatrixCoefficient &A)
   : MatrixCoefficient(A.GetHeight(), A.GetWidth()), a(&A)
{
   MFEM_ASSERT(A.GetHeight() == A.GetWidth(),
               "InverseMatrixCoefficient:  "
               "Argument must be a square matrix.");
}

void InverseMatrixCoefficient::Eval(DenseMatrix &M,
                                    ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   a->Eval(M, T, ip);
   M.Invert();
}

OuterProductCoefficient::OuterProductCoefficient(VectorCoefficient &A,
                                                 VectorCoefficient &B)
   : MatrixCoefficient(A.GetVDim(), B.GetVDim()), a(&A), b(&B),
     va(A.GetVDim()), vb(B.GetVDim())
{}

void OuterProductCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   a->Eval(va, T, ip);
   b->Eval(vb, T, ip);
   M.SetSize(va.Size(), vb.Size());
   for (int i=0; i<va.Size(); i++)
   {
      for (int j=0; j<vb.Size(); j++)
      {
         M(i, j) = va[i] * vb[j];
      }
   }
}

double LpNormLoop(double p, Coefficient &coeff, Mesh &mesh,
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
         if (p < infinity())
         {
            norm += ip.weight * tr->Weight() * pow(val, p);
         }
         else
         {
            if (norm < val)
            {
               norm = val;
            }
         }
      }
   }
   return norm;
}

double LpNormLoop(double p, VectorCoefficient &coeff, Mesh &mesh,
                  const IntegrationRule *irs[])
{
   double norm = 0.0;
   ElementTransformation *tr;
   int vdim = coeff.GetVDim();
   Vector vval(vdim);
   double val;

   for (int i = 0; i < mesh.GetNE(); i++)
   {
      tr = mesh.GetElementTransformation(i);
      const IntegrationRule &ir = *irs[mesh.GetElementType(i)];
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         tr->SetIntPoint(&ip);
         coeff.Eval(vval, *tr, ip);
         if (p < infinity())
         {
            for (int idim(0); idim < vdim; ++idim)
            {
               norm += ip.weight * tr->Weight() * pow(fabs( vval(idim) ), p);
            }
         }
         else
         {
            for (int idim(0); idim < vdim; ++idim)
            {
               val = fabs(vval(idim));
               if (norm < val)
               {
                  norm = val;
               }
            }
         }
      }
   }

   return norm;
}

double ComputeLpNorm(double p, Coefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[])
{
   double norm = LpNormLoop(p, coeff, mesh, irs);

   if (p < infinity())
   {
      // negative quadrature weights may cause norm to be negative
      if (norm < 0.0)
      {
         norm = -pow(-norm, 1.0/p);
      }
      else
      {
         norm = pow(norm, 1.0/p);
      }
   }

   return norm;
}

double ComputeLpNorm(double p, VectorCoefficient &coeff, Mesh &mesh,
                     const IntegrationRule *irs[])
{
   double norm = LpNormLoop(p, coeff, mesh, irs);

   if (p < infinity())
   {
      // negative quadrature weights may cause norm to be negative
      if (norm < 0.0)
      {
         norm = -pow(-norm, 1.0/p);
      }
      else
      {
         norm = pow(norm, 1.0/p);
      }
   }

   return norm;
}

#ifdef MFEM_USE_MPI
double ComputeGlobalLpNorm(double p, Coefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[])
{
   double loc_norm = LpNormLoop(p, coeff, pmesh, irs);
   double glob_norm = 0;

   MPI_Comm comm = pmesh.GetComm();

   if (p < infinity())
   {
      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

      // negative quadrature weights may cause norm to be negative
      if (glob_norm < 0.0)
      {
         glob_norm = -pow(-glob_norm, 1.0/p);
      }
      else
      {
         glob_norm = pow(glob_norm, 1.0/p);
      }
   }
   else
   {
      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, comm);
   }

   return glob_norm;
}

double ComputeGlobalLpNorm(double p, VectorCoefficient &coeff, ParMesh &pmesh,
                           const IntegrationRule *irs[])
{
   double loc_norm = LpNormLoop(p, coeff, pmesh, irs);
   double glob_norm = 0;

   MPI_Comm comm = pmesh.GetComm();

   if (p < infinity())
   {
      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

      // negative quadrature weights may cause norm to be negative
      if (glob_norm < 0.0)
      {
         glob_norm = -pow(-glob_norm, 1.0/p);
      }
      else
      {
         glob_norm = pow(glob_norm, 1.0/p);
      }
   }
   else
   {
      MPI_Allreduce(&loc_norm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, comm);
   }

   return glob_norm;
}
#endif

VectorQuadratureFunctionCoefficient::VectorQuadratureFunctionCoefficient(
   QuadratureFunction &qf)
   : VectorCoefficient(qf.GetVDim()), QuadF(qf), index(0) { }

void VectorQuadratureFunctionCoefficient::SetComponent(int _index, int _length)
{
   MFEM_VERIFY(_index >= 0, "Index must be >= 0");
   MFEM_VERIFY(_index < QuadF.GetVDim(),
               "Index must be < QuadratureFunction length");
   index = _index;

   MFEM_VERIFY(_length > 0, "Length must be > 0");
   MFEM_VERIFY(_length <= QuadF.GetVDim() - index,
               "Length must be <= (QuadratureFunction length - index)");

   vdim = _length;
}

void VectorQuadratureFunctionCoefficient::Eval(Vector &V,
                                               ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   QuadF.HostRead();

   if (index == 0 && vdim == QuadF.GetVDim())
   {
      QuadF.GetElementValues(T.ElementNo, ip.index, V);
   }
   else
   {
      Vector temp;
      QuadF.GetElementValues(T.ElementNo, ip.index, temp);
      V.SetSize(vdim);
      for (int i = 0; i < vdim; i++)
      {
         V(i) = temp(index + i);
      }
   }

   return;
}

QuadratureFunctionCoefficient::QuadratureFunctionCoefficient(
   QuadratureFunction &qf) : QuadF(qf)
{
   MFEM_VERIFY(qf.GetVDim() == 1, "QuadratureFunction's vdim must be 1");
}

double QuadratureFunctionCoefficient::Eval(ElementTransformation &T,
                                           const IntegrationPoint &ip)
{
   QuadF.HostRead();
   Vector temp(1);
   QuadF.GetElementValues(T.ElementNo, ip.index, temp);
   return temp[0];
}

}
