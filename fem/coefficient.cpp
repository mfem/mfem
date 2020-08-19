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
#include "../linalg/dtensor.hpp"

namespace mfem
{

using namespace std;

void Coefficient::Eval(const FiniteElementSpace &fes, const IntegrationRule &ir,
                       Vector &qcoeff)
{
   const int ne = fes.GetMesh()->GetNE();
   const int nq = ir.GetNPoints();
   qcoeff.SetSize(nq * ne);
   auto C = Reshape(qcoeff.HostWrite(), nq, ne);
   for (int e = 0; e < ne; ++e)
   {
      ElementTransformation& T = *fes.GetElementTransformation(e);
      for (int q = 0; q < nq; ++q)
      {
         C(q,e) = this->Eval(T, ir.IntPoint(q));
      }
   }
}

void Coefficient::Eval(const FiniteElementSpace &fes,
                       const IntegrationRule &ir,
                       const FaceType type,
                       Vector &qcoeff)
{
   const int nf = fes.GetNFbyType(type);
   const int nq = ir.GetNPoints();
   const int dim = fes.GetMesh()->Dimension();
   const int quad1D =
      fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0))
      ->GetDofToQuad(ir, DofToQuad::TENSOR).nqpt;
   qcoeff.SetSize(nq * nf);
   auto C = Reshape(qcoeff.HostWrite(), nq, nf);
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      int e1, e2;
      int inf1, inf2;
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      int face_id = inf1 / 64;
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {
         ElementTransformation& T = *fes.GetMesh()->GetFaceTransformation(f);
         for (int q = 0; q < nq; ++q)
         {
            // Convert to lexicographic ordering
            int iq = ToLexOrdering(dim, face_id, quad1D, q);
            C(iq,f_ind) = Eval(T, ir.IntPoint(q));
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
}

void ConstantCoefficient::Eval(const FiniteElementSpace &fes,
                               const IntegrationRule &ir,
                               Vector &qcoeff)
{
   qcoeff.SetSize(1);
   qcoeff(0) = constant;
}

void ConstantCoefficient::Eval(const FiniteElementSpace &fes,
                               const IntegrationRule &ir,
                               const FaceType type,
                               Vector &qcoeff)
{
   qcoeff.SetSize(1);
   qcoeff(0) = constant;
}

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

void QuadratureFunctionCoefficient::Eval(const FiniteElementSpace &fes,
                                         const IntegrationRule &ir,
                                         Vector &qcoeff)
{
   const int ne = fes.GetMesh()->GetNE();
   const int nq = ir.GetNPoints();
   MFEM_VERIFY(QuadF.Size() == nq * ne,
               "Incompatible QuadratureFunction dimension \n");

   MFEM_VERIFY(&ir == &QuadF.GetSpace()->GetElementIntRule(0),
               "IntegrationRule used within integrator and in"
               " QuadratureFunction appear to be different");
   QuadF.Read();
   qcoeff.MakeRef(const_cast<QuadratureFunction &>(QuadF),0);
}

void QuadratureFunctionCoefficient::Eval(const FiniteElementSpace &fes,
                                         const IntegrationRule &ir,
                                         const FaceType type,
                                         Vector &qcoeff)
{
   const int nf = fes.GetNFbyType(type);
   const int nq = ir.GetNPoints();
   MFEM_VERIFY(QuadF.Size() == nq * nf,
               "Incompatible QuadratureFunction dimension \n");

   MFEM_VERIFY(&ir == &QuadF.GetSpace()->GetElementIntRule(0),
               "IntegrationRule used within integrator and in"
               " QuadratureFunction appear to be different");
   QuadF.Read();
   qcoeff.MakeRef(const_cast<QuadratureFunction &>(QuadF),0);
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

void VectorCoefficient::Eval(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             Vector &qcoeff)
{
   const int ne = fes.GetMesh()->GetNE();
   const int nq = ir.GetNPoints();
   qcoeff.SetSize(vdim * nq * ne);
   auto C = Reshape(qcoeff.HostWrite(), vdim, nq, ne);
   DenseMatrix M(vdim, nq);
   for (int e = 0; e < ne; ++e)
   {
      ElementTransformation& T = *fes.GetElementTransformation(e);
      Eval(M, T, ir);
      for (int q = 0; q < nq; ++q)
      {
         for (int d = 0; d < vdim; d++)
         {
            C(d,q,e) = M(d,q);
         }
      }
   }
}

void VectorCoefficient::Eval(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             const FaceType type,
                             Vector &qcoeff)
{
   const int nf = fes.GetNFbyType(type);
   const int nq = ir.GetNPoints();
   const int dim = fes.GetMesh()->Dimension();
   const int quad1D =
      fes.GetTraceElement(0, fes.GetMesh()->GetFaceBaseGeometry(0))
      ->GetDofToQuad(ir, DofToQuad::TENSOR).nqpt;
   qcoeff.SetSize(dim * nq * nf);
   auto C = Reshape(qcoeff.HostWrite(), dim, nq, nf);
   Vector Vq(dim);
   int f_ind = 0;
   for (int f = 0; f < fes.GetNF(); ++f)
   {
      int e1, e2;
      int inf1, inf2;
      fes.GetMesh()->GetFaceElements(f, &e1, &e2);
      fes.GetMesh()->GetFaceInfos(f, &inf1, &inf2);
      int face_id = inf1 / 64;
      if ((type==FaceType::Interior && (e2>=0 || (e2<0 && inf2>=0))) ||
          (type==FaceType::Boundary && e2<0 && inf2<0) )
      {
         ElementTransformation& T = *fes.GetMesh()->GetFaceTransformation(f);
         for (int q = 0; q < nq; ++q)
         {
            // Convert to lexicographic ordering
            int iq = ToLexOrdering(dim, face_id, quad1D, q);
            Eval(Vq, T, ir.IntPoint(q));
            for (int i = 0; i < dim; ++i)
            {
               C(i,iq,f_ind) = Vq(i);
            }
         }
         f_ind++;
      }
   }
   MFEM_VERIFY(f_ind==nf, "Incorrect number of faces.");
}

void VectorConstantCoefficient::Eval(const FiniteElementSpace &fes,
                                     const IntegrationRule &ir,
                                     Vector &qcoeff)
{
   qcoeff.SetSize(vdim);
   for (int d = 0; d < vdim; d++)
   {
      qcoeff(d) = vec(d);
   }
}


void VectorConstantCoefficient::Eval(const FiniteElementSpace &fes,
                                     const IntegrationRule &ir,
                                     const FaceType type,
                                     Vector &qcoeff)
{
   qcoeff.SetSize(vdim);
   for (int d = 0; d < vdim; d++)
   {
      qcoeff(d) = vec(d);
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

void VectorQuadratureFunctionCoefficient::Eval(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               Vector &qcoeff)
{
   const int ne = fes.GetMesh()->GetNE();
   const int nq = ir.GetNPoints();
   const int dim = fes.GetMesh()->Dimension();
   MFEM_VERIFY(QuadF.Size() == dim * nq * ne,
               "Incompatible QuadratureFunction dimension \n");

   MFEM_VERIFY(&ir == &QuadF.GetSpace()->GetElementIntRule(0),
               "IntegrationRule used within integrator and in"
               " QuadratureFunction appear to be different");

   QuadF.Read();
   qcoeff.MakeRef(const_cast<QuadratureFunction &>(QuadF),0);
}

void VectorQuadratureFunctionCoefficient::Eval(const FiniteElementSpace &fes,
                                               const IntegrationRule &ir,
                                               const FaceType type,
                                               Vector &qcoeff)
{
   const int nf = fes.GetNFbyType(type);
   const int nq = ir.GetNPoints();
   const int dim = fes.GetMesh()->Dimension();
   // Assumed to be in lexicographical ordering
   MFEM_VERIFY(QuadF.Size() == dim * nq * nf,
               "Incompatible QuadratureFunction dimension \n");

   MFEM_VERIFY(&ir == &QuadF.GetSpace()->GetElementIntRule(0),
               "IntegrationRule used within integrator and in"
               " QuadratureFunction appear to be different");
   qcoeff.Read();
   qcoeff.MakeRef(const_cast<QuadratureFunction &>(QuadF),0);
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

CurlGridFunctionCoefficient::CurlGridFunctionCoefficient(
   const GridFunction *gf)
   : VectorCoefficient(0)
{
   SetGridFunction(gf);
}

void CurlGridFunctionCoefficient::SetGridFunction(const GridFunction *gf)
{
   if (gf)
   {
      int sdim = gf -> FESpace() -> GetMesh() -> SpaceDimension();
      MFEM_VERIFY(sdim == 2 || sdim == 3,
                  "CurlGridFunctionCoefficient "
                  "only defind for spaces of dimension 2 or 3.");
   }
   GridFunc = gf;
   vdim = (gf) ? (2 * gf -> FESpace() -> GetMesh() -> SpaceDimension() - 3) : 0;
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
void MatrixCoefficient::Eval(const FiniteElementSpace &fes,
                             const IntegrationRule &ir,
                             Vector &qcoeff)
{
   if (!IsSymmetric())
   {
      const int ne = fes.GetMesh()->GetNE();
      const int nq = ir.GetNPoints();
      qcoeff.SetSize(height * width * nq * ne);
      auto C = Reshape(qcoeff.HostWrite(), height, width, nq, ne);
      DenseMatrix K(height, width);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            Eval(K, T, ir.IntPoint(q));
            for (int w = 0; w < width; w++)
            {
               for (int h = 0; h < height; h++)
               {
                  C(h,w,q,e) = K(h,w);
               }
            }
         }
      }
   }
   else
   {
      const int ne = fes.GetMesh()->GetNE();
      const int nq = ir.GetNPoints();
      const int symmDim = width*(width+1)/2;
      qcoeff.SetSize(symmDim * nq * ne);
      auto C = Reshape(qcoeff.HostWrite(), symmDim, nq, ne);
      Vector K(symmDim);
      for (int e = 0; e < ne; ++e)
      {
         ElementTransformation& T = *fes.GetElementTransformation(e);
         for (int q = 0; q < nq; ++q)
         {
            EvalSymmetric(K, T, ir.IntPoint(q));
            for (int c = 0; c < symmDim; c++)
            {
               C(c,q,e) = K(c);
            }
         }
      }
   }
}

void MatrixConstantCoefficient::Eval(const FiniteElementSpace &fes,
                                     const IntegrationRule &ir,
                                     Vector &qcoeff)
{
   qcoeff.SetSize(height * width);
   auto C = Reshape(qcoeff.HostWrite(), height, width);
   for (int w = 0; w < width; w++)
   {
      for (int h = 0; h < height; h++)
      {
         C(h,w) = mat(h,w);
      }
   }
}

void MatrixFunctionCoefficient::EvalSymmetric(Vector &K,
                                              ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_VERIFY(symmetric && height == width && height < 4 && SymmFunction,
               "MatrixFunctionCoefficient is not symmetric");

   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   K.SetSize((width * (width + 1)) / 2); // 1x1: 1, 2x2: 3, 3x3: 6

   if (SymmFunction)
   {
      (*SymmFunction)(transip, K);
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

VectorSumCoefficient::VectorSumCoefficient(int dim)
   : VectorCoefficient(dim),
     ACoef(NULL), BCoef(NULL),
     A(dim), B(dim),
     alphaCoef(NULL), betaCoef(NULL),
     alpha(1.0), beta(1.0)
{
   A = 0.0; B = 0.0;
}

VectorSumCoefficient::VectorSumCoefficient(VectorCoefficient &_A,
                                           VectorCoefficient &_B,
                                           double _alpha, double _beta)
   : VectorCoefficient(_A.GetVDim()),
     ACoef(&_A), BCoef(&_B),
     A(_A.GetVDim()), B(_A.GetVDim()),
     alphaCoef(NULL), betaCoef(NULL),
     alpha(_alpha), beta(_beta)
{
   MFEM_ASSERT(_A.GetVDim() == _B.GetVDim(),
               "VectorSumCoefficient:  "
               "Arguments must have the same dimension.");
}

VectorSumCoefficient::VectorSumCoefficient(VectorCoefficient &_A,
                                           VectorCoefficient &_B,
                                           Coefficient &_alpha,
                                           Coefficient &_beta)
   : VectorCoefficient(_A.GetVDim()),
     ACoef(&_A), BCoef(&_B),
     A(_A.GetVDim()),
     B(_A.GetVDim()),
     alphaCoef(&_alpha),
     betaCoef(&_beta),
     alpha(0.0), beta(0.0)
{
   MFEM_ASSERT(_A.GetVDim() == _B.GetVDim(),
               "VectorSumCoefficient:  "
               "Arguments must have the same dimension.");
}

void VectorSumCoefficient::Eval(Vector &V, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   V.SetSize(A.Size());
   if (    ACoef) { ACoef->Eval(A, T, ip); }
   if (    BCoef) { BCoef->Eval(B, T, ip); }
   if (alphaCoef) { alpha = alphaCoef->Eval(T, ip); }
   if ( betaCoef) { beta  = betaCoef->Eval(T, ip); }
   add(alpha, A, beta, B, V);
}

ScalarVectorProductCoefficient::ScalarVectorProductCoefficient(
   double A,
   VectorCoefficient &B)
   : VectorCoefficient(B.GetVDim()), aConst(A), a(NULL), b(&B)
{}

ScalarVectorProductCoefficient::ScalarVectorProductCoefficient(
   Coefficient &A,
   VectorCoefficient &B)
   : VectorCoefficient(B.GetVDim()), aConst(0.0), a(&A), b(&B)
{}

void ScalarVectorProductCoefficient::Eval(Vector &V, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   double sa = (a == NULL) ? aConst : a->Eval(T, ip);
   b->Eval(V, T, ip);
   V *= sa;
}

NormalizedVectorCoefficient::NormalizedVectorCoefficient(VectorCoefficient &A,
                                                         double _tol)
   : VectorCoefficient(A.GetVDim()), a(&A), tol(_tol)
{}

void NormalizedVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   a->Eval(V, T, ip);
   double nv = V.Norml2();
   V *= (nv > tol) ? (1.0/nv) : 0.0;
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

MatrixVectorProductCoefficient::MatrixVectorProductCoefficient(
   MatrixCoefficient &A, VectorCoefficient &B)
   : VectorCoefficient(A.GetHeight()), a(&A), b(&B),
     ma(A.GetHeight(), A.GetWidth()), vb(B.GetVDim())
{
   MFEM_ASSERT(A.GetWidth() == B.GetVDim(),
               "MatrixVectorProductCoefficient:  "
               "Arguments have incompatible dimensions.");
}

void MatrixVectorProductCoefficient::Eval(Vector &V, ElementTransformation &T,
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
   double A,
   MatrixCoefficient &B)
   : MatrixCoefficient(B.GetHeight(), B.GetWidth()), aConst(A), a(NULL), b(&B)
{}

ScalarMatrixProductCoefficient::ScalarMatrixProductCoefficient(
   Coefficient &A,
   MatrixCoefficient &B)
   : MatrixCoefficient(B.GetHeight(), B.GetWidth()), aConst(0.0), a(&A), b(&B)
{}

void ScalarMatrixProductCoefficient::Eval(DenseMatrix &M,
                                          ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   double sa = (a == NULL) ? aConst : a->Eval(T, ip);
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

CrossCrossCoefficient::CrossCrossCoefficient(Coefficient &A,
                                             VectorCoefficient &K)
   : MatrixCoefficient(K.GetVDim(), K.GetVDim()), aConst(0.0), a(&A), k(&K),
     vk(K.GetVDim())
{}

void CrossCrossCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                 const IntegrationPoint &ip)
{
   k->Eval(vk, T, ip);
   M.SetSize(vk.Size(), vk.Size());
   M = 0.0;
   double k2 = vk*vk;
   for (int i=0; i<vk.Size(); i++)
   {
      M(i, i) = k2;
      for (int j=0; j<vk.Size(); j++)
      {
         M(i, j) -= vk[i] * vk[j];
      }
   }
   M *= ((a == NULL ) ? aConst : a->Eval(T, ip) );
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
