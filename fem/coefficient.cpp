// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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

// Given an ElementTransformation and IntegrationPoint in a refined mesh,
// return the ElementTransformation of the parent coarse element, and set
// coarse_ip to the location of the original ip within the coarse element.
ElementTransformation *RefinedToCoarse(
   Mesh &coarse_mesh, const ElementTransformation &T,
   const IntegrationPoint &ip, IntegrationPoint &coarse_ip)
{
   Mesh &fine_mesh = *T.mesh;
   // Get the element transformation of the coarse element containing the
   // fine element.
   int fine_element = T.ElementNo;
   const CoarseFineTransformations &cf = fine_mesh.GetRefinementTransforms();
   int coarse_element = cf.embeddings[fine_element].parent;
   ElementTransformation *coarse_T = coarse_mesh.GetElementTransformation(
                                        coarse_element);
   // Transform the integration point from fine element coordinates to coarse
   // element coordinates.
   Geometry::Type geom = T.GetGeometryType();
   IntegrationPointTransformation fine_to_coarse;
   IsoparametricTransformation &emb_tr = fine_to_coarse.Transf;
   emb_tr.SetIdentityTransformation(geom);
   emb_tr.SetPointMat(cf.point_matrices[geom](cf.embeddings[fine_element].matrix));
   fine_to_coarse.Transform(ip, coarse_ip);
   coarse_T->SetIntPoint(&coarse_ip);
   return coarse_T;
}

double PWConstCoefficient::Eval(ElementTransformation & T,
                                const IntegrationPoint & ip)
{
   int att = T.Attribute;
   return (constants(att-1));
}

void PWCoefficient::InitMap(const Array<int> & attr,
                            const Array<Coefficient*> & coefs)
{
   MFEM_VERIFY(attr.Size() == coefs.Size(),
               "PWCoefficient:  "
               "Attribute and coefficient arrays have incompatible "
               "dimensions.");

   for (int i=0; i<attr.Size(); i++)
   {
      if (coefs[i] != NULL)
      {
         UpdateCoefficient(attr[i], *coefs[i]);
      }
   }
}

void PWCoefficient::SetTime(double t)
{
   Coefficient::SetTime(t);

   std::map<int, Coefficient*>::iterator p = pieces.begin();
   for (; p != pieces.end(); p++)
   {
      if (p->second != NULL)
      {
         p->second->SetTime(t);
      }
   }
}

double PWCoefficient::Eval(ElementTransformation &T,
                           const IntegrationPoint &ip)
{
   const int att = T.Attribute;
   std::map<int, Coefficient*>::const_iterator p = pieces.find(att);
   if (p != pieces.end())
   {
      if ( p->second != NULL)
      {
         return p->second->Eval(T, ip);
      }
   }
   return 0.0;
}

double FunctionCoefficient::Eval(ElementTransformation & T,
                                 const IntegrationPoint & ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   if (Function)
   {
      return Function(transip);
   }
   else
   {
      return TDFunction(transip, GetTime());
   }
}

double GridFunctionCoefficient::Eval (ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   Mesh *gf_mesh = GridF->FESpace()->GetMesh();
   if (T.mesh == gf_mesh)
   {
      return GridF->GetValue(T, ip, Component);
   }
   else
   {
      IntegrationPoint coarse_ip;
      ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      return GridF->GetValue(*coarse_T, coarse_ip, Component);
   }
}

void TransformedCoefficient::SetTime(double t)
{
   if (Q1) { Q1->SetTime(t); }
   if (Q2) { Q2->SetTime(t); }
   this->Coefficient::SetTime(t);
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

void DeltaCoefficient::SetTime(double t)
{
   if (weight) { weight->SetTime(t); }
   this->Coefficient::SetTime(t);
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

void RestrictedCoefficient::SetTime(double t)
{
   if (c) { c->SetTime(t); }
   this->Coefficient::SetTime(t);
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

void PWVectorCoefficient::InitMap(const Array<int> & attr,
                                  const Array<VectorCoefficient*> & coefs)
{
   MFEM_VERIFY(attr.Size() == coefs.Size(),
               "PWVectorCoefficient:  "
               "Attribute and coefficient arrays have incompatible "
               "dimensions.");

   for (int i=0; i<attr.Size(); i++)
   {
      if (coefs[i] != NULL)
      {
         UpdateCoefficient(attr[i], *coefs[i]);
      }
   }
}

void PWVectorCoefficient::UpdateCoefficient(int attr, VectorCoefficient & coef)
{
   MFEM_VERIFY(coef.GetVDim() == vdim,
               "PWVectorCoefficient::UpdateCoefficient:  "
               "VectorCoefficient has incompatible dimension.");
   pieces[attr] = &coef;
}

void PWVectorCoefficient::SetTime(double t)
{
   VectorCoefficient::SetTime(t);

   std::map<int, VectorCoefficient*>::iterator p = pieces.begin();
   for (; p != pieces.end(); p++)
   {
      if (p->second != NULL)
      {
         p->second->SetTime(t);
      }
   }
}

void PWVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   const int att = T.Attribute;
   std::map<int, VectorCoefficient*>::const_iterator p = pieces.find(att);
   if (p != pieces.end())
   {
      if ( p->second != NULL)
      {
         p->second->Eval(V, T, ip);
         return;
      }
   }

   V.SetSize(vdim);
   V = 0.0;
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
      Function(transip, V);
   }
   else
   {
      TDFunction(transip, GetTime(), V);
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

void VectorArrayCoefficient::SetTime(double t)
{
   for (int i = 0; i < vdim; i++)
   {
      if (Coeff[i]) { Coeff[i]->SetTime(t); }
   }
   this->VectorCoefficient::SetTime(t);
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
   Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
   if (T.mesh == gf_mesh)
   {
      GridFunc->GetVectorValue(T, ip, V);
   }
   else
   {
      IntegrationPoint coarse_ip;
      ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      GridFunc->GetVectorValue(*coarse_T, coarse_ip, V);
   }
}

void VectorGridFunctionCoefficient::Eval(
   DenseMatrix &M, ElementTransformation &T, const IntegrationRule &ir)
{
   if (T.mesh == GridFunc->FESpace()->GetMesh())
   {
      GridFunc->GetVectorValues(T, ir, M);
   }
   else
   {
      VectorCoefficient::Eval(M, T, ir);
   }
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
   Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
   if (T.mesh == gf_mesh)
   {
      GridFunc->GetGradient(T, V);
   }
   else
   {
      IntegrationPoint coarse_ip;
      ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      GridFunc->GetGradient(*coarse_T, V);
   }
}

void GradientGridFunctionCoefficient::Eval(
   DenseMatrix &M, ElementTransformation &T, const IntegrationRule &ir)
{
   if (T.mesh == GridFunc->FESpace()->GetMesh())
   {
      GridFunc->GetGradients(T, ir, M);
   }
   else
   {
      VectorCoefficient::Eval(M, T, ir);
   }
}

CurlGridFunctionCoefficient::CurlGridFunctionCoefficient(
   const GridFunction *gf)
   : VectorCoefficient(0)
{
   SetGridFunction(gf);
}

void CurlGridFunctionCoefficient::SetGridFunction(const GridFunction *gf)
{
   GridFunc = gf; vdim = (gf) ? gf -> CurlDim() : 0;
}

void CurlGridFunctionCoefficient::Eval(Vector &V, ElementTransformation &T,
                                       const IntegrationPoint &ip)
{
   Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
   if (T.mesh == gf_mesh)
   {
      GridFunc->GetCurl(T, V);
   }
   else
   {
      IntegrationPoint coarse_ip;
      ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      GridFunc->GetCurl(*coarse_T, V);
   }
}

DivergenceGridFunctionCoefficient::DivergenceGridFunctionCoefficient (
   const GridFunction *gf) : Coefficient()
{
   GridFunc = gf;
}

double DivergenceGridFunctionCoefficient::Eval(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   Mesh *gf_mesh = GridFunc->FESpace()->GetMesh();
   if (T.mesh == gf_mesh)
   {
      return GridFunc->GetDivergence(T);
   }
   else
   {
      IntegrationPoint coarse_ip;
      ElementTransformation *coarse_T = RefinedToCoarse(*gf_mesh, T, ip, coarse_ip);
      return GridFunc->GetDivergence(*coarse_T);
   }
}

void VectorDeltaCoefficient::SetTime(double t)
{
   d.SetTime(t);
   this->VectorCoefficient::SetTime(t);
}

void VectorDeltaCoefficient::SetDirection(const Vector &d_)
{
   dir = d_;
   (*this).vdim = dir.Size();
}

void VectorDeltaCoefficient::EvalDelta(
   Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
{
   V = dir;
   d.SetTime(GetTime());
   V *= d.EvalDelta(T, ip);
}

void VectorRestrictedCoefficient::SetTime(double t)
{
   if (c) { c->SetTime(t); }
   this->VectorCoefficient::SetTime(t);
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

void PWMatrixCoefficient::InitMap(const Array<int> & attr,
                                  const Array<MatrixCoefficient*> & coefs)
{
   MFEM_VERIFY(attr.Size() == coefs.Size(),
               "PWMatrixCoefficient:  "
               "Attribute and coefficient arrays have incompatible "
               "dimensions.");

   for (int i=0; i<attr.Size(); i++)
   {
      if (coefs[i] != NULL)
      {
         UpdateCoefficient(attr[i], *coefs[i]);
      }
   }
}

void PWMatrixCoefficient::UpdateCoefficient(int attr, MatrixCoefficient & coef)
{
   MFEM_VERIFY(coef.GetHeight() == height,
               "PWMatrixCoefficient::UpdateCoefficient:  "
               "MatrixCoefficient has incompatible height.");
   MFEM_VERIFY(coef.GetWidth() == width,
               "PWMatrixCoefficient::UpdateCoefficient:  "
               "MatrixCoefficient has incompatible width.");
   if (symmetric)
   {
      MFEM_VERIFY(coef.IsSymmetric(),
                  "PWMatrixCoefficient::UpdateCoefficient:  "
                  "MatrixCoefficient has incompatible symmetry.");
   }
   pieces[attr] = &coef;
}

void PWMatrixCoefficient::SetTime(double t)
{
   MatrixCoefficient::SetTime(t);

   std::map<int, MatrixCoefficient*>::iterator p = pieces.begin();
   for (; p != pieces.end(); p++)
   {
      if (p->second != NULL)
      {
         p->second->SetTime(t);
      }
   }
}

void PWMatrixCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                               const IntegrationPoint &ip)
{
   const int att = T.Attribute;
   std::map<int, MatrixCoefficient*>::const_iterator p = pieces.find(att);
   if (p != pieces.end())
   {
      if ( p->second != NULL)
      {
         p->second->Eval(K, T, ip);
         return;
      }
   }

   K.SetSize(height, width);
   K = 0.0;
}

void MatrixFunctionCoefficient::SetTime(double t)
{
   if (Q) { Q->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
}

void MatrixFunctionCoefficient::Eval(DenseMatrix &K, ElementTransformation &T,
                                     const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   K.SetSize(height, width);

   if (symmetric) // Use SymmFunction (deprecated version)
   {
      MFEM_VERIFY(height == width && SymmFunction,
                  "MatrixFunctionCoefficient is not symmetric");

      Vector Ksym((width * (width + 1)) / 2); // 1x1: 1, 2x2: 3, 3x3: 6

      SymmFunction(transip, Ksym);

      // Copy upper triangular values from Ksym to the full matrix K
      int os = 0;
      for (int i=0; i<height; ++i)
      {
         for (int j=i; j<width; ++j)
         {
            const double Kij = Ksym[j - i + os];
            K(i,j) = Kij;
            if (j != i) { K(j,i) = Kij; }
         }

         os += width - i;
      }
   }
   else
   {
      if (Function)
      {
         Function(transip, K);
      }
      else if (TDFunction)
      {
         TDFunction(transip, GetTime(), K);
      }
      else
      {
         K = mat;
      }
   }

   if (Q)
   {
      K *= Q->Eval(T, ip, GetTime());
   }
}

void MatrixFunctionCoefficient::EvalSymmetric(Vector &K,
                                              ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   MFEM_VERIFY(symmetric && height == width && SymmFunction,
               "MatrixFunctionCoefficient is not symmetric");

   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   K.SetSize((width * (width + 1)) / 2); // 1x1: 1, 2x2: 3, 3x3: 6

   if (SymmFunction)
   {
      SymmFunction(transip, K);
   }

   if (Q)
   {
      K *= Q->Eval(T, ip, GetTime());
   }
}

void SymmetricMatrixFunctionCoefficient::SetTime(double t)
{
   if (Q) { Q->SetTime(t); }
   this->SymmetricMatrixCoefficient::SetTime(t);
}

void SymmetricMatrixFunctionCoefficient::Eval(DenseSymmetricMatrix &K,
                                              ElementTransformation &T,
                                              const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   T.Transform(ip, transip);

   K.SetSize(dim);

   if (Function)
   {
      Function(transip, K);
   }
   else if (TDFunction)
   {
      TDFunction(transip, GetTime(), K);
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

void MatrixArrayCoefficient::SetTime(double t)
{
   for (int i=0; i < height*width; i++)
   {
      if (Coeff[i]) { Coeff[i]->SetTime(t); }
   }
   this->MatrixCoefficient::SetTime(t);
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
   K.SetSize(height, width);
   for (int i = 0; i < height; i++)
   {
      for (int j = 0; j < width; j++)
      {
         K(i,j) = this->Eval(i, j, T, ip);
      }
   }
}

void MatrixRestrictedCoefficient::SetTime(double t)
{
   if (c) { c->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
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

void SumCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->Coefficient::SetTime(t);
}

void ProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->Coefficient::SetTime(t);
}

void RatioCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->Coefficient::SetTime(t);
}

void PowerCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   this->Coefficient::SetTime(t);
}

InnerProductCoefficient::InnerProductCoefficient(VectorCoefficient &A,
                                                 VectorCoefficient &B)
   : a(&A), b(&B), va(A.GetVDim()), vb(B.GetVDim())
{
   MFEM_ASSERT(A.GetVDim() == B.GetVDim(),
               "InnerProductCoefficient:  "
               "Arguments have incompatible dimensions.");
}

void InnerProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->Coefficient::SetTime(t);
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

void VectorRotProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->Coefficient::SetTime(t);
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

void DeterminantCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   this->Coefficient::SetTime(t);
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

VectorSumCoefficient::VectorSumCoefficient(VectorCoefficient &A_,
                                           VectorCoefficient &B_,
                                           double alpha_, double beta_)
   : VectorCoefficient(A_.GetVDim()),
     ACoef(&A_), BCoef(&B_),
     A(A_.GetVDim()), B(A_.GetVDim()),
     alphaCoef(NULL), betaCoef(NULL),
     alpha(alpha_), beta(beta_)
{
   MFEM_ASSERT(A_.GetVDim() == B_.GetVDim(),
               "VectorSumCoefficient:  "
               "Arguments must have the same dimension.");
}

VectorSumCoefficient::VectorSumCoefficient(VectorCoefficient &A_,
                                           VectorCoefficient &B_,
                                           Coefficient &alpha_,
                                           Coefficient &beta_)
   : VectorCoefficient(A_.GetVDim()),
     ACoef(&A_), BCoef(&B_),
     A(A_.GetVDim()),
     B(A_.GetVDim()),
     alphaCoef(&alpha_),
     betaCoef(&beta_),
     alpha(0.0), beta(0.0)
{
   MFEM_ASSERT(A_.GetVDim() == B_.GetVDim(),
               "VectorSumCoefficient:  "
               "Arguments must have the same dimension.");
}

void VectorSumCoefficient::SetTime(double t)
{
   if (ACoef) { ACoef->SetTime(t); }
   if (BCoef) { BCoef->SetTime(t); }
   if (alphaCoef) { alphaCoef->SetTime(t); }
   if (betaCoef) { betaCoef->SetTime(t); }
   this->VectorCoefficient::SetTime(t);
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

void ScalarVectorProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->VectorCoefficient::SetTime(t);
}

void ScalarVectorProductCoefficient::Eval(Vector &V, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   double sa = (a == NULL) ? aConst : a->Eval(T, ip);
   b->Eval(V, T, ip);
   V *= sa;
}

NormalizedVectorCoefficient::NormalizedVectorCoefficient(VectorCoefficient &A,
                                                         double tol_)
   : VectorCoefficient(A.GetVDim()), a(&A), tol(tol_)
{}

void NormalizedVectorCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   this->VectorCoefficient::SetTime(t);
}

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

void VectorCrossProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->VectorCoefficient::SetTime(t);
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

void MatrixVectorProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->VectorCoefficient::SetTime(t);
}

void MatrixVectorProductCoefficient::Eval(Vector &V, ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
   a->Eval(ma, T, ip);
   b->Eval(vb, T, ip);
   V.SetSize(vdim);
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
                                           double alpha_, double beta_)
   : MatrixCoefficient(A.GetHeight(), A.GetWidth()),
     a(&A), b(&B), alpha(alpha_), beta(beta_),
     ma(A.GetHeight(), A.GetWidth())
{
   MFEM_ASSERT(A.GetHeight() == B.GetHeight() && A.GetWidth() == B.GetWidth(),
               "MatrixSumCoefficient:  "
               "Arguments must have the same dimensions.");
}

void MatrixSumCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
}

void MatrixSumCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   b->Eval(M, T, ip);
   if ( beta != 1.0 ) { M *= beta; }
   a->Eval(ma, T, ip);
   M.Add(alpha, ma);
}

MatrixProductCoefficient::MatrixProductCoefficient(MatrixCoefficient &A,
                                                   MatrixCoefficient &B)
   : MatrixCoefficient(A.GetHeight(), B.GetWidth()),
     a(&A), b(&B),
     ma(A.GetHeight(), A.GetWidth()),
     mb(B.GetHeight(), B.GetWidth())
{
   MFEM_ASSERT(A.GetWidth() == B.GetHeight(),
               "MatrixProductCoefficient:  "
               "Arguments must have compatible dimensions.");
}

void MatrixProductCoefficient::Eval(DenseMatrix &M, ElementTransformation &T,
                                    const IntegrationPoint &ip)
{
   a->Eval(ma, T, ip);
   b->Eval(mb, T, ip);
   Mult(ma, mb, M);
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

void ScalarMatrixProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
}

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

void TransposeMatrixCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
}

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

void InverseMatrixCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
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

void OuterProductCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (b) { b->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
}

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

CrossCrossCoefficient::CrossCrossCoefficient(double A, VectorCoefficient &K)
   : MatrixCoefficient(K.GetVDim(), K.GetVDim()), aConst(A), a(NULL), k(&K),
     vk(K.GetVDim())
{}

CrossCrossCoefficient::CrossCrossCoefficient(Coefficient &A,
                                             VectorCoefficient &K)
   : MatrixCoefficient(K.GetVDim(), K.GetVDim()), aConst(0.0), a(&A), k(&K),
     vk(K.GetVDim())
{}

void CrossCrossCoefficient::SetTime(double t)
{
   if (a) { a->SetTime(t); }
   if (k) { k->SetTime(t); }
   this->MatrixCoefficient::SetTime(t);
}

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

void VectorQuadratureFunctionCoefficient::SetComponent(int index_, int length_)
{
   MFEM_VERIFY(index_ >= 0, "Index must be >= 0");
   MFEM_VERIFY(index_ < QuadF.GetVDim(),
               "Index must be < QuadratureFunction length");
   index = index_;

   MFEM_VERIFY(length_ > 0, "Length must be > 0");
   MFEM_VERIFY(length_ <= QuadF.GetVDim() - index,
               "Length must be <= (QuadratureFunction length - index)");

   vdim = length_;
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
