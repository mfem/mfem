// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Finite Element Base classes

#include "fe_base.hpp"
#include "face_map_utils.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

DofToQuad DofToQuad::Abs() const
{
   DofToQuad d2q(*this);
   d2q.B.Abs();
   d2q.Bt.Abs();
   d2q.G.Abs();
   d2q.Gt.Abs();
   return d2q;
}

FiniteElement::FiniteElement(int D, Geometry::Type G,
                             int Do, int O, int F)
   : Nodes(Do)
{
   dim = D ; geom_type = G ; dof = Do ; order = O ; func_space = F;
   vdim = 0 ; cdim = 0;
   range_type = SCALAR;
   map_type = VALUE;
   deriv_type = NONE;
   deriv_range_type = SCALAR;
   deriv_map_type = VALUE;
   for (int i = 0; i < Geometry::MaxDim; i++) { orders[i] = -1; }
#ifndef MFEM_THREAD_SAFE
   vshape.SetSize(dof, dim);
#endif
}

void FiniteElement::CalcVShape(
   const IntegrationPoint &ip, DenseMatrix &shape) const
{
   MFEM_ABORT("method is not implemented for this class");
}

void FiniteElement::CalcVShape(
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   MFEM_ABORT("method is not implemented for this class");
}

void FiniteElement::CalcDivShape(
   const IntegrationPoint &ip, Vector &divshape) const
{
   MFEM_ABORT("method is not implemented for this class");
}

void FiniteElement::CalcPhysDivShape(
   ElementTransformation &Trans, Vector &div_shape) const
{
   CalcDivShape(Trans.GetIntPoint(), div_shape);
   div_shape *= (1.0 / Trans.Weight());
}

void FiniteElement::CalcCurlShape(const IntegrationPoint &ip,
                                  DenseMatrix &curl_shape) const
{
   MFEM_ABORT("method is not implemented for this class");
}

void FiniteElement::CalcPhysCurlShape(ElementTransformation &Trans,
                                      DenseMatrix &curl_shape) const
{
   switch (dim)
   {
      case 3:
      {
#ifdef MFEM_THREAD_SAFE
         DenseMatrix vshape(dof, dim);
#endif
         CalcCurlShape(Trans.GetIntPoint(), vshape);
         MultABt(vshape, Trans.Jacobian(), curl_shape);
         curl_shape *= (1.0 / Trans.Weight());
         break;
      }
      case 2:
         // This is valid for both 2x2 and 3x2 Jacobians
         CalcCurlShape(Trans.GetIntPoint(), curl_shape);
         curl_shape *= (1.0 / Trans.Weight());
         break;
      default:
         MFEM_ABORT("Invalid dimension, Dim = " << dim);
   }
}

void FiniteElement::GetFaceDofs(int face, int **dofs, int *ndofs) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::CalcHessian(const IntegrationPoint &ip,
                                DenseMatrix &h) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::GetLocalInterpolation(ElementTransformation &Trans,
                                          DenseMatrix &I) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::GetLocalRestriction(ElementTransformation &,
                                        DenseMatrix &) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::GetTransferMatrix(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &I) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::Project(
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::ProjectFromNodes(Vector &vc, ElementTransformation &Trans,
                                     Vector &dofs) const
{
   mfem_error("FiniteElement::ProjectFromNodes() (vector) is not overloaded!");
}

void FiniteElement::ProjectMatrixCoefficient(
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   MFEM_ABORT("method is not overloaded");
}

void FiniteElement::ProjectDelta(int vertex, Vector &dofs) const
{
   MFEM_ABORT("method is not implemented for this element");
}

void FiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   MFEM_ABORT("method is not implemented for this element");
}

void FiniteElement::ProjectGrad(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &grad) const
{
   MFEM_ABORT("method is not implemented for this element");
}

void FiniteElement::ProjectCurl(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &curl) const
{
   MFEM_ABORT("method is not implemented for this element");
}

void FiniteElement::ProjectDiv(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &div) const
{
   MFEM_ABORT("method is not implemented for this element");
}

void FiniteElement::CalcPhysShape(ElementTransformation &Trans,
                                  Vector &shape) const
{
   CalcShape(Trans.GetIntPoint(), shape);
   if (map_type == INTEGRAL)
   {
      shape /= Trans.Weight();
   }
}

void FiniteElement::CalcPhysDShape(ElementTransformation &Trans,
                                   DenseMatrix &dshape) const
{
   MFEM_ASSERT(map_type == VALUE, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif
   CalcDShape(Trans.GetIntPoint(), vshape);
   Mult(vshape, Trans.InverseJacobian(), dshape);
}

void FiniteElement::CalcPhysLaplacian(ElementTransformation &Trans,
                                      Vector &Laplacian) const
{
   MFEM_ASSERT(map_type == VALUE, "");

   // Simpler routine if mapping is affine
   if (Trans.Hessian().FNorm2() < 1e-20)
   {
      CalcPhysLinLaplacian(Trans, Laplacian);
      return;
   }

   // Compute full Hessian first if non-affine
   int size = (dim*(dim+1))/2;
   DenseMatrix hess(dof, size);
   CalcPhysHessian(Trans,hess);

   if (dim == 3)
   {
      for (int nd = 0; nd < dof; nd++)
      {
         Laplacian[nd] = hess(nd,0) + hess(nd,4) + hess(nd,5);
      }
   }
   else if (dim == 2)
   {
      for (int nd = 0; nd < dof; nd++)
      {
         Laplacian[nd] = hess(nd,0) + hess(nd,2);
      }
   }
   else
   {
      for (int nd = 0; nd < dof; nd++)
      {
         Laplacian[nd] = hess(nd,0);
      }
   }
}

// Assume a linear mapping
void FiniteElement::CalcPhysLinLaplacian(ElementTransformation &Trans,
                                         Vector &Laplacian) const
{
   MFEM_ASSERT(map_type == VALUE, "");
   int size = (dim*(dim+1))/2;
   DenseMatrix hess(dof, size);
   DenseMatrix Gij(dim,dim);
   Vector scale(size);

   CalcHessian(Trans.GetIntPoint(), hess);
   MultAAt(Trans.InverseJacobian(), Gij);

   if (dim == 3)
   {
      scale[0] =   Gij(0,0);
      scale[1] = 2*Gij(0,1);
      scale[2] = 2*Gij(0,2);

      scale[3] = 2*Gij(1,2);
      scale[4] =   Gij(2,2);

      scale[5] =   Gij(1,1);
   }
   else if (dim == 2)
   {
      scale[0] =   Gij(0,0);
      scale[1] = 2*Gij(0,1);
      scale[2] =   Gij(1,1);
   }
   else
   {
      scale[0] =   Gij(0,0);
   }

   for (int nd = 0; nd < dof; nd++)
   {
      Laplacian[nd] = 0.0;
      for (int ii = 0; ii < size; ii++)
      {
         Laplacian[nd] += hess(nd,ii)*scale[ii];
      }
   }
}

void  FiniteElement::CalcPhysHessian(ElementTransformation &Trans,
                                     DenseMatrix& Hessian) const
{
   MFEM_ASSERT(map_type == VALUE, "");

   // Roll 2-Tensors in vectors and 4-Tensor in Matrix, exploiting symmetry
   Array<int> map(dim*dim);
   if (dim == 3)
   {
      map[0] = 0;
      map[1] = 1;
      map[2] = 2;

      map[3] = 1;
      map[4] = 5;
      map[5] = 3;

      map[6] = 2;
      map[7] = 3;
      map[8] = 4;
   }
   else if (dim == 2)
   {
      map[0] = 0;
      map[1] = 1;

      map[2] = 1;
      map[3] = 2;
   }
   else
   {
      map[0] = 0;
   }

   // Hessian in ref coords
   int size = (dim*(dim+1))/2;
   DenseMatrix hess(dof, size);
   CalcHessian(Trans.GetIntPoint(), hess);

   // Gradient in physical coords
   if (Trans.Hessian().FNorm2() > 1e-10)
   {
      DenseMatrix grad(dof, dim);
      CalcPhysDShape(Trans, grad);
      DenseMatrix gmap(dof, size);
      Mult(grad,Trans.Hessian(),gmap);
      hess -= gmap;
   }

   // LHM
   DenseMatrix lhm(size,size);
   DenseMatrix invJ = Trans.Jacobian();
   lhm = 0.0;
   for (int i = 0; i < dim; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         for (int k = 0; k < dim; k++)
         {
            for (int l = 0; l < dim; l++)
            {
               lhm(map[i*dim+j],map[k*dim+l]) += invJ(i,k)*invJ(j,l);
            }
         }
      }
   }
   // Correct multiplicity
   Vector mult(size);
   mult = 0.0;
   for (int i = 0; i < dim*dim; i++) { mult[map[i]]++; }
   lhm.InvRightScaling(mult);

   // Hessian in physical coords
   lhm.Invert();
   Mult(hess, lhm, Hessian);
}

const DofToQuad &FiniteElement::GetDofToQuad(const IntegrationRule &ir,
                                             DofToQuad::Mode mode) const
{
   DofToQuad *d2q = nullptr;
   MFEM_VERIFY(mode == DofToQuad::FULL, "invalid mode requested");

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   #pragma omp critical (DofToQuad)
#endif
   {
      for (int i = 0; i < dof2quad_array.Size(); i++)
      {
         d2q = dof2quad_array[i];
         if (d2q->IntRule != &ir || d2q->mode != mode) { d2q = nullptr; }
      }
      if (!d2q)
      {
#ifdef MFEM_THREAD_SAFE
         DenseMatrix vshape(dof, dim);
#endif
         d2q = new DofToQuad;
         const int nqpt = ir.GetNPoints();
         d2q->FE = this;
         d2q->IntRule = &ir;
         d2q->mode = mode;
         d2q->ndof = dof;
         d2q->nqpt = nqpt;
         switch (range_type)
         {
            case SCALAR:
            {
               d2q->B.SetSize(nqpt*dof);
               d2q->Bt.SetSize(dof*nqpt);

               Vector shape;
               vshape.GetColumnReference(0, shape);
               for (int i = 0; i < nqpt; i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  CalcShape(ip, shape);
                  for (int j = 0; j < dof; j++)
                  {
                     d2q->B[i+nqpt*j] = d2q->Bt[j+dof*i] = shape(j);
                  }
               }
               break;
            }
            case VECTOR:
            {
               d2q->B.SetSize(nqpt*dim*dof);
               d2q->Bt.SetSize(dof*nqpt*dim);

               for (int i = 0; i < nqpt; i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  CalcVShape(ip, vshape);
                  for (int d = 0; d < dim; d++)
                  {
                     for (int j = 0; j < dof; j++)
                     {
                        d2q->B[i+nqpt*(d+dim*j)] =
                           d2q->Bt[j+dof*(i+nqpt*d)] = vshape(j, d);
                     }
                  }
               }
               break;
            }
            case UNKNOWN_RANGE_TYPE:
               // Skip B and Bt for unknown range type
               break;
         }
         switch (deriv_type)
         {
            case GRAD:
            {
               d2q->G.SetSize(nqpt*dim*dof);
               d2q->Gt.SetSize(dof*nqpt*dim);

               for (int i = 0; i < nqpt; i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  CalcDShape(ip, vshape);
                  for (int d = 0; d < dim; d++)
                  {
                     for (int j = 0; j < dof; j++)
                     {
                        d2q->G[i+nqpt*(d+dim*j)] =
                           d2q->Gt[j+dof*(i+nqpt*d)] = vshape(j, d);
                     }
                  }
               }
               break;
            }
            case DIV:
            {
               d2q->G.SetSize(nqpt*dof);
               d2q->Gt.SetSize(dof*nqpt);

               Vector divshape;
               vshape.GetColumnReference(0, divshape);
               for (int i = 0; i < nqpt; i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  CalcDivShape(ip, divshape);
                  for (int j = 0; j < dof; j++)
                  {
                     d2q->G[i+nqpt*j] = d2q->Gt[j+dof*i] = divshape(j);
                  }
               }
               break;
            }
            case CURL:
            {
               d2q->G.SetSize(nqpt*cdim*dof);
               d2q->Gt.SetSize(dof*nqpt*cdim);

               DenseMatrix curlshape(vshape.GetData(), dof, cdim);  // cdim <= dim
               for (int i = 0; i < nqpt; i++)
               {
                  const IntegrationPoint &ip = ir.IntPoint(i);
                  CalcCurlShape(ip, curlshape);
                  for (int d = 0; d < cdim; d++)
                  {
                     for (int j = 0; j < dof; j++)
                     {
                        d2q->G[i+nqpt*(d+cdim*j)] =
                           d2q->Gt[j+dof*(i+nqpt*d)] = curlshape(j, d);
                     }
                  }
               }
               break;
            }
            case NONE:
               // Skip G and Gt for unknown derivative type
               break;
         }
         dof2quad_array.Append(d2q);
      }
   }
   return *d2q;
}

void FiniteElement::GetFaceMap(const int face_id,
                               Array<int> &face_map) const
{
   MFEM_ABORT("method is not implemented for this element");
}

FiniteElement::~FiniteElement()
{
   for (int i = 0; i < dof2quad_array.Size(); i++)
   {
      delete dof2quad_array[i];
   }
}


void ScalarFiniteElement::NodalLocalInterpolation(
   ElementTransformation &Trans, DenseMatrix &I,
   const ScalarFiniteElement &fine_fe) const
{
   real_t v[Geometry::MaxDim];
   Vector vv(v, dim);
   IntegrationPoint f_ip;

#ifdef MFEM_THREAD_SAFE
   Vector shape(dof);
#else
   Vector shape;
   vshape.GetColumnReference(0, shape);
#endif

   MFEM_ASSERT(map_type == fine_fe.GetMapType(), "");

   I.SetSize(fine_fe.dof, dof);
   for (int i = 0; i < fine_fe.dof; i++)
   {
      Trans.Transform(fine_fe.Nodes.IntPoint(i), vv);
      f_ip.Set(v, dim);
      CalcShape(f_ip, shape);
      for (int j = 0; j < dof; j++)
      {
         if (fabs(I(i,j) = shape(j)) < 1.0e-12)
         {
            I(i,j) = 0.0;
         }
      }
   }
   if (map_type == INTEGRAL)
   {
      // assuming Trans is linear; this should be ok for all refinement types
      Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
      I *= Trans.Weight();
   }
}

void ScalarFiniteElement::ScalarLocalInterpolation(
   ElementTransformation &Trans, DenseMatrix &I,
   const ScalarFiniteElement &fine_fe) const
{
   // General "interpolation", defined by L2 projection

   real_t v[Geometry::MaxDim];
   Vector vv(v, dim);
   IntegrationPoint f_ip;

   const int fs = fine_fe.GetDof(), cs = this->GetDof();
   I.SetSize(fs, cs);
   Vector fine_shape(fs), coarse_shape(cs);
   DenseMatrix fine_mass(fs), fine_coarse_mass(fs, cs); // initialized with 0
   const int ir_order =
      std::max(GetOrder(), fine_fe.GetOrder()) + fine_fe.GetOrder();
   const IntegrationRule &ir = IntRules.Get(fine_fe.GetGeomType(), ir_order);

   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      fine_fe.CalcShape(ip, fine_shape);
      Trans.Transform(ip, vv);
      f_ip.Set(v, dim);
      this->CalcShape(f_ip, coarse_shape);

      AddMult_a_VVt(ip.weight, fine_shape, fine_mass);
      AddMult_a_VWt(ip.weight, fine_shape, coarse_shape, fine_coarse_mass);
   }

   DenseMatrixInverse fine_mass_inv(fine_mass);
   fine_mass_inv.Mult(fine_coarse_mass, I);

   if (map_type == INTEGRAL)
   {
      // assuming Trans is linear; this should be ok for all refinement types
      Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
      I *= Trans.Weight();
   }
}

void ScalarFiniteElement::ScalarLocalL2Restriction(
   ElementTransformation &Trans, DenseMatrix &R,
   const ScalarFiniteElement &coarse_fe) const
{
   // General "restriction", defined by L2 projection
   real_t v[Geometry::MaxDim];
   Vector vv(v, dim);

   const int cs = coarse_fe.GetDof(), fs = this->GetDof();
   R.SetSize(cs, fs);
   Vector fine_shape(fs), coarse_shape(cs);
   DenseMatrix coarse_mass(cs), coarse_fine_mass(cs, fs); // initialized with 0
   const int ir_order = GetOrder() + coarse_fe.GetOrder();
   const IntegrationRule &ir = IntRules.Get(coarse_fe.GetGeomType(), ir_order);

   // integrate coarse_mass in the coarse space
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &c_ip = ir.IntPoint(i);
      coarse_fe.CalcShape(c_ip, coarse_shape);
      AddMult_a_VVt(c_ip.weight, coarse_shape, coarse_mass);
   }

   // integrate coarse_fine_mass in the fine space
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &f_ip = ir.IntPoint(i);
      this->CalcShape(f_ip, fine_shape);
      Trans.Transform(f_ip, vv);

      IntegrationPoint c_ip;
      c_ip.Set(v, dim);
      coarse_fe.CalcShape(c_ip, coarse_shape);
      AddMult_a_VWt(f_ip.weight*Trans.Weight(), coarse_shape, fine_shape,
                    coarse_fine_mass);
   }

   DenseMatrixInverse coarse_mass_inv(coarse_mass);
   coarse_mass_inv.Mult(coarse_fine_mass, R);

   if (map_type == INTEGRAL)
   {
      // assuming Trans is linear; this should be ok for all refinement types
      Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
      R *= 1.0 / Trans.Weight();
   }
}

void NodalFiniteElement::CreateLexicographicFullMap(const IntegrationRule &ir)
const
{
   // Get the FULL version of the map.
   auto &d2q = GetDofToQuad(ir, DofToQuad::FULL);
   //Undo the native ordering which is what FiniteElement::GetDofToQuad returns.
   auto *d2q_new = new DofToQuad(d2q);
   d2q_new->mode = DofToQuad::LEXICOGRAPHIC_FULL;
   const int nqpt = ir.GetNPoints();

   const int b_dim = (range_type == VECTOR) ? dim : 1;

   for (int i = 0; i < nqpt; i++)
   {
      for (int d = 0; d < b_dim; d++)
      {
         for (int j = 0; j < dof; j++)
         {
            const double val = d2q.B[i + nqpt*(d+b_dim*lex_ordering[j])];
            d2q_new->B[i+nqpt*(d+b_dim*j)] = val;
            d2q_new->Bt[j+dof*(i+nqpt*d)] = val;
         }
      }
   }

   const int g_dim = [this]()
   {
      switch (deriv_type)
      {
         case GRAD: return dim;
         case DIV: return 1;
         case CURL: return cdim;
         default: return 0;
      }
   }();

   for (int i = 0; i < nqpt; i++)
   {
      for (int d = 0; d < g_dim; d++)
      {
         for (int j = 0; j < dof; j++)
         {
            const double val = d2q.G[i + nqpt*(d+g_dim*lex_ordering[j])];
            d2q_new->G[i+nqpt*(d+g_dim*j)] = val;
            d2q_new->Gt[j+dof*(i+nqpt*d)] = val;
         }
      }
   }

   dof2quad_array.Append(d2q_new);
}

const DofToQuad &NodalFiniteElement::GetDofToQuad(const IntegrationRule &ir,
                                                  DofToQuad::Mode mode) const
{
   //Should make this loop a function of FiniteElement
   for (int i = 0; i < dof2quad_array.Size(); i++)
   {
      const DofToQuad &d2q = *dof2quad_array[i];
      if (d2q.IntRule == &ir && d2q.mode == mode) { return d2q; }
   }

   if (mode != DofToQuad::LEXICOGRAPHIC_FULL)
   {
      return FiniteElement::GetDofToQuad(ir, mode);
   }
   else
   {
      CreateLexicographicFullMap(ir);
      return NodalFiniteElement::GetDofToQuad(ir, mode);
   }
}

void NodalFiniteElement::ProjectCurl_2D(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), 1);

   curl.SetSize(dof, fe.GetDof());
   for (int i = 0; i < dof; i++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(i), curl_shape);

      real_t w = 1.0;
      if (GetMapType() == FiniteElement::VALUE)
      {
         Trans.SetIntPoint(&Nodes.IntPoint(i));
         w /= Trans.Weight();
      }
      for (int j = 0; j < fe.GetDof(); j++)
      {
         curl(i,j) = w * curl_shape(j,0);
      }
   }
}

void InvertLinearTrans(ElementTransformation &trans,
                       const IntegrationPoint &pt, Vector &x)
{
   // invert a linear transform with one Newton step
   IntegrationPoint p0;
   p0.Set3(0, 0, 0);
   trans.Transform(p0, x);

   real_t store[3];
   Vector v(store, x.Size());
   pt.Get(store, x.Size());
   v -= x;

   trans.InverseJacobian().Mult(v, x);
}

void NodalFiniteElement::GetLocalRestriction(ElementTransformation &Trans,
                                             DenseMatrix &R) const
{
   IntegrationPoint ipt;
   Vector pt(&ipt.x, dim);

#ifdef MFEM_THREAD_SAFE
   Vector shape(dof);
#else
   Vector shape;
   vshape.GetColumnReference(0, shape);
#endif

   Trans.SetIntPoint(&Nodes[0]);

   for (int j = 0; j < dof; j++)
   {
      InvertLinearTrans(Trans, Nodes[j], pt);
      if (Geometries.CheckPoint(geom_type, ipt)) // do we need an epsilon here?
      {
         CalcShape(ipt, shape);
         R.SetRow(j, shape);
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}

void NodalFiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   for (int i = 0; i < dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      // some coefficients expect that Trans.IntPoint is the same
      // as the second argument of Eval
      Trans.SetIntPoint(&ip);
      dofs(i) = coeff.Eval(Trans, ip);
      if (map_type == INTEGRAL)
      {
         dofs(i) *= Trans.Weight();
      }
   }
}

void NodalFiniteElement::Project(
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*dof, "");
   Vector x(vc.GetVDim());

   for (int i = 0; i < dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      vc.Eval (x, Trans, ip);
      if (map_type == INTEGRAL)
      {
         x *= Trans.Weight();
      }
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(dof*j+i) = x(j);
      }
   }
}

void NodalFiniteElement::ProjectMatrixCoefficient(
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   // (mc.height x mc.width) @ DOFs -> (dof x mc.width x mc.height) in dofs
   MFEM_ASSERT(dofs.Size() == mc.GetHeight()*mc.GetWidth()*dof, "");
   DenseMatrix MQ(mc.GetHeight(), mc.GetWidth());

   for (int k = 0; k < dof; k++)
   {
      T.SetIntPoint(&Nodes.IntPoint(k));
      mc.Eval(MQ, T, Nodes.IntPoint(k));
      if (map_type == INTEGRAL) { MQ *= T.Weight(); }
      for (int r = 0; r < MQ.Height(); r++)
      {
         for (int d = 0; d < MQ.Width(); d++)
         {
            dofs(k+dof*(d+MQ.Width()*r)) = MQ(r,d);
         }
      }
   }
}

void NodalFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      Vector shape(fe.GetDof());

      I.SetSize(dof, fe.GetDof());
      if (map_type == fe.GetMapType())
      {
         for (int k = 0; k < dof; k++)
         {
            fe.CalcShape(Nodes.IntPoint(k), shape);
            for (int j = 0; j < shape.Size(); j++)
            {
               I(k,j) = (fabs(shape(j)) < 1e-12) ? 0.0 : shape(j);
            }
         }
      }
      else
      {
         for (int k = 0; k < dof; k++)
         {
            Trans.SetIntPoint(&Nodes.IntPoint(k));
            fe.CalcPhysShape(Trans, shape);
            if (map_type == INTEGRAL)
            {
               shape *= Trans.Weight();
            }
            for (int j = 0; j < shape.Size(); j++)
            {
               I(k,j) = (fabs(shape(j)) < 1e-12) ? 0.0 : shape(j);
            }
         }
      }
   }
   else
   {
      DenseMatrix vshape(fe.GetDof(), std::max(Trans.GetSpaceDim(),
                                               fe.GetRangeDim()));

      I.SetSize(vshape.Width()*dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         Trans.SetIntPoint(&Nodes.IntPoint(k));
         fe.CalcVShape(Trans, vshape);
         if (map_type == INTEGRAL)
         {
            vshape *= Trans.Weight();
         }
         for (int j = 0; j < vshape.Height(); j++)
            for (int d = 0; d < vshape.Width(); d++)
            {
               I(k+d*dof,j) = vshape(j,d);
            }
      }
   }
}

void NodalFiniteElement::ProjectGrad(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &grad) const
{
   MFEM_ASSERT(fe.GetMapType() == VALUE, "");
   MFEM_ASSERT(Trans.GetSpaceDim() == dim, "")

   DenseMatrix dshape(fe.GetDof(), dim), grad_k(fe.GetDof(), dim), Jinv(dim);

   grad.SetSize(dim*dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      fe.CalcDShape(ip, dshape);
      Trans.SetIntPoint(&ip);
      CalcInverse(Trans.Jacobian(), Jinv);
      Mult(dshape, Jinv, grad_k);
      if (map_type == INTEGRAL)
      {
         grad_k *= Trans.Weight();
      }
      for (int j = 0; j < grad_k.Height(); j++)
         for (int d = 0; d < dim; d++)
         {
            grad(k+d*dof,j) = grad_k(j,d);
         }
   }
}

void NodalFiniteElement::ProjectDiv(
   const FiniteElement &fe, ElementTransformation &Trans,
   DenseMatrix &div) const
{
   real_t detJ;
   Vector div_shape(fe.GetDof());

   div.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);
      fe.CalcDivShape(ip, div_shape);
      if (map_type == VALUE)
      {
         Trans.SetIntPoint(&ip);
         detJ = Trans.Weight();
         for (int j = 0; j < div_shape.Size(); j++)
         {
            div(k,j) = (fabs(div_shape(j)) < 1e-12) ? 0.0 : div_shape(j)/detJ;
         }
      }
      else
      {
         for (int j = 0; j < div_shape.Size(); j++)
         {
            div(k,j) = (fabs(div_shape(j)) < 1e-12) ? 0.0 : div_shape(j);
         }
      }
   }
}

void NodalFiniteElement::ReorderLexToNative(int ncomp,
                                            Vector &dofs) const
{
   MFEM_ASSERT(lex_ordering.Size() == dof, "Permutation is not defined by FE.");
   MFEM_ASSERT(dofs.Size() == ncomp * dof, "Wrong input size.");

   Vector dofs_native(ncomp * dof);
   for (int i = 0; i < dof; i++)
   {
      for (int c = 0; c < ncomp; c++)
      {
         dofs_native(c*dof + lex_ordering[i]) = dofs(c*dof + i);
      }
   }
   dofs = dofs_native;
}

VectorFiniteElement::VectorFiniteElement(int D, Geometry::Type G,
                                         int Do, int O, int M, int F)
   : FiniteElement(D, G, Do, O, F)
{
   range_type = VECTOR;
   map_type = M;
   SetDerivMembers();
   is_nodal = true;
   vdim = dim;
   if (map_type == H_CURL)
   {
      cdim = (dim == 3) ? 3 : 1;
   }
}

void VectorFiniteElement::CalcShape(
   const IntegrationPoint &ip, Vector &shape) const
{
   mfem_error("Error: Cannot use scalar CalcShape(...) function with\n"
              "   VectorFiniteElements!");
}

void VectorFiniteElement::CalcDShape(
   const IntegrationPoint &ip, DenseMatrix &dshape) const
{
   mfem_error("Error: Cannot use scalar CalcDShape(...) function with\n"
              "   VectorFiniteElements!");
}

void VectorFiniteElement::SetDerivMembers()
{
   switch (map_type)
   {
      case H_DIV:
         deriv_type = DIV;
         deriv_range_type = SCALAR;
         deriv_map_type = INTEGRAL;
         break;
      case H_CURL:
         switch (dim)
         {
            case 3: // curl: 3D H_CURL -> 3D H_DIV
               deriv_type = CURL;
               deriv_range_type = VECTOR;
               deriv_map_type = H_DIV;
               break;
            case 2:
               // curl: 2D H_CURL -> INTEGRAL
               deriv_type = CURL;
               deriv_range_type = SCALAR;
               deriv_map_type = INTEGRAL;
               break;
            case 1:
               deriv_type = NONE;
               deriv_range_type = SCALAR;
               deriv_map_type = INTEGRAL;
               break;
            default:
               MFEM_ABORT("Invalid dimension, Dim = " << dim);
         }
         break;
      default:
         MFEM_ABORT("Invalid MapType = " << map_type);
   }
}

void VectorFiniteElement::CalcVShape_RT(
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   MFEM_ASSERT(map_type == H_DIV, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif
   CalcVShape(Trans.GetIntPoint(), vshape);
   MultABt(vshape, Trans.Jacobian(), shape);
   shape *= (1.0 / Trans.Weight());
}

void VectorFiniteElement::CalcVShape_ND(
   ElementTransformation &Trans, DenseMatrix &shape) const
{
   MFEM_ASSERT(map_type == H_CURL, "");
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif
   CalcVShape(Trans.GetIntPoint(), vshape);
   Mult(vshape, Trans.InverseJacobian(), shape);
}

void VectorFiniteElement::Project_RT(
   const real_t *nk, const Array<int> &d2n,
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   real_t vk[Geometry::MaxDim];
   const int sdim = Trans.GetSpaceDim();
   MFEM_ASSERT(vc.GetVDim() == sdim, "");
   Vector xk(vk, sdim);
   const bool square_J = (dim == sdim);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));
      vc.Eval(xk, Trans, Nodes.IntPoint(k));
      // dof_k = nk^t adj(J) xk
      dofs(k) = Trans.AdjugateJacobian().InnerProduct(vk, nk + d2n[k]*dim);
      if (!square_J) { dofs(k) /= Trans.Weight(); }
   }
}

void VectorFiniteElement::Project_RT(
   const real_t *nk, const Array<int> &d2n,
   Vector &vc, ElementTransformation &Trans, Vector &dofs) const
{
   const int sdim = Trans.GetSpaceDim();
   const bool square_J = (dim == sdim);

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));
      // dof_k = nk^t adj(J) xk
      dofs(k) = Trans.AdjugateJacobian().InnerProduct(
                   &vc[k*sdim], nk + d2n[k]*dim);
      if (!square_J) { dofs(k) /= Trans.Weight(); }
   }
}

void VectorFiniteElement::ProjectMatrixCoefficient_RT(
   const real_t *nk, const Array<int> &d2n,
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   // project the rows of the matrix coefficient in an RT space

   const int sdim = T.GetSpaceDim();
   MFEM_ASSERT(mc.GetWidth() == sdim, "");
   const bool square_J = (dim == sdim);
   DenseMatrix MQ(mc.GetHeight(), mc.GetWidth());
   Vector nk_phys(sdim), dofs_k(MQ.Height());
   MFEM_ASSERT(dofs.Size() == dof*MQ.Height(), "");

   for (int k = 0; k < dof; k++)
   {
      T.SetIntPoint(&Nodes.IntPoint(k));
      mc.Eval(MQ, T, Nodes.IntPoint(k));
      // nk_phys = adj(J)^t nk
      T.AdjugateJacobian().MultTranspose(nk + d2n[k]*dim, nk_phys);
      if (!square_J) { nk_phys /= T.Weight(); }
      MQ.Mult(nk_phys, dofs_k);
      for (int r = 0; r < MQ.Height(); r++)
      {
         dofs(k+dof*r) = dofs_k(r);
      }
   }
}

void VectorFiniteElement::Project_RT(
   const real_t *nk, const Array<int> &d2n, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      real_t vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());
      int sdim = Trans.GetSpaceDim();

      I.SetSize(dof, sdim*fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // Transform RT face normals from reference to physical space
         // vk = adj(J)^T nk
         Trans.AdjugateJacobian().MultTranspose(nk + d2n[k]*dim, vk);
         if (fe.GetMapType() == INTEGRAL)
         {
            real_t w = 1.0/Trans.Weight();
            for (int d = 0; d < dim; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            real_t s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            // Project scalar basis function multiplied by each coordinate
            // direction onto the transformed face normals
            for (int d = 0; d < sdim; d++)
            {
               I(k,j+d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      int sdim = Trans.GetSpaceDim();
      real_t vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), sdim);
      Vector vshapenk(fe.GetDof());
      const bool square_J = (dim == sdim);

      I.SetSize(dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Trans.SetIntPoint(&ip);
         // Transform RT face normals from reference to physical space
         // vk = adj(J)^T nk
         Trans.AdjugateJacobian().MultTranspose(nk + d2n[k]*dim, vk);
         // Compute fe basis functions in physical space
         fe.CalcVShape(Trans, vshape);
         // Project fe basis functions onto transformed face normals
         vshape.Mult(vk, vshapenk);
         if (!square_J) { vshapenk /= Trans.Weight(); }
         for (int j=0; j<vshapenk.Size(); j++)
         {
            I(k,j) = vshapenk(j);
         }
      }
   }
}

void VectorFiniteElement::ProjectGrad_RT(
   const real_t *nk, const Array<int> &d2n, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &grad) const
{
   if (dim != 2)
   {
      mfem_error("VectorFiniteElement::ProjectGrad_RT works only in 2D!");
   }

   DenseMatrix dshape(fe.GetDof(), fe.GetDim());
   Vector grad_k(fe.GetDof());
   real_t tk[2];

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      tk[0] = nk[d2n[k]*dim+1];
      tk[1] = -nk[d2n[k]*dim];
      dshape.Mult(tk, grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}

void VectorFiniteElement::ProjectCurl_ND(
   const real_t *tk, const Array<int> &d2t, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &curl) const
{
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(fe.GetDof(), dim);
   DenseMatrix curlshape_J(fe.GetDof(), dim);
   DenseMatrix JtJ(dim, dim);
#else
   curlshape.SetSize(fe.GetDof(), dim);
   curlshape_J.SetSize(fe.GetDof(), dim);
   JtJ.SetSize(dim, dim);
#endif

   Vector curl_k(fe.GetDof());

   curl.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(k);

      // calculate J^t * J / |J|
      Trans.SetIntPoint(&ip);
      MultAtB(Trans.Jacobian(), Trans.Jacobian(), JtJ);
      JtJ *= 1.0 / Trans.Weight();

      // transform curl of shapes (rows) by J^t * J / |J|
      fe.CalcCurlShape(ip, curlshape);
      Mult(curlshape, JtJ, curlshape_J);

      curlshape_J.Mult(tk + d2t[k]*dim, curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

void VectorFiniteElement::ProjectCurl_RT(
   const real_t *nk, const Array<int> &d2n, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &curl) const
{
   DenseMatrix curl_shape(fe.GetDof(), dim);
   Vector curl_k(fe.GetDof());

   curl.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcCurlShape(Nodes.IntPoint(k), curl_shape);
      curl_shape.Mult(nk + d2n[k]*dim, curl_k);
      for (int j = 0; j < curl_k.Size(); j++)
      {
         curl(k,j) = (fabs(curl_k(j)) < 1e-12) ? 0.0 : curl_k(j);
      }
   }
}

void VectorFiniteElement::Project_ND(
   const real_t *tk, const Array<int> &d2t,
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   real_t vk[Geometry::MaxDim];
   Vector xk(vk, vc.GetVDim());

   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));

      vc.Eval(xk, Trans, Nodes.IntPoint(k));
      // dof_k = xk^t J tk
      dofs(k) = Trans.Jacobian().InnerProduct(tk + d2t[k]*dim, vk);
   }
}

void VectorFiniteElement::Project_ND(
   const real_t *tk, const Array<int> &d2t,
   Vector &vc, ElementTransformation &Trans, Vector &dofs) const
{
   for (int k = 0; k < dof; k++)
   {
      Trans.SetIntPoint(&Nodes.IntPoint(k));
      // dof_k = xk^t J tk
      dofs(k) = Trans.Jacobian().InnerProduct(tk + d2t[k]*dim, &vc[k*dim]);
   }
}

void VectorFiniteElement::ProjectMatrixCoefficient_ND(
   const real_t *tk, const Array<int> &d2t,
   MatrixCoefficient &mc, ElementTransformation &T, Vector &dofs) const
{
   // project the rows of the matrix coefficient in an ND space

   const int sdim = T.GetSpaceDim();
   MFEM_ASSERT(mc.GetWidth() == sdim, "");
   DenseMatrix MQ(mc.GetHeight(), mc.GetWidth());
   Vector tk_phys(sdim), dofs_k(MQ.Height());
   MFEM_ASSERT(dofs.Size() == dof*MQ.Height(), "");

   for (int k = 0; k < dof; k++)
   {
      T.SetIntPoint(&Nodes.IntPoint(k));
      mc.Eval(MQ, T, Nodes.IntPoint(k));
      // tk_phys = J tk
      T.Jacobian().Mult(tk + d2t[k]*dim, tk_phys);
      MQ.Mult(tk_phys, dofs_k);
      for (int r = 0; r < MQ.Height(); r++)
      {
         dofs(k+dof*r) = dofs_k(r);
      }
   }
}

void VectorFiniteElement::Project_ND(
   const real_t *tk, const Array<int> &d2t, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   if (fe.GetRangeType() == SCALAR)
   {
      int sdim = Trans.GetSpaceDim();
      real_t vk[Geometry::MaxDim];
      Vector shape(fe.GetDof());

      I.SetSize(dof, sdim*fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         fe.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // Transform ND edge tengents from reference to physical space
         // vk = J tk
         Trans.Jacobian().Mult(tk + d2t[k]*dim, vk);
         if (fe.GetMapType() == INTEGRAL)
         {
            real_t w = 1.0/Trans.Weight();
            for (int d = 0; d < sdim; d++)
            {
               vk[d] *= w;
            }
         }

         for (int j = 0; j < shape.Size(); j++)
         {
            real_t s = shape(j);
            if (fabs(s) < 1e-12)
            {
               s = 0.0;
            }
            // Project scalar basis function multiplied by each coordinate
            // direction onto the transformed edge tangents
            for (int d = 0; d < sdim; d++)
            {
               I(k, j + d*shape.Size()) = s*vk[d];
            }
         }
      }
   }
   else
   {
      int sdim = Trans.GetSpaceDim();
      real_t vk[Geometry::MaxDim];
      DenseMatrix vshape(fe.GetDof(), sdim);
      Vector vshapetk(fe.GetDof());

      I.SetSize(dof, fe.GetDof());
      for (int k = 0; k < dof; k++)
      {
         const IntegrationPoint &ip = Nodes.IntPoint(k);

         Trans.SetIntPoint(&ip);
         // Transform ND edge tangents from reference to physical space
         // vk = J tk
         Trans.Jacobian().Mult(tk + d2t[k]*dim, vk);
         // Compute fe basis functions in physical space
         fe.CalcVShape(Trans, vshape);
         // Project fe basis functions onto transformed edge tangents
         vshape.Mult(vk, vshapetk);
         for (int j=0; j<vshapetk.Size(); j++)
         {
            I(k, j) = vshapetk(j);
         }
      }
   }
}

void VectorFiniteElement::ProjectGrad_ND(
   const real_t *tk, const Array<int> &d2t, const FiniteElement &fe,
   ElementTransformation &Trans, DenseMatrix &grad) const
{
   MFEM_ASSERT(fe.GetMapType() == VALUE, "");

   DenseMatrix dshape(fe.GetDof(), fe.GetDim());
   Vector grad_k(fe.GetDof());

   grad.SetSize(dof, fe.GetDof());
   for (int k = 0; k < dof; k++)
   {
      fe.CalcDShape(Nodes.IntPoint(k), dshape);
      dshape.Mult(tk + d2t[k]*dim, grad_k);
      for (int j = 0; j < grad_k.Size(); j++)
      {
         grad(k,j) = (fabs(grad_k(j)) < 1e-12) ? 0.0 : grad_k(j);
      }
   }
}

void VectorFiniteElement::LocalL2Projection_RT(
   const VectorFiniteElement &cfe, ElementTransformation &Trans,
   DenseMatrix &I) const
{
   Vector v(dim);
   IntegrationPoint tr_ip;

   const int fs = dof, cs = cfe.GetDof();
   I.SetSize(fs, cs);
   DenseMatrix fine_shape(fs, dim), coarse_shape(cs, cfe.GetDim());
   DenseMatrix fine_mass(fs), fine_coarse_mass(fs, cs); // initialized with 0
   const int ir_order =
      std::max(GetOrder(), this->GetOrder()) + this->GetOrder();
   const IntegrationRule &ir = IntRules.Get(this->GetGeomType(), ir_order);

   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &adjJ = Trans.AdjugateJacobian();
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      real_t w = ip.weight;
      this->CalcVShape(ip, fine_shape);
      Trans.Transform(ip, v);
      tr_ip.Set(v.GetData(), dim);
      cfe.CalcVShape(tr_ip, coarse_shape);

      AddMult_a_AAt(w, fine_shape, fine_mass);
      for (int k=0; k<fs; ++k)
      {
         for (int j=0; j<cs; ++j)
         {
            real_t Mkj = 0.0;
            for (int d1=0; d1<dim; ++d1)
            {
               for (int d2=0; d2<dim; ++d2)
               {
                  Mkj += w*fine_shape(k,d1)*adjJ(d2,d1)*coarse_shape(j,d2);
               }
            }
            fine_coarse_mass(k,j) += (fabs(Mkj) < 1e-12) ? 0.0 : Mkj;
         }
      }
   }
   DenseMatrixInverse fine_mass_inv(fine_mass);
   fine_mass_inv.Mult(fine_coarse_mass, I);
}

void VectorFiniteElement::LocalInterpolation_RT(
   const VectorFiniteElement &cfe, const real_t *nk, const Array<int> &d2n,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   MFEM_ASSERT(map_type == cfe.GetMapType(), "");

   if (!is_nodal) { return LocalL2Projection_RT(cfe, Trans, I); }

   real_t vk[Geometry::MaxDim];
   Vector xk(vk, dim);
   IntegrationPoint ip;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(cfe.GetDof(), cfe.GetDim());
#else
   DenseMatrix vshape(cfe.vshape.Data(), cfe.GetDof(), cfe.GetDim());
#endif
   I.SetSize(dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &adjJ = Trans.AdjugateJacobian();
   for (int k = 0; k < dof; k++)
   {
      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = |J| J^{-t} n_k
      adjJ.MultTranspose(nk + d2n[k]*dim, vk);
      // I_k = vshape_k.adj(J)^t.n_k, k=1,...,dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         real_t Ikj = 0.;
         for (int i = 0; i < dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void VectorFiniteElement::LocalL2Projection_ND(
   const VectorFiniteElement &cfe,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   Vector v(dim);
   IntegrationPoint tr_ip;

   const int fs = dof, cs = cfe.GetDof();
   I.SetSize(fs, cs);
   DenseMatrix fine_shape(fs, dim), coarse_shape(cs, cfe.GetDim());
   DenseMatrix fine_mass(fs), fine_coarse_mass(fs, cs); // initialized with 0
   const int ir_order =
      std::max(GetOrder(), this->GetOrder()) + this->GetOrder();
   const IntegrationRule &ir = IntRules.Get(this->GetGeomType(), ir_order);

   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &J = Trans.Jacobian();
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      this->CalcVShape(ip, fine_shape);
      Trans.Transform(ip, v);
      tr_ip.Set(v.GetData(), dim);
      cfe.CalcVShape(tr_ip, coarse_shape);

      AddMult_a_AAt(ip.weight, fine_shape, fine_mass);
      for (int k=0; k<fs; ++k)
      {
         for (int j=0; j<cs; ++j)
         {
            real_t Mkj = 0.0;
            for (int d1=0; d1<dim; ++d1)
            {
               for (int d2=0; d2<dim; ++d2)
               {
                  Mkj += ip.weight*fine_shape(k,d1)*J(d1,d2)*coarse_shape(j,d2);
               }
            }
            fine_coarse_mass(k,j) += (fabs(Mkj) < 1e-12) ? 0.0 : Mkj;
         }
      }
   }
   DenseMatrixInverse fine_mass_inv(fine_mass);
   fine_mass_inv.Mult(fine_coarse_mass, I);
}

void VectorFiniteElement::LocalInterpolation_ND(
   const VectorFiniteElement &cfe, const real_t *tk, const Array<int> &d2t,
   ElementTransformation &Trans, DenseMatrix &I) const
{
   if (!is_nodal) { return LocalL2Projection_ND(cfe, Trans, I); }

   real_t vk[Geometry::MaxDim];
   Vector xk(vk, dim);
   IntegrationPoint ip;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(cfe.GetDof(), cfe.GetDim());
#else
   DenseMatrix vshape(cfe.vshape.Data(), cfe.GetDof(), cfe.GetDim());
#endif
   I.SetSize(dof, vshape.Height());

   // assuming Trans is linear; this should be ok for all refinement types
   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &J = Trans.Jacobian();
   for (int k = 0; k < dof; k++)
   {
      Trans.Transform(Nodes.IntPoint(k), xk);
      ip.Set3(vk);
      cfe.CalcVShape(ip, vshape);
      // xk = J t_k
      J.Mult(tk + d2t[k]*dim, vk);
      // I_k = vshape_k.J.t_k, k=1,...,Dof
      for (int j = 0; j < vshape.Height(); j++)
      {
         real_t Ikj = 0.;
         for (int i = 0; i < dim; i++)
         {
            Ikj += vshape(j, i) * vk[i];
         }
         I(k, j) = (fabs(Ikj) < 1e-12) ? 0.0 : Ikj;
      }
   }
}

void VectorFiniteElement::LocalRestriction_RT(
   const real_t *nk, const Array<int> &d2n, ElementTransformation &Trans,
   DenseMatrix &R) const
{
   real_t pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &J = Trans.Jacobian();
   const real_t weight = Trans.Weight();
   for (int j = 0; j < dof; j++)
   {
      InvertLinearTrans(Trans, Nodes.IntPoint(j), pt);
      ip.Set(pt_data, dim);
      if (Geometries.CheckPoint(geom_type, ip)) // do we need an epsilon here?
      {
         CalcVShape(ip, vshape);
         J.MultTranspose(nk+dim*d2n[j], pt_data);
         pt /= weight;
         for (int k = 0; k < dof; k++)
         {
            real_t R_jk = 0.0;
            for (int d = 0; d < dim; d++)
            {
               R_jk += vshape(k,d)*pt_data[d];
            }
            R(j,k) = R_jk;
         }
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}

void VectorFiniteElement::LocalRestriction_ND(
   const real_t *tk, const Array<int> &d2t, ElementTransformation &Trans,
   DenseMatrix &R) const
{
   real_t pt_data[Geometry::MaxDim];
   IntegrationPoint ip;
   Vector pt(pt_data, dim);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(dof, dim);
#endif

   Trans.SetIntPoint(&Geometries.GetCenter(geom_type));
   const DenseMatrix &Jinv = Trans.InverseJacobian();
   for (int j = 0; j < dof; j++)
   {
      InvertLinearTrans(Trans, Nodes.IntPoint(j), pt);
      ip.Set(pt_data, dim);
      if (Geometries.CheckPoint(geom_type, ip)) // do we need an epsilon here?
      {
         CalcVShape(ip, vshape);
         Jinv.Mult(tk+dim*d2t[j], pt_data);
         for (int k = 0; k < dof; k++)
         {
            real_t R_jk = 0.0;
            for (int d = 0; d < dim; d++)
            {
               R_jk += vshape(k,d)*pt_data[d];
            }
            R(j,k) = R_jk;
         }
      }
      else
      {
         // Set the whole row to avoid valgrind warnings in R.Threshold().
         R.SetRow(j, infinity());
      }
   }
   R.Threshold(1e-12);
}


Poly_1D::Basis::Basis(const int p, const real_t *nodes, EvalType etype)
   : etype(etype), auxiliary_basis(NULL), scale_integrated(false)
{
   switch (etype)
   {
      case ChangeOfBasis:
      {
         x.SetSize(p + 1);
         w.SetSize(p + 1);
         DenseMatrix A(p + 1);
         for (int i = 0; i <= p; i++)
         {
            CalcBasis(p, nodes[i], A.GetColumn(i));
         }
         Ai.Factor(A);
         // mfem::out << "Poly_1D::Basis(" << p << ",...) : "; Ai.TestInversion();
         break;
      }
      case Barycentric:
      {
         x.SetSize(p + 1);
         w.SetSize(p + 1);
         x = nodes;
         w = 1.0;
         for (int i = 0; i <= p; i++)
         {
            for (int j = 0; j < i; j++)
            {
               real_t xij = x(i) - x(j);
               w(i) *=  xij;
               w(j) *= -xij;
            }
         }
         for (int i = 0; i <= p; i++)
         {
            w(i) = 1.0/w(i);
         }

#ifdef MFEM_DEBUG
         // Make sure the nodes are increasing
         for (int i = 0; i < p; i++)
         {
            if (x(i) >= x(i+1))
            {
               mfem_error("Poly_1D::Basis::Basis : nodes are not increasing!");
            }
         }
#endif
         break;
      }
      case Positive:
         x.SetDataAndSize(NULL, p + 1); // use x to store (p + 1)
         break;
      case Integrated:
         auxiliary_basis = new Basis(
            p+1, poly1d.GetPoints(p+1, BasisType::GaussLobatto), Barycentric);
         u_aux.SetSize(p+2);
         d_aux.SetSize(p+2);
         d2_aux.SetSize(p+2);
         break;
      default: break;
   }
}

void Poly_1D::Basis::Eval(const real_t y, Vector &u) const
{
   switch (etype)
   {
      case ChangeOfBasis:
      {
         CalcBasis(Ai.Width() - 1, y, x);
         Ai.Mult(x, u);
         break;
      }
      case Barycentric:
      {
         int i, k, p = x.Size() - 1;
         real_t l, lk;

         if (p == 0)
         {
            u(0) = 1.0;
            return;
         }

         lk = 1.0;
         for (k = 0; k < p; k++)
         {
            if (y >= (x(k) + x(k+1))/2)
            {
               lk *= y - x(k);
            }
            else
            {
               for (i = k+1; i <= p; i++)
               {
                  lk *= y - x(i);
               }
               break;
            }
         }
         l = lk * (y - x(k));

         for (i = 0; i < k; i++)
         {
            u(i) = l * w(i) / (y - x(i));
         }
         u(k) = lk * w(k);
         for (i++; i <= p; i++)
         {
            u(i) = l * w(i) / (y - x(i));
         }
         break;
      }
      case Positive:
         CalcBernstein(x.Size() - 1, y, u);
         break;
      case Integrated:
         auxiliary_basis->Eval(y, u_aux, d_aux);
         EvalIntegrated(d_aux, u);
         break;
      default: break;
   }
}

void Poly_1D::Basis::Eval(const real_t y, Vector &u, Vector &d) const
{
   switch (etype)
   {
      case ChangeOfBasis:
      {
         CalcBasis(Ai.Width() - 1, y, x, w);
         Ai.Mult(x, u);
         Ai.Mult(w, d);
         break;
      }
      case Barycentric:
      {
         int i, k, p = x.Size() - 1;
         real_t l, lp, lk, sk, si;

         if (p == 0)
         {
            u(0) = 1.0;
            d(0) = 0.0;
            return;
         }

         lk = 1.0;
         for (k = 0; k < p; k++)
         {
            if (y >= (x(k) + x(k+1))/2)
            {
               lk *= y - x(k);
            }
            else
            {
               for (i = k+1; i <= p; i++)
               {
                  lk *= y - x(i);
               }
               break;
            }
         }
         l = lk * (y - x(k));

         sk = 0.0;
         for (i = 0; i < k; i++)
         {
            si = 1.0/(y - x(i));
            sk += si;
            u(i) = l * si * w(i);
         }
         u(k) = lk * w(k);
         for (i++; i <= p; i++)
         {
            si = 1.0/(y - x(i));
            sk += si;
            u(i) = l * si * w(i);
         }
         lp = l * sk + lk;

         for (i = 0; i < k; i++)
         {
            d(i) = (lp * w(i) - u(i))/(y - x(i));
         }
         d(k) = sk * u(k);
         for (i++; i <= p; i++)
         {
            d(i) = (lp * w(i) - u(i))/(y - x(i));
         }
         break;
      }
      case Positive:
         CalcBernstein(x.Size() - 1, y, u, d);
         break;
      case Integrated:
         auxiliary_basis->Eval(y, u_aux, d_aux, d2_aux);
         EvalIntegrated(d_aux,u);
         EvalIntegrated(d2_aux,d);
         break;
      default: break;
   }
}

void Poly_1D::Basis::Eval(const real_t y, Vector &u, Vector &d,
                          Vector &d2) const
{
   MFEM_VERIFY(etype == Barycentric,
               "Basis::Eval with second order derivatives not implemented for"
               " etype = " << etype);
   switch (etype)
   {
      case ChangeOfBasis:
      {
         CalcBasis(Ai.Width() - 1, y, x, w);
         Ai.Mult(x, u);
         Ai.Mult(w, d);
         // set d2 (not implemented yet)
         break;
      }
      case Barycentric:
      {
         int i, k, p = x.Size() - 1;
         real_t l, lp, lp2, lk, sk, si, sk2;

         if (p == 0)
         {
            u(0) = 1.0;
            d(0) = 0.0;
            d2(0) = 0.0;
            return;
         }

         lk = 1.0;
         for (k = 0; k < p; k++)
         {
            if (y >= (x(k) + x(k+1))/2)
            {
               lk *= y - x(k);
            }
            else
            {
               for (i = k+1; i <= p; i++)
               {
                  lk *= y - x(i);
               }
               break;
            }
         }
         l = lk * (y - x(k));

         sk = 0.0;
         sk2 = 0.0;
         for (i = 0; i < k; i++)
         {
            si = 1.0/(y - x(i));
            sk += si;
            sk2 -= si * si;
            u(i) = l * si * w(i);
         }
         u(k) = lk * w(k);
         for (i++; i <= p; i++)
         {
            si = 1.0/(y - x(i));
            sk += si;
            sk2 -= si * si;
            u(i) = l * si * w(i);
         }
         lp = l * sk + lk;
         lp2 = lp * sk + l * sk2 + sk * lk;

         for (i = 0; i < k; i++)
         {
            d(i) = (lp * w(i) - u(i))/(y - x(i));
            d2(i) = (lp2 * w(i) - 2 * d(i))/(y - x(i));
         }
         d(k) = sk * u(k);
         d2(k) = sk2 * u(k) + sk * d(k);
         for (i++; i <= p; i++)
         {
            d(i) = (lp * w(i) - u(i))/(y - x(i));
            d2(i) = (lp2 * w(i) - 2 * d(i))/(y - x(i));
         }
         break;
      }
      case Positive:
         CalcBernstein(x.Size() - 1, y, u, d);
         break;
      case Integrated:
         MFEM_ABORT("Integrated basis must be evaluated with EvalIntegrated");
         break;
      default: break;
   }
}

void Poly_1D::Basis::EvalIntegrated(const Vector &d_aux_, Vector &u) const
{
   MFEM_VERIFY(etype == Integrated,
               "EvalIntegrated is only valid for Integrated basis type");
   int p = d_aux_.Size() - 1;
   // See Gerritsma, M. (2010).  "Edge functions for spectral element methods",
   // in Lecture Notes in Computational Science and Engineering, 199--207.
   u[0] = -d_aux_[0];
   for (int j=1; j<p; ++j)
   {
      u[j] = u[j-1] - d_aux_[j];
   }
   // If scale_integrated is true, the degrees of freedom represent mean values,
   // otherwise they represent subcell integrals. Generally, scale_integrated
   // should be true for MapType::VALUE, and false for other map types.
   if (scale_integrated)
   {
      Vector &aux_nodes = auxiliary_basis->x;
      for (int j=0; j<aux_nodes.Size()-1; ++j)
      {
         u[j] *= aux_nodes[j+1] - aux_nodes[j];
      }
   }
}

void Poly_1D::Basis::ScaleIntegrated(bool scale_integrated_)
{
   scale_integrated = scale_integrated_;
}

Poly_1D::Basis::~Basis()
{
   delete auxiliary_basis;
}

const int *Poly_1D::Binom(const int p)
{
   if (binom.NumCols() <= p)
   {
      binom.SetSize(p + 1, p + 1);
      for (int i = 0; i <= p; i++)
      {
         binom(i,0) = binom(i,i) = 1;
         for (int j = 1; j < i; j++)
         {
            binom(i,j) = binom(i-1,j) + binom(i-1,j-1);
         }
      }
   }
   return binom[p];
}

void Poly_1D::ChebyshevPoints(const int p, real_t *x)
{
   for (int i = 0; i <= p; i++)
   {
      // x[i] = 0.5*(1. + cos(M_PI*(p - i + 0.5)/(p + 1)));
      real_t s = sin(M_PI_2*(i + 0.5)/(p + 1));
      x[i] = s*s;
   }
}

void Poly_1D::CalcMono(const int p, const real_t x, real_t *u)
{
   real_t xn;
   u[0] = xn = 1.;
   for (int n = 1; n <= p; n++)
   {
      u[n] = (xn *= x);
   }
}

void Poly_1D::CalcMono(const int p, const real_t x, real_t *u, real_t *d)
{
   real_t xn;
   u[0] = xn = 1.;
   d[0] = 0.;
   for (int n = 1; n <= p; n++)
   {
      d[n] = n * xn;
      u[n] = (xn *= x);
   }
}

void Poly_1D::CalcBinomTerms(const int p, const real_t x, const real_t y,
                             real_t *u)
{
   if (p == 0)
   {
      u[0] = 1.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      real_t z = x;

      for (i = 1; i < p; i++)
      {
         u[i] = b[i]*z;
         z *= x;
      }
      u[p] = z;
      z = y;
      for (i--; i > 0; i--)
      {
         u[i] *= z;
         z *= y;
      }
      u[0] = z;
   }
}

void Poly_1D::CalcBinomTerms(const int p, const real_t x, const real_t y,
                             real_t *u, real_t *d)
{
   if (p == 0)
   {
      u[0] = 1.;
      d[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      const real_t xpy = x + y, ptx = p*x;
      real_t z = 1.;

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
         u[i] = b[i]*z;
      }
      d[p] = p*z;
      u[p] = z*x;
      z = 1.;
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
         u[i] *= z;
      }
      d[0] = -p*z;
      u[0] = z*y;
   }
}

void Poly_1D::CalcDBinomTerms(const int p, const real_t x, const real_t y,
                              real_t *d)
{
   if (p == 0)
   {
      d[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      const real_t xpy = x + y, ptx = p*x;
      real_t z = 1.;

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
      }
      d[p] = p*z;
      z = 1.;
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
      }
      d[0] = -p*z;
   }
}

void Poly_1D::CalcDxBinomTerms(const int p, const real_t x, const real_t y,
                               real_t *u)
{
   if (p == 0)
   {
      u[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      real_t z = 1.;

      for (i = 1; i < p; i++)
      {
         u[i] = i * b[i]*z;
         z *= x;
      }
      u[p] = i * z;
      z = y;
      for (i--; i > 0; i--)
      {
         u[i] *= z;
         z *= y;
      }
      u[0] = 0;
   }
}

void Poly_1D::CalcDyBinomTerms(const int p, const real_t x, const real_t y,
                               real_t *u)
{
   if (p == 0)
   {
      u[0] = 0.;
   }
   else
   {
      int i;
      const int *b = Binom(p);
      real_t z = x;

      for (i = 1; i < p; i++)
      {
         u[i] = b[i]*z;
         z *= x;
      }
      u[p] = 0.;
      z = 1.;
      for (i--; i > 0; i--)
      {
         u[i] *= (p - i) * z;
         z *= y;
      }
      u[0] = p * z;
   }
}

void Poly_1D::CalcLegendre(const int p, const real_t x, real_t *u)
{
   // use the recursive definition for [-1,1]:
   // (n+1)*P_{n+1}(z) = (2*n+1)*z*P_n(z)-n*P_{n-1}(z)
   real_t z;
   u[0] = 1.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = ((2*n + 1)*z*u[n] - n*u[n-1])/(n + 1);
   }
}

void Poly_1D::CalcLegendre(const int p, const real_t x, real_t *u, real_t *d)
{
   // use the recursive definition for [-1,1]:
   // (n+1)*P_{n+1}(z) = (2*n+1)*z*P_n(z)-n*P_{n-1}(z)
   // for the derivative use, z in [-1,1]:
   // P'_{n+1}(z) = (2*n+1)*P_n(z)+P'_{n-1}(z)
   real_t z;
   u[0] = 1.;
   d[0] = 0.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   d[1] = 2.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = ((2*n + 1)*z*u[n] - n*u[n-1])/(n + 1);
      d[n+1] = (4*n + 2)*u[n] + d[n-1];
   }
}

void Poly_1D::CalcChebyshev(const int p, const real_t x, real_t *u)
{
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   real_t z;
   u[0] = 1.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = 2*z*u[n] - u[n-1];
   }
}

void Poly_1D::CalcChebyshev(const int p, const real_t x, real_t *u, real_t *d)
{
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   // T'_n(z) = n*U_{n-1}(z)
   // U_0(z) = 1  U_1(z) = 2*z
   // U_{n+1}(z) = 2*z*U_n(z) - U_{n-1}(z)
   // U_n(z) = z*U_{n-1}(z) + T_n(z) = z*T'_n(z)/n + T_n(z)
   // T'_{n+1}(z) = (n + 1)*(z*T'_n(z)/n + T_n(z))
   real_t z;
   u[0] = 1.;
   d[0] = 0.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   d[1] = 2.;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = 2*z*u[n] - u[n-1];
      d[n+1] = (n + 1)*(z*d[n]/n + 2*u[n]);
   }
}

void Poly_1D::CalcChebyshev(const int p, const real_t x, real_t *u, real_t *d,
                            real_t *dd)
{
   // recursive definition, z in [-1,1]
   // T_0(z) = 1,  T_1(z) = z
   // T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
   // T'_n(z) = n*U_{n-1}(z)
   // U_0(z) = 1  U_1(z) = 2*z
   // U_{n+1}(z) = 2*z*U_n(z) - U_{n-1}(z)
   // U_n(z) = z*U_{n-1}(z) + T_n(z) = z*T'_n(z)/n + T_n(z)
   // T'_{n+1}(z) = (n + 1)*(z*T'_n(z)/n + T_n(z))
   // T''_{n+1}(z) = (n + 1)*(2*(n + 1)*T'_n(z) + z*T''_n(z)) / n
   real_t z;
   u[0] = 1.;
   d[0] = 0.;
   dd[0]= 0.;
   if (p == 0) { return; }
   u[1] = z = 2.*x - 1.;
   d[1] = 2.;
   dd[1] = 0;
   for (int n = 1; n < p; n++)
   {
      u[n+1] = 2*z*u[n] - u[n-1];
      d[n+1] = (n + 1)*(z*d[n]/n + 2*u[n]);
      dd[n+1] = (n + 1)*(2.*(n + 1)*d[n] + z*dd[n])/n;
   }
}

const Array<real_t>* Poly_1D::GetPointsArray(const int p, const int btype)
{
   Array<real_t> *val;
   BasisType::Check(btype);
   const int qtype = BasisType::GetQuadrature1D(btype);
   if (qtype == Quadrature1D::Invalid) { return nullptr; }

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   #pragma omp critical (Poly1DGetPoints)
#endif
   {
      std::pair<int, int> key(btype, p);
      auto it = points_container.find(key);
      if (it == points_container.end())
      {
         it = points_container.emplace(key, new Array<real_t>(p + 1, h_mt)).first;
         val = it->second.get();
         real_t* hptr = val->HostWrite();
         quad_func.GivePolyPoints(p + 1, hptr, qtype);
      }
      else
      {
         val = it->second.get();
      }
   }
   return val;
}

Poly_1D::Basis &Poly_1D::GetBasis(const int p, const int btype)
{
   BasisType::Check(btype);
   Basis* val;

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   #pragma omp critical (Poly1DGetBasis)
#endif
   {
      std::pair<int, int> key(btype, p);
      auto it = bases_container.find(key);
      if (it == bases_container.end())
      {
         EvalType etype;
         if (btype == BasisType::Positive) { etype = Positive; }
         else if (btype == BasisType::IntegratedGLL) { etype = Integrated; }
         else { etype = Barycentric; }
         it = bases_container
              .emplace(key, new Basis(p, GetPoints(p, btype), etype))
              .first;
      }
      val = it->second.get();
   }
   return *val;
}


TensorBasisElement::TensorBasisElement(const int dims, const int p,
                                       const int btype, const DofMapType dmtype)
   : b_type(btype),
     basis1d(poly1d.GetBasis(p, b_type))
{
   if (dmtype == H1_DOF_MAP || dmtype == Sr_DOF_MAP)
   {
      switch (dims)
      {
         case 1:
         {
            dof_map.SetSize(p + 1);
            dof_map[0] = 0;
            dof_map[p] = 1;
            for (int i = 1; i < p; i++)
            {
               dof_map[i] = i+1;
            }
            break;
         }
         case 2:
         {
            const int p1 = p + 1;
            dof_map.SetSize(p1*p1);

            // vertices
            dof_map[0 + 0*p1] = 0;
            dof_map[p + 0*p1] = 1;
            dof_map[p + p*p1] = 2;
            dof_map[0 + p*p1] = 3;

            // edges
            int o = 4;
            for (int i = 1; i < p; i++)
            {
               dof_map[i + 0*p1] = o++;
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + i*p1] = o++;
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[(p-i) + p*p1] = o++;
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (p-i)*p1] = o++;
            }

            // interior
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + j*p1] = o++;
               }
            }
            break;
         }
         case 3:
         {
            const int p1 = p + 1;
            dof_map.SetSize(p1*p1*p1);

            // vertices
            dof_map[0 + (0 + 0*p1)*p1] = 0;
            dof_map[p + (0 + 0*p1)*p1] = 1;
            dof_map[p + (p + 0*p1)*p1] = 2;
            dof_map[0 + (p + 0*p1)*p1] = 3;
            dof_map[0 + (0 + p*p1)*p1] = 4;
            dof_map[p + (0 + p*p1)*p1] = 5;
            dof_map[p + (p + p*p1)*p1] = 6;
            dof_map[0 + (p + p*p1)*p1] = 7;

            // edges (see Hexahedron::edges in mesh/hexahedron.cpp).
            // edges (see Constants<Geometry::CUBE>::Edges in fem/geom.cpp).
            int o = 8;
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (0 + 0*p1)*p1] = o++;   // (0,1)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (i + 0*p1)*p1] = o++;   // (1,2)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (p + 0*p1)*p1] = o++;   // (3,2)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (i + 0*p1)*p1] = o++;   // (0,3)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (0 + p*p1)*p1] = o++;   // (4,5)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (i + p*p1)*p1] = o++;   // (5,6)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[i + (p + p*p1)*p1] = o++;   // (7,6)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (i + p*p1)*p1] = o++;   // (4,7)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (0 + i*p1)*p1] = o++;   // (0,4)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (0 + i*p1)*p1] = o++;   // (1,5)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[p + (p + i*p1)*p1] = o++;   // (2,6)
            }
            for (int i = 1; i < p; i++)
            {
               dof_map[0 + (p + i*p1)*p1] = o++;   // (3,7)
            }

            // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + ((p-j) + 0*p1)*p1] = o++;   // (3,2,1,0)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + (0 + j*p1)*p1] = o++;   // (0,1,5,4)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[p + (i + j*p1)*p1] = o++;   // (1,2,6,5)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[(p-i) + (p + j*p1)*p1] = o++;   // (2,3,7,6)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[0 + ((p-i) + j*p1)*p1] = o++;   // (3,0,4,7)
               }
            }
            for (int j = 1; j < p; j++)
            {
               for (int i = 1; i < p; i++)
               {
                  dof_map[i + (j + p*p1)*p1] = o++;   // (4,5,6,7)
               }
            }

            // interior
            for (int k = 1; k < p; k++)
            {
               for (int j = 1; j < p; j++)
               {
                  for (int i = 1; i < p; i++)
                  {
                     dof_map[i + (j + k*p1)*p1] = o++;
                  }
               }
            }
            break;
         }
         default:
            MFEM_ABORT("invalid dimension: " << dims);
            break;
      }
   }
   else if (dmtype == L2_DOF_MAP)
   {
      // leave dof_map empty, indicating that the dofs are ordered
      // lexicographically, i.e. the dof_map is identity
   }
   else
   {
      MFEM_ABORT("invalid DofMapType: " << dmtype);
   }
}

const DofToQuad &TensorBasisElement::GetTensorDofToQuad(
   const FiniteElement &fe, const IntegrationRule &ir,
   DofToQuad::Mode mode, const Poly_1D::Basis &basis, bool closed,
   Array<DofToQuad*> &dof2quad_array)
{
   DofToQuad *d2q = nullptr;
   MFEM_VERIFY(mode == DofToQuad::TENSOR, "invalid mode requested");

#if defined(MFEM_THREAD_SAFE) && defined(MFEM_USE_OPENMP)
   #pragma omp critical (DofToQuad)
#endif
   {
      for (int i = 0; i < dof2quad_array.Size(); i++)
      {
         d2q = dof2quad_array[i];
         if (d2q->IntRule != &ir || d2q->mode != mode) { d2q = nullptr; }
      }
      if (!d2q)
      {
         d2q = new DofToQuad;
         const int ndof = closed ? fe.GetOrder() + 1 : fe.GetOrder();
         const int nqpt = (int)floor(pow(ir.GetNPoints(), 1.0/fe.GetDim()) + 0.5);
         d2q->FE = &fe;
         d2q->IntRule = &ir;
         d2q->mode = mode;
         d2q->ndof = ndof;
         d2q->nqpt = nqpt;
         d2q->B.SetSize(nqpt*ndof);
         d2q->Bt.SetSize(ndof*nqpt);
         d2q->G.SetSize(nqpt*ndof);
         d2q->Gt.SetSize(ndof*nqpt);
         Vector val(ndof), grad(ndof);
         for (int i = 0; i < nqpt; i++)
         {
            // The first 'nqpt' points in 'ir' have the same x-coordinates as those
            // of the 1D rule.
            basis.Eval(ir.IntPoint(i).x, val, grad);
            for (int j = 0; j < ndof; j++)
            {
               d2q->B[i+nqpt*j] = d2q->Bt[j+ndof*i] = val(j);
               d2q->G[i+nqpt*j] = d2q->Gt[j+ndof*i] = grad(j);
            }
         }
         dof2quad_array.Append(d2q);
      }
   }
   return *d2q;
}

NodalTensorFiniteElement::NodalTensorFiniteElement(const int dims,
                                                   const int p,
                                                   const int btype,
                                                   const DofMapType dmtype)
   : NodalFiniteElement(dims, GetTensorProductGeometry(dims), Pow(p + 1, dims),
                        p, dims > 1 ? FunctionSpace::Qk : FunctionSpace::Pk),
     TensorBasisElement(dims, p, btype, dmtype)
{
   lex_ordering = dof_map;
}

void NodalTensorFiniteElement::SetMapType(const int map_type)
{
   ScalarFiniteElement::SetMapType(map_type);
   // If we are using the "integrated" basis, the basis functions should be
   // scaled for MapType::VALUE, and not scaled for MapType::INTEGRAL. This
   // ensures spectral equivalence of the mass matrix with its low-order-refined
   // counterpart (cf. LORDiscretization)
   if (basis1d.IsIntegratedType())
   {
      basis1d.ScaleIntegrated(map_type == VALUE);
   }
}

const DofToQuad &NodalTensorFiniteElement::GetDofToQuad(
   const IntegrationRule &ir,
   DofToQuad::Mode mode) const
{
   if (mode != DofToQuad::TENSOR)
   {
      return NodalFiniteElement::GetDofToQuad(ir, mode);
   }
   else
   {
      return GetTensorDofToQuad(*this, ir, mode, basis1d, true, dof2quad_array);
   }
}

void NodalTensorFiniteElement::GetFaceMap(const int face_id,
                                          Array<int> &face_map) const
{
   internal::GetTensorFaceMap(dim, order, face_id, face_map);
}

VectorTensorFiniteElement::VectorTensorFiniteElement(const int dims,
                                                     const int d,
                                                     const int p,
                                                     const int cbtype,
                                                     const int obtype,
                                                     const int M,
                                                     const DofMapType dmtype)
   : VectorFiniteElement(dims, GetTensorProductGeometry(dims), d,
                         p, M, FunctionSpace::Qk),
     TensorBasisElement(dims, p, VerifyNodal(VerifyClosed(cbtype)), dmtype),
     obasis1d(poly1d.GetBasis(p - 1, VerifyOpen(obtype)))
{
   MFEM_VERIFY(dims > 1, "Constructor for VectorTensorFiniteElement with both "
               "open and closed bases is not valid for 1D elements.");
}

VectorTensorFiniteElement::VectorTensorFiniteElement(const int dims,
                                                     const int d,
                                                     const int p,
                                                     const int obtype,
                                                     const int M,
                                                     const DofMapType dmtype)
   : VectorFiniteElement(dims, GetTensorProductGeometry(dims), d,
                         p, M, FunctionSpace::Pk),
     TensorBasisElement(dims, p, VerifyOpen(obtype), dmtype),
     obasis1d(poly1d.GetBasis(p, VerifyOpen(obtype)))
{
   MFEM_VERIFY(dims == 1, "Constructor for VectorTensorFiniteElement without "
               "closed basis is only valid for 1D elements.");
}

VectorTensorFiniteElement::~VectorTensorFiniteElement()
{
   for (int i = 0; i < dof2quad_array_open.Size(); i++)
   {
      delete dof2quad_array_open[i];
   }
}

}
