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

// H1 Finite Element classes utilizing the Bernstein basis

#include "fe_pos.hpp"
#include "face_map_utils.hpp"
#include "../bilininteg.hpp"
#include "../lininteg.hpp"
#include "../coefficient.hpp"

namespace mfem
{

using namespace std;

void PositiveFiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   for (int i = 0; i < dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      dofs(i) = coeff.Eval(Trans, ip);
   }
}

void PositiveFiniteElement::Project(
   VectorCoefficient &vc, ElementTransformation &Trans, Vector &dofs) const
{
   MFEM_ASSERT(dofs.Size() == vc.GetVDim()*dof, "");
   Vector x(vc.GetVDim());

   for (int i = 0; i < dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      vc.Eval (x, Trans, ip);
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(dof*j+i) = x(j);
      }
   }
}

void PositiveFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   const NodalFiniteElement *nfe =
      dynamic_cast<const NodalFiniteElement *>(&fe);

   if (nfe && dof == nfe->GetDof())
   {
      nfe->Project(*this, Trans, I);
      I.Invert();
   }
   else
   {
      // local L2 projection
      DenseMatrix pos_mass, mixed_mass;
      MassIntegrator mass_integ;

      mass_integ.AssembleElementMatrix(*this, Trans, pos_mass);
      mass_integ.AssembleElementMatrix2(fe, *this, Trans, mixed_mass);

      DenseMatrixInverse pos_mass_inv(pos_mass);
      I.SetSize(dof, fe.GetDof());
      pos_mass_inv.Mult(mixed_mass, I);
   }
}


PositiveTensorFiniteElement::PositiveTensorFiniteElement(
   const int dims, const int p, const DofMapType dmtype)
   : PositiveFiniteElement(dims, GetTensorProductGeometry(dims),
                           Pow(p + 1, dims), p,
                           dims > 1 ? FunctionSpace::Qk : FunctionSpace::Pk),
     TensorBasisElement(dims, p, BasisType::Positive, dmtype) { }

void PositiveTensorFiniteElement::GetFaceMap(const int face_id,
                                             Array<int> &face_map) const
{
   internal::GetTensorFaceMap(dim, order, face_id, face_map);
}


BiQuadPos2DFiniteElement::BiQuadPos2DFiniteElement()
   : PositiveFiniteElement(2, Geometry::SQUARE, 9, 2, FunctionSpace::Qk)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(0).y = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(1).y = 0.0;
   Nodes.IntPoint(2).x = 1.0;
   Nodes.IntPoint(2).y = 1.0;
   Nodes.IntPoint(3).x = 0.0;
   Nodes.IntPoint(3).y = 1.0;
   Nodes.IntPoint(4).x = 0.5;
   Nodes.IntPoint(4).y = 0.0;
   Nodes.IntPoint(5).x = 1.0;
   Nodes.IntPoint(5).y = 0.5;
   Nodes.IntPoint(6).x = 0.5;
   Nodes.IntPoint(6).y = 1.0;
   Nodes.IntPoint(7).x = 0.0;
   Nodes.IntPoint(7).y = 0.5;
   Nodes.IntPoint(8).x = 0.5;
   Nodes.IntPoint(8).y = 0.5;
}

void BiQuadPos2DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   real_t x = ip.x, y = ip.y;
   real_t l1x, l2x, l3x, l1y, l2y, l3y;

   l1x = (1. - x) * (1. - x);
   l2x = 2. * x * (1. - x);
   l3x = x * x;
   l1y = (1. - y) * (1. - y);
   l2y = 2. * y * (1. - y);
   l3y = y * y;

   shape(0) = l1x * l1y;
   shape(4) = l2x * l1y;
   shape(1) = l3x * l1y;
   shape(7) = l1x * l2y;
   shape(8) = l2x * l2y;
   shape(5) = l3x * l2y;
   shape(3) = l1x * l3y;
   shape(6) = l2x * l3y;
   shape(2) = l3x * l3y;
}

void BiQuadPos2DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   real_t x = ip.x, y = ip.y;
   real_t l1x, l2x, l3x, l1y, l2y, l3y;
   real_t d1x, d2x, d3x, d1y, d2y, d3y;

   l1x = (1. - x) * (1. - x);
   l2x = 2. * x * (1. - x);
   l3x = x * x;
   l1y = (1. - y) * (1. - y);
   l2y = 2. * y * (1. - y);
   l3y = y * y;

   d1x = 2. * x - 2.;
   d2x = 2. - 4. * x;
   d3x = 2. * x;
   d1y = 2. * y - 2.;
   d2y = 2. - 4. * y;
   d3y = 2. * y;

   dshape(0,0) = d1x * l1y;
   dshape(0,1) = l1x * d1y;

   dshape(4,0) = d2x * l1y;
   dshape(4,1) = l2x * d1y;

   dshape(1,0) = d3x * l1y;
   dshape(1,1) = l3x * d1y;

   dshape(7,0) = d1x * l2y;
   dshape(7,1) = l1x * d2y;

   dshape(8,0) = d2x * l2y;
   dshape(8,1) = l2x * d2y;

   dshape(5,0) = d3x * l2y;
   dshape(5,1) = l3x * d2y;

   dshape(3,0) = d1x * l3y;
   dshape(3,1) = l1x * d3y;

   dshape(6,0) = d2x * l3y;
   dshape(6,1) = l2x * d3y;

   dshape(2,0) = d3x * l3y;
   dshape(2,1) = l3x * d3y;
}

void BiQuadPos2DFiniteElement::GetLocalInterpolation(
   ElementTransformation &Trans, DenseMatrix &I) const
{
   real_t s[9];
   IntegrationPoint tr_ip;
   Vector xx(&tr_ip.x, 2), shape(s, 9);

   for (int i = 0; i < 9; i++)
   {
      Trans.Transform(Nodes.IntPoint(i), xx);
      CalcShape(tr_ip, shape);
      for (int j = 0; j < 9; j++)
         if (fabs(I(i,j) = s[j]) < 1.0e-12)
         {
            I(i,j) = 0.0;
         }
   }
   for (int i = 0; i < 9; i++)
   {
      real_t *d = &I(0,i);
      d[4] = 2. * d[4] - 0.5 * (d[0] + d[1]);
      d[5] = 2. * d[5] - 0.5 * (d[1] + d[2]);
      d[6] = 2. * d[6] - 0.5 * (d[2] + d[3]);
      d[7] = 2. * d[7] - 0.5 * (d[3] + d[0]);
      d[8] = 4. * d[8] - 0.5 * (d[4] + d[5] + d[6] + d[7]) -
             0.25 * (d[0] + d[1] + d[2] + d[3]);
   }
}

void BiQuadPos2DFiniteElement::Project(
   Coefficient &coeff, ElementTransformation &Trans, Vector &dofs) const
{
   real_t *d = dofs.GetData();

   for (int i = 0; i < 9; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      d[i] = coeff.Eval(Trans, ip);
   }
   d[4] = 2. * d[4] - 0.5 * (d[0] + d[1]);
   d[5] = 2. * d[5] - 0.5 * (d[1] + d[2]);
   d[6] = 2. * d[6] - 0.5 * (d[2] + d[3]);
   d[7] = 2. * d[7] - 0.5 * (d[3] + d[0]);
   d[8] = 4. * d[8] - 0.5 * (d[4] + d[5] + d[6] + d[7]) -
          0.25 * (d[0] + d[1] + d[2] + d[3]);
}

void BiQuadPos2DFiniteElement::Project (
   VectorCoefficient &vc, ElementTransformation &Trans,
   Vector &dofs) const
{
   real_t v[3];
   Vector x (v, vc.GetVDim());

   for (int i = 0; i < 9; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      Trans.SetIntPoint(&ip);
      vc.Eval (x, Trans, ip);
      for (int j = 0; j < x.Size(); j++)
      {
         dofs(9*j+i) = v[j];
      }
   }
   for (int j = 0; j < x.Size(); j++)
   {
      real_t *d = &dofs(9*j);

      d[4] = 2. * d[4] - 0.5 * (d[0] + d[1]);
      d[5] = 2. * d[5] - 0.5 * (d[1] + d[2]);
      d[6] = 2. * d[6] - 0.5 * (d[2] + d[3]);
      d[7] = 2. * d[7] - 0.5 * (d[3] + d[0]);
      d[8] = 4. * d[8] - 0.5 * (d[4] + d[5] + d[6] + d[7]) -
             0.25 * (d[0] + d[1] + d[2] + d[3]);
   }
}


QuadPos1DFiniteElement::QuadPos1DFiniteElement()
   : PositiveFiniteElement(1, Geometry::SEGMENT, 3, 2)
{
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   Nodes.IntPoint(2).x = 0.5;
}

void QuadPos1DFiniteElement::CalcShape(const IntegrationPoint &ip,
                                       Vector &shape) const
{
   const real_t x = ip.x, x1 = 1. - x;

   shape(0) = x1 * x1;
   shape(1) = x * x;
   shape(2) = 2. * x * x1;
}

void QuadPos1DFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                        DenseMatrix &dshape) const
{
   const real_t x = ip.x;

   dshape(0,0) = 2. * x - 2.;
   dshape(1,0) = 2. * x;
   dshape(2,0) = 2. - 4. * x;
}


H1Pos_SegmentElement::H1Pos_SegmentElement(const int p)
   : PositiveTensorFiniteElement(1, p, H1_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   // thread private versions; see class header.
   shape_x.SetSize(p+1);
   dshape_x.SetSize(p+1);
#endif

   // Endpoints need to be first in the list, so reorder them.
   Nodes.IntPoint(0).x = 0.0;
   Nodes.IntPoint(1).x = 1.0;
   for (int i = 1; i < p; i++)
   {
      Nodes.IntPoint(i+1).x = real_t(i)/p;
   }
}

void H1Pos_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData() );

   // Endpoints need to be first in the list, so reorder them.
   shape(0) = shape_x(0);
   shape(1) = shape_x(p);
   for (int i = 1; i < p; i++)
   {
      shape(i+1) = shape_x(i);
   }
}

void H1Pos_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), dshape_x(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData(), dshape_x.GetData() );

   // Endpoints need to be first in the list, so reorder them.
   dshape(0,0) = dshape_x(0);
   dshape(1,0) = dshape_x(p);
   for (int i = 1; i < p; i++)
   {
      dshape(i+1,0) = dshape_x(i);
   }
}

void H1Pos_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex] = 1.0;
}


H1Pos_QuadrilateralElement::H1Pos_QuadrilateralElement(const int p)
   : PositiveTensorFiniteElement(2, p, H1_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
#endif

   int o = 0;
   for (int j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(dof_map[o++]).Set2(real_t(i)/p, real_t(j)/p);
      }
}

void H1Pos_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y);

   // Reorder so that vertices are at the beginning of the list
   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(dof_map[o++]) = shape_x(i)*shape_y(j);
      }
}

void H1Pos_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x, dshape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y, dshape_y);

   // Reorder so that vertices are at the beginning of the list
   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         dshape(dof_map[o],0) = dshape_x(i)* shape_y(j);
         dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j);  o++;
      }
}

void H1Pos_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex] = 1.0;
}


H1Pos_HexahedronElement::H1Pos_HexahedronElement(const int p)
   : PositiveTensorFiniteElement(3, p, H1_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   const int p1 = p + 1;

   shape_x.SetSize(p1);
   shape_y.SetSize(p1);
   shape_z.SetSize(p1);
   dshape_x.SetSize(p1);
   dshape_y.SetSize(p1);
   dshape_z.SetSize(p1);
#endif

   int o = 0;
   for (int k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
            Nodes.IntPoint(dof_map[o++]).Set3(real_t(i)/p, real_t(j)/p,
                                              real_t(k)/p);
}

void H1Pos_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData() );
   Poly_1D::CalcBernstein(p, ip.y, shape_y.GetData() );
   Poly_1D::CalcBernstein(p, ip.z, shape_z.GetData() );

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(dof_map[o++]) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void H1Pos_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x.GetData(), dshape_x.GetData() );
   Poly_1D::CalcBernstein(p, ip.y, shape_y.GetData(), dshape_y.GetData() );
   Poly_1D::CalcBernstein(p, ip.z, shape_z.GetData(), dshape_z.GetData() );

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(dof_map[o],0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(dof_map[o],1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(dof_map[o],2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void H1Pos_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex] = 1.0;
}


H1Pos_TriangleElement::H1Pos_TriangleElement(const int p)
   : PositiveFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                           FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   m_shape.SetSize(dof);
   dshape_1d.SetSize(p + 1);
   m_dshape.SetSize(dof, dim);
#endif
   dof_map.SetSize(dof);

   struct Index
   {
      int p2p3;
      Index(int p) { p2p3 = 2*p + 3; }
      int operator()(int i, int j) { return ((p2p3-j)*j)/2+i; }
   };
   Index idx(p);

   // vertices
   dof_map[idx(0,0)] = 0;
   Nodes.IntPoint(0).Set2(0., 0.);
   dof_map[idx(p,0)] = 1;
   Nodes.IntPoint(1).Set2(1., 0.);
   dof_map[idx(0,p)] = 2;
   Nodes.IntPoint(2).Set2(0., 1.);

   // edges
   int o = 3;
   for (int i = 1; i < p; i++)
   {
      dof_map[idx(i,0)] = o;
      Nodes.IntPoint(o++).Set2(real_t(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)
   {
      dof_map[idx(p-i,i)] = o;
      Nodes.IntPoint(o++).Set2(real_t(p-i)/p, real_t(i)/p);
   }
   for (int i = 1; i < p; i++)
   {
      dof_map[idx(0,p-i)] = o;
      Nodes.IntPoint(o++).Set2(0., real_t(p-i)/p);
   }

   // interior
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)
      {
         dof_map[idx(i,j)] = o;
         Nodes.IntPoint(o++).Set2(real_t(i)/p, real_t(j)/p);
      }
}

// static method
void H1Pos_TriangleElement::CalcShape(
   const int p, const real_t l1, const real_t l2, real_t *shape)
{
   const real_t l3 = 1. - l1 - l2;

   // The (i,j) basis function is given by: T(i,j,p-i-j) l1^i l2^j l3^{p-i-j},
   // where T(i,j,k) = (i+j+k)! / (i! j! k!)
   // Another expression is given by the terms of the expansion:
   //    (l1 + l2 + l3)^p =
   //       \sum_{j=0}^p \binom{p}{j} l2^j
   //          \sum_{i=0}^{p-j} \binom{p-j}{i} l1^i l3^{p-j-i}
   const int *bp = Poly_1D::Binom(p);
   real_t z = 1.;
   for (int o = 0, j = 0; j <= p; j++)
   {
      Poly_1D::CalcBinomTerms(p - j, l1, l3, &shape[o]);
      real_t s = bp[j]*z;
      for (int i = 0; i <= p - j; i++)
      {
         shape[o++] *= s;
      }
      z *= l2;
   }
}

// static method
void H1Pos_TriangleElement::CalcDShape(
   const int p, const real_t l1, const real_t l2,
   real_t *dshape_1d, real_t *dshape)
{
   const int dof = ((p + 1)*(p + 2))/2;
   const real_t l3 = 1. - l1 - l2;

   const int *bp = Poly_1D::Binom(p);
   real_t z = 1.;
   for (int o = 0, j = 0; j <= p; j++)
   {
      Poly_1D::CalcDBinomTerms(p - j, l1, l3, dshape_1d);
      real_t s = bp[j]*z;
      for (int i = 0; i <= p - j; i++)
      {
         dshape[o++] = s*dshape_1d[i];
      }
      z *= l2;
   }
   z = 1.;
   for (int i = 0; i <= p; i++)
   {
      Poly_1D::CalcDBinomTerms(p - i, l2, l3, dshape_1d);
      real_t s = bp[i]*z;
      for (int o = i, j = 0; j <= p - i; j++)
      {
         dshape[dof + o] = s*dshape_1d[j];
         o += p + 1 - j;
      }
      z *= l1;
   }
}

void H1Pos_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape(dof);
#endif
   CalcShape(order, ip.x, ip.y, m_shape.GetData());
   for (int i = 0; i < dof; i++)
   {
      shape(dof_map[i]) = m_shape(i);
   }
}

void H1Pos_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(order + 1);
   DenseMatrix m_dshape(dof, dim);
#endif
   CalcDShape(order, ip.x, ip.y, dshape_1d.GetData(), m_dshape.Data());
   for (int d = 0; d < 2; d++)
   {
      for (int i = 0; i < dof; i++)
      {
         dshape(dof_map[i],d) = m_dshape(i,d);
      }
   }
}


H1Pos_TetrahedronElement::H1Pos_TetrahedronElement(const int p)
   : PositiveFiniteElement(3, Geometry::TETRAHEDRON,
                           ((p + 1)*(p + 2)*(p + 3))/6, p, FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   m_shape.SetSize(dof);
   dshape_1d.SetSize(p + 1);
   m_dshape.SetSize(dof, dim);
#endif
   dof_map.SetSize(dof);

   struct Index
   {
      int p, dof;
      int tri(int k) { return (k*(k + 1))/2; }
      int tet(int k) { return (k*(k + 1)*(k + 2))/6; }
      Index(int p_) { p = p_; dof = tet(p + 1); }
      int operator()(int i, int j, int k)
      { return dof - tet(p - k) - tri(p + 1 - k - j) + i; }
   };
   Index idx(p);

   // vertices
   dof_map[idx(0,0,0)] = 0;
   Nodes.IntPoint(0).Set3(0., 0., 0.);
   dof_map[idx(p,0,0)] = 1;
   Nodes.IntPoint(1).Set3(1., 0., 0.);
   dof_map[idx(0,p,0)] = 2;
   Nodes.IntPoint(2).Set3(0., 1., 0.);
   dof_map[idx(0,0,p)] = 3;
   Nodes.IntPoint(3).Set3(0., 0., 1.);

   // edges (see Tetrahedron::edges in mesh/tetrahedron.cpp)
   int o = 4;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      dof_map[idx(i,0,0)] = o;
      Nodes.IntPoint(o++).Set3(real_t(i)/p, 0., 0.);
   }
   for (int i = 1; i < p; i++)  // (0,2)
   {
      dof_map[idx(0,i,0)] = o;
      Nodes.IntPoint(o++).Set3(0., real_t(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      dof_map[idx(0,0,i)] = o;
      Nodes.IntPoint(o++).Set3(0., 0., real_t(i)/p);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      dof_map[idx(p-i,i,0)] = o;
      Nodes.IntPoint(o++).Set3(real_t(p-i)/p, real_t(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)  // (1,3)
   {
      dof_map[idx(p-i,0,i)] = o;
      Nodes.IntPoint(o++).Set3(real_t(p-i)/p, 0., real_t(i)/p);
   }
   for (int i = 1; i < p; i++)  // (2,3)
   {
      dof_map[idx(0,p-i,i)] = o;
      Nodes.IntPoint(o++).Set3(0., real_t(p-i)/p, real_t(i)/p);
   }

   // faces (see Mesh::GenerateFaces in mesh/mesh.cpp)
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1,2,3)
      {
         dof_map[idx(p-i-j,i,j)] = o;
         Nodes.IntPoint(o++).Set3(real_t(p-i-j)/p, real_t(i)/p, real_t(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,3,2)
      {
         dof_map[idx(0,j,i)] = o;
         Nodes.IntPoint(o++).Set3(0., real_t(j)/p, real_t(i)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,1,3)
      {
         dof_map[idx(i,0,j)] = o;
         Nodes.IntPoint(o++).Set3(real_t(i)/p, 0., real_t(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0,2,1)
      {
         dof_map[idx(j,i,0)] = o;
         Nodes.IntPoint(o++).Set3(real_t(j)/p, real_t(i)/p, 0.);
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
         for (int i = 1; i + j + k < p; i++)
         {
            dof_map[idx(i,j,k)] = o;
            Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, real_t(k)/p);
         }
}

// static method
void H1Pos_TetrahedronElement::CalcShape(
   const int p, const real_t l1, const real_t l2, const real_t l3,
   real_t *shape)
{
   const real_t l4 = 1. - l1 - l2 - l3;

   // The basis functions are the terms in the expansion:
   //   (l1 + l2 + l3 + l4)^p =
   //      \sum_{k=0}^p \binom{p}{k} l3^k
   //         \sum_{j=0}^{p-k} \binom{p-k}{j} l2^j
   //            \sum_{i=0}^{p-k-j} \binom{p-k-j}{i} l1^i l4^{p-k-j-i}
   const int *bp = Poly_1D::Binom(p);
   real_t l3k = 1.;
   for (int o = 0, k = 0; k <= p; k++)
   {
      const int *bpk = Poly_1D::Binom(p - k);
      const real_t ek = bp[k]*l3k;
      real_t l2j = 1.;
      for (int j = 0; j <= p - k; j++)
      {
         Poly_1D::CalcBinomTerms(p - k - j, l1, l4, &shape[o]);
         real_t ekj = ek*bpk[j]*l2j;
         for (int i = 0; i <= p - k - j; i++)
         {
            shape[o++] *= ekj;
         }
         l2j *= l2;
      }
      l3k *= l3;
   }
}

// static method
void H1Pos_TetrahedronElement::CalcDShape(
   const int p, const real_t l1, const real_t l2, const real_t l3,
   real_t *dshape_1d, real_t *dshape)
{
   const int dof = ((p + 1)*(p + 2)*(p + 3))/6;
   const real_t l4 = 1. - l1 - l2 - l3;

   // For the x derivatives, differentiate the terms of the expression:
   //   \sum_{k=0}^p \binom{p}{k} l3^k
   //      \sum_{j=0}^{p-k} \binom{p-k}{j} l2^j
   //         \sum_{i=0}^{p-k-j} \binom{p-k-j}{i} l1^i l4^{p-k-j-i}
   const int *bp = Poly_1D::Binom(p);
   real_t l3k = 1.;
   for (int o = 0, k = 0; k <= p; k++)
   {
      const int *bpk = Poly_1D::Binom(p - k);
      const real_t ek = bp[k]*l3k;
      real_t l2j = 1.;
      for (int j = 0; j <= p - k; j++)
      {
         Poly_1D::CalcDBinomTerms(p - k - j, l1, l4, dshape_1d);
         real_t ekj = ek*bpk[j]*l2j;
         for (int i = 0; i <= p - k - j; i++)
         {
            dshape[o++] = dshape_1d[i]*ekj;
         }
         l2j *= l2;
      }
      l3k *= l3;
   }
   // For the y derivatives, differentiate the terms of the expression:
   //   \sum_{k=0}^p \binom{p}{k} l3^k
   //      \sum_{i=0}^{p-k} \binom{p-k}{i} l1^i
   //         \sum_{j=0}^{p-k-i} \binom{p-k-i}{j} l2^j l4^{p-k-j-i}
   l3k = 1.;
   for (int ok = 0, k = 0; k <= p; k++)
   {
      const int *bpk = Poly_1D::Binom(p - k);
      const real_t ek = bp[k]*l3k;
      real_t l1i = 1.;
      for (int i = 0; i <= p - k; i++)
      {
         Poly_1D::CalcDBinomTerms(p - k - i, l2, l4, dshape_1d);
         real_t eki = ek*bpk[i]*l1i;
         int o = ok + i;
         for (int j = 0; j <= p - k - i; j++)
         {
            dshape[dof + o] = dshape_1d[j]*eki;
            o += p - k - j + 1;
         }
         l1i *= l1;
      }
      l3k *= l3;
      ok += ((p - k + 2)*(p - k + 1))/2;
   }
   // For the z derivatives, differentiate the terms of the expression:
   //   \sum_{j=0}^p \binom{p}{j} l2^j
   //      \sum_{i=0}^{p-j} \binom{p-j}{i} l1^i
   //         \sum_{k=0}^{p-j-i} \binom{p-j-i}{k} l3^k l4^{p-k-j-i}
   real_t l2j = 1.;
   for (int j = 0; j <= p; j++)
   {
      const int *bpj = Poly_1D::Binom(p - j);
      const real_t ej = bp[j]*l2j;
      real_t l1i = 1.;
      for (int i = 0; i <= p - j; i++)
      {
         Poly_1D::CalcDBinomTerms(p - j - i, l3, l4, dshape_1d);
         real_t eji = ej*bpj[i]*l1i;
         int m = ((p + 2)*(p + 1))/2;
         int n = ((p - j + 2)*(p - j + 1))/2;
         for (int o = i, k = 0; k <= p - j - i; k++)
         {
            // m = ((p - k + 2)*(p - k + 1))/2;
            // n = ((p - k - j + 2)*(p - k - j + 1))/2;
            o += m;
            dshape[2*dof + o - n] = dshape_1d[k]*eji;
            m -= p - k + 1;
            n -= p - k - j + 1;
         }
         l1i *= l1;
      }
      l2j *= l2;
   }
}

void H1Pos_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape(dof);
#endif
   CalcShape(order, ip.x, ip.y, ip.z, m_shape.GetData());
   for (int i = 0; i < dof; i++)
   {
      shape(dof_map[i]) = m_shape(i);
   }
}

void H1Pos_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(order + 1);
   DenseMatrix m_dshape(dof, dim);
#endif
   CalcDShape(order, ip.x, ip.y, ip.z, dshape_1d.GetData(), m_dshape.Data());
   for (int d = 0; d < 3; d++)
   {
      for (int i = 0; i < dof; i++)
      {
         dshape(dof_map[i],d) = m_dshape(i,d);
      }
   }
}


H1Pos_WedgeElement::H1Pos_WedgeElement(const int p)
   : PositiveFiniteElement(3, Geometry::PRISM,
                           ((p + 1)*(p + 1)*(p + 2))/2, p, FunctionSpace::Qk),
     TriangleFE(p),
     SegmentFE(p)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(dof);
   s_dof.SetSize(dof);

   // Nodal DoFs
   t_dof[0] = 0; s_dof[0] = 0;
   t_dof[1] = 1; s_dof[1] = 0;
   t_dof[2] = 2; s_dof[2] = 0;
   t_dof[3] = 0; s_dof[3] = 1;
   t_dof[4] = 1; s_dof[4] = 1;
   t_dof[5] = 2; s_dof[5] = 1;

   // Edge DoFs
   int ne = p-1;
   for (int i=1; i<p; i++)
   {
      t_dof[5 + 0 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 0 * ne + i] = 0;
      t_dof[5 + 1 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 1 * ne + i] = 0;
      t_dof[5 + 2 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 2 * ne + i] = 0;
      t_dof[5 + 3 * ne + i] = 2 + 0 * ne + i; s_dof[5 + 3 * ne + i] = 1;
      t_dof[5 + 4 * ne + i] = 2 + 1 * ne + i; s_dof[5 + 4 * ne + i] = 1;
      t_dof[5 + 5 * ne + i] = 2 + 2 * ne + i; s_dof[5 + 5 * ne + i] = 1;
      t_dof[5 + 6 * ne + i] = 0;              s_dof[5 + 6 * ne + i] = i + 1;
      t_dof[5 + 7 * ne + i] = 1;              s_dof[5 + 7 * ne + i] = i + 1;
      t_dof[5 + 8 * ne + i] = 2;              s_dof[5 + 8 * ne + i] = i + 1;
   }

   // Triangular Face DoFs
   int k=0;
   int nt = (p-1)*(p-2)/2;
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<j; i++)
      {
         t_dof[6 + 9 * ne + k]      = 3 * p + k; s_dof[6 + 9 * ne + k]      = 0;
         t_dof[6 + 9 * ne + nt + k] = 3 * p + k; s_dof[6 + 9 * ne + nt + k] = 1;
         k++;
      }
   }

   // Quadrilateral Face DoFs
   k=0;
   int nq = (p-1)*(p-1);
   for (int j=1; j<p; j++)
   {
      for (int i=1; i<p; i++)
      {
         t_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 2 + 0 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 2 + 1 * ne + i;
         t_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 2 + 2 * ne + i;

         s_dof[6 + 9 * ne + 2 * nt + 0 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 1 * nq + k] = 1 + j;
         s_dof[6 + 9 * ne + 2 * nt + 2 * nq + k] = 1 + j;

         k++;
      }
   }

   // Interior DoFs
   int m=0;
   for (k=1; k<p; k++)
   {
      int l=0;
      for (int j=1; j<p; j++)
      {
         for (int i=1; i<j; i++)
         {
            t_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 3 * p + l;
            s_dof[6 + 9 * ne + 2 * nt + 3 * nq + m] = 1 + k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void H1Pos_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void H1Pos_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}

H1Pos_PyramidElement::H1Pos_PyramidElement(const int p)
   : PositiveFiniteElement(3, Geometry::PYRAMID,
                           ((p + 1)*(p + 2)*(2 * p + 3))/6, p,
                           FunctionSpace::Uk),
     nterms(((p + 1)*(p + 2)*(p + 3)*(p + 4))/24)
{
#ifndef MFEM_THREAD_SAFE
   m_shape_1d.SetSize(order + 1);
   m_shape.SetSize(nterms);
   m_dshape.SetSize(nterms, dim);
#endif

   Index idx;

   // vertices
   dof_map[idx(p,0,0,0,0)] = 0;
   Nodes.IntPoint(0).Set3(0., 0., 0.);
   dof_map[idx(0,p,0,0,0)] = 1;
   Nodes.IntPoint(1).Set3(1., 0., 0.);
   dof_map[idx(0,0,p,0,0)] = 2;
   Nodes.IntPoint(2).Set3(1., 1., 0.);
   dof_map[idx(0,0,0,p,0)] = 3;
   Nodes.IntPoint(3).Set3(0., 1., 0.);
   dof_map[idx(0,0,0,0,p)] = 4;
   Nodes.IntPoint(4).Set3(0., 0., 1.);

   // edges (see Geometry::Constants<Geometry::PYRAMID>::Edges
   // in fem/geom.cpp)
   int o = 5;
   for (int i = 1; i < p; i++)  // (0,1)
   {
      dof_map[idx(p-i,i,0,0,0)] = o;
      Nodes.IntPoint(o++).Set3(real_t(i)/p, 0., 0.);
   }
   for (int i = 1; i < p; i++)  // (1,2)
   {
      dof_map[idx(0,p-i,i,0,0)] = o;
      Nodes.IntPoint(o++).Set3(1.0, real_t(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)  // (3,2)
   {
      dof_map[idx(0,0,i,p-i,0)] = o;
      Nodes.IntPoint(o++).Set3(real_t(i)/p, 1., 0.);
   }
   for (int i = 1; i < p; i++)  // (0,3)
   {
      dof_map[idx(p-i,0,0,i,0)] = o;
      Nodes.IntPoint(o++).Set3(0., real_t(i)/p, 0.);
   }
   for (int i = 1; i < p; i++)  // (0,4)
   {
      dof_map[idx(p-i,0,0,0,i)] = o;
      Nodes.IntPoint(o++).Set3(0., 0., real_t(i)/p);
   }
   for (int i = 1; i < p; i++)  // (1,4)
   {
      dof_map[idx(0,p-i,0,0,i)] = o;
      Nodes.IntPoint(o++).Set3(real_t(p-i)/p, 0., real_t(i)/p);
   }
   for (int i = 1; i < p; i++)  // (2,4)
   {
      dof_map[idx(0,0,p-i,0,i)] = o;
      Nodes.IntPoint(o++).Set3(real_t(p-i)/p, real_t(p-i)/p, real_t(i)/p);
   }
   for (int i = 1; i < p; i++)  // (3,4)
   {
      dof_map[idx(0,0,0,p-i,i)] = o;
      Nodes.IntPoint(o++).Set3(0., real_t(p-i)/p, real_t(i)/p);
   }

   // faces (see Geometry::Constants<Geometry::PYRAMID>::FaceVert
   // in fem/geom.cpp)
   for (int j = 1; j < p; j++)
   {
      int i1 = j;
      int i2 = 0;
      int i3 = 0;
      int i4 = p - j;
      const int i5 = 0;

      for (int i = 1; i <= p - j; i++)  // (3,2,1,0)
      {
         i3++;
         i4--;
         dof_map[idx(i1,i2,i3,i4,i5)] = o;
         Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(p-j)/p, 0);
      }
      for (int i = p - j + 1; i < p; i++)  // (3,2,1,0)
      {
         i1--;
         i2++;
         dof_map[idx(i1,i2,i3,i4,i5)] = o;
         Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(p-j)/p, 0);
      }
   }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (0, 1, 4)
      {
         dof_map[idx(p-i-j,i,0,0,j)] = o;
         Nodes.IntPoint(o++).Set3(real_t(i)/p, 0., real_t(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (1, 2, 4)
      {
         dof_map[idx(0,p-i-j,i,0,j)] = o;
         Nodes.IntPoint(o++).Set3(real_t(p-j)/p, real_t(i)/p, real_t(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (2, 3, 4)
      {
         dof_map[idx(0,0,p-i-j,i,j)] = o;
         Nodes.IntPoint(o++).Set3(real_t(p-i-j)/p, real_t(p-j)/p, real_t(j)/p);
      }
   for (int j = 1; j < p; j++)
      for (int i = 1; i + j < p; i++)  // (3, 0, 4)
      {
         dof_map[idx(i,0,0,p-i-j,j)] = o;
         Nodes.IntPoint(o++).Set3(0., real_t(p-i-j)/p, real_t(j)/p);
      }

   // interior
   for (int k = 1; k < p; k++)
      for (int j = 1; j + k < p; j++)
      {
         int i1 = p - j - k;
         int i2 = 0;
         int i3 = 0;
         int i4 = j;
         const int i5 = k;

         for (int i = 1; i <= j; i++)
         {
            i3++;
            i4--;
            dof_map[idx(i1,i2,i3,i4,i5)] = o;
            Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, 0);
         }
         for (int i = j + 1; i + k < p; i++)
         {
            i1--;
            i2++;
            dof_map[idx(i1,i2,i3,i4,i5)] = o;
            Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, 0);
         }
      }
}

// static method
void H1Pos_PyramidElement::CalcShape(const int p, const real_t x,
                                     const real_t y, const real_t z,
                                     real_t *shape_1d,
                                     real_t *shape)
{
   const int lshape = ((p + 1)*(p + 2)*(p + 3)*(p + 4))/24;
   for (int i=0; i<lshape; i++) { shape[i] = 0.0; }

   const real_t l1 = lam1(x, y, z);
   const real_t l2 = lam2(x, y, z);
   const real_t l3 = lam3(x, y, z);
   const real_t l4 = lam4(x, y, z);
   const real_t l5 = lam5(x, y, z);

   // The basis functions are the terms in the expansion:
   //   (l1 + l2 + l3 + l4 + l5)^p =
   //      \sum_{l=0}^p \binom{p}{l} l5^l
   //         \sum_{k=0}^{p-l} \binom{p-l}{k} l4^k
   //            \sum_{j=0}^{p-l-k} \binom{p-l-k}{j} l3^j
   //               \sum_{i=0}^{p-l-k-j} \binom{p-l-k-j}{i} l2^i l1^{p-l-k-j-i}
   Index idx;
   const int *bp = Poly_1D::Binom(p);
   real_t l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, shape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               shape_1d[i2] *= ei345;
               shape[o] += shape_1d[i2];
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }
}

// static method
void H1Pos_PyramidElement::CalcDShape(const int p, const real_t x,
                                      const real_t y, const real_t z,
                                      real_t *dshape_1d, real_t *dshape)
{
   const int nterms = ((p + 1)*(p + 2)*(p + 3)*(p + 4))/24;
   for (int i=0; i<3*nterms; i++) { dshape[i] = 0.0; }

   const real_t l1 = lam1(x, y, z);
   const real_t l2 = lam2(x, y, z);
   const real_t l3 = lam3(x, y, z);
   const real_t l4 = lam4(x, y, z);
   const real_t l5 = lam5(x, y, z);

   const Vector dl1 = grad_lam1(x, y, z);
   const Vector dl2 = grad_lam2(x, y, z);
   const Vector dl3 = grad_lam3(x, y, z);
   const Vector dl4 = grad_lam4(x, y, z);
   const Vector dl5 = grad_lam5(x, y, z);

   // The basis functions are the terms in the expansion:
   //   (l1 + l2 + l3 + l4 + l5)^p
   // We will compute the derivative by first computing the derivatives
   // of these terms w.r.t each of the l1, l2, l3, l4, and l5 and summing
   // the results together.
   Index idx;

   // Derivative w.r.t. l1 times grad(l1)
   const int *bp = Poly_1D::Binom(p);
   real_t l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcDyBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl1 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl1 * dl1[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l2 times grad(l2)
   l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcDxBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl2 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl2*dl2[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l3 times grad(l3)
   l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 1; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = i3*ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl3 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl3*dl3[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l4 times grad(l4)
   l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 1; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = i4*ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl4 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl4*dl4[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l5 times grad(l5)
   l5i5 = 1.;
   for (int i5 = 1; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = i5*bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl5 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl5*dl5[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }
}

void H1Pos_PyramidElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape_1d(order + 1);
   Vector m_shape(nterms);
#endif

   CalcShape(order, ip.x, ip.y, ip.z, m_shape_1d.GetData(), m_shape.GetData());

   for (auto const& it : dof_map)
   {
      if (it.first < m_shape.Size()) { shape[it.second] = m_shape[it.first]; }
   }
}

void H1Pos_PyramidElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape_1d(order + 1);
   DenseMatrix m_dshape(nterms, 3);
#endif

   CalcDShape(order, ip.x, ip.y, ip.z,
              m_shape_1d.GetData(), m_dshape.GetData());

   for (auto const& it : dof_map)
      for (int d=0; d<3; d++)
      {
         dshape(it.second, d) = m_dshape(it.first, d);
      }

}

L2Pos_SegmentElement::L2Pos_SegmentElement(const int p)
   : PositiveTensorFiniteElement(1, p, L2_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   dshape_x.SetDataAndSize(NULL, p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).x = 0.5;
   }
   else
   {
      for (int i = 0; i <= p; i++)
      {
         Nodes.IntPoint(i).x = real_t(i)/p;
      }
   }
}

void L2Pos_SegmentElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
   Poly_1D::CalcBernstein(order, ip.x, shape);
}

void L2Pos_SegmentElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector shape_x(dof), dshape_x(dshape.Data(), dof);
#else
   dshape_x.SetData(dshape.Data());
#endif
   Poly_1D::CalcBernstein(order, ip.x, shape_x, dshape_x);
}

void L2Pos_SegmentElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   dofs[vertex*order] = 1.0;
}


L2Pos_QuadrilateralElement::L2Pos_QuadrilateralElement(const int p)
   : PositiveTensorFiniteElement(2, p, L2_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set2(0.5, 0.5);
   }
   else
   {
      for (int o = 0, j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            Nodes.IntPoint(o++).Set2(real_t(i)/p, real_t(j)/p);
         }
   }
}

void L2Pos_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         shape(o++) = shape_x(i)*shape_y(j);
      }
}

void L2Pos_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), dshape_x(p+1), dshape_y(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x, dshape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y, dshape_y);

   for (int o = 0, j = 0; j <= p; j++)
      for (int i = 0; i <= p; i++)
      {
         dshape(o,0) = dshape_x(i)* shape_y(j);
         dshape(o,1) =  shape_x(i)*dshape_y(j);  o++;
      }
}

void L2Pos_QuadrilateralElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;

   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[p] = 1.0; break;
      case 2: dofs[p*(p + 2)] = 1.0; break;
      case 3: dofs[p*(p + 1)] = 1.0; break;
   }
}


L2Pos_HexahedronElement::L2Pos_HexahedronElement(const int p)
   : PositiveTensorFiniteElement(3, p, L2_DOF_MAP)
{
#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p + 1);
   shape_y.SetSize(p + 1);
   shape_z.SetSize(p + 1);
   dshape_x.SetSize(p + 1);
   dshape_y.SetSize(p + 1);
   dshape_z.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set3(0.5, 0.5, 0.5);
   }
   else
   {
      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j <= p; j++)
            for (int i = 0; i <= p; i++)
            {
               Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, real_t(k)/p);
            }
   }
}

void L2Pos_HexahedronElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1), shape_y(p+1), shape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y);
   Poly_1D::CalcBernstein(p, ip.z, shape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            shape(o++) = shape_x(i)*shape_y(j)*shape_z(k);
         }
}

void L2Pos_HexahedronElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+1),  shape_y(p+1),  shape_z(p+1);
   Vector dshape_x(p+1), dshape_y(p+1), dshape_z(p+1);
#endif

   Poly_1D::CalcBernstein(p, ip.x, shape_x, dshape_x);
   Poly_1D::CalcBernstein(p, ip.y, shape_y, dshape_y);
   Poly_1D::CalcBernstein(p, ip.z, shape_z, dshape_z);

   for (int o = 0, k = 0; k <= p; k++)
      for (int j = 0; j <= p; j++)
         for (int i = 0; i <= p; i++)
         {
            dshape(o,0) = dshape_x(i)* shape_y(j)* shape_z(k);
            dshape(o,1) =  shape_x(i)*dshape_y(j)* shape_z(k);
            dshape(o,2) =  shape_x(i)* shape_y(j)*dshape_z(k);  o++;
         }
}

void L2Pos_HexahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   const int p = order;

   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[p] = 1.0; break;
      case 2: dofs[p*(p + 2)] = 1.0; break;
      case 3: dofs[p*(p + 1)] = 1.0; break;
      case 4: dofs[p*(p + 1)*(p + 1)] = 1.0; break;
      case 5: dofs[p + p*(p + 1)*(p + 1)] = 1.0; break;
      case 6: dofs[dof - 1] = 1.0; break;
      case 7: dofs[dof - p - 1] = 1.0; break;
   }
}


L2Pos_TriangleElement::L2Pos_TriangleElement(const int p)
   : PositiveFiniteElement(2, Geometry::TRIANGLE, ((p + 1)*(p + 2))/2, p,
                           FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   dshape_1d.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set2(1./3, 1./3);
   }
   else
   {
      for (int o = 0, j = 0; j <= p; j++)
         for (int i = 0; i + j <= p; i++)
         {
            Nodes.IntPoint(o++).Set2(real_t(i)/p, real_t(j)/p);
         }
   }
}

void L2Pos_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                      Vector &shape) const
{
   H1Pos_TriangleElement::CalcShape(order, ip.x, ip.y, shape.GetData());
}

void L2Pos_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                       DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(order + 1);
#endif

   H1Pos_TriangleElement::CalcDShape(order, ip.x, ip.y, dshape_1d.GetData(),
                                     dshape.Data());
}

void L2Pos_TriangleElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[order] = 1.0; break;
      case 2: dofs[dof-1] = 1.0; break;
   }
}


L2Pos_TetrahedronElement::L2Pos_TetrahedronElement(const int p)
   : PositiveFiniteElement(3, Geometry::TETRAHEDRON,
                           ((p + 1)*(p + 2)*(p + 3))/6, p, FunctionSpace::Pk)
{
#ifndef MFEM_THREAD_SAFE
   dshape_1d.SetSize(p + 1);
#endif

   if (p == 0)
   {
      Nodes.IntPoint(0).Set3(0.25, 0.25, 0.25);
   }
   else
   {
      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
            for (int i = 0; i + j + k <= p; i++)
            {
               Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, real_t(k)/p);
            }
   }
}

void L2Pos_TetrahedronElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   H1Pos_TetrahedronElement::CalcShape(order, ip.x, ip.y, ip.z,
                                       shape.GetData());
}

void L2Pos_TetrahedronElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector dshape_1d(order + 1);
#endif

   H1Pos_TetrahedronElement::CalcDShape(order, ip.x, ip.y, ip.z,
                                        dshape_1d.GetData(), dshape.Data());
}

void L2Pos_TetrahedronElement::ProjectDelta(int vertex, Vector &dofs) const
{
   dofs = 0.0;
   switch (vertex)
   {
      case 0: dofs[0] = 1.0; break;
      case 1: dofs[order] = 1.0; break;
      case 2: dofs[(order*(order+3))/2] = 1.0; break;
      case 3: dofs[dof-1] = 1.0; break;
   }
}


L2Pos_WedgeElement::L2Pos_WedgeElement(const int p)
   : PositiveFiniteElement(3, Geometry::PRISM,
                           ((p + 1)*(p + 1)*(p + 2))/2, p, FunctionSpace::Qk),
     TriangleFE(p),
     SegmentFE(p)
{
#ifndef MFEM_THREAD_SAFE
   t_shape.SetSize(TriangleFE.GetDof());
   s_shape.SetSize(SegmentFE.GetDof());
   t_dshape.SetSize(TriangleFE.GetDof(), 2);
   s_dshape.SetSize(SegmentFE.GetDof(), 1);
#endif

   t_dof.SetSize(dof);
   s_dof.SetSize(dof);

   // Interior DoFs
   int m=0;
   for (int k=0; k<=p; k++)
   {
      int l=0;
      for (int j=0; j<=p; j++)
      {
         for (int i=0; i<=j; i++)
         {
            t_dof[m] = l;
            s_dof[m] = k;
            l++; m++;
         }
      }
   }

   // Define Nodes
   const IntegrationRule & t_Nodes = TriangleFE.GetNodes();
   const IntegrationRule & s_Nodes = SegmentFE.GetNodes();
   for (int i=0; i<dof; i++)
   {
      Nodes.IntPoint(i).x = t_Nodes.IntPoint(t_dof[i]).x;
      Nodes.IntPoint(i).y = t_Nodes.IntPoint(t_dof[i]).y;
      Nodes.IntPoint(i).z = s_Nodes.IntPoint(s_dof[i]).x;
   }
}

void L2Pos_WedgeElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector t_shape(TriangleFE.GetDof());
   Vector s_shape(SegmentFE.GetDof());
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   SegmentFE.CalcShape(ipz, s_shape);

   for (int i=0; i<dof; i++)
   {
      shape[i] = t_shape[t_dof[i]] * s_shape[s_dof[i]];
   }
}

void L2Pos_WedgeElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector      t_shape(TriangleFE.GetDof());
   DenseMatrix t_dshape(TriangleFE.GetDof(), 2);
   Vector      s_shape(SegmentFE.GetDof());
   DenseMatrix s_dshape(SegmentFE.GetDof(), 1);
#endif

   IntegrationPoint ipz; ipz.x = ip.z; ipz.y = 0.0; ipz.z = 0.0;

   TriangleFE.CalcShape(ip, t_shape);
   TriangleFE.CalcDShape(ip, t_dshape);
   SegmentFE.CalcShape(ipz, s_shape);
   SegmentFE.CalcDShape(ipz, s_dshape);

   for (int i=0; i<dof; i++)
   {
      dshape(i, 0) = t_dshape(t_dof[i],0) * s_shape[s_dof[i]];
      dshape(i, 1) = t_dshape(t_dof[i],1) * s_shape[s_dof[i]];
      dshape(i, 2) = t_shape[t_dof[i]] * s_dshape(s_dof[i],0);
   }
}

L2Pos_PyramidElement::L2Pos_PyramidElement(const int p)
   : PositiveFiniteElement(3, Geometry::PYRAMID,
                           ((p + 1)*(p + 2)*(2 * p + 3))/6, p,
                           FunctionSpace::Uk),
     nterms(((p + 1)*(p + 2)*(p + 3)*(p + 4))/24)
{
#ifndef MFEM_THREAD_SAFE
   m_shape_1d.SetSize(order + 1);
   m_shape.SetSize(nterms);
   m_dshape.SetSize(nterms, dim);
#endif

   Index idx;

   if (p == 0)
   {
      dof_map[idx(0,0,0,0,0)] = 0;
      Nodes.IntPoint(0).Set3(0.375, 0.375, 0.25);
   }
   else
   {
      for (int o = 0, k = 0; k <= p; k++)
         for (int j = 0; j + k <= p; j++)
         {
            int i1 = p - j - k;
            int i2 = 0;
            int i3 = -1;
            int i4 = j + 1;
            const int i5 = k;

            for (int i = 0; i <= j; i++)
            {
               i3++;
               i4--;
               dof_map[idx(i1,i2,i3,i4,i5)] = o;
               Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, 0);
            }
            for (int i = j + 1; i + k <= p; i++)
            {
               i1--;
               i2++;
               dof_map[idx(i1,i2,i3,i4,i5)] = o;
               Nodes.IntPoint(o++).Set3(real_t(i)/p, real_t(j)/p, 0);
            }
         }
   }
}

// static method
void L2Pos_PyramidElement::CalcShape(const int p, const real_t x,
                                     const real_t y, const real_t z,
                                     real_t *shape_1d,
                                     real_t *shape)
{
   const int lshape = ((p + 1)*(p + 2)*(p + 3)*(p + 4))/24;
   for (int i=0; i<lshape; i++) { shape[i] = 0.0; }

   const real_t l1 = lam1(x, y, z);
   const real_t l2 = lam2(x, y, z);
   const real_t l3 = lam3(x, y, z);
   const real_t l4 = lam4(x, y, z);
   const real_t l5 = lam5(x, y, z);

   // The basis functions are the terms in the expansion:
   //   (l1 + l2 + l3 + l4 + l5)^p =
   //      \sum_{l=0}^p \binom{p}{l} l5^l
   //         \sum_{k=0}^{p-l} \binom{p-l}{k} l4^k
   //            \sum_{j=0}^{p-l-k} \binom{p-l-k}{j} l3^j
   //               \sum_{i=0}^{p-l-k-j} \binom{p-l-k-j}{i} l2^i l1^{p-l-k-j-i}
   Index idx;
   const int *bp = Poly_1D::Binom(p);
   real_t l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, shape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               shape_1d[i2] *= ei345;
               shape[o] += shape_1d[i2];
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }
}

// static method
void L2Pos_PyramidElement::CalcDShape(const int p, const real_t x,
                                      const real_t y, const real_t z,
                                      real_t *dshape_1d, real_t *dshape)
{
   const int nterms = ((p + 1)*(p + 2)*(p + 3)*(p + 4))/24;
   for (int i=0; i<3*nterms; i++) { dshape[i] = 0.0; }

   const real_t l1 = lam1(x, y, z);
   const real_t l2 = lam2(x, y, z);
   const real_t l3 = lam3(x, y, z);
   const real_t l4 = lam4(x, y, z);
   const real_t l5 = lam5(x, y, z);

   const Vector dl1 = grad_lam1(x, y, z);
   const Vector dl2 = grad_lam2(x, y, z);
   const Vector dl3 = grad_lam3(x, y, z);
   const Vector dl4 = grad_lam4(x, y, z);
   const Vector dl5 = grad_lam5(x, y, z);

   // The basis functions are the terms in the expansion:
   //   (l1 + l2 + l3 + l4 + l5)^p
   // We will compute the derivative by first computing the derivatives
   // of these terms w.r.t each of the l1, l2, l3, l4, and l5 and summing
   // the results together.
   Index idx;

   // Derivative w.r.t. l1 times grad(l1)
   const int *bp = Poly_1D::Binom(p);
   real_t l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcDyBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl1 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl1 * dl1[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l2 times grad(l2)
   l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcDxBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl2 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl2*dl2[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l3 times grad(l3)
   l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 1; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = i3*ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl3 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl3*dl3[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l4 times grad(l4)
   l5i5 = 1.;
   for (int i5 = 0; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 1; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = i4*ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl4 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl4*dl4[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }

   // Derivative w.r.t. l5 times grad(l5)
   l5i5 = 1.;
   for (int i5 = 1; i5 <= p; i5++)
   {
      const int *bpi5 = Poly_1D::Binom(p - i5);
      const real_t ei5 = i5*bp[i5]*l5i5;
      real_t l4i4 = 1.;
      for (int i4 = 0; i4 <= p - i5; i4++)
      {
         const int *bpi45 = Poly_1D::Binom(p - i5 - i4);
         const real_t ei45 = ei5*bpi5[i4]*l4i4;
         real_t l3i3 = 1.;
         for (int i3 = 0; i3 <= p - i5 - i4; i3++)
         {
            Poly_1D::CalcBinomTerms(p - i5 - i4 - i3, l2, l1, dshape_1d);
            real_t ei345 = ei45*bpi45[i3]*l3i3;
            for (int i2 = 0; i2 <= p - i5 - i4 - i3; i2++)
            {
               const int i1 = p - i5 - i4 - i3 - i2;
               const int o = idx(i1,i2,i3,i4,i5);
               const real_t dshape_dl5 = dshape_1d[i2]*ei345;
               for (int d = 0; d < 3; d++)
               {
                  dshape[o + d * nterms] += dshape_dl5*dl5[d];
               }
            }
            l3i3 *= l3;
         }
         l4i4 *= l4;
      }
      l5i5 *= l5;
   }
}

void L2Pos_PyramidElement::CalcShape(const IntegrationPoint &ip,
                                     Vector &shape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape_1d(order + 1);
   Vector m_shape(nterms);
#endif

   CalcShape(order, ip.x, ip.y, ip.z, m_shape_1d.GetData(), m_shape.GetData());

   for (auto const& it : dof_map)
   {
      if (it.first < m_shape.Size()) { shape[it.second] = m_shape[it.first]; }
   }
}

void L2Pos_PyramidElement::CalcDShape(const IntegrationPoint &ip,
                                      DenseMatrix &dshape) const
{
#ifdef MFEM_THREAD_SAFE
   Vector m_shape_1d(order + 1);
   DenseMatrix m_dshape(nterms, 3);
#endif

   CalcDShape(order, ip.x, ip.y, ip.z,
              m_shape_1d.GetData(), m_dshape.GetData());

   for (auto const& it : dof_map)
      for (int d=0; d<3; d++)
      {
         dshape(it.second, d) = m_dshape(it.first, d);
      }
}

}
