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

// Serendipity Finite Element classes

#include "fe_ser.hpp"
#include "fe_fixed_order.hpp"

namespace mfem
{

using namespace std;

H1Ser_QuadrilateralElement::H1Ser_QuadrilateralElement(const int p)
   : ScalarFiniteElement(2, Geometry::SQUARE, (p*p + 3*p +6) / 2, p,
                         FunctionSpace::Qk)
{
   // Store the dof_map of the associated TensorBasisElement, which will be used
   // to create the serendipity dof map.  Its size is larger than the size of
   // the serendipity element.
   TensorBasisElement tbeTemp =
      TensorBasisElement(2, p, BasisType::GaussLobatto,
                         TensorBasisElement::DofMapType::Sr_DOF_MAP);
   const Array<int> tp_dof_map = tbeTemp.GetDofMap();

   const real_t *cp = poly1d.ClosedPoints(p, BasisType::GaussLobatto);

   // Fixing the Nodes is exactly the same as the H1_QuadrilateralElement
   // constructor except we only use those values of the associated tensor
   // product dof_map that are <= the number of serendipity Dofs e.g. only DoFs
   // 0-7 out of the 9 tensor product dofs (at quadratic order)
   int o = 0;

   for (int j = 0; j <= p; j++)
   {
      for (int i = 0; i <= p; i++)
      {
         if (tp_dof_map[o] < Nodes.Size())
         {
            Nodes.IntPoint(tp_dof_map[o]).x = cp[i];
            Nodes.IntPoint(tp_dof_map[o]).y = cp[j];
         }
         o++;
      }
   }
}

void H1Ser_QuadrilateralElement::CalcShape(const IntegrationPoint &ip,
                                           Vector &shape) const
{
   int p = (this)->GetOrder();
   real_t x = ip.x, y = ip.y;

   Poly_1D::Basis &edgeNodalBasis = poly1d.GetBasis(p, BasisType::GaussLobatto);
   Vector nodalX(p+1);
   Vector nodalY(p+1);

   edgeNodalBasis.Eval(x, nodalX);
   edgeNodalBasis.Eval(y, nodalY);

   // First, fix edge-based shape functions. Use a nodal interpolant for edge
   // points, weighted by the linear function that vanishes on opposite edge.
   for (int i = 0; i < p-1; i++)
   {
      shape(4 + 0*(p-1) + i) = (nodalX(i+1))*(1.-y);         // south edge 0->1
      shape(4 + 1*(p-1) + i) = (nodalY(i+1))*x;              // east edge  1->2
      shape(4 + 3*(p-1) - i - 1) = (nodalX(i+1)) * y;        // north edge 3->2
      shape(4 + 4*(p-1) - i - 1) = (nodalY(i+1)) * (1. - x); // west edge  0->3
   }

   BiLinear2DFiniteElement bilinear = BiLinear2DFiniteElement();
   Vector bilinearsAtIP(4);
   bilinear.CalcShape(ip, bilinearsAtIP);

   const real_t *edgePts(poly1d.ClosedPoints(p, BasisType::GaussLobatto));

   // Next, set the shape function associated with vertex V, evaluated at (x,y)
   // to be: bilinear function associated to V, evaluated at (x,y) - sum (shape
   // function at edge point P, weighted by bilinear function for V evaluated at
   // P) where the sum is taken only for points P on edges incident to V.

   real_t vtx0fix =0;
   real_t vtx1fix =0;
   real_t vtx2fix =0;
   real_t vtx3fix =0;
   for (int i = 0; i<p-1; i++)
   {
      vtx0fix += (1-edgePts[i+1])*(shape(4 + i) +
                                   shape(4 + 4*(p-1) - i - 1)); // bot+left edge
      vtx1fix += (1-edgePts[i+1])*(shape(4 + 1*(p-1) + i) +
                                   shape(4 + (p-2)-i));        // right+bot edge
      vtx2fix += (1-edgePts[i+1])*(shape(4 + 2*(p-1) + i) +
                                   shape(1 + 2*p-i));          // top+right edge
      vtx3fix += (1-edgePts[i+1])*(shape(4 + 3*(p-1) + i) +
                                   shape(3*p - i));            // left+top edge
   }
   shape(0) = bilinearsAtIP(0) - vtx0fix;
   shape(1) = bilinearsAtIP(1) - vtx1fix;
   shape(2) = bilinearsAtIP(2) - vtx2fix;
   shape(3) = bilinearsAtIP(3) - vtx3fix;

   // Interior basis functions appear starting at order p=4. These are non-nodal
   // bubble functions.
   if (p > 3)
   {
      real_t *legX = new real_t[p-1];
      real_t *legY = new real_t[p-1];

      Poly_1D::CalcLegendre(p-2, x, legX);
      Poly_1D::CalcLegendre(p-2, y, legY);

      int interior_total = 0;
      for (int j = 4; j < p + 1; j++)
      {
         for (int k = 0; k < j-3; k++)
         {
            shape(4 + 4*(p-1) + interior_total)
               = legX[k] * legY[j-4-k] * x * (1. - x) * y * (1. - y);
            interior_total++;
         }
      }

      delete[] legX;
      delete[] legY;
   }
}

void H1Ser_QuadrilateralElement::CalcDShape(const IntegrationPoint &ip,
                                            DenseMatrix &dshape) const
{
   int p = (this)->GetOrder();
   real_t x = ip.x, y = ip.y;

   Poly_1D::Basis &edgeNodalBasis = poly1d.GetBasis(p, BasisType::GaussLobatto);
   Vector nodalX(p+1);
   Vector DnodalX(p+1);
   Vector nodalY(p+1);
   Vector DnodalY(p+1);

   edgeNodalBasis.Eval(x, nodalX, DnodalX);
   edgeNodalBasis.Eval(y, nodalY, DnodalY);

   for (int i = 0; i < p-1; i++)
   {
      dshape(4 + 0*(p-1) + i,0) =  DnodalX(i+1) * (1.-y);
      dshape(4 + 0*(p-1) + i,1) = -nodalX(i+1);
      dshape(4 + 1*(p-1) + i,0) =  nodalY(i+1);
      dshape(4 + 1*(p-1) + i,1) =  DnodalY(i+1)*x;
      dshape(4 + 3*(p-1) - i - 1,0) =  DnodalX(i+1)*y;
      dshape(4 + 3*(p-1) - i - 1,1) =  nodalX(i+1);
      dshape(4 + 4*(p-1) - i - 1,0) = -nodalY(i+1);
      dshape(4 + 4*(p-1) - i - 1,1) =  DnodalY(i+1) * (1.-x);
   }

   BiLinear2DFiniteElement bilinear = BiLinear2DFiniteElement();
   DenseMatrix DbilinearsAtIP(4);
   bilinear.CalcDShape(ip, DbilinearsAtIP);

   const real_t *edgePts(poly1d.ClosedPoints(p, BasisType::GaussLobatto));

   dshape(0,0) = DbilinearsAtIP(0,0);
   dshape(0,1) = DbilinearsAtIP(0,1);
   dshape(1,0) = DbilinearsAtIP(1,0);
   dshape(1,1) = DbilinearsAtIP(1,1);
   dshape(2,0) = DbilinearsAtIP(2,0);
   dshape(2,1) = DbilinearsAtIP(2,1);
   dshape(3,0) = DbilinearsAtIP(3,0);
   dshape(3,1) = DbilinearsAtIP(3,1);

   for (int i = 0; i<p-1; i++)
   {
      dshape(0,0) -= (1-edgePts[i+1])*(dshape(4 + 0*(p-1) + i, 0) +
                                       dshape(4 + 4*(p-1) - i - 1,0));
      dshape(0,1) -= (1-edgePts[i+1])*(dshape(4 + 0*(p-1) + i, 1) +
                                       dshape(4 + 4*(p-1) - i - 1,1));
      dshape(1,0) -= (1-edgePts[i+1])*(dshape(4 + 1*(p-1) + i, 0) +
                                       dshape(4 + (p-2)-i, 0));
      dshape(1,1) -= (1-edgePts[i+1])*(dshape(4 + 1*(p-1) + i, 1) +
                                       dshape(4 + (p-2)-i, 1));
      dshape(2,0) -= (1-edgePts[i+1])*(dshape(4 + 2*(p-1) + i, 0) +
                                       dshape(1 + 2*p-i, 0));
      dshape(2,1) -= (1-edgePts[i+1])*(dshape(4 + 2*(p-1) + i, 1) +
                                       dshape(1 + 2*p-i, 1));
      dshape(3,0) -= (1-edgePts[i+1])*(dshape(4 + 3*(p-1) + i, 0) +
                                       dshape(3*p - i, 0));
      dshape(3,1) -= (1-edgePts[i+1])*(dshape(4 + 3*(p-1) + i, 1) +
                                       dshape(3*p - i, 1));
   }

   if (p > 3)
   {
      real_t *legX = new real_t[p-1];
      real_t *legY = new real_t[p-1];
      real_t *DlegX = new real_t[p-1];
      real_t *DlegY = new real_t[p-1];

      Poly_1D::CalcLegendre(p-2, x, legX, DlegX);
      Poly_1D::CalcLegendre(p-2, y, legY, DlegY);

      int interior_total = 0;
      for (int j = 4; j < p + 1; j++)
      {
         for (int k = 0; k < j-3; k++)
         {
            dshape(4 + 4*(p-1) + interior_total, 0) =
               legY[j-4-k]*y*(1-y) * (DlegX[k]*x*(1-x) + legX[k]*(1-2*x));
            dshape(4 + 4*(p-1) + interior_total, 1) =
               legX[k]*x*(1-x) * (DlegY[j-4-k]*y*(1-y) + legY[j-4-k]*(1-2*y));
            interior_total++;
         }
      }
      delete[] legX;
      delete[] legY;
      delete[] DlegX;
      delete[] DlegY;
   }
}

void H1Ser_QuadrilateralElement::GetLocalInterpolation(ElementTransformation
                                                       &Trans,
                                                       DenseMatrix &I) const
{
   // For p<=4, the basis is nodal; for p>4, the quad-interior functions are
   // non-nodal.
   if (order <= 4)
   {
      NodalLocalInterpolation(Trans, I, *this);
   }
   else
   {
      ScalarLocalInterpolation(Trans, I, *this);
   }
}

}
