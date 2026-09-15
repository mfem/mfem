// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.


#include "fe_bell.hpp"
#include "../eltrans.hpp"

namespace mfem
{
namespace
{
constexpr real_t vertices[3][2] = {{0,0}, {1,0}, {0,1}};
constexpr int edges[3][2] = {{0,1}, {1,2}, {2,0}};
}

BellTriangleFiniteElement::BellTriangleFiniteElement()
   : FiniteElement(2, Geometry::TRIANGLE, 18, 5, FunctionSpace::Pk)
{
   deriv_type = GRAD;
   deriv_range_type = VECTOR;
   for (int i = 0; i < dof; i++) { Nodes.IntPoint(i) = argyris.GetNodes().IntPoint(i); }
   DenseMatrix J(2);
   J = 0.0; J(0,0) = J(1,1) = 1.0;
   GetEmbedding(J, reference_embedding);
}

void BellTriangleFiniteElement::GetEmbedding(const DenseMatrix &J,
                                             DenseMatrix &E) const
{
   E.SetSize(21,18);
   E = 0.0;
   for (int i = 0; i < 18; i++) { E(i,i) = 1.0; }
   for (int e = 0; e < 3; e++)
   {
      const int a = edges[e][0], b = edges[e][1];
      const real_t dx = vertices[b][0]-vertices[a][0];
      const real_t dy = vertices[b][1]-vertices[a][1];
      const real_t tx = J(0,0)*dx+J(0,1)*dy;
      const real_t ty = J(1,0)*dx+J(1,1)*dy;
      const real_t nx = ty, ny = -tx;
      // Cubic Hermite interpolation of g = n.grad(u) along the edge:
      // g(1/2) = (g(0)+g(1))/2 + (g'(0)-g'(1))/8.
      for (int end = 0; end < 2; end++)
      {
         const int o = 6*edges[e][end];
         const real_t s = end ? -0.125 : 0.125;
         E(18+e,o+1) = nx/2;
         E(18+e,o+2) = ny/2;
         E(18+e,o+3) = s*tx*nx;
         E(18+e,o+4) = s*(tx*ny+ty*nx);
         E(18+e,o+5) = s*ty*ny;
      }
   }
}

void BellTriangleFiniteElement::CalcShape(const IntegrationPoint &ip,
                                          Vector &shape) const
{
   Vector full(21);
   argyris.CalcShape(ip,full);
   reference_embedding.MultTranspose(full,shape);
}

void BellTriangleFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                           DenseMatrix &shape) const
{
   DenseMatrix full(21,2);
   argyris.CalcDShape(ip,full);
   MultAtB(reference_embedding,full,shape);
}

void BellTriangleFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                            DenseMatrix &shape) const
{
   DenseMatrix full(21,3);
   argyris.CalcHessian(ip,full);
   MultAtB(reference_embedding,full,shape);
}

void BellTriangleFiniteElement::CalcPhysShape(ElementTransformation &T,
                                              Vector &shape) const
{
   Vector full(21);
   DenseMatrix E;
   argyris.CalcPhysShape(T,full);
   GetEmbedding(T.Jacobian(),E);
   E.MultTranspose(full,shape);
}

void BellTriangleFiniteElement::CalcPhysDShape(ElementTransformation &T,
                                               DenseMatrix &shape) const
{
   DenseMatrix full(21,2), E;
   argyris.CalcPhysDShape(T,full);
   GetEmbedding(T.Jacobian(),E);
   MultAtB(E,full,shape);
}

void BellTriangleFiniteElement::CalcPhysHessian(ElementTransformation &T,
                                                DenseMatrix &shape) const
{
   DenseMatrix full(21,3), E;
   argyris.CalcPhysHessian(T,full);
   GetEmbedding(T.Jacobian(),E);
   MultAtB(E,full,shape);
}

void BellTriangleFiniteElement::Interpolate(ElementTransformation &child,
                                            ElementTransformation &coarse,
                                            DenseMatrix &I) const
{
   I.SetSize(dof);
   Vector point(2), value(dof);
   DenseMatrix grad(dof,2), hessian(dof,3);
   for (int v = 0; v < 3; v++)
   {
      child.Transform(Nodes.IntPoint(6*v),point);
      IntegrationPoint ip;
      ip.Set2(point(0),point(1));
      coarse.SetIntPoint(&ip);
      CalcPhysShape(coarse,value);
      CalcPhysDShape(coarse,grad);
      CalcPhysHessian(coarse,hessian);
      for (int j = 0; j < dof; j++)
      {
         I(6*v,j) = value(j);
         for (int d = 0; d < 2; d++) { I(6*v+1+d,j) = grad(j,d); }
         for (int d = 0; d < 3; d++) { I(6*v+3+d,j) = hessian(j,d); }
      }
   }
   coarse.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
}

void BellTriangleFiniteElement::GetTransferMatrix(const FiniteElement &fe,
                                                  ElementTransformation &T,
                                                  DenseMatrix &I) const
{
   MFEM_VERIFY(dynamic_cast<const BellTriangleFiniteElement *>(&fe),
               "Bell transfer requires a Bell source element");
   T.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
   MFEM_VERIFY(T.GetSpaceDim() == 2 && T.Hessian().FNorm2() < 1e-20,
               "Bell transfer requires affine 2D transformations");
   IsoparametricTransformation coarse;
   coarse.SetIdentityTransformation(Geometry::TRIANGLE);
   Interpolate(T,coarse,I);
}

void BellTriangleFiniteElement::GetPhysicalTransferMatrix(
   const DenseMatrix &, ElementTransformation &child,
   ElementTransformation &fine, DenseMatrix &I) const
{
   const auto &center = Geometries.GetCenter(Geometry::TRIANGLE);
   child.SetIntPoint(&center);
   fine.SetIntPoint(&center);
   MFEM_VERIFY(child.GetSpaceDim() == 2 && fine.GetSpaceDim() == 2 &&
               child.Hessian().FNorm2() < 1e-20 && fine.Hessian().FNorm2() < 1e-20,
               "Bell transfer requires affine 2D transformations");
   DenseMatrix K(2), J(2), points(2,3);
   CalcInverse(child.Jacobian(),K);
   Mult(fine.Jacobian(),K,J);
   points = 0.0;
   for (int d = 0; d < 2; d++)
   { points(d,1) = J(d,0); points(d,2) = J(d,1); }
   IsoparametricTransformation coarse;
   coarse.SetIdentityTransformation(Geometry::TRIANGLE);
   coarse.SetPointMat(points);
   // The reduced reference matrix cannot recover the physical constraints.
   // Re-evaluate coarse physical vertex jets at the child vertices instead.
   Interpolate(child,coarse,I);
}

void BellTriangleFiniteElement::GetFaceDofs(int face, int **dofs,
                                            int *ndofs) const
{
   static int indices[3][12] = {{0,1,2,3,4,5,6,7,8,9,10,11},
      {6,7,8,9,10,11,12,13,14,15,16,17},
      {12,13,14,15,16,17,0,1,2,3,4,5}
   };
   MFEM_ASSERT(face >= 0 && face < 3, "invalid face index");
   *dofs = indices[face];
   *ndofs = 12;
}

} // namespace mfem
