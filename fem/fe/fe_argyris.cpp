// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "fe_argyris.hpp"
#include "../eltrans.hpp"

namespace mfem
{
namespace
{
constexpr int argyris_dof = 21;
constexpr real_t vertices[3][2] = {{0,0}, {1,0}, {0,1}};
constexpr int edge_vertices[3][2] = {{0,1}, {1,2}, {2,0}};
constexpr int derivatives[6][2] = {{0,0}, {1,0}, {0,1}, {2,0}, {1,1}, {0,2}};

void Monomials(const IntegrationPoint &ip, int dx, int dy, Vector &values)
{
   int m = 0;
   for (int degree = 0; degree <= 5; degree++)
   {
      for (int py = 0; py <= degree; py++)
      {
         const int px = degree - py;
         real_t value = (px >= dx && py >= dy) ? 1.0 : 0.0;
         for (int i = 0; i < dx; i++) { value *= px-i; }
         for (int i = 0; i < dy; i++) { value *= py-i; }
         for (int i = dx; i < px; i++) { value *= ip.x; }
         for (int i = dy; i < py; i++) { value *= ip.y; }
         values(m++) = value;
      }
   }
}
// Fill @a I with the values of a C1 triangle's physical degrees of freedom
// applied to the shape functions of @a coarse_fe. The element has
// @a vertex_dofs degrees of freedom at each vertex, ordered value, gradient
// and then Hessian, and one oriented normal derivative on each edge when
// @a edge_dofs is true. @a child maps the child reference cell into the
// reference cell of @a coarse_fe, and @a coarse maps that cell into the frame
// in which the degrees of freedom are expressed.
void InterpolateC1Dofs(const FiniteElement &coarse_fe, int vertex_dofs,
                       bool edge_dofs, ElementTransformation &child,
                       ElementTransformation &coarse, DenseMatrix &I)
{
   const IntegrationPoint &center = Geometries.GetCenter(Geometry::TRIANGLE);
   child.SetIntPoint(&center);
   coarse.SetIntPoint(&center);
   DenseMatrix jacobian(2);
   Mult(coarse.Jacobian(), child.Jacobian(), jacobian);

   const int coarse_dof = coarse_fe.GetDof();
   I.SetSize(3*vertex_dofs + (edge_dofs ? 3 : 0), coarse_dof);
   I = 0.0;
   Vector point(2), value(coarse_dof);
   DenseMatrix grad(coarse_dof, 2), hessian(coarse_dof, 3);
   IntegrationPoint fine_ip, coarse_ip;
   for (int vertex = 0; vertex < 3; vertex++)
   {
      fine_ip.Set2(vertices[vertex][0], vertices[vertex][1]);
      child.Transform(fine_ip, point);
      coarse_ip.Set2(point(0), point(1));
      coarse.SetIntPoint(&coarse_ip);
      coarse_fe.CalcPhysShape(coarse, value);
      coarse_fe.CalcPhysDShape(coarse, grad);
      if (vertex_dofs > 3) { coarse_fe.CalcPhysHessian(coarse, hessian); }
      const int row = vertex_dofs*vertex;
      for (int j = 0; j < coarse_dof; j++)
      {
         I(row, j) = value(j);
         I(row + 1, j) = grad(j, 0);
         I(row + 2, j) = grad(j, 1);
         for (int d = 3; d < vertex_dofs; d++)
         {
            I(row + d, j) = hessian(j, d - 3);
         }
      }
   }
   for (int edge = 0; edge_dofs && edge < 3; edge++)
   {
      const int v0 = edge_vertices[edge][0], v1 = edge_vertices[edge][1];
      fine_ip.Set2(0.5*(vertices[v0][0] + vertices[v1][0]),
                   0.5*(vertices[v0][1] + vertices[v1][1]));
      child.Transform(fine_ip, point);
      coarse_ip.Set2(point(0), point(1));
      coarse.SetIntPoint(&coarse_ip);
      coarse_fe.CalcPhysDShape(coarse, grad);
      const real_t tx = vertices[v1][0] - vertices[v0][0];
      const real_t ty = vertices[v1][1] - vertices[v0][1];
      // The same oriented, unnormalized physical normal as the edge DOF.
      const real_t nx = jacobian(1,0)*tx + jacobian(1,1)*ty;
      const real_t ny = -jacobian(0,0)*tx - jacobian(0,1)*ty;
      const int row = 3*vertex_dofs + edge;
      for (int j = 0; j < coarse_dof; j++)
      {
         I(row, j) = nx*grad(j, 0) + ny*grad(j, 1);
      }
   }
   coarse.SetIntPoint(&center);
}

// Place the coarse reference cell in the frame in which physical degrees of
// freedom are measured: the child cell maps onto the true fine element.
void SetRelativeCoarseTransformation(ElementTransformation &child,
                                     ElementTransformation &fine,
                                     IsoparametricTransformation &coarse)
{
   const IntegrationPoint &center = Geometries.GetCenter(Geometry::TRIANGLE);
   child.SetIntPoint(&center);
   fine.SetIntPoint(&center);
   MFEM_VERIFY(child.GetSpaceDim() == 2 && fine.GetSpaceDim() == 2 &&
               child.Hessian().FNorm2() < 1e-20 &&
               fine.Hessian().FNorm2() < 1e-20,
               "C1 transfer requires affine two-dimensional transformations");
   DenseMatrix inverse(2), jacobian(2), points(2, 3);
   CalcInverse(child.Jacobian(), inverse);
   Mult(fine.Jacobian(), inverse, jacobian);
   points = 0.0;
   for (int d = 0; d < 2; d++)
   {
      points(d, 1) = jacobian(d, 0);
      points(d, 2) = jacobian(d, 1);
   }
   coarse.SetIdentityTransformation(Geometry::TRIANGLE);
   coarse.SetPointMat(points);
}

} // namespace

ArgyrisTriangleFiniteElement::ArgyrisTriangleFiniteElement()
   : FiniteElement(2, Geometry::TRIANGLE, argyris_dof, 5, FunctionSpace::Pk),
     basis(argyris_dof)
{
   deriv_type = GRAD;
   deriv_range_type = VECTOR;
   DenseMatrix M(argyris_dof);
   Vector raw(argyris_dof), dy(argyris_dof);
   for (int v = 0; v < 3; v++)
   {
      for (int d = 0; d < 6; d++)
      {
         IntegrationPoint &ip = Nodes.IntPoint(6*v+d);
         ip.Set2(vertices[v][0], vertices[v][1]);
         Monomials(ip, derivatives[d][0], derivatives[d][1], raw);
         for (int j = 0; j < argyris_dof; j++) { M(j,6*v+d) = raw(j); }
      }
   }
   for (int e = 0; e < 3; e++)
   {
      const real_t *a = vertices[edge_vertices[e][0]];
      const real_t *b = vertices[edge_vertices[e][1]];
      IntegrationPoint &ip = Nodes.IntPoint(18+e);
      ip.Set2((a[0]+b[0])/2, (a[1]+b[1])/2);
      Monomials(ip, 1, 0, raw);
      Monomials(ip, 0, 1, dy);
      for (int j = 0; j < argyris_dof; j++)
      {
         M(j,18+e) = (b[1]-a[1])*raw(j) - (b[0]-a[0])*dy(j);
      }
   }
   DenseMatrixInverse(M).GetInverseMatrix(basis);
}

void ArgyrisTriangleFiniteElement::CalcShape(const IntegrationPoint &ip,
                                             Vector &shape) const
{
   Vector raw(argyris_dof);
   Monomials(ip, 0, 0, raw);
   basis.Mult(raw, shape);
}

void ArgyrisTriangleFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                              DenseMatrix &dshape) const
{
   Vector raw(argyris_dof), values(argyris_dof);
   for (int d = 0; d < 2; d++)
   {
      Monomials(ip, d == 0, d == 1, raw);
      basis.Mult(raw, values);
      dshape.SetCol(d, values);
   }
}

void ArgyrisTriangleFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                               DenseMatrix &hessian) const
{
   Vector raw(argyris_dof), values(argyris_dof);
   for (int d = 0; d < 3; d++)
   {
      Monomials(ip, 2-d, d, raw);
      basis.Mult(raw, values);
      hessian.SetCol(d, values);
   }
}

void ArgyrisTriangleFiniteElement::GetPhysicalDofMatrix(
   ElementTransformation &Trans, DenseMatrix &M) const
{
   MFEM_VERIFY(Trans.GetSpaceDim() == 2 && Trans.Hessian().FNorm2() < 1e-20,
               "Argyris elements require affine two-dimensional transformations");
   M.SetSize(argyris_dof);
   M = 0.0;
   const DenseMatrix &K = Trans.InverseJacobian();
   const DenseMatrix &J = Trans.Jacobian();
   const real_t a = K(0,0), b = K(0,1), c = K(1,0), d = K(1,1);
   for (int v = 0; v < 3; v++)
   {
      const int o = 6*v;
      M(o,o) = 1.0;
      M(o+1,o+1) = a; M(o+1,o+2) = c;
      M(o+2,o+1) = b; M(o+2,o+2) = d;
      M(o+3,o+3) = a*a; M(o+3,o+4) = 2*a*c; M(o+3,o+5) = c*c;
      M(o+4,o+3) = a*b; M(o+4,o+4) = a*d+b*c; M(o+4,o+5) = c*d;
      M(o+5,o+3) = b*b; M(o+5,o+4) = 2*b*d; M(o+5,o+5) = d*d;
   }
   DenseMatrix grad(argyris_dof, 2);
   for (int e = 0; e < 3; e++)
   {
      const real_t *v0 = vertices[edge_vertices[e][0]];
      const real_t *v1 = vertices[edge_vertices[e][1]];
      const real_t tx = v1[0]-v0[0], ty = v1[1]-v0[1];
      const real_t nx = J(1,0)*tx+J(1,1)*ty;
      const real_t ny = -J(0,0)*tx-J(0,1)*ty;
      const real_t ax = a*nx+b*ny, ay = c*nx+d*ny;
      CalcDShape(Nodes.IntPoint(18+e), grad);
      for (int j = 0; j < argyris_dof; j++)
      {
         M(18+e,j) = ax*grad(j,0)+ay*grad(j,1);
      }
   }
}

void ArgyrisTriangleFiniteElement::CalcPhysShape(ElementTransformation &Trans,
                                                 Vector &shape) const
{
   Vector reference(argyris_dof);
   CalcShape(Trans.GetIntPoint(), reference);
   DenseMatrix M(argyris_dof), inverse(argyris_dof);
   GetPhysicalDofMatrix(Trans, M);
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   inverse.MultTranspose(reference, shape);
}

void ArgyrisTriangleFiniteElement::CalcPhysDShape(ElementTransformation &Trans,
                                                  DenseMatrix &dshape) const
{
   DenseMatrix reference(argyris_dof, 2), mapped(argyris_dof, 2);
   CalcDShape(Trans.GetIntPoint(), reference);
   DenseMatrix M(argyris_dof), inverse(argyris_dof);
   GetPhysicalDofMatrix(Trans, M);
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   MultAtB(inverse, reference, mapped);
   Mult(mapped, Trans.InverseJacobian(), dshape);
}

void ArgyrisTriangleFiniteElement::CalcPhysHessian(
   ElementTransformation &Trans, DenseMatrix &hessian) const
{
   MFEM_VERIFY(Trans.Hessian().FNorm2() < 1e-20,
               "Argyris elements currently require affine transformations");
   DenseMatrix reference(argyris_dof, 3), mapped(argyris_dof, 3);
   CalcHessian(Trans.GetIntPoint(), reference);
   DenseMatrix M(argyris_dof), inverse(argyris_dof);
   GetPhysicalDofMatrix(Trans, M);
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   MultAtB(inverse, reference, mapped);

   const DenseMatrix &K = Trans.InverseJacobian();
   const real_t a = K(0,0), b = K(0,1);
   const real_t c = K(1,0), d = K(1,1);
   for (int i = 0; i < argyris_dof; i++)
   {
      const real_t xx = mapped(i,0);
      const real_t xy = mapped(i,1);
      const real_t yy = mapped(i,2);
      hessian(i,0) = a*a*xx + 2.0*a*c*xy + c*c*yy;
      hessian(i,1) = a*b*xx + (a*d + b*c)*xy + c*d*yy;
      hessian(i,2) = b*b*xx + 2.0*b*d*xy + d*d*yy;
   }
}


void ArgyrisTriangleFiniteElement::GetTransferMatrix(const FiniteElement &fe,
                                                     ElementTransformation &T, DenseMatrix &I) const
{
   MFEM_VERIFY(dynamic_cast<const ArgyrisTriangleFiniteElement *>(&fe),
               "Argyris transfer requires a Argyris source element");
   T.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
   MFEM_VERIFY(T.GetSpaceDim() == 2 && T.Hessian().FNorm2() < 1e-20,
               "Argyris transfer requires affine two-dimensional "
               "transformations");
   IsoparametricTransformation coarse;
   coarse.SetIdentityTransformation(Geometry::TRIANGLE);
   InterpolateC1Dofs(fe, 6, true, T, coarse, I);
}

void ArgyrisTriangleFiniteElement::GetPhysicalTransferMatrix(
   const DenseMatrix &, ElementTransformation &child,
   ElementTransformation &fine, DenseMatrix &I) const
{
   // The reference matrix cannot carry the physical DOF bases; interpolate the
   // coarse physical jets at the child nodes instead.
   IsoparametricTransformation coarse;
   SetRelativeCoarseTransformation(child, fine, coarse);
   InterpolateC1Dofs(*this, 6, true, child, coarse, I);
}

} // namespace mfem
