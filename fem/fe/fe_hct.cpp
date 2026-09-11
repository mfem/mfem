// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "fe_hct.hpp"
#include "../eltrans.hpp"

namespace mfem
{

namespace
{

constexpr int hct_dof = 12;
constexpr int raw_dof = 30;
constexpr int monomial_powers[10][2] =
{
   {0,0}, {1,0}, {0,1}, {2,0}, {1,1},
   {0,2}, {3,0}, {2,1}, {1,2}, {0,3}
};
constexpr int edge_vertices[3][2] = {{0,1}, {1,2}, {2,0}};
constexpr real_t vertices[3][2] = {{0.0,0.0}, {1.0,0.0}, {0.0,1.0}};
constexpr real_t edge_tangents[3][2] = {{1.0,0.0}, {-1.0,1.0}, {0.0,-1.0}};

real_t IntPower(real_t x, int p)
{
   return p == 0 ? 1.0 : (p == 1 ? x : (p == 2 ? x*x : x*x*x));
}

int DerivativeFactor(int p, int d)
{
   if (d == 0) { return 1; }
   if (p < d) { return 0; }
   return d == 1 ? p : p*(p - 1);
}

int Binomial(int n, int k)
{
   if (k < 0 || k > n) { return 0; }
   if (n <= 1 || k == 0 || k == n) { return 1; }
   return n == 2 ? 2 : (k == 1 || k == 2 ? 3 : 1);
}

real_t MonomialDerivative(int monomial, int dx, int dy,
                          real_t x, real_t y)
{
   const int px = monomial_powers[monomial][0];
   const int py = monomial_powers[monomial][1];
   const int factor = DerivativeFactor(px, dx)*DerivativeFactor(py, dy);
   if (!factor) { return 0.0; }
   return factor*IntPower(x, px - dx)*IntPower(y, py - dy);
}

real_t PowerLineCoefficient(int p, real_t a, real_t b, int k)
{
   return Binomial(p,k)*IntPower(a,p-k)*IntPower(b,k);
}

real_t MonomialLineDerivativeCoefficient(int monomial, int dx, int dy,
                                         real_t x0, real_t xt,
                                         real_t y0, real_t yt, int k)
{
   const int px = monomial_powers[monomial][0] - dx;
   const int py = monomial_powers[monomial][1] - dy;
   const int factor =
      DerivativeFactor(monomial_powers[monomial][0], dx)*
      DerivativeFactor(monomial_powers[monomial][1], dy);
   if (!factor || k > px + py) { return 0.0; }
   real_t value = 0.0;
   for (int i = 0; i <= px; i++)
   {
      const int j = k - i;
      if (0 <= j && j <= py)
      {
         value += PowerLineCoefficient(px, x0, xt, i)*
                  PowerLineCoefficient(py, y0, yt, j);
      }
   }
   return factor*value;
}

} // namespace

HCTTriangleFiniteElement::HCTTriangleFiniteElement()
   : FiniteElement(2, Geometry::TRIANGLE, hct_dof, 3, FunctionSpace::Pk),
     basis(hct_dof, raw_dof)
{
   deriv_type = GRAD;
   deriv_range_type = VECTOR;

   // The first twelve columns impose the HCT degrees of freedom. The final
   // eighteen impose C1 continuity across the three internal split edges.
   DenseMatrix dof_matrix(raw_dof);
   dof_matrix = 0.0;
   int functional = 0;

   const int vertex_child[3] = {0, 0, 1};
   for (int vertex = 0; vertex < 3; vertex++)
   {
      for (int derivative = 0; derivative < 3; derivative++)
      {
         const int dx = derivative == 1;
         const int dy = derivative == 2;
         for (int monomial = 0; monomial < 10; monomial++)
         {
            const int raw = 10*vertex_child[vertex] + monomial;
            dof_matrix(raw,functional) =
               MonomialDerivative(monomial, dx, dy,
                                  vertices[vertex][0], vertices[vertex][1]);
         }
         functional++;
      }
   }

   for (int edge = 0; edge < 3; edge++)
   {
      const real_t tx = edge_tangents[edge][0];
      const real_t ty = edge_tangents[edge][1];
      const real_t nx = ty;
      const real_t ny = -tx;
      const int v0 = edge_vertices[edge][0];
      const int v1 = edge_vertices[edge][1];
      const real_t x = 0.5*(vertices[v0][0] + vertices[v1][0]);
      const real_t y = 0.5*(vertices[v0][1] + vertices[v1][1]);
      for (int monomial = 0; monomial < 10; monomial++)
      {
         dof_matrix(10*edge + monomial,functional) =
            nx*MonomialDerivative(monomial, 1, 0, x, y) +
            ny*MonomialDerivative(monomial, 0, 1, x, y);
      }
      functional++;
   }

   struct Constraint { int edge, derivative, coefficient; };
   const Constraint constraints[18] =
   {
      {0,0,0}, {0,0,1}, {0,0,2}, {0,0,3},
      {0,1,0}, {0,1,1}, {0,1,2},
      {1,0,0}, {1,0,1}, {1,0,2}, {1,0,3},
      {1,1,0}, {1,1,1}, {1,1,2},
      {2,0,0}, {2,0,1}, {2,1,0}, {2,1,1}
   };
   const int child_a[3] = {0, 1, 2};
   const int child_b[3] = {1, 2, 0};
   const int shared_vertex[3] = {1, 2, 0};
   for (const Constraint &constraint : constraints)
   {
      const int edge = constraint.edge;
      const int vertex = shared_vertex[edge];
      const real_t x0 = vertices[vertex][0];
      const real_t y0 = vertices[vertex][1];
      const real_t xt = 1.0/3.0 - x0;
      const real_t yt = 1.0/3.0 - y0;
      for (int monomial = 0; monomial < 10; monomial++)
      {
         const real_t value = MonomialLineDerivativeCoefficient(
                                 monomial, constraint.derivative, 0, x0, xt, y0, yt,
                                 constraint.coefficient);
         dof_matrix(10*child_a[edge] + monomial,functional) = value;
         dof_matrix(10*child_b[edge] + monomial,functional) = -value;
      }
      functional++;
   }
   MFEM_ASSERT(functional == raw_dof, "incorrect HCT constraint count");

   DenseMatrix inverse(raw_dof);
   DenseMatrixInverse(dof_matrix).GetInverseMatrix(inverse);
   for (int i = 0; i < hct_dof; i++)
   {
      for (int j = 0; j < raw_dof; j++) { basis(i,j) = inverse(i,j); }
   }

   for (int vertex = 0; vertex < 3; vertex++)
   {
      for (int j = 0; j < 3; j++)
      {
         Nodes.IntPoint(3*vertex + j).Set2(vertices[vertex][0],
                                           vertices[vertex][1]);
      }
   }
   for (int edge = 0; edge < 3; edge++)
   {
      const int v0 = edge_vertices[edge][0];
      const int v1 = edge_vertices[edge][1];
      Nodes.IntPoint(9 + edge).Set2(
         0.5*(vertices[v0][0] + vertices[v1][0]),
         0.5*(vertices[v0][1] + vertices[v1][1]));
   }
}

int HCTTriangleFiniteElement::GetSubTriangle(const IntegrationPoint &ip)
{
   const real_t lambda[3] = {1.0 - ip.x - ip.y, ip.x, ip.y};
   if (lambda[2] <= lambda[0] && lambda[2] <= lambda[1]) { return 0; }
   if (lambda[0] <= lambda[1] && lambda[0] <= lambda[2]) { return 1; }
   return 2;
}

void HCTTriangleFiniteElement::CalcRawShape(const IntegrationPoint &ip,
                                            Vector &raw)
{
   raw = 0.0;
   const int offset = 10*GetSubTriangle(ip);
   for (int monomial = 0; monomial < 10; monomial++)
   {
      raw(offset + monomial) = MonomialDerivative(monomial, 0, 0, ip.x, ip.y);
   }
}

void HCTTriangleFiniteElement::CalcRawDShape(const IntegrationPoint &ip,
                                             DenseMatrix &raw)
{
   raw = 0.0;
   const int offset = 10*GetSubTriangle(ip);
   for (int monomial = 0; monomial < 10; monomial++)
   {
      raw(offset + monomial,0) =
         MonomialDerivative(monomial, 1, 0, ip.x, ip.y);
      raw(offset + monomial,1) =
         MonomialDerivative(monomial, 0, 1, ip.x, ip.y);
   }
}

void HCTTriangleFiniteElement::CalcRawHessian(const IntegrationPoint &ip,
                                              DenseMatrix &raw)
{
   raw = 0.0;
   const int offset = 10*GetSubTriangle(ip);
   for (int monomial = 0; monomial < 10; monomial++)
   {
      raw(offset + monomial,0) =
         MonomialDerivative(monomial, 2, 0, ip.x, ip.y);
      raw(offset + monomial,1) =
         MonomialDerivative(monomial, 1, 1, ip.x, ip.y);
      raw(offset + monomial,2) =
         MonomialDerivative(monomial, 0, 2, ip.x, ip.y);
   }
}

void HCTTriangleFiniteElement::CalcShape(const IntegrationPoint &ip,
                                         Vector &shape) const
{
   Vector raw(raw_dof);
   CalcRawShape(ip, raw);
   basis.Mult(raw, shape);
}

void HCTTriangleFiniteElement::CalcDShape(const IntegrationPoint &ip,
                                          DenseMatrix &dshape) const
{
   DenseMatrix raw(raw_dof, 2);
   CalcRawDShape(ip, raw);
   Mult(basis, raw, dshape);
}

void HCTTriangleFiniteElement::CalcHessian(const IntegrationPoint &ip,
                                           DenseMatrix &hessian) const
{
   DenseMatrix raw(raw_dof, 3);
   CalcRawHessian(ip, raw);
   Mult(basis, raw, hessian);
}

void HCTTriangleFiniteElement::GetPhysicalDofMatrix(
   ElementTransformation &Trans, DenseMatrix &M) const
{
   MFEM_VERIFY(Trans.GetSpaceDim() == 2,
               "HCT elements require a two-dimensional physical mesh");
   M.SetSize(hct_dof);
   M = 0.0;
   for (int i = 0; i < hct_dof; i++) { M(i,i) = 1.0; }

   const DenseMatrix &invJ = Trans.InverseJacobian();
   for (int vertex = 0; vertex < 3; vertex++)
   {
      const int offset = 3*vertex + 1;
      for (int i = 0; i < 2; i++)
      {
         for (int j = 0; j < 2; j++) { M(offset+i,offset+j) = 0.0; }
      }
      M(offset,offset)       = invJ(0,0);
      M(offset,offset + 1)   = invJ(1,0);
      M(offset + 1,offset)   = invJ(0,1);
      M(offset + 1,offset+1) = invJ(1,1);
   }

   const DenseMatrix &J = Trans.Jacobian();
   for (int edge = 0; edge < 3; edge++)
   {
      const int row = 9 + edge;
      for (int j = 0; j < hct_dof; j++) { M(row,j) = 0.0; }
      const real_t tx = edge_tangents[edge][0];
      const real_t ty = edge_tangents[edge][1];
      const real_t tpx = J(0,0)*tx + J(0,1)*ty;
      const real_t tpy = J(1,0)*tx + J(1,1)*ty;
      const real_t mpx = tpy;
      const real_t mpy = -tpx;
      const real_t ax = invJ(0,0)*mpx + invJ(0,1)*mpy;
      const real_t ay = invJ(1,0)*mpx + invJ(1,1)*mpy;
      const real_t norm2 = tx*tx + ty*ty;
      const real_t alpha = (ax*ty - ay*tx)/norm2;
      const real_t beta = (ax*tx + ay*ty)/norm2;
      M(row,row) = alpha;

      const int v0 = edge_vertices[edge][0];
      const int v1 = edge_vertices[edge][1];
      M(row,3*v0) += -1.5*beta;
      M(row,3*v1) +=  1.5*beta;
      M(row,3*v0 + 1) += -0.25*beta*tx;
      M(row,3*v0 + 2) += -0.25*beta*ty;
      M(row,3*v1 + 1) += -0.25*beta*tx;
      M(row,3*v1 + 2) += -0.25*beta*ty;
   }
}

void HCTTriangleFiniteElement::CalcPhysShape(ElementTransformation &Trans,
                                             Vector &shape) const
{
   Vector reference(hct_dof);
   CalcShape(Trans.GetIntPoint(), reference);
   DenseMatrix M(hct_dof), inverse(hct_dof);
   GetPhysicalDofMatrix(Trans, M);
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   inverse.MultTranspose(reference, shape);
}

void HCTTriangleFiniteElement::CalcPhysDShape(ElementTransformation &Trans,
                                              DenseMatrix &dshape) const
{
   DenseMatrix reference(hct_dof, 2), mapped(hct_dof, 2);
   CalcDShape(Trans.GetIntPoint(), reference);
   DenseMatrix M(hct_dof), inverse(hct_dof);
   GetPhysicalDofMatrix(Trans, M);
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   MultAtB(inverse, reference, mapped);
   Mult(mapped, Trans.InverseJacobian(), dshape);
}

void HCTTriangleFiniteElement::CalcPhysHessian(
   ElementTransformation &Trans, DenseMatrix &hessian) const
{
   MFEM_VERIFY(Trans.Hessian().FNorm2() < 1e-20,
               "HCT elements currently require affine transformations");
   DenseMatrix reference(hct_dof, 3), mapped(hct_dof, 3);
   CalcHessian(Trans.GetIntPoint(), reference);
   DenseMatrix M(hct_dof), inverse(hct_dof);
   GetPhysicalDofMatrix(Trans, M);
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   MultAtB(inverse, reference, mapped);

   const DenseMatrix &K = Trans.InverseJacobian();
   const real_t a = K(0,0), b = K(0,1);
   const real_t c = K(1,0), d = K(1,1);
   for (int i = 0; i < hct_dof; i++)
   {
      const real_t xx = mapped(i,0);
      const real_t xy = mapped(i,1);
      const real_t yy = mapped(i,2);
      hessian(i,0) = a*a*xx + 2.0*a*c*xy + c*c*yy;
      hessian(i,1) = a*b*xx + (a*d + b*c)*xy + c*d*yy;
      hessian(i,2) = b*b*xx + 2.0*b*d*xy + d*d*yy;
   }
}

} // namespace mfem
