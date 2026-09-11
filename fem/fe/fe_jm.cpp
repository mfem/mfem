// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "fe_jm.hpp"
#include "../eltrans.hpp"
#include "../../linalg/densemat.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace mfem
{

namespace
{

constexpr int jm_dof = 15;
constexpr int raw_dof = 27;

const real_t vertices[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
const int edge_vertices[3][2] = {{0, 1}, {1, 2}, {2, 0}};
const int edge_children[3] = {2, 0, 1};

struct Point
{
   real_t x, y;
};

inline real_t Cross(const Point &a, const Point &b, const Point &p)
{
   return (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x);
}

std::vector<Point> ClipPolygon(const std::vector<Point> &polygon,
                               const Point &a, const Point &b)
{
   std::vector<Point> clipped;
   if (polygon.empty()) { return clipped; }

   Point p = polygon.back();
   real_t fp = Cross(a, b, p);
   for (const Point &q : polygon)
   {
      const real_t fq = Cross(a, b, q);
      const bool p_inside = fp >= -1e-14;
      const bool q_inside = fq >= -1e-14;
      if (p_inside != q_inside)
      {
         const real_t theta = fp/(fp - fq);
         clipped.push_back({p.x + theta*(q.x - p.x),
                            p.y + theta*(q.y - p.y)});
      }
      if (q_inside) { clipped.push_back(q); }
      p = q;
      fp = fq;
   }
   return clipped;
}

void AddAlfeldBreakpoints(const Point &a, const Point &b,
                          std::vector<real_t> &points)
{
   const real_t lambda_a[3] = {1.0 - a.x - a.y, a.x, a.y};
   const real_t lambda_b[3] = {1.0 - b.x - b.y, b.x, b.y};
   for (int i = 0; i < 3; i++)
   {
      for (int j = i + 1; j < 3; j++)
      {
         const real_t f0 = lambda_a[i] - lambda_a[j];
         const real_t f1 = lambda_b[i] - lambda_b[j];
         const real_t denominator = f0 - f1;
         if (std::abs(denominator) > 1e-14)
         {
            const real_t r = f0/denominator;
            if (r > 1e-14 && r < 1.0 - 1e-14) { points.push_back(r); }
         }
      }
   }
   std::sort(points.begin(), points.end());
   points.erase(std::unique(points.begin(), points.end(),
                            [](real_t x, real_t y)
   {
      return std::abs(x - y) < 1e-13;
   }), points.end());
}

inline real_t Monomial(const int m, const real_t x, const real_t y)
{
   return m == 0 ? 1.0 : (m == 1 ? x : y);
}

inline real_t MonomialDerivative(const int m, const int d)
{
   return m == d + 1 ? 1.0 : 0.0;
}

void RawMatrix(const int raw, const real_t x, const real_t y,
               real_t s[2][2])
{
   s[0][0] = s[0][1] = s[1][0] = s[1][1] = 0.0;
   const int comp = (raw % 9) / 3;
   const real_t p = Monomial(raw % 3, x, y);
   if (comp == 0) { s[0][0] = p; }
   else if (comp == 1) { s[0][1] = s[1][0] = p; }
   else { s[1][1] = p; }
}

real_t BoundaryMoment(const int raw, const int edge, const int mode,
                      const bool normal_normal)
{
   if (raw / 9 != edge_children[edge]) { return 0.0; }

   const real_t *a = vertices[edge_vertices[edge][0]];
   const real_t *b = vertices[edge_vertices[edge][1]];
   const real_t dx = b[0] - a[0];
   const real_t dy = b[1] - a[1];
   const real_t length = std::sqrt(dx*dx + dy*dy);
   const real_t t[2] = {dx/length, dy/length};
   const real_t n[2] = {t[1], -t[0]};
   constexpr real_t q[2] = {0.21132486540518711775,
                            0.78867513459481288225
                           };
   real_t value = 0.0;
   for (int k = 0; k < 2; k++)
   {
      const real_t r = q[k];
      const real_t x = a[0] + r*dx;
      const real_t y = a[1] + r*dy;
      real_t s[2][2];
      RawMatrix(raw, x, y, s);
      const real_t sn0 = s[0][0]*n[0] + s[0][1]*n[1];
      const real_t sn1 = s[1][0]*n[0] + s[1][1]*n[1];
      const real_t component = normal_normal ? n[0]*sn0 + n[1]*sn1
                               : t[0]*sn0 + t[1]*sn1;
      value += 0.5*length*(mode == 0 ? 1.0 : 2.0*r - 1.0)*component;
   }
   return value;
}

real_t CellMoment(const int raw, const int component)
{
   const int child = raw / 9;
   const int a = (child + 1) % 3;
   const int b = (child + 2) % 3;
   const real_t x = (vertices[a][0] + vertices[b][0] + 1.0/3.0)/3.0;
   const real_t y = (vertices[a][1] + vertices[b][1] + 1.0/3.0)/3.0;
   real_t s[2][2];
   RawMatrix(raw, x, y, s);
   return s[component == 2 ? 1 : 0][component == 0 ? 0 : 1] / 6.0;
}

real_t InternalConstraint(const int raw, const int vertex, const int endpoint,
                          const int component)
{
   const int child_a = (vertex + 1) % 3;
   const int child_b = (vertex + 2) % 3;
   const int raw_child = raw / 9;
   if (raw_child != child_a && raw_child != child_b) { return 0.0; }

   const real_t cx = 1.0/3.0;
   const real_t cy = 1.0/3.0;
   const real_t x = endpoint == 0 ? cx : vertices[vertex][0];
   const real_t y = endpoint == 0 ? cy : vertices[vertex][1];
   const real_t tx = vertices[vertex][0] - cx;
   const real_t ty = vertices[vertex][1] - cy;
   const real_t length = std::sqrt(tx*tx + ty*ty);
   const real_t n[2] = {ty/length, -tx/length};
   real_t s[2][2];
   RawMatrix(raw, x, y, s);
   const real_t traction = s[component][0]*n[0] + s[component][1]*n[1];
   return (raw_child == child_a ? 1.0 : -1.0)*traction;
}

} // namespace

JohnsonMercierTriangleFiniteElement::JohnsonMercierTriangleFiniteElement()
   : FiniteElement(2, Geometry::TRIANGLE, jm_dof, 1, FunctionSpace::Pk),
     basis(jm_dof, raw_dof)
{
   range_type = MATRIX;
   map_type = DOUBLE_CONTRAVARIANT_PIOLA;
   deriv_type = DIV;
   deriv_range_type = VECTOR;
   deriv_map_type = UNKNOWN_MAP_TYPE;
   vdim = 2;

   DenseMatrix dof_matrix(raw_dof);
   for (int raw = 0; raw < raw_dof; raw++)
   {
      int functional = 0;
      for (int edge = 0; edge < 3; edge++)
      {
         for (int mode = 0; mode < 2; mode++)
         {
            dof_matrix(raw, functional++) =
               BoundaryMoment(raw, edge, mode, true);
            dof_matrix(raw, functional++) =
               BoundaryMoment(raw, edge, mode, false);
         }
      }
      for (int component = 0; component < 3; component++)
      {
         dof_matrix(raw, functional++) = CellMoment(raw, component);
      }
      for (int vertex = 0; vertex < 3; vertex++)
      {
         for (int endpoint = 0; endpoint < 2; endpoint++)
         {
            for (int component = 0; component < 2; component++)
            {
               dof_matrix(raw, functional++) =
                  InternalConstraint(raw, vertex, endpoint, component);
            }
         }
      }
   }

   DenseMatrix inverse(raw_dof);
   DenseMatrixInverse(dof_matrix).GetInverseMatrix(inverse);
   for (int i = 0; i < jm_dof; i++)
   {
      for (int j = 0; j < raw_dof; j++) { basis(i,j) = inverse(i,j); }
   }

   for (int edge = 0; edge < 3; edge++)
   {
      const real_t *a = vertices[edge_vertices[edge][0]];
      const real_t *b = vertices[edge_vertices[edge][1]];
      for (int j = 0; j < 4; j++)
      {
         Nodes.IntPoint(4*edge + j).Set2(0.5*(a[0] + b[0]),
                                         0.5*(a[1] + b[1]));
      }
   }
   for (int j = 12; j < 15; j++)
   {
      Nodes.IntPoint(j).Set2(1.0/3.0, 1.0/3.0);
   }
}

int JohnsonMercierTriangleFiniteElement::GetSubTriangle(
   const IntegrationPoint &ip)
{
   const real_t lambda[3] = {1.0 - ip.x - ip.y, ip.x, ip.y};
   int child = 0;
   if (lambda[1] < lambda[child]) { child = 1; }
   if (lambda[2] < lambda[child]) { child = 2; }
   return child;
}

void JohnsonMercierTriangleFiniteElement::CalcRawShape(
   const IntegrationPoint &ip, Vector &raw)
{
   raw = 0.0;
   const int offset = 9*GetSubTriangle(ip);
   for (int comp = 0; comp < 3; comp++)
   {
      raw(offset + 3*comp) = 1.0;
      raw(offset + 3*comp + 1) = ip.x;
      raw(offset + 3*comp + 2) = ip.y;
   }
}

void JohnsonMercierTriangleFiniteElement::CalcRawDivShape(
   const IntegrationPoint &ip, DenseMatrix &raw_div)
{
   raw_div = 0.0;
   const int offset = 9*GetSubTriangle(ip);
   for (int m = 0; m < 3; m++)
   {
      raw_div(offset + m, 0) = MonomialDerivative(m, 0);
      raw_div(offset + 3 + m, 0) = MonomialDerivative(m, 1);
      raw_div(offset + 3 + m, 1) = MonomialDerivative(m, 0);
      raw_div(offset + 6 + m, 1) = MonomialDerivative(m, 1);
   }
}

void JohnsonMercierTriangleFiniteElement::CalcMShape(
   const IntegrationPoint &ip, DenseTensor &shape) const
{
   MFEM_ASSERT(shape.SizeI() == 2 && shape.SizeJ() == 2 &&
               shape.SizeK() == jm_dof, "invalid matrix shape size");
   Vector raw(raw_dof);
   CalcRawShape(ip, raw);
   for (int i = 0; i < jm_dof; i++)
   {
      shape(0,0,i) = 0.0;
      shape(0,1,i) = shape(1,0,i) = shape(1,1,i) = 0.0;
      for (int child = 0; child < 3; child++)
      {
         const int o = 9*child;
         for (int m = 0; m < 3; m++)
         {
            shape(0,0,i) += basis(i,o+m)*raw(o+m);
            shape(0,1,i) += basis(i,o+3+m)*raw(o+3+m);
            shape(1,1,i) += basis(i,o+6+m)*raw(o+6+m);
         }
      }
      shape(1,0,i) = shape(0,1,i);
   }
}

void JohnsonMercierTriangleFiniteElement::CalcDivShape(
   const IntegrationPoint &ip, DenseMatrix &divshape) const
{
   DenseMatrix raw_div(raw_dof, 2);
   CalcRawDivShape(ip, raw_div);
   Mult(basis, raw_div, divshape);
}

void JohnsonMercierTriangleFiniteElement::GetFacetTransform(
   const DenseMatrix &J, DenseMatrix &A) const
{
   A.SetSize(jm_dof);
   A = 0.0;
   for (int i = 0; i < jm_dof; i++) { A(i,i) = 1.0; }

   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   for (int edge = 0; edge < 3; edge++)
   {
      const real_t *a = vertices[edge_vertices[edge][0]];
      const real_t *b = vertices[edge_vertices[edge][1]];
      const real_t dx = b[0] - a[0];
      const real_t dy = b[1] - a[1];
      const real_t ref_length = std::sqrt(dx*dx + dy*dy);
      const real_t t[2] = {dx/ref_length, dy/ref_length};
      const real_t n[2] = {t[1], -t[0]};
      const real_t jt[2] = {J(0,0)*t[0] + J(0,1)*t[1],
                            J(1,0)*t[0] + J(1,1)*t[1]
                           };
      const real_t jn[2] = {J(0,0)*n[0] + J(0,1)*n[1],
                            J(1,0)*n[0] + J(1,1)*n[1]
                           };
      const real_t length = std::sqrt(jt[0]*jt[0] + jt[1]*jt[1]);
      const real_t alpha = jn[0]*jt[0] + jn[1]*jt[1];
      // The double Piola image preserves the traction space but mixes its
      // normal-normal and normal-tangential moments. This change of basis
      // makes the physical edge moments canonical again.
      for (int mode = 0; mode < 2; mode++)
      {
         const int nn = 4*edge + 2*mode;
         const int nt = nn + 1;
         A(nn,nn) = length;
         A(nn,nt) = -alpha/length;
         A(nt,nt) = det/length;
      }
   }
}

void JohnsonMercierTriangleFiniteElement::CalcMShape(
   ElementTransformation &Trans, DenseTensor &shape) const
{
   MFEM_ASSERT(Trans.GetSpaceDim() == 2, "Johnson-Mercier is a 2D element");
   MFEM_ASSERT(shape.SizeI() == 2 && shape.SizeJ() == 2 &&
               shape.SizeK() == jm_dof, "invalid matrix shape size");
   DenseTensor ref_shape(2, 2, jm_dof);
   CalcMShape(Trans.GetIntPoint(), ref_shape);
   const DenseMatrix &J = Trans.Jacobian();
   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   const real_t scale = 1.0/(det*det);
   DenseTensor mapped(2, 2, jm_dof);
   for (int k = 0; k < jm_dof; k++)
   {
      for (int i = 0; i < 2; i++)
      {
         for (int j = 0; j < 2; j++)
         {
            real_t value = 0.0;
            for (int a = 0; a < 2; a++)
            {
               for (int b = 0; b < 2; b++)
               {
                  value += J(i,a)*ref_shape(a,b,k)*J(j,b);
               }
            }
            mapped(i,j,k) = scale*value;
         }
      }
   }

   DenseMatrix A;
   GetFacetTransform(J, A);
   for (int k = 0; k < jm_dof; k++)
   {
      for (int i = 0; i < 2; i++)
      {
         for (int j = 0; j < 2; j++)
         {
            shape(i,j,k) = 0.0;
            for (int l = 0; l < jm_dof; l++)
            {
               shape(i,j,k) += A(k,l)*mapped(i,j,l);
            }
         }
      }
   }
}

void JohnsonMercierTriangleFiniteElement::CalcPhysDivShape(
   ElementTransformation &Trans, DenseMatrix &divshape) const
{
   MFEM_ASSERT(divshape.Height() == jm_dof && divshape.Width() == 2,
               "invalid matrix divergence shape size");
   DenseMatrix ref_div(jm_dof, 2), mapped(jm_dof, 2), A;
   CalcDivShape(Trans.GetIntPoint(), ref_div);
   const DenseMatrix &J = Trans.Jacobian();
   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   for (int k = 0; k < jm_dof; k++)
   {
      for (int i = 0; i < 2; i++)
      {
         mapped(k,i) = (J(i,0)*ref_div(k,0) + J(i,1)*ref_div(k,1)) /
                       (det*det);
      }
   }
   GetFacetTransform(J, A);
   Mult(A, mapped, divshape);
}

void JohnsonMercierTriangleFiniteElement::GetTransferMatrix(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   MFEM_VERIFY(fe.GetDim() == 2 && fe.GetGeomType() == Geometry::TRIANGLE &&
               fe.GetRangeType() == MATRIX &&
               fe.GetMapType() == DOUBLE_CONTRAVARIANT_PIOLA,
               "incompatible coarse finite element");
   MFEM_VERIFY(Trans.GetSpaceDim() == 2,
               "Johnson-Mercier transfer requires a 2D transformation");

   const int coarse_dof = fe.GetDof();
   I.SetSize(jm_dof, coarse_dof);
   I = 0.0;

   // Refinement transformations are affine. Store their Jacobian at the
   // center since Transform() calls below do not set the integration point.
   Trans.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
   DenseMatrix J(2), inverse_J(2);
   J = Trans.Jacobian();
   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   const real_t abs_det = std::abs(det);
   MFEM_VERIFY(abs_det > 0.0, "singular refinement transformation");
   CalcInverse(J, inverse_J);

   Point fine_vertices[3];
   Vector point(2);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0], vertices[v][1]);
      Trans.Transform(ip, point);
      fine_vertices[v] = {point(0), point(1)};
   }

   DenseTensor coarse_shape(2, 2, coarse_dof);

   // Apply the twelve physical edge moments. Split each integral where the
   // fine edge crosses an interface of the coarse Alfeld split, making the
   // two-point Gauss rule exact on every piece.
   constexpr real_t gauss[2] = {0.21132486540518711775,
                                0.78867513459481288225
                               };
   for (int edge = 0; edge < 3; edge++)
   {
      const Point &a = fine_vertices[edge_vertices[edge][0]];
      const Point &b = fine_vertices[edge_vertices[edge][1]];
      const real_t dx = b.x - a.x;
      const real_t dy = b.y - a.y;
      const real_t length = std::sqrt(dx*dx + dy*dy);
      const real_t t[2] = {dx/length, dy/length};
      const real_t n[2] = {t[1], -t[0]};
      std::vector<real_t> breaks = {0.0, 1.0};
      AddAlfeldBreakpoints(a, b, breaks);

      for (size_t interval = 0; interval + 1 < breaks.size(); interval++)
      {
         const real_t r0 = breaks[interval];
         const real_t r1 = breaks[interval + 1];
         for (int q = 0; q < 2; q++)
         {
            const real_t r = r0 + (r1 - r0)*gauss[q];
            IntegrationPoint ip;
            ip.Set2(a.x + r*dx, a.y + r*dy);
            fe.CalcMShape(ip, coarse_shape);
            const real_t weight = 0.5*length*(r1 - r0);
            for (int k = 0; k < coarse_dof; k++)
            {
               const real_t sn0 = coarse_shape(0,0,k)*n[0] +
                                  coarse_shape(0,1,k)*n[1];
               const real_t sn1 = coarse_shape(1,0,k)*n[0] +
                                  coarse_shape(1,1,k)*n[1];
               const real_t nn = n[0]*sn0 + n[1]*sn1;
               const real_t nt = t[0]*sn0 + t[1]*sn1;
               I(4*edge, k) += weight*nn;
               I(4*edge + 1, k) += weight*nt;
               I(4*edge + 2, k) += weight*(2.0*r - 1.0)*nn;
               I(4*edge + 3, k) += weight*(2.0*r - 1.0)*nt;
            }
         }
      }
   }

   // The three interior functionals are the reference cell moments of the
   // double-Piola pullback. Integrate them over the exact overlay between the
   // fine triangle and each coarse Alfeld subtriangle. On every overlay cell
   // the pulled-back coarse shape is affine, so one centroid value is exact.
   const Point center = {1.0/3.0, 1.0/3.0};
   const Point coarse_children[3][3] =
   {
      {{1.0, 0.0}, {0.0, 1.0}, center},
      {{0.0, 1.0}, {0.0, 0.0}, center},
      {{0.0, 0.0}, {1.0, 0.0}, center}
   };
   const int component_row[3] = {0, 0, 1};
   const int component_col[3] = {0, 1, 1};
   for (int child = 0; child < 3; child++)
   {
      std::vector<Point> polygon(fine_vertices, fine_vertices + 3);
      for (int edge = 0; edge < 3; edge++)
      {
         polygon = ClipPolygon(polygon, coarse_children[child][edge],
                               coarse_children[child][(edge + 1) % 3]);
      }
      if (polygon.size() < 3) { continue; }

      for (size_t q = 1; q + 1 < polygon.size(); q++)
      {
         const Point &a = polygon[0];
         const Point &b = polygon[q];
         const Point &c = polygon[q + 1];
         const real_t area = 0.5*std::abs(Cross(a, b, c));
         if (area < 1e-15) { continue; }
         IntegrationPoint ip;
         ip.Set2((a.x + b.x + c.x)/3.0, (a.y + b.y + c.y)/3.0);
         fe.CalcMShape(ip, coarse_shape);
         const real_t reference_weight = area/abs_det;

         for (int component = 0; component < 3; component++)
         {
            const int row = component_row[component];
            const int col = component_col[component];
            for (int k = 0; k < coarse_dof; k++)
            {
               real_t pullback = 0.0;
               for (int i = 0; i < 2; i++)
               {
                  for (int j = 0; j < 2; j++)
                  {
                     pullback += inverse_J(row,i)*coarse_shape(i,j,k)*
                                 inverse_J(col,j);
                  }
               }
               I(12 + component, k) +=
                  reference_weight*det*det*pullback;
            }
         }
      }
   }

   for (int i = 0; i < I.Height(); i++)
   {
      for (int j = 0; j < I.Width(); j++)
      {
         if (std::abs(I(i,j)) < 1e-12) { I(i,j) = 0.0; }
      }
   }
}

void JohnsonMercierTriangleFiniteElement::GetFaceDofs(
   int face, int **dofs, int *ndofs) const
{
   static int face_dofs[3][4] = {{0, 1, 2, 3}, {4, 5, 6, 7},
      {8, 9, 10, 11}
   };
   MFEM_ASSERT(face >= 0 && face < 3, "invalid face index");
   *dofs = face_dofs[face];
   *ndofs = 4;
}

void JohnsonMercierTriangleFiniteElement::GetPhysicalTransferMatrix(
   const DenseMatrix &reference_transfer, ElementTransformation &child,
   ElementTransformation &fine, DenseMatrix &I) const
{
   const IntegrationPoint &center = Geometries.GetCenter(Geometry::TRIANGLE);
   child.SetIntPoint(&center);
   fine.SetIntPoint(&center);
   DenseMatrix inverse_child(2), coarse_J(2);
   CalcInverse(child.Jacobian(), inverse_child);
   Mult(fine.Jacobian(), inverse_child, coarse_J);

   DenseMatrix Ac, Af, Ar, inverse_Af(jm_dof);
   GetFacetTransform(coarse_J, Ac);
   GetFacetTransform(fine.Jacobian(), Af);
   GetFacetTransform(child.Jacobian(), Ar);
   DenseMatrixInverse(Af).GetInverseMatrix(inverse_Af);

   // A^T maps physical moment coefficients to double-Piola coefficients.
   // Piola maps compose, but the geometry-dependent facet corrections do not:
   // I_phys = A_f^{-T} A_child^T I_ref A_coarse^T.
   DenseMatrix right(jm_dof), middle(jm_dof);
   MultABt(reference_transfer, Ac, right);
   MultAtB(Ar, right, middle);
   I.SetSize(jm_dof);
   MultAtB(inverse_Af, middle, I);
}

} // namespace mfem
