// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "fe_aw.hpp"
#include "../eltrans.hpp"
#include <cmath>
#include <functional>

namespace mfem
{
namespace
{
constexpr int aw_dof = 24;
constexpr int raw_dof = 30;
constexpr real_t vertices[3][2] = {{0,0}, {1,0}, {0,1}};
constexpr int edge_vertices[3][2] = {{0,1}, {1,2}, {2,0}};
constexpr int powers[10][2] =
{{0,0}, {1,0}, {0,1}, {2,0}, {1,1}, {0,2}, {3,0}, {2,1}, {1,2}, {0,3}};
constexpr int rows[3] = {0,0,1};
constexpr int cols[3] = {0,1,1};

real_t Monomial(int m, const IntegrationPoint &ip, int derivative = -1)
{
   int px = powers[m][0], py = powers[m][1];
   real_t value = 1.0;
   if (derivative == 0) { value = px--; }
   if (derivative == 1) { value = py--; }
   for (int i = 0; i < px; i++) { value *= ip.x; }
   for (int i = 0; i < py; i++) { value *= ip.y; }
   return value;
}

void RawShape(const IntegrationPoint &ip, DenseTensor &shape)
{
   shape = 0.0;
   for (int c = 0; c < 3; c++)
   {
      for (int m = 0; m < 10; m++)
      {
         shape(rows[c],cols[c],10*c+m) =
            shape(cols[c],rows[c],10*c+m) = Monomial(m, ip);
      }
   }
}

// Apply canonical DOFs to physical tensor values evaluated at reference
// points of T. Keeping this common to construction, projection, and transfer
// makes their moment and component conventions identical.
void ApplyDofs(ElementTransformation &T, int n, int order,
               const std::function<void(const IntegrationPoint &, DenseTensor &)> &eval,
               DenseMatrix &I)
{
   T.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
   MFEM_VERIFY(T.GetSpaceDim() == 2 && T.Hessian().FNorm2() < 1e-20,
               "Arnold-Winther elements require affine 2D transformations");
   I.SetSize(aw_dof,n);
   I = 0.0;
   DenseTensor shape(2,2,n);
   for (int v = 0; v < 3; v++)
   {
      IntegrationPoint ip;
      ip.Set2(vertices[v][0],vertices[v][1]);
      eval(ip,shape);
      for (int c = 0; c < 3; c++)
      {
         for (int j = 0; j < n; j++) { I(3*v+c,j) = shape(rows[c],cols[c],j); }
      }
   }
   const IntegrationRule &edge_rule = IntRules.Get(Geometry::SEGMENT,order+1);
   Vector pa(2),pb(2);
   for (int e = 0; e < 3; e++)
   {
      const real_t *a = vertices[edge_vertices[e][0]];
      const real_t *b = vertices[edge_vertices[e][1]];
      IntegrationPoint ip;
      ip.Set2(a[0],a[1]); T.Transform(ip,pa);
      ip.Set2(b[0],b[1]); T.Transform(ip,pb);
      const real_t length = std::hypot(pb(0)-pa(0),pb(1)-pa(1));
      const real_t tx = (pb(0)-pa(0))/length, ty = (pb(1)-pa(1))/length;
      const real_t nx = ty, ny = -tx;
      for (int q = 0; q < edge_rule.GetNPoints(); q++)
      {
         const IntegrationPoint &qp = edge_rule.IntPoint(q);
         ip.Set2(a[0]+qp.x*(b[0]-a[0]),a[1]+qp.x*(b[1]-a[1]));
         eval(ip,shape);
         for (int j = 0; j < n; j++)
         {
            const real_t snx = shape(0,0,j)*nx+shape(0,1,j)*ny;
            const real_t sny = shape(0,1,j)*nx+shape(1,1,j)*ny;
            const real_t nn = nx*snx+ny*sny, nt = tx*snx+ty*sny;
            for (int mode = 0; mode < 2; mode++)
            {
               const real_t w = qp.weight*length*(mode ? 2*qp.x-1 : 1);
               I(9+4*e+2*mode,j) += w*nn;
               I(10+4*e+2*mode,j) += w*nt;
            }
         }
      }
   }
   T.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
   const DenseMatrix K(T.InverseJacobian());
   const real_t det2 = T.Jacobian().Det()*T.Jacobian().Det();
   const IntegrationRule &rule = IntRules.Get(Geometry::TRIANGLE,order);
   for (int q = 0; q < rule.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = rule.IntPoint(q);
      eval(ip,shape);
      for (int c = 0; c < 3; c++)
      {
         for (int j = 0; j < n; j++)
         {
            real_t value = 0.0;
            for (int a = 0; a < 2; a++)
            {
               for (int b = 0; b < 2; b++)
               { value += K(rows[c],a)*shape(a,b,j)*K(cols[c],b); }
            }
            I(21+c,j) += ip.weight*det2*value;
         }
      }
   }
   T.SetIntPoint(&Geometries.GetCenter(Geometry::TRIANGLE));
}
} // namespace

ArnoldWintherTriangleFiniteElement::ArnoldWintherTriangleFiniteElement()
   : FiniteElement(2, Geometry::TRIANGLE, aw_dof, 3, FunctionSpace::Pk),
     basis(aw_dof,raw_dof)
{
   range_type = MATRIX;
   map_type = DOUBLE_CONTRAVARIANT_PIOLA;
   deriv_type = DIV;
   deriv_range_type = VECTOR;
   deriv_map_type = UNKNOWN_MAP_TYPE;
   vdim = 2;
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TRIANGLE);
   DenseMatrix moments, M(raw_dof), inverse(raw_dof);
   ApplyDofs(T,raw_dof,3,RawShape,moments);
   M = 0.0;
   for (int r = 0; r < raw_dof; r++)
   {
      for (int j = 0; j < aw_dof; j++) { M(r,j) = moments(j,r); }
      const int c = r/10, m = r%10;
      // Remove all quadratic coefficients of both components of divergence.
      for (int d = 0; d < 2; d++)
      {
         const int derivative = c == 1 ? 1-d : d;
         if ((c == 0 && d == 1) || (c == 2 && d == 0)) { continue; }
         for (int p = 3; p < 6; p++)
         {
            if (powers[m][0]-(derivative == 0) == powers[p][0] &&
                powers[m][1]-(derivative == 1) == powers[p][1])
            { M(r,24+3*d+p-3) = powers[m][derivative]; }
         }
      }
   }
   DenseMatrixInverse(M).GetInverseMatrix(inverse);
   for (int i = 0; i < aw_dof; i++)
   {
      for (int j = 0; j < raw_dof; j++) { basis(i,j) = inverse(i,j); }
   }
   for (int v = 0; v < 3; v++)
   {
      for (int c = 0; c < 3; c++)
      { Nodes.IntPoint(3*v+c).Set2(vertices[v][0],vertices[v][1]); }
   }
   for (int e = 0; e < 3; e++)
   {
      const real_t *a = vertices[edge_vertices[e][0]];
      const real_t *b = vertices[edge_vertices[e][1]];
      for (int j = 0; j < 4; j++)
      { Nodes.IntPoint(9+4*e+j).Set2((a[0]+b[0])/2,(a[1]+b[1])/2); }
   }
   for (int j = 21; j < 24; j++) { Nodes.IntPoint(j).Set2(1.0/3,1.0/3); }
}

void ArnoldWintherTriangleFiniteElement::CalcMShape(
   const IntegrationPoint &ip, DenseTensor &shape) const
{
   shape = 0.0;
   for (int i = 0; i < aw_dof; i++)
   {
      for (int c = 0; c < 3; c++)
      {
         real_t value = 0.0;
         for (int m = 0; m < 10; m++) { value += basis(i,10*c+m)*Monomial(m,ip); }
         shape(rows[c],cols[c],i) = shape(cols[c],rows[c],i) = value;
      }
   }
}

void ArnoldWintherTriangleFiniteElement::CalcDivShape(
   const IntegrationPoint &ip, DenseMatrix &divshape) const
{
   divshape = 0.0;
   for (int i = 0; i < aw_dof; i++)
   {
      for (int m = 0; m < 10; m++)
      {
         const real_t dx = Monomial(m,ip,0), dy = Monomial(m,ip,1);
         divshape(i,0) += basis(i,m)*dx + basis(i,10+m)*dy;
         divshape(i,1) += basis(i,10+m)*dx + basis(i,20+m)*dy;
      }
   }
}

void ArnoldWintherTriangleFiniteElement::GetBasisTransform(
   const DenseMatrix &J, DenseMatrix &A) const
{
   A.SetSize(aw_dof);
   A = 0.0;
   for (int i = 0; i < aw_dof; i++) { A(i,i) = 1.0; }

   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   DenseMatrix K(2);
   CalcInverse(J, K);
   // Vertex values are physical Cartesian components. The rows of A are
   // columns of the inverse double-Piola component map.
   for (int v = 0; v < 3; v++)
   {
      for (int r = 0; r < 3; r++)
      {
         for (int c = 0; c < 3; c++)
         {
            real_t value = K(rows[c],rows[r])*K(cols[c],cols[r]);
            if (r == 1) { value += K(rows[c],cols[r])*K(cols[c],rows[r]); }
            A(3*v+r,3*v+c) = det*det*value;
         }
      }
   }
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
         const int nn = 9 + 4*edge + 2*mode;
         const int nt = nn + 1;
         A(nn,nn) = length;
         A(nn,nt) = -alpha/length;
         A(nt,nt) = det/length;
      }
   }
}

void ArnoldWintherTriangleFiniteElement::CalcMShape(
   ElementTransformation &Trans, DenseTensor &shape) const
{
   MFEM_VERIFY(Trans.GetSpaceDim() == 2 && Trans.Hessian().FNorm2() < 1e-20,
               "Arnold-Winther elements require affine 2D transformations");
   MFEM_ASSERT(Trans.GetSpaceDim() == 2, "Arnold-Winther is a 2D element");
   MFEM_ASSERT(shape.SizeI() == 2 && shape.SizeJ() == 2 &&
               shape.SizeK() == aw_dof, "invalid matrix shape size");
   DenseTensor ref_shape(2, 2, aw_dof);
   CalcMShape(Trans.GetIntPoint(), ref_shape);
   const DenseMatrix &J = Trans.Jacobian();
   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   const real_t scale = 1.0/(det*det);
   DenseTensor mapped(2, 2, aw_dof);
   for (int k = 0; k < aw_dof; k++)
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
   GetBasisTransform(J, A);
   for (int k = 0; k < aw_dof; k++)
   {
      for (int i = 0; i < 2; i++)
      {
         for (int j = 0; j < 2; j++)
         {
            shape(i,j,k) = 0.0;
            for (int l = 0; l < aw_dof; l++)
            {
               shape(i,j,k) += A(k,l)*mapped(i,j,l);
            }
         }
      }
   }
}

void ArnoldWintherTriangleFiniteElement::CalcPhysDivShape(
   ElementTransformation &Trans, DenseMatrix &divshape) const
{
   MFEM_VERIFY(Trans.GetSpaceDim() == 2 && Trans.Hessian().FNorm2() < 1e-20,
               "Arnold-Winther elements require affine 2D transformations");
   MFEM_ASSERT(divshape.Height() == aw_dof && divshape.Width() == 2,
               "invalid matrix divergence shape size");
   DenseMatrix ref_div(aw_dof, 2), mapped(aw_dof, 2), A;
   CalcDivShape(Trans.GetIntPoint(), ref_div);
   const DenseMatrix &J = Trans.Jacobian();
   const real_t det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
   for (int k = 0; k < aw_dof; k++)
   {
      for (int i = 0; i < 2; i++)
      {
         mapped(k,i) = (J(i,0)*ref_div(k,0) + J(i,1)*ref_div(k,1)) /
                       (det*det);
      }
   }
   GetBasisTransform(J, A);
   Mult(A, mapped, divshape);
}

void ArnoldWintherTriangleFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   MFEM_VERIFY(fe.GetDim() == 2 && fe.GetGeomType() == Geometry::TRIANGLE &&
               fe.GetRangeType() == SCALAR && fe.GetMapType() == VALUE,
               "Arnold-Winther projection requires a scalar H1 triangle "
               "with three byVDIM components");
   const int n = fe.GetDof();
   Vector values(n);
   ApplyDofs(Trans,3*n,fe.GetOrder(),
             [&](const IntegrationPoint &ip, DenseTensor &shape)
   {
      Trans.SetIntPoint(&ip);
      fe.CalcPhysShape(Trans,values);
      shape = 0.0;
      for (int c = 0; c < 3; c++)
      {
         for (int j = 0; j < n; j++)
         { shape(rows[c],cols[c],c*n+j) = shape(cols[c],rows[c],c*n+j) = values(j); }
      }
   },I);
}

void ArnoldWintherTriangleFiniteElement::GetTransferMatrix(
   const FiniteElement &fe, ElementTransformation &Trans, DenseMatrix &I) const
{
   MFEM_VERIFY(dynamic_cast<const ArnoldWintherTriangleFiniteElement *>(&fe),
               "Arnold-Winther transfer requires an Arnold-Winther source");
   Vector point(2);
   ApplyDofs(Trans,fe.GetDof(),fe.GetOrder(),
             [&](const IntegrationPoint &ip, DenseTensor &shape)
   {
      Trans.Transform(ip,point);
      IntegrationPoint coarse_ip;
      coarse_ip.Set2(point(0),point(1));
      fe.CalcMShape(coarse_ip,shape);
   },I);
}

void ArnoldWintherTriangleFiniteElement::GetFaceDofs(
   int face, int **dofs, int *ndofs) const
{
   static int face_dofs[3][10] =
   {
      {0,1,2,3,4,5,9,10,11,12},
      {3,4,5,6,7,8,13,14,15,16},
      {6,7,8,0,1,2,17,18,19,20}
   };
   MFEM_ASSERT(face >= 0 && face < 3, "invalid face index");
   *dofs = face_dofs[face];
   *ndofs = 10;
}

void ArnoldWintherTriangleFiniteElement::GetPhysicalTransferMatrix(
   const DenseMatrix &reference_transfer, ElementTransformation &child,
   ElementTransformation &fine, DenseMatrix &I) const
{
   const IntegrationPoint &center = Geometries.GetCenter(Geometry::TRIANGLE);
   child.SetIntPoint(&center);
   fine.SetIntPoint(&center);
   DenseMatrix inverse_child(2), coarse_J(2);
   CalcInverse(child.Jacobian(), inverse_child);
   Mult(fine.Jacobian(), inverse_child, coarse_J);

   DenseMatrix Ac, Af, Ar, inverse_Af(aw_dof);
   GetBasisTransform(coarse_J, Ac);
   GetBasisTransform(fine.Jacobian(), Af);
   GetBasisTransform(child.Jacobian(), Ar);
   DenseMatrixInverse(Af).GetInverseMatrix(inverse_Af);

   // A^T maps physical moment coefficients to double-Piola coefficients.
   // Piola maps compose, but the geometry-dependent facet corrections do not:
   // I_phys = A_f^{-T} A_child^T I_ref A_coarse^T.
   DenseMatrix right(aw_dof), middle(aw_dof);
   MultABt(reference_transfer, Ac, right);
   MultAtB(Ar, right, middle);
   I.SetSize(aw_dof);
   MultAtB(inverse_Af, middle, I);
}

} // namespace mfem
