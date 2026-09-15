// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.


#include "fe_hzzz.hpp"
#include "../eltrans.hpp"
#include <cmath>

namespace mfem
{
namespace
{
constexpr real_t vertices[3][2] = {{0,0}, {1,0}, {0,1}};
constexpr int edges[3][2] = {{0,1}, {1,2}, {2,0}};
constexpr int retained[21] = {0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,17,18,19,21,22,23};
}

HuangZhangZhouZhuTriangleFiniteElement::HuangZhangZhouZhuTriangleFiniteElement()
   : FiniteElement(2, Geometry::TRIANGLE, 21, 3, FunctionSpace::Pk)
{
   range_type = MATRIX;
   map_type = DOUBLE_CONTRAVARIANT_PIOLA;
   deriv_type = DIV;
   deriv_range_type = VECTOR;
   deriv_map_type = UNKNOWN_MAP_TYPE;
   vdim = 2;
   for (int i = 0; i < dof; i++)
   { Nodes.IntPoint(i) = aw.GetNodes().IntPoint(retained[i]); }
   DenseMatrix J(2);
   J = 0.0; J(0,0) = J(1,1) = 1.0;
   GetEmbedding(J,reference_embedding);
}

void HuangZhangZhouZhuTriangleFiniteElement::GetEmbedding(
   const DenseMatrix &J, DenseMatrix &E) const
{
   E.SetSize(24,21);
   E = 0.0;
   for (int i = 0; i < dof; i++) { E(retained[i],i) = 1.0; }
   for (int e = 0; e < 3; e++)
   {
      const int a = edges[e][0], b = edges[e][1];
      const real_t dx = vertices[b][0]-vertices[a][0];
      const real_t dy = vertices[b][1]-vertices[a][1];
      const real_t tx = J(0,0)*dx+J(0,1)*dy;
      const real_t ty = J(1,0)*dx+J(1,1)*dy;
      const real_t scale = 1.0/(6*std::hypot(tx,ty));
      // For quadratic q(s), integral_0^1 (2s-1)q(s) ds = (q(1)-q(0))/6.
      // This removes the cubic nt trace from the physical AW space.
      const real_t nt[3] = {tx*ty, ty*ty-tx*tx, -tx*ty};
      for (int c = 0; c < 3; c++)
      {
         E(12+4*e,3*a+c) = -scale*nt[c];
         E(12+4*e,3*b+c) = scale*nt[c];
      }
   }
}

void HuangZhangZhouZhuTriangleFiniteElement::ReduceShape(
   const DenseMatrix &E, const DenseTensor &full, DenseTensor &shape) const
{
   shape = 0.0;
   for (int k = 0; k < dof; k++)
   {
      for (int l = 0; l < 24; l++)
      {
         for (int i = 0; i < 2; i++)
         {
            for (int j = 0; j < 2; j++) { shape(i,j,k) += E(l,k)*full(i,j,l); }
         }
      }
   }
}

void HuangZhangZhouZhuTriangleFiniteElement::SelectDofs(
   const DenseMatrix &full, DenseMatrix &I) const
{
   I.SetSize(dof,full.Width());
   for (int i = 0; i < dof; i++)
   {
      for (int j = 0; j < full.Width(); j++) { I(i,j) = full(retained[i],j); }
   }
}

void HuangZhangZhouZhuTriangleFiniteElement::CalcMShape(
   const IntegrationPoint &ip, DenseTensor &shape) const
{
   DenseTensor full(2,2,24);
   aw.CalcMShape(ip,full);
   ReduceShape(reference_embedding,full,shape);
}

void HuangZhangZhouZhuTriangleFiniteElement::CalcMShape(
   ElementTransformation &T, DenseTensor &shape) const
{
   DenseTensor full(2,2,24);
   DenseMatrix E;
   aw.CalcMShape(T,full);
   GetEmbedding(T.Jacobian(),E);
   ReduceShape(E,full,shape);
}

void HuangZhangZhouZhuTriangleFiniteElement::CalcDivShape(
   const IntegrationPoint &ip, DenseMatrix &shape) const
{
   DenseMatrix full(24,2);
   aw.CalcDivShape(ip,full);
   MultAtB(reference_embedding,full,shape);
}

void HuangZhangZhouZhuTriangleFiniteElement::CalcPhysDivShape(
   ElementTransformation &T, DenseMatrix &shape) const
{
   DenseMatrix full(24,2), E;
   aw.CalcPhysDivShape(T,full);
   GetEmbedding(T.Jacobian(),E);
   MultAtB(E,full,shape);
}

void HuangZhangZhouZhuTriangleFiniteElement::Project(
   const FiniteElement &fe, ElementTransformation &T, DenseMatrix &I) const
{
   DenseMatrix full;
   aw.Project(fe,T,full);
   SelectDofs(full,I);
}

void HuangZhangZhouZhuTriangleFiniteElement::GetTransferMatrix(
   const FiniteElement &fe, ElementTransformation &T, DenseMatrix &I) const
{
   MFEM_VERIFY(dynamic_cast<const HuangZhangZhouZhuTriangleFiniteElement *>(&fe),
               "HZZZ transfer requires an HZZZ source element");
   DenseMatrix full, reduced(24,dof);
   aw.GetLocalInterpolation(T,full);
   Mult(full,reference_embedding,reduced);
   SelectDofs(reduced,I);
}

void HuangZhangZhouZhuTriangleFiniteElement::GetPhysicalTransferMatrix(
   const DenseMatrix &, ElementTransformation &child,
   ElementTransformation &fine, DenseMatrix &I) const
{
   const auto &center = Geometries.GetCenter(Geometry::TRIANGLE);
   child.SetIntPoint(&center);
   fine.SetIntPoint(&center);
   MFEM_VERIFY(child.GetSpaceDim() == 2 && fine.GetSpaceDim() == 2 &&
               child.Hessian().FNorm2() < 1e-20 && fine.Hessian().FNorm2() < 1e-20,
               "HZZZ transfer requires affine 2D transformations");
   DenseMatrix K(2), J(2), E, reference, physical, reduced(24,dof);
   CalcInverse(child.Jacobian(),K);
   Mult(fine.Jacobian(),K,J);
   GetEmbedding(J,E);
   // Retain the full AW transfer until after the physical change of basis:
   // the eliminated nt moments depend on the coarse physical geometry.
   aw.GetLocalInterpolation(child,reference);
   aw.GetPhysicalTransferMatrix(reference,child,fine,physical);
   Mult(physical,E,reduced);
   SelectDofs(reduced,I);
}

void HuangZhangZhouZhuTriangleFiniteElement::GetFaceDofs(
   int face, int **dofs, int *ndofs) const
{
   static int indices[3][9] = {{0,1,2,3,4,5,9,10,11},
      {3,4,5,6,7,8,12,13,14},
      {6,7,8,0,1,2,15,16,17}
   };
   MFEM_ASSERT(face >= 0 && face < 3, "invalid face index");
   *dofs = indices[face];
   *ndofs = 9;
}

} // namespace mfem
