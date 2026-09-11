// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.

#include "mfem.hpp"
#include "unit_tests.hpp"
#include <cmath>
#include <memory>

using namespace mfem;

namespace
{

const real_t vertices[3][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
const int edge_vertices[3][2] = {{0, 1}, {1, 2}, {2, 0}};

void ReferenceDofMatrix(const JohnsonMercierTriangleFiniteElement &fe,
                        DenseMatrix &dofs)
{
   dofs.SetSize(15);
   dofs = 0.0;
   const IntegrationRule &edge_rule = IntRules.Get(Geometry::SEGMENT, 3);
   DenseTensor shape(2, 2, 15);
   for (int edge = 0; edge < 3; edge++)
   {
      const real_t *a = vertices[edge_vertices[edge][0]];
      const real_t *b = vertices[edge_vertices[edge][1]];
      const real_t dx = b[0] - a[0];
      const real_t dy = b[1] - a[1];
      const real_t length = std::sqrt(dx*dx + dy*dy);
      const real_t t[2] = {dx/length, dy/length};
      const real_t n[2] = {t[1], -t[0]};
      for (int q = 0; q < edge_rule.GetNPoints(); q++)
      {
         const IntegrationPoint &sip = edge_rule.IntPoint(q);
         IntegrationPoint ip;
         ip.Set2(a[0] + sip.x*dx, a[1] + sip.x*dy);
         fe.CalcMShape(ip, shape);
         for (int mode = 0; mode < 2; mode++)
         {
            const real_t polynomial = mode == 0 ? 1.0 : 2.0*sip.x - 1.0;
            for (int k = 0; k < 15; k++)
            {
               const real_t sn0 = shape(0,0,k)*n[0] + shape(0,1,k)*n[1];
               const real_t sn1 = shape(1,0,k)*n[0] + shape(1,1,k)*n[1];
               dofs(4*edge + 2*mode, k) += sip.weight*length*polynomial*
                                           (n[0]*sn0 + n[1]*sn1);
               dofs(4*edge + 2*mode + 1, k) += sip.weight*length*polynomial*
                                               (t[0]*sn0 + t[1]*sn1);
            }
         }
      }
   }

   const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 2);
   std::unique_ptr<IntegrationRule> ir(base.ApplyToTriangleAlfeldSplit());
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      fe.CalcMShape(ip, shape);
      for (int k = 0; k < 15; k++)
      {
         dofs(12,k) += ip.weight*shape(0,0,k);
         dofs(13,k) += ip.weight*shape(0,1,k);
         dofs(14,k) += ip.weight*shape(1,1,k);
      }
   }
}

real_t EvalAffineComponent(const real_t coefficients[3][3], int component,
                           real_t x, real_t y)
{
   return coefficients[component][0] + coefficients[component][1]*x +
          coefficients[component][2]*y;
}

void AffineMatrixDofs(const real_t coefficients[3][3], Vector &dofs)
{
   dofs.SetSize(15);
   dofs = 0.0;
   const real_t gauss[2] = {0.21132486540518711775,
                            0.78867513459481288225
                           };
   for (int edge = 0; edge < 3; edge++)
   {
      const real_t *a = vertices[edge_vertices[edge][0]];
      const real_t *b = vertices[edge_vertices[edge][1]];
      const real_t dx = b[0] - a[0];
      const real_t dy = b[1] - a[1];
      const real_t length = std::sqrt(dx*dx + dy*dy);
      const real_t t[2] = {dx/length, dy/length};
      const real_t n[2] = {t[1], -t[0]};
      for (int q = 0; q < 2; q++)
      {
         const real_t r = gauss[q];
         const real_t x = a[0] + r*dx;
         const real_t y = a[1] + r*dy;
         const real_t s00 = EvalAffineComponent(coefficients, 0, x, y);
         const real_t s01 = EvalAffineComponent(coefficients, 1, x, y);
         const real_t s11 = EvalAffineComponent(coefficients, 2, x, y);
         const real_t sn0 = s00*n[0] + s01*n[1];
         const real_t sn1 = s01*n[0] + s11*n[1];
         dofs(4*edge) += 0.5*length*(n[0]*sn0 + n[1]*sn1);
         dofs(4*edge + 1) += 0.5*length*(t[0]*sn0 + t[1]*sn1);
         dofs(4*edge + 2) +=
            0.5*length*(2.0*r - 1.0)*(n[0]*sn0 + n[1]*sn1);
         dofs(4*edge + 3) +=
            0.5*length*(2.0*r - 1.0)*(t[0]*sn0 + t[1]*sn1);
      }
   }
   for (int component = 0; component < 3; component++)
   {
      dofs(12 + component) = 0.5*EvalAffineComponent(
                                coefficients, component, 1.0/3.0, 1.0/3.0);
   }
}

} // namespace

TEST_CASE("Johnson-Mercier triangle element", "[FiniteElement][JohnsonMercier]")
{
   JohnsonMercierTriangleFiniteElement fe;

   SECTION("metadata and collection")
   {
      REQUIRE(fe.GetDim() == 2);
      REQUIRE(fe.GetDof() == 15);
      REQUIRE(fe.GetOrder() == 1);
      REQUIRE(fe.GetRangeType() == FiniteElement::MATRIX);
      REQUIRE(fe.GetMapType() == FiniteElement::DOUBLE_CONTRAVARIANT_PIOLA);
      REQUIRE(fe.GetDerivType() == FiniteElement::DIV);
      REQUIRE(fe.GetDerivRangeType() == FiniteElement::VECTOR);

      JohnsonMercierFECollection fec;
      REQUIRE(fec.DofForGeometry(Geometry::POINT) == 0);
      REQUIRE(fec.DofForGeometry(Geometry::SEGMENT) == 4);
      REQUIRE(fec.DofForGeometry(Geometry::TRIANGLE) == 3);
      REQUIRE(fec.GetContType() == FiniteElementCollection::NORMAL);
      std::unique_ptr<FiniteElementCollection> factory(
         FiniteElementCollection::New("JM_2D_P1"));
      REQUIRE(factory != nullptr);
   }

   SECTION("reference degrees of freedom are unisolvent")
   {
      DenseMatrix dofs;
      ReferenceDofMatrix(fe, dofs);
      for (int i = 0; i < 15; i++)
      {
         for (int j = 0; j < 15; j++)
         {
            CAPTURE(i, j, dofs(i,j));
            REQUIRE(dofs(i,j) == MFEM_Approx(i == j ? 1.0 : 0.0));
         }
      }
   }

   SECTION("shape symmetry and row divergence")
   {
      IntegrationPoint ip, xp, xm, yp, ym;
      ip.Set2(0.2, 0.1);
      const real_t eps = 1e-6;
      xp.Set2(ip.x + eps, ip.y);
      xm.Set2(ip.x - eps, ip.y);
      yp.Set2(ip.x, ip.y + eps);
      ym.Set2(ip.x, ip.y - eps);
      DenseTensor shape(2,2,15), sxp(2,2,15), sxm(2,2,15);
      DenseTensor syp(2,2,15), sym(2,2,15);
      DenseMatrix div(15,2);
      fe.CalcMShape(ip, shape);
      fe.CalcMShape(xp, sxp);
      fe.CalcMShape(xm, sxm);
      fe.CalcMShape(yp, syp);
      fe.CalcMShape(ym, sym);
      fe.CalcDivShape(ip, div);
      for (int k = 0; k < 15; k++)
      {
         REQUIRE(shape(0,1,k) == MFEM_Approx(shape(1,0,k)));
         for (int i = 0; i < 2; i++)
         {
            const real_t finite_difference =
               (sxp(i,0,k) - sxm(i,0,k) + syp(i,1,k) - sym(i,1,k)) /
               (2.0*eps);
            CAPTURE(k, i, finite_difference, div(k,i));
            REQUIRE(div(k,i) == MFEM_Approx(finite_difference)
                    .epsilon(1e-7).margin(1e-7));
         }
      }
   }

   SECTION("normal traction is continuous across the Alfeld split")
   {
      const real_t center[2] = {1.0/3.0, 1.0/3.0};
      const real_t eps = 1e-8;
      for (int vertex = 0; vertex < 3; vertex++)
      {
         const real_t tx = vertices[vertex][0] - center[0];
         const real_t ty = vertices[vertex][1] - center[1];
         const real_t length = std::sqrt(tx*tx + ty*ty);
         const real_t n[2] = {ty/length, -tx/length};
         IntegrationPoint plus, minus;
         plus.Set2(center[0] + 0.6*tx + eps*n[0],
                   center[1] + 0.6*ty + eps*n[1]);
         minus.Set2(center[0] + 0.6*tx - eps*n[0],
                    center[1] + 0.6*ty - eps*n[1]);
         DenseTensor sp(2,2,15), sm(2,2,15);
         fe.CalcMShape(plus, sp);
         fe.CalcMShape(minus, sm);
         for (int k = 0; k < 15; k++)
         {
            for (int i = 0; i < 2; i++)
            {
               const real_t tp = sp(i,0,k)*n[0] + sp(i,1,k)*n[1];
               const real_t tm = sm(i,0,k)*n[0] + sm(i,1,k)*n[1];
               CAPTURE(vertex, k, i, tp, tm);
               REQUIRE(tp == MFEM_Approx(tm).margin(1e-5));
            }
         }
      }
   }

   SECTION("physical divergence on an affine element")
   {
      Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE, true);
      for (int v = 0; v < mesh.GetNV(); v++)
      {
         real_t *point = mesh.GetVertex(v);
         const real_t x = point[0];
         const real_t y = point[1];
         point[0] = 1.7*x + 0.3*y;
         point[1] = 0.2*x + 1.4*y;
      }
      ElementTransformation *T = mesh.GetElementTransformation(0);
      IntegrationPoint ip;
      ip.Set2(0.2, 0.1);
      T->SetIntPoint(&ip);
      DenseMatrix inverse_jacobian(2);
      CalcInverse(T->Jacobian(), inverse_jacobian);
      const real_t eps = 1e-6;
      IntegrationPoint xp, xm, yp, ym;
      xp.Set2(ip.x + eps*inverse_jacobian(0,0),
              ip.y + eps*inverse_jacobian(1,0));
      xm.Set2(ip.x - eps*inverse_jacobian(0,0),
              ip.y - eps*inverse_jacobian(1,0));
      yp.Set2(ip.x + eps*inverse_jacobian(0,1),
              ip.y + eps*inverse_jacobian(1,1));
      ym.Set2(ip.x - eps*inverse_jacobian(0,1),
              ip.y - eps*inverse_jacobian(1,1));
      DenseTensor sxp(2,2,15), sxm(2,2,15), syp(2,2,15), sym(2,2,15);
      T->SetIntPoint(&xp);
      fe.CalcMShape(*T, sxp);
      T->SetIntPoint(&xm);
      fe.CalcMShape(*T, sxm);
      T->SetIntPoint(&yp);
      fe.CalcMShape(*T, syp);
      T->SetIntPoint(&ym);
      fe.CalcMShape(*T, sym);
      T->SetIntPoint(&ip);
      DenseMatrix div(15,2);
      fe.CalcPhysDivShape(*T, div);
      for (int k = 0; k < 15; k++)
      {
         for (int i = 0; i < 2; i++)
         {
            const real_t finite_difference =
               (sxp(i,0,k) - sxm(i,0,k) + syp(i,1,k) - sym(i,1,k)) /
               (2.0*eps);
            CAPTURE(k, i, finite_difference, div(k,i));
            REQUIRE(div(k,i) == MFEM_Approx(finite_difference)
                    .epsilon(1e-7).margin(1e-7));
         }
      }
   }

   SECTION("transfer matrix applies fine Johnson-Mercier moments")
   {
      IsoparametricTransformation transformation;
      transformation.SetIdentityTransformation(Geometry::TRIANGLE);
      DenseMatrix identity;
      fe.GetTransferMatrix(fe, transformation, identity);
      for (int i = 0; i < 15; i++)
      {
         for (int j = 0; j < 15; j++)
         {
            CAPTURE(i, j, identity(i,j));
            REQUIRE(identity(i,j) == MFEM_Approx(i == j ? 1.0 : 0.0)
                    .margin(1e-12));
         }
      }

      static real_t child_data[24] =
      {
         0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
         0.5, 0.5, 0.0, 0.5, 0.5, 0.0,
         0.5, 0.0, 1.0, 0.0, 0.5, 0.5,
         0.0, 0.5, 0.5, 0.5, 0.0, 1.0
      };
      DenseTensor children(child_data, 2, 3, 4);
      const real_t coefficients[3][3] =
      {
         {1.2, 0.3, -0.2},
         {-0.4, 0.7, 0.1},
         {2.1, -0.5, 0.8}
      };
      Vector coarse_dofs, fine_dofs;
      AffineMatrixDofs(coefficients, coarse_dofs);
      fine_dofs.SetSize(15);
      const real_t sample_points[3][2] =
      {
         {0.15, 0.15}, {0.65, 0.15}, {0.15, 0.65}
      };
      for (int child = 0; child < 4; child++)
      {
         transformation.SetPointMat(children(child));
         DenseMatrix transfer;
         fe.GetTransferMatrix(fe, transformation, transfer);
         transfer.Mult(coarse_dofs, fine_dofs);
         for (int q = 0; q < 3; q++)
         {
            IntegrationPoint ip;
            ip.Set2(sample_points[q][0], sample_points[q][1]);
            Vector mapped_point(2);
            transformation.Transform(ip, mapped_point);
            transformation.SetIntPoint(&ip);
            DenseTensor shape(2, 2, 15);
            fe.CalcMShape(transformation, shape);
            for (int row = 0; row < 2; row++)
            {
               for (int col = 0; col < 2; col++)
               {
                  real_t value = 0.0;
                  for (int k = 0; k < 15; k++)
                  {
                     value += fine_dofs(k)*shape(row,col,k);
                  }
                  CAPTURE(child, q, row, col, value);
                  const int component = row == col ? 2*row : 1;
                  const real_t exact = EvalAffineComponent(
                                          coefficients, component,
                                          mapped_point(0), mapped_point(1));
                  REQUIRE(value == MFEM_Approx(exact)
                          .margin(1e-11));
               }
            }
         }
      }
   }
}

TEST_CASE("Johnson-Mercier finite element space transfer",
          "[Transfer][JohnsonMercier]")
{
   Mesh coarse_mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();
   JohnsonMercierFECollection fec;
   const int vdim = GENERATE(1, 2);
   const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
   FiniteElementSpace coarse_fes(&coarse_mesh, &fec, vdim, ordering);
   FiniteElementSpace fine_fes(&fine_mesh, &fec, vdim, ordering);

   OperatorHandle assembled(Operator::MFEM_SPARSEMAT);
   OperatorHandle matrix_free(Operator::ANY_TYPE);
   fine_fes.GetTransferOperator(coarse_fes, assembled);
   fine_fes.GetTransferOperator(coarse_fes, matrix_free);
   REQUIRE(assembled->Height() == fine_fes.GetVSize());
   REQUIRE(assembled->Width() == coarse_fes.GetVSize());

   Vector x(coarse_fes.GetVSize()), y_assembled(fine_fes.GetVSize());
   Vector y_matrix_free(fine_fes.GetVSize());
   for (int i = 0; i < x.Size(); i++) { x(i) = std::sin(real_t(i + 1)); }
   Array<int> orientation_dofs;
   DofTransformation orientation_transform;
   fine_fes.GetElementDofs(0, orientation_dofs, orientation_transform);
   REQUIRE(orientation_transform.IsIdentity());
   assembled->Mult(x, y_assembled);
   matrix_free->Mult(x, y_matrix_free);
   y_matrix_free -= y_assembled;
   REQUIRE(y_matrix_free.Norml2() < 1e-12*y_assembled.Norml2());

   Vector z(fine_fes.GetVSize()), x_assembled(coarse_fes.GetVSize());
   Vector x_matrix_free(coarse_fes.GetVSize());
   for (int i = 0; i < z.Size(); i++) { z(i) = std::cos(real_t(i + 1)); }
   assembled->MultTranspose(z, x_assembled);
   matrix_free->MultTranspose(z, x_matrix_free);
   x_matrix_free -= x_assembled;
   REQUIRE(x_matrix_free.Norml2() < 1e-12*x_assembled.Norml2());

   // Updating a space in place uses GetLocalInterpolation, whereas a hierarchy
   // uses GetTransferMatrix. Both must apply the same physical basis change.
   Mesh updated_mesh(coarse_mesh);
   FiniteElementSpace updated(&updated_mesh, &fec, vdim, ordering);
   updated.GetElementToDofTable();
   updated_mesh.UniformRefinement();
   updated.Update();
   Vector y_updated(updated.GetVSize());
   updated.GetUpdateOperator()->Mult(x, y_updated);
   y_updated -= y_assembled;
   REQUIRE(y_updated.Norml2() < 1e-12*y_assembled.Norml2());
}

TEST_CASE("Johnson-Mercier assembly", "[BilinearForm][JohnsonMercier]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   JohnsonMercierFECollection fec;
   FiniteElementSpace fes(&mesh, &fec);
   REQUIRE(fes.GetNDofs() == 4*mesh.GetNEdges() + 3*mesh.GetNE());

   BilinearForm form(&fes);
   form.AddDomainIntegrator(new MatrixFEMassIntegrator);
   form.AddDomainIntegrator(new MatrixDivDivIntegrator);
   form.Assemble();
   const SparseMatrix &matrix = form.SpMat();
   Vector x(fes.GetVSize()), y(fes.GetVSize());
   for (int i = 0; i < x.Size(); i++) { x(i) = std::sin(real_t(i + 1)); }
   matrix.Mult(x, y);
   REQUIRE((x*y) > 0.0);

   MatrixFunctionCoefficient coefficient(2,
                                         [](const Vector &p, DenseMatrix &value)
   {
      value.SetSize(2);
      value(0,0) = 1.0 + p(0);
      value(0,1) = value(1,0) = p(0)*p(1);
      value(1,1) = 1.0 + p(1);
   });
   LinearForm load(&fes);
   load.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(coefficient));
   load.Assemble();
   REQUIRE(std::isfinite(load.Norml2()));
   REQUIRE(load.Norml2() > 0.0);
}

TEST_CASE("Johnson-Mercier physical traction continuity",
          "[FiniteElementSpace][JohnsonMercier]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::TRIANGLE, true);
   JohnsonMercierFECollection fec;
   FiniteElementSpace fes(&mesh, &fec);
   Vector x(fes.GetVSize());
   for (int i = 0; i < x.Size(); i++) { x(i) = std::sin(real_t(i + 1)); }
   real_t jump = 0.0;
   for (int f = 0; f < mesh.GetNumFaces(); f++)
   {
      auto *T = mesh.GetFaceElementTransformations(f);
      if (T->Elem2No < 0) { continue; }
      IntegrationPoint ip;
      ip.Set1w(0.37, 1.0);
      T->SetAllIntPoints(&ip);
      Vector normal(2);
      CalcOrtho(T->Jacobian(), normal);
      Vector traction[2] = {Vector(2), Vector(2)};
      for (int side = 0; side < 2; side++)
      {
         const int e = side ? T->Elem2No : T->Elem1No;
         Array<int> dofs;
         DofTransformation trans;
         fes.GetElementDofs(e, dofs, trans);
         Vector local;
         x.GetSubVector(dofs, local);
         trans.InvTransformPrimal(local);
         DenseTensor shape(2, 2, 15);
         fes.GetFE(e)->CalcMShape(side ? *T->Elem2 : *T->Elem1, shape);
         traction[side] = 0.0;
         for (int k = 0; k < 15; k++)
         {
            for (int i = 0; i < 2; i++)
            {
               for (int j = 0; j < 2; j++)
               { traction[side](i) += local(k)*shape(i,j,k)*normal(j); }
            }
         }
      }
      traction[0] -= traction[1];
      jump = std::max(jump, traction[0].Norml2());
   }
   REQUIRE(jump < 1e-10);
}

TEST_CASE("Johnson-Mercier physical interpolation invariants",
          "[Transfer][JohnsonMercier]")
{
   Mesh coarse_mesh = Mesh::MakeCartesian2D(1, 1, Element::TRIANGLE, true);
   // A shear and unequal scaling exercise the geometry-dependent facet basis.
   for (int v = 0; v < coarse_mesh.GetNV(); v++)
   {
      real_t *p = coarse_mesh.GetVertex(v);
      const real_t x = p[0], y = p[1];
      p[0] = 0.3 + 1.7*x + 0.4*y;
      p[1] = -0.2 + 0.2*x + 0.9*y;
   }
   Mesh fine_mesh(coarse_mesh);
   fine_mesh.UniformRefinement();
   JohnsonMercierFECollection fec;
   FiniteElementSpace coarse(&coarse_mesh, &fec), fine(&fine_mesh, &fec);
   OperatorHandle P(Operator::MFEM_SPARSEMAT);
   fine.GetTransferOperator(coarse, P);

   SECTION("reproduces physical affine stresses")
   {
      MatrixFunctionCoefficient exact(2, [](const Vector &p, DenseMatrix &s)
      {
         s.SetSize(2);
         s(0,0) = 1.2 + 0.3*p(0) - 0.2*p(1);
         s(0,1) = s(1,0) = -0.4 + 0.7*p(0) + 0.1*p(1);
         s(1,1) = 2.1 - 0.5*p(0) + 0.8*p(1);
      });
      BilinearForm mass(&coarse);
      mass.AddDomainIntegrator(new MatrixFEMassIntegrator);
      mass.Assemble(); mass.Finalize();
      LinearForm load(&coarse);
      load.AddDomainIntegrator(new MatrixFEDomainLFIntegrator(exact));
      load.Assemble();
      DenseMatrix M;
      mass.SpMat().ToDenseMatrix(M);
      Vector x(coarse.GetVSize()), y(fine.GetVSize());
      DenseMatrixInverse(M, true).Mult(load, x);
      P->Mult(x, y);
      const IntegrationRule &base = IntRules.Get(Geometry::TRIANGLE, 2);
      std::unique_ptr<IntegrationRule> rule(base.ApplyToTriangleAlfeldSplit());
      for (int level = 0; level < 2; level++)
      {
         FiniteElementSpace &fes = level ? fine : coarse;
         const Vector &coefficients = level ? y : x;
         real_t error = 0.0;
         for (int e = 0; e < fes.GetNE(); e++)
         {
            Array<int> dofs;
            fes.GetElementDofs(e, dofs);
            Vector local;
            coefficients.GetSubVector(dofs, local);
            auto *T = fes.GetElementTransformation(e);
            DenseTensor shape(2, 2, 15);
            DenseMatrix value(2);
            for (int q = 0; q < rule->GetNPoints(); q++)
            {
               const auto &ip = rule->IntPoint(q);
               T->SetIntPoint(&ip);
               fes.GetFE(e)->CalcMShape(*T, shape);
               exact.Eval(value, *T, ip);
               for (int i = 0; i < 2; i++)
               {
                  for (int j = 0; j < 2; j++)
                  {
                     real_t diff = -value(i,j);
                     for (int k = 0; k < 15; k++)
                     { diff += local(k)*shape(i,j,k); }
                     error += ip.weight*T->Weight()*diff*diff;
                  }
               }
            }
         }
         CAPTURE(level, error);
         REQUIRE(std::sqrt(error) < 1e-10);
      }
   }

   SECTION("preserves the divergence-free subspace")
   {
      // Six independent divergence values per macro triangle. Project an
      // arbitrary coefficient vector into their common nullspace.
      DenseMatrix D(6*coarse.GetNE(), coarse.GetVSize());
      D = 0.0;
      const real_t samples[3][2] =
      {{4.0/9.0, 4.0/9.0}, {1.0/9.0, 4.0/9.0}, {4.0/9.0, 1.0/9.0}};
      for (int e = 0; e < coarse.GetNE(); e++)
      {
         Array<int> dofs;
         coarse.GetElementDofs(e, dofs);
         auto *T = coarse.GetElementTransformation(e);
         DenseMatrix div(15,2);
         for (int q = 0; q < 3; q++)
         {
            IntegrationPoint ip;
            ip.Set2(samples[q][0], samples[q][1]);
            T->SetIntPoint(&ip);
            coarse.GetFE(e)->CalcPhysDivShape(*T, div);
            for (int k = 0; k < 15; k++)
            {
               for (int i = 0; i < 2; i++)
               {
                  D(6*e+2*q+i, UnsignIndex(dofs[k])) =
                     (dofs[k] < 0 ? -1.0 : 1.0)*div(k,i);
               }
            }
         }
      }
      DenseMatrix DDt(D.Height());
      MultAAt(D, DDt);
      Vector x(coarse.GetVSize()), dx(D.Height()), correction(x.Size());
      Vector multiplier(D.Height()), y(fine.GetVSize());
      for (int i = 0; i < x.Size(); i++) { x(i) = std::sin(real_t(i+1)); }
      D.Mult(x, dx);
      DenseMatrixInverse(DDt, true).Mult(dx, multiplier);
      D.MultTranspose(multiplier, correction);
      x -= correction;
      D.Mult(x, dx);
      REQUIRE(dx.Norml2() < 1e-10);
      P->Mult(x, y);
      real_t max_div = 0.0;
      for (int e = 0; e < fine.GetNE(); e++)
      {
         Array<int> dofs;
         fine.GetElementDofs(e, dofs);
         Vector local, div_value(2);
         y.GetSubVector(dofs, local);
         auto *T = fine.GetElementTransformation(e);
         DenseMatrix div(15,2);
         for (int q = 0; q < 3; q++)
         {
            IntegrationPoint ip;
            ip.Set2(samples[q][0], samples[q][1]);
            T->SetIntPoint(&ip);
            fine.GetFE(e)->CalcPhysDivShape(*T, div);
            div.MultTranspose(local, div_value);
            max_div = std::max(max_div, div_value.Norml2());
         }
      }
      REQUIRE(max_div < 1e-9);
   }
}
