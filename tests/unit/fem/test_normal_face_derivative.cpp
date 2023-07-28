// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

constexpr double __C[] = {1.0, 2.0, 3.1415};

double f(const Vector& x)
{
   double F = 0.0;
   for (int d = 0; d < x.Size(); ++d)
   {
      F += __C[d] * x[d];
   }

   return F;
}

// Constructs a sparse matrix representation of op by applying op to the columns
// of the identity matrix. SLOW but just for testing.
SparseMatrix PA2Sparse(const Operator * op);

TEST_CASE("Normal Face Derivative", "[FaceRestriction]")
{
   // Set grid function f(x) = dot(c, x) and verify that normal derivatives
   // computed by L2NormalDerivativeFaceRestriction are dot(c, n).

   const int dim = GENERATE(2, 3);
   const int nx = 3;
   const int order = 4;

   Mesh mesh = (dim == 2) ? Mesh::MakeCartesian2D(nx, nx, Element::QUADRILATERAL,
                                                  true) : Mesh::MakeCartesian3D(nx, nx, nx, Element::HEXAHEDRON);
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);

   FiniteElementSpace fes(&mesh, &fec);

   GridFunction gf(&fes);

   FaceType face_type = GENERATE(FaceType::Boundary, FaceType::Interior);

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule& ir = irs.Get(mesh.GetFaceGeometry(0), 2 * order - 1);
   FaceQuadratureSpace fqs(mesh, ir, face_type);

   L2NormalDerivativeFaceRestriction dfr(fes, ElementDofOrdering::LEXICOGRAPHIC,
                                         face_type);

   QuadratureFunction qf(fqs, 2);
   QuadratureFunction qfref(fqs, 2);
   QuadratureFunction qff(fqs, 1);
   FunctionCoefficient coef(f);
   coef.Project(qff);
   gf.ProjectCoefficient(coef);

   auto geom = mesh.GetFaceGeometricFactors(ir, FaceGeometricFactors::NORMALS,
                                            face_type);
   auto ns = geom->normal.Read();

   const int np = ir.Size();

   for (int f=0; f < fqs.GetNumFaces(); ++f)
   {
      for (int p=0; p < np; ++p)
      {
         double val = 0.0;
         for (int d = 0; d < dim; ++d)
         {
            double n = std::abs(ns[p + np * (d + dim * f)]);

            val += __C[d] * n;
         }

         qfref[p + np * (0 + 2 * f)] = val / nx;
         qfref[p + np * (1 + 2 * f)] = (face_type == FaceType::Interior) ?
                                       (val / nx) : 0.0;
      }
   }

   dfr.Mult(gf, qf);
   qf -= qfref;

   REQUIRE(qf.Normlinf() == MFEM_Approx(0.0));

   // test transpose
   SparseMatrix A = PA2Sparse(&dfr);

   Vector y(qf.Size());
   y.Randomize(1);

   Vector x(gf.Size());
   x = 0.0;
   Vector xref(gf.Size());
   xref = 0.0;

   dfr.AddMultTranspose(y, x);
   A.AddMultTranspose(y, xref);

   x -= xref;

   REQUIRE(x.Normlinf() == MFEM_Approx(0.0));
}
void rotate_3d_vertices(int *v, int ref_face, int rot);

Mesh mesh_3d_orientation1(int face_perm_1, int face_perm_2)
{
   static const int dim = 3;
   static const int nv = 12;
   static const int nel = 2;
   Mesh mesh(dim, nv, nel);
   double x[dim];
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 0.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 0.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 0.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 1.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);
   x[0] = 2.0;   x[1] = 1.0;   x[2] = 1.0;
   mesh.AddVertex(x);

   int el[8];

   el[0] = 0;
   el[1] = 1;
   el[2] = 4;
   el[3] = 3;
   el[4] = 6;
   el[5] = 7;
   el[6] = 10;
   el[7] = 9;
   rotate_3d_vertices(el, face_perm_1/4, face_perm_1%4);
   mesh.AddHex(el);

   el[0] = 1;
   el[1] = 2;
   el[2] = 5;
   el[3] = 4;
   el[4] = 7;
   el[5] = 8;
   el[6] = 11;
   el[7] = 10;
   rotate_3d_vertices(el, face_perm_2/4, face_perm_2%4);
   mesh.AddHex(el);

   mesh.FinalizeHexMesh(true);
   mesh.GenerateBoundaryElements();
   mesh.Finalize();
   return mesh;
}

TEST_CASE("normal perm")
{
   const int order = 2;
   const int perm1 = GENERATE(range(0, 24));
   const int perm2 = GENERATE(range(0, 24));

   Mesh mesh = mesh_3d_orientation1(perm1, perm2);
   const int dim = mesh.Dimension();

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   GridFunction gf(&fes);

   FaceType face_type = FaceType::Interior;

   IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   const IntegrationRule& ir = irs.Get(mesh.GetFaceGeometry(0), 2 * order - 1);
   FaceQuadratureSpace fqs(mesh, ir, face_type);

   L2NormalDerivativeFaceRestriction dfr(fes, ElementDofOrdering::LEXICOGRAPHIC,
                                         face_type);

   QuadratureFunction qf(fqs, 2);
   QuadratureFunction qfref(fqs, 2);
   QuadratureFunction qff(fqs, 1);
   FunctionCoefficient coef([](const Vector& x)->double{return x[0]*x[1]*x[2];});
   coef.Project(qff);
   gf.ProjectCoefficient(coef);

   const int np = ir.Size();

   FunctionCoefficient dercoef([](const Vector& x)->double{return x[1]*x[2];});
   GridFunction dgf(&fes);
   dgf.ProjectCoefficient(dercoef);
   qff.ProjectGridFunction(dgf);

   for (int f = 0; f < fqs.GetNumFaces(); ++f)
   {
      for (int p = 0; p < np; ++p)
      {
         qfref(p + np * (0 + 2 * f)) = qff(p + np * f);
         qfref(p + np * (1 + 2 * f)) = qff(p + np * f);
      }
   }

   dfr.Mult(gf, qf);

   // for (int i=0; i < qf.Size(); ++i)
   // {
   //    std::cout << qf(i) << " " << qfref(i) << "\n";
   // }

   double e = 0;

   for (int i=0; i < qf.Size(); ++i)
   {
      e = std::max(e, std::abs( std::abs(qf(i)) - std::abs(qfref(i)) ));
   }

   // qf -= qfref;

   // REQUIRE(qf.Normlinf() == MFEM_Approx(0.0));
   REQUIRE(e == MFEM_Approx(0.0));

}


SparseMatrix PA2Sparse(const Operator * op)
{
   const int m = op->NumRows();
   const int n = op->NumCols();

   SparseMatrix A(m, n);

   Vector x(n), y(m);
   x = 0.0;

   for (int j = 0; j < n; ++j)
   {
      x(j) = 1.0;

      op->Mult(x, y);

      for (int i = 0; i < m; ++i)
      {
         if (y(i) != 0.0)
         {
            A.Set(i, j, y(i));
         }
      }

      x(j) = 0.0;
   }

   A.Finalize();
   return A;
}