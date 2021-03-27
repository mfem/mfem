// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "catch.hpp"

using namespace mfem;

namespace surf_blf
{

Mesh * GetCylMesh(int type);

void trans_skew_cyl(const Vector &x, Vector &r);

void sigmaFunc(const Vector &x, DenseMatrix &s);

double uExact(const Vector &x)
{
   return (0.25 * (2.0 + x[0]) - x[2]) * (x[2] + 0.25 * (2.0 + x[0]));
}

TEST_CASE("Embedded Surface Diffusion",
          "[DiffusionIntegrator]")
{
   for (int type = (int) Element::TRIANGLE;
        type <= (int) Element::QUADRILATERAL;
        type++)
   {
      int order = 3;

      Mesh *mesh = GetCylMesh(type);
      int dim = mesh->Dimension();

      mesh->SetCurvature(3);
      mesh->Transform(trans_skew_cyl);

      H1_FECollection fec(order, dim);
      FiniteElementSpace fespace(mesh, &fec);

      Array<int> ess_tdof_list;
      if (mesh->bdr_attributes.Size())
      {
         Array<int> ess_bdr(mesh->bdr_attributes.Max());
         ess_bdr = 1;
         fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }

      LinearForm b(&fespace);
      ConstantCoefficient one(1.0);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      GridFunction x(&fespace);
      x = 0.0;

      BilinearForm a(&fespace);
      MatrixFunctionCoefficient sigma(3, sigmaFunc);
      a.AddDomainIntegrator(new DiffusionIntegrator(sigma));
      a.Assemble();

      OperatorPtr A;
      Vector B, X;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 0, 200, 1e-12, 0.0);

      a.RecoverFEMSolution(X, b, x);

      FunctionCoefficient uCoef(uExact);
      double err = x.ComputeL2Error(uCoef);

      REQUIRE(err < 1.5e-3);

      delete mesh;
   }
}

Mesh * GetCylMesh(int type)
{
   Mesh * mesh = NULL;

   if (type == (int) Element::TRIANGLE)
   {
      mesh = new Mesh(2, 12, 16, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);
      mesh->AddVertex( 0.0, -1.0, 0.5);
      mesh->AddVertex( 1.0,  0.0, 0.5);
      mesh->AddVertex( 0.0,  1.0, 0.5);
      mesh->AddVertex(-1.0,  0.0, 0.5);

      mesh->AddTriangle(0, 1, 8);
      mesh->AddTriangle(1, 5, 8);
      mesh->AddTriangle(5, 4, 8);
      mesh->AddTriangle(4, 0, 8);
      mesh->AddTriangle(1, 2, 9);
      mesh->AddTriangle(2, 6, 9);
      mesh->AddTriangle(6, 5, 9);
      mesh->AddTriangle(5, 1, 9);
      mesh->AddTriangle(2, 3, 10);
      mesh->AddTriangle(3, 7, 10);
      mesh->AddTriangle(7, 6, 10);
      mesh->AddTriangle(6, 2, 10);
      mesh->AddTriangle(3, 0, 11);
      mesh->AddTriangle(0, 4, 11);
      mesh->AddTriangle(4, 7, 11);
      mesh->AddTriangle(7, 3, 11);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else if (type == (int) Element::QUADRILATERAL)
   {
      mesh = new Mesh(2, 8, 4, 8, 3);

      mesh->AddVertex(-1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0, -1.0, 0.0);
      mesh->AddVertex( 1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0,  1.0, 0.0);
      mesh->AddVertex(-1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0, -1.0, 1.0);
      mesh->AddVertex( 1.0,  1.0, 1.0);
      mesh->AddVertex(-1.0,  1.0, 1.0);

      mesh->AddQuad(0, 1, 5, 4);
      mesh->AddQuad(1, 2, 6, 5);
      mesh->AddQuad(2, 3, 7, 6);
      mesh->AddQuad(3, 0, 4, 7);

      mesh->AddBdrSegment(0, 1, 1);
      mesh->AddBdrSegment(1, 2, 1);
      mesh->AddBdrSegment(2, 3, 1);
      mesh->AddBdrSegment(3, 0, 1);
      mesh->AddBdrSegment(5, 4, 2);
      mesh->AddBdrSegment(6, 5, 2);
      mesh->AddBdrSegment(7, 6, 2);
      mesh->AddBdrSegment(4, 7, 2);
   }
   else
   {
      MFEM_ABORT("Unrecognized mesh type " << type << "!");
   }
   mesh->FinalizeTopology();

   return mesh;
}

void trans_skew_cyl(const Vector &x, Vector &r)
{
   r.SetSize(3);

   double tol = 1e-6;
   double theta = 0.0;
   if (fabs(x[1] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (x[0] - 2.0);
   }
   else if (fabs(x[0] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * x[1];
   }
   else if (fabs(x[1] - 1.0) < tol)
   {
      theta = 0.25 * M_PI * (2.0 - x[0]);
   }
   else if (fabs(x[0] + 1.0) < tol)
   {
      theta = 0.25 * M_PI * (4.0 - x[1]);
   }
   else
   {
      mfem::out << "side not recognized "
                << x[0] << " " << x[1] << " " << x[2] << std::endl;
   }

   r[0] = cos(theta);
   r[1] = sin(theta);
   r[2] = 0.25 * (2.0 * x[2] - 1.0) * (r[0] + 2.0);
}

void sigmaFunc(const Vector &x, DenseMatrix &s)
{
   s.SetSize(3);
   double a = 17.0 - 2.0 * x[0] * (1.0 + x[0]);
   s(0,0) = 0.5 + x[0] * x[0] * (8.0 / a - 0.5);
   s(0,1) = x[0] * x[1] * (8.0 / a - 0.5);
   s(0,2) = 0.0;
   s(1,0) = s(0,1);
   s(1,1) = 0.5 * x[0] * x[0] + 8.0 * x[1] * x[1] / a;
   s(1,2) = 0.0;
   s(2,0) = 0.0;
   s(2,1) = 0.0;
   s(2,2) = a / 32.0;
}

} // namespace surf_blf
