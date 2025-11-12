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

#include "mfem.hpp"
#include "unit_tests.hpp"
#include "make_permuted_mesh.hpp"

using namespace mfem;

real_t x_fn(const Vector &xvec) { return xvec[0]; }
real_t y_fn(const Vector &xvec) { return xvec[1]; }
real_t z_fn(const Vector &xvec) { return xvec[2]; }

real_t TestFaceRestriction(Mesh &mesh, int order)
{
   int dim = mesh.Dimension();
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction gf(&fes);

   L2FaceRestriction restr(fes, ElementDofOrdering::LEXICOGRAPHIC,
                           FaceType::Interior, L2FaceValues::DoubleValued);

   const int ndof_face = static_cast<int>(pow(order+1, dim-1));

   Vector face_values(ndof_face*2);

   real_t max_err = 0.0;
   for (int d=0; d<dim; ++d)
   {
      real_t (*fn)(const Vector &);
      if (d == 0)
      {
         fn = x_fn;
      }
      else if (d == 1)
      {
         fn = y_fn;
      }
      else if (d == 2)
      {
         fn = z_fn;
      }
      else
      {
         MFEM_ABORT("Bad dimension");
         return infinity();
      }
      FunctionCoefficient coeff(fn);
      gf.ProjectCoefficient(coeff);
      restr.Mult(gf, face_values);
      face_values.HostReadWrite();

      for (int i=0; i<ndof_face; ++i)
      {
         real_t error = std::abs(face_values(i) - face_values(i + ndof_face));
         max_err = std::max(max_err, error);
      }
   }
   return max_err;
}

TEST_CASE("2D Face Permutation", "[Face Permutation]")
{
   int order = 3;
   real_t max_err = 0.0;
   for (int fp2=0; fp2<4; ++fp2)
   {
      for (int fp1=0; fp1<4; ++fp1)
      {
         Mesh mesh = MeshOrientation(2, fp1, fp2);
         real_t error = TestFaceRestriction(mesh, order);
         max_err = std::max(max_err, error);
      }
   }
   REQUIRE(max_err < 1e-15);
}

TEST_CASE("3D Face Permutation", "[Face Permutation]")
{
   int order = 3;
   real_t max_err = 0.0;
   for (int fp2=0; fp2<24; ++fp2)
   {
      for (int fp1=0; fp1<24; ++fp1)
      {
         Mesh mesh = MeshOrientation(3, fp1, fp2);
         real_t error = TestFaceRestriction(mesh, order);
         max_err = std::max(max_err, error);
      }
   }
   REQUIRE(max_err < 1e-15);
}
