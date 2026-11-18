// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
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

namespace entity_dof_map
{

// The element matrices of @a integ in the layout expected by
// FillSparseMatrix(): test dof x test vdim x trial dof x trial vdim x element,
// with the element dofs in native order.
static void ElementMatrices(BilinearFormIntegrator &integ,
                            FiniteElementSpace &test_fes,
                            FiniteElementSpace &trial_fes, Vector &ea)
{
   Mesh &mesh = *test_fes.GetMesh();
   const int ne = mesh.GetNE();
   const int test_size = test_fes.GetFE(0)->GetDof()*test_fes.GetVDim();
   const int trial_size = trial_fes.GetFE(0)->GetDof()*trial_fes.GetVDim();
   const int size = test_size*trial_size;
   ea.SetSize(size*ne);
   real_t *h_ea = ea.HostWrite();
   DenseMatrix elmat;
   for (int e = 0; e < ne; e++)
   {
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      if (&test_fes == &trial_fes)
      {
         integ.AssembleElementMatrix(*test_fes.GetFE(e), T, elmat);
      }
      else
      {
         integ.AssembleElementMatrix2(*trial_fes.GetFE(e), *test_fes.GetFE(e),
                                      T, elmat);
      }
      REQUIRE(elmat.Height() == test_size);
      REQUIRE(elmat.Width() == trial_size);
      for (int k = 0; k < size; k++) { h_ea[e*size + k] = elmat.GetData()[k]; }
   }
}

// Check the matrix that FillSparseMatrix() builds from the element matrices of
// @a integ against the one assembled by BilinearForm (MixedBilinearForm for
// different spaces). Takes ownership of @a integ.
static void CheckFillSparseMatrix(BilinearFormIntegrator *integ,
                                  FiniteElementSpace &test_fes,
                                  FiniteElementSpace &trial_fes)
{
   constexpr auto ordering = ElementDofOrdering::NATIVE;
   const EntityDofMap test =
      test_fes.GetElementRestriction(ordering)->GetEntityDofMap();
   const EntityDofMap trial =
      trial_fes.GetElementRestriction(ordering)->GetEntityDofMap();
   REQUIRE(test.ne == test_fes.GetNE());
   REQUIRE(test.vdim == test_fes.GetVDim());
   REQUIRE(test.ndofs == test_fes.GetNDofs());

   Vector ea;
   ElementMatrices(*integ, test_fes, trial_fes, ea);
   SparseMatrix A;
   A.OverrideSize(test_fes.GetVSize(), trial_fes.GetVSize());
   FillSparseMatrix(ea, test, trial, A);

   std::unique_ptr<Operator> form;
   const SparseMatrix *A_ref;
   if (&test_fes == &trial_fes)
   {
      auto *a = new BilinearForm(&test_fes);
      form.reset(a);
      a->AddDomainIntegrator(integ);
      a->Assemble();
      a->Finalize();
      A_ref = &a->SpMat();
   }
   else
   {
      auto *a = new MixedBilinearForm(&trial_fes, &test_fes);
      form.reset(a);
      a->AddDomainIntegrator(integ);
      a->Assemble();
      a->Finalize();
      A_ref = &a->SpMat();
   }

   Vector x(trial_fes.GetVSize()), y(test_fes.GetVSize());
   Vector y_ref(y.Size()), xt(x.Size()), xt_ref(x.Size());
   x.Randomize(1);
   y.Randomize(2);
   A.Mult(x, y_ref);
   A_ref->AddMult(x, y_ref, -1.0);
   A.MultTranspose(y, xt);
   A_ref->MultTranspose(y, xt_ref);
   xt -= xt_ref;
   REQUIRE(y_ref.Normlinf() == MFEM_Approx(0.0));
   REQUIRE(xt.Normlinf() == MFEM_Approx(0.0));
}

} // namespace entity_dof_map

TEST_CASE("FillSparseMatrix from element matrices",
          "[ElementRestriction][FillSparseMatrix]")
{
   using namespace entity_dof_map;
   const int dim = GENERATE(2, 3);
   CAPTURE(dim);
   Mesh mesh = (dim == 2)
               ? Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL)
               : Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);

   SECTION("Mixed H1 spaces")
   {
      H1_FECollection test_fec(2, dim), trial_fec(1, dim);
      FiniteElementSpace test_fes(&mesh, &test_fec);
      FiniteElementSpace trial_fes(&mesh, &trial_fec);
      CheckFillSparseMatrix(new MassIntegrator, test_fes, trial_fes);
   }

   SECTION("Vector H1 space")
   {
      const auto ordering = GENERATE(Ordering::byNODES, Ordering::byVDIM);
      CAPTURE(ordering);
      H1_FECollection fec(2, dim);
      FiniteElementSpace fes(&mesh, &fec, dim, ordering);
      CheckFillSparseMatrix(new VectorMassIntegrator, fes, fes);
   }

   SECTION("L2 space")
   {
      L2_FECollection fec(1, dim);
      FiniteElementSpace fes(&mesh, &fec);
      CheckFillSparseMatrix(new MassIntegrator, fes, fes);
   }
}
