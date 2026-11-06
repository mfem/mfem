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

#include "unit_tests.hpp"
#include "mfem.hpp"

namespace mfem
{

real_t f1(const Vector &x)
{
   real_t r = pow(x(0),2);
   if (x.Size() >= 2) { r += pow(x(1), 3); }
   if (x.Size() >= 3) { r += pow(x(2), 4); }
   return r;
}

void gradf1(const Vector &x, Vector &u)
{
   u(0) = 2*x(0);
   if (x.Size() >= 2) { u(1) = 3*pow(x(1), 2); }
   if (x.Size() >= 3) { u(2) = 4*pow(x(2), 3); }
}

Mesh create2x2MeshWithBoundaryAttributes()
{
   Mesh mesh = Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL, 0, 1.0, 1.0);
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      Element *el = mesh.GetElement(i);
      Array<int> vertices;
      el->GetVertices(vertices);

      real_t xavg = 0.0;
      for (int j = 0; j < vertices.Size(); j++)
      {
         xavg += mesh.GetVertex(vertices[j])[0];
      }
      xavg /= vertices.Size();
      el->SetAttribute((xavg < 0.5) ? 1 : 2);
   }

   mesh.bdr_attributes.SetSize(mesh.GetNBE());
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      mesh.GetBdrElement(i)->SetAttribute(i + 1);
      mesh.bdr_attributes[i] = i + 1;
   }
   mesh.SetAttributes(true, true);
   return mesh;
}

TEST_CASE("MixedBilinearFormSubMeshTrial", "[FormRectangularSystemMatrix]")
{
   Mesh mesh = create2x2MeshWithBoundaryAttributes();

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;
   auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

   const int dim = mesh.Dimension();
   const int order = 2;

   H1_FECollection fec(order, dim);
   FiniteElementSpace submesh_fes(&submesh, &fec);
   FiniteElementSpace trial_fes(&mesh, &fec);
   FiniteElementSpace test_fes(&mesh, &fec);

   FunctionCoefficient xcoeff([](const Vector &x)
   {
      return 1.0 + 2.0 * x(0) - x(1);
   });

   GridFunction submesh_trial_gf(&submesh_fes);
   submesh_trial_gf.ProjectCoefficient(xcoeff);

   GridFunction parent_trial_gf(&trial_fes);
   parent_trial_gf = 0.0;
   SubMesh::Transfer(submesh_trial_gf, parent_trial_gf);

   ConstantCoefficient one(1.0);

   MixedBilinearForm submesh_trial_form(&submesh_fes, &test_fes);
   submesh_trial_form.AddDomainIntegrator(new MassIntegrator(one));
   submesh_trial_form.Assemble();
   submesh_trial_form.Finalize();

   MixedBilinearForm submesh_test_form(&trial_fes, &submesh_fes);
   submesh_test_form.AddDomainIntegrator(new MassIntegrator(one));
   submesh_test_form.Assemble();
   submesh_test_form.Finalize();

   Array<int> attr_marker(mesh.attributes.Max());
   attr_marker = 0;
   attr_marker[1] = 1;

   MixedBilinearForm reference_form(&trial_fes, &test_fes);
   reference_form.AddDomainIntegrator(new MassIntegrator(one), attr_marker);
   reference_form.Assemble();
   reference_form.Finalize();

   Vector submesh_trial_result(test_fes.GetVSize());
   Vector submesh_test_result(test_fes.GetVSize());
   Vector reference_result(test_fes.GetVSize());

   submesh_trial_form.Mult(submesh_trial_gf, submesh_trial_result);
   submesh_test_form.MultTranspose(submesh_trial_gf, submesh_test_result);
   reference_form.Mult(parent_trial_gf, reference_result);

   REQUIRE(submesh_trial_result.Norml2() > 1e-3);
   REQUIRE(submesh_test_result.Norml2() > 1e-3);
   submesh_trial_result -= reference_result;
   submesh_test_result -= reference_result;
   REQUIRE(submesh_trial_result.Norml2() == MFEM_Approx(0.0));
   REQUIRE(submesh_test_result.Norml2() == MFEM_Approx(0.0));
}

TEST_CASE("MixedBilinearFormSubMeshTrialBoundary",
          "[FormRectangularSystemMatrix]")
{
   Mesh mesh = create2x2MeshWithBoundaryAttributes();

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;
   auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

   const int dim = mesh.Dimension();
   const int order = 2;

   H1_FECollection fec(order, dim);
   FiniteElementSpace submesh_fes(&submesh, &fec);
   FiniteElementSpace trial_fes(&mesh, &fec);
   FiniteElementSpace test_fes(&mesh, &fec);

   FunctionCoefficient xcoeff([](const Vector &x)
   {
      return 1.0 + 2.0 * x(0) - x(1);
   });

   GridFunction submesh_trial_gf(&submesh_fes);
   submesh_trial_gf.ProjectCoefficient(xcoeff);

   GridFunction parent_trial_gf(&trial_fes);
   parent_trial_gf = 0.0;
   SubMesh::Transfer(submesh_trial_gf, parent_trial_gf);

   ConstantCoefficient one(1.0);

   Array<int> ref_bdr_marker(mesh.bdr_attributes.Max());
   ref_bdr_marker = 0;
   ref_bdr_marker[1] = 1;

   MixedBilinearForm submesh_trial_form(&submesh_fes, &test_fes);
   submesh_trial_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(one),
                                            ref_bdr_marker);
   submesh_trial_form.Assemble();
   submesh_trial_form.Finalize();

   MixedBilinearForm submesh_test_form(&trial_fes, &submesh_fes);
   submesh_test_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(one),
                                           ref_bdr_marker);
   submesh_test_form.Assemble();
   submesh_test_form.Finalize();

   MixedBilinearForm reference_form(&trial_fes, &test_fes);
   reference_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(one),
                                        ref_bdr_marker);
   reference_form.Assemble();
   reference_form.Finalize();

   Vector submesh_trial_result(test_fes.GetVSize());
   Vector submesh_test_result(test_fes.GetVSize());
   Vector reference_result(test_fes.GetVSize());

   submesh_trial_form.Mult(submesh_trial_gf, submesh_trial_result);
   submesh_test_form.MultTranspose(submesh_trial_gf, submesh_test_result);
   reference_form.Mult(parent_trial_gf, reference_result);

   REQUIRE(submesh_trial_result.Norml2() > 1e-3);
   REQUIRE(submesh_test_result.Norml2() > 1e-3);
   submesh_trial_result -= reference_result;
   submesh_test_result -= reference_result;
   REQUIRE(submesh_trial_result.Norml2() == MFEM_Approx(0.0));
   REQUIRE(submesh_test_result.Norml2() == MFEM_Approx(0.0));
}

// Cartesian mesh of the unit square or cube, with attribute 1 for x < 0.5 and
// attribute 2 otherwise.
Mesh createHalvedMesh(int dim)
{
   Mesh mesh = (dim == 2) ?
               Mesh::MakeCartesian2D(4, 4, Element::QUADRILATERAL) :
               Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON);
   Vector center;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      mesh.GetElementCenter(i, center);
      mesh.SetAttribute(i, (center(0) < 0.5) ? 1 : 2);
   }
   mesh.SetAttributes();
   return mesh;
}

// Mixed integrators supporting partial assembly, with their trial and test
// spaces.
enum class SubMeshPACase { H1_ND_GRADIENT, ND_ND_MASS, RT_L2_DIVERGENCE };

void makeSubMeshPACollections(SubMeshPACase c, int dim,
                              std::unique_ptr<FiniteElementCollection> &trial,
                              std::unique_ptr<FiniteElementCollection> &test)
{
   const int order = 2;
   switch (c)
   {
      case SubMeshPACase::H1_ND_GRADIENT:
         trial.reset(new H1_FECollection(order, dim));
         test.reset(new ND_FECollection(order, dim));
         break;
      case SubMeshPACase::ND_ND_MASS:
         trial.reset(new ND_FECollection(order, dim));
         test.reset(new ND_FECollection(order, dim));
         break;
      case SubMeshPACase::RT_L2_DIVERGENCE:
         trial.reset(new RT_FECollection(order - 1, dim));
         test.reset(new L2_FECollection(order - 1, dim));
         break;
   }
}

BilinearFormIntegrator *newSubMeshPAIntegrator(SubMeshPACase c, int dim,
                                               Coefficient &q)
{
   BilinearFormIntegrator *integ = nullptr;
   switch (c)
   {
      case SubMeshPACase::H1_ND_GRADIENT:
         integ = new MixedVectorGradientIntegrator(q);
         break;
      case SubMeshPACase::ND_ND_MASS:
         integ = new VectorFEMassIntegrator(q);
         break;
      case SubMeshPACase::RT_L2_DIVERGENCE:
         integ = new VectorFEDivergenceIntegrator(q);
         break;
   }
   // Partial and full assembly may choose different default quadrature
   // rules, which matters for a non-constant coefficient.
   const Geometry::Type geom = (dim == 2) ? Geometry::SQUARE : Geometry::CUBE;
   integ->SetIntRule(&IntRules.Get(geom, 5));
   return integ;
}

real_t subMeshPACoefficient(const Vector &x)
{
   return 1.0 + x(0) + 0.5 * x(1) * x(1);
}

TEST_CASE("MixedBilinearFormSubMeshPA",
          "[FormRectangularSystemMatrix], [PartialAssembly]")
{
   const int dim = GENERATE(2, 3);
   const bool trial_on_submesh = GENERATE(true, false);
   const auto c = GENERATE(SubMeshPACase::H1_ND_GRADIENT,
                           SubMeshPACase::ND_ND_MASS,
                           SubMeshPACase::RT_L2_DIVERGENCE);
   CAPTURE(dim, trial_on_submesh, static_cast<int>(c));

   Mesh mesh = createHalvedMesh(dim);
   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;
   auto submesh = SubMesh::CreateFromDomain(mesh, subdomain_attributes);

   std::unique_ptr<FiniteElementCollection> trial_fec, test_fec;
   makeSubMeshPACollections(c, dim, trial_fec, test_fec);
   FiniteElementSpace trial_fes(trial_on_submesh ? &submesh : &mesh,
                                trial_fec.get());
   FiniteElementSpace test_fes(trial_on_submesh ? &mesh : &submesh,
                               test_fec.get());

   FunctionCoefficient q(subMeshPACoefficient);

   MixedBilinearForm fa_form(&trial_fes, &test_fes);
   fa_form.AddDomainIntegrator(newSubMeshPAIntegrator(c, dim, q));
   fa_form.Assemble();
   fa_form.Finalize();

   MixedBilinearForm pa_form(&trial_fes, &test_fes);
   pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_form.AddDomainIntegrator(newSubMeshPAIntegrator(c, dim, q));
   pa_form.Assemble();

   Vector x(trial_fes.GetVSize());
   Vector y_fa(test_fes.GetVSize()), y_pa(test_fes.GetVSize());
   x.Randomize(1);
   fa_form.Mult(x, y_fa);
   pa_form.Mult(x, y_pa);
   REQUIRE(y_fa.Norml2() > 1e-3);
   y_pa -= y_fa;
   REQUIRE(y_pa.Norml2() == MFEM_Approx(0.0));

   Vector xt(test_fes.GetVSize());
   Vector yt_fa(trial_fes.GetVSize()), yt_pa(trial_fes.GetVSize());
   xt.Randomize(2);
   fa_form.MultTranspose(xt, yt_fa);
   pa_form.MultTranspose(xt, yt_pa);
   REQUIRE(yt_fa.Norml2() > 1e-3);
   yt_pa -= yt_fa;
   REQUIRE(yt_pa.Norml2() == MFEM_Approx(0.0));

   if (c == SubMeshPACase::RT_L2_DIVERGENCE)
   {
      // Only VectorFEDivergenceIntegrator implements AssembleDiagonalPA_ADAt.
      // On this conforming mesh the true dofs coincide with the vdofs, and
      // with an L2 test space diag(A D A^T)_i = sum_j A_ij^2 D_j exactly.
      Vector D(trial_fes.GetTrueVSize());
      D.Randomize(3);
      Vector diag_pa(test_fes.GetTrueVSize());
      pa_form.AssembleDiagonal_ADAt(D, diag_pa);

      const SparseMatrix &A = fa_form.SpMat();
      Vector diag_fa(A.Height());
      diag_fa = 0.0;
      for (int i = 0; i < A.Height(); i++)
      {
         const int *cols = A.GetRowColumns(i);
         const real_t *vals = A.GetRowEntries(i);
         for (int k = 0; k < A.RowSize(i); k++)
         {
            diag_fa(i) += vals[k] * vals[k] * D(cols[k]);
         }
      }
      REQUIRE(diag_fa.Norml2() > 1e-3);
      diag_pa -= diag_fa;
      REQUIRE(diag_pa.Norml2() == MFEM_Approx(0.0));
   }
}

TEST_CASE("FormRectangular", "[FormRectangularSystemMatrix]")
{
   enum FECType { H1, L2_VALUE, L2_INTEGRAL };
   auto fec_type = GENERATE(FECType::H1, FECType::L2_VALUE,
                            FECType::L2_INTEGRAL);

   SECTION("MixedBilinearForm::FormRectangularSystemMatrix")
   {
      Mesh mesh = Mesh::MakeCartesian2D(
                     10, 10, Element::QUADRILATERAL, 0, 1.0, 1.0);
      int dim = mesh.Dimension();
      int order = 4;

      int nattr = mesh.bdr_attributes.Max();
      Array<int> ess_trial_tdof_list, ess_test_tdof_list;

      Array<int> ess_bdr(nattr);
      ess_bdr = 0;
      ess_bdr[0] = 1;

      // Scalar
      H1_FECollection fec1(order, dim);
      FiniteElementSpace fes1(&mesh, &fec1);
      fes1.GetEssentialTrueDofs(ess_bdr, ess_trial_tdof_list);
      GridFunction field(&fes1);

      // Vector valued
      std::unique_ptr<FiniteElementCollection> fec2;
      switch (fec_type)
      {
         case FECType::H1:
            fec2.reset(new H1_FECollection(order, dim));
            break;
         case FECType::L2_VALUE:
            fec2.reset(new L2_FECollection(order, dim, BasisType::GaussLegendre,
                                           FiniteElement::VALUE));
            break;
         case FECType::L2_INTEGRAL:
            fec2.reset(new L2_FECollection(order, dim, BasisType::GaussLegendre,
                                           FiniteElement::INTEGRAL));
            break;
      }
      FiniteElementSpace fes2(&mesh, fec2.get(), dim);
      fes2.GetEssentialTrueDofs(ess_bdr, ess_test_tdof_list);
      GridFunction field2(&fes2);

      MixedBilinearForm gform(&fes1, &fes2);
      gform.AddDomainIntegrator(new GradientIntegrator);
      gform.Assemble();

      // Project u = f1
      FunctionCoefficient fcoeff1(f1);
      field.ProjectCoefficient(fcoeff1);

      VectorFunctionCoefficient fcoeff2(dim, gradf1);
      LinearForm lf(&fes2);
      lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff2));
      lf.Assemble();

      OperatorHandle G;
      Vector X, B;
      gform.FormRectangularLinearSystem(ess_trial_tdof_list,
                                        ess_test_tdof_list,
                                        field,
                                        lf,
                                        G,
                                        X,
                                        B);

      G->Mult(field, field2);

      subtract(B, field2, field2);
      REQUIRE(field2.Norml2() == MFEM_Approx(0.0));
   }
}

#ifdef MFEM_USE_MPI

TEST_CASE("ParMixedBilinearFormSubMeshTrial",
          "[Parallel], [FormRectangularSystemMatrix]")
{
   Mesh mesh = create2x2MeshWithBoundaryAttributes();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;
   auto submesh = ParSubMesh::CreateFromDomain(pmesh,
                                               subdomain_attributes);

   const int dim = pmesh.Dimension();
   const int order = 2;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace submesh_fes(&submesh, &fec);
   ParFiniteElementSpace trial_fes(&pmesh, &fec);
   ParFiniteElementSpace test_fes(&pmesh, &fec);

   FunctionCoefficient xcoeff([](const Vector &x)
   {
      return 1.0 + 2.0 * x(0) - x(1);
   });

   ParGridFunction submesh_trial_gf(&submesh_fes);
   submesh_trial_gf.ProjectCoefficient(xcoeff);

   ParGridFunction parent_trial_gf(&trial_fes);
   parent_trial_gf = 0.0;
   ParSubMesh::Transfer(submesh_trial_gf, parent_trial_gf);

   ConstantCoefficient one(1.0);

   ParMixedBilinearForm submesh_trial_form(&submesh_fes, &test_fes);
   submesh_trial_form.AddDomainIntegrator(new MassIntegrator(one));
   submesh_trial_form.Assemble();
   submesh_trial_form.Finalize();
   std::unique_ptr<HypreParMatrix> submesh_trial_mat(
      submesh_trial_form.ParallelAssemble());

   ParMixedBilinearForm submesh_test_form(&trial_fes, &submesh_fes);
   submesh_test_form.AddDomainIntegrator(new MassIntegrator(one));
   submesh_test_form.Assemble();
   submesh_test_form.Finalize();
   std::unique_ptr<HypreParMatrix> submesh_test_mat(
      submesh_test_form.ParallelAssemble());

   Array<int> attr_marker(mesh.attributes.Max());
   attr_marker = 0;
   attr_marker[1] = 1;

   ParMixedBilinearForm reference_form(&trial_fes, &test_fes);
   reference_form.AddDomainIntegrator(new MassIntegrator(one), attr_marker);
   reference_form.Assemble();
   reference_form.Finalize();
   std::unique_ptr<HypreParMatrix> reference_mat(
      reference_form.ParallelAssemble());

   std::unique_ptr<HypreParVector> trial_true(submesh_trial_gf.ParallelProject());
   std::unique_ptr<HypreParVector> parent_trial_true(
      parent_trial_gf.ParallelProject());
   std::unique_ptr<HypreParVector> submesh_trial_result(
      test_fes.NewTrueDofVector());
   std::unique_ptr<HypreParVector> submesh_test_result(
      test_fes.NewTrueDofVector());
   std::unique_ptr<HypreParVector> reference_result(test_fes.NewTrueDofVector());

   submesh_trial_mat->Mult(*trial_true, *submesh_trial_result);
   submesh_test_mat->MultTranspose(*trial_true, *submesh_test_result);
   reference_mat->Mult(*parent_trial_true, *reference_result);

   *submesh_trial_result -= *reference_result;
   *submesh_test_result -= *reference_result;
   REQUIRE(submesh_trial_result->Norml2() == MFEM_Approx(0.0));
   REQUIRE(submesh_test_result->Norml2() == MFEM_Approx(0.0));
}

TEST_CASE("ParMixedBilinearFormSubMeshTrialBoundary",
          "[Parallel], [FormRectangularSystemMatrix]")
{
   Mesh mesh = create2x2MeshWithBoundaryAttributes();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);

   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;
   auto submesh = ParSubMesh::CreateFromDomain(pmesh,
                                               subdomain_attributes);

   const int dim = pmesh.Dimension();
   const int order = 2;

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace submesh_fes(&submesh, &fec);
   ParFiniteElementSpace trial_fes(&pmesh, &fec);
   ParFiniteElementSpace test_fes(&pmesh, &fec);

   FunctionCoefficient xcoeff([](const Vector &x)
   {
      return 1.0 + 2.0 * x(0) - x(1);
   });

   ParGridFunction submesh_trial_gf(&submesh_fes);
   submesh_trial_gf.ProjectCoefficient(xcoeff);

   ParGridFunction parent_trial_gf(&trial_fes);
   parent_trial_gf = 0.0;
   ParSubMesh::Transfer(submesh_trial_gf, parent_trial_gf);

   ConstantCoefficient one(1.0);

   Array<int> ref_bdr_marker(mesh.bdr_attributes.Max());
   ref_bdr_marker = 0;
   ref_bdr_marker[1] = 1;

   ParMixedBilinearForm submesh_trial_form(&submesh_fes, &test_fes);
   submesh_trial_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(one),
                                            ref_bdr_marker);
   submesh_trial_form.Assemble();
   submesh_trial_form.Finalize();
   std::unique_ptr<HypreParMatrix> submesh_trial_mat(
      submesh_trial_form.ParallelAssemble());

   ParMixedBilinearForm submesh_test_form(&trial_fes, &submesh_fes);
   submesh_test_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(one),
                                           ref_bdr_marker);
   submesh_test_form.Assemble();
   submesh_test_form.Finalize();
   std::unique_ptr<HypreParMatrix> submesh_test_mat(
      submesh_test_form.ParallelAssemble());

   ParMixedBilinearForm reference_form(&trial_fes, &test_fes);
   reference_form.AddBoundaryIntegrator(new BoundaryMassIntegrator(one),
                                        ref_bdr_marker);
   reference_form.Assemble();
   reference_form.Finalize();
   std::unique_ptr<HypreParMatrix> reference_mat(
      reference_form.ParallelAssemble());

   std::unique_ptr<HypreParVector> trial_true(submesh_trial_gf.ParallelProject());
   std::unique_ptr<HypreParVector> parent_trial_true(
      parent_trial_gf.ParallelProject());
   std::unique_ptr<HypreParVector> submesh_trial_result(
      test_fes.NewTrueDofVector());
   std::unique_ptr<HypreParVector> submesh_test_result(
      test_fes.NewTrueDofVector());
   std::unique_ptr<HypreParVector> reference_result(test_fes.NewTrueDofVector());

   submesh_trial_mat->Mult(*trial_true, *submesh_trial_result);
   submesh_test_mat->MultTranspose(*trial_true, *submesh_test_result);
   reference_mat->Mult(*parent_trial_true, *reference_result);

   *submesh_trial_result -= *reference_result;
   *submesh_test_result -= *reference_result;
   REQUIRE(submesh_trial_result->Norml2() == MFEM_Approx(0.0));
   REQUIRE(submesh_test_result->Norml2() == MFEM_Approx(0.0));
}

TEST_CASE("ParMixedBilinearFormSubMeshPA",
          "[Parallel], [FormRectangularSystemMatrix], [PartialAssembly]")
{
   const int dim = GENERATE(2, 3);
   const bool trial_on_submesh = GENERATE(true, false);
   const auto c = GENERATE(SubMeshPACase::H1_ND_GRADIENT,
                           SubMeshPACase::ND_ND_MASS,
                           SubMeshPACase::RT_L2_DIVERGENCE);
   CAPTURE(dim, trial_on_submesh, static_cast<int>(c));

   Mesh mesh = createHalvedMesh(dim);
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   Array<int> subdomain_attributes(1);
   subdomain_attributes[0] = 2;
   auto submesh = ParSubMesh::CreateFromDomain(pmesh, subdomain_attributes);

   std::unique_ptr<FiniteElementCollection> trial_fec, test_fec;
   makeSubMeshPACollections(c, dim, trial_fec, test_fec);
   ParFiniteElementSpace trial_fes(trial_on_submesh ? &submesh : &pmesh,
                                   trial_fec.get());
   ParFiniteElementSpace test_fes(trial_on_submesh ? &pmesh : &submesh,
                                  test_fec.get());

   FunctionCoefficient q(subMeshPACoefficient);

   ParMixedBilinearForm fa_form(&trial_fes, &test_fes);
   fa_form.AddDomainIntegrator(newSubMeshPAIntegrator(c, dim, q));
   fa_form.Assemble();
   fa_form.Finalize();
   std::unique_ptr<HypreParMatrix> fa_mat(fa_form.ParallelAssemble());

   ParMixedBilinearForm pa_form(&trial_fes, &test_fes);
   pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_form.AddDomainIntegrator(newSubMeshPAIntegrator(c, dim, q));
   pa_form.Assemble();
   Array<int> empty;
   OperatorHandle pa_op;
   pa_form.FormRectangularSystemMatrix(empty, empty, pa_op);

   auto GlobalNorm = [](const Vector &v)
   {
      return std::sqrt(InnerProduct(MPI_COMM_WORLD, v, v));
   };

   Vector x(trial_fes.GetTrueVSize());
   Vector y_fa(test_fes.GetTrueVSize()), y_pa(test_fes.GetTrueVSize());
   x.Randomize(1);
   fa_mat->Mult(x, y_fa);
   pa_op->Mult(x, y_pa);
   REQUIRE(GlobalNorm(y_fa) > 1e-3);
   y_pa -= y_fa;
   REQUIRE(GlobalNorm(y_pa) == MFEM_Approx(0.0));

   Vector xt(test_fes.GetTrueVSize());
   Vector yt_fa(trial_fes.GetTrueVSize()), yt_pa(trial_fes.GetTrueVSize());
   xt.Randomize(2);
   fa_mat->MultTranspose(xt, yt_fa);
   pa_op->MultTranspose(xt, yt_pa);
   REQUIRE(GlobalNorm(yt_fa) > 1e-3);
   yt_pa -= yt_fa;
   REQUIRE(GlobalNorm(yt_pa) == MFEM_Approx(0.0));
}

TEST_CASE("ParallelFormRectangular",
          "[Parallel], [FormRectangularSystemMatrix]")
{
   SECTION("ParMixedBilinearForm::FormRectangularSystemMatrix")
   {
      Mesh mesh = Mesh::MakeCartesian2D(
                     10, 10, Element::QUADRILATERAL, 0, 1.0, 1.0);
      int dim = mesh.Dimension();
      int order = 4;

      int nattr = mesh.bdr_attributes.Max();
      Array<int> ess_trial_tdof_list, ess_test_tdof_list;

      Array<int> ess_bdr(nattr);
      ess_bdr = 0;
      ess_bdr[0] = 1;

      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      // Scalar
      H1_FECollection fec1(order, dim);
      ParFiniteElementSpace fes1(&pmesh, &fec1);

      // Vector valued
      H1_FECollection fec2(order, dim);
      ParFiniteElementSpace fes2(&pmesh, &fec2, dim);

      fes1.GetEssentialTrueDofs(ess_bdr, ess_trial_tdof_list);
      fes2.GetEssentialTrueDofs(ess_bdr, ess_test_tdof_list);

      ParGridFunction field(&fes1), field2(&fes2);

      ParMixedBilinearForm gform(&fes1, &fes2);
      gform.AddDomainIntegrator(new GradientIntegrator);
      gform.Assemble();

      // Project u = f1
      FunctionCoefficient fcoeff1(f1);
      field.ProjectCoefficient(fcoeff1);

      VectorFunctionCoefficient fcoeff2(dim, gradf1);
      ParLinearForm lf(&fes2);
      lf.AddDomainIntegrator(new VectorDomainLFIntegrator(fcoeff2));
      lf.Assemble();

      OperatorHandle G;
      Vector X, B;
      gform.FormRectangularLinearSystem(ess_trial_tdof_list,
                                        ess_test_tdof_list,
                                        field,
                                        lf,
                                        G,
                                        X,
                                        B);

      Vector *field_tdof = field.ParallelProject();
      Vector *field2_tdof = field2.ParallelProject();

      G->Mult(*field_tdof, *field2_tdof);

      subtract(B, *field2_tdof, *field2_tdof);
      REQUIRE(field2_tdof->Norml2() == MFEM_Approx(0.0));

      delete field2_tdof;
      delete field_tdof;
   }
}

TEST_CASE("HypreParMatrixBlocksRectangular",
          "[Parallel], [BlockMatrix]")
{
   SECTION("HypreParMatrixFromBlocks")
   {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      Mesh mesh = Mesh::MakeCartesian2D(
                     10, 10, Element::QUADRILATERAL, 0, 1.0, 1.0);
      int dim = mesh.Dimension();
      int order = 2;

      int nattr = mesh.bdr_attributes.Max();
      Array<int> ess_trial_tdof_list, ess_test_tdof_list;

      Array<int> ess_bdr(nattr);
      ess_bdr = 0;
      ess_bdr[0] = 1;

      ParMesh pmesh(MPI_COMM_WORLD, mesh);

      FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
      FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

      ParFiniteElementSpace R_space(&pmesh, hdiv_coll);
      ParFiniteElementSpace W_space(&pmesh, l2_coll);

      ParBilinearForm RmVarf(&R_space);
      ParBilinearForm WmVarf(&W_space);
      ParMixedBilinearForm bVarf(&R_space, &W_space);

      HypreParMatrix *MR, *MW, *B;

      RmVarf.AddDomainIntegrator(new VectorFEMassIntegrator());
      RmVarf.Assemble();
      RmVarf.Finalize();
      MR = RmVarf.ParallelAssemble();

      WmVarf.AddDomainIntegrator(new MassIntegrator());
      WmVarf.Assemble();
      WmVarf.Finalize();
      MW = WmVarf.ParallelAssemble();

      bVarf.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
      bVarf.Assemble();
      bVarf.Finalize();
      B = bVarf.ParallelAssemble();
      (*B) *= -1;

      HypreParMatrix *BT = B->Transpose();

      Array<int> blockRow_trueOffsets(3); // number of variables + 1
      blockRow_trueOffsets[0] = 0;
      blockRow_trueOffsets[1] = R_space.TrueVSize();
      blockRow_trueOffsets[2] = W_space.TrueVSize();
      blockRow_trueOffsets.PartialSum();

      Array<int> blockCol_trueOffsets(4); // number of variables + 1
      blockCol_trueOffsets[0] = 0;
      blockCol_trueOffsets[1] = R_space.TrueVSize();
      blockCol_trueOffsets[2] = W_space.TrueVSize();
      blockCol_trueOffsets[3] = W_space.TrueVSize();
      blockCol_trueOffsets.PartialSum();

      BlockOperator blockOper(blockRow_trueOffsets, blockCol_trueOffsets);
      blockOper.SetBlock(0, 0, MR);
      blockOper.SetBlock(0, 1, BT);
      blockOper.SetBlock(1, 0, B);
      blockOper.SetBlock(0, 2, BT, 3.14);
      blockOper.SetBlock(1, 2, MW);

      Array2D<const HypreParMatrix*> hBlocks(2,3);
      hBlocks = NULL;
      hBlocks(0, 0) = MR;
      hBlocks(0, 1) = BT;
      hBlocks(1, 0) = B;
      hBlocks(0, 2) = BT;
      hBlocks(1, 2) = MW;

      Array2D<real_t> blockCoeff(2,3);
      blockCoeff = 1.0;
      blockCoeff(0, 2) = 3.14;
      HypreParMatrix *H = HypreParMatrixFromBlocks(hBlocks, &blockCoeff);

      Vector x(blockCol_trueOffsets[3]);
      Vector yB(blockRow_trueOffsets[2]);
      Vector yH(blockRow_trueOffsets[2]);

      x.Randomize();
      yB = 0.0;
      yH = 0.0;

      blockOper.Mult(x, yB);
      H->Mult(x, yH);

      yH -= yB;
      real_t error = yH.Norml2();
      mfem::out << "  order: " << order
                << ", block matrix error norm on rank " << rank << ": " << error << std::endl;
      REQUIRE(error < 1.e-12);

      delete H;
      delete BT;
      delete B;
      delete MW;
      delete MR;
      delete l2_coll;
      delete hdiv_coll;
   }
}

#endif

} // namespace mfem
