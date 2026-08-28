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

using namespace mfem;

namespace elasticity_pa
{

constexpr real_t q_lambda = 2.3;
constexpr real_t q_mu = 0.7;

real_t MaterialFunction(const Vector &x)
{
   real_t value =
      1.45 + 0.19 * sin(2.0 * M_PI * x[0])
      + 0.13 * cos(3.0 * M_PI * x[1]);
   if (x.Size() == 3)
   {
      value += 0.08 * x[2] + 0.05 * sin(M_PI * x[2]);
   }
   return value;
}

MFEM_HOST_DEVICE inline int ComponentVDof(const int dof,
                                          const int component,
                                          const int ndofs,
                                          const int vdim,
                                          const int bynodes)
{
   return vdim == 1 ? dof
          : (bynodes ? dof + ndofs * component : component + vdim * dof);
}

void FillVector(Vector &v, const int seed)
{
   const int n = v.Size();
   const real_t s = seed + 1;
   auto *data = v.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t k = i + 1;
      data[i] = sin((0.31 + 0.017 * s) * k)
                + 0.23 * cos((0.13 + 0.011 * s) * k) + 0.001 * s;
   });
}

void ScatterScalarComponent(const Vector &scalar,
                            Vector &vector,
                            const FiniteElementSpace &vector_fes,
                            const int component)
{
   const int n = scalar.Size();
   const int ndofs = vector_fes.GetNDofs();
   const int vdim = vector_fes.GetVDim();
   const int bynodes = vector_fes.GetOrdering() == Ordering::byNODES;
   vector = 0.0;
   const auto *src = scalar.Read();
   auto *dest = vector.ReadWrite();
   mfem::forall(n, [=] MFEM_HOST_DEVICE(int k)
   {
      dest[ComponentVDof(k, component, ndofs, vdim, bynodes)] = src[k];
   });
}

void GatherScalarComponent(const Vector &vector,
                           Vector &scalar,
                           const FiniteElementSpace &vector_fes,
                           const int component)
{
   const int n = scalar.Size();
   const int ndofs = vector_fes.GetNDofs();
   const int vdim = vector_fes.GetVDim();
   const int bynodes = vector_fes.GetOrdering() == Ordering::byNODES;
   const auto *src = vector.Read();
   auto *dest = scalar.Write();
   mfem::forall(n, [=] MFEM_HOST_DEVICE(int k)
   {
      dest[k] = src[ComponentVDof(k, component, ndofs, vdim, bynodes)];
   });
}

void RequireClose(const Vector &actual, const Vector &expected)
{
   Vector difference(actual);
   difference -= expected;
   REQUIRE(difference.Norml2() == MFEM_Approx(0.0));
}

Mesh MakeMesh(const int dim)
{
   constexpr int n = 2;
   if (dim == 2)
   {
      return Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, 1, 1.2, 0.9);
   }
   return Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON, 1.2, 0.9, 1.1);
}

void AddElasticityAndMass(BilinearForm &form,
                          Coefficient &material,
                          Coefficient &mass,
                          const bool elasticity_first)
{
   if (elasticity_first)
   {
      form.AddDomainIntegrator(
         new ElasticityIntegrator(material, q_lambda, q_mu));
      form.AddDomainIntegrator(new VectorMassIntegrator(mass));
   }
   else
   {
      form.AddDomainIntegrator(new VectorMassIntegrator(mass));
      form.AddDomainIntegrator(
         new ElasticityIntegrator(material, q_lambda, q_mu));
   }
}

TEST_CASE("Elasticity PA action and diagonal",
          "[PartialAssembly][ElasticityPA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(dim, order);

   Mesh mesh = MakeMesh(dim);
   H1_FECollection fec(order, dim);
   FiniteElementSpace vector_fes(&mesh, &fec, dim, Ordering::byNODES);
   FunctionCoefficient material(MaterialFunction);

   BilinearForm pa_form(&vector_fes);
   pa_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   pa_form.Assemble();

   BilinearForm full_form(&vector_fes);
   full_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   full_form.Assemble();
   full_form.Finalize();

   Vector x(vector_fes.GetVSize());
   Vector y_pa(vector_fes.GetVSize());
   Vector y_full(vector_fes.GetVSize());
   FillVector(x, 1);
   y_pa = 0.0;
   y_full = 0.0;
   pa_form.Mult(x, y_pa);
   full_form.Mult(x, y_full);
   RequireClose(y_pa, y_full);

   Vector diagonal_pa(vector_fes.GetVSize());
   Vector diagonal_full(vector_fes.GetVSize());
   pa_form.AssembleDiagonal(diagonal_pa);
   full_form.SpMat().GetDiag(diagonal_full);
   RequireClose(diagonal_pa, diagonal_full);
}

TEST_CASE("Elasticity PA diagonal accumulation",
          "[PartialAssembly][ElasticityPA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(dim, order);

   Mesh mesh = MakeMesh(dim);
   H1_FECollection fec(order, dim);
   FiniteElementSpace vector_fes(&mesh, &fec, dim, Ordering::byNODES);
   FunctionCoefficient material(MaterialFunction);

   ElasticityIntegrator integrator(material, q_lambda, q_mu);
   integrator.AssemblePA(vector_fes);

   const int scalar_dofs = vector_fes.GetTypicalFE()->GetDof();
   const int e_size = scalar_dofs * dim * vector_fes.GetNE();

   Vector contribution(e_size);
   contribution = 0.0;
   integrator.AssembleDiagonalPA(contribution);

   Vector accumulated(e_size);
   FillVector(accumulated, 3);
   Vector expected(accumulated);
   expected += contribution;
   integrator.AssembleDiagonalPA(accumulated);

   RequireClose(accumulated, expected);
}

TEST_CASE("Elasticity PA multi-integrator diagonal",
          "[PartialAssembly][ElasticityPA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(dim, order);

   Mesh mesh = MakeMesh(dim);
   H1_FECollection fec(order, dim);
   FiniteElementSpace vector_fes(&mesh, &fec, dim, Ordering::byNODES);
   FunctionCoefficient material(MaterialFunction);
   ConstantCoefficient mass_coefficient(0.37);

   BilinearForm full_form(&vector_fes);
   AddElasticityAndMass(full_form, material, mass_coefficient, true);
   full_form.Assemble();
   full_form.Finalize();
   Vector diagonal_full(vector_fes.GetVSize());
   full_form.SpMat().GetDiag(diagonal_full);

   BilinearForm pa_elasticity_first(&vector_fes);
   pa_elasticity_first.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   AddElasticityAndMass(pa_elasticity_first, material, mass_coefficient, true);
   pa_elasticity_first.Assemble();
   Vector diagonal_elasticity_first(vector_fes.GetVSize());
   pa_elasticity_first.AssembleDiagonal(diagonal_elasticity_first);

   BilinearForm pa_mass_first(&vector_fes);
   pa_mass_first.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   AddElasticityAndMass(pa_mass_first, material, mass_coefficient, false);
   pa_mass_first.Assemble();
   Vector diagonal_mass_first(vector_fes.GetVSize());
   pa_mass_first.AssembleDiagonal(diagonal_mass_first);

   RequireClose(diagonal_elasticity_first, diagonal_full);
   RequireClose(diagonal_mass_first, diagonal_full);
   RequireClose(diagonal_mass_first, diagonal_elasticity_first);
}

TEST_CASE("Elasticity component PA", "[PartialAssembly][ElasticityPA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(dim, order);

   Mesh mesh = MakeMesh(dim);
   H1_FECollection fec(order, dim);
   FiniteElementSpace scalar_fes(&mesh, &fec, 1, Ordering::byNODES);
   FiniteElementSpace vector_fes(&mesh, &fec, dim, Ordering::byNODES);
   FunctionCoefficient material(MaterialFunction);

   BilinearForm full_form(&vector_fes);
   full_form.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   full_form.Assemble();
   full_form.Finalize();

   // Deliberately do not call parent.AssemblePA(vector_fes). The first
   // component assembly must initialize the shared parent state correctly.
   ElasticityIntegrator parent(material, q_lambda, q_mu);

   Vector x_scalar(scalar_fes.GetVSize());
   Vector x_vector(vector_fes.GetVSize());
   Vector y_vector(vector_fes.GetVSize());
   Vector y_block(scalar_fes.GetVSize());
   Vector y_reference(scalar_fes.GetVSize());

   for (int j = 0; j < dim; ++j)
   {
      FillVector(x_scalar, 10 + j);
      ScatterScalarComponent(x_scalar, x_vector, vector_fes, j);

      y_vector = 0.0;
      full_form.Mult(x_vector, y_vector);

      for (int i = 0; i < dim; ++i)
      {
         CAPTURE(i, j);
         BilinearForm component_form(&scalar_fes);
         component_form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         component_form.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i, j));
         component_form.Assemble();

         y_block = 0.0;
         component_form.Mult(x_scalar, y_block);
         GatherScalarComponent(y_vector, y_reference, vector_fes, i);
         RequireClose(y_block, y_reference);
      }
   }
}

TEST_CASE("Elasticity component EA", "[PartialAssembly][ElasticityPA][GPU]")
{
   const int dim = GENERATE(2, 3);
   const int order = GENERATE(1, 2);
   CAPTURE(dim, order);

   Mesh mesh = MakeMesh(dim);
   H1_FECollection fec(order, dim);
   FiniteElementSpace scalar_fes(&mesh, &fec, 1, Ordering::byNODES);
   FiniteElementSpace vector_fes(&mesh, &fec, dim, Ordering::byNODES);
   FunctionCoefficient material(MaterialFunction);

   BilinearForm full_elasticity(&vector_fes);
   full_elasticity.AddDomainIntegrator(
      new ElasticityIntegrator(material, q_lambda, q_mu));
   full_elasticity.Assemble();
   full_elasticity.Finalize();

   ConstantCoefficient mass_coefficient(0.41);
   BilinearForm full_mass(&scalar_fes);
   full_mass.AddDomainIntegrator(new MassIntegrator(mass_coefficient));
   full_mass.Assemble();
   full_mass.Finalize();

   // Deliberately share one parent across all blocks. The first EA call must
   // initialize it, and later calls must reuse compatible PA data.
   ElasticityIntegrator parent(material, q_lambda, q_mu);

   Vector x_scalar(scalar_fes.GetVSize());
   Vector x_vector(vector_fes.GetVSize());
   Vector y_vector(vector_fes.GetVSize());
   Vector y_block(scalar_fes.GetVSize());
   Vector y_mass(scalar_fes.GetVSize());
   Vector expected_with_mass(scalar_fes.GetVSize());
   Vector y_component_only(scalar_fes.GetVSize());
   Vector y_component_first(scalar_fes.GetVSize());
   Vector y_component_second(scalar_fes.GetVSize());

   for (int j = 0; j < dim; ++j)
   {
      FillVector(x_scalar, 30 + j);
      ScatterScalarComponent(x_scalar, x_vector, vector_fes, j);

      y_vector = 0.0;
      full_elasticity.Mult(x_vector, y_vector);
      full_mass.Mult(x_scalar, y_mass);

      for (int i = 0; i < dim; ++i)
      {
         CAPTURE(i, j);
         GatherScalarComponent(y_vector, y_block, vector_fes, i);
         expected_with_mass = y_block;
         expected_with_mass += y_mass;

         BilinearForm component_only(&scalar_fes);
         component_only.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         component_only.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i, j));
         component_only.Assemble();
         component_only.Mult(x_scalar, y_component_only);
         RequireClose(y_component_only, y_block);

         BilinearForm component_first(&scalar_fes);
         component_first.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         component_first.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i, j));
         component_first.AddDomainIntegrator(
            new MassIntegrator(mass_coefficient));
         component_first.Assemble();
         component_first.Mult(x_scalar, y_component_first);
         RequireClose(y_component_first, expected_with_mass);

         BilinearForm component_second(&scalar_fes);
         component_second.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         component_second.AddDomainIntegrator(
            new MassIntegrator(mass_coefficient));
         component_second.AddDomainIntegrator(
            new ElasticityComponentIntegrator(parent, i, j));
         component_second.Assemble();
         component_second.Mult(x_scalar, y_component_second);
         RequireClose(y_component_second, expected_with_mass);
      }
   }
}

} // namespace elasticity_pa
