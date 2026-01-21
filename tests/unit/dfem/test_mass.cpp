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

#include "../unit_tests.hpp"
#include "../linalg/test_same_matrices.hpp"
#include "mfem.hpp"

#ifdef MFEM_USE_MPI

using namespace mfem;
using namespace mfem::future;
using mfem::future::tensor;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using mfem::future::dual;
using dscalar_t = dual<real_t, real_t>;
#endif

template <int DIM>
void mass_action(const char *filename, int p)
{
   constexpr int BDIM = DIM - 1;
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec);

   ParGridFunction x(&fes), y(&fes), z(&fes);
   Vector X(fes.GetTrueVSize()), Y(fes.GetTrueVSize()), Z(fes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);

   SECTION("domain")
   {
      const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

      Array<int> all_domain_attr;
      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }

      ParBilinearForm blf(&fes);
      blf.AddDomainIntegrator(new MassIntegrator(one, ir));
      blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf.Assemble();
      blf.Mult(x, y);
      fes.GetProlongationMatrix()->MultTranspose(y, Y);

      static constexpr int U = 0, Coords = 1;
      const auto sol = std::vector{ FieldDescriptor{ U, &fes } };
      DifferentiableOperator dop(sol, {{Coords, nodes->ParFESpace()}}, pmesh);
      const auto mf_mass_qf =
         [] MFEM_HOST_DEVICE(const real_t &u,
                             const tensor<real_t, DIM, DIM> &J, const real_t &w)
      { return tuple{u * w * det(J)}; };
      dop.AddDomainIntegrator(mf_mass_qf,
                              tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                              tuple{ Value<U>{} },
                              *ir, all_domain_attr);
      dop.SetParameters({ nodes });

      fes.GetRestrictionMatrix()->Mult(x, X);
      dop.Mult(X, Z);
      Y -= Z;

      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Test boundary
   // This ensures that we're not trying to test on fully periodic meshes
   if (!((std::string("../../data/periodic-square.mesh").compare(filename) == 0) ||
         (std::string("../../data/periodic-cube.mesh").compare(filename) == 0)))
   {
      SECTION("boundary")
      {
         const auto *ir = &IntRules.Get(pmesh.GetTypicalFaceGeometry(), 2 * p);

         Array<int> all_bdr_attr;
         if (pmesh.bdr_attributes.Size() > 0)
         {
            all_bdr_attr.SetSize(pmesh.bdr_attributes.Max());
            all_bdr_attr = 1;
         }

         ParBilinearForm blf(&fes);
         blf.AddBoundaryIntegrator(new MassIntegrator(one, ir));
         blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         blf.Assemble();
         blf.Mult(x, y);
         fes.GetProlongationMatrix()->MultTranspose(y, Y);

         static constexpr int U = 0, Coords = 1;
         const auto sol = std::vector{FieldDescriptor{U, &fes}};
         DifferentiableOperator dop(sol, {{Coords, nodes->ParFESpace()}}, pmesh);

         const auto mf_mass_qf =
            [] MFEM_HOST_DEVICE(const dscalar_t &u,
                                const tensor<real_t, DIM, BDIM> &J,
                                const real_t &w)
         {
            return tuple{u * weight(J) * w};
         };

         auto derivatives = std::integer_sequence<size_t, U> {};
         dop.AddBoundaryIntegrator(mf_mass_qf,
                                   tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                                   tuple{ Value<U>{} },
                                   *ir, all_bdr_attr, derivatives);
         dop.SetParameters({nodes});

         fes.GetRestrictionMatrix()->Mult(x, X);
         dop.Mult(X, Z);

         Y -= Z;
         real_t norm_g, norm_l = Y.Normlinf();
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));

         auto dRdU = dop.GetDerivative(U, {&x}, {nodes});
         dRdU->Mult(X, Z);

         fes.GetProlongationMatrix()->MultTranspose(y, Y);
         Y -= Z;
         norm_l = Y.Normlinf();
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
}

template <int DIM> void mass_mat_mixed(const char* filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec0(p, DIM);
   H1_FECollection fec1(p + 1, DIM);
   ParFiniteElementSpace fes0(&pmesh, &fec0);
   ParFiniteElementSpace fes1(&pmesh, &fec1);

   const auto* ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   ConstantCoefficient one(1.0);
   ParMixedBilinearForm blf(&fes1, &fes0);
   blf.AddDomainIntegrator(new MassIntegrator(one, ir));
   blf.SetAssemblyLevel(AssemblyLevel::FULL);
   blf.Assemble();
   blf.Finalize();

   blf.SpMat().Finalize();

   static constexpr int U = 0, P = 1, Coords = 2;
   const auto sol = std::vector{FieldDescriptor{U, &fes1}};
   DifferentiableOperator dop(sol, {{P, &fes0}, {Coords, nodes->ParFESpace()}},
   pmesh);
   const auto mf_mass_qf = [] MFEM_HOST_DEVICE(
                              const dscalar_t& u,
                              const tensor<real_t, DIM, DIM>& J,
                              const real_t& w)
   {
      return tuple{u * w * det(J)};
   };

   auto derivatives = std::integer_sequence<size_t, U> {};
   dop.AddDomainIntegrator(mf_mass_qf,
                           tuple{Value<U>{}, Gradient<Coords>{}, Weight{}},
                           tuple{Value<P>{}},
                           *ir, all_domain_attr, derivatives);

   ParGridFunction ugf(&fes1);
   ugf = 0.0;

   ParGridFunction pgf(&fes0);
   pgf = 0.0;

   dop.SetParameters({&pgf, nodes});

   auto ddopdu = dop.GetDerivative(U, {&ugf}, {&pgf, nodes});

   SECTION("spmat")
   {
      SparseMatrix *A;
      ddopdu->Assemble(A);
      TestSameMatrices(*A, blf.SpMat());
      delete A;
   }

   SECTION("hypre parallel mat")
   {
      HypreParMatrix *Amfem = blf.ParallelAssemble();

      HypreParMatrix *Adfem;
      ddopdu->Assemble(Adfem);
      TestSameMatrices(*Adfem, *Amfem);
      delete Amfem;
      delete Adfem;
   }
}

// no GPU tag to avoid failing 'hypre parallel mat' section
TEST_CASE("dFEM Mass", "[Parallel][dFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);

   SECTION("2d")
   {
      const auto filename2d =
         GENERATE(
            "../../data/star.mesh",
            "../../data/star-q3.mesh",
            "../../data/rt-2d-q3.mesh",
            "../../data/inline-quad.mesh",
            "../../data/periodic-square.mesh"
         );
      mass_action<2>(filename2d, p);
      mass_mat_mixed<2>(filename2d, p);
   }

   SECTION("3d")
   {
      const auto filename3d =
         GENERATE(
            "../../data/fichera.mesh",
            "../../data/fichera-q3.mesh",
            "../../data/inline-hex.mesh",
            "../../data/toroid-hex.mesh",
            "../../data/periodic-cube.mesh"
         );
      mass_action<3>(filename3d, p);
      mass_mat_mixed<3>(filename3d, p);
   }
}

#endif // MFEM_USE_MPI
