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

#include "mfem.hpp"
#include "../../../fem/dfem/doperator.hpp"

#ifdef MFEM_USE_ENZYME
#include "../linalg/test_same_matrices.hpp"
#endif

#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

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

template <int DIM, typename QFBackend = LocalQFBackend>
void mass_action(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   dbg("{} {} {}", filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   H1_FECollection fec(p, DIM);

   ParFiniteElementSpace pfes(&pmesh, &fec);

   ParGridFunction x(&pfes), y(&pfes), z(&pfes);
   Vector X(pfes.GetTrueVSize()), Y(pfes.GetTrueVSize()), Z(pfes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);

   ParBilinearForm blf(&pfes);
   blf.AddDomainIntegrator(new MassIntegrator(one, ir));
   blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   blf.Assemble();

   static constexpr int U = 0, Coords = 1;
   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &pfes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

   const auto mf_mass_qf =
      [] MFEM_HOST_DEVICE(const dscalar_t &u,
                          const tensor<real_t, DIM, DIM> &J,
                          const real_t &w,
                          dscalar_t &v)
   {
      v = u * w * det(J);
   };

   SECTION("Action")
   {
      blf.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddDomainIntegrator<QFBackend>(
         mf_mass_qf,
         tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
         tuple{ Value<U>{} },
         *ir, all_domain_attr);

      Vector N;
      nodes->GetTrueDofs(N);
      pfes.GetRestrictionMatrix()->Mult(x, X);

      MultiVector MX{X, N}, MZ{Z};
      dop.Mult(MX, MZ);
      Y -= Z;

      real_t norm_global = 0.0;
      real_t norm_local = Y.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());
      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("Assemble Diagonal")
   {
      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      auto derivatives = std::integer_sequence<size_t, U> {};
      dop.AddDomainIntegrator<QFBackend>(
         mf_mass_qf,
         tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
         tuple{ Value<U>{} },
         *ir, all_domain_attr, derivatives);

      Vector N;
      nodes->GetTrueDofs(N);
      pfes.GetRestrictionMatrix()->Mult(x, X);
      MultiVector MX{X, N};

      auto dRdU = dop.GetDerivative(U, MX);

      Vector dfem_diagonal(pfes.GetTrueVSize());
      dRdU->AssembleDiagonal(dfem_diagonal);

      Vector mfem_diagonal(pfes.GetTrueVSize());
      blf.AssembleDiagonal(mfem_diagonal);

      dfem_diagonal -= mfem_diagonal;

      real_t norm_global = 0.0;
      real_t norm_local = dfem_diagonal.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());

      dbg("Assemble Diagonal");
      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Test boundary
#if 0 // TODO: Boundary tests 
   // This ensures that we're not trying to test on fully periodic meshes
   if (!((std::string("../../data/periodic-square.mesh").compare(filename) == 0) ||
         (std::string("../../data/periodic-cube.mesh").compare(filename) == 0)))
   {
      constexpr int BDIM = DIM - 1;
      SECTION("boundary")
      {
         Array<int> all_bdr_attr;
         if (pmesh.bdr_attributes.Size() > 0)
         {
            all_bdr_attr.SetSize(pmesh.bdr_attributes.Max());
            all_bdr_attr = 1;
         }

         ParBilinearForm blf(&pfes);
         blf.AddBoundaryIntegrator(new MassIntegrator(one, ir));
         blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         blf.Assemble();
         blf.Mult(x, y);
         pfes.GetProlongationMatrix()->MultTranspose(y, Y);

         static constexpr int U = 0, Coords = 1;
         const auto in_fds = std::vector<FieldDescriptor> {{ U, &pfes }, { Coords, mfes }};
         const auto out_fds = std::vector<FieldDescriptor> {{ U, &pfes }};
         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         const auto mf_mass_qf =
            [] MFEM_HOST_DEVICE(const dscalar_t &u,
                                const tensor<real_t, DIM, BDIM> &J,
                                const real_t &w,
                                dscalar_t& v)
         {
            v = u * weight(J) * w;
         };

         auto derivatives = std::integer_sequence<size_t, U> {};
         dop.AddBoundaryIntegrator<QFBackend>(mf_mass_qf,
                                              tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
                                              tuple{ Value<U>{} },
                                              *ir, all_bdr_attr, derivatives);

         pfes.GetRestrictionMatrix()->Mult(x, X);

         Vector N;
         nodes->GetTrueDofs(N);
         MultiVector MX{X, N}, MZ{Z};
         dop.Mult(MX, MZ);

         Y -= MZ[0];
         real_t norm_g, norm_l = Y.Normlinf();
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));

         nodes->GetTrueDofs(N);
         auto dRdU = dop.GetDerivative(U, MX);
         dRdU->Mult(MX[0], MZ);

         pfes.GetProlongationMatrix()->MultTranspose(y, Y);
         Y -= Z;
         norm_l = Y.Normlinf();
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }
#endif // TODO: Boundary tests 
}

template <int DIM, typename QFBackend = LocalQFBackend>
void mass_mat_mixed(const char* filename, int p)
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
   DifferentiableOperator dop(
   /* inputs  */ {{ U, &fes1 }, { Coords, nodes->ParFESpace() }},
   /* outputs */ {{ P, &fes0 }}, pmesh);
   const auto mf_mass_qf = [] MFEM_HOST_DEVICE(
                              const dscalar_t& u,
                              const tensor<real_t, DIM, DIM>& J,
                              const real_t& w,
                              dscalar_t& p)
   {
      p = u * w * det(J);
   };
   const auto derivatives = std::integer_sequence<size_t, U> {};
   dop.AddDomainIntegrator<QFBackend>(
      mf_mass_qf,
      tuple{Value<U>{}, Gradient<Coords>{}, Weight{}},
      tuple{Value<P>{}},
      *ir, all_domain_attr, derivatives);


   ParGridFunction ugf(&fes1);
   ugf = 0.0;

   ParGridFunction pgf(&fes0);
   pgf = 0.0;

   Vector xtvec(fes1.GetTrueVSize()), ytvec(fes0.GetTrueVSize());
   Vector nodestv;

   xtvec.Randomize(1);
   ugf.SetFromTrueDofs(xtvec);
   nodes->GetTrueDofs(nodestv);

   fes1.GetRestrictionMatrix()->Mult(ugf, xtvec);
   MultiVector X{xtvec, nodestv};

   auto ddopdu = dop.GetDerivative(U, X);

   // Action linearized
   {
      xtvec.Randomize(567);
      ugf.SetFromTrueDofs(xtvec);

      Vector dztvec(fes0.GetTrueVSize());
      MultiVector DZ{dztvec};
      ddopdu->Mult(X[0], DZ);

      blf.Mult(ugf, pgf);
      fes0.GetProlongationMatrix()->MultTranspose(pgf, ytvec);

      ytvec -= dztvec;

      real_t norm_global = 0.0;
      real_t norm_local = ytvec.Normlinf();
      MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                    pmesh.GetComm());

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // spmat
   {
#ifdef MFEM_USE_ENZYME
      SparseMatrix *A;
      ddopdu->Assemble(A);
      TestSameMatrices(*A, blf.SpMat());
      delete A;
#endif // MFEM_USE_ENZYME
   }

   // hypre parallel mat
   // Warning: "derivative can't be assembled into a HypreParMatrix"
   {
#if defined(MFEM_USE_ENZYME) && 0 // TODO
      HypreParMatrix *Amfem = blf.ParallelAssemble();

      HypreParMatrix *Adfem;
      ddopdu->Assemble(Adfem);
      TestSameMatrices(*Adfem, *Amfem);
      delete Amfem;
      delete Adfem;
#endif // MFEM_USE_ENZYME
   }
}

TEST_CASE("dFEM Mass 2D", "[Parallel][dFEM][GPU][MASS]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   const auto mesh2d =
      GENERATE("../../data/star.mesh",
               "../../data/star-q3.mesh",
               "../../data/rt-2d-q3.mesh",
               "../../data/inline-quad.mesh",
               "../../data/periodic-square.mesh");
   mass_action<2>(mesh2d, p);
   mass_mat_mixed<2>(mesh2d, p);
}

TEST_CASE("dFEM Mass 3D", "[Parallel][dFEM][GPU][MASS]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);
   const auto mesh3d =
      GENERATE("../../data/fichera.mesh",
               "../../data/fichera-q3.mesh",
               "../../data/inline-hex.mesh",
               "../../data/toroid-hex.mesh",
               "../../data/periodic-cube.mesh");
   mass_action<3>(mesh3d, p);
   mass_mat_mixed<3>(mesh3d, p);
}

#endif // MFEM_USE_MPI
