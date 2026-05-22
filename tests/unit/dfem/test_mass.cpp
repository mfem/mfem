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

// FIXME: update this test to work with the new dFEM API.
#if 0

#include "../unit_tests.hpp"

#include "mfem.hpp"
#include "../../../fem/dfem/doperator.hpp"

#include "../fem/dfem/doperator.hpp"
#include "../linalg/test_same_matrices.hpp"

#include "../fem/dfem/backends/local_qf/prelude.hpp"
using LocalQFDefaultBackend = mfem::future::LocalQFBackend;
#include "../fem/dfem/backends/local_qf/qf_local_kernels.hpp"
using LocalQFKernelsBackend = mfem::future::LocalQFKernelsBackend;

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

template <int DIM, typename QFBackend>
void mass_action(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   dbg("{} {} {}", filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   ParGridFunction x(&pfes), y(&pfes), z(&pfes);
   Vector X(pfes.GetTrueVSize()), Y(pfes.GetTrueVSize()), Z(pfes.GetTrueVSize());

   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);

   // Action matrix free
   {
      dbg();
      const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

      Array<int> all_domain_attr;
      if (pmesh.attributes.Size() > 0)
      {
         all_domain_attr.SetSize(pmesh.attributes.Max());
         all_domain_attr = 1;
      }

      ParBilinearForm blf(&pfes);
      blf.AddDomainIntegrator(new MassIntegrator(one, ir));
      blf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      blf.Assemble();
      blf.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);

      static constexpr int U = 0, Coords = 1;
      const auto in_fds = std::vector
      {
         FieldDescriptor{ U, &pfes },
         FieldDescriptor{ Coords, mfes }
      };
      const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      const auto mf_mass_qf =
         [] MFEM_HOST_DEVICE(const real_t &u,
                             const tensor<real_t, DIM, DIM> &J,
                             const real_t &w,
                             real_t &v)
      {
         v = u * w * det(J);
      };
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

      real_t norm_g, norm_l = Y.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   // Test boundary
   // This ensures that we're not trying to test on fully periodic meshes
   /*if (!((std::string("../../data/periodic-square.mesh").compare(filename) == 0) ||
         (std::string("../../data/periodic-cube.mesh").compare(filename) == 0)))
   {
      constexpr int BDIM = DIM - 1;
      // SECTION("boundary")
      {
         const auto *ir = &IntRules.Get(pmesh.GetTypicalFaceGeometry(), 2 * p);

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

         // Vector N;
         // nodes->GetTrueDofs(N);
         // MultiVector MX{X, N}, MZ{Z};
         // auto dRdU = dop.GetDerivative(U, MX);
         // dRdU->Mult(MX, MZ);

         pfes.GetProlongationMatrix()->MultTranspose(y, Y);
         Y -= Z;
         norm_l = Y.Normlinf();
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }
   }*/
}

template <int DIM, typename QFBackend>
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
   if constexpr (std::is_same_v<QFBackend, LocalQFDefaultBackend>)
   {
      SparseMatrix *A;
      ddopdu->Assemble(A);
      TestSameMatrices(*A, blf.SpMat());
      delete A;
   }

   // hypre parallel mat
   if constexpr (std::is_same_v<QFBackend, LocalQFDefaultBackend>)
   {
      HypreParMatrix *Amfem = blf.ParallelAssemble();

      HypreParMatrix *Adfem;
      ddopdu->Assemble(Adfem);
      TestSameMatrices(*Adfem, *Amfem);
      delete Amfem;
      delete Adfem;
   }
}

TEST_CASE("dFEM Mass 2D", "[Parallel][dFEM][GPU][KER][MASS]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 2 : GENERATE(1, 2, 3);
   const auto mesh2d =
      GENERATE("../../data/star.mesh",
               "../../data/star-q3.mesh",
               "../../data/rt-2d-q3.mesh",
               "../../data/inline-quad.mesh",
               "../../data/periodic-square.mesh");

   SECTION("LocalQF Default")
   {
      mass_action<2, LocalQFDefaultBackend>(mesh2d, p);
   }

   SECTION("LocalQF Kernels")
   {
      mass_action<2, LocalQFKernelsBackend>(mesh2d, p);
   }

   // Avoiding failing 'hypre parallel mat' section
#ifndef MFEM_USE_CUDA_OR_HIP
#ifdef MFEM_USE_ENZYME
   SECTION("2D Mixed Default")
   {
      mass_mat_mixed<2, LocalQFDefaultBackend>(mesh2d, p);
   }
#else
   SECTION("2D Mixed Kernels")
   {
      // 2D not supported yet
      // mass_mat_mixed<2, LocalQFKernelsBackend>(mesh2d, p);
   }
#endif // MFEM_USE_ENZYME
#endif // MFEM_USE_CUDA_OR_HIP
}

TEST_CASE("dFEM Mass 3D", "[Parallel][dFEM][GPU][KER][MASS]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);
   const auto mesh3d =
      GENERATE("../../data/fichera.mesh",
               "../../data/fichera-q3.mesh",
               "../../data/inline-hex.mesh",
               "../../data/toroid-hex.mesh",
               "../../data/periodic-cube.mesh");

   SECTION("LocalQF Default")
   {
      mass_action<3, LocalQFDefaultBackend>(mesh3d, p);
   }

   SECTION("LocalQF Kernels")
   {
      mass_action<3, LocalQFKernelsBackend>(mesh3d, p);
   }

   // Avoiding failing 'hypre parallel mat' section
#ifndef MFEM_USE_CUDA_OR_HIP
#ifdef MFEM_USE_ENZYME
   SECTION("3D Mixed Default")
   {
      mass_mat_mixed<3, LocalQFDefaultBackend>(mesh3d, p);
   }
#else
   SECTION("3D Mixed Kernels")
   {
      mass_mat_mixed<3, LocalQFKernelsBackend>(mesh3d, p);
   }
#endif // MFEM_USE_ENZYME
#endif // MFEM_USE_CUDA_OR_HIP
}

#endif // MFEM_USE_MPI

#endif // # if 0
