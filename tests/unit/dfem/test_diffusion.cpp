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
#include "../linalg/test_same_matrices.hpp"

#include "../../../fem/dfem/backends/local_qf/prelude.hpp"
using LocalQFDefaultBackend = mfem::future::LocalQFBackend;
#include "../fem/dfem/backends/local_qf/qf_local_kernels.hpp"
using LocalQFKernelsBackend = mfem::future::LocalQFKernelsBackend;

#include <utility>

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

template <int DIM> struct Diffusion
{
   using dvecd_t = tensor<dscalar_t, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   struct MFApply
   {
      MFEM_HOST_DEVICE inline auto operator()(
         const dvecd_t &dudxi,
         const matd_t &J,
         const real_t &w,
         dvecd_t &dvdxi) const
      {
         const auto invJ = inv(J);
         const auto invJt = transpose(invJ);
         dvdxi = (dudxi * invJ) * invJt * det(J) * w;
      }
   };

   struct PASetup
   {
      MFEM_HOST_DEVICE inline auto operator()(
         const matd_t &J,
         const real_t &w,
         matd_t &qdata) const
      {
         qdata = inv(J) * transpose(inv(J)) * det(J) * w;
      }
   };

   struct PAApply
   {
      MFEM_HOST_DEVICE inline auto operator()(
         const dvecd_t &dudxi,
         const matd_t &qdata,
         dvecd_t &dvdxi) const
      {
         dvdxi = qdata * dudxi;
      };
   };
};

template <int DIM> struct VectorDiffusion
{
   using dmatd_t = tensor<dscalar_t, DIM, DIM>;
   using matd_t = tensor<real_t, DIM, DIM>;

   struct MFApply
   {
      MFEM_HOST_DEVICE inline auto operator()(
         const dmatd_t &dudxi,
         const matd_t &J,
         const real_t &w,
         dmatd_t &dvdxi) const
      {
         const auto invJ = inv(J);
         const auto invJt = transpose(invJ);
         dvdxi = (dudxi * invJ) * invJt * det(J) * w;
      }
   };
};

template <int DIM, typename QFBackend>
void diffusion(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);
   dbg("{} {} {}", filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   smesh.Clear();

   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   const int q = 2 * p;

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);

   H1_FECollection fec(p, DIM);

   static constexpr int U = 0, Coords = 1;

   SECTION("Scalar")
   {
#undef DFEM_RUN_SCALAR_DIFFUSION_TESTS
#ifdef DFEM_RUN_SCALAR_DIFFUSION_TESTS
      ParFiniteElementSpace pfes(&pmesh, &fec);

      ParGridFunction x(&pfes), y(&pfes), z(&pfes);
      Vector xtvec(pfes.GetTrueVSize()), ytvec(pfes.GetTrueVSize()),
             ztvec(pfes.GetTrueVSize());

      xtvec.Randomize(1);
      x.SetFromTrueDofs(xtvec);

      ParBilinearForm blf_fa(&pfes);
      blf_fa.AddDomainIntegrator(new DiffusionIntegrator(ir));
      blf_fa.SetAssemblyLevel(AssemblyLevel::FULL);
      blf_fa.Assemble();
      blf_fa.Finalize();

      const auto in_fds = std::vector
      {
         FieldDescriptor{ U, &pfes },
         FieldDescriptor{ Coords, mfes }
      };
      const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

      SECTION("Action")
      {
         DifferentiableOperator dop_mf(in_fds, out_fds, pmesh);
         typename Diffusion<DIM>::MFApply mf_apply_qf;
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_apply_qf,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            *ir, all_domain_attr);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);

         pfes.GetRestrictionMatrix()->Mult(x, xtvec);

         MultiVector X{xtvec, nodestv};
         MultiVector Z{ztvec};

         dop_mf.Mult(X, Z);

         blf_fa.Mult(x, y);
         pfes.GetProlongationMatrix()->MultTranspose(y, ytvec);
         ytvec -= ztvec;

         real_t norm_global = 0.0;
         real_t norm_local = ytvec.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Scalar Action");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      SECTION("Action Partial Assembly")
      {
         static constexpr int QData = 2;
         QuadratureSpace qspace(pmesh, *ir);
         VectorQuadratureSpace qspace_vec(qspace, DIM * DIM);
         QuadratureFunction qd(qspace_vec);

         DifferentiableOperator setupPAData(
         {
            {Coords, mfes}
         },
         {
            {QData, &qspace_vec}
         }, pmesh);

         typename Diffusion<DIM>::PASetup pa_setup_qf;
         setupPAData.AddDomainIntegrator<LocalQFBackend>(
            pa_setup_qf,
            tuple{Gradient<Coords>{}, Weight{}},
            tuple{Identity<QData>{}},
            *ir, all_domain_attr);

         {
            Vector nodestv;
            nodes->GetTrueDofs(nodestv);
            MultiVector X{nodestv};
            MultiVector Y{qd};
            setupPAData.Mult(X, Y);
         }

         DifferentiableOperator applyPAData(
         {
            {U, &pfes}, {QData, &qspace_vec}
         },
         {
            {U, &pfes}
         }, pmesh);
         typename Diffusion<DIM>::PAApply pa_apply_qf;
         applyPAData.AddDomainIntegrator<LocalQFBackend>(
            pa_apply_qf,
            tuple{ Gradient<U>{}, Identity<QData>{} },
            tuple{ Gradient<U>{} },
            *ir, all_domain_attr);

         {
            pfes.GetRestrictionMatrix()->Mult(x, xtvec);
            MultiVector X{xtvec, qd};
            MultiVector Z{ztvec};
            applyPAData.Mult(X, Z);
         }

         blf_fa.Mult(x, y);
         pfes.GetProlongationMatrix()->MultTranspose(y, ytvec);
         ytvec -= ztvec;

         real_t norm_global = 0.0;
         real_t norm_local = ytvec.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Action Partial Assembly");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      // derivative action with native-dual LocalQFKernelsBackend (3D)
      if constexpr (DIM == 3 && std::is_same_v<QFBackend, LocalQFKernelsBackend>)
      {
         DifferentiableOperator dop_mf(in_fds, out_fds, pmesh);
         typename Diffusion<DIM>::MFApply mf_apply_qf;
         const auto derivatives = std::integer_sequence<size_t, U> {};
         dop_mf.AddDomainIntegrator<LocalQFKernelsBackend>(
            mf_apply_qf,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            *ir, all_domain_attr, derivatives);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);

         pfes.GetRestrictionMatrix()->Mult(x, xtvec);
         MultiVector X{xtvec, nodestv};
         auto ddop = dop_mf.GetDerivative(U, X);

         xtvec.Randomize(567);
         x.SetFromTrueDofs(xtvec);

         Vector dztvec(ztvec.Size());
         MultiVector DZ{dztvec};
         ddop->Mult(X[0], DZ);

         blf_fa.Mult(x, y);
         pfes.GetProlongationMatrix()->MultTranspose(y, ytvec);

         ytvec -= dztvec;

         real_t norm_global = 0.0;
         real_t norm_local = ytvec.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Derivative Action with native-dual LocalQFKernelsBackend");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      SECTION("Action Linearized")
      {
#ifdef MFEM_USE_ENZYME
         DifferentiableOperator dop_mf(in_fds, out_fds, pmesh);
         typename Diffusion<DIM>::MFApply mf_apply_qf;
         auto derivatives = std::integer_sequence<size_t, U> {};
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_apply_qf,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            *ir, all_domain_attr, derivatives);

         pfes.GetRestrictionMatrix()->Mult(x, xtvec);
         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{xtvec, nodestv};
         MultiVector Z{ztvec};
         auto ddop = dop_mf.GetDerivative(U, X);

         // Randomize again s.t. the PA setup like cache can't
         // trivially succeed by caching one direction only.
         xtvec.Randomize(567);
         x.SetFromTrueDofs(xtvec);

         Vector dztvec(ztvec.Size());
         MultiVector DZ{dztvec};
         ddop->Mult(X[0], DZ);

         blf_fa.Mult(x, y);
         pfes.GetProlongationMatrix()->MultTranspose(y, ytvec);

         ytvec -= dztvec;

         real_t norm_global = 0.0;
         real_t norm_local = ytvec.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Action Linearized");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_ENZYME
      }

      SECTION("SparseMatrix")
      {
#ifdef MFEM_USE_ENZYME
         DifferentiableOperator dop_mf(in_fds, out_fds, pmesh);
         typename Diffusion<DIM>::MFApply mf_apply_qf;
         auto derivatives = std::integer_sequence<size_t, U> {};
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_apply_qf,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            *ir, all_domain_attr, derivatives);

         pfes.GetRestrictionMatrix()->Mult(x, xtvec);
         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{xtvec, nodestv};
         auto dRdU = dop_mf.GetDerivative(U, X);

         SparseMatrix *A = nullptr;
         dRdU->Assemble(A);

         dbg("SparseMatrix");
         TestSameMatrices(*A, blf_fa.SpMat());
         delete A;
         MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_ENZYME
      }

      SECTION("Assemble Diagonal")
      {
#ifdef MFEM_USE_ENZYME
         DifferentiableOperator dop_mf(in_fds, out_fds, pmesh);
         typename Diffusion<DIM>::MFApply mf_apply_qf;
         auto derivatives = std::integer_sequence<size_t, U> {};
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_apply_qf,
            tuple{Gradient<U>{}, Gradient<Coords>{}, Weight{}},
            tuple{Gradient<U>{}},
            *ir, all_domain_attr, derivatives);

         pfes.GetRestrictionMatrix()->Mult(x, xtvec);
         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{xtvec, nodestv};
         auto dRdU = dop_mf.GetDerivative(U, X);

         Vector dfem_diagonal(pfes.GetTrueVSize());
         dRdU->AssembleDiagonal(dfem_diagonal);

         Vector mfem_diagonal(pfes.GetTrueVSize());
         blf_fa.AssembleDiagonal(mfem_diagonal);

         dfem_diagonal -= mfem_diagonal;

         real_t norm_global = 0.0;
         real_t norm_local = dfem_diagonal.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Assemble Diagonal");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_ENZYME
      }
#endif // DFEM_RUN_SCALAR_DIFFUSION_TESTS
   }

   SECTION("Vector")
   {
#define DFEM_RUN_VECTOR_DIFFUSION_TESTS
#ifdef DFEM_RUN_VECTOR_DIFFUSION_TESTS

      ParFiniteElementSpace vpfes(&pmesh, &fec, DIM);
      ParGridFunction vx(&vpfes), vy(&vpfes);
      Vector vX(vpfes.GetTrueVSize()), vY(vpfes.GetTrueVSize()),
             vZ(vpfes.GetTrueVSize());

      SECTION("Action")
      {
         vX.Randomize(1);
         vx.SetFromTrueDofs(vX);

         DifferentiableOperator dop_mf(
         {
            {U, &vpfes},
            {Coords, mfes},
         },
         {
            {U, &vpfes}
         }, pmesh);

         typename VectorDiffusion<DIM>::MFApply mf_vector_diffusion_qf;
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_vector_diffusion_qf,
            tuple{ Gradient<U>{}, Gradient<Coords>{}, Weight{} },
            tuple{ Gradient<U>{} },
            *ir, all_domain_attr);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{vX, nodestv};
         MultiVector Z{vZ};
         dop_mf.Mult(X, Z);

         ParBilinearForm vblf_fa(&vpfes);
         vblf_fa.AddDomainIntegrator(new VectorDiffusionIntegrator(ir));
         vblf_fa.SetAssemblyLevel(AssemblyLevel::LEGACYFULL);
         vblf_fa.Assemble();
         vblf_fa.Finalize();
         vblf_fa.Mult(vx, vy);
         vpfes.GetProlongationMatrix()->MultTranspose(vy, vY);

         vY -= vZ;
         real_t norm_global = 0.0;
         real_t norm_local = vY.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Vector Action");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
      }

      SECTION("Action Linearized")
      {
#ifdef MFEM_USE_ENZYME
         vX.Randomize(1);
         vx.SetFromTrueDofs(vX);

         DifferentiableOperator dop_mf(
         {
            {U, &vpfes},
            {Coords, mfes},
         },
         {
            {U, &vpfes}
         }, pmesh);

         typename VectorDiffusion<DIM>::MFApply mf_vector_diffusion_qf;
         const auto derivatives = std::integer_sequence<size_t, U> {};
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_vector_diffusion_qf,
            tuple{ Gradient<U>{}, Gradient<Coords>{}, Weight{} },
            tuple{ Gradient<U>{} },
            *ir, all_domain_attr, derivatives);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{vX, nodestv};
         const auto ddop = dop_mf.GetDerivative(U, X);

         MultiVector Z{vZ};
         ddop->Mult(vX, Z);

         ParBilinearForm vblf_fa(&vpfes);
         vblf_fa.AddDomainIntegrator(new VectorDiffusionIntegrator(ir));
         vblf_fa.SetAssemblyLevel(AssemblyLevel::LEGACYFULL);
         vblf_fa.Assemble();
         vblf_fa.Finalize();
         vblf_fa.Mult(vx, vy);
         vpfes.GetProlongationMatrix()->MultTranspose(vy, vY);

         vY -= vZ;
         real_t norm_global = 0.0;
         real_t norm_local = vY.Normlinf();
         MPI_Allreduce(&norm_local, &norm_global, 1, MPI_DOUBLE, MPI_MAX,
                       pmesh.GetComm());

         dbg("Vector Action Linearized");
         REQUIRE(norm_global == MFEM_Approx(0.0));
         MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_ENZYME
      }

      SECTION("SparseMatrix")
      {
#ifdef MFEM_USE_ENZYME
         vX.Randomize(1);
         vx.SetFromTrueDofs(vX);

         DifferentiableOperator dop_mf(
         {
            {U, &vpfes},
            {Coords, mfes},
         },
         {
            {U, &vpfes}
         }, pmesh);

         typename VectorDiffusion<DIM>::MFApply mf_vector_diffusion_qf;
         const auto derivatives = std::integer_sequence<size_t, U> {};
         dop_mf.AddDomainIntegrator<LocalQFBackend>(
            mf_vector_diffusion_qf,
            tuple{ Gradient<U>{}, Gradient<Coords>{}, Weight{} },
            tuple{ Gradient<U>{} },
            *ir, all_domain_attr, derivatives);

         Vector nodestv;
         nodes->GetTrueDofs(nodestv);
         MultiVector X{vX, nodestv};
         const auto ddop = dop_mf.GetDerivative(U, X);

         MultiVector Z{vZ};
         ddop->Mult(vX, Z);

         ParBilinearForm vblf_fa(&vpfes);
         vblf_fa.AddDomainIntegrator(new VectorDiffusionIntegrator(ir));
         vblf_fa.SetAssemblyLevel(AssemblyLevel::LEGACYFULL);
         vblf_fa.Assemble();
         vblf_fa.Finalize();

         SparseMatrix *A = nullptr;
         ddop->Assemble(A);

         dbg("Vector SparseMatrix");
         TestSameMatrices(*A, vblf_fa.SpMat());
         delete A;
         MPI_Barrier(MPI_COMM_WORLD);
#endif // MFEM_USE_ENZYME
      }
#endif // DFEM_RUN_VECTOR_DIFFUSION_TESTS
   }
}

// TEST_CASE("dFEM Diffusion 2D", "[Parallel][dFEM][GPU][KER][DIFFUSION]")
// {
//    const auto all_tests = launch_all_non_regression_tests;
//    const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);
//    const auto mesh2d =
//       GENERATE("../../data/star.mesh",
//                "../../data/star-q3.mesh",
//                "../../data/rt-2d-q3.mesh",
//                "../../data/inline-quad.mesh",
//                "../../data/periodic-square.mesh");

//    SECTION("LocalQF Default")
//    {
//       diffusion<2, LocalQFDefaultBackend>(mesh2d, p);
//    }

//    // SECTION("LocalQF Kernels")
//    // {
//    //    diffusion<2, LocalQFKernelsBackend>(mesh2d, p);
//    // }
// }

TEST_CASE("dFEM Diffusion 3D", "[Parallel][dFEM][GPU][KER][DIFFUSION]")
{
   const auto all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);
   const auto mesh3d =
      GENERATE("../../data/fichera.mesh",
               "../../data/fichera-q3.mesh",
               "../../data/inline-hex.mesh",
               "../../data/toroid-hex.mesh",
               "../../data/periodic-cube.mesh");

   // SECTION("LocalQF Default")
   // {
   //    diffusion<3, LocalQFDefaultBackend>(mesh3d, p);
   // }

   SECTION("LocalQF Kernels")
   {
      diffusion<3, LocalQFKernelsBackend>(mesh3d, p);
   }
}

#endif // MFEM_USE_MPI
