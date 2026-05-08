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
#include "../fem/dfem/doperator.hpp"
#include "../fem/dfem/backends/local_qf/prelude.hpp"
#include "../linalg/test_same_matrices.hpp"

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
      MFEM_HOST_DEVICE inline auto operator()(const dvecd_t &dudxi,
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
      MFEM_HOST_DEVICE inline auto operator()(const matd_t &J,
                                              const real_t &w,
                                              matd_t &qdata) const
      {
         qdata = inv(J) * transpose(inv(J)) * det(J) * w;
      }
   };

   struct PAApply
   {
      MFEM_HOST_DEVICE inline auto operator()(const dvecd_t &dudxi,
                                              const matd_t &qdata,
                                              dvecd_t &dvdxi) const
      {
         dvdxi = qdata * dudxi;
      };
   };
};

template <int DIM>
void diffusion(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   p = std::max(p, pmesh.GetNodalFESpace()->GetMaxElementOrder());
   smesh.Clear();

   Array<int> all_domain_attr;
   if (pmesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(pmesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   const int NE = pfes.GetNE(), d1d(p + 1), q = 2 * p;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q);

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

   static constexpr int U = 0, Coords = 1, Rho = 2;
   const auto in = std::vector
   {
      FieldDescriptor{ U, &pfes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out = std::vector{ FieldDescriptor{ U, &pfes } };

   SECTION("action")
   {
      DifferentiableOperator dop_mf(in, out, pmesh);
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

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("action partial assembly")
   {
      static constexpr int QData = 2;
      QuadratureSpace qspace(pmesh, *ir);
      QuadratureFunction qd(qspace, DIM * DIM);

      DifferentiableOperator setupPAData(
      {
         {Coords, mfes}
      },
      {
         {QData, &qd}
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
         {U, &pfes}, {QData, &qd}
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

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("action linearized")
   {
      DifferentiableOperator dop_mf(in, out, pmesh);
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

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("action vector")
   {
      ParFiniteElementSpace vpfes(&pmesh, &fec, DIM);
      ParGridFunction vx(&vpfes), vy(&vpfes);
      Vector vX(vpfes.GetTrueVSize()), vY(vpfes.GetTrueVSize()),
             vZ(vpfes.GetTrueVSize());

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

      const auto mf_vector_diffusion_qf =
         [] MFEM_HOST_DEVICE (const tensor<dscalar_t, DIM, DIM> &dudxi,
                              const tensor<real_t, DIM, DIM> &J,
                              const real_t &w,
                              tensor<dscalar_t, DIM, DIM> &dvdxi)
      {
         const auto invJ = inv(J);
         const auto invJt = transpose(invJ);
         dvdxi = (dudxi * invJ) * invJt * det(J) * w;
      };

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

      REQUIRE(norm_global == MFEM_Approx(0.0));
      MPI_Barrier(MPI_COMM_WORLD);
   }

   SECTION("SparseMatrix")
   {
      DifferentiableOperator dop_mf(in, out, pmesh);
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

      TestSameMatrices(*A, blf_fa.SpMat());
      delete A;

      MPI_Barrier(MPI_COMM_WORLD);
   }
}

TEST_CASE("dFEM Diffusion", "[Parallel][dFEM][GPU]")
{
   const bool all_tests = launch_all_non_regression_tests;
   const auto p = !all_tests ? 1 : GENERATE(1, 2, 3);

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
      diffusion<2>(filename2d, p);
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
      diffusion<3>(filename3d, p);
   }
}

#endif // MFEM_USE_MPI
