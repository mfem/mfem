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
#include "fem/dfem/doperator.hpp"

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
void transpose(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh serial_mesh(filename);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   mesh.EnsureNodes();
   auto* nodes = static_cast<ParGridFunction*>(mesh.GetNodes());
   p = std::max(p, mesh.GetNodalFESpace()->GetMaxElementOrder());

   Array<int> all_domain_attr;
   if (mesh.attributes.Size() > 0)
   {
      all_domain_attr.SetSize(mesh.attributes.Max());
      all_domain_attr = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace scalar_fes(&mesh, &fec);
   ParFiniteElementSpace vector_fes(&mesh, &fec, DIM);

   ParGridFunction sgf(&scalar_fes);

   auto f0 = [](const Vector &x)
   {
      if constexpr (DIM == 3)
      {
         return M_PI*cos(M_PI*x[0]) * sin(M_PI*x[1]) * sin(M_PI*x[2]);
      }
      return M_PI*cos(M_PI*x[0]) * sin(M_PI*x[1]);
   };

   FunctionCoefficient f0_coeff(f0);
   sgf.ProjectCoefficient(f0_coeff);

   ParGridFunction vgf(&vector_fes);

   auto gradf1 = [](const Vector &x, Vector &u)
   {
      if constexpr (DIM == 3)
      {
         u(0) = M_PI*cos(M_PI*x[0]) * sin(M_PI*x[1]) * sin(M_PI*x[2]);
         u(1) = M_PI*sin(M_PI*x[0]) * cos(M_PI*x[1]) * sin(M_PI*x[2]);
         u(2) = M_PI*sin(M_PI*x[0]) * sin(M_PI*x[1]) * cos(M_PI*x[2]);
         return;
      }
      u(0) = M_PI*cos(M_PI*x[0]) * sin(M_PI*x[1]);
      u(1) = M_PI*sin(M_PI*x[0]) * cos(M_PI*x[1]);
   };

   VectorFunctionCoefficient gradf1_coeff(DIM, gradf1);
   vgf.ProjectCoefficient(gradf1_coeff);

   const auto* ir = &IntRules.Get(mesh.GetTypicalElementGeometry(), 2 * p);

   SECTION("Mass Transpose Action")
   {
      ParBilinearForm Mblf(&scalar_fes);
      auto mass_integ = new MassIntegrator;
      mass_integ->SetIntegrationRule(*ir);
      Mblf.AddDomainIntegrator(mass_integ);
      Mblf.Assemble();
      Mblf.Finalize();
      auto Mmat = Mblf.ParallelAssemble();

      static constexpr int SCALAR = 0, COORDINATES = 1;
      const auto sol = std::vector{FieldDescriptor{SCALAR, &scalar_fes}};
      const auto par = std::vector{FieldDescriptor{COORDINATES, nodes->ParFESpace()}};
      DifferentiableOperator dop(sol, par, mesh);
      const auto gradient_qf = [] MFEM_HOST_DEVICE(
                                  const dscalar_t &u,
                                  const tensor<real_t, DIM, DIM> &J,
                                  const real_t &w)
      {
         return tuple{u * w * det(J)};
      };

      auto derivatives = std::integer_sequence<size_t, SCALAR> {};
      dop.AddDomainIntegrator(gradient_qf,
                              tuple{Value<SCALAR>{}, Gradient<COORDINATES>{}, Weight{}},
                              tuple{Value<SCALAR>{}},
                              *ir, all_domain_attr, derivatives);
      dop.SetParameters({nodes});

      Vector S, T, U;
      S.SetSize(scalar_fes.GetTrueVSize());
      T.SetSize(scalar_fes.GetTrueVSize());
      U.SetSize(scalar_fes.GetTrueVSize());

      sgf.GetTrueDofs(S);

      Mmat->MultTranspose(S, T);

      auto ddop = dop.GetDerivative(SCALAR, {&sgf}, {nodes});
      ddop->MultTranspose(S, U);

      T -= U;
      real_t norm_g, norm_l = T.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));

      delete Mmat;
   }

   SECTION("Vector Mass Transpose Action")
   {
      ParBilinearForm Mvblf(&vector_fes);
      auto mass_integ = new VectorMassIntegrator;
      mass_integ->SetIntegrationRule(*ir);
      Mvblf.AddDomainIntegrator(mass_integ);
      Mvblf.Assemble();
      Mvblf.Finalize();
      auto Mvmat = Mvblf.ParallelAssemble();

      static constexpr int VECTOR = 0, COORDINATES = 1;
      const auto sol = std::vector{FieldDescriptor{VECTOR, &vector_fes}};
      const auto par = std::vector{FieldDescriptor{COORDINATES, nodes->ParFESpace()}};
      DifferentiableOperator dop(sol, par, mesh);
      const auto gradient_qf = [] MFEM_HOST_DEVICE(
                                  const tensor<dscalar_t, DIM> &u,
                                  const tensor<real_t, DIM, DIM> &J,
                                  const real_t &w)
      {
         return tuple{u * w * det(J)};
      };

      auto derivatives = std::integer_sequence<size_t, VECTOR> {};
      dop.AddDomainIntegrator(gradient_qf,
                              tuple{Value<VECTOR>{}, Gradient<COORDINATES>{}, Weight{}},
                              tuple{Value<VECTOR>{}},
                              *ir, all_domain_attr, derivatives);
      dop.SetParameters({nodes});

      Vector V, W, Z;
      V.SetSize(vector_fes.GetTrueVSize());
      W.SetSize(vector_fes.GetTrueVSize());
      Z.SetSize(vector_fes.GetTrueVSize());

      vgf.GetTrueDofs(V);

      Mvmat->MultTranspose(V, W);

      auto ddop = dop.GetDerivative(VECTOR, {&vgf}, {nodes});
      ddop->MultTranspose(V, Z);

      W -= Z;
      real_t norm_g, norm_l = W.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));

      delete Mvmat;
   }

   SECTION("Discrete Gradient Transpose Action")
   {
      ParMixedBilinearForm Gblf(&scalar_fes, &vector_fes);
      auto grad_integ = new GradientIntegrator;
      grad_integ->SetIntegrationRule(*ir);
      Gblf.AddDomainIntegrator(grad_integ);
      Gblf.Assemble();
      Gblf.Finalize();
      auto Gmat = Gblf.ParallelAssemble();

      static constexpr int SCALAR = 0, VECTOR = 2, COORDINATES = 1;
      const auto sol = std::vector{FieldDescriptor{SCALAR, &scalar_fes}};
      const auto par = std::vector
      {
         FieldDescriptor{VECTOR, &vector_fes},
         FieldDescriptor{COORDINATES, nodes->ParFESpace()}
      };
      DifferentiableOperator dop(sol, par, mesh);
      const auto gradient_qf = [] MFEM_HOST_DEVICE(
                                  const tensor<dscalar_t, DIM> &dudxi,
                                  const tensor<real_t, DIM, DIM> &J,
                                  const real_t &w)
      {
         const auto dudx = dudxi * inv(J);
         return tuple{dudx * w * det(J)};
      };

      auto derivatives = std::integer_sequence<size_t, SCALAR> {};
      dop.AddDomainIntegrator(gradient_qf,
                              tuple{Gradient<SCALAR>{}, Gradient<COORDINATES>{}, Weight{}},
                              tuple{Value<VECTOR>{}},
                              *ir, all_domain_attr, derivatives);
      dop.SetParameters({&vgf, nodes});

      Vector S, T, V;
      S.SetSize(scalar_fes.GetTrueVSize());
      T.SetSize(scalar_fes.GetTrueVSize());
      vgf.GetTrueDofs(V);

      Gmat->MultTranspose(V, S);

      auto ddop = dop.GetDerivative(SCALAR, {&sgf}, {&vgf, nodes});
      ddop->MultTranspose(V, T);

      S -= T;
      real_t norm_g, norm_l = S.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));

      delete Gmat;
   }
}

TEST_CASE("dFEM Transpose", "[Parallel][dFEM][XXX]")
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
      transpose<2>(filename2d, p);
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
      transpose<3>(filename3d, p);
   }
}

#endif
