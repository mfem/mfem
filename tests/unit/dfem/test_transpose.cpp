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

   SECTION("Scalar Convection Transpose Action")
   {
      auto b_func = [](const Vector &x, Vector &b)
      {
         b(0) = cos(x[0] * 2.0 * M_PI);
         b(1) = 1.0 + cos(x[1] * 2.0 * M_PI);
         if constexpr (DIM == 3)
         {
            b(2) = 2.0 + cos(x[2] * 2.0 * M_PI);
         }
      };
      VectorFunctionCoefficient b_coeff(DIM, b_func);

      ParBilinearForm Gblf(&scalar_fes);
      auto conv_integ = new ConvectionIntegrator(b_coeff);
      conv_integ->SetIntegrationRule(*ir);
      Gblf.AddDomainIntegrator(conv_integ);
      Gblf.Assemble();
      Gblf.Finalize();
      auto Gmat = Gblf.ParallelAssemble();

      // Gmat->PrintMatlab(std::cout, 0, 0);
      // Gmat->PrintMatlabTranspose(std::cout, 0, 0);

      static constexpr int SCALAR = 0, COORDINATES = 1;
      const auto sol = std::vector{FieldDescriptor{SCALAR, &scalar_fes}};
      const auto par = std::vector
      {
         FieldDescriptor{COORDINATES, nodes->ParFESpace()}
      };
      DifferentiableOperator dop(sol, par, mesh);

      const auto convection_qf =
         [] MFEM_HOST_DEVICE(
            const tensor<dscalar_t, DIM> &dudxi,
            const tensor<real_t, DIM> &x,
            const tensor<real_t, DIM, DIM> &J,
            const real_t &w)
      {
         const auto dudx = dudxi * inv(J);
         tensor<dscalar_t, DIM> b{};
         b(0) = cos(x[0] * 2.0 * M_PI);
         b(1) = 1.0 + cos(x[1] * 2.0 * M_PI);
         if constexpr (DIM == 3)
         {
            b(2) = 2.0 + cos(x[2] * 2.0 * M_PI);
         }
         return tuple{dot(b, dudx) * w * det(J)};
      };

      auto derivatives = std::integer_sequence<size_t, SCALAR> {};
      dop.AddDomainIntegrator(convection_qf,
                              tuple{Gradient<SCALAR>{}, Value<COORDINATES>{}, Gradient<COORDINATES>{}, Weight{}},
                              tuple{Value<SCALAR>{}},
                              *ir, all_domain_attr, derivatives);
      dop.SetParameters({nodes});

      DifferentiableOperator dop_tr(sol, par, mesh);
      const auto convection_transpose_qf =
         [] MFEM_HOST_DEVICE(
            const dscalar_t &u,
            const tensor<real_t, DIM, DIM> &J,
            const real_t &w)
      {
         tensor<dscalar_t, DIM> b{};
         b(0) = 1.0;
         b(1) = 1.0;
         if constexpr (DIM == 3)
         {
            b(2) = 1.0;
         }
         return tuple{b * u * w * det(J) * transpose(inv(J))};
      };
      dop_tr.AddDomainIntegrator(convection_transpose_qf,
                                 tuple{Value<SCALAR>{}, Gradient<COORDINATES>{}, Weight{}},
                                 tuple{Gradient<SCALAR>{}},
                                 *ir, all_domain_attr, derivatives);
      dop_tr.SetParameters({nodes});

      Vector S, T, U;
      S.SetSize(scalar_fes.GetTrueVSize());
      T.SetSize(scalar_fes.GetTrueVSize());
      U.SetSize(scalar_fes.GetTrueVSize());
      U.Randomize(1);

      // printf("Handcoded dFEM convection transpose\n");
      // // Mmat->MultTranspose(U, S);
      // Gmat->MultTranspose(U, S);
      // dop_tr.Mult(U, T);
      // printf("S: ");
      // pretty_print(S);
      // printf("T: ");
      // pretty_print(T);

      // S -= T;
      // real_t norm_g, norm_l = S.Normlinf();
      // MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      // REQUIRE(norm_g == MFEM_Approx(0.0));

      // {
      //    printf("\nHandcoded dFEM convection using qpdc\n");
      //    Gmat->Mult(U, S);
      //    auto ddop = dop.GetDerivative(SCALAR, {&sgf}, {nodes});
      //    ddop->Mult(U, T);
      //    printf("S: ");
      //    pretty_print(S);
      //    printf("T: ");
      //    pretty_print(T);

      //    S -= T;
      //    real_t norm_g, norm_l = S.Normlinf();
      //    MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      //    // REQUIRE(norm_g == MFEM_Approx(0.0));
      // }

      // {
      //    Gmat->MultTranspose(U, S);
      //    auto ddop_tr = dop_tr.GetDerivative(SCALAR, {&sgf}, {nodes});
      //    ddop_tr->Mult(U, T);
      //    printf("S: ");
      //    pretty_print(S);
      //    printf("T: ");
      //    pretty_print(T);

      //    S -= T;
      //    real_t norm_g, norm_l = S.Normlinf();
      //    MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      //    // REQUIRE(norm_g == MFEM_Approx(0.0));
      // }

      {
         Gmat->MultTranspose(U, S);

         auto ddop = dop.GetDerivative(SCALAR, {&sgf}, {nodes});
         // ddop->PrintMatlab(std::cout, 0);
         ddop->MultTranspose(U, T);

         // ddop->PrintMatlabTranspose(std::cout, 0, 0);
         // std::cout << std::flush;

         // printf("S: ");
         // pretty_print(S);
         // printf("T: ");
         // pretty_print(T);

         S -= T;
         real_t norm_g, norm_l = S.Normlinf();
         MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
         REQUIRE(norm_g == MFEM_Approx(0.0));
      }

      delete Gmat;
   }

   SECTION("Nonlinear VectorConvection Transpose Action")
   {
      auto b_func = [](const Vector &x, Vector &b)
      {
         // b(0) = x[0] * (1.0 - x[0]);
         // b(1) = x[1] * (1.0 - x[1]);
         // if constexpr (DIM == 3)
         // {
         //    b(2) = x[2] * (1.0 - x[2]);
         // }
         b(0) = cos(x[0]) * sin(x[0]) * x[1];
         b(1) = cos(x[1]) * sin(x[1]) * x[0];
         if constexpr (DIM == 3)
         {
            b(2) = cos(x[2]) * sin(x[2]) * x[0];
         }
      };
      VectorFunctionCoefficient b_coeff(DIM, b_func);

      ParGridFunction ugf(&vector_fes);
      ugf.ProjectCoefficient(b_coeff);

      Vector U(vector_fes.GetTrueVSize());
      ugf.GetTrueDofs(U);

      ParNonlinearForm nlf(&vector_fes);
      const auto vcinteg = new VectorConvectionNLFIntegrator();
      vcinteg->SetIntegrationRule(*ir);
      nlf.AddDomainIntegrator(vcinteg);
      HypreParMatrix &Nmat = dynamic_cast<HypreParMatrix&>(nlf.GetGradient(U));

      static constexpr int VELOCITY = 0, COORDINATES = 1;
      const auto sol = std::vector{FieldDescriptor{VELOCITY, &vector_fes}};
      const auto par = std::vector
      {
         FieldDescriptor{COORDINATES, nodes->ParFESpace()}
      };
      DifferentiableOperator dop(sol, par, mesh);

      const auto nlconvection_qf =
         [] MFEM_HOST_DEVICE(
            const tensor<dscalar_t, DIM> &u,
            const tensor<dscalar_t, DIM, DIM> &dudxi,
            const tensor<real_t, DIM, DIM> &J,
            const real_t &w)
      {
         const auto invJ = inv(J);
         const auto dudx = dudxi * invJ;
         return tuple{dot(dudx, u) * w * det(J)};
      };

      auto derivatives = std::integer_sequence<size_t, VELOCITY> {};
      dop.AddDomainIntegrator(nlconvection_qf,
                              tuple{Value<VELOCITY>{}, Gradient<VELOCITY>{}, Gradient<COORDINATES>{}, Weight{}},
                              tuple{Value<VELOCITY>{}},
                              *ir, all_domain_attr, derivatives);
      dop.SetParameters({nodes});

      auto ddop = dop.GetDerivative(VELOCITY, {&ugf}, {nodes});

      Vector S(U.Size()), T(U.Size()), Se(vector_fes.GetVSize());

      Nmat.MultTranspose(U, S);
      ddop->MultTranspose(U, T);

      // {
      //    print_mpi_root("S: ");
      //    pretty_print_mpi(S);
      //    print_mpi_root("T: ");
      //    pretty_print_mpi(T);
      // }

      S -= T;
      real_t norm_g, norm_l = S.Normlinf();
      MPI_Allreduce(&norm_l, &norm_g, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());
      REQUIRE(norm_g == MFEM_Approx(0.0));
   }
}

TEST_CASE("dFEM Transpose", "[Parallel][dFEM][XXX]")
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
            "../../data/inline-quad.mesh"
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
            "../../data/toroid-hex.mesh"
         );
      transpose<3>(filename3d, p);
   }
}

#endif
