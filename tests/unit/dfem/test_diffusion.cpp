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

#include "unit_tests.hpp"
#include "mfem.hpp"

#include "fem/dfem/doperator.hpp"
#include "linalg/tensor.hpp"

using namespace mfem;
using mfem::internal::tensor;

namespace dfem_pa_kernels
{

/*real_t rho(const Vector &x)
{
   real_t r = pow(x(0), 2);
   if (x.Size() >= 2) { r += pow(x(1), 3); }
   if (x.Size() >= 3) { r += pow(x(2), 4); }
   return r;
}*/

TEST_CASE("DFEM Diffusion", "[Parallel][DFEM]")
{
   const bool all_tests = launch_all_non_regression_tests;

   const auto filename = GENERATE("../../data/star.mesh",
                                  "../../data/star-q3.mesh",
                                  "../../data/fichera.mesh",
                                  "../../data/fichera-q3.mesh");

   const auto order = !all_tests ? 2 : GENERATE(1, 2, 3);
   const auto q_order_inc = !all_tests ? 0 : GENERATE(0, 1, 2, 3);

   Mesh smesh(filename);
   smesh.EnsureNodes();
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   pmesh.SetCurvature(order); // 🔥 necessary with 3D q3 ?!
   auto *nodes = static_cast<ParGridFunction*>(pmesh.GetNodes());
   smesh.Clear();

   const int dim = pmesh.Dimension();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParFiniteElementSpace *mfes = nodes->ParFESpace();

   // L2_FECollection l2fec(0, dim);
   // ParFiniteElementSpace l2fes(&pmesh, &l2fec);

   const int q_order = 2 * order + q_order_inc;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), q_order);

   const int d1d(order + 1),
         q1d(IntRules.Get(Geometry::SEGMENT, ir->GetOrder()).GetNPoints());
   mfem::out << "\x1b[33m"
             << " filename: " << filename
             << " order: " << order
             << " dim: " << dim
             << " d1d: " << d1d
             << " q1d: " << q1d
             << "\x1b[m" << std::endl;

   if (dim == 2) { assert(q1d*q1d == ir->GetNPoints()); }
   if (dim == 3) { assert(q1d*q1d*q1d == ir->GetNPoints()); }

   ParGridFunction x(&pfes), y_fa(&pfes), y_dfem(&pfes);
   x.Randomize(1);

   // FunctionCoefficient rho_coeff(rho);
   ConstantCoefficient rho_coeff(1.0);

   ParBilinearForm blf_fa(&pfes);
   blf_fa.AddDomainIntegrator(new DiffusionIntegrator(rho_coeff, ir));
   blf_fa.Assemble();
   blf_fa.Finalize();

   Array<int> all_domain_attr(pmesh.bdr_attributes.Max());
   all_domain_attr = 1;

   {
      // ParGridFunction l2_rho_gf(&l2fes);
      // l2_rho_gf.ProjectCoefficient(rho_coeff);

      // ParGridFunction rho_gf(&l2fes0);
      // rho_gf.ProjectCoefficient(rho_coeff);
      // rho_gf.ProjectGridFunction(l2_rho_gf);
      // rho_g = 2.0;

      // QuadratureSpace qs(pmesh, *ir);
      // CoefficientVector coeff(rho_coeff, qs, CoefficientStorage::FULL);
      // assert(coeff.GetVDim() == 1);

      static constexpr int Potential = 0, Coordinates = 1, QData = 2;
      const auto solution = std::vector{FieldDescriptor{Potential, &pfes}};

      // Matrix free
      {
         DifferentiableOperator dOpMF(solution,
                                      std::vector{FieldDescriptor{Coordinates, mfes}},
                                      pmesh);
         dOpMF.SetParameters({nodes});

         auto apply_mf_qf_2d = [] MFEM_HOST_DEVICE(const tensor<real_t, 2>& dudxi,
                                                   const tensor<real_t, 2, 2> &J,
                                                   const real_t &w)
         {
            auto invJ = inv(J);
            return mfem::tuple{((dudxi * invJ)) * transpose(invJ) * det(J) * w};
         };
         auto apply_mf_qf_3d = [] MFEM_HOST_DEVICE(const tensor<real_t, 3>& dudxi,
                                                   const tensor<real_t, 3, 3> &J,
                                                   const real_t &w)
         {
            auto invJ = inv(J);
            return mfem::tuple{((dudxi * invJ)) * transpose(invJ) * det(J) * w};
         };
         if (dim == 2)
         {
            dOpMF.AddDomainIntegrator(apply_mf_qf_2d,
                                      mfem::tuple { Gradient<Potential>{},
                                                    Gradient<Coordinates> {},
                                                    Weight{} },
                                      mfem::tuple{ Gradient<Potential>{}},
                                      *ir, all_domain_attr);
         }
         else if (dim == 3)
         {
            dOpMF.AddDomainIntegrator(apply_mf_qf_3d,
                                      mfem::tuple { Gradient<Potential>{},
                                                    Gradient<Coordinates> {},
                                                    Weight{} },
                                      mfem::tuple{ Gradient<Potential>{}},
                                      *ir, all_domain_attr);
         }
         else { MFEM_ABORT("Not implemented"); }
         dOpMF.Mult(x, y_dfem);

         y_fa = 0.0;
         blf_fa.Mult(x, y_fa);
         y_fa -= y_dfem;
         REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
      }

      // Partial assembly
      {
         const int elem_size(dim * dim * ir->GetNPoints()),
               total_size(elem_size * pmesh.GetNE());
         // 🔥 2D workaround for is_none_fop Reshape access
         // ParametricSpace qs(dim, dim * dim, elem_size, total_size);
         ParametricSpace qs(dim, dim * dim, elem_size, total_size,
                            dim == 3 ? d1d : d1d*d1d, dim == 3 ? q1d : q1d*q1d);
         ParametricFunction qdata(qs);
         qdata.UseDevice(true);

         // setup
         {
            DifferentiableOperator dSetup(solution,
                                          std::vector{FieldDescriptor{Coordinates, mfes},
                                                      FieldDescriptor{QData, &qs}},
                                          pmesh);
            dSetup.SetParameters({nodes, &qdata});
            auto setup_qf_2d = [] MFEM_HOST_DEVICE(const real_t &u,
                                                   const tensor<real_t, 2, 2> &J,
                                                   const real_t &w)
            {
               return mfem::tuple{inv(J) * transpose(inv(J)) * det(J) * w};
            };
            auto setup_qf_3d = [] MFEM_HOST_DEVICE(const real_t &u,
                                                   const tensor<real_t, 3, 3> &J,
                                                   const real_t &w)
            {
               return mfem::tuple{inv(J) * transpose(inv(J)) * det(J) * w};
            };
            if (dim == 2)
            {
               dSetup.AddDomainIntegrator(setup_qf_2d,
                                          mfem::tuple { None<Potential> {},
                                                        Gradient<Coordinates> {},
                                                        Weight{} },
                                          mfem::tuple{ None<QData> {}},
                                          *ir, all_domain_attr);
            }
            else if (dim == 3)
            {
               dSetup.AddDomainIntegrator(setup_qf_3d,
                                          mfem::tuple { None<Potential> {},
                                                        Gradient<Coordinates> {},
                                                        Weight{} },
                                          mfem::tuple{ None<QData> {}},
                                          *ir, all_domain_attr);
            }
            else { MFEM_ABORT("Not implemented"); }
            dSetup.Mult(x, qdata);
         }

         // Apply
         DifferentiableOperator dApply(solution,
                                       std::vector{FieldDescriptor{QData, &qs}},
                                       pmesh);
         dApply.SetParameters({&qdata});
         auto apply_qf_2d = [] MFEM_HOST_DEVICE (const tensor<real_t, 2> &dudxi,
                                                 const tensor<real_t, 2, 2> &q)
         {
            return mfem::tuple{q * dudxi};
         };
         auto apply_qf_3d = [] MFEM_HOST_DEVICE (const tensor<real_t, 3> &dudxi,
                                                 const tensor<real_t, 3, 3> &q)
         {
            return mfem::tuple{q * dudxi};
         };
         if (dim == 2)
         {
            dApply.AddDomainIntegrator(apply_qf_2d,
                                       mfem::tuple{ Gradient<Potential>{}, None<QData>{} },
                                       mfem::tuple{ Gradient<Potential>{}},
                                       *ir, all_domain_attr);
         }
         else if (dim == 3)
         {
            dApply.AddDomainIntegrator(apply_qf_3d,
                                       mfem::tuple{ Gradient<Potential>{}, None<QData>{} },
                                       mfem::tuple{ Gradient<Potential>{}},
                                       *ir, all_domain_attr);
         }
         else { MFEM_ABORT("Not implemented"); }
         dApply.Mult(x, y_dfem);

         y_fa = 0.0;
         blf_fa.Mult(x, y_fa);
         y_fa -= y_dfem;
         REQUIRE(y_fa.Normlinf() == MFEM_Approx(0.0));
      }
   }
}

} // namespace dfem_pa_kernels
