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

#include "../unit_tests.hpp"

#include "mfem.hpp"

#ifdef MFEM_USE_MPI

#include "../linalg/test_same_matrices.hpp"

#include "../../../fem/dfem/doperator.hpp"
#include "../../../fem/dfem/backends/local_qf/prelude.hpp"

using namespace mfem;
using namespace mfem::future;

#ifdef MFEM_USE_ENZYME
using dscalar_t = real_t;
#else
using dscalar_t = dual<real_t, real_t>;
#endif

// ────────────────────────────────────────────────────────────────────────────
template <int DIM> struct global_mf_mass_qf
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &v) const
   {
      mfem::forall<UseEnzyme>(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = (dscalar_t)(u(q)) * w(q) * det(J(q));
      });
   }
};

template <int DIM> struct global_bdr_mass_qf
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t, DIM, DIM - 1> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t> &v) const
   {
      mfem::forall<UseEnzyme>(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = (dscalar_t)(u(q)) * w(q) * weight(J(q));
      });
   }
};

// This tests global bdry integrator applied to sub entity
// which includes also an Identity fop
template <int DIM> struct global_rho_mass_qf
{
   void operator()(tensor_array<const dscalar_t> &u,
                   tensor_array<const real_t, DIM, DIM> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<const real_t> &rho,
                   tensor_array<dscalar_t> &v) const
   {
      mfem::forall<UseEnzyme>(u.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = (dscalar_t)(u(q)) * rho(q) * w(q) * det(J(q));
      });
   }
};

template <int DIM> struct local_rho_mass_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(const dscalar_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   const real_t &rho,
                   dscalar_t &v) const
   {
      v = rho * u * w * det(J);
   }
};

template <int DIM> struct local_mf_mass_qf
{
   inline MFEM_HOST_DEVICE
   void operator()(const dscalar_t &u,
                   const tensor<real_t, DIM, DIM> &J,
                   const real_t &w,
                   dscalar_t &v) const
   {
      v = u * w * det(J);
   }
};

template <int DIM, int VDIM> struct global_bdr_vector_mass_qf
{
   void operator()(tensor_array<const dscalar_t, VDIM> &u,
                   tensor_array<const real_t, DIM, DIM - 1> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<dscalar_t, VDIM> &v) const
   {
      mfem::forall<UseEnzyme>(w.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         v(q) = u(q) * w(q) * weight(J(q));
      });
   }
};

// ────────────────────────────────────────────────────────────────────────────
// Boundary integrators of a scalar H1 space, alone and with domain integrators,
// against the legacy forms.

// TODO: Boundaries of 2D meshes are 1D, which the LocalQF
// kernels do not support yet, so only 3D meshes are tested.
template <int DIM>
void bdr_mass(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");
   smesh.Clear();

   // Check for periodic meshes,since AddBoundaryIntegrator needs every
   // boundary element to be a boundary face, otherwise it breaks.
   int bdr_elements_are_faces =
      pmesh.GetNFbyType(FaceType::Boundary) == pmesh.GetNBE();
   MPI_Allreduce(MPI_IN_PLACE, &bdr_elements_are_faces, 1, MPI_INT, MPI_MIN,
                 pmesh.GetComm());
   if (pmesh.bdr_attributes.Size() == 0 || !bdr_elements_are_faces) { return; }

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   p = std::max(p, mfes->GetMaxElementOrder());
   Vector N;
   nodes->GetTrueDofs(N);

   Array<int> all_domain_attr(pmesh.attributes.Max());
   all_domain_attr = 1;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), 2 * p);

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   const int tvsize = pfes.GetTrueVSize();

   ParGridFunction x(&pfes), y(&pfes);
   Vector X(tvsize), Y(tvsize), Z(tvsize);
   X.Randomize(1);
   x.SetFromTrueDofs(X);

   ConstantCoefficient one(1.0);

   static constexpr int U = 0, Coords = 1;
   const auto in_fds = std::vector
   {
      FieldDescriptor{ U, &pfes },
      FieldDescriptor{ Coords, mfes }
   };
   const auto out_fds = std::vector{ FieldDescriptor{ U, &pfes } };

   using IT = Inputs<Value<U>, Gradient<Coords>, Weight>;
   using OT = Outputs<Value<U>>;
   using DT = Derivatives<U>;

   constexpr int BDIM = DIM - 1;
   const auto *bdr_ir =
      &IntRules.Get(pmesh.GetTypicalFaceGeometry(), 2 * p);

   Array<int> all_bdr_attr(pmesh.bdr_attributes.Max());
   all_bdr_attr = 1;

   auto bdr_mass_qf =
      [] MFEM_HOST_DEVICE(const dscalar_t &u,
                          const tensor<real_t, DIM, BDIM> &J,
                          const real_t &w,
                          dscalar_t &v)
   {
      v = u * weight(J) * w;
   };

   // Checks the action, the direct and the cached derivative, and the
   // transpose of @a dop against @a a, which is symmetric. The inputs of
   // @a dop are U, Coords and the fields @a params.
   const auto check = [&](DifferentiableOperator &dop, ParBilinearForm &a,
                          auto &...params)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.Assemble();
      a.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);

      MultiVector MX{X, N, params...}, MZ{Z};
      dop.Mult(MX, MZ);
      Z -= Y;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));

      for (const bool use_cache : { false, true })
      {
         dop.GetDerivative(U, MX, use_cache)->Mult(MX[0], MZ);
         Z -= Y;
         REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) ==
                 MFEM_Approx(0.0));
      }

      MultiVector direction{X};
      dop.GetDerivative(U, MX)->MultTranspose(direction, Z);
      Z -= Y;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));
   };
   constexpr auto action_kernels =
      DerivativeKernels::Action | DerivativeKernels::PA;

   // Checks the assembled derivative of @a dop and its diagonal against
   // the legacy assembly of @a a
   const auto check_assembly =
      [&](DifferentiableOperator &dop, ParBilinearForm &a)
   {
      a.Assemble();
      a.Finalize();

      MultiVector MX{X, N};
      auto dRdU = dop.GetDerivative(U, MX);

      SparseMatrix *A = nullptr;
      dRdU->Assemble(A);
      TestSameMatrices(*A, a.SpMat());
      TestSameMatrices(a.SpMat(), *A);
      delete A;

      HypreParMatrix *Amfem = a.ParallelAssemble();
      HypreParMatrix *Adfem = nullptr;
      dRdU->Assemble(Adfem);
      TestSameMatrices(*Adfem, *Amfem);
      TestSameMatrices(*Amfem, *Adfem);

      Vector dfem_D(tvsize), mfem_D(tvsize), mfem_D_l;
      dRdU->AssembleDiagonal(dfem_D);
      a.SpMat().GetDiag(mfem_D_l);
      pfes.GetProlongationMatrix()->MultTranspose(mfem_D_l, mfem_D);
      mfem_D -= dfem_D;
      REQUIRE(ParNormlp(mfem_D, infinity(), pmesh.GetComm()) ==
              MFEM_Approx(0.0));
      delete Amfem;
      delete Adfem;
   };

   constexpr auto assembly_kernels =
      DerivativeKernels::AssembleMatrix |
      DerivativeKernels::AssembleDiagonal;

   global_bdr_mass_qf<DIM> global_bdr_qf;

   SECTION("GlobalQF Boundary Action")
   {
      ParBilinearForm a(&pfes);
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddBoundaryIntegrator<GlobalQFBackend, action_kernels>(
         global_bdr_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
      check(dop, a);
   }

   SECTION("GlobalQF Domain and Boundary Action")
   {
      ParBilinearForm a(&pfes);
      a.AddDomainIntegrator(new MassIntegrator(one, ir));
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      global_mf_mass_qf<DIM> global_domain_qf;
      dop.AddDomainIntegrator<GlobalQFBackend, action_kernels>(
         global_domain_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});
      dop.AddBoundaryIntegrator<GlobalQFBackend, action_kernels>(
         global_bdr_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
      check(dop, a);
   }

   SECTION("GlobalQF Boundary Assemble")
   {
      ParBilinearForm a(&pfes);
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddBoundaryIntegrator<GlobalQFBackend, assembly_kernels>(
         global_bdr_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
      check_assembly(dop, a);
   }

   SECTION("Boundary Assemble")
   {
      ParBilinearForm a(&pfes);
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddBoundaryIntegrator<LocalQFBackend, assembly_kernels>(
         bdr_mass_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
      check_assembly(dop, a);
   }

   SECTION("Domain and Boundary Assemble")
   {
      ParBilinearForm a(&pfes);
      a.AddDomainIntegrator(new MassIntegrator(one, ir));
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      local_mf_mass_qf<DIM> domain_mass_qf;
      dop.AddDomainIntegrator<LocalQFBackend, assembly_kernels>(
         domain_mass_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});
      dop.AddBoundaryIntegrator<LocalQFBackend, assembly_kernels>(
         bdr_mass_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
      check_assembly(dop, a);
   }

   SECTION("Two Domain and one Boundary Assemble")
   {
      // The domain integrators add up their element matrices before the
      // fill, see detail::EntityMatrices.
      ParBilinearForm a(&pfes);
      a.AddDomainIntegrator(new MassIntegrator(one, ir));
      a.AddDomainIntegrator(new MassIntegrator(one, ir));
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      local_mf_mass_qf<DIM> domain_mass_qf;
      for (int i = 0; i < 2; i++)
      {
         dop.AddDomainIntegrator<LocalQFBackend, assembly_kernels>(
            domain_mass_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});
      }
      dop.AddBoundaryIntegrator<LocalQFBackend, assembly_kernels>(
         bdr_mass_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
      check_assembly(dop, a);
   }

   SECTION("Boundary Action")
   {
      ParBilinearForm a(&pfes);
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddBoundaryIntegrator<LocalQFBackend, action_kernels>(
         bdr_mass_qf, IT{}, OT{}, *bdr_ir,
         all_bdr_attr, DT{});
      check(dop, a);
   }

   SECTION("Domain and Boundary Action")
   {
      ParBilinearForm a(&pfes);
      a.AddDomainIntegrator(new MassIntegrator(one, ir));
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      local_mf_mass_qf<DIM> domain_mass_qf;
      dop.AddDomainIntegrator<LocalQFBackend, action_kernels>(
         domain_mass_qf, IT{}, OT{}, *ir, all_domain_attr, DT{});
      dop.AddBoundaryIntegrator<LocalQFBackend, action_kernels>(
         bdr_mass_qf, IT{}, OT{}, *bdr_ir,
         all_bdr_attr, DT{});
      check(dop, a);
   }

   SECTION("Domain integrator with a parameter, Boundary Action")
   {
      // A field that only the domain integrator uses, so it should
      // not be restricted by the boudnary entity inside dFEM
      QuadratureSpace qs(pmesh, *ir);
      VectorQuadratureSpace vqs(qs, 1);
      QuadratureFunction rho(vqs);
      rho = 1.0;
      static constexpr int Rho = 2;
      const auto param_in_fds = std::vector
      {
         FieldDescriptor{ U, &pfes },
         FieldDescriptor{ Coords, mfes },
         FieldDescriptor{ Rho, &vqs }
      };

      ParBilinearForm a(&pfes);
      a.AddDomainIntegrator(new MassIntegrator(one, ir));
      a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.Assemble();
      a.Mult(x, y);
      pfes.GetProlongationMatrix()->MultTranspose(y, Y);

      auto domain_rho_mass_qf =
         [] MFEM_HOST_DEVICE(const dscalar_t &u,
                             const tensor<real_t, DIM, DIM> &J,
                             const real_t &w,
                             const real_t &rho,
                             dscalar_t &v)
      {
         v = rho * u * w * det(J);
      };

      DifferentiableOperator dop(param_in_fds, out_fds, pmesh);
      dop.AddDomainIntegrator<LocalQFBackend, DerivativeKernels::Action>(
         domain_rho_mass_qf,
         Inputs<Value<U>, Gradient<Coords>, Weight, Identity<Rho>> {},
         OT{}, *ir, all_domain_attr, DT{});
      dop.AddBoundaryIntegrator<LocalQFBackend, DerivativeKernels::Action>(
         bdr_mass_qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});

      MultiVector MX{X, N, rho}, MZ{Z};
      dop.Mult(MX, MZ);
      Z -= Y;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));

      dop.GetDerivative(U, MX)->Mult(MX[0], MZ);
      Z -= Y;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));
   }

   if (pmesh.bdr_attributes.Max() > 1)
   {
      SECTION("Boundary Action on an attribute subset")
      {
         Array<int> bdr_attr(pmesh.bdr_attributes.Max());
         bdr_attr = 0;
         bdr_attr[0] = 1;

         ParBilinearForm a(&pfes);
         a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir), bdr_attr);

         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         SECTION("LocalQF")
         {
            dop.AddBoundaryIntegrator<LocalQFBackend, action_kernels>(
               bdr_mass_qf, IT{}, OT{}, *bdr_ir, bdr_attr, DT{});
         }
         SECTION("GlobalQF")
         {
            dop.AddBoundaryIntegrator<GlobalQFBackend, action_kernels>(
               global_bdr_qf, IT{}, OT{}, *bdr_ir, bdr_attr, DT{});
         }
         check(dop, a);
      }

      SECTION("Boundary Assemble on two attribute subsets")
      {
         // Two boundary integrators on complementary attributes add up
         // their face matrices before the fill.
         Array<int> first(pmesh.bdr_attributes.Max()), rest;
         first = 0;
         first[0] = 1;
         rest = first;
         for (int &r : rest) { r = 1 - r; }

         ParBilinearForm a(&pfes);
         a.AddBoundaryIntegrator(new MassIntegrator(one, bdr_ir));

         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         SECTION("LocalQF")
         {
            for (const Array<int> *attr : { &first, &rest })
            {
               dop.AddBoundaryIntegrator<LocalQFBackend, assembly_kernels>(
                  bdr_mass_qf, IT{}, OT{}, *bdr_ir, *attr, DT{});
            }
         }
         SECTION("GlobalQF")
         {
            for (const Array<int> *attr : { &first, &rest })
            {
               dop.AddBoundaryIntegrator<GlobalQFBackend, assembly_kernels>(
                  global_bdr_qf, IT{}, OT{}, *bdr_ir, *attr, DT{});
            }
         }
         check_assembly(dop, a);
      }
   }

   SECTION("Domain Action with a parameter on an attribute subset")
   {
      // Every test mesh has one element attribute: put odd elements on attribute
      // 2 so that marking attribute 1 selects a strict subset.
      for (int e = 1; e < pmesh.GetNE(); e += 2) { pmesh.SetAttribute(e, 2); }
      pmesh.SetAttributes();
      Array<int> domain_attr({1, 0});

      QuadratureSpace qs(pmesh, *ir);
      VectorQuadratureSpace vqs(qs, 1);
      QuadratureFunction rho(vqs);
      rho.Randomize(2);
      static constexpr int Rho = 2;
      const auto param_in_fds = std::vector
      {
         FieldDescriptor{ U, &pfes },
         FieldDescriptor{ Coords, mfes },
         FieldDescriptor{ Rho, &vqs }
      };
      using RT = Inputs<Value<U>, Gradient<Coords>, Weight, Identity<Rho>>;

      QuadratureFunctionCoefficient rho_coeff(rho);
      ParBilinearForm a(&pfes);
      a.AddDomainIntegrator(new MassIntegrator(rho_coeff, ir), domain_attr);

      DifferentiableOperator dop(param_in_fds, out_fds, pmesh);
      SECTION("LocalQF")
      {
         local_rho_mass_qf<DIM> qf;
         dop.AddDomainIntegrator<LocalQFBackend, action_kernels>(
            qf, RT{}, OT{}, *ir, domain_attr, DT{});
      }
      SECTION("GlobalQF")
      {
         global_rho_mass_qf<DIM> qf;
         dop.AddDomainIntegrator<GlobalQFBackend, action_kernels>(
            qf, RT{}, OT{}, *ir, domain_attr, DT{});
      }
      check(dop, a, rho);
   }
}

// ────────────────────────────────────────────────────────────────────────────
// Boundary mass of an L2 space and of a vector H1 space, against
// the legacy assembly. 
template <int DIM>
void bdr_mass_spaces(const char *filename, int p)
{
   CAPTURE(filename, DIM, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == DIM, "Mesh dimension mismatch");
   smesh.Clear();

   int bdr_elements_are_faces =
      pmesh.GetNFbyType(FaceType::Boundary) == pmesh.GetNBE();
   MPI_Allreduce(MPI_IN_PLACE, &bdr_elements_are_faces, 1, MPI_INT, MPI_MIN,
                 pmesh.GetComm());
   if (pmesh.bdr_attributes.Size() == 0 || !bdr_elements_are_faces) { return; }

   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   p = std::max(p, mfes->GetMaxElementOrder());
   Vector N;
   nodes->GetTrueDofs(N);

   constexpr int BDIM = DIM - 1, VDIM = DIM;
   const auto *bdr_ir = &IntRules.Get(pmesh.GetTypicalFaceGeometry(), 2 * p);
   Array<int> all_bdr_attr(pmesh.bdr_attributes.Max());
   all_bdr_attr = 1;
   ConstantCoefficient one(1.0);

   static constexpr int U = 0, Coords = 1;
   using IT = Inputs<Value<U>, Gradient<Coords>, Weight>;
   using OT = Outputs<Value<U>>;
   using DT = Derivatives<U>;
   constexpr auto kernels =
      DerivativeKernels::Action | DerivativeKernels::AssembleMatrix;

   // Checks the action, the derivative and the assembled derivative of @a dop
   // against the legacy assembly of @a a
   const auto check = [&](DifferentiableOperator &dop, ParBilinearForm &a)
   {
      a.Assemble();
      a.Finalize();
      std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());

      Vector X(A->Width()), Y(A->Height()), Z(A->Height());
      X.Randomize(1);
      A->Mult(X, Y);

      MultiVector MX{X, N}, MZ{Z};
      dop.Mult(MX, MZ);
      Z -= Y;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));

      auto dRdU = dop.GetDerivative(U, MX);
      dRdU->Mult(MX[0], MZ);
      Z -= Y;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));

      SparseMatrix *Ad = nullptr;
      dRdU->Assemble(Ad);
      TestSameMatrices(*Ad, a.SpMat());
      TestSameMatrices(a.SpMat(), *Ad);
      delete Ad;
   };

   SECTION("L2 Boundary")
   {
      L2_FECollection fec(p, DIM, BasisType::GaussLobatto);
      ParFiniteElementSpace fes(&pmesh, &fec);
      const std::vector<FieldDescriptor> in_fds{{U, &fes}, {Coords, mfes}};
      const std::vector<FieldDescriptor> out_fds{{U, &fes}};

      ParBilinearForm a(&fes);
      auto *integ = new BoundaryMassIntegrator(one);
      integ->SetIntRule(bdr_ir);
      a.AddBdrFaceIntegrator(integ);

      SECTION("LocalQF")
      {
         auto qf = [] MFEM_HOST_DEVICE(const dscalar_t &u,
                                       const tensor<real_t, DIM, BDIM> &J,
                                       const real_t &w, dscalar_t &v)
         {
            v = u * weight(J) * w;
         };
         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         dop.AddBoundaryIntegrator<LocalQFBackend, kernels>(
            qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
         check(dop, a);
      }

      SECTION("GlobalQF")
      {
         global_bdr_mass_qf<DIM> qf;
         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         dop.AddBoundaryIntegrator<GlobalQFBackend, kernels>(
            qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
         check(dop, a);
      }
   }

   SECTION("Vector Boundary")
   {
      H1_FECollection fec(p, DIM);
      ParFiniteElementSpace fes(&pmesh, &fec, VDIM);
      const std::vector<FieldDescriptor> in_fds{{U, &fes}, {Coords, mfes}};
      const std::vector<FieldDescriptor> out_fds{{U, &fes}};

      ParBilinearForm a(&fes);
      auto *integ = new VectorMassIntegrator(one, bdr_ir);
      integ->SetVDim(VDIM);
      a.AddBoundaryIntegrator(integ);

      SECTION("LocalQF")
      {
         auto qf = [] MFEM_HOST_DEVICE(const tensor<dscalar_t, VDIM> &u,
                                       const tensor<real_t, DIM, BDIM> &J,
                                       const real_t &w,
                                       tensor<dscalar_t, VDIM> &v)
         {
            v = u * weight(J) * w;
         };
         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         dop.AddBoundaryIntegrator<LocalQFBackend, kernels>(
            qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
         check(dop, a);
      }

      SECTION("GlobalQF")
      {
         global_bdr_vector_mass_qf<DIM, VDIM> qf;
         DifferentiableOperator dop(in_fds, out_fds, pmesh);
         dop.AddBoundaryIntegrator<GlobalQFBackend, kernels>(
            qf, IT{}, OT{}, *bdr_ir, all_bdr_attr, DT{});
         check(dop, a);
      }
   }
}

// ────────────────────────────────────────────────────────────────────────────
// Boundary flux of a vector field against each test function, using
// normal(J, s) with the sign of the Orientation input, checked through the
// divergence theorem:
//    ∫_∂Ω (F·n) v ds = ∫_Ω div(F) v + F·∇v dx.

// Field: F(x) = [xy + 1, y - z^2, x + 2z], div(F) = y + 3
// used for the MFEM coefficient and by the dFEM qfunctions
MFEM_HOST_DEVICE inline tensor<real_t, 3> FluxField(const tensor<real_t, 3> &x)
{
   return {x[0]*x[1] + 1.0, x[1] - x[2]*x[2], x[0] + 2.0*x[2]};
}

struct global_bdr_flux_qf
{
   void operator()(tensor_array<const real_t, 3> &x,
                   tensor_array<const real_t, 3, 2> &J,
                   tensor_array<const real_t> &w,
                   tensor_array<const real_t> &s,
                   tensor_array<real_t> &v) const
   {
      mfem::forall<UseEnzyme>(w.size(), [=] MFEM_HOST_DEVICE (int q)
      {
         const real_t s_q = s(q);
         v(q) = dot(FluxField(x(q)), normal(J(q), s_q)) * weight(J(q)) * w(q);
      });
   }
};

void bdr_normal_flux(const char *filename, int p)
{
   CAPTURE(filename, p);

   Mesh smesh(filename);
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   MFEM_VERIFY(pmesh.Dimension() == 3, "Mesh dimension mismatch");
   smesh.Clear();

   int bdr_elements_are_faces =
      pmesh.GetNFbyType(FaceType::Boundary) == pmesh.GetNBE();
   MPI_Allreduce(MPI_IN_PLACE, &bdr_elements_are_faces, 1, MPI_INT, MPI_MIN,
                 pmesh.GetComm());
   if (pmesh.bdr_attributes.Size() == 0 || !bdr_elements_are_faces) { return; }

   ///<--- Preamble
   pmesh.EnsureNodes();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   ParFiniteElementSpace *mfes = nodes->ParFESpace();
   Vector N;
   nodes->GetTrueDofs(N);

   H1_FECollection fec(p, 3);
   ParFiniteElementSpace pfes(&pmesh, &fec);

   // F is quadratic, so both sides are integrated exactly on affine meshes.
   const int order = p + 2 * mfes->GetMaxElementOrder() + 2;
   const auto *ir = &IntRules.Get(pmesh.GetTypicalElementGeometry(), order);
   const auto *bdr_ir = &IntRules.Get(pmesh.GetTypicalFaceGeometry(), order);
   Array<int> all_bdr_attr(pmesh.bdr_attributes.Max());
   all_bdr_attr = 1;

   ///<--- MFEM setup
   // Right-hand side of the divergence theorem, with F from FluxField()
   auto f_coef = [](const Vector &x, Vector &f)
   {
      const auto fx = FluxField({x(0), x(1), x(2)});
      for (int i = 0; i < 3; i++) { f(i) = fx[i]; }
   };
   VectorFunctionCoefficient F_coeff(3, f_coef);
   FunctionCoefficient divF_coeff([](const Vector &x) { return x(1) + 3.0; });

   ParLinearForm b(&pfes);
   b.AddDomainIntegrator(new DomainLFIntegrator(divF_coeff, ir));
   b.AddDomainIntegrator(new DomainLFGradIntegrator(F_coeff));
   b.GetDLFI()->Last()->SetIntRule(ir);
   b.Assemble();
   std::unique_ptr<HypreParVector> B(b.ParallelAssemble());

   ///<--- dFEM setup
   static constexpr int U = 0, Coords = 1;
   // normal(J) alone points into the domain on some faces, the Orientation
   // input provides the sign that makes normal(J, s) outward.
   const std::vector<FieldDescriptor> in_fds{{Coords, mfes}};
   const std::vector<FieldDescriptor> out_fds{{U, &pfes}};
   using IT = Inputs<Value<Coords>, Gradient<Coords>, Weight, Orientation>;
   using OT = Outputs<Value<U>>;

   ///<--- Check that dFEM satisfies the divergence theorem for each test function
   ///   ∫_∂Ω (F·n) φ_i ds = ∫_Ω div(F) φ_i + F·∇φ_i dx.
   Vector Z(pfes.GetTrueVSize());
   MultiVector MX{N}, MZ{Z};
   auto check = [&](DifferentiableOperator &dop)
   {
      dop.Mult(MX, MZ);
      Z -= *B;
      REQUIRE(ParNormlp(Z, infinity(), pmesh.GetComm()) == MFEM_Approx(0.0));
   };

   SECTION("LocalQF")
   {
      auto qf = [] MFEM_HOST_DEVICE(const tensor<real_t, 3> &x,
                                    const tensor<real_t, 3, 2> &J,
                                    const real_t &w,
                                    const real_t &orient,
                                    real_t &v)
      {
         v = dot(FluxField(x), normal(J, orient)) * weight(J) * w;
      };
      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddBoundaryIntegrator<LocalQFBackend>(
         qf, IT{}, OT{}, *bdr_ir, all_bdr_attr);
      check(dop);
   }

   SECTION("GlobalQF")
   {
      global_bdr_flux_qf qf;
      DifferentiableOperator dop(in_fds, out_fds, pmesh);
      dop.AddBoundaryIntegrator<GlobalQFBackend>(
         qf, IT{}, OT{}, *bdr_ir, all_bdr_attr);
      check(dop);
   }
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM Boundary Normal 3D", "[Parallel][dFEM][GPU]")
{
   const auto p = GenAll({1}, {2, 3});
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh" };
   bdr_normal_flux(GenAll(meshs, extra), p);
}

// ────────────────────────────────────────────────────────────────────────────
TEST_CASE("dFEM Boundary Mass 3D", "[Parallel][dFEM][GPU]")
{
   const auto p = GenAll({1}, {2, 3});
   const auto meshs = { "../../data/inline-hex.mesh" };
   const auto extra = { "../../data/fichera.mesh",
                        "../../data/fichera-q3.mesh",
                        "../../data/toroid-hex.mesh",
                        "../../data/periodic-cube.mesh"
                      };
   SECTION("Scalar Mass")
   {
      bdr_mass<3>(GenAll(meshs, extra), p);
   }
   SECTION("Spaces")
   {
      bdr_mass_spaces<3>(GenAll(meshs, extra), p);
   }
}

#endif // MFEM_USE_MPI
