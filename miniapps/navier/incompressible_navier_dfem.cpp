// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
using namespace mfem;
using namespace mfem::future;

#include "linalg/tensor.hpp"
using mfem::future::tensor;

#define NVTX_COLOR ::gpu::nvtx::kOrchid
#include "incompressible_navier_nvtx.hpp"

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
void DiffusionSetup(const ParFiniteElementSpace &sfes,
                    const IntegrationRule &ir,
                    const Array<int> &domain_attributes,
                    ParameterFunction &qdata)
{
   NVTX_MARK_FUNCTION;
   auto pmesh = sfes.GetParMesh();
   auto nodes = static_cast<ParGridFunction *>(pmesh->GetNodes());
   auto mfes = nodes->ParFESpace();

   constexpr int U = 0, Ξ = 1, Δ = 2;

   DifferentiableOperator dop(
   {{ U, &sfes }},
   {
      {  { Ξ, mfes },
         { Δ, &qdata.GetParameterSpace() }
      }
   },
   *pmesh);
   const auto qfunc =
      [] MFEM_HOST_DEVICE(const tensor<real_t, DIM, DIM> &J,
                          const real_t &w)
   {
      auto invJ = inv(J);
      tensor<real_t, DIM, DIM> C{};
      C(0, 0) = M_PI;
      C(0, 1) = 0, C(1, 0) = 0;
      assert(C(0, 1) == 0 && C(1, 0) == 0); // diff otherwise
      C(1, 1) = 1.0 / M_PI;
      return tuple{ C * invJ * transpose(invJ) * det(J) * w };
   };
   dop.AddDomainIntegrator(qfunc,
                           tuple{ Gradient<Ξ>{}, Weight{} }, // inputs
                           tuple{ Identity<Δ>{} },           // outputs
                           ir, domain_attributes);
   dop.SetParameters({ nodes, &qdata });

   Vector unused(sfes.GetTrueVSize());
   dop.Mult(unused, qdata);
   qdata.HostRead();
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
void DiffusionApply(const ParFiniteElementSpace &sfes,
                    const IntegrationRule &ir,
                    const Array<int> &domain_attributes,
                    ParameterFunction &qdata,
                    const Vector &x, Vector &y)
{
   NVTX_MARK_FUNCTION;
   auto pmesh = sfes.GetParMesh();
   constexpr int U = 0, Q = 1;
   auto qd_ps = &qdata.GetParameterSpace();
   DifferentiableOperator dop({ { U, &sfes } }, { { Q, qd_ps } }, *pmesh);
   const auto qfunc =[] MFEM_HOST_DEVICE(const tensor<real_t, DIM> &∇u,
                                         const tensor<real_t, DIM, DIM> &Q)
   {
      return tuple{ Q * ∇u };
   };
   dop.AddDomainIntegrator(
      qfunc,
      tuple{ Gradient<U>{}, Identity<Q>{} },
      tuple{ Gradient<U>{} },
      ir, domain_attributes);
   dop.SetParameters({ &qdata });
   dop.Mult(x, y);
   y.HostRead();
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
int DiffVerification(ParFiniteElementSpace &h1fes,
                     const IntegrationRule &ir,
                     const Vector &qdata,
                     const Vector &x, const Vector &y)
{
   NVTX_MARK_FUNCTION;
   constexpr real_t ϵ = 1e-12;

   MatrixFunctionCoefficient matrix_coeff(
      DIM, [](const Vector &, DenseMatrix &C)
   {
      C.SetSize(DIM);
      C(0, 0) = M_PI;
      C(0, 1) = 0, C(1, 0) = 0;
      assert(C(0, 1) == 0 && C(1, 0) == 0); // diff otherwise
      C(1, 1) = 1.0 / M_PI;
   });

   ParBilinearForm a(&h1fes);

   auto diff_integ = new DiffusionIntegrator(matrix_coeff);
   diff_integ->SetIntRule(&ir);
   a.AddDomainIntegrator(diff_integ);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   OperatorPtr A;
   a.Assemble(), a.Finalize();
   a.FormSystemMatrix(Array<int> {}, A);

   Vector y2(h1fes.TrueVSize());
   y2 = 0.0;
   A->Mult(x, y2);
   y2.HostRead();

   Vector diff(y2);
   diff -= y;
   const auto diff_norm = diff.Norml2();
   if (diff_norm > ϵ)
   {
      dbg("\x1B[31m||dFdu_FD u^* - ex||_l2 = {}", diff_norm);
      return EXIT_FAILURE;
   }

   dbg("\x1B[32m||dFdu_FD u^* - ex||_l2 = {}", diff_norm);
   return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
void MassApply(const ParFiniteElementSpace &sfes,
               const IntegrationRule &ir,
               const Array<int> &domain_attributes,
               const Vector &x, Vector &y)
{
   dbg();
   auto &pmesh = *sfes.GetParMesh();
   auto *nodes = static_cast<ParGridFunction *>(pmesh.GetNodes());
   auto *mfes = nodes->ParFESpace();
   constexpr int U = 0, Coords = 1;
   DifferentiableOperator dop({{ U, &sfes }}, {{ Coords, mfes }}, pmesh);
   const auto mf_mass_qf =
      [](const real_t &dudxi,
         const tensor<real_t, DIM, DIM> &J,
         const real_t &w)
   {
      return tuple{ dudxi * w * det(J) };
   };
   dop.AddDomainIntegrator(
      mf_mass_qf,
      tuple{ Value<U>{}, Gradient<Coords>{}, Weight{} },
      tuple{ Value<U>{} },
      ir, domain_attributes);
   dop.SetParameters({ nodes });
   // Vector X(sfes.GetTrueVSize()), Y(sfes.GetTrueVSize());
   // sfes.GetRestrictionMatrix()->Mult(x, X);
   // dop.Mult(X, Y);
   dop.Mult(x, y);
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
int MassVerification(ParFiniteElementSpace &h1fes,
                     const IntegrationRule &ir,
                     const Vector &x, Vector &y)
{
   ParBilinearForm a(&h1fes);
   auto mass_integ = new MassIntegrator;
   mass_integ->SetIntRule(&ir);
   a.AddDomainIntegrator(mass_integ);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble(), a.Finalize();
   Vector y2(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2.HostRead();

   Vector diff(y2);
   diff -= y;

   const auto diff_norm = diff.Norml2();
   constexpr real_t ϵ = 1e-12;
   if (diff_norm > ϵ)
   {
      dbg("\x1B[31m||dFdu_FD u^* - ex||_l2 = {}", diff_norm);
      return EXIT_FAILURE;
   }
   dbg("\x1B[32m||dFdu_FD u^* - ex||_l2 = {}", diff_norm);
   return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM>
void VectorDiffApply(const ParFiniteElementSpace &vfes,
                     const IntegrationRule &ir,
                     const Array<int> &domain_attributes,
                     const Vector &x,
                     Vector &y)
{
   NVTX_MARK_FUNCTION;
   auto pmesh = vfes.GetParMesh();
   auto nodes = static_cast<ParGridFunction *>(pmesh->GetNodes());
   auto mfes = nodes->ParFESpace();
   constexpr int U = 0, Coords = 1;
   DifferentiableOperator dop({{ U, &vfes }}, {{ Coords, mfes }}, *pmesh);
   const auto qfunc =
      [] MFEM_HOST_DEVICE(const tensor<real_t, DIM, DIM> &∇u,
                          const tensor<real_t, DIM, DIM> &J, const real_t &w)
   {
      return tuple{ ∇u * inv(J) * det(J) * w * transpose(inv(J)) };
   };
   dop.AddDomainIntegrator(qfunc,
                           tuple{ Gradient<U>{}, Gradient<Coords>{}, Weight{} },
                           tuple{ Gradient<U>{} },
                           ir, domain_attributes);
   dop.SetParameters({ nodes });
   dop.Mult(x, y);
   y.HostRead();
}

///////////////////////////////////////////////////////////////////////////////
template <int DIM, int VDIM = DIM>
int VectorDiffVerif(ParFiniteElementSpace &h1fes,
                    const IntegrationRule &ir, const Vector &x,
                    Vector &y)
{
   NVTX_MARK_FUNCTION;
   ParBilinearForm a(&h1fes);
   auto A_integ = new VectorDiffusionIntegrator(VDIM);
   A_integ->SetIntRule(&ir);
   a.AddDomainIntegrator(A_integ);
   a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a.Assemble(), a.Finalize();
   Vector y2(h1fes.TrueVSize());
   a.Mult(x, y2);
   y2.HostRead();

   Vector diff(y2);
   diff -= y;

   const auto diff_norm = diff.Norml2();
   constexpr real_t ϵ = 1e-12;
   if (diff_norm > ϵ)
   {
      dbg("\x1B[31m||dFdu_FD u^* - ex||_l2 = {}", diff_norm);
      return EXIT_FAILURE;
   }
   dbg("\x1B[32m||dFdu_FD u^* - ex||_l2 = {}", diff_norm);
   return EXIT_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) try
{
   NVTX_MARK_FUNCTION;
   constexpr int DIM = 2, VDIM = DIM;

   static mfem::MPI_Session mpi(argc, argv);
   const int myid = mpi.WorldRank();
   Hypre::Init();

   const char *device_config = "cpu";
   const char *mesh_file = "none";
   int serial_refinements = 0;
   int nx = 1, ny = 1;
   int p = 1;
   bool visualization = false;
   bool pa = false;

   std::cout.precision(8);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&p, "-o", "--order", "Finite element order.");
   args.AddOption(&serial_refinements, "-sr", "--serial-refinements",
                  "Number serial refinements.");
   args.AddOption(&nx, "-nx", "--nx", "Number of elements in X.");
   args.AddOption(&ny, "-ny", "--ny", "Number of elements in Y.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly",
                  "Enable or disable partial assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(mfem::out); }
      return EXIT_FAILURE;
   }
   if (myid == 0) { args.PrintOptions(mfem::out); }

   Mesh smesh;
   if (std::string(mesh_file) != "none")
   {
      smesh = Mesh(mesh_file);
   }
   else
   {
      const real_t sx = 3.0, sy = 1.0;
      const bool generate_edges = true;
      const auto QUAD = Element::QUADRILATERAL;
      smesh = Mesh::MakeCartesian2D(nx, ny, QUAD, generate_edges, sx, sy);
   }
   MFEM_ASSERT(smesh.Dimension() == 2, "2D mesh required!");

   for (int i = 0; i < serial_refinements; ++i) { smesh.UniformRefinement(); }

   dbg("Number of elements: {}", smesh.GetNE());
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   smesh.Clear();

   pmesh.EnsureNodes();
   pmesh.SetCurvature(p);
   assert(DIM == pmesh.Dimension());

   Array<int> domain_attributes;
   if (pmesh.attributes.Size() > 0)
   {
      domain_attributes.SetSize(pmesh.attributes.Max());
      domain_attributes = 1;
   }

   H1_FECollection fec(p, DIM);
   ParFiniteElementSpace fes(&pmesh, &fec), vfes(&pmesh, &fec, VDIM);
   dbg("#dofs:{} ", fes.GetTrueVSize());

   const auto &fe = *fes.GetFE(0);
   const auto &ir =
      IntRules.Get(fe.GetGeomType(), fe.GetOrder() + fe.GetOrder() + fe.GetDim() - 1);

   dbg("#ndof per el = {}", fe.GetDof());
   dbg("#nqp = {}", ir.GetNPoints());
   dbg("#q1d = {}", (int)floor(pow(ir.GetNPoints(), 1.0 / DIM) + 0.5));

   ParGridFunction f1_gf(&fes);

   auto f1 = [](const Vector &coords)
   {
      assert(DIM == 2);
      const double x = coords(0), y = coords(1);
      return M_PI + x + x * x + x * y + y;
   };
   FunctionCoefficient f1_c(f1);
   f1_gf.ProjectCoefficient(f1_c);

   Vector x(f1_gf), y(fes.GetTrueVSize());

   UniformParameterSpace qd_ps(pmesh, ir, DIM * DIM);
   ParameterFunction qdata(qd_ps);

   dbg("Diffusion setup, apply & verification");
   DiffusionSetup<DIM>(fes, ir, domain_attributes, qdata);
   DiffusionApply<DIM>(fes, ir, domain_attributes, qdata, x, y);
   if (DiffVerification<DIM>(fes, ir, qdata, x, y) != EXIT_SUCCESS) { return EXIT_FAILURE; }

   dbg("Mass apply & verification");
   MassApply<DIM>(fes, ir, domain_attributes, x, y);
   if (MassVerification<DIM>(fes, ir, x, y) != EXIT_SUCCESS) { return EXIT_FAILURE; }

   dbg("Vector diffusion apply");
   VectorFunctionCoefficient vf1_c(VDIM, [](const Vector &coords, Vector &u)
   {
      assert(DIM == 2);
      const double x = coords(0), y = coords(1);
      u(0) = M_PI + 0.25 * x * x * y + y * y * x;
      u(1) = M_PI - 0.25 * x * y * y + y * x * x;
   });
   ParGridFunction vf1_gf(&vfes);
   vf1_gf.ProjectCoefficient(vf1_c);
   Vector vx(vf1_gf), vy(vfes.GetTrueVSize());
   VectorDiffApply<DIM>(vfes, ir, domain_attributes, vx, vy);
   if (VectorDiffVerif<DIM>(vfes, ir, vx, vy) != EXIT_SUCCESS) { return EXIT_FAILURE; }

   return EXIT_SUCCESS;
}
catch (std::exception &e)
{
   std::cerr << "\033[31m..xxxXXX[ERROR]XXXxxx.." << std::endl;
   std::cerr << "\033[31m{}" << e.what() << std::endl;
   return EXIT_FAILURE;
}
