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
// #include <numeric>

#include "bench.hpp" // IWYU pragma: keep

#ifdef MFEM_USE_BENCHMARK

#include <fem/qinterp/det.cpp>
#include <fem/qinterp/grad.hpp> // IWYU pragma: keep

#include <fem/dfem/doperator.hpp>
#include <linalg/tensor.hpp>

#undef NVTX_COLOR
#define NVTX_COLOR nvtx::kAquamarine
#include "general/nvtx.hpp"

using namespace mfem;
using mfem::internal::tensor;

namespace mfem
{

/// Arguments /////////////////////////////////////////////////////////////////
#ifndef MFEM_USE_HIP
#define MAX_NDOFS 128 * 1024
#else
#define MAX_NDOFS 10 * 1024 * 1024
#endif

static void KerOrderSideArgs(bmi::Benchmark *b)
{
   const auto est = [](int c) { return (c + 1) * (c + 1) * (c + 1); };
   const auto versions = { 1 }; // only one version of the kernel yet
   for (const auto k : versions)
   {
      // for (int p = 6; p >= 1; p -= 1)
      for (int p = 1; p <= 6; p += 1)
      {
         for (int c = 25; est(c) <= MAX_NDOFS; c += 25)
         {
            b->Args({ k, p, c });
         }
      }
   }
}

/// Globals ///////////////////////////////////////////////////////////////////
Device *device_ptr = nullptr;

///////////////////////////////////////////////////////////////////////////////
struct ∂DiffusionIntegrator : public BilinearFormIntegrator
{
   ParMesh *pmesh;
   ParGridFunction *nodes;
   const ParFiniteElementSpace *pfes, *mesh_pfes;
   int P1d, Q1d;

   static constexpr int U = 0, Ξ = 1; // potential, coordinates

public:

   ////////////////////////////////////////////////////////////////////////////
   ∂DiffusionIntegrator() = default;

   ////////////////////////////////////////////////////////////////////////////
   void AddMultPA(const Vector &x, Vector &y) const override
   {
      dbg();

      const auto p = pfes->GetFE(0)->GetOrder();
      // const auto q = 2 * p + pmesh->GetElementTransformation(0)->OrderW();
      const auto q = 2 * p + 3;
      const auto type = pmesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);

      static auto solutions = std::vector{FieldDescriptor{U, pfes}};
      static auto parameters = std::vector{FieldDescriptor{Ξ, mesh_pfes}};
      // static const std::vector<FieldDescriptor> solutions = { { U, pfes } };
      // static const std::vector<FieldDescriptor> parameters = { { Ξ, mesh_pfes } };
      static DifferentiableOperator ∂_op(solutions, parameters, *pmesh);

      static auto input_operators = mfem::tuple{Gradient<U>{}, Gradient<Ξ>{}, Weight{}};
      static auto output_operator = mfem::tuple{Gradient<U>{}};

      auto qDiffusionMF = [] (const tensor<real_t, 3> &∇u,
                              const tensor<real_t, 3, 3> &J,
                              const real_t &w)
      {
         const auto invJ = inv(J);
         // return mfem::tuple{(invJ * transpose(invJ)) * ∇u} * det(J) * w;
         return mfem::tuple{((∇u * invJ)) * transpose(invJ) * det(J) * w};
      };

      static bool setup = true;
      if (setup)
      {
         dbg("\x1b[33mSetup");
         ∂_op.AddDomainIntegrator(qDiffusionMF,
                                    input_operators, // mfem::tuple{ Gradient<U>{}, Gradient<Ξ>{}, Weight{} },
                                    output_operator, // mfem::tuple{ Gradient<U>{} },
                                    ir);
         ∂_op.SetParameters({ nodes });
         setup = false;
      }

      ∂_op.Mult(x, y);
      // assert(false);
   }

   ////////////////////////////////////////////////////////////////////////////
   void AssemblePA(const FiniteElementSpace &fes) override
   {
      dbg();
      pfes = dynamic_cast<const ParFiniteElementSpace*>(&fes);
      assert(pfes);

      pmesh = pfes->GetParMesh();
      nodes = static_cast<ParGridFunction *>(pmesh->GetNodes());
      assert(nodes);

      mesh_pfes = nodes->ParFESpace();
      assert(mesh_pfes);

      const auto p = pfes->GetFE(0)->GetOrder();
      // const auto q = 2 * p + pmesh->GetElementTransformation(0)->OrderW();
      const auto q = 2 * p + 3;
      dbg("p:{} q:{}", p, q);
      const auto type = pmesh->GetElementBaseGeometry(0);
      const IntegrationRule &ir = IntRules.Get(type, q);

      P1d = p + 1;
      Q1d = IntRules.Get(Geometry::SEGMENT, ir.GetOrder()).GetNPoints();
      dbg("P1d:{} Q1d:{} ", P1d, Q1d);
      // assert(false);

      // constexpr int DIM = 3;
      // const int spatial_dim = DIM,
      //           local_size = DIM * DIM,
      //           element_size = ir.GetNPoints(),
      //           total_size = DIM * DIM * ir.GetNPoints() * pmesh->GetNE();
      // ParametricSpace qdata_space(spatial_dim, local_size, element_size, total_size);
      // ParametricFunction qdata(qdata_space);
      // dbg("qdata size:{} ", qdata.Size());

#if 0
      constexpr int Ξ = 1, Δ = 2;

      DifferentiableOperator ∂_op(
      { {}},                                     // solutions
      { { Ξ, mesh_pfes }, { Δ, &qdata.space } }, // parameters
      *pmesh);

      auto qSetup =
         [] MFEM_HOST_DEVICE(const tensor<real_t, DIM, DIM> &J,
                             const real_t &w)
      {
         const auto invJ = inv(J);
         return mfem::tuple{ invJ * transpose(invJ) * det(J) * w };
      };
      ∂_op.AddDomainIntegrator(qSetup,                                   // kernel
                                 mfem::tuple{ Gradient<Ξ>{}, Weight{} }, // inputs
                                 mfem::tuple{ None<Δ>{} },               // outputs
                                 ir);
      ∂_op.SetParameters({ nodes, &qdata });

      Vector x(pfes->GetTrueVSize());
      x = 0.0;
      ∂_op.Mult(x, qdata);
      qdata.HostRead();
      assert(false);
#elif 0
      constexpr int U = 0, Ξ = 1, Δ = 2;
      DifferentiableOperator ∂_op(
      { {U, pfes}},                           // solutions
      { { Ξ, mesh_pfes } }, // parameters
      // { { Ξ, mesh_pfes }, { Δ, &qdata.space } }, // parameters
      *pmesh);
      auto qMass = [](const real_t &u,
                      const tensor<real_t, DIM, DIM> &J,
                      const real_t &w)
      {
         return mfem::tuple{ u * w * det(J) };
      };
      ∂_op.AddDomainIntegrator(qMass,
                                 mfem::tuple{ Value<U>{}, Gradient<Ξ>{}, Weight{} }, // inputs
                                 mfem::tuple{ Value<U>{} },                          // outputs
                                 ir);
      ∂_op.SetParameters({ nodes });
#else
      // nothing to do in setup
#endif
   }
};

} // namespace mfem

/// BakeOff ///////////////////////////////////////////////////////////////////
template <int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int DIM = 3;
   const int p, c, q, n, nx, ny, nz;
   const bool check_x, check_y, check_z, checked;
   Mesh smesh;
   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const int dofs;
   ParGridFunction x, y;
   ParBilinearForm a;
   double mdofs{};

   BakeOff(int p, int side):
      p(p), c(side), q(2 * p + (GLL ? -1 : 3)), n((assert(c >= p), c / p)),
      nx(n + (p * (n + 1) * p * n * p * n < c * c * c ? 1 : 0)),
      ny(n + (p * (n + 1) * p * (n + 1) * p * n < c * c * c ? 1 : 0)), nz(n),
      check_x(p * nx * p * ny * p * nz <= c * c * c),
      check_y(p * (nx + 1) * p * (ny + 1) * p * nz > c * c * c),
      check_z(p * (nx + 1) * p * (ny + 1) * p * (nz + 1) > c * c * c),
      checked((assert(check_x &&check_y &&check_z), true)),
      smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON)),
      pmesh(MPI_COMM_WORLD, (smesh.EnsureNodes(), smesh)),
      fec(p, DIM, BasisType::GaussLobatto),
      pfes(&pmesh, &fec, VDIM, Ordering::byNODES),
      geom_type(pmesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q)), one(1.0), uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(pfes.GetTrueVSize()), x(&pfes), y(&pfes), a(&pfes)
   {
      dbg("p:{} q:{}", p, q);
      pmesh.SetCurvature(p);
      smesh.Clear();
      x = 0.0;
   }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Problems //////////////////////////////////////////////////////////////////
template <typename BFI, int VDIM = 1, bool GLL = false>
struct Problem : public BakeOff<VDIM, GLL>
{
   const real_t rtol = 0.0;
   const int max_it = 32, print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ParLinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   using BakeOff<VDIM, GLL>::a;
   using BakeOff<VDIM, GLL>::ir;
   using BakeOff<VDIM, GLL>::one;
   using BakeOff<VDIM, GLL>::pmesh;
   using BakeOff<VDIM, GLL>::pfes;
   using BakeOff<VDIM, GLL>::x;
   using BakeOff<VDIM, GLL>::y;
   using BakeOff<VDIM, GLL>::mdofs;

   Problem(int order, int side):
      BakeOff<VDIM, GLL>(order, side),
      ess_bdr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      cg(MPI_COMM_WORLD)
   {
      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      static_assert(VDIM ==1);
      b.AddDomainIntegrator(new DomainLFIntegrator(this->one));
      b.UseFastAssembly(true);
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI());
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetOperator(*A);
      cg.iterative_mode = false;
      if constexpr (true) // check
      {
         dbg("Check");
         cg.SetPrintLevel(1);
         cg.SetMaxIter(100);
         cg.SetRelTol(1e-8);
         cg.SetAbsTol(0.0);
         cg.Mult(B, X);
         MFEM_VERIFY(cg.GetConverged(), "CG solver did not converge.");
         MFEM_DEVICE_SYNC;
         mfem::out << "✅" << std::endl;
      }
      cg.SetAbsTol(0.0);
      cg.SetRelTol(rtol);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      cg.Mult(B, X);
      MFEM_DEVICE_SYNC;
      mdofs += this->MDofs() * cg.GetNumIterations();
   }
};


///////////////////////////////////////////////////////////////////////////////
static void BP(bm::State &state)
{
   const int version = static_cast<int>(state.range(0));
   const auto order = static_cast<int>(state.range(1));
   const auto side = static_cast<int>(state.range(2));
   Problem<∂DiffusionIntegrator> ker(order, side);
   // Problem<DiffusionIntegrator> ker(order, side);
   // device_ptr->SetKernelsVersion(version);
   // if (k > 1) { device_ptr->EnableFastKernels(); }
   while (state.KeepRunning()) { ker.benchmark(); }
   bm::Counter::Flags flags = bm::Counter::kIsRate;
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags);
   state.counters["Dofs"] = bm::Counter(ker.dofs);
   state.counters["p"] = bm::Counter(order);
   state.counters["version"] = bm::Counter(version);
}
BENCHMARK(BP)->Apply(KerOrderSideArgs)
->Unit(bm::kMillisecond)
   ->Iterations(10);

/// Specializations ///////////////////////////////////////////////////////////
void AddKernelSpecializations()
{
   dbg();
   using Det = QuadratureInterpolator::DetKernels;
   // Det::Specialization<3, 3, 2, 2>::Add();
   // Det::Specialization<3, 3, 2, 3>::Add();
   // Det::Specialization<3, 3, 2, 5>::Add();
   // Det::Specialization<3, 3, 2, 6>::Add();
   Det::Specialization<3, 3, 7, 7>::Add();

   // using Grad = QuadratureInterpolator::GradKernels;
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 7, 8>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 3>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 4>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 5>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 6>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 7>::Add();
   // Grad::Specialization<3, QVectorLayout::byNODES, false, 3, 2, 8>::Add();
}

/// main //////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
   static mfem::MPI_Session mpi(argc, argv);
   // const int myid = mpi.WorldRank();
   // Hypre::Init();

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   AddKernelSpecializations();

   // Device setup, cpu by default
   std::string device_config = "cpu";
   const auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   dbg("device_config: {}", device_config);
   Device device(device_config.c_str());
   device_ptr = &device;
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return EXIT_FAILURE; }

   bm::RunSpecifiedBenchmarks(&CR);

   return EXIT_SUCCESS;
}

#endif // MFEM_USE_BENCHMARK
