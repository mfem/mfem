// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include <cmath>
#include <cassert>

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor/lor.hpp"

#define MFEM_DEBUG_COLOR 119
#include "general/debug.hpp"

#define MFEM_NVTX_COLOR Lime
#include "general/nvtx.hpp"

Mesh MakeCartesianMesh(int p, int requested_ndof, int dim)
{
   const int ne = std::max(1, (int)std::ceil(requested_ndof / pow(p, dim)));
   if (dim == 2)
   {
      const int nx = sqrt(ne);
      const int ny = ne / nx;
      return Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL);
   }
   else
   {
      const int nx = cbrt(ne);
      const int ny = sqrt(ne / nx);
      const int nz = ne / nx / ny;
      return Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON);
   }
}

struct LORBench
{
   GeometricFactors::FactorFlags DETERMINANTS = GeometricFactors::DETERMINANTS;
   const int p; // polynomial degree
   const int dim; // space dimension
   Mesh mesh;
   const int ne; // number of elements

   H1_FECollection fec_ho;
   FiniteElementSpace fes_ho;

   IntegrationRules irs;
   const IntegrationRule &ir;

   LORDiscretization lor;
   BilinearForm a_ho, a_lor;

   OperatorHandle A_ho, A_lor;

   HYPRE_Int row_starts[2];
   HypreParMatrix *A = nullptr;
   HypreBoomerAMG amg;

   Array<int> ess_dofs;

   const int ndofs;
   double mdof;
   Vector x, y;

   LORBench(int p_, int requested_ndof, int dim_, const std::string &name) :
      p(p_),
      dim(dim_),
      mesh(MakeCartesianMesh(p, requested_ndof, dim)),
      ne(mesh.GetNE()), // might differ from n_requested if not evenly divisible
      fec_ho(p, dim),
      fes_ho(&mesh, &fec_ho),
      irs(0, Quadrature1D::GaussLobatto),
      ir(irs.Get(mesh.GetElementGeometry(0), 1)),
      lor(fes_ho),
      a_ho(&fes_ho),
      a_lor(&lor.GetFESpace()),
      ndofs(fes_ho.GetTrueVSize()),
      mdof(0.0),
      x(ndofs),
      y(ndofs)
   {
      // std::cout << "Requested ndof: "
      //           << std::setw(10) << requested_ndof
      //           << " Actual: "
      //           << std::setw(10) << ndofs << '\n';
      fes_ho.GetBoundaryTrueDofs(ess_dofs);

      a_ho.AddDomainIntegrator(new DiffusionIntegrator(&IntRules.Get(mesh.GetElementGeometry(0), 2*p)));
      a_ho.AddDomainIntegrator(new MassIntegrator(&IntRules.Get(mesh.GetElementGeometry(0), 2*p)));
      a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);

      a_lor.AddDomainIntegrator(new DiffusionIntegrator(&ir));
      a_lor.AddDomainIntegrator(new MassIntegrator(&ir));
      a_lor.SetAssemblyLevel(AssemblyLevel::FULL);

      x.Randomize(1);
      y.Randomize(2);

      // warm up
      if (name == "AssembleHO" || name == "ApplyHO") { AssembleHO(); }
      if (name == "AssembleBatched") { AssembleBatched(); }
      if (name == "AssembleFull") { AssembleFull(); }
      if (name == "AMGSetup" || name == "Vcycle")
      {
         AssembleBatched();
         SparseMatrix &A_serial = lor.GetAssembledMatrix();
         row_starts[0] = 0;
         row_starts[1] = A_serial.Height();
         A = new HypreParMatrix(MPI_COMM_WORLD, A_serial.Height(), row_starts, &A_serial);
         amg.SetOperator(*A);
         amg.SetPrintLevel(0);
      }
      if (name == "Vcycle") { amg.Setup(x,y); }
   }

   void AssembleHO()
   {
      NVTX("AssembleHO");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * ndofs;
      a_ho.Assemble();
      a_ho.FormSystemMatrix(ess_dofs, A_ho);
   }

   void AssembleFull()
   {
      NVTX("AssembleFull");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * ndofs;
      a_lor.Assemble();
      a_lor.FormSystemMatrix(ess_dofs, A_ho);
   }

   void AssembleBatched()
   {
      NVTX("AssembleBatched");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * ndofs;
      lor.AssembleSystem(a_ho, ess_dofs);
   }

   void ApplyHO()
   {
      NVTX("ApplyHO");
      MFEM_DEVICE_SYNC;
      mdof += 1e-6 * ndofs;
      A_ho->Mult(x, y);
   }

   void AMGSetup()
   {
      NVTX("AMG Setup");
      MFEM_DEVICE_SYNC;
      amg.SetOperator(*A);
      amg.Setup(x, y);
   }

   void Vcycle()
   {
      NVTX("Vcycle");
      MFEM_DEVICE_SYNC;
      amg.Mult(x, y);
   }

   ~LORBench() { delete A; }

   // void AllLegacy()
   // {
   //    NVTX("AllLegacy");
   //    MFEM_DEVICE_SYNC;
   //    mdof += 1e-6 * ndofs;
   //    LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
   //    FiniteElementSpace &fes_lo = lor_disc.GetFESpace();
   //    BilinearForm bf_legacy(&fes_lo);
   //    bf_legacy.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
   //    bf_legacy.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
   //    bf_legacy.SetAssemblyLevel(AssemblyLevel::LEGACY);
   //    bf_legacy.Assemble();
   //    // no EliminateVDofs
   // }

   // void AllFull()
   // {
   //    NVTX("AllFull");
   //    MFEM_DEVICE_SYNC;
   //    mdof += 1e-6 * ndofs;
   //    LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
   //    FiniteElementSpace &fes_lo = lor_disc.GetFESpace();
   //    BilinearForm bf_full(&fes_lo);
   //    bf_full.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
   //    bf_full.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
   //    bf_full.SetAssemblyLevel(AssemblyLevel::FULL);
   //    bf_full.Assemble();
   //    // no EliminateVDofs
   // }

   // void AllBatched()
   // {
   //    NVTX("AllBatched");
   //    MFEM_DEVICE_SYNC;
   //    mdof += 1e-6 * ndofs;
   //    LORDiscretization lor_disc(fes_ho, BasisType::GaussLobatto);
   //    BilinearForm bf_ho(&fes_ho);
   //    bf_ho.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff, ir));
   //    bf_ho.AddDomainIntegrator(new MassIntegrator(mass_coeff, ir));
   //    lor_disc.AssembleSystem(bf_ho, ess_tdofs_empty);
   //    // working with no BC
   // }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// The different sides of the mesh
#define LOG_NDOFS bm::CreateDenseRange(7,23,1)

// Dimensions: 2 or 3
#define DIMS bm::CreateDenseRange(2,3,1)

/// Kernels definitions and registrations
#define Benchmark(Name)\
static void Name(bm::State &state){\
   const int p = state.range(0);\
   const int log_ndof = state.range(1);\
   const int requested_ndof = pow(2, log_ndof);\
   const int dim = state.range(2);\
   if (p == 1 && log_ndof >= 21) { state.SkipWithError("Problem size"); return; }\
   if (p == 2 && log_ndof >= 23) { state.SkipWithError("Problem size"); return; }\
   if (p == 3 && log_ndof >= 23) { state.SkipWithError("Problem size"); return; }\
   LORBench lor(p, requested_ndof, dim, #Name);\
   while (state.KeepRunning()) { lor.Name(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["MDof/s"] = bm::Counter(1e-6*lor.ndofs, flags);\
   state.counters["dofs"] = bm::Counter(lor.ndofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({P_ORDERS, LOG_NDOFS, DIMS})\
            -> Unit(bm::kMillisecond)\
            -> Iterations(10);

Benchmark(AssembleHO)
Benchmark(AssembleFull)
Benchmark(AssembleBatched)

Benchmark(ApplyHO)
Benchmark(AMGSetup)
Benchmark(Vcycle)

int main(int argc, char *argv[])
{
   Mpi::Init();

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   if (bmi::global_context != nullptr)
   {
      const auto device = bmi::global_context->find("device");
      if (device != bmi::global_context->end())
      {
         mfem::out << device->first << " : " << device->second << std::endl;
         device_config = device->second;
      }
   }
   Device device(device_config.c_str());
   device.Print();

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }
   bm::RunSpecifiedBenchmarks(&CR);
   return 0;
}

#endif // MFEM_USE_BENCHMARK
