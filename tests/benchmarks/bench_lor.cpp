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

#include "fem/lor/lor_batched.hpp"
#include "fem/lor/lor_ads.hpp"
#include "fem/lor/lor_ams.hpp"

#ifdef MFEM_USE_BENCHMARK

#include "fem/lor/lor.hpp"

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
      const bool sfc_ordering = p > 1;
      return Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON,
                                   1.0,1.0,1.0, sfc_ordering);
   }
}

ParMesh MakeParCartesianMesh(int p, int requested_ndof, int dim)
{
   Mesh mesh = MakeCartesianMesh(p, requested_ndof, dim);
   return ParMesh(MPI_COMM_WORLD, mesh);
}

struct RT_LORBench
{
   ParMesh mesh;
   RT_FECollection fec_ho;
   ParFiniteElementSpace fes_ho;

   BatchedLORAssembly lor;
   BatchedLOR_ADS ads;
   OperatorHandle A_lor;

   LORSolver<HypreADS> *amg = nullptr;

   ParBilinearForm a_ho;
   Array<int> ess_dofs;

   const int ndofs;
   double mdof;

   Vector x, y;

   RT_LORBench(int p, int requested_ndof, int dim, const std::string &name) :
      mesh(MakeParCartesianMesh(p, requested_ndof, dim)),
      fec_ho(p - 1, dim, BasisType::GaussLobatto, BasisType::IntegratedGLL),
      fes_ho(&mesh, &fec_ho),
      lor(fes_ho),
      ads(fes_ho, lor.GetLORVertexCoordinates()),
      a_ho(&fes_ho),
      ndofs(fes_ho.GetTrueVSize()),
      mdof(0.0)
   {
      fes_ho.GetBoundaryTrueDofs(ess_dofs);

      a_ho.AddDomainIntegrator(new VectorFEMassIntegrator);
      a_ho.AddDomainIntegrator(new DivDivIntegrator);

      if (name == "RTAssembleBatched") { RTAssembleBatched(); }
      if (name == "RTDiscreteCurl") { RTDiscreteCurl(); }
      if (name == "ADSApply")
      {
         amg = new LORSolver<HypreADS>(a_ho, ess_dofs);
         x.SetSize(ndofs);
         y.SetSize(ndofs);
         ADSApply();
      }
   }

   void RTAssembleBatched()
   {
      NVTX("RTAssembleBatched");
      MFEM_DEVICE_SYNC;

      lor.AssembleWithoutBC(a_ho, A_lor);
      A_lor.As<SparseMatrix>()->EliminateBC(ess_dofs,
                                            Operator::DiagonalPolicy::DIAG_KEEP);
   }

   void RTDiscreteCurl()
   {
      NVTX("DiscreteCurl");
      MFEM_DEVICE_SYNC;

      ads.FormCurlMatrix();
   }

   void ADSApply()
   {
      NVTX("ADSApply");
      MFEM_DEVICE_SYNC;
      amg->Mult(x, y);
   }

   ~RT_LORBench() { delete amg; }
};

struct ND_LORBench
{
   ParMesh mesh;
   ND_FECollection fec_ho;
   ParFiniteElementSpace fes_ho;

   BatchedLORAssembly lor;
   BatchedLOR_AMS ams;
   OperatorHandle A_lor;

   LORSolver<HypreAMS> *amg = nullptr;

   ParBilinearForm a_ho;
   Array<int> ess_dofs;

   Vector evec, xvert;

   const int ndofs;
   double mdof;

   Vector x, y;

   ND_LORBench(int p, int requested_ndof, int dim, const std::string &name) :
      mesh(MakeParCartesianMesh(p, requested_ndof, dim)),
      fec_ho(p, dim, BasisType::GaussLobatto, BasisType::IntegratedGLL),
      fes_ho(&mesh, &fec_ho),
      lor(fes_ho),
      ams(fes_ho, lor.GetLORVertexCoordinates()),
      a_ho(&fes_ho),
      ndofs(fes_ho.GetTrueVSize()),
      mdof(0.0)
   {
      fes_ho.GetBoundaryTrueDofs(ess_dofs);

      a_ho.AddDomainIntegrator(new VectorFEMassIntegrator);
      a_ho.AddDomainIntegrator(new CurlCurlIntegrator);

      if (name == "NDAssembleBatched") { NDAssembleBatched(); }
      if (name == "DiscreteGradient") { DiscreteGradient(); }
      if (name == "CoordinateVectors") { CoordinateVectors(); }
      if (name == "VertexCoordinates") { VertexCoordinates(); }
      if (name == "AMSApply")
      {
         amg = new LORSolver<HypreAMS>(a_ho, ess_dofs);
         x.SetSize(ndofs);
         y.SetSize(ndofs);
         AMSApply();
      }
   }

   void NDAssembleBatched()
   {
      NVTX("NDAssembleBatched");
      MFEM_DEVICE_SYNC;

      lor.AssembleWithoutBC(a_ho, A_lor);
      A_lor.As<SparseMatrix>()->EliminateBC(ess_dofs,
                                            Operator::DiagonalPolicy::DIAG_KEEP);
   }

   void DiscreteGradient()
   {
      NVTX("DiscreteGradient");
      MFEM_DEVICE_SYNC;
      MFEM_ABORT("Not available anymore, as part of FormGradientMatrix now!");
      // ams.FormGradientMatrixLocal();
   }

   void CoordinateVectors()
   {
      NVTX("CoordinateVectors");
      MFEM_DEVICE_SYNC;

      ams.FormCoordinateVectors(lor.GetLORVertexCoordinates());
   }

   void VertexCoordinates()
   {
      NVTX("VertexCoordinates");
      MFEM_DEVICE_SYNC;

      MFEM_ABORT("Not available anymore, API has changed!");
      BatchedLORAssembly::FormLORVertexCoordinates(fes_ho, xvert);
   }

   void AMSApply()
   {
      NVTX("AMSApply");
      MFEM_DEVICE_SYNC;
      amg->Mult(x, y);
   }

   ~ND_LORBench() { delete amg; }
};

struct H1_LORBench
{
   Mesh mesh;

   H1_FECollection fec_ho;
   FiniteElementSpace fes_ho;

   IntegrationRules irs;
   const IntegrationRule &ir;

   LORDiscretization lor;
   BatchedLORAssembly lor_b;
   BilinearForm a_ho, a_lor;

   OperatorHandle A_ho, A_lor;

   HYPRE_Int row_starts[2];
   HypreParMatrix *A = nullptr;
   HypreBoomerAMG amg;

   Array<int> ess_dofs;

   const int ndofs;
   double mdof;
   Vector x, y;

   H1_LORBench(int p, int requested_ndof, int dim, const std::string &name) :
      mesh(MakeCartesianMesh(p, requested_ndof, dim)),
      fec_ho(p, dim),
      fes_ho(&mesh, &fec_ho),
      irs(0, Quadrature1D::GaussLobatto),
      ir(irs.Get(mesh.GetElementGeometry(0), 1)),
      lor(fes_ho),
      lor_b(fes_ho),
      a_ho(&fes_ho),
      a_lor(&lor.GetFESpace()),
      ndofs(fes_ho.GetTrueVSize()),
      mdof(0.0)
   {
      // std::cout << "Requested ndof: "
      //           << std::setw(10) << requested_ndof
      //           << " Actual: "
      //           << std::setw(10) << ndofs << '\n';
      fes_ho.GetBoundaryTrueDofs(ess_dofs);

      a_ho.AddDomainIntegrator(new DiffusionIntegrator);
      a_ho.AddDomainIntegrator(new MassIntegrator);
      a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);

      a_lor.AddDomainIntegrator(new DiffusionIntegrator(&ir));
      a_lor.AddDomainIntegrator(new MassIntegrator(&ir));
      a_lor.SetAssemblyLevel(AssemblyLevel::FULL);

      if (name == "ApplyHO" || name =="Vcycle")
      {
         x.SetSize(ndofs);
         y.SetSize(ndofs);
         x.Randomize(1);
         y.Randomize(2);
      }

      // warm up
      if (name == "AssembleHO" || name == "ApplyHO") { AssembleHO(); }
      if (name == "ApplyHO") { ApplyHO(); }
      if (name == "H1AssembleBatched") { H1AssembleBatched(); }
      if (name == "AssembleFull") { AssembleFull(); }

      if (name == "AMGSetup" || name == "Vcycle")
      {
         H1AssembleBatched();
         SparseMatrix &A_serial = lor.GetAssembledMatrix();
         row_starts[0] = 0;
         row_starts[1] = A_serial.Height();

         MFEM_ABORT("Not available anymore, API has changed!");
         // A = new HypreParMatrix(MPI_COMM_WORLD, A_serial.Height(), row_starts, &A_serial);
         amg.SetOperator(*A);
         amg.SetPrintLevel(0);
      }
      if (name == "Vcycle") { amg.Setup(x,y); }
   }

   void AssembleHO()
   {
      NVTX("AssembleHO");
      MFEM_DEVICE_SYNC;
      a_ho.Assemble();
      a_ho.FormSystemMatrix(ess_dofs, A_ho);
   }

   void AssembleFull()
   {
      NVTX("AssembleFull");
      MFEM_DEVICE_SYNC;
      a_lor.Assemble();
      a_lor.FormSystemMatrix(ess_dofs, A_lor);
   }

   void H1AssembleBatched()
   {
      NVTX("AssembleBatched");
      MFEM_DEVICE_SYNC;
      //lor.AssembleSystem(a_ho, ess_dofs);
      lor_b.Assemble(a_ho, ess_dofs, A_lor);
   }

   void ApplyHO()
   {
      NVTX("ApplyHO");
      MFEM_DEVICE_SYNC;
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

   ~H1_LORBench()
   {
      delete A;
   }
};

// The different orders the tests can run
#define P_ORDERS bm::CreateDenseRange(1,8,1)

// The different sides of the mesh
#define LOG_NDOFS bm::CreateDenseRange(7,23,1)

// Dimensions: 2 or 3
#define DIMS bm::CreateDenseRange(2,3,1)

/// Kernels definitions and registrations
#define Benchmark(Class, Name)\
static void Name(bm::State &state){\
   const int p = state.range(2);\
   const int log_ndof = state.range(1);\
   const int requested_ndof = pow(2, log_ndof);\
   const int dim = state.range(0);\
   const std::string name = #Name;\
   if (p == 1 && log_ndof >= 21) { state.SkipWithError("Problem size"); return; }\
   if (p == 2 && log_ndof >= 22) { state.SkipWithError("Problem size"); return; }\
   if (p == 3 && log_ndof >= 22) { state.SkipWithError("Problem size"); return; }\
   if ((name == "ADSApply" || name == "AMSApply") && log_ndof >= 21)\
   { state.SkipWithError("Problem size"); return; }\
   Class lor(p, requested_ndof, dim, name);\
   while (state.KeepRunning()) { lor.Name(); }\
   bm::Counter::Flags flags = bm::Counter::kIsIterationInvariantRate;\
   state.counters["MDof/s"] = bm::Counter(1e-6*lor.ndofs, flags);\
   state.counters["dofs"] = bm::Counter(lor.ndofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(Name)\
            -> ArgsProduct({DIMS, LOG_NDOFS, P_ORDERS})\
-> Unit(bm::kMillisecond);
//  -> Iterations(10);

// H1
Benchmark(H1_LORBench, AssembleHO)
Benchmark(H1_LORBench, AssembleFull)
Benchmark(H1_LORBench, H1AssembleBatched)

#ifdef MFEM_USE_MPI
Benchmark(H1_LORBench, ApplyHO)
Benchmark(H1_LORBench, AMGSetup)
Benchmark(H1_LORBench, Vcycle)

// ND
Benchmark(ND_LORBench, NDAssembleBatched)
Benchmark(ND_LORBench, DiscreteGradient)
Benchmark(ND_LORBench, CoordinateVectors)
Benchmark(ND_LORBench, VertexCoordinates)
Benchmark(ND_LORBench, AMSApply)

// RT
Benchmark(RT_LORBench, RTAssembleBatched)
Benchmark(RT_LORBench, RTDiscreteCurl)
Benchmark(RT_LORBench, ADSApply)
#endif

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
