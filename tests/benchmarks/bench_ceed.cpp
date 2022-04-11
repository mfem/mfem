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

#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

/*
  This benchmark contains the implementation of the CEED's bake-off problems:
  high-order kernels/benchmarks designed to test and compare the performance
  of high-order codes.

  See: ceed.exascaleproject.org/bps and github.com/CEED/benchmarks
*/

namespace mfem
{

////////////////////////////////////////////////////////////////////////////////
#ifdef MFEM_USE_MPI
static MPI_Session *mpi = nullptr;
#define mpiRoot mpi->Root()
#define mpiWorldSize mpi->WorldSize()
#define mpiWorldRank mpi->WorldRank()
#else
typedef int MPI_Session;
#define mpiRoot true
#define mpiWorldSize 1
#define mpiWorldRank 0
#define GlobalTrueVSize GetVSize
#define HYPRE_Int int
#define MPI_COMM_WORLD
#define ParMesh Mesh
#define GetParMesh GetMesh
#define HypreParMatrix SparseMatrix
#define ParGridFunction GridFunction
#define ParBilinearForm BilinearForm
#define ParLinearForm LinearForm
#define ParFiniteElementSpace FiniteElementSpace
#define ParFiniteElementSpaceHierarchy FiniteElementSpaceHierarchy
#endif

////////////////////////////////////////////////////////////////////////////////
static int config_ndev = 4; // default 4 GPU per node

////////////////////////////////////////////////////////////////////////////////
struct BakeOff
{
   const int p, c, q, n, nx, ny, nz, dim = 3;
   const bool check_x, check_y, check_z, checked;
   std::function<ParMesh()> GetMesh = [&]()
   {
      Mesh smesh = Mesh::MakeCartesian3D(nx,ny,nz, Element::HEXAHEDRON);
#ifdef MFEM_USE_MPI
      ParMesh mesh(MPI_COMM_WORLD, smesh);
#else
      ParMesh mesh(smesh);
#endif // MFEM_USE_MPI
      return mesh;
   };
   ParMesh pmesh;
   H1_FECollection fec;
   ParFiniteElementSpace pfes;
   const Geometry::Type geom_type;
   IntegrationRules IntRulesGLL;
   const IntegrationRule *irGLL;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   const int dofs;
   ParGridFunction x,y;
   ParBilinearForm a;
   double mdofs;

   BakeOff(int p, int side, int vdim, bool gll):
      p(p),
      c(side),
      q(2*p + (gll?-1:3)),
      n((assert(c>=p),c/p)),
      nx(n + (p*(n+1)*p*n*p*n < c*c*c ?1:0)),
      ny(n + (p*(n+1)*p*(n+1)*p*n < c*c*c ?1:0)),
      nz(n),
      check_x(p*nx * p*ny * p*nz <= c*c*c),
      check_y(p*(nx+1) * p*(ny+1) * p*nz > c*c*c),
      check_z(p*(nx+1) * p*(ny+1) * p*(nz+1) > c*c*c),
      checked((assert(check_x && check_y && check_z), true)),
      pmesh(GetMesh()),
      fec(p, dim, BasisType::GaussLobatto),
      pfes(&pmesh, &fec, vdim),
      geom_type(pfes.GetFE(0)->GetGeomType()),
      IntRulesGLL(0, Quadrature1D::GaussLobatto),
      irGLL(&IntRulesGLL.Get(geom_type, q)),
      ir(&IntRules.Get(geom_type, q)),
      one(1.0),
      dofs(pfes.GlobalTrueVSize()),
      x(&pfes),
      y(&pfes),
      a(&pfes),
      mdofs(0.0) { x = 0.0; }

   virtual void benchmark() = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Bake-off Problems (BPs)
template<typename BFI, int VDIM = 1, bool GLL = false>
struct Problem: public BakeOff
{
   const double rtol = 1e-12;
   const int max_it = 32;
   const int print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   ParLinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;

   Problem(int order, int side):
      BakeOff(order,side,VDIM,GLL),
      ess_bdr(pmesh.bdr_attributes.Max()),
      b(&pfes),
      cg(MPI_COMM_WORLD)
   {
      ess_bdr = 1;
      pfes.GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
      b.AddDomainIntegrator(new DomainLFIntegrator(one));
      b.Assemble();

      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, GLL?irGLL:ir));
      a.Assemble();
      a.Mult(x, y);

      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

      cg.SetRelTol(rtol);
      cg.SetOperator(*A);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      cg.iterative_mode = false;
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      cg.Mult(B,X);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs() * cg.GetNumIterations();
   }
};

static void OrderSideArgs(bmi::Benchmark *b)
{
   const auto est = [](int c) { return (c+1)*(c+1)*(c+1); };
   for (int p = 6; p > 0; p--)
   {
      for (int c = 12; est(c) <= 2*1024*1024; c += 6)
      {
         b->Args({p, c});
      }
   }
}

/// Bake-off Problems (BPs)
/// const int nranks = mpiWorldSize;
#define BakeOff_Problem(i,Kernel,VDIM,GLL)\
static void BP##i(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   Problem<Kernel##Integrator,VDIM,GLL> ker(p,side);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   bm::Counter::Flags flags = bm::Counter::kIsRate;\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), flags);\
   state.counters["Dofs"] = bm::Counter(ker.dofs);\
   state.counters["p"] = bm::Counter(p);\
}\
BENCHMARK(BP##i)\
       -> Apply(OrderSideArgs)\
       -> Unit(bm::kMillisecond)\
       -> Iterations(100);

/// BP1: scalar PCG with mass matrix, q=p+2
BakeOff_Problem(1,Mass,1,false)

/// BP2: vector PCG with mass matrix, q=p+2
BakeOff_Problem(2,VectorMass,3,false)

/// BP3: scalar PCG with stiffness matrix, q=p+2
BakeOff_Problem(3,Diffusion,1,false)

/// BP4: vector PCG with stiffness matrix, q=p+2
BakeOff_Problem(4,VectorDiffusion,3,false)

/// BP5: scalar PCG with stiffness matrix, q=p+1
BakeOff_Problem(5,Diffusion,1,true)

/// BP6: vector PCG with stiffness matrix, q=p+1
BakeOff_Problem(6,VectorDiffusion,3,true)


/// Bake-off Kernels (BKs)
template <typename BFI, int VDIM = 1, bool GLL = false>
struct Kernel: public BakeOff
{
   ParGridFunction y;

   Kernel(int order, int side): BakeOff(order, side, VDIM, GLL), y(&pfes)
   {
      x.Randomize(1);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, GLL?irGLL:ir));
      a.Assemble();
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
   }

   void benchmark() override
   {
      a.Mult(x, y);
      MFEM_DEVICE_SYNC;
      mdofs += MDofs();
   }
};

} // namespace mfem

/// Generic CEED BKi
#define BakeOff_Kernel(i,KER,VDIM,GLL)\
static void BK##i(bm::State &state){\
   const int p = state.range(0);\
   const int side = state.range(1);\
   mfem::Kernel<KER##Integrator,VDIM,GLL> ker(p,side);\
   while (state.KeepRunning()) { ker.benchmark(); }\
   state.counters["MDof/s"] = bm::Counter(ker.SumMdofs(), bm::Counter::kIsRate);\
   state.counters["Dofs"] = bm::Counter(ker.dofs);\
   state.counters["p"] = bm::Counter(p);\
 }\
 BENCHMARK(BK##i)\
        -> Apply(OrderSideArgs)\
        -> Unit(bm::kMillisecond);

/// BK1: scalar E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(1,Mass,1,false)

/// BK2: vector E-vector-to-E-vector evaluation of mass matrix, q=p+2
BakeOff_Kernel(2,VectorMass,3,false)

/// BK3: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(3,Diffusion,1,false)

/// BK4: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+2
BakeOff_Kernel(4,VectorDiffusion,3,false)

/// BK5: scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(5,Diffusion,1,true)

/// BK6: vector E-vector-to-E-vector evaluation of stiffness matrix, q=p+1
BakeOff_Kernel(6,VectorDiffusion,3,true)

/**
 * @brief main entry point
 * --benchmark_filter=BP1/6
 * --benchmark_context=device=cpu
 * --benchmark_out=bp1_cpu_fast.org
 * --benchmark_out_format=csv
 * --benchmark_min_time=2
 */
int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   mfem::MPI_Session main_mpi(argc, argv);
   mpi = &main_mpi;
#endif

   bm::ConsoleReporter CR;
   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string config_device = "cpu";

   if (bmi::global_context != nullptr)
   {
      bmi::FindInContext("device", config_device); // device=cuda/hip
      bmi::FindInContext("ndev", config_ndev); // ndev=1 when biding is used
   }

   const int mpi_rank = mpiWorldRank;
   const int device_id = mpi_rank % config_ndev;
   Device device(config_device.c_str(), device_id);
   if (mpiRoot) { device.Print(); }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

#ifndef MFEM_USE_MPI
   bm::RunSpecifiedBenchmarks(&CR);
#else
   if (mpi->Root()) { bm::RunSpecifiedBenchmarks(&CR); }
   else
   {
      // No default display_reporter and file_reporter
      bm::BenchmarkReporter *file_reporter = new NoReporter();
      bm::BenchmarkReporter *display_reporter = new NoReporter();
      bm::RunSpecifiedBenchmarks(display_reporter, file_reporter);
   }
#endif

   return 0;
}

#endif // MFEM_USE_BENCHMARK
