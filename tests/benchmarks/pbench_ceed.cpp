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

// Compile with: make pbench_ceed
//
// Sample runs:
//    mpirun -np 4 pbench_ceed
//    mpirun -np 4 pbench_ceed --benchmark_filter=BP3
//    mpirun -np 4 pbench_ceed --benchmark_filter=BP3 --benchmark_context=local_size=5e4
//    mpirun -np 6 pbench_ceed --benchmark_filter=BP3 --benchmark_context=proc_grid=3x2x1,local_size=5e4
//
// Device sample runs:
//    mpirun -np 4 pbench_ceed --benchmark_context=device=cuda,local_size=1e6
//    mpirun -np 4 pbench_ceed --benchmark_filter=BP3 --benchmark_context=device=cuda,local_size=1e7
//
// Description:
//    This benchmark contains the implementation of the CEED's bake-off
//    problems, BP1-BP6, and bake-off kernels, BK1-BK6: high-order benchmarks
//    designed to test and compare the performance of high-order codes.
//
//    See: ceed.exascaleproject.org/bps and github.com/CEED/benchmarks


#include "bench.hpp"

#ifdef MFEM_USE_BENCHMARK

// for instantiating more kernels:
#include <fem/integ/bilininteg_mass_kernels.hpp>
#include <fem/integ/bilininteg_diffusion_kernels.hpp>

// Global parameters

// local_size: desired approximate MPI-local problem size; this local size and
// the polynomial order determine the local mesh size so that the resulting
// problem size is (approximately) equal to local_size for all polynomial
// orders, see MakeParMesh().
double local_size = 61*61*61; // exact size match for all p=1,...,6

// proc_grid: use processor grid given by proc_grid[0,1,2].
int proc_grid[3] = {0,0,0};

// q_gl_inc: increment for the number of GL points: q = p + 1 + q_gl_inc
int q_gl_inc = 0;

// q_gll_inc: increment for the number of GLL points: q = p + 1 + q_gll_inc
int q_gll_inc = 0;

// verbose: verbosity level: 0, 1, 2
int verbose = 0;


// If running on GPU, wait for GPU tasks to finish:
inline void DeviceSync()
{
   if (Device::Allows(Backend::DEVICE_MASK & ~Backend::DEBUG_DEVICE))
   {
      MFEM_STREAM_SYNC;
      // MFEM_DEVICE_SYNC;
   }
}

void MakeExp2ProcGrid(int np)
{
   proc_grid[0] = proc_grid[1] = proc_grid[2] = 1;
   for ( ; np >= 8; np /= 8)
   {
      proc_grid[0] *= 2; proc_grid[1] *= 2; proc_grid[2] *= 2;
   }
   if (np == 4) { proc_grid[0] *= 2; proc_grid[1] *= 2; }
   else if (np == 2) { proc_grid[0] *= 2; }
}

// Construct the parallel mesh based on the polynomial order, p, and the
// local_size:
ParMesh MakeParMesh(int p)
{
   int nx = 0, ny = 0, nz = 0;
   int par_ref = 0;
   if (verbose && Mpi::Root()) { std::cout << _MFEM_FUNC_NAME << std::endl; }

   const double s = local_size;

   int m = floor((pow(s, 1./3)-1)/p);
   double s_l, s_u, s_c;
   while ((s_l=(  m  *p+1)*(  m  *p+1)*(  m  *p+1), s_l > s))  { m--; }
   m = std::max(m, 1);
   while ((s_u=((m+1)*p+1)*((m+1)*p+1)*((m+1)*p+1), s_u <= s)) { m++; }
   s_l = (m*p+1)*(m*p+1)*(m*p+1);
   if ((s_c=((m+1)*p+1)*(m*p+1)*(m*p+1), s_c > s))
   {
      if (s/s_l <= s_c/s) { nx = m; ny = m; nz = m; }
      else                { nx = m; ny = m; nz = m + 1; }
   }
   else if ((s_l=s_c, s_c=((m+1)*p+1)*((m+1)*p+1)*(m*p+1), s_c > s))
   {
      if (s/s_l <= s_c/s) { nx = m; ny = m;     nz = m + 1; }
      else                { nx = m; ny = m + 1; nz = m + 1; }
   }
   else
   {
      s_l=s_c, s_c=s_u;
      if (s/s_l <= s_c/s) { nx = m;     ny = m + 1; nz = m + 1; }
      else                { nx = m + 1; ny = m + 1; nz = m + 1; }
   }
   while (nx%2 == 0 && ny%2 == 0 && nz%2 == 0)
   {
      par_ref++;
      nx /= 2; ny /= 2; nz /= 2;
   }

   nx *= proc_grid[0];
   ny *= proc_grid[1];
   nz *= proc_grid[2];

   if (verbose && Mpi::Root())
   {
      std::cout
            << '\n'
            << "   order: " << p << '\n'
            << "   nx: " << nx << ", ny: " << ny << ", nz: " << nz << '\n'
            << "   px: " << proc_grid[0] << ", py: " << proc_grid[1]
            << ", pz: " << proc_grid[2] << '\n'
            << "   par_ref: " << par_ref << '\n'
            << std::endl;
   }

   StopWatch timer;
   timer.Start();
   double t_start = timer.RealTime();

   Mesh smesh(Mesh::MakeCartesian3D(nx, ny, nz, Element::HEXAHEDRON));
   if (verbose && Mpi::Root())
   {
      double t_elapsed = timer.RealTime() - t_start;
      std::cout << "   Mesh: " << 1e3*t_elapsed << " ms" << std::endl;
   }
   t_start = timer.RealTime();
   Array<int> partitioning;
   partitioning.MakeRef(smesh.CartesianPartitioning(proc_grid), smesh.GetNE(),
                        MemoryType::HOST, true);
   ParMesh pmesh(MPI_COMM_WORLD, smesh, partitioning.HostRead());
   smesh.Clear();
   for (int i = 0; i < par_ref; i++)
   {
      pmesh.UniformRefinement();
   }
   if (verbose && Mpi::Root())
   {
      double t_elapsed = timer.RealTime() - t_start;
      std::cout << "   ParMesh: " << 1e3*t_elapsed << " ms" << std::endl;
   }
   return pmesh;
}

template <int VDIM, bool GLL>
struct BakeOff
{
   static constexpr int DIM = 3;
   const int p, q, q_order;
   ParMesh mesh;
   H1_FECollection fec;
   ParFiniteElementSpace fes;
   const Geometry::Type geom_type;
   IntegrationRules irs;
   const IntegrationRule *ir;
   ConstantCoefficient one;
   Vector uvec;
   VectorConstantCoefficient unit_vec;
   const HYPRE_BigInt dofs;
   ParGridFunction x, y;
   ParBilinearForm a;
   double mdofs{};
   StopWatch timer;

   BakeOff(int p):
      p(p),
      q(GLL ? p + 1 + q_gll_inc : p + 1 + q_gl_inc),
      q_order(2 * q + (GLL ? -3 : -1)),
      mesh(MakeParMesh(p)),
      fec(p, DIM, BasisType::GaussLobatto),
      fes(&mesh, &fec, VDIM, VDIM == 3 ? Ordering::byVDIM : Ordering::byNODES),
      geom_type(mesh.GetTypicalElementGeometry()),
      irs(0, GLL ? Quadrature1D::GaussLobatto : Quadrature1D::GaussLegendre),
      ir(&irs.Get(geom_type, q_order)),
      one(1.0),
      uvec(DIM),
      unit_vec((uvec = 1.0, uvec /= uvec.Norml2(), uvec)),
      dofs(fes.GlobalTrueVSize()),
      x(&fes),
      y(&fes),
      a(&fes)
   {
      x = 0.0;
      if (verbose && Mpi::Root())
      {
         std::cout << "q: " << q << ", dofs: " << dofs << std::endl;
         // std::cout << _MFEM_FUNC_NAME << std::endl;
      }
      timer.Start();
   }

   virtual void benchmark(benchmark::State &state) = 0;

   double SumMdofs() const { return mdofs; }

   double MDofs() const { return 1e-6 * dofs; }
};

/// Bake-off Problems (BPs)
template <typename BFI, int VDIM, bool GLL>
struct Problem : public BakeOff<VDIM, GLL>
{
   const double rtol = 1e-16;
   const int max_it = 20;
   const int print_lvl = -1;

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   LinearForm b;
   OperatorPtr A;
   Vector B, X;
   CGSolver cg;
   int bench_call_counter = 0;

   using base = BakeOff<VDIM, GLL>;
   using base::a;
   using base::ir;
   using base::one;
   using base::mesh;
   using base::fes;
   using base::x;
   using base::y;
   using base::mdofs;
   using base::timer;

   Problem(int order):
      BakeOff<VDIM, GLL>(order),
      ess_bdr(mesh.bdr_attributes.Max()),
      b(&fes),
      cg(fes.GetComm())
   {
      if (verbose && Mpi::Root()) { std::cout << _MFEM_FUNC_NAME << std::endl; }
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      LinearFormIntegrator *integ;
      if (VDIM == 1)
      {
         integ = new DomainLFIntegrator(this->one);
      }
      else
      {
         integ = new VectorDomainLFIntegrator(this->unit_vec);
      }
      integ->SetIntRule(ir);
      b.AddDomainIntegrator(integ); // b takes ownership of integ
      b.UseFastAssembly(true);
      b.Assemble();

      double t_start = timer.RealTime();
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, ir));
      a.Assemble();
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      if (verbose && Mpi::Root())
      {
         double t_elapsed = timer.RealTime() - t_start;
         std::cout << "   assemble a: " << 1e3*t_elapsed << " ms" << std::endl;
      }

      cg.SetRelTol(rtol);
      cg.SetOperator(*A);
      cg.SetMaxIter(max_it);
      cg.SetPrintLevel(print_lvl);
      cg.iterative_mode = false;
      // warmup cg
      cg.SetMaxIter(2);
      cg.Mult(B, X);
      cg.SetMaxIter(max_it);
   }

   virtual ~Problem()
   {
      if (verbose && Mpi::Root())
      {
         std::cout << _MFEM_FUNC_NAME << '\n'
                   << "   call counter: " << bench_call_counter << '\n'
                   << "   MDofs: " << mdofs << std::endl;
      }
   }

   void benchmark(benchmark::State &state) override
   {
      if (verbose > 1 && Mpi::Root())
      {
         std::cout << _MFEM_FUNC_NAME << std::endl;
      }

      DeviceSync();
      MPI_Barrier(cg.GetComm());
      double t_start = timer.RealTime();

      cg.Mult(B, X);

      DeviceSync();
      MPI_Barrier(cg.GetComm());
      double t_elapsed = timer.RealTime() - t_start;
      // Ensure every ranks gets the same time, otherwise google-benchmark may
      // behave differently on different ranks.
      MPI_Bcast(&t_elapsed, 1, MPI_DOUBLE, 0, cg.GetComm());

      state.SetIterationTime(t_elapsed);
      if (verbose > 1 && Mpi::Root())
      {
         std::cout << "   bench time: " << 1e3*t_elapsed << " ms" << std::endl;
      }

      mdofs += this->MDofs() * cg.GetNumIterations();
      bench_call_counter++;
   }
};

/// Bake-off Problems (BPs)
#define BakeOff_Problem(i, Kernel, VDIM, GLL)                        \
   static void BP##i(bm::State &state)                               \
   {                                                                 \
      Problem<Kernel##Integrator, VDIM, GLL> ker(state.range(0));    \
      for (auto z : state) { ker.benchmark(state); }                 \
      state.counters["Num Dofs"] = ker.dofs;                         \
      state.counters["|      Dof/s"] =                               \
         bm::Counter(1e6*ker.SumMdofs(), bm::Counter::kIsRate);      \
      state.counters["|   Dof/s/NP"] =                               \
         bm::Counter(1e6*ker.SumMdofs()/ker.fes.GetNRanks(),         \
                     bm::Counter::kIsRate);                          \
   }                                                                 \
   BENCHMARK(BP##i)->DenseRange(1, 6)->Unit(bm::kMillisecond)->UseManualTime();

// state.counters[" Q1D"] = ker.q;


/// BP1: scalar PCG with mass matrix, GL
BakeOff_Problem(1, Mass, 1, false)

/// BP2: vector PCG with mass matrix, GL
BakeOff_Problem(2, VectorMass, 3, false)

/// BP3: scalar PCG with stiffness matrix, GL
BakeOff_Problem(3, Diffusion, 1, false)

/// BP4: vector PCG with stiffness matrix, GL
BakeOff_Problem(4, VectorDiffusion, 3, false)

/// BP5: scalar PCG with stiffness matrix, GLL
BakeOff_Problem(5, Diffusion, 1, true)

/// BP6: vector PCG with stiffness matrix, GLL
BakeOff_Problem(6, VectorDiffusion, 3, true)

/// Bake-off Kernels (BKs)
template <typename BFI, int VDIM, bool GLL>
struct Kernel : public BakeOff<VDIM, GLL>
{
   using base = BakeOff<VDIM, GLL>;
   using base::a;
   using base::ir;
   using base::one;
   using base::fes;
   using base::x;
   using base::y;
   using base::mdofs;
   using base::timer;

   Kernel(int order) : base(order)
   {
      x.Randomize(1);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new BFI(one, ir));
      a.Assemble();
      // warmup
      for (int i = 0; i < 2; i++) { a.Mult(x, y); }
   }

   void benchmark(benchmark::State &state) override
   {
      DeviceSync();
      MPI_Barrier(fes.GetComm());
      double t_start = timer.RealTime();

      a.Mult(x, y);

      DeviceSync();
      MPI_Barrier(fes.GetComm());
      double t_elapsed = timer.RealTime() - t_start;
      // Ensure every ranks gets the same time, otherwise google-benchmark may
      // behave differently on different ranks.
      MPI_Bcast(&t_elapsed, 1, MPI_DOUBLE, 0, fes.GetComm());

      state.SetIterationTime(t_elapsed);
      mdofs += this->MDofs();
   }
};

/// Generic CEED BKi
#define BakeOff_Kernel(i, KER, VDIM, GLL)                       \
   static void BK##i(bm::State &state)                          \
   {                                                            \
      Kernel<KER##Integrator, VDIM, GLL> ker(state.range(0));   \
      for (auto z : state) { ker.benchmark(state); }            \
      state.counters["Num Dofs"] = ker.dofs;                    \
      state.counters["|      Dof/s"] =                          \
         bm::Counter(1e6*ker.SumMdofs(), bm::Counter::kIsRate); \
      state.counters["|   Dof/s/NP"] =                          \
         bm::Counter(1e6*ker.SumMdofs()/ker.fes.GetNRanks(),    \
                     bm::Counter::kIsRate);                     \
   }                                                            \
   BENCHMARK(BK##i)->DenseRange(1, 6)->Unit(bm::kMillisecond)->UseManualTime();

// state.counters[" Q1D"] = ker.q;


/// BK1: scalar E-vector-to-E-vector evaluation of mass matrix, GL
BakeOff_Kernel(1, Mass, 1, false)

/// BK2: vector E-vector-to-E-vector evaluation of mass matrix, GL
BakeOff_Kernel(2, VectorMass, 3, false)

/// BK3: scalar E-vector-to-E-vector evaluation of stiffness matrix, GL
BakeOff_Kernel(3, Diffusion, 1, false)

/// BK4: vector E-vector-to-E-vector evaluation of stiffness matrix, GL
BakeOff_Kernel(4, VectorDiffusion, 3, false)

/// BK5: scalar E-vector-to-E-vector evaluation of stiffness matrix, GLL
BakeOff_Kernel(5, Diffusion, 1, true)

/// BK6: vector E-vector-to-E-vector evaluation of stiffness matrix, GLL
BakeOff_Kernel(6, VectorDiffusion, 3, true)


int main(int argc, char *argv[])
{
   // MassIntegrator specializations by <DIM, D1D, Q1D>
   MassIntegrator::AddSpecialization<3, 3, 3>();
   MassIntegrator::AddSpecialization<3, 4, 4>();
   MassIntegrator::AddSpecialization<3, 5, 5>();
   MassIntegrator::AddSpecialization<3, 6, 6>();
   MassIntegrator::AddSpecialization<3, 7, 7>();

   // DiffusionIntegrator specializations by <DIM, D1D, Q1D>
   DiffusionIntegrator::AddSpecialization<3, 3, 3>();
   DiffusionIntegrator::AddSpecialization<3, 4, 4>();
   DiffusionIntegrator::AddSpecialization<3, 5, 5>();
   DiffusionIntegrator::AddSpecialization<3, 6, 6>();
   DiffusionIntegrator::AddSpecialization<3, 7, 7>();

   Mpi::Init();
   Hypre::Init();

   bm::Initialize(&argc, argv);

   // Device setup, cpu by default
   std::string device_config = "cpu";
   bool gpu_aware_mpi = false;
   std::string proc_grid_str = "";

   auto global_context = bmi::GetGlobalContext();
   if (global_context != nullptr)
   {
      const auto device = global_context->find("device");
      if (device != global_context->end())
      {
         device_config = device->second;
      }
      const auto ctx_gpu_aware_mpi = global_context->find("gpu_aware_mpi");
      if (ctx_gpu_aware_mpi != global_context->end())
      {
         gpu_aware_mpi = std::atoi(ctx_gpu_aware_mpi->second.c_str());
      }
      const auto ctx_local_size = global_context->find("local_size");
      if (ctx_local_size != global_context->end())
      {
         std::size_t pos;
         local_size = std::stof(ctx_local_size->second, &pos);
         if (ctx_local_size->second.size() != pos)
         {
            if (Mpi::Root())
            {
               std::cout << "\nerror reading local_size: "
                         << ctx_local_size->second << '\n' << std::endl;
            }
            return 1;
         }
         if (local_size < 64.0 || local_size > std::exp2(30.0))
         {
            if (Mpi::Root())
            {
               std::cout << "\nlocal_size must be in [2^6,2^30]! local_size: "
                         << local_size << '\n' << std::endl;
            }
            return 1;
         }
      }
      const auto ctx_proc_grid = global_context->find("proc_grid");
      if (ctx_proc_grid != global_context->end())
      {
         proc_grid_str = ctx_proc_grid->second;
      }
      const auto ctx_verbose = global_context->find("verbose");
      if (ctx_verbose != global_context->end())
      {
         verbose = std::atoi(ctx_verbose->second.c_str());
      }
   }
   const int num_procs = Mpi::WorldSize();
   if (proc_grid_str == "" || proc_grid_str == "2^n")
   {
      if (((num_procs-1)&num_procs) != 0)
      {
         if (Mpi::Root())
         {
            std::cout << "\nthe number of processors is not a power of 2!"
                      << " num_procs: " << num_procs
                      << "\nuse a processor grid, e.g. "
                      << "--benchmark_context=proc_grid=3x5x7\n"
                      << std::endl;
         }
         return 1;
      }
      MakeExp2ProcGrid(num_procs);
   }
   else
   {
      int n = std::sscanf(proc_grid_str.c_str(), "%d x %d x %d",
                          &proc_grid[0], &proc_grid[1], &proc_grid[2]);
      if (n != 3)
      {
         if (Mpi::Root())
         {
            std::cout << "\ninvalid processor grid input: "
                      << proc_grid_str << "\n" << std::endl;
         }
         return 1;
      }
      if (proc_grid[0]*proc_grid[1]*proc_grid[2] != num_procs ||
          proc_grid[0] < 1 || proc_grid[1] < 1 || proc_grid[2] < 1)
      {
         if (Mpi::Root())
         {
            std::cout << "\ninvalid processor grid: " << proc_grid[0] << " x "
                      << proc_grid[1] << " x " << proc_grid[2] << " != "
                      << num_procs << '\n' << std::endl;
         }
         return 1;
      }
   }

   if (bm::ReportUnrecognizedArguments(argc, argv)) { return 1; }

   Device device(device_config.c_str());
   device.SetGPUAwareMPI(gpu_aware_mpi);
   if (Mpi::Root())
   {
      device.Print();
      std::cout << "    num procs: " << num_procs << std::endl;
      std::cout << "gpu aware mpi: " << gpu_aware_mpi << std::endl;
      std::cout << "   local size: " << local_size << std::endl;
      std::cout << "    proc grid: " << proc_grid[0] << 'x'
                /**/                 << proc_grid[1] << 'x'
                /**/                 << proc_grid[2] << std::endl;
      std::cout << "     GL q_inc: " << q_gl_inc << std::endl;
      std::cout << "    GLL q_inc: " << q_gll_inc << std::endl;
   }

   DeviceSync();
   MPI_Barrier(MPI_COMM_WORLD);

   if (Mpi::Root())
   {
      bm::ConsoleReporter CR;
      bm::RunSpecifiedBenchmarks(&CR);
   }
   else
   {
      NoReporter NR;
      bm::RunSpecifiedBenchmarks(&NR);
   }

   return 0;
}

#endif // MFEM_USE_BENCHMARK
