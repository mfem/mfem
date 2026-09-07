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

#define CATCH_CONFIG_RUNNER
#include "mfem.hpp"
using namespace mfem;

#define CATCH_CONFIG_RUNNER
#include "run_unit_tests.hpp"

using namespace mfem;

#ifndef _WIN32 // Debug device specific tests, not supported on Windows
#include <unistd.h>

struct NullBuf: public std::streambuf { int overflow(int c) override { return c; }};

#include <iosfwd>
#include <csetjmp>
#include <csignal>

static void TestMemoryTypes(MemoryType mt, bool use_dev, int N = 1024)
{
   Memory<real_t> mem(N, mt);
   REQUIRE(mem.Capacity() == N);
   Vector y;
   y.NewMemoryAndSize(mem, N, true);
   y.UseDevice(use_dev);
   y = 0.0;
   y.HostWrite();
   y[0] = -1.0;
   y.Write();
   y = 1.0;
   y.HostReadWrite();
   y[0] = 0.0;
   REQUIRE(y*y == MFEM_Approx(N-1));
   y.Destroy();
}

static void ScanMemoryTypes()
{
   const auto h_mt = mm.GetHostMemoryType(), d_mt = mm.GetDeviceMemoryType();
   TestMemoryTypes(h_mt, true), TestMemoryTypes(d_mt, true);
   TestMemoryTypes(h_mt, false), TestMemoryTypes(d_mt, false);
}

static void MmuCatch(const int N = 1024)
{
   Vector Y(N);
   Y.UseDevice(true);
   static real_t *h_Y = Y.GetData(); // store host address
   Y = 0.0; // use Y on the device
   // using h_Y raises an MFEM abort that needs to be caught with a new handler
   static jmp_buf env;
   struct sigaction sa;
   sa.sa_flags = SA_SIGINFO;
   sigemptyset(&sa.sa_mask);
   static volatile bool caught_illegal_memory_access = false;
   sa.sa_sigaction = [](int, siginfo_t *si, void*)
   {
      REQUIRE(si->si_addr == h_Y);
      caught_illegal_memory_access = true;
      mfem::out << "Illegal memory access caught at " << si->si_addr << std::endl;
      std::longjmp(env, EXIT_FAILURE); // noreturn, setjmp returns EXIT_FAILURE
   };
   // set the new handlers
   REQUIRE(sigaction(SIGBUS, &sa, nullptr) != -1); // macOS
   REQUIRE(sigaction(SIGSEGV, &sa, nullptr) != -1); // Linux

   if (setjmp(env) == EXIT_SUCCESS) // save the execution context to env
   {
      h_Y[0] = 0.0; // raises a SIGBUS, handler, longjmp
      REQUIRE(false); // should not be here
   }
   REQUIRE(caught_illegal_memory_access); // rewinding to env through setjmp
}

static void Aliases(const int N = 0x1234)
{
   Vector S(2*3*N + N);
   S.UseDevice(true);
   S = -1.0;
   GridFunction X,V,E;
   const int Xsz = 3*N, Vsz = 3*N, Esz = N;
   X.NewMemoryAndSize(Memory<real_t>(S.GetMemory(), 0, Xsz), Xsz, true);
   V.NewMemoryAndSize(Memory<real_t>(S.GetMemory(), Xsz, Vsz), Vsz, true);
   E.NewMemoryAndSize(Memory<real_t>(S.GetMemory(), Xsz + Vsz, Esz), Esz, true);
   X = 1.0;
   X.SyncAliasMemory(S);
   S.HostWrite();
   S = -1.0;
   X.Write();
   X = 1.0;
   S.HostRead();
   REQUIRE(S*S == MFEM_Approx(7.0*N));
   V = 2.0;
   V.SyncAliasMemory(S);
   REQUIRE(S*S == MFEM_Approx(16.0*N));
   E = 3.0;
   E.SyncAliasMemory(S);
   REQUIRE(S*S == MFEM_Approx(24.0*N));
}

TEST_CASE("Array::MakeRef", "[DebugDevice]")
{
   Array<int> x(1), y;
   y.MakeRef(x);
   x.Read();
   REQUIRE_NOTHROW(y.Read());
}

TEST_CASE("MemoryManager/DebugDevice", "[DebugDevice]")
{
   // If MFEM_MEMORY is set, we can start with some non-empty maps,
   // we need to use the number of pointers and aliases there already are
   // present in the maps
   struct NullBuffer: public std::streambuf
   {
      int overflow(int c) override { return c; }
   } null_buffer;
   std::ostream dev_null(&null_buffer);
   const auto n_ptr = mm.PrintPtrs(dev_null);
   const auto n_alias = mm.PrintAliases(dev_null);
   const auto pagesize = sysconf(_SC_PAGE_SIZE);
   REQUIRE(pagesize > 0);

   for (int n = 1; n < 2*pagesize; n+=7)
   {
      Aliases(n);
      REQUIRE(mm.PrintPtrs(dev_null) == n_ptr);
      REQUIRE(mm.PrintAliases(dev_null) == n_alias);
   }
   MmuCatch();
   ScanMemoryTypes();

   REQUIRE(mm.PrintPtrs(dev_null) == n_ptr);
   REQUIRE(mm.PrintAliases(dev_null) == n_alias);
}

namespace darcy_alias
{

class FixedTau : public HDGStabilization
{
public:
   explicit FixedTau(real_t t) : tau(t) { }
   bool IsConstant() const override { return true; }
   real_t Eval(real_t, real_t, real_t, real_t,
               ElementTransformation &) const override { return tau; }
private:
   real_t tau;
};

/// A linear hybridized Darcy solve. @a sync selects how the caller gets the
/// potential load into its own BlockVector: through the block with a
/// SyncAliasMemory afterwards, or host-explicitly. The two must agree.
void Solve(bool sync, Vector &trace, Vector &pot)
{
   const int n = 4, order = 1, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim, BasisType::GaussLobatto);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   ConstantCoefficient one(1.0), src(1.0);
   FixedTau tau(1.0);
   DarcyForm darcy(&Vh, &Wh);
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));

   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   BilinearForm *M_p = darcy.GetPotentialMassForm();
   auto *fi = new HDGDiffusionIntegrator(one, 1.0);
   auto *fb = new HDGDiffusionIntegrator(one, 1.0);
   fi->SetStabilization(tau);
   fb->SetStabilization(tau);
   M_p->AddInteriorFaceIntegrator(fi);
   M_p->AddBdrFaceIntegrator(fb, all);

   MixedBilinearForm *B = darcy.GetFluxDivForm();
   B->AddDomainIntegrator(new VectorDivergenceIntegrator());
   B->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   B->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-2.0)), all);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   darcy.GetHybridization()->SetEssentialBC(all);
   darcy.Assemble();

   Array<int> offs(4);
   offs[0] = 0;
   offs[1] = Vh.GetVSize();
   offs[2] = Wh.GetVSize();
   offs[3] = Mh.GetVSize();
   offs.PartialSum();
   BlockVector sol(offs), rhs(offs);
   sol = 0.0;
   rhs = 0.0;
   darcy.GetPotentialRHS()->Assemble();

   if (sync)
   {
      // The documented contract: `+=` through the block is a DEVICE operation
      // on an alias, so its result has to be propagated to the parent.
      rhs.GetBlock(1) += *darcy.GetPotentialRHS();
      rhs.GetBlock(1).SyncAliasMemory(rhs);
   }
   else
   {
      Vector &rb = rhs.GetBlock(1);
      const Vector &lb = *darcy.GetPotentialRHS();
      real_t *h = rb.HostReadWrite();
      const real_t *l = lb.HostRead();
      for (int i = 0; i < rb.Size(); i++) { h[i] += l[i]; }
      rb.SyncAliasMemory(rhs);
   }

   Vector X, RHS;
   X.MakeRef(sol, offs[2], Mh.GetVSize());
   RHS.MakeRef(rhs, offs[2], Mh.GetVSize());
   BlockVector dsol(sol, darcy.GetOffsets()), drhs(rhs, darcy.GetOffsets());
   OperatorPtr R;
   darcy.FormLinearSystem(ess_flux, dsol, drhs, R, X, RHS, true);

   SparseMatrix *H = dynamic_cast<SparseMatrix *>(R.Ptr());
   REQUIRE(H != nullptr);
   RHS.HostReadWrite();
   X.HostReadWrite();
   UMFPackSolver lin(*H);
   lin.Mult(RHS, X);
   BlockVector csol(sol, darcy.GetOffsets());
   darcy.RecoverFEMSolution(X, csol);

   trace.SetSize(X.Size());
   trace = X;
   pot.SetSize(csol.GetBlock(1).Size());
   pot = csol.GetBlock(1);
   trace.HostRead();
   pot.HostRead();
}

} // namespace darcy_alias

/**
 * @brief A caller accumulating into a block of its own BlockVector owes
 * SyncAliasMemory, and under a device it is not optional.
 *
 * `rhs.GetBlock(1) += *darcy.GetPotentialRHS()` is a DEVICE operation on an
 * ALIAS of rhs. It leaves the result in that alias's device buffer, and the
 * second view the caller builds to hand FormLinearSystem() gets a FRESH alias
 * marked host-valid whatever the real state -- so DarcyHybridization's host
 * loops read stale zeros. Measured before the contract was documented: the
 * reduced trace right-hand side came back EXACTLY zero, the trace solve
 * returned zero, the recovered fields were quietly wrong, and nothing errored
 * anywhere. The validity flags say hostvalid=1 devvalid=0 on the very block
 * whose host buffer is zeros, so no HostRead() and no guard can catch it --
 * which is why this is pinned by a test rather than by an assertion in the
 * library.
 *
 * DarcyForm::Assemble() already does exactly this for its own b_u and b_p;
 * the contract is on GetPotentialRHS().
 *
 * This case lives here, and not in the globbed set, because it needs a Device
 * and the Device here is the binary's: constructing one inside a TEST_CASE
 * linked into unit_tests would run mm.Destroy() at the end of the case and
 * leave the remaining cases against a destroyed MemoryManager, and would trip
 * "the mfem::Device is already configured!" in punit_tests.
 */
TEST_CASE("DarcyForm/BlockVector alias sync", "[DebugDevice]")
{
   using namespace darcy_alias;

   Vector tr_sync, pot_sync, tr_host, pot_host;
   Solve(true, tr_sync, pot_sync);
   Solve(false, tr_host, pot_host);

   // There is something to get wrong: without the sync the trace came back
   // identically zero.
   REQUIRE(tr_sync.Norml2() > 1e-6);
   REQUIRE(pot_sync.Norml2() > 1e-6);

   REQUIRE(tr_sync.Size() == tr_host.Size());
   Vector d(tr_sync);
   d -= tr_host;
   REQUIRE(d.Norml2() <= 1e-12 * tr_host.Norml2());

   Vector dp(pot_sync);
   dp -= pot_host;
   REQUIRE(dp.Norml2() <= 1e-12 * pot_host.Norml2());
}

#endif // _WIN32

int main(int argc, char *argv[])
{
   Device device("debug");
   return RunCatchSession(argc, argv, {"[DebugDevice]"});
}
