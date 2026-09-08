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
#include <memory>

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
///
/// @a mode selects how the element-local blocks are factored and solved.
/// LocalFactorMode::Batched is the device-shaped route -- the gather, the
/// factorisation, the Schur complement, the local solves and the scatter are
/// all BatchedLinAlg calls or mfem::forall kernels -- and with a Device
/// configured it leaves the recovered fields DEVICE-valid, which is the
/// second thing this file is for.
void Solve(bool sync, Vector &trace, Vector &pot,
           DarcyHybridization::LocalFactorMode mode =
              DarcyHybridization::LocalFactorMode::Serial,
           Vector *flux = NULL)
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
   darcy.GetHybridization()->SetLocalFactorMode(mode);
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

   // Read the recovered fields back through a SECOND view over the range
   // RecoverFEMSolution wrote, not through the block objects it wrote
   // THROUGH. That is this file's other case seen from the far side, and it
   // is the only way the library's own alias discipline is observable: a
   // fresh alias comes back marked host-valid whatever the underlying state,
   // so if ComputeSolution() leaves its answer in a block's device buffer
   // without propagating it, this reads stale host memory and says nothing.
   Vector qv, pv;
   qv.MakeRef(csol, 0, csol.GetBlock(0).Size());
   pv.MakeRef(csol, csol.GetBlock(0).Size(), csol.GetBlock(1).Size());
   pot.SetSize(pv.Size());
   pot = pv;
   if (flux)
   {
      flux->SetSize(qv.Size());
      *flux = qv;
      flux->HostRead();
   }
   trace.HostRead();
   pot.HostRead();
}

/// (c p^2, w) on the potential mass form, which is what puts
/// DarcyHybridization into a nonlinear local operator and so onto the NPC
/// route -- NPCReduce() and NPCRecover(), which a linear problem never
/// reaches.
class SquareSource : public NonlinearFormIntegrator
{
public:
   explicit SquareSource(real_t c_) : c(c_) { }

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elvect.SetSize(dof);
      elvect = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         elvect.Add(ip.weight * Tr.Weight() * c * u * u, shape);
      }
   }

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   {
      const int dof = el.GetDof();
      shape.SetSize(dof);
      elmat.SetSize(dof);
      elmat = 0.0;
      const IntegrationRule &ir = IntRules.Get(el.GetGeomType(),
                                               2*el.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         el.CalcPhysShape(Tr, shape);
         const real_t u = shape * elfun;
         AddMult_a_VVt(ip.weight * Tr.Weight() * 2.0 * c * u, shape, elmat);
      }
   }

private:
   real_t c;
   Vector shape;
};

/// One NPC Newton step on the semilinear problem, in the given local factor
/// and gradient modes. Reaches NPCReduce()/NPCRecover() -- whose gather and
/// scatter are the kernels LocalFactorMode::Batched installs -- and, under
/// GradientMode::MatrixFree, ComputeHMode::GradientFactorOnly, whose whole
/// body is the batched factorisation.
void NPCStep(DarcyHybridization::LocalFactorMode mode,
             DarcyHybridization::GradientMode gmode,
             Vector &dq, Vector &dp, Vector &dtr_out)
{
   const int n = 4, order = 1, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   NonlinearForm *Mnl_p = darcy.GetPotentialMassNonlinearForm();
   Mnl_p->AddDomainIntegrator(new SquareSource(5.0));
   Mnl_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   Mnl_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetLocalFactorMode(mode);
   dh->SetGradientMode(gmode);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   x = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;

   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);

   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   {
      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
      std::unique_ptr<GSSmoother> prec;
      if (Sm) { prec.reset(new GSSmoother(*Sm)); }
      GMRESSolver gmres;
      gmres.SetOperator(S);
      if (prec) { gmres.SetPreconditioner(*prec); }
      gmres.SetKDim(200);
      gmres.SetMaxIter(2000);
      gmres.SetRelTol(1e-14);
      gmres.SetAbsTol(0.0);
      gmres.SetPrintLevel(-1);
      gmres.Mult(b_tr, dtr);
   }

   BlockVector dx(darcy.GetOffsets());
   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);

   // Through a SECOND view over the range NPCRecover wrote, for the reason
   // Solve() gives.
   Vector qv, pv;
   qv.MakeRef(dx, 0, dx.GetBlock(0).Size());
   pv.MakeRef(dx, dx.GetBlock(0).Size(), dx.GetBlock(1).Size());
   dq.SetSize(qv.Size());
   dq = qv;
   dp.SetSize(pv.Size());
   dp = pv;
   dtr_out.SetSize(dtr.Size());
   dtr_out = dtr;
   dq.HostRead();
   dp.HostRead();
   dtr_out.HostRead();
}

/// One NPC Newton step on a LINEAR problem whose potential-mass face
/// constraint is a SumIntegrator of a diffusion and an upwinded convection
/// term -- which is what AssemblyMode::Batched's face kernel covers, and what
/// the nonlinear NPCStep() above does not reach: a nonlinear potential mass
/// takes the constraint to c_nlfi_p, where there is no batched kernel at all.
void FaceKernelStep(DarcyHybridization::AssemblyMode am,
                    Vector &dq, Vector &dp, Vector &dtr_out, bool &taken)
{
   const int n = 4, order = 1, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   VectorFunctionCoefficient vel(dim, [](const Vector &X, Vector &v)
   {
      v(0) = 1.0 + 0.5 * std::sin(M_PI * X(1));
      v(1) = -0.7 + 0.3 * std::cos(M_PI * X(0));
   });

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxMassForm()->AddDomainIntegrator(new VectorMassIntegrator(one));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddInteriorFaceIntegrator(
      new HDGConvectionUpwindedIntegrator(vel, 1.0, 0.5));
   // A boundary term the kernel does NOT cover, so the host loop that reads
   // D immediately after it is in play -- which is the whole point here.
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();
   taken = dh->CanBatchPotFaceAssembly();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   x = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;
   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);

   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   {
      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
      REQUIRE(Sm != nullptr);
      b_tr.HostReadWrite();
      dtr.HostReadWrite();
      UMFPackSolver umf(*Sm);
      umf.Mult(b_tr, dtr);
   }

   BlockVector dx(darcy.GetOffsets());
   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);

   Vector qv, pv;
   qv.MakeRef(dx, 0, dx.GetBlock(0).Size());
   pv.MakeRef(dx, dx.GetBlock(0).Size(), dx.GetBlock(1).Size());
   dq.SetSize(qv.Size());
   dq = qv;
   dp.SetSize(pv.Size());
   dp = pv;
   dtr_out.SetSize(dtr.Size());
   dtr_out = dtr;
   dq.HostRead();
   dp.HostRead();
   dtr_out.HostRead();
}

/** @brief One NPC residual on the `-nld` shape, in the given assembly mode.

    The flux law is a MixedConductionNLFIntegrator over a LinearDiffusionFlux
    on the BlockNonlinearForm and the HDG stabilization is a plain
    HDGDiffusionIntegrator on the LINEAR potential mass form, which is what
    puts the law on m_nlfi and takes the constraint to c_bfi_p. That is the
    only shape DarcyHybridization::CanBatchLocalResidual() admits, and 15 of
    the 88 hybridized regression references have it.

    Under a device the residual kernel writes ru_batched device-side and the
    element loop that follows reads it on the HOST -- so a missing HostRead()
    faults here under `debug` and, per this file's face-kernel case, would
    silently return a stale buffer under CUDA. That is the whole reason this
    case exists rather than only the globbed one.

    Everything read back at the end goes through HostRead(), as the other
    fixtures here do. */
void LocalResidualStep(DarcyHybridization::AssemblyMode am, int order,
                       Vector &rq, Vector &rp, Vector &rtr, bool &taken)
{
   const int n = 4, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);
   FunctionCoefficient ikappa([](const Vector &X)
   {
      return 1.3 + 0.4 * std::sin(M_PI * X(0)) * X(1);
   });
   LinearDiffusionFlux law(dim, ikappa);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));
   darcy.GetBlockNonlinearForm()->AddDomainIntegrator(
      new MixedConductionNLFIntegrator(law));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();
   taken = dh->CanBatchLocalResidual();

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   // A state with structure: for a law linear in the flux, x = 0 makes the
   // flux row identically zero and the comparison vacuous.
   x.HostWrite();
   for (int i = 0; i < x.Size(); i++)
   {
      x(i) = std::sin(0.71 * i + 0.5) + 0.5 * std::cos(0.113 * i);
   }
   Vector x_tr(Mh.GetVSize());
   x_tr.HostWrite();
   for (int i = 0; i < x_tr.Size(); i++)
   {
      x_tr(i) = std::sin(0.71 * i + 2.1) + 0.5 * std::cos(0.113 * i);
   }

   BlockVector r(darcy.GetOffsets());
   Vector r_tr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);

   Vector qv, pv;
   qv.MakeRef(r, 0, r.GetBlock(0).Size());
   pv.MakeRef(r, r.GetBlock(0).Size(), r.GetBlock(1).Size());
   rq.SetSize(qv.Size());
   rq = qv;
   rp.SetSize(pv.Size());
   rp = pv;
   rtr.SetSize(r_tr.Size());
   rtr = r_tr;
   rq.HostRead();
   rp.HostRead();
   rtr.HostRead();
}

/** A flux-mass boundary face integrator: s <C q, v>_F with C a non-symmetric
    constant coupling of the vdim components. It has to be written here because
    the library has no BilinearFormIntegrator returning the one-sided element
    block of a vector flux space on a boundary face -- which is also why the
    batched boundary flux pass batches the SCATTER and not the quadrature. The
    fuller note is on tests/unit/fem/test_darcy_batched_bdrflux.cpp. */
class BdrFluxMass : public BilinearFormIntegrator
{
   const real_t s;
   const int vd;

public:
   BdrFluxMass(real_t s_, int vd_) : s(s_), vd(vd_) { }

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      const int dof = el1.GetDof();
      elmat.SetSize(dof * vd);
      elmat = 0.0;
      Vector shape(dof);
      const IntegrationRule &ir =
         IntRules.Get(Trans.GetGeometryType(), 2 * el1.GetOrder() + 2);
      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Trans.SetAllIntPoints(&ip);
         el1.CalcShape(Trans.GetElement1IntPoint(), shape);
         const real_t w = s * ip.weight * Trans.Weight();
         for (int d = 0; d < vd; d++)
            for (int e = 0; e < vd; e++)
            {
               const real_t wc = w * (1.0 + 0.5*d - 0.25*e + ((d > e) ? 0.75 : 0.));
               for (int i = 0; i < dof; i++)
                  for (int j = 0; j < dof; j++)
                  {
                     elmat(d*dof + i, e*dof + j) += wc * shape(i) * shape(j);
                  }
            }
      }
   }
};

/// One NPC Newton step on a LINEAR problem whose FLUX mass carries two
/// BOUNDARY face integrators -- which is what
/// AssembleFluxMassBdrMatricesBatched() covers, and what no miniapp and no
/// regression reference in the tree installs.
void BdrFluxStep(DarcyHybridization::AssemblyMode am,
                 Vector &dq, Vector &dp, Vector &dtr_out, bool &taken)
{
   const int n = 4, order = 1, dim = 2;
   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL, false,
                                     0.8, 1.2);
   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);
   FiniteElementSpace Vh(&mesh, &u_coll, dim), Wh(&mesh, &p_coll),
                      Mh(&mesh, &t_coll);

   DarcyForm darcy(&Vh, &Wh);
   ConstantCoefficient one(1.0), src(1.0);

   darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
   BilinearForm *M_u = darcy.GetFluxMassForm();
   M_u->AddDomainIntegrator(new VectorMassIntegrator(one));
   M_u->AddBdrFaceIntegrator(new BdrFluxMass(0.7, dim));
   M_u->AddBdrFaceIntegrator(new BdrFluxMass(0.5, dim));

   darcy.GetFluxDivForm()->AddDomainIntegrator(
      new VectorDivergenceIntegrator());
   darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

   BilinearForm *M_p = darcy.GetPotentialMassForm();
   M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
   M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

   Array<int> ess_flux;
   darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);
   DarcyHybridization *dh = darcy.GetHybridization();
   dh->SetAssemblyMode(am);
   dh->EnableNPC();
   Array<int> all(mesh.bdr_attributes.Max());
   all = 1;
   dh->SetEssentialBC(all);

   darcy.Assemble();
   darcy.Finalize();
   taken = dh->CanBatchFluxMassBdrFaces(M_u);

   BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
   b = 0.0;
   x = 0.0;
   darcy.GetPotentialRHS()->Assemble();
   b.GetBlock(1) += *darcy.GetPotentialRHS();
   b.GetBlock(1).SyncAliasMemory(b);

   Vector x_tr(Mh.GetVSize());
   x_tr = 0.0;
   BlockVector r(darcy.GetOffsets());
   Vector r_tr, b_tr, dtr;
   dh->NPCResidual(b, x, x_tr, r, r_tr);

   Operator &S = dh->NPCGradient(x, x_tr);
   dh->NPCReduce(r, r_tr, b_tr);

   dtr.SetSize(b_tr.Size());
   dtr = 0.0;
   {
      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
      REQUIRE(Sm != nullptr);
      b_tr.HostReadWrite();
      dtr.HostReadWrite();
      UMFPackSolver umf(*Sm);
      umf.Mult(b_tr, dtr);
   }

   BlockVector dx(darcy.GetOffsets());
   dx = 0.0;
   dh->NPCRecover(r, dtr, dx);

   Vector qv, pv;
   qv.MakeRef(dx, 0, dx.GetBlock(0).Size());
   pv.MakeRef(dx, dx.GetBlock(0).Size(), dx.GetBlock(1).Size());
   dq.SetSize(qv.Size());
   dq = qv;
   dp.SetSize(pv.Size());
   dp = pv;
   dtr_out.SetSize(dtr.Size());
   dtr_out = dtr;
   dq.HostRead();
   dp.HostRead();
   dtr_out.HostRead();
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

/**
 * @brief LocalFactorMode::Batched under a Device gives the element loop's
 * answer, and the fields survive coming back through the caller's blocks.
 *
 * The globbed [DarcyHybridization][BatchedLinAlg] cases run this comparison
 * with no Device, where every kernel degrades to a host loop and the memory
 * validity machinery is inert -- so they cannot see any of what is under test
 * here. With one configured:
 *
 *  - ReduceRHS() gathers the element-blocked right-hand side with one
 *    mfem::forall over el_u_dofs instead of a per-element host loop, and the
 *    result never touches the host;
 *  - ComputeH() factors every element and forms every Schur complement in one
 *    batch of BatchedLinAlg calls;
 *  - ComputeSolution() scatters the recovered fields back with one kernel and
 *    deliberately does NOT read them back, so they reach the caller
 *    device-valid through two levels of BlockVector alias.
 *
 * That last one is what needs pinning, and it is exactly the failure this
 * file's other case documents seen from inside the library. Dropping
 * ComputeSolution()'s SyncFromBlocks() makes this case fail, and fail with
 * the same signature: the potential comes back EXACTLY zero against a
 * reference of -6.4e-03, because the caller's second view over that range
 * reports host-valid while the answer sits in a block's device buffer.
 *
 * It only fails because Solve() reads the fields back through that second
 * view. A first version read them through the very block objects the library
 * had written, and passed with the sync removed -- the block knows where its
 * own data is. Checking that the guard fires is what found that, and it is
 * the difference between pinning the contract and pinning nothing.
 *
 * Bitwise, and that is a fact about this build rather than about batching:
 * BatchedLinAlg's native backend runs the same kernels::LUFactor/LUSolve that
 * LUFactors does without LAPACK, and kernels::AddMult runs mfem::AddMult's own
 * j-k-i loop. The debug Device has no GPU_BLAS behind it, so the native
 * backend is what runs.
 */
TEST_CASE("Darcy batched local algebra under a device", "[DebugDevice]")
{
   using namespace darcy_alias;
   using LFM = DarcyHybridization::LocalFactorMode;

   Vector tr_ref, pot_ref, q_ref, tr_bat, pot_bat, q_bat;
   Solve(true, tr_ref, pot_ref, LFM::Serial, &q_ref);
   Solve(true, tr_bat, pot_bat, LFM::Batched, &q_bat);

   // There is something to get wrong.
   REQUIRE(tr_ref.Norml2() > 1e-6);
   REQUIRE(pot_ref.Norml2() > 1e-6);
   REQUIRE(q_ref.Norml2() > 1e-6);

   REQUIRE(tr_bat.Size() == tr_ref.Size());
   REQUIRE(pot_bat.Size() == pot_ref.Size());
   REQUIRE(q_bat.Size() == q_ref.Size());

   // Vector::Norml2() reads through Read(), whose default is on_dev = true,
   // so the three checks above left these DEVICE-valid; operator() below is a
   // host access and the debug backend mprotects the host pointer. Reading a
   // device-resident field on the host is exactly what this mode makes the
   // caller say out loud.
   tr_ref.HostRead(); tr_bat.HostRead();
   pot_ref.HostRead(); pot_bat.HostRead();
   q_ref.HostRead(); q_bat.HostRead();

   for (int i = 0; i < tr_ref.Size(); i++)
   {
      REQUIRE(tr_bat(i) == tr_ref(i));
   }
   for (int i = 0; i < pot_ref.Size(); i++)
   {
      REQUIRE(pot_bat(i) == pot_ref(i));
   }
   for (int i = 0; i < q_ref.Size(); i++)
   {
      REQUIRE(q_bat(i) == q_ref(i));
   }
}

/**
 * @brief The same, on the NPC route, in both gradient modes.
 *
 * The linear case above reaches ReduceRHS() and ComputeSolution(). A
 * NONLINEAR problem never reaches either -- NPCEnabled() short-circuits
 * ReduceRHS() -- so NPCReduce()'s gather and NPCRecover()'s scatter, which
 * are the other two kernels LocalFactorMode::Batched installs, have no
 * coverage under a Device without this.
 *
 * GradientMode::MatrixFree is the half that matters most, and it is here
 * because a claim about it turned out to be wrong. ComputeHMode::
 * GradientFactorOnly has no face loop, so it looked like the one end of this
 * chain with nothing to read back -- and ComputeH() was written to skip the
 * host sync there on that reasoning. But the apply that follows,
 * MultNL(GradMult), calls the PER-ELEMENT MultInv(), which reads Af_data,
 * Bf_data and the pivots through raw pointers. Reading the code said so;
 * running this said so louder, since the debug backend mprotects a host
 * pointer whose device copy is the valid one and the case segfaults outright.
 * There is no device-resident end here yet, and this is what keeps that
 * honest.
 */
TEST_CASE("Darcy batched local algebra under a device on the NPC route",
          "[DebugDevice]")
{
   using namespace darcy_alias;
   using LFM = DarcyHybridization::LocalFactorMode;
   using GM = DarcyHybridization::GradientMode;

   const GM gmode = GENERATE(GM::Assembled, GM::MatrixFree);

   Vector q_ref, p_ref, tr_ref, q_bat, p_bat, tr_bat;
   NPCStep(LFM::Serial, gmode, q_ref, p_ref, tr_ref);
   NPCStep(LFM::Batched, gmode, q_bat, p_bat, tr_bat);

   REQUIRE(tr_ref.Norml2() > 1e-6);
   REQUIRE(p_ref.Norml2() > 1e-6);
   REQUIRE(q_ref.Norml2() > 1e-6);

   tr_ref.HostRead(); tr_bat.HostRead();
   p_ref.HostRead(); p_bat.HostRead();
   q_ref.HostRead(); q_bat.HostRead();

   REQUIRE(tr_bat.Size() == tr_ref.Size());
   for (int i = 0; i < tr_ref.Size(); i++) { REQUIRE(tr_bat(i) == tr_ref(i)); }
   for (int i = 0; i < p_ref.Size(); i++) { REQUIRE(p_bat(i) == p_ref(i)); }
   for (int i = 0; i < q_ref.Size(); i++) { REQUIRE(q_bat(i) == q_ref(i)); }
}

/**
 * @brief AssemblyMode::Batched's face kernel under a Device gives the
 * per-face loop's answer, and its results reach the host code that follows.
 *
 * The kernel writes E, G, H and D on the device, and the very next loop --
 * the BOUNDARY face pass, which is not batched -- accumulates into D on the
 * host through DenseMatrix::operator+=. Nothing in the globbed test set can
 * see that: with no Device configured every one of those writes is a host
 * write and the ordering is invisible.
 *
 * It was not hypothetical. This mode had been unreachable for every caller in
 * the tree, so the first time it ran on a device was the first time anyone
 * looked: under Device("debug") it faulted inside AssemblePotMassMatrix(),
 * naming neither the array nor the routine that had left it there, and under
 * CUDA it did not fault at all -- it read a stale host buffer and returned an
 * answer 60% wrong. DarcyHybridization::AssemblePotFaceMatricesBatched()
 * ends with SyncLocalBlocksToHost() for that reason; remove it and this case
 * faults rather than fails, which is the debug backend doing its job.
 */
TEST_CASE("The batched HDG face kernel under a device", "[DebugDevice]")
{
   using namespace darcy_alias;
   using AM = DarcyHybridization::AssemblyMode;

   Vector q_ref, p_ref, tr_ref, q_bat, p_bat, tr_bat;
   bool taken_ref = true, taken_bat = false;
   FaceKernelStep(AM::Serial, q_ref, p_ref, tr_ref, taken_ref);
   FaceKernelStep(AM::Batched, q_bat, p_bat, tr_bat, taken_bat);

   // The kernel was actually taken. Two fallbacks would agree perfectly and
   // test nothing.
   REQUIRE_FALSE(taken_ref);
   REQUIRE(taken_bat);

   REQUIRE(tr_ref.Norml2() > 1e-6);
   REQUIRE(p_ref.Norml2() > 1e-6);
   REQUIRE(q_ref.Norml2() > 1e-6);

   tr_ref.HostRead(); tr_bat.HostRead();
   p_ref.HostRead(); p_bat.HostRead();
   q_ref.HostRead(); q_bat.HostRead();

   // Round-off and not bitwise: the kernel accumulates point by point where
   // the per-face route adds one element matrix. See
   // tests/unit/fem/test_darcy_batched_face.cpp, which measures the level.
   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      REQUIRE(d.Normlinf() <= 1e-12 * std::max(a.Normlinf(), 1e-30));
   };
   close(tr_ref, tr_bat);
   close(p_ref, p_bat);
   close(q_ref, q_bat);
}

/**
 * @brief AssemblyMode::Batched's local-residual kernel under a Device gives
 * the per-element integrator's answer, and its result reaches the host loop
 * that consumes it.
 *
 * The kernel computes every element's flux row on the device; the element
 * loop in MultNL() then reads element el's slice on the HOST. Nothing in the
 * globbed test set can see the difference -- with no Device configured the
 * kernel's Write() and the loop's read are the same buffer. Under `debug`
 * they are not, which is exactly the hazard this file's face-kernel case
 * records: remove the HostRead() and this faults rather than fails.
 */
TEST_CASE("The batched HDG local residual under a device", "[DebugDevice]")
{
   using namespace darcy_alias;
   using AM = DarcyHybridization::AssemblyMode;

   const int order = GENERATE(1, 2);

   Vector q_ref, p_ref, tr_ref, q_bat, p_bat, tr_bat;
   bool taken_ref = true, taken_bat = false;
   LocalResidualStep(AM::Serial, order, q_ref, p_ref, tr_ref, taken_ref);
   LocalResidualStep(AM::Batched, order, q_bat, p_bat, tr_bat, taken_bat);

   // Two fallbacks agree perfectly and test nothing.
   REQUIRE_FALSE(taken_ref);
   REQUIRE(taken_bat);

   REQUIRE(tr_ref.Normlinf() > 1e-3);
   REQUIRE(p_ref.Normlinf() > 1e-3);
   REQUIRE(q_ref.Normlinf() > 1e-3);

   // Round-off and not bitwise, and the kernel is not why: AssemblyMode::
   // Batched switches the face, mass and divergence kernels on as well, and
   // those accumulate point by point. The bitwise pin on the residual kernel
   // alone is in tests/unit/fem/test_darcy_batched_residual.cpp.
   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      REQUIRE(d.Normlinf() <= 1e-11 * std::max(a.Normlinf(), 1e-30));
   };
   close(tr_ref, tr_bat);
   close(p_ref, p_bat);
   close(q_ref, q_bat);
}

/**
 * @brief The batched BOUNDARY flux-mass pass under a Device gives the per-face
 * loop's answer, and it is the pass that found the hazard.
 *
 * DarcyForm::AssembleFluxMassBdrFaces() was the last assembly loop on the
 * hybridized path with no kernel. Batching it made this the SECOND kernel in
 * the flux mass group, and that is what nothing had been before: the element
 * pass hands hat_offsets, Af_f_offsets and hat_dofs_marker to a kernel, and
 * Array<int>::Read() defaults to on_dev = true, so they come back
 * DEVICE-valid while this pass's host half indexes them raw as
 * hat_offsets[e+1]. The first run of it under Device("debug") was a SIGSEGV
 * inside AssembleFluxMassBdrMatricesBatched() with an address and nothing
 * else; under CUDA it would have sized the blocks from stale memory. The three
 * HostRead() calls at the top of that routine are the fix, and removing them
 * makes this case fault rather than fail -- which is the debug backend doing
 * its job.
 *
 * Nothing in the globbed test set can see it: with no Device configured every
 * one of those reads is a host read and the ordering is invisible.
 *
 * **The general shape is worth more than the fix.** The offset arrays are
 * SHARED between the passes, so the second kernel of a chain has to host-read
 * whatever the first one made device-valid. Every kernel added after this one
 * inherits the same obligation.
 */
TEST_CASE("The batched boundary flux mass under a device", "[DebugDevice]")
{
   using namespace darcy_alias;
   using AM = DarcyHybridization::AssemblyMode;

   Vector q_ref, p_ref, tr_ref, q_bat, p_bat, tr_bat;
   bool taken_ref = true, taken_bat = false;
   BdrFluxStep(AM::Serial, q_ref, p_ref, tr_ref, taken_ref);
   BdrFluxStep(AM::Batched, q_bat, p_bat, tr_bat, taken_bat);

   // The kernel was actually taken. Two fallbacks would agree perfectly.
   REQUIRE_FALSE(taken_ref);
   REQUIRE(taken_bat);

   REQUIRE(tr_ref.Norml2() > 1e-6);
   REQUIRE(p_ref.Norml2() > 1e-6);
   REQUIRE(q_ref.Norml2() > 1e-6);

   tr_ref.HostRead(); tr_bat.HostRead();
   p_ref.HostRead(); p_bat.HostRead();
   q_ref.HostRead(); q_bat.HostRead();

   // Round-off and not bitwise, because switching AssemblyMode switches the
   // element mass, the divergence and the face kernels as well. The boundary
   // pass ON ITS OWN is bit-for-bit the loop -- one thread per element summing
   // that element's faces in the loop's order -- and that is measured in
   // tests/unit/fem/test_darcy_batched_bdrflux.cpp.
   auto close = [](const Vector &a, const Vector &b)
   {
      REQUIRE(a.Size() == b.Size());
      Vector d(a);
      d -= b;
      REQUIRE(d.Normlinf() <= 1e-12 * std::max(a.Normlinf(), 1e-30));
   };
   close(tr_ref, tr_bat);
   close(p_ref, p_bat);
   close(q_ref, q_bat);
}

#endif // _WIN32

int main(int argc, char *argv[])
{
   Device device("debug");
   return RunCatchSession(argc, argv, {"[DebugDevice]"});
}
