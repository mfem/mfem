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
//
//               ------------------------------------------------
//               HDG device harness: what runs where, stage by stage
//               ------------------------------------------------
//
// Compile: only in an MFEM_USE_CUDA build. See miniapps/hdg/makefile -- the
// target drops out of SEQ_MINIAPPS otherwise, rather than building a binary
// that would report the same numbers as convdiff.
//
// Sample runs:
//    hdgdevice -d cuda -n 64 -o 2
//    hdgdevice -d cuda -n 96 -o 2 -cudss
//    hdgdevice -d cpu  -n 64 -o 2            (the control)
//
// WHAT THIS IS FOR, and it is not a speed claim.
//
// doc/HDG-DEVICE-OFFLOAD.md's target is a full-device path with no transfer
// back to the host until the outer driver writes output, and its gate is that
// no step of it can be landed alone and show a gain -- a device kernel whose
// neighbours are on the host pays more in transfer than it saves. Every
// device-shaped setting in DarcyHybridization is therefore off by default,
// and the honest question a caller can ask today is not "how much faster" but
// "which stages have somewhere to run, and where does the data still have to
// come back". That question was previously answerable only by a scratch probe
// and a reading of the source, which is how AssemblyMode::Batched managed to
// be unreachable for every caller in the tree for three commits without
// anyone noticing.
//
// So this prints a LEDGER: per stage, whether the device path was actually
// taken -- asked of the library, never inferred from a timing -- and what the
// stage cost. It also solves the problem twice, once with every device
// setting on and once with none, and compares the answers, because a stage
// that silently fell back and a stage that ran are indistinguishable from the
// wall clock alone.
//
// The problem is a hybridized convection-diffusion system with NPC enabled: a
// diffusion and an upwinded convection term on the potential mass constraint,
// which is a SumIntegrator of two and is what the batched face kernel covers.
// NPC is not incidental -- the kernel writes H into H_data, which only the NPC
// route reads.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

#ifndef MFEM_USE_CUDA
#error This miniapp exists to exercise the device path and needs MFEM_USE_CUDA.
#endif

using namespace std;
using namespace mfem;

namespace
{

/// One stage of the hybridized solve, and what is known about it.
struct Stage
{
   const char *name;
   const char *device;   ///< "device", "host", or why not
   real_t secs;
};

void PrintLedger(const Array<Stage*> &stages, real_t total)
{
   cout << "\n"
        << "  stage                        where            time (s)\n"
        << "  ---------------------------------------------------------\n";
   for (Stage *s : stages)
   {
      cout << "  " << left << setw(28) << s->name
           << setw(17) << s->device
           << right << fixed << setprecision(4) << s->secs << "\n";
   }
   cout << "  ---------------------------------------------------------\n"
        << "  " << left << setw(45) << "total" << right << fixed
        << setprecision(4) << total << "\n\n";
}

} // namespace

int main(int argc, char *argv[])
{
   const char *device_config = "cuda";
   int order = 2, n = 64;
   bool use_cudss = false;
   bool baseline = true;
   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure(). "
                  "Pass 'cpu' for the control: every setting below is then a "
                  "different route to the same host loops.");
   args.AddOption(&order, "-o", "--order", "Finite element order.");
   args.AddOption(&n, "-n", "--ncells", "Cells per side of the unit square.");
   args.AddOption(&use_cudss, "-cudss", "--cudss", "-no-cudss", "--no-cudss",
                  "Solve the trace system with cuDSS, which is a DIRECT solve "
                  "that stays on the device -- the one thing that keeps the "
                  "end of the chain off the host. Needs MFEM_USE_CUDSS; "
                  "without it the trace solve falls back to UMFPACK and the "
                  "ledger says so.");
   args.AddOption(&baseline, "-cmp", "--compare", "-no-cmp", "--no-compare",
                  "Also solve with every device setting off and compare, "
                  "which is what separates a stage that ran from a stage that "
                  "silently fell back.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   Mesh mesh = Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
   const int dim = mesh.Dimension();

   L2_FECollection u_coll(order, dim, BasisType::GaussLobatto);
   L2_FECollection p_coll(order, dim);
   DG_Interface_FECollection t_coll(order, dim);

   Vector sol_ref;
   real_t tr_norm_ref = 0.;

   // Two passes: the device settings, then the control. The second is skipped
   // with -no-cmp.
   for (int pass = 0; pass < (baseline ? 2 : 1); pass++)
   {
      const bool on = (pass == 0);
      FiniteElementSpace Vh(&mesh, &u_coll, dim);
      FiniteElementSpace Wh(&mesh, &p_coll);
      FiniteElementSpace Mh(&mesh, &t_coll);

      DarcyForm darcy(&Vh, &Wh);
      ConstantCoefficient one(1.0);
      VectorFunctionCoefficient vel(dim, [](const Vector &X, Vector &v)
      {
         v(0) = 1.0 + 0.5 * std::sin(M_PI * X(1));
         v(1) = -0.7 + 0.3 * std::cos(M_PI * X(0));
      });
      FunctionCoefficient src([](const Vector &X)
      {
         return std::sin(M_PI * X(0)) * std::sin(M_PI * X(1));
      });

      darcy.GetPotentialRHS()->AddDomainIntegrator(new DomainLFIntegrator(src));
      darcy.GetFluxMassForm()->AddDomainIntegrator(
         new VectorMassIntegrator(one));
      darcy.GetFluxDivForm()->AddDomainIntegrator(
         new VectorDivergenceIntegrator());
      darcy.GetFluxDivForm()->AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGNormalTraceIntegrator(-1.0)));

      BilinearForm *M_p = darcy.GetPotentialMassForm();
      M_p->AddInteriorFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));
      M_p->AddInteriorFaceIntegrator(
         new HDGConvectionUpwindedIntegrator(vel, 1.0, 0.5));
      M_p->AddBdrFaceIntegrator(new HDGDiffusionIntegrator(one, 1.0));

      Array<int> ess_flux;
      darcy.EnableHybridization(&Mh, new NormalTraceJumpIntegrator(), ess_flux);

      DarcyHybridization *dh = darcy.GetHybridization();
      if (on)
      {
         dh->SetAssemblyMode(DarcyHybridization::AssemblyMode::Batched);
         dh->SetLocalFactorMode(
            DarcyHybridization::LocalFactorMode::Batched);
      }
      dh->EnableNPC();
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      dh->SetEssentialBC(ess_bdr);

      StopWatch sw, total;
      total.Start();

      sw.Start();
      darcy.Assemble();
      darcy.Finalize();
      sw.Stop();
      // EVERY line is queried. This one used to read a literal "host" and went
      // on reading it after the element kernels landed -- the exact staleness
      // this harness exists to prevent, in the harness itself.
      const DarcyForm *cdarcy = &darcy;
      auto *fm = const_cast<BilinearForm*>(cdarcy->GetFluxMassForm());
      auto *pm = const_cast<BilinearForm*>(cdarcy->GetPotentialMassForm());
      auto *bb = const_cast<MixedBilinearForm*>(cdarcy->GetFluxDivForm());
      const bool el_dev = dh->CanBatchFluxMass(fm) && dh->CanBatchDiv(bb);
      Stage s_asm{"assembly, total", "mixed", sw.RealTime()};
      Stage s_mass{"  flux mass + divergence",
                   el_dev ? "device kernel" : "host per-element", 0.};
      Stage s_pmass{"  potential mass",
                    dh->CanBatchPotMass(pm) ? "device kernel"
                    : "host (or no domain term)", 0.};
      Stage s_face{"  face constraint, interior",
                   dh->CanBatchPotFaceAssembly() ? "device kernel"
                   : "host per-face", 0.};
      Stage s_bface{"  face constraint, boundary",
                    dh->CanBatchPotBdrFaceAssembly() ? "device kernel"
                    : "host per-face", 0.};
      // Three states here, not two, and the middle one is why this line was
      // wrong twice over: it read a LITERAL "host per-face" after
      // AssembleFluxMassBdrMatricesBatched() landed, and this problem has no
      // flux mass boundary face integrator at all, so "host per-face" named a
      // loop that never runs. What that pass batches is the SCATTER only --
      // no BilinearFormIntegrator in the library returns the one-sided block
      // of a vector L2 or an H(div) flux space, so there is nothing for a
      // quadrature kernel to dispatch on. See CanBatchFluxMassBdrFaces().
      Array<BilinearFormIntegrator*> *fm_bfbfi = fm->GetBFBFI();
      const bool fbdr_any = fm_bfbfi && fm_bfbfi->Size() > 0;
      Stage s_fbdr{"  flux mass boundary faces",
                   !fbdr_any ? "no such term"
                   : (dh->CanBatchFluxMassBdrFaces(fm) ? "device scatter"
                      : "host per-face"), 0.};

      BlockVector b(darcy.GetOffsets()), x(darcy.GetOffsets());
      b = 0.0;
      x = 0.0;
      darcy.GetPotentialRHS()->Assemble();
      b.GetBlock(1) += *darcy.GetPotentialRHS();
      // The caller contract: a device operation on a BlockVector's block
      // leaves its result in that alias. See doc/HDG-DEVICE-OFFLOAD.md step 0.
      b.GetBlock(1).SyncAliasMemory(b);

      Vector x_tr(Mh.GetVSize());
      x_tr = 0.0;
      BlockVector r(darcy.GetOffsets());
      Vector r_tr, b_tr, dtr;

      sw.Clear();
      sw.Start();
      dh->NPCResidual(b, x, x_tr, r, r_tr);
      sw.Stop();
      // A query, for the same reason the assembly lines are: the batched
      // local residual refuses an integrator it cannot weigh, silently, and
      // its consumer is still the host element loop.
      Stage s_res{"NPC residual",
                  dh->CanBatchLocalResidual() ? "device kernel"
                  : "host integrators", sw.RealTime()};

      sw.Clear();
      sw.Start();
      Operator &S = dh->NPCGradient(x, x_tr);
      sw.Stop();
      Stage s_grad{"NPC gradient (factor+Schur)",
                   dh->CanBatchLocalSolve() ? "device batched" : "host loop",
                   sw.RealTime()};
      // The STATE-CARRYING face constraint, re-evaluated once per element
      // per Newton step, which is a different question from the
      // assembly-time s_face above. This problem's constraint is linear, so
      // the honest answer is that there is no such term here.
      Stage s_gface{"  of which NL constraint",
                    dh->CanBatchNLFaceGrad() ? "device kernel"
                    : "host or linear", 0.};

      sw.Clear();
      sw.Start();
      dh->NPCReduce(r, r_tr, b_tr);
      sw.Stop();
      Stage s_red{"NPC reduce (local solves)",
                  dh->CanBatchLocalSolve() ? "device batched" : "host loop",
                  sw.RealTime()};

      SparseMatrix *Sm = dynamic_cast<SparseMatrix*>(&S);
      MFEM_VERIFY(Sm, "the assembled trace operator is wanted here");

      dtr.SetSize(b_tr.Size());
      dtr = 0.0;

      const char *tr_where = "host (UMFPack)";
      sw.Clear();
      sw.Start();
      if (on && use_cudss)
      {
#ifdef MFEM_USE_CUDSS
         CuDSSSolver cudss;
         cudss.SetOperator(*Sm);
         cudss.Mult(b_tr, dtr);
         tr_where = "device (cuDSS)";
#else
         MFEM_ABORT("-cudss needs MFEM_USE_CUDSS; configure with "
                    "MFEM_USE_CUDSS=YES CUDSS_INCLUDE_DIR=... "
                    "CUDSS_LIBRARY_DIR=... (they are multiarch, so both are "
                    "needed -- CUDSS_DIR=/usr yields /usr/lib, which has no "
                    "libcudss.so).");
#endif
      }
      else
      {
#ifdef MFEM_USE_SUITESPARSE
         UMFPackSolver umf(*Sm);
         umf.Mult(b_tr, dtr);
#else
         GSSmoother gs(*Sm);
         GMRESSolver lin;
         lin.SetOperator(*Sm);
         lin.SetPreconditioner(gs);
         lin.SetKDim(200);
         lin.SetMaxIter(2000);
         lin.SetRelTol(1e-12);
         lin.SetPrintLevel(-1);
         lin.Mult(b_tr, dtr);
         tr_where = "host (GMRES+GS)";
#endif
      }
      sw.Stop();
      Stage s_tr{"trace solve", tr_where, sw.RealTime()};

      BlockVector dx(darcy.GetOffsets());
      dx = 0.0;
      sw.Clear();
      sw.Start();
      dh->NPCRecover(r, dtr, dx);
      sw.Stop();
      Stage s_rec{"NPC recover (local solves)",
                  dh->CanBatchLocalSolve() ? "device batched, no readback"
                  : "host loop", sw.RealTime()};

      total.Stop();

      cout << "\n=== " << (on ? "device settings ON" : "control: all OFF")
           << " ===\n"
           << "  elements " << mesh.GetNE()
           << ", trace dofs " << Mh.GetVSize()
           << ", face constraint integrators "
           << dh->NumPotFaceConstraintIntegrators() << "\n";

      Array<Stage*> stages;
      stages.Append(&s_asm);
      stages.Append(&s_mass);
      stages.Append(&s_pmass);
      stages.Append(&s_face);
      stages.Append(&s_bface);
      stages.Append(&s_fbdr);
      stages.Append(&s_res);
      stages.Append(&s_grad);
      stages.Append(&s_gface);
      stages.Append(&s_red);
      stages.Append(&s_tr);
      stages.Append(&s_rec);
      PrintLedger(stages, total.RealTime());

      // The answer, read on the host -- which is itself the transfer the
      // target forbids, and is deliberately the LAST thing that happens.
      Vector sol(dx.GetBlock(1));
      sol.HostRead();
      if (on)
      {
         sol_ref = sol;
         tr_norm_ref = dtr.Norml2();
      }
      else
      {
         Vector d(sol_ref);
         d -= sol;
         cout << "  potential:  |device - control| / |control| = "
              << scientific << setprecision(3)
              << ((sol.Norml2() > 0.) ? (d.Norml2() / sol.Norml2()) : 0.)
              << "\n"
              << "  trace norm: " << tr_norm_ref << " against "
              << dtr.Norml2() << "\n\n";
      }
   }

   // Four of this list's five bullets have been struck since it was written,
   // each by a kernel that landed after it -- the NPC residual, the
   // state-carrying face constraint, the flux mass boundary faces' scatter
   // and ComputeElementH()'s face-PAIR loop. That is the same staleness the
   // ledger above was carrying, so what is left is written as what the
   // library REFUSES rather than as what nobody has built.
   cout << "What is NOT on the device:\n"
        << "  * the flux mass boundary faces' INTEGRATOR. The scatter is a\n"
        << "    kernel, but no BilinearFormIntegrator in the library returns\n"
        << "    the one-sided block of a vector L2 or an H(div) flux space,\n"
        << "    so there is nothing for a quadrature kernel to dispatch on;\n"
        << "  * any residual or face-constraint integrator the kernels\n"
        << "    refuse -- a conductivity that is a host std::function of the\n"
        << "    current potential above all. The ledger's own lines say\n"
        << "    which of them THIS problem hits, and a linear constraint\n"
        << "    reads as no such term rather than as a refusal;\n"
        << "  * the scatter into the trace SparseMatrix;\n"
        << "  * the readbacks. SyncLocalBlocksToHost() is still called from\n"
        << "    the assembly and from the local-block routines, so the chain\n"
        << "    returns to the host between groups -- COUNTED, not inferred:\n"
        << "    nine call sites across darcyform.cpp and\n"
        << "    darcyhybridization.cpp as this was written.\n";

   return 0;
}
