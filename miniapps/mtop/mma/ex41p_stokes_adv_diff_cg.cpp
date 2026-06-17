//                       MFEM: Stokes + CG IMEX Advection-Diffusion (Parallel)
//                       Based on ex41p.cpp (IMEX) + ex5p.cpp (Stokes)
//
// Compile: place in an MFEM build directory alongside the other examples,
//          add to the CMakeLists/Makefile, then: make ex41p_stokes_adv_diff_cg
//
// Sample runs:
//   # Periodic square, Gaussian blob transported by a sinusoidal body force
//   mpirun -np 4 ./ex41p_stokes_adv_diff_cg \
//     -m ../data/periodic-square.mesh -rs 2 -rp 1 -o 2 \
//     -fa 5 -nu 0.01 -dt 0.005 -tf 2.0 -pv -pvs 20
//
//   # Non-periodic (no-slip walls), higher forcing
//   mpirun -np 4 ./ex41p_stokes_adv_diff_cg \
//     -m ../data/square-disc.mesh -rs 2 -rp 1 -o 2 \
//     -fa 10 -nu 0.01 -dt 0.005 -tf 1.0 -pv
//
// ---------------------------------------------------------------------------
// Description:
//
//   Solves the coupled system:
//
//     STOKES (quasi-static, one solve per time step):
//       -ν Δu + ∇p = f,   ∇·u = 0   in Ω
//
//     ADVECTION-DIFFUSION (IMEX RK time integration):
//       ∂c/∂t + u·∇c - ∇·(κ∇c) = 0  in Ω
//
//   Stokes:  Taylor-Hood H1^d(p+1) × H1(p).
//            The saddle-point system is solved with GMRES preconditioned
//            by a block-diagonal preconditioner (AMG for velocity, scaled
//            mass matrix for pressure, following Cahouet-Chabard).
//            On periodic meshes the pressure null space is fixed by an
//            explicit mean-pressure subtraction after each solve.
//            The linear form f is recreated and assembled each step.
//
//   Body force:  f = amp · (sin 2πy,  sin 2πx)
//            Divergence-free, periodic, drives counter-rotating vortices.
//            Controlled by -fa <amplitude> (default 1.0).
//
//   Scalar:  CG H1(p).  Continuous Galerkin ConvectionIntegrator
//            (explicit); standard DiffusionIntegrator (implicit).
//            IMEX split via MFEM 4.9:
//              IMEXExpImplEuler (1), IMEXRK2 (2), IMEX_DIRK_RK3 (3).

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

using namespace std;
using namespace mfem;

static const real_t pi = std::acos(real_t(-1));

// ============================================================
//  VelocityCoeff — wraps a ParGridFunction as a VectorCoefficient
//  so MFEM advection integrators can read it at quadrature points.
// ============================================================
class VelocityCoeff : public VectorCoefficient
{
   ParGridFunction *vel;
public:
   VelocityCoeff(int vdim, ParGridFunction &u)
      : VectorCoefficient(vdim), vel(&u) {}
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   { vel->GetVectorValue(T, ip, V); }
};

// ============================================================
//  StokesOperator
//
//  Solves the block saddle-point system
//      [ A   B^T ] [ u ]   [ f ]
//      [ B    0  ] [ p ] = [ 0 ]
//  with  A = ν(∇u,∇v),  B = -(p,∇·v).
//
//  Key design choices:
//
//  1. FormSystemMatrix on a_form gives A with essential BCs eliminated.
//     ParallelAssemble on b_form gives B; B^T is computed by Transpose().
//     Essential velocity DOFs eliminate columns of B; B^T is formed
//     afterward so the transpose stays consistent.
//     EliminateRowsCols / EliminateCols return new HypreParMatrix* values;
//     delete those returned matrices.
//
//  2. The body-force linear form f_form is recreated fresh each Solve():
//     delete, new ParLinearForm, AddDomainIntegrator, Assemble().
//     This avoids LinearForm state/accumulation issues across steps.
//
//  3. Pressure null space on periodic / pure-Neumann meshes:
//     GMRES finds a solution; mean-p subtraction handles the null space.
//     We then project to zero mean by subtracting ∫p / ∫1.
//     The weight vector vol_tdof = (1, φ_i) is computed once.
//
//  4. The preconditioner mass matrix Mp_mat must outlive pres_prec
//     (HypreDiagScale stores a raw pointer).  It is kept as a member.
// ============================================================
class StokesOperator
{
public:
   ParFiniteElementSpace *vel_fes, *pres_fes;

   // Coefficients — must outlive the integrators that reference them
   ConstantCoefficient  nu_cf;
   ConstantCoefficient  neg_one_cf;
   ConstantCoefficient *eps_cf = nullptr;   // mass regularization (owned)

   Array<int> vel_ess_tdofs;
   Array<int> block_trueOffsets;   // [0, nU_true, nU_true+nP_true]

   // Forms
   ParBilinearForm      *a_form = nullptr;
   ParMixedBilinearForm *b_form = nullptr;
   ParLinearForm        *f_form = nullptr;   // body force (recreated each step)
   ParLinearForm        *g_form = nullptr;   // divergence source (= 0, cached)
   VectorCoefficient    *f_body_coeff = nullptr;  // non-owning ptr to body force

   // True-dof RHS vectors assembled once
   Vector G_tdof;     // zero divergence source
   Vector vol_tdof;   // (1, φ_i) for mean-pressure removal

   // True-dof system matrices
   HypreParMatrix *A_mat  = nullptr;   // velocity stiffness (owned)
   HypreParMatrix *B_mat  = nullptr;
   HypreParMatrix *Bt_mat = nullptr;

   // Preconditioner components — Mp_mat must outlive pres_prec
   HypreBoomerAMG             *vel_prec  = nullptr;
   HypreParMatrix             *Mp_mat    = nullptr;
   HypreDiagScale             *pres_prec = nullptr;
   BlockDiagonalPreconditioner *blk_prec = nullptr;
   BlockOperator              *stokes_op = nullptr;
   GMRESSolver                *solver    = nullptr;

   ParGridFunction *u_gf = nullptr;
   ParGridFunction *p_gf = nullptr;

   // ── Constructor ─────────────────────────────────────────
   StokesOperator(ParFiniteElementSpace *vfes,
                  ParFiniteElementSpace *pfes,
                  real_t viscosity)
      : vel_fes(vfes), pres_fes(pfes),
        nu_cf(viscosity), neg_one_cf(-1.0)
   {
      u_gf = new ParGridFunction(vel_fes);  *u_gf = 0.0;
      p_gf = new ParGridFunction(pres_fes); *p_gf = 0.0;

      block_trueOffsets.SetSize(3);
      block_trueOffsets[0] = 0;
      block_trueOffsets[1] = vel_fes->GetTrueVSize();
      block_trueOffsets[2] = pres_fes->GetTrueVSize();
      block_trueOffsets.PartialSum();

      a_form = new ParBilinearForm(vel_fes);
      a_form->AddDomainIntegrator(new VectorDiffusionIntegrator(nu_cf));
      // Small mass regularization ε(u,v) makes A strictly positive definite
      // on periodic meshes where constant velocities are in the null space of
      // the pure diffusion operator.  ε = ν·1e-10 is negligible physically
      // but prevents BoomerAMG from breaking on a singular/near-singular matrix.
      eps_cf = new ConstantCoefficient(nu_cf.constant * 1e-10);
      a_form->AddDomainIntegrator(new VectorMassIntegrator(*eps_cf));

      b_form = new ParMixedBilinearForm(vel_fes, pres_fes);
      b_form->AddDomainIntegrator(new VectorDivergenceIntegrator(neg_one_cf));

      f_form = new ParLinearForm(vel_fes);
      g_form = new ParLinearForm(pres_fes);
   }

   // Register body force.  Call before Assemble().
   // The coefficient must outlive this object.
   void SetBodyForce(VectorCoefficient &f_coeff)
   {
      f_body_coeff = &f_coeff;
   }

   // ── Assemble ─────────────────────────────────────────────
   // Builds system matrices, preconditioner, and cached RHS vectors.
   // Call once (or after mesh changes).
   void Assemble(const Array<int> &ess_vel_bdr)
   {
      vel_fes->GetEssentialTrueDofs(ess_vel_bdr, vel_ess_tdofs);

      // System matrices (time-invariant for linear Stokes)
      a_form->Assemble(); a_form->Finalize();
      b_form->Assemble(); b_form->Finalize();
      // Velocity stiffness matrix A in true-dof space.
      // Use ParallelAssemble() directly rather than FormSystemMatrix():
      // on periodic meshes FormSystemMatrix can store the result under
      // a non-HypreParMatrix type in the OperatorHandle, causing
      // A_op.As<HypreParMatrix>() to return nullptr → NaN in AMG.
      // ParallelAssemble() always returns a HypreParMatrix*.
      // Essential-BC elimination on A is handled separately below.
      {
         HypreParMatrix *A_full = a_form->ParallelAssemble();
         // Collective call: every MPI rank must enter elimination, even when
         // vel_ess_tdofs is empty on that rank.  The returned eliminated
         // entries matrix is owned by the caller and must be deleted.
         {
            HypreParMatrix *A_e = A_full->EliminateRowsCols(vel_ess_tdofs);
            delete A_e;
         }
         delete A_mat;
         A_mat = A_full;   // take ownership
      }

      delete B_mat; delete Bt_mat;
      B_mat  = b_form->ParallelAssemble();

      // b_form has trial space = velocity and test space = pressure. Thus B
      // has pressure rows and velocity columns, so essential velocity true DOFs
      // correspond to columns of B.  This elimination is collective: every MPI
      // rank must enter it, even when vel_ess_tdofs is empty on that rank.
      // Build B^T after column elimination to keep the two blocks consistent.
      {
         HypreParMatrix *B_e = B_mat->EliminateCols(vel_ess_tdofs);
         delete B_e;
      }
      Bt_mat = B_mat->Transpose();

      // Cache zero divergence source as a true-dof vector
      g_form->Assemble();
      { HypreParVector *G = g_form->ParallelAssemble();
        G_tdof = *G; delete G; }

      // Cache (1, φ_i) for mean-pressure removal (time-invariant)
      { ConstantCoefficient one(1.0);
        ParLinearForm vf(pres_fes);
        vf.AddDomainIntegrator(new DomainLFIntegrator(one));
        vf.Assemble();
        HypreParVector *V = vf.ParallelAssemble();
        vol_tdof = *V; delete V; }

      // Velocity block preconditioner — BoomerAMG in systems (vector) mode.
      // SetSystemsOptions works with the default byNODES ordering.
      // SetElasticityOptions would be more accurate but requires byVDIM ordering.
      // The mass regularization ε(u,v) already removes the null space, so
      // standard systems-AMG converges correctly.
      // Chebyshev smoother (relax type 16) is symmetric.
      delete vel_prec;
      vel_prec = new HypreBoomerAMG(*A_mat);
      vel_prec->SetSystemsOptions(vel_fes->GetMesh()->Dimension(), true);
      vel_prec->SetPrintLevel(0);
      vel_prec->SetCycleNumSweeps(2, 2);
      vel_prec->SetRelaxType(16);   // Chebyshev (symmetric)

      // Pressure block preconditioner — diagonal scaling of (1/ν) M_p
      // (Cahouet-Chabard).  Mp_mat must outlive pres_prec.
      { ConstantCoefficient inv_nu(1.0 / nu_cf.constant);
        ParBilinearForm mp(pres_fes);
        mp.AddDomainIntegrator(new MassIntegrator(inv_nu));
        mp.Assemble(); mp.Finalize();
        delete pres_prec; delete Mp_mat;
        Mp_mat    = mp.ParallelAssemble();
        pres_prec = new HypreDiagScale(*Mp_mat); }

      // Block saddle-point operator and preconditioner
      delete stokes_op;
      stokes_op = new BlockOperator(block_trueOffsets);
      stokes_op->SetBlock(0, 0, A_mat);
      stokes_op->SetBlock(0, 1, Bt_mat);
      stokes_op->SetBlock(1, 0, B_mat);

      delete blk_prec;
      blk_prec = new BlockDiagonalPreconditioner(block_trueOffsets);
      blk_prec->SetDiagonalBlock(0, vel_prec);
      blk_prec->SetDiagonalBlock(1, pres_prec);

      // Use GMRES: does not require a symmetric preconditioner,
      // more robust than MINRES for saddle-point systems with
      // approximate (non-symmetric) block preconditioners.
      delete solver;
      solver = new GMRESSolver(vel_fes->GetComm());
      solver->SetRelTol(1e-8);
      solver->SetAbsTol(0.0);
      solver->SetMaxIter(3000);
      solver->SetKDim(100);          // restart dimension
      solver->SetPrintLevel(0);
      solver->iterative_mode = true;
      solver->SetOperator(*stokes_op);
      solver->SetPreconditioner(*blk_prec);
   }

   // ── Solve ─────────────────────────────────────────────────
   // Re-assembles the body-force RHS each call. The form is recreated
   // instead of reused, avoiding any LinearForm state from prior steps.
   void Solve()
   {
      // Assemble body-force RHS fresh each call.
      // Delete and recreate avoids any state issues with Update()/Assemble().
      delete f_form;
      f_form = new ParLinearForm(vel_fes);
      if (f_body_coeff)
         f_form->AddDomainIntegrator(new VectorDomainLFIntegrator(*f_body_coeff));
      f_form->Assemble();

      HypreParVector *F = f_form->ParallelAssemble();
      for (int i = 0; i < vel_ess_tdofs.Size(); i++)
         (*F)[vel_ess_tdofs[i]] = 0.0;

      BlockVector RHS(block_trueOffsets), X(block_trueOffsets);
      RHS.GetBlock(0) = *F;
      RHS.GetBlock(1) = G_tdof;
      delete F;

      // Warm start from previous solution; enforce BCs on initial guess
      u_gf->GetTrueDofs(X.GetBlock(0));
      p_gf->GetTrueDofs(X.GetBlock(1));
      for (int i = 0; i < vel_ess_tdofs.Size(); i++)
         X.GetBlock(0)[vel_ess_tdofs[i]] = 0.0;

      solver->Mult(RHS, X);

      u_gf->SetFromTrueDofs(X.GetBlock(0));
      p_gf->SetFromTrueDofs(X.GetBlock(1));

      // Remove mean pressure (null-space fix for periodic/Neumann meshes).
      {
         Vector p_tdof(pres_fes->GetTrueVSize());
         p_gf->GetTrueDofs(p_tdof);
         Vector ones(vol_tdof.Size()); ones = 1.0;
         real_t pmom = InnerProduct(pres_fes->GetComm(), vol_tdof, p_tdof);
         real_t vol  = InnerProduct(pres_fes->GetComm(), vol_tdof, ones);
         *p_gf -= pmom / vol;
      }
   }

   ~StokesOperator()
   {
      // Destroy objects in dependency order: solvers/preconditioners and block
      // operators keep non-owning pointers to the matrices below.
      delete solver;
      delete blk_prec;
      delete stokes_op;
      delete vel_prec;
      delete pres_prec;
      delete Mp_mat;
      delete A_mat; delete B_mat; delete Bt_mat;
      delete a_form; delete b_form; delete f_form; delete g_form;
      delete eps_cf;
      delete u_gf; delete p_gf;
   }
};


// ============================================================
//  CGAdvDiffOperator — IMEX TimeDependentOperator, H1/CG scalar space
//
//    Mult()          -> explicit: M^{-1} K_adv c
//    ImplicitSolve() -> implicit: (M + dt D)^{-1} M c
//
//  This is the continuous-Galerkin counterpart of the DG operator in
//  ex41p_stokes_adv_diff.cpp.  The scalar finite element space is H1(p).
//  There are no DG trace or SIPG face terms here:
//
//    K_adv(c,v) = -(u · grad c, v)
//    D(c,v)     =  (kappa grad c, grad v)
//
//  The implicit step solves
//
//    (M + dt D) c_new = M c_old,
//
//  which corresponds to backward Euler for the diffusive part in the IMEX
//  split expected by MFEM's ODE solvers.
// ============================================================
class CGAdvDiffOperator : public TimeDependentOperator
{
   ParFiniteElementSpace *fes;
   ParMesh               *pmesh;   // non-owning, kept for symmetry/context

   // Velocity coefficient used by ConvectionIntegrator.
   VelocityCoeff         *vel_coeff_owned = nullptr; // owned if created here
   VelocityCoeff         *vel_coeff       = nullptr; // non-owning ptr in use

   // Diffusion coefficient.  CG diffusion only uses volume integrators, so a
   // QuadratureFunctionCoefficient is valid here; no face coefficient is needed.
   ConstantCoefficient    kappa_const_cf;
   Coefficient           *diff_coeff      = nullptr;  // non-owning

   ParBilinearForm *M_form = nullptr;
   ParBilinearForm *K_form = nullptr;
   ParBilinearForm *D_form = nullptr;

   HypreParMatrix *Mmat = nullptr;
   HypreParMatrix *Kmat = nullptr;
   HypreParMatrix *Dmat = nullptr;

   // Cached implicit system A(dt) = M + dt D.
   mutable HypreParMatrix *impl_mat    = nullptr;
   mutable real_t          impl_gamma  = -1.0;
   mutable HyprePCG       *impl_solver = nullptr;
   mutable HypreBoomerAMG *impl_prec   = nullptr;

   void InvalidateImplicitSystem()
   {
      delete impl_solver; impl_solver = nullptr;
      delete impl_prec;   impl_prec   = nullptr;
      delete impl_mat;    impl_mat    = nullptr;
      impl_gamma = -1.0;
   }

   void BuildDiffusion()
   {
      MFEM_VERIFY(diff_coeff != nullptr,
                  "CGAdvDiffOperator: diff_coeff is null!");

      delete D_form;
      D_form = new ParBilinearForm(fes);
      D_form->AddDomainIntegrator(new DiffusionIntegrator(*diff_coeff));
      D_form->Assemble(0);
      D_form->Finalize(0);

      delete Dmat;
      Dmat = D_form->ParallelAssemble();
      InvalidateImplicitSystem();
   }

   void BuildAdvectionForm()
   {
      delete K_form;
      K_form = new ParBilinearForm(fes);
      // Negative sign gives c_t = - u · grad(c) after mass inversion.
      K_form->AddDomainIntegrator(new ConvectionIntegrator(*vel_coeff, -1.0));
   }

public:
   CGAdvDiffOperator(ParFiniteElementSpace *fes_,
                     ParMesh *pmesh_,
                     VelocityCoeff *vc,
                     real_t kappa)
      : TimeDependentOperator(fes_->GetTrueVSize(), 0.0,
                              TimeDependentOperator::IMPLICIT),
        fes(fes_), pmesh(pmesh_), vel_coeff(vc), kappa_const_cf(kappa)
   {
      diff_coeff = &kappa_const_cf;

      M_form = new ParBilinearForm(fes);
      M_form->AddDomainIntegrator(new MassIntegrator);
      M_form->Assemble(0);
      M_form->Finalize(0);
      Mmat = M_form->ParallelAssemble();

      BuildAdvectionForm();
      BuildDiffusion();
   }

   void SetDiffusionCoeff(Coefficient &kappa_cf)
   {
      diff_coeff = &kappa_cf;
   }

   void UpdateDiffusion()
   {
      BuildDiffusion();
   }

   void SetVelocityGF(ParGridFunction &u_new)
   {
      delete Kmat; Kmat = nullptr;
      delete vel_coeff_owned;
      vel_coeff_owned = new VelocityCoeff(u_new.VectorDim(), u_new);
      vel_coeff       = vel_coeff_owned;
      BuildAdvectionForm();
   }

   void UpdateVelocity()
   {
      MFEM_VERIFY(K_form != nullptr,
                  "CGAdvDiffOperator: advection form is null!");
      delete Kmat; Kmat = nullptr;
      K_form->Update();
      K_form->Assemble(0);
      K_form->Finalize(0);
      Kmat = K_form->ParallelAssemble();
   }

   void Mult(const Vector &c, Vector &dc_dt) const override
   {
      MFEM_VERIFY(Kmat != nullptr, "Call UpdateVelocity() first.");

      Vector Kc(c.Size());
      Kmat->Mult(c, Kc);

      HyprePCG ms(fes->GetComm());
      HypreDiagScale mp(*Mmat);
      ms.SetPreconditioner(mp);
      ms.SetOperator(*Mmat);
      ms.SetTol(1e-12);
      ms.SetMaxIter(500);
      ms.SetPrintLevel(0);
      dc_dt = 0.0;
      ms.Mult(Kc, dc_dt);
   }

   void ImplicitSolve(const real_t dt, const Vector &c,
                      Vector &dc_dt) override
   {
      bool need_rebuild = (!impl_mat)
                       || (std::abs(dt - impl_gamma)
                           > 1e-12 * std::max(std::abs(impl_gamma), real_t(1.0)));
      if (need_rebuild)
      {
         InvalidateImplicitSystem();

         impl_mat   = Add(1.0, *Mmat, dt, *Dmat);
         impl_gamma = dt;
         impl_prec = new HypreBoomerAMG(*impl_mat);
         impl_prec->SetPrintLevel(0);
         impl_solver = new HyprePCG(fes->GetComm());
         impl_solver->SetPreconditioner(*impl_prec);
         impl_solver->SetOperator(*impl_mat);
         impl_solver->SetTol(1e-10);
         impl_solver->SetMaxIter(500);
         impl_solver->SetPrintLevel(0);
      }

      Vector rhs(c.Size());
      Mmat->Mult(c, rhs);

      Vector c_new(c.Size());
      c_new = 0.0;
      impl_solver->Mult(rhs, c_new);

      add(1.0 / dt, c_new, -1.0 / dt, c, dc_dt);
   }

   ~CGAdvDiffOperator()
   {
      InvalidateImplicitSystem();
      delete Mmat; delete Kmat; delete Dmat;
      delete M_form; delete K_form; delete D_form;
      delete vel_coeff_owned;
   }
};

// ============================================================
//  main
// ============================================================
int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();

   const char *mesh_file  = "../data/periodic-square.mesh";
   int    ser_ref   = 2;
   int    par_ref   = 1;
   int    order     = 2;
   real_t dt        = -1.0;    // <0 means use CFL condition
   real_t cfl       = 0.3;     // CFL number for automatic dt selection
   real_t t_final   = 1.0;
   real_t nu        = 1e-2;
   real_t kappa     = 1e-3;
   real_t force_amp = 1.0;
   int    ode_id    = 3;
   bool   vis       = true;
   bool   paraview  = false;
   int    pv_steps  = 5;
   const char *pv_dir = "stokes_adv_diff_cg_paraview";
   bool   use_qf    = false;   // Example B: QuadratureFunction κ + external u

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file,  "-m",   "--mesh",            "Mesh file.");
   args.AddOption(&ser_ref,    "-rs",  "--refine-serial",   "Serial refs.");
   args.AddOption(&par_ref,    "-rp",  "--refine-parallel", "Parallel refs.");
   args.AddOption(&order,      "-o",   "--order",           "FE order.");
   args.AddOption(&dt,         "-dt",  "--time-step",
                  "Time step. Negative → use CFL condition.");
   args.AddOption(&cfl,        "-cfl", "--cfl",
                  "CFL number for automatic time step (used when -dt < 0).");
   args.AddOption(&t_final,    "-tf",  "--t-final",         "Final time.");
   args.AddOption(&nu,         "-nu",  "--viscosity",       "Stokes viscosity.");
   args.AddOption(&kappa,      "-kap", "--diffusivity",     "Scalar diffusivity.");
   args.AddOption(&force_amp,  "-fa",  "--force-amp",
                  "Body-force amplitude: f=(amp*sin2πy, amp*sin2πx).");
   args.AddOption(&ode_id,     "-ode", "--ode-solver",
                  "IMEX: 1=IMEXExpImplEuler 2=IMEXRK2 3=IMEX_DIRK_RK3.");
   args.AddOption(&vis,  "-vis","--visualization",
                  "-no-vis","--no-visualization", "GLVis.");
   args.AddOption(&paraview, "-pv","--paraview",
                  "-no-pv","--no-paraview", "ParaView output.");
   args.AddOption(&pv_steps, "-pvs","--paraview-steps",
                  "Save ParaView output every N steps.");
   args.AddOption(&pv_dir,   "-pvd","--paraview-dir",
                  "ParaView output directory.");
   args.AddOption(&use_qf, "-qf","--quadrature-diffusion",
                  "-no-qf","--no-quadrature-diffusion",
                  "Example B: spatially varying κ via QuadratureFunction "
                  "+ external rotational velocity via ParGridFunction.");
   args.Parse();
   if (!args.Good()) { if (myid == 0) args.PrintUsage(cout); return 1; }
   if (myid == 0) args.PrintOptions(cout);

   MFEM_VERIFY(ser_ref >= 0, "Serial refinement level must be nonnegative.");
   MFEM_VERIFY(par_ref >= 0, "Parallel refinement level must be nonnegative.");
   MFEM_VERIFY(order >= 1, "Taylor-Hood Stokes spaces require -o/--order >= 1.");
   MFEM_VERIFY(t_final >= 0.0, "Final time must be nonnegative.");
   MFEM_VERIFY(nu > 0.0, "Stokes viscosity must be positive.");
   MFEM_VERIFY(kappa >= 0.0, "Scalar diffusivity must be nonnegative.");
   MFEM_VERIFY(cfl > 0.0, "CFL number must be positive.");
   MFEM_VERIFY(pv_steps > 0, "ParaView output stride must be positive.");

   // ── Mesh ─────────────────────────────────────────────────
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();
   MFEM_VERIFY(dim == 2,
               "ex41p_stokes_adv_diff currently assumes a 2D mesh.");
   for (int l = 0; l < ser_ref; l++) mesh->UniformRefinement();
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   std::unique_ptr<ParMesh> pmesh_owner(pmesh);
   delete mesh;
   for (int l = 0; l < par_ref; l++) pmesh->UniformRefinement();

   // Collective calls — all ranks participate
   { long long gne = pmesh->GetGlobalNE();
     if (myid == 0) cout << "dim=" << dim
                         << "  global_elements=" << gne << "\n"; }

   H1_FECollection vel_fec(order + 1, dim);
   H1_FECollection pres_fec(order, dim);
   // Continuous Galerkin scalar space for the new CG advection-diffusion miniapp.
   H1_FECollection scal_fec(order, dim);

   ParFiniteElementSpace vel_fes(pmesh, &vel_fec, dim);
   ParFiniteElementSpace pres_fes(pmesh, &pres_fec);
   ParFiniteElementSpace scal_fes(pmesh, &scal_fec);

   HYPRE_BigInt vn = vel_fes.GlobalTrueVSize();
   HYPRE_BigInt pn = pres_fes.GlobalTrueVSize();
   HYPRE_BigInt sn = scal_fes.GlobalTrueVSize();
   if (myid == 0)
      cout << "Velocity DOFs: " << vn << "  Pressure DOFs: " << pn
           << "  Scalar DOFs: " << sn << "\n";

   // Essential BCs: empty for periodic mesh, no-slip for others
   Array<int> vel_ess_bdr;
   if (pmesh->bdr_attributes.Size() > 0)
   {
      vel_ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_bdr = 1;
   }

   // ── Stokes ───────────────────────────────────────────────
   // Body force f = force_amp * (sin 2πy, sin 2πx).
   // Divergence-free and periodic → compatible with ∇·u=0 and the mesh.
   VectorFunctionCoefficient stokes_force(dim,
      [&force_amp](const Vector &x, Vector &f)
      {
         const real_t tp = 2.0 * pi;
         f.SetSize(x.Size());
         f = 0.0;
         f(0) = force_amp * sin(tp * x(1));
         f(1) = force_amp * sin(tp * x(0));
      });

   StokesOperator stokes(&vel_fes, &pres_fes, nu);
   stokes.SetBodyForce(stokes_force);
   stokes.Assemble(vel_ess_bdr);
   stokes.Solve();
   ParGridFunction &u_gf = *stokes.u_gf;

   // ── CFL-based time step ───────────────────────────────────
   // Explicit-advection CFL estimate: dt <= C_cfl*h_min/((2p+1)*||u||_max)
   // h_min: smallest element size (collective min over all ranks).
   // ||u||_max: maximum velocity magnitude (collective max over all ranks).
   bool dt_is_auto = (dt < 0.0);

   // Compute dt_cfl = cfl * h_min / ((2p+1) * umax) using all quadrature points.
   // When print=true, rank 0 prints:  CFL: h_min=...  umax=...  dt_cfl=...
   // Both MPI reductions are collective — all ranks must call this together.
   auto compute_cfl_dt = [&](real_t cfl_num, bool print) -> real_t
   {
      // h_min: minimum element size across all local elements
      real_t h_min_local = std::numeric_limits<real_t>::max();
      for (int i = 0; i < pmesh->GetNE(); i++)
         h_min_local = std::min(h_min_local,
                                pmesh->GetElementSize(i, 1));
      real_t h_min;
      MPI_Allreduce(&h_min_local, &h_min, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MIN,
                    pmesh->GetComm());

      // umax: maximum velocity magnitude over all quadrature points
      real_t umax_local = 0.0;
      for (int i = 0; i < pmesh->GetNE(); i++)
      {
         const IntegrationRule &ir =
            IntRules.Get(pmesh->GetElementGeometry(i),
                         vel_fes.GetOrder(i) + 1);
         ElementTransformation *T = pmesh->GetElementTransformation(i);
         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            T->SetIntPoint(&ir.IntPoint(q));
            Vector uq(dim);
            u_gf.GetVectorValue(*T, ir.IntPoint(q), uq);
            umax_local = std::max(umax_local, uq.Norml2());
         }
      }
      real_t umax;
      MPI_Allreduce(&umax_local, &umax, 1,
                    MPITypeMap<real_t>::mpi_type, MPI_MAX,
                    pmesh->GetComm());

      if (umax < 1e-14) return t_final;   // zero velocity: no CFL limit
      real_t dt_cfl = cfl_num * h_min / ((2.0 * order + 1.0) * umax);
      if (print && myid == 0)
         cout << "  CFL: h_min=" << scientific << setprecision(3) << h_min
              << "  umax=" << umax
              << "  dt_cfl=" << dt_cfl
              << "  target_CFL#=" << cfl_num
              << defaultfloat << "\n";
      return dt_cfl;
   };

   if (dt_is_auto)
      dt = compute_cfl_dt(cfl, true);

   // ── Initial scalar ────────────────────────────────────────
   ParGridFunction c_gf(&scal_fes);
   FunctionCoefficient c0([](const Vector &x) -> real_t {
      real_t r2 = (x(0)-0.5)*(x(0)-0.5) + (x(1)-0.5)*(x(1)-0.5);
      return exp(-40.0 * r2);
   });
   c_gf.ProjectCoefficient(c0);

   // ── CG IMEX advection-diffusion ──────────────────────────
   //
   // EXAMPLE A (default): constant kappa, velocity from Stokes u_gf.
   //
   // EXAMPLE B: spatially varying diffusion via QuadratureFunction,
   //   and velocity from a separate ParGridFunction.
   //   Activated by -qf flag below.  Demonstrates:
   //     1. Build a QuadratureSpace on the scalar FE mesh.
   //     2. Project a FunctionCoefficient κ(x) onto it → QuadratureFunction.
   //     3. Wrap it in a QuadratureFunctionCoefficient.
   //     4. Pass it to SetDiffusionCoeff() + UpdateDiffusion().
   //     5. Set a custom velocity ParGridFunction via SetVelocityGF().

   if (use_qf && myid == 0) cout << "Using QuadratureFunction diffusion + "
                                    "ParGridFunction velocity (Example B)\n";

   // ── QuadratureSpace + QuadratureFunction for κ(x) ────────
   // Use the same integration order as the diffusion form (order+1).
   // κ(x,y) = kappa * (1 + 0.9*sin(2πx)*sin(2πy))
   //          — smoothly varying between 0.1·kappa and 1.9·kappa.
   std::unique_ptr<QuadratureSpace> qs;
   std::unique_ptr<QuadratureFunction> kappa_qf;
   std::unique_ptr<QuadratureFunctionCoefficient> kappa_qfc;
   // kappa_func_cf: the plain FunctionCoefficient for κ(x).
   // It is used to fill the QuadratureFunction when -qf is enabled.
   std::unique_ptr<FunctionCoefficient> kappa_func_cf(
      new FunctionCoefficient([&kappa](const Vector &x) -> real_t {
         return kappa * (1.0 + 0.9*sin(2.0*pi*x(0))*sin(2.0*pi*x(1)));
      }));

   if (use_qf)
   {
      int qorder = 2 * order + 2;
      qs.reset(new QuadratureSpace(pmesh, qorder));
      kappa_qf.reset(new QuadratureFunction(qs.get(), 1));

      // Fill QF by evaluating kappa_func_cf at each quadrature point
      {
         for (int e = 0; e < pmesh->GetNE(); e++)
         {
            const IntegrationRule &ir = qs->GetIntRule(e);
            ElementTransformation *T  = pmesh->GetElementTransformation(e);
            Vector vals;
            kappa_qf->GetValues(e, vals);
            for (int q = 0; q < ir.GetNPoints(); q++)
            {
               T->SetIntPoint(&ir.IntPoint(q));
               vals(q) = kappa_func_cf->Eval(*T, ir.IntPoint(q));
            }
         }
      }

      kappa_qfc.reset(new QuadratureFunctionCoefficient(*kappa_qf));

      if (myid == 0)
         cout << "Projected κ(x) onto QuadratureSpace (order=" << qorder
              << ", npts=" << qs->GetSize() << " local)\n";
   }

   // ── Optional separate velocity ParGridFunction ─────────────
   // For Example B we create a rotational velocity field
   //   u_ext = (-sin(πy)cos(πx),  sin(πx)cos(πy))
   // projected onto the velocity FE space, independent of Stokes.
   std::unique_ptr<ParGridFunction> u_ext;
   if (use_qf)
   {
      u_ext.reset(new ParGridFunction(&vel_fes));
      VectorFunctionCoefficient rot_vel(dim,
         [](const Vector &x, Vector &v) {
            v.SetSize(x.Size()); v = 0.0;
            v(0) = -sin(pi * x(1)) * cos(pi * x(0));
            v(1) =  sin(pi * x(0)) * cos(pi * x(1));
         });
      u_ext->ProjectCoefficient(rot_vel);
      if (myid == 0) cout << "External rotational velocity projected.\n";
   }

   // ── Construct CGAdvDiffOperator ────────────────────────────
   VelocityCoeff vel_coeff(dim, u_gf);           // default: Stokes velocity
   CGAdvDiffOperator ad_op(&scal_fes, pmesh, &vel_coeff, kappa);

   if (use_qf)
   {
      // CG diffusion has only domain integrators, so the QFC is safe here.
      ad_op.SetDiffusionCoeff(*kappa_qfc);
      ad_op.UpdateDiffusion();

      // Switch to the external rotational velocity ParGridFunction
      ad_op.SetVelocityGF(*u_ext);
   }

   ad_op.UpdateVelocity();

   ODESolver *ode = nullptr;
   switch (ode_id)
   {
      case 1:  ode = new IMEXExpImplEuler; break;
      case 2:  ode = new IMEXRK2;          break;
      case 3:  ode = new IMEX_DIRK_RK3;    break;
      default:
         if (myid == 0) cerr << "Unknown -ode " << ode_id << "\n";
         return 3;
   }
   ode->Init(ad_op);

   Vector C(scal_fes.GetTrueVSize());
   c_gf.GetTrueDofs(C);
   // ── Scalar mass diagnostic: mass(c) = ∫_Ω c dx ─────────────
   ConstantCoefficient one_cf(1.0);
   ParLinearForm mass_lf(&scal_fes);
   mass_lf.AddDomainIntegrator(new DomainLFIntegrator(one_cf));
   mass_lf.Assemble();

   HypreParVector *mass_vec_h = mass_lf.ParallelAssemble();
   Vector mass_vec = *mass_vec_h;
   delete mass_vec_h;

   auto scalar_mass = [&]() -> real_t
   {
      return InnerProduct(scal_fes.GetComm(), mass_vec, C);
   };

   const real_t mass0 = scalar_mass();
   if (myid == 0)
   {
      cout << "initial mass(c)=" << setprecision(16) << mass0 << "\n";
   }

   // ── GLVis ─────────────────────────────────────────────────
   socketstream c_sock, u_sock;
   if (vis)
   {
      char vishost[] = "localhost"; int visport = 19916;
      c_sock.open(vishost, visport); c_sock.precision(8);
      u_sock.open(vishost, visport); u_sock.precision(8);
      c_sock << "parallel " << Mpi::WorldSize() << " " << myid
             << "\nsolution\n";
      pmesh->Print(c_sock); c_gf.Save(c_sock);
      c_sock << "window_title 'c t=0'\nkeys cmm\n" << flush;
      u_sock << "parallel " << Mpi::WorldSize() << " " << myid
             << "\nsolution\n";
      pmesh->Print(u_sock); u_gf.Save(u_sock);
      u_sock << "window_title 'Stokes u'\nkeys Rjlmm\n" << flush;
   }

   // ── ParaView ──────────────────────────────────────────────
   ParaViewDataCollection *pvdc = nullptr;
   if (paraview)
   {
      pvdc = new ParaViewDataCollection(pv_dir, pmesh);
      pvdc->SetPrefixPath("");
      pvdc->SetLevelsOfDetail(order);
      pvdc->SetDataFormat(VTKFormat::BINARY);
      pvdc->SetHighOrderOutput(true);
      pvdc->RegisterField("velocity", &u_gf);
      pvdc->RegisterField("pressure", stokes.p_gf);
      pvdc->RegisterField("scalar",   &c_gf);
      pvdc->SetCycle(0); pvdc->SetTime(0.0);
      pvdc->Save();
      if (myid == 0) cout << "ParaView → " << pv_dir << "/\n";
   }

   // ── Time loop ─────────────────────────────────────────────
   real_t t = 0.0; bool last = false; int step = 0;
   while (!last)
   {
      step++;

      stokes.Solve();
      ad_op.UpdateVelocity();

      // Compute CFL every step (collective); print alongside step info.
      // If in auto-dt mode, use dt_cfl as the current step size.
      real_t dt_cfl = compute_cfl_dt(cfl, false);
      if (dt_is_auto) dt = dt_cfl;

      // Clamp after the optional CFL update. Otherwise automatic time-step
      // selection can overwrite the final-step clamp and step past t_final.
      if (t + dt >= t_final - 1e-10*dt)
      {
         dt = t_final - t;
         last = true;
      }
      if (dt <= 0.0) { break; }

      ode->Step(C, t, dt);
      c_gf.SetFromTrueDofs(C);

      // Per-step output — ComputeL2Error and scalar mass are collective
      {
         ConstantCoefficient zero(0.0);
         real_t l2 = c_gf.ComputeL2Error(zero);
         real_t mass = scalar_mass();
         real_t rel_mass_err = std::abs(mass - mass0) /
                               std::max(std::abs(mass0), real_t(1e-30));

         if (myid == 0)
         {
            // CFL# = dt / dt_cfl * target_cfl
            real_t cfl_used = (dt_cfl > 0.0) ? dt / dt_cfl * cfl : 0.0;
            cout << "step " << setw(4) << step
                 << "  t="          << scientific << setprecision(4) << t
                 << "  dt="         << dt
                 << "  dt_cfl="     << dt_cfl
                 << "  CFL#="       << setprecision(3) << cfl_used
                 << "  ||c||="      << setprecision(5) << l2
                 << "  mass(c)="    << setprecision(12) << mass
                 << "  rel_mass_err=" << setprecision(3) << rel_mass_err
                 << defaultfloat << "\n";
         }
      }

      if (vis && step % 5 == 0)
      {
         c_sock << "parallel " << Mpi::WorldSize() << " " << myid
                << "\nsolution\n";
         pmesh->Print(c_sock); c_gf.Save(c_sock);
         c_sock << "window_title 'c t=" << t << "'\n" << flush;
      }
      if (pvdc && step % pv_steps == 0)
      {
         pvdc->SetCycle(step); pvdc->SetTime(t);
         pvdc->Save();
      }
   }

   // ── Save final solution ───────────────────────────────────
   { ostringstream fn;
     fn << "scalar_cg_c." << setfill('0') << setw(6) << myid << ".gf";
     ofstream ofs(fn.str()); ofs.precision(8); c_gf.Save(ofs); }
   { ostringstream fn;
     fn << "velocity_cg_u." << setfill('0') << setw(6) << myid << ".gf";
     ofstream ofs(fn.str()); ofs.precision(8); u_gf.Save(ofs); }

   if (myid == 0) cout << "Done.  t=" << t << "\n";

   if (pvdc) { pvdc->SetCycle(step); pvdc->SetTime(t); pvdc->Save();
               delete pvdc; }

   delete ode;

   // The optional qf/velocity coefficient objects are owned by unique_ptrs
   // declared before ad_op, so they outlive ad_op and are released during
   // stack unwinding.

   // pmesh is owned by pmesh_owner, which was constructed before the finite
   // element spaces and operators. It is therefore destroyed after all stack
   // MFEM objects that refer to it. Hypre::Init() handles HYPRE finalization;
   // do not call HYPRE_Finalize() while Hypre-backed MFEM objects are still
   // unwinding.
   return 0;
}
