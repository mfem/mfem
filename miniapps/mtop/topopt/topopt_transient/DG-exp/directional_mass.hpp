// directional_mass.hpp -- directional accumulated-mass constraint for
//                         density-based topology optimization (MFEM).
//
// Adds, on top of the standard volume budget, a constraint that limits how much
// material ACCUMULATES along a constant direction beta.  The accumulation field
// m is the streamline integral of the (physical) density along beta:
//
//        beta . grad m = rho~     in Omega,        m = 0 on the inflow boundary
//                                                  Gamma_in = { beta.n < 0 },
//
// and the constrained quantity is the total accumulated mass
//
//        g_dm(rho) = INT_Omega m dx .
//
// Because the transport solve, the source rho~, and the integral are all LINEAR,
// g_dm is AFFINE in the design: its sensitivity is a CONSTANT vector
//
//        d g_dm / d rho~ = S^T T^{-T} c_m   (one adjoint transport solve),
//
// where T is the DG-upwind transport matrix, S the (H1 -> DG) source coupling,
// and c_m the DG volume weights (c_m . m = INT m dx).  The driver filters this
// with the existing M F^{-1} chain rule to get d g_dm / d rho and hands it to
// MMA as a second linear inequality constraint -- no per-iteration transport
// solve is needed.
//
// Discretization (steady linear advection, MFEM ex9 building blocks):
//   T = ConvectionIntegrator(beta, +1)                       (volume  INT (beta.grad m) w)
//     + NonconservativeDGTraceIntegrator(beta, +1)  on        interior + boundary faces
//   == full upwind flux; the homogeneous inflow m=0 is the natural BC (no inflow
//   load is assembled), and the outflow face term is included on boundary faces.
//   This is exactly the negative of the ex9 advection operator (which assembles
//   the same integrators with alpha = -1 for du/dt = -beta.grad u).
//
// Mirrors topopt.hpp: a serial DirectionalMass and an MFEM_USE_MPI parallel
// ParDirectionalMass with identical mathematics on local true-dof chunks.

#ifndef DIRECTIONAL_MASS_HPP
#define DIRECTIONAL_MASS_HPP

#include "mfem.hpp"
#include <memory>

using namespace mfem;

// GMRES + BlockILU solve of a (possibly non-symmetric) DG operator.  Shared by
// the forward (T m = b) and adjoint (T^T lambda = c) transport solves.  The
// GMRES solver is supplied by the caller so the parallel class can construct it
// with its MPI communicator (global inner products) while the serial class uses
// the default (communicator-free) solver.
inline int DGTransportSolve(GMRESSolver &gmres, const Operator &A, Solver &prec,
                            const Vector &b, Vector &x,
                            real_t rtol = 1e-12, int maxit = 5000)
{
   gmres.SetKDim(100);
   gmres.SetRelTol(rtol);
   gmres.SetAbsTol(0.0);
   gmres.SetMaxIter(maxit);
   gmres.SetPrintLevel(0);
   gmres.SetOperator(A);
   gmres.SetPreconditioner(prec);
   x = 0.0;
   gmres.Mult(b, x);
   return gmres.GetNumIterations();
}

// ===========================================================================
//  SERIAL directional-mass operator.
// ===========================================================================
class DirectionalMass
{
   Vector beta_vec;
   VectorConstantCoefficient beta_cf;
   ConstantCoefficient one;

   DG_FECollection dg_fec;                 // broken L2 transport space
   FiniteElementSpace fes_m;

   FiniteElementSpace &fes_rho;            // H1 density space (caller-owned)

   BilinearForm Tform;                     // upwind transport, beta.grad
   std::unique_ptr<SparseMatrix> Tt;       // explicit transpose for the adjoint
   std::unique_ptr<BlockILU> precT, precTt;

   MixedBilinearForm Sform;                // H1 (rho~) -> DG (source) mass
   Vector cm;                              // DG volume weights: c_m . m = INT m

public:
   DirectionalMass(FiniteElementSpace &fes_rho_, Mesh &mesh,
                   int dg_order, const Vector &beta)
      : beta_vec(beta), beta_cf(beta_vec), one(1.0),
        dg_fec(dg_order, mesh.Dimension()),
        fes_m(&mesh, &dg_fec),
        fes_rho(fes_rho_),
        Tform(&fes_m), Sform(&fes_rho, &fes_m)
   {
      // T = INT (beta.grad m) w  + upwind face fluxes  (== -ex9 operator).
      Tform.AddDomainIntegrator(new ConvectionIntegrator(beta_cf, 1.0));
      Tform.AddInteriorFaceIntegrator(
         new NonconservativeDGTraceIntegrator(beta_cf, 1.0));
      Tform.AddBdrFaceIntegrator(
         new NonconservativeDGTraceIntegrator(beta_cf, 1.0));
      Tform.Assemble(0);
      Tform.Finalize(0);

      // S: (S rho~)_i = INT rho~ phi_i^DG  (test = DG, trial = H1 density).
      Sform.AddDomainIntegrator(new MixedScalarMassIntegrator());
      Sform.Assemble();
      Sform.Finalize();

      // DG volume weights c_m,i = INT phi_i^DG.
      LinearForm cmlf(&fes_m);
      cmlf.AddDomainIntegrator(new DomainLFIntegrator(one));
      cmlf.Assemble();
      cm.SetSize(fes_m.GetVSize());
      cm = cmlf;

      const int bs = fes_m.GetFE(0)->GetDof();
      Tt.reset(Transpose(Tform.SpMat()));
      precT  = std::make_unique<BlockILU>(Tform.SpMat(), bs);
      precTt = std::make_unique<BlockILU>(*Tt, bs);
   }

   FiniteElementSpace &TransportSpace() { return fes_m; }

   // Constant constraint sensitivity w = S^T T^{-T} c_m  (density dofs).
   // One adjoint transport solve; the caller filters w with M F^{-1}.
   int AdjointWeights(Vector &w)
   {
      Vector lambda(fes_m.GetVSize());
      GMRESSolver gmres;
      const int it = DGTransportSolve(gmres, *Tt, *precTt, cm, lambda);
      w.SetSize(fes_rho.GetVSize());
      Sform.MultTranspose(lambda, w);       // S^T lambda
      return it;
   }

   // Forward accumulation field m solving T m = S rho~ (for value / plotting).
   int Solve(const Vector &rho_tilde, Vector &m)
   {
      Vector src(fes_m.GetVSize());
      Sform.Mult(rho_tilde, src);           // S rho~
      m.SetSize(fes_m.GetVSize());
      GMRESSolver gmres;
      return DGTransportSolve(gmres, Tform.SpMat(), *precT, src, m);
   }

   // Total accumulated mass INT m dx = c_m . m.
   real_t Integral(const Vector &m) const { return cm * m; }

   // Directional mass of a (filtered) design without forming m, via the
   // precomputed weights:  INT m dx = w . rho~.
   real_t DirMass(const Vector &w, const Vector &rho_tilde) const
   { return w * rho_tilde; }
};

// ===========================================================================
//  PARALLEL (MPI) directional-mass operator -- same mathematics on local
//  true-dof chunks; mirrors ParDensityFilter / ParEllipticOperator.
// ===========================================================================
#ifdef MFEM_USE_MPI
class ParDirectionalMass
{
   MPI_Comm comm;
   Vector beta_vec;
   VectorConstantCoefficient beta_cf;
   ConstantCoefficient one;

   DG_FECollection dg_fec;
   ParFiniteElementSpace fes_m;
   ParFiniteElementSpace &fes_rho;

   ParBilinearForm Tform;
   OperatorHandle Th;                      // HypreParMatrix  (beta.grad, upwind)
   std::unique_ptr<HypreParMatrix> Tt;     // explicit transpose for the adjoint
   std::unique_ptr<BlockILU> precT, precTt;

   ParMixedBilinearForm Sform;
   OperatorHandle Sh;                       // HypreParMatrix  H1 -> DG
   Vector cm;                               // DG volume weights (true dofs)

public:
   ParDirectionalMass(ParFiniteElementSpace &fes_rho_, ParMesh &pmesh,
                      int dg_order, const Vector &beta)
      : comm(fes_rho_.GetComm()),
        beta_vec(beta), beta_cf(beta_vec), one(1.0),
        dg_fec(dg_order, pmesh.Dimension()),
        fes_m(&pmesh, &dg_fec),
        fes_rho(fes_rho_),
        Tform(&fes_m), Sform(&fes_rho, &fes_m)
   {
      Tform.AddDomainIntegrator(new ConvectionIntegrator(beta_cf, 1.0));
      Tform.AddInteriorFaceIntegrator(
         new NonconservativeDGTraceIntegrator(beta_cf, 1.0));
      Tform.AddBdrFaceIntegrator(
         new NonconservativeDGTraceIntegrator(beta_cf, 1.0));
      Tform.Assemble(0);
      Tform.Finalize(0);
      Th.Reset(Tform.ParallelAssemble());

      Sform.AddDomainIntegrator(new MixedScalarMassIntegrator());
      Sform.Assemble();
      Sform.Finalize();
      Sh.Reset(Sform.ParallelAssemble());

      ParLinearForm cmlf(&fes_m);
      cmlf.AddDomainIntegrator(new DomainLFIntegrator(one));
      cmlf.Assemble();
      std::unique_ptr<HypreParVector> cmtv(cmlf.ParallelAssemble());
      cm.SetSize(fes_m.GetTrueVSize());
      cm = *cmtv;

      const int bs = fes_m.GetTypicalFE()->GetDof();
      Tt.reset(Th.As<HypreParMatrix>()->Transpose());
      precT  = std::make_unique<BlockILU>(*Th.As<HypreParMatrix>(), bs);
      precTt = std::make_unique<BlockILU>(*Tt, bs);
   }

   ParFiniteElementSpace &TransportSpace() { return fes_m; }

   int AdjointWeights(Vector &w)
   {
      Vector lambda(fes_m.GetTrueVSize());
      GMRESSolver gmres(comm);
      const int it = DGTransportSolve(gmres, *Tt, *precTt, cm, lambda);
      w.SetSize(fes_rho.GetTrueVSize());
      Sh.As<HypreParMatrix>()->MultTranspose(lambda, w);
      return it;
   }

   int Solve(const Vector &rho_tilde, Vector &m)
   {
      Vector src(fes_m.GetTrueVSize());
      Sh.As<HypreParMatrix>()->Mult(rho_tilde, src);
      m.SetSize(fes_m.GetTrueVSize());
      GMRESSolver gmres(comm);
      return DGTransportSolve(gmres, *Th.As<HypreParMatrix>(), *precT, src, m);
   }

   real_t Integral(const Vector &m) const
   { return InnerProduct(comm, cm, m); }

   real_t DirMass(const Vector &w, const Vector &rho_tilde) const
   { return InnerProduct(comm, w, rho_tilde); }
};
#endif // MFEM_USE_MPI

#endif // DIRECTIONAL_MASS_HPP
