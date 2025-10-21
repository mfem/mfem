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
//          ----------------------------------------------------------
//               Bramble-Pasciak preconditioning for Darcy problem
//          ----------------------------------------------------------
//
// Main idea is to transform the block system
//                 Ax = [ M  B^T ] [u] = [f] = b
//                      [ B   0  ] [p] = [g]
//     where:
//        M = \int_\Omega (k u_h) \cdot v_h dx,
//        B = -\int_\Omega (div_h u_h) q_h dx,
//        u_h, v_h \in R_h (Raviart-Thomas finite element space),
//        q_h \in W_h (piecewise discontinuous polynomials),
// with a block transformation of the form X = A*N - Id
//                  X = [ M*invQ - Id    0   ]
//                      [     B*invQ    -Id  ]
// where N is defined by
//                  N = [ invQ    0 ]
//                      [   0     0 ]
// and Q is constructed such that Q and M-Q are both s.p.d..
//
// The solution x is then obtained by solving XAx = Xb with PCG as XA is s.p.d.
//
// The codes allows the user to provide such Q, or to construct it from the
// element matrices M_T. Moreover, the user can provide a block preconditioner
//                  P = [ M_0    0  ]
//                      [  0    M_1 ]
// for the transformed system XA.
//
// The code also allows the user to use BPCG, which is a special implementation
// of the PCG iteration with the particular preconditioner H, defined as
//                  H = [ M - Q    0  ]
//                      [  0      M_1 ]
// BPCG is efficient as it avoids the direct application of invH and X.

#ifndef MFEM_BP_SOLVER_HPP
#define MFEM_BP_SOLVER_HPP

#include "darcy_solver.hpp"
#include <memory>

namespace mfem::blocksolvers
{

/// Parameters for the BramblePasciakSolver method
struct BPSParameters : IterSolveParameters
{
   bool use_bpcg = true;   // whether to use BPCG
   real_t q_scaling = 0.5; // scaling (> 0 and < 1) of the Q preconditioner
};

/// Bramble-Pasciak Conjugate Gradient
class BPCGSolver : public IterativeSolver
{
protected:
   mutable Vector r, p, g, t, r_bar, r_red, g_red;
   const Operator *iprec, *pprec;
   void UpdateVectors();

public:
   BPCGSolver(const Operator *ipc, const Operator *ppc): iprec(ipc), pprec(ppc) {}

#ifdef MFEM_USE_MPI
   BPCGSolver(MPI_Comm comm_, const Operator *ipc, const Operator *ppc)
      : IterativeSolver(comm_), iprec(ipc), pprec(ppc) { }
#endif

   void SetOperator(const Operator &op) override
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   void SetPreconditioner(Solver &pc) override
   { if (Mpi::Root()) { MFEM_WARNING("SetPreconditioner has no effect on BPCGSolver.\n"); } }

   virtual void SetIncompletePreconditioner(const Operator *ipc) { iprec = ipc; }

   virtual void SetParticularPreconditioner(const Operator *ppc) { pprec = ppc; }

   void Mult(const Vector &b, Vector &x) const override;
};

/// Bramble-Pasciak Solver for Darcy equation.
/** Bramble-Pasciak Solver for Darcy equation.

    The basic idea is to precondition the mass matrix M with a s.p.d. matrix Q
    such that M - Q remains s.p.d. Then we can transform the block operator into
    a s.p.d. operator under a modified inner product. In particular, this enable
    us to implement modified versions of CG iterations, that rely on efficient
    applications of the required transformations.

    We offer a mass preconditioner based on a rescaling of the diagonal of the
    element mass matrices M_T.

    We consider Q_T := alpha * lambda_min * D_T, where D_T := diag(M_T), and
    lambda_min is the smallest eigenvalue of the following problem

                       M_T x = lambda * D_T x.

    Alpha is a parameter that is strictly between 0 and 1.

    For more details, see:

    1. P. Vassilevski, Multilevel Block Factorization Preconditioners (Appendix
       F.3), Springer, 2008.

    2. J. Bramble and J. Pasciak. A Preconditioning Technique for Indefinite
       Systems Resulting From Mixed Approximations of Elliptic Problems,
       Mathematics of Computation, 50:1-17, 1988. */
class BramblePasciakSolver : public DarcySolver
{
   mutable bool use_bpcg;
   std::unique_ptr<IterativeSolver> solver_;
   std::unique_ptr<BlockOperator> oop_, ipc_;
   std::unique_ptr<ProductOperator> mop_, ppc_;
   std::unique_ptr<SumOperator> map_;
   std::unique_ptr<BlockDiagonalPreconditioner> cpc_;
   std::unique_ptr<HypreParMatrix> M_, B_, Q_, S_;
   std::unique_ptr<TransposeOperator> Bt_;
   OperatorPtr M0_, M1_;
   Array<int> ess_zero_dofs_;

   void Init(HypreParMatrix &M, HypreParMatrix &B,
             HypreParMatrix &Q,
             Solver &M0, Solver &M1,
             const BPSParameters &param);
public:
   /// System and mass preconditioner are constructed from bilinear forms
   BramblePasciakSolver(
      ParBilinearForm &mVarf,
      ParMixedBilinearForm &bVarf,
      const BPSParameters &param);

   /// System and mass preconditioner are user-provided
   BramblePasciakSolver(
      HypreParMatrix &M, HypreParMatrix &B, HypreParMatrix &Q,
      Solver &M0, Solver &M1,
      const BPSParameters &param);

   /// Assemble a preconditioner for the mass matrix
   /** Mass preconditioner corresponds to a local re-scaling based on the
       smallest eigenvalue of the generalized eigenvalue problem locally on each
       element T:
                                M_T x_T = lambda_T diag(M_T) x_T.
       We set Q_T = alpha * min(lambda_T) * diag(M_T), 0 < alpha < 1. */
   static HypreParMatrix *ConstructMassPreconditioner(const ParBilinearForm &mVarf,
                                                      const real_t alpha = 0.5);

   void Mult(const Vector &x, Vector &y) const override;
   void SetOperator(const Operator &op) override { }
   void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
   int GetNumIterations() const override { return solver_->GetNumIterations(); }
};

} // namespace mfem::blocksolvers

#endif // MFEM_BP_SOLVER_HPP
