// Copyright (c) 2023-2023, Lawrence Livermore National Security, LLC. Produced
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
// Main idea is to precondition the block system
//                 Ax = [ M  B^T ] [u] = [f]
//                      [ B   0  ] [p] = [g]
//     where:
//        M = \int_\Omega (k u_h) \cdot v_h dx,
//        B = -\int_\Omega (div_h u_h) q_h dx,
//        f = \int_\Omega f_exact v_h dx + \int_D natural_bc v_h dS,
//        g = \int_\Omega g_exact q_h dx,
//        u_h, v_h \in R_h (Raviart-Thomas finite element space),
//        q_h \in W_h (piecewise discontinuous polynomials),
//        D: subset of the boundary where natural boundary condition is imposed.
// with a block transformation of the form X = AN - Id
//                  X = [ A*invQ - Id    0   ]
//                      [     B*invQ    -Id  ]
// where N is defined by
//                  N = [ invQ    0 ]
//                      [   0     0 ]
// and Q is constructed such that Q and M-Q are both s.p.d.
//
// The codes allows the user to provide such Q, or to construct it from the
// element matrices A_T. Moreover, the user can provide a block preconditioner
//                  P = [ M_1    0  ]
//                      [  0    M_2 ]
// Using the particular preconditioner H, defined as
//                  H = [ A - Q    0  ]
//                      [  0      M_2 ]
// (where M_1 = Q), enables a simplified version of a CG iteration (BPCG), as it avoids
// the direct application of invH and X.
//
// The code allows to use (P)CG with P or H, and BPCG.

#ifndef MFEM_BP_SOLVER_HPP
#define MFEM_BP_SOLVER_HPP

#include "darcy_solver.hpp"
#include <memory>

namespace mfem
{
namespace blocksolvers
{

/// Parameters for the BramblePasciakSolver method
struct BPSParameters : IterSolveParameters
{
   /* These are parameters for the scaling of the Q preconditioner
    * the usage of BPCG method, and the definition of the H preconditioner */
   bool use_bpcg = true;
   double q_scaling = 0.5;
};

/// Bramble-Pasciak Conjugate Gradient
class BPCGSolver : public IterativeSolver
{
protected:
   mutable Vector r, p, g, t, r_bar, r_red, g_red;
   /// Remaining required operators
   /*  Operator list
    *  From IterativeSolver:
    *  *oper  -> A  = [M, Bt; B, 0]
    *  *prec  -> P  = diag(M0, M1) // Not used
    *  From this class:
    *  *iprec -> N  = diag(M0, 0)
    *  *pprec -> P' = P * [Id, 0; B*M0, -Id]
    */
   const Operator *iprec, *pprec;
   void UpdateVectors();

public:
   BPCGSolver() { }
   BPCGSolver(const Operator &ipc, const Operator &ppc) { pprec = &ppc; iprec = &ipc; }

#ifdef MFEM_USE_MPI
   BPCGSolver(MPI_Comm comm_) : IterativeSolver(comm_) { }
   BPCGSolver(MPI_Comm comm_, const Operator &ipc,
              const Operator &ppc) : IterativeSolver(comm_) { pprec = &ppc; iprec = &ipc; }
#endif

   virtual void SetOperator(const Operator &op)
   { IterativeSolver::SetOperator(op); UpdateVectors(); }

   virtual void SetPreconditioner(const Operator &pc)
   { if (Mpi::Root()) { MFEM_WARNING("No explicit preconditioner required for BPCG.\n"); } }

   virtual void SetIncompletePreconditioner(const Operator &ipc)
   { iprec = &ipc; }

   virtual void SetParticularPreconditioner(const Operator &ppc)
   { pprec = &ppc; }

   virtual void Mult(const Vector &b, Vector &x) const;
};

/// Bramble-Pasciak Solver for Darcy equation.
/** Bramble-Pasciak Solver for Darcy equation.
 *  The basic idea is to precondition the mass matrix M with a s.p.d. matrix Q
 *  such that M - Q remains s.p.d. Then we can transform the block operator into a
 *  s.p.d. operator under a modified inner product.
 *  In particular, this enable us to implement modified versions of CG iterations,
 *  that rely on efficient applications of the required transformations.
 *
 *  We offer a mass preconditioner based on a rescalling of the diagonal of the
 *  element mass matrices M_T.
 *  We consider Q_T := alpha * lambda_min * D_T, where D_T := diag(M_T), and
 *  lambda_min is the smallest eigenvalue of the following problem
 *                M_T x = lambda * D_T x.
 *  alpha is a parameter that is stricly between 0 and 1.
 *
 *  For more details, see:
 *  1. Vassilevski, Multilevel Block Factorization Preconditioners (Appendix F.3),
 *     Springer, 2008.
 *  2. James H. Bramble and Joseph E. Pasciak.
 *     A Preconditioning Technique for Indefinite Systems Resulting From Mixed
 *     Approximations of Elliptic Problems. Mathematics of Computation, 50:1–17, 1988.
 */
class BramblePasciakSolver : public DarcySolver
{
   mutable bool use_bpcg;
   std::unique_ptr<IterativeSolver> solver_;
   BlockOperator *oop_, *ipc_;
   ProductOperator *mop_;
   SumOperator *map_;
   ProductOperator *ppc_;
   BlockDiagonalPreconditioner *cpc_, *hpc_;
   std::unique_ptr<HypreParMatrix> M_;
   std::unique_ptr<HypreParMatrix> B_;
   std::unique_ptr<HypreParMatrix> Q_;
   OperatorPtr M0_;
   OperatorPtr M1_;
   Array<int> ess_zero_dofs_;

   void Init(HypreParMatrix &M, HypreParMatrix &B,
             HypreParMatrix &Q,
             Solver &M0, Solver &M1,
             const BPSParameters &param);
public:
   /// System and mass preconditioner are constructed from bilinear forms
   BramblePasciakSolver(
      const std::shared_ptr<ParBilinearForm> &mVarf,
      const std::shared_ptr<ParMixedBilinearForm> &bVarf,
      const BPSParameters &param);

   /// System and mass preconditioner are user-provided
   BramblePasciakSolver(
      HypreParMatrix &M, HypreParMatrix &B, HypreParMatrix &Q,
      Solver &M0, Solver &M1,
      const BPSParameters &param);

   /// Assemble a preconditioner for the mass matrix
   /** Mass preconditioner corresponds to a local re-scaling
    * based on the smallest eigenvalue of the generalized
    * eigenvalue problem locally on each element T:
    *         M_T x_T = lambda_T diag(M_T) x_T
    * and we set Q_T = 0.5 * min(lambda_T) * diag(M_T).
   */
   static HypreParMatrix *ConstructMassPreconditioner(ParBilinearForm &mVarf,
                                                      double alpha = 0.5);

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void SetOperator(const Operator &op) { }
   void SetEssZeroDofs(const Array<int>& dofs) { dofs.Copy(ess_zero_dofs_); }
   virtual int GetNumIterations() const { return solver_->GetNumIterations(); }
};

} // namespace blocksolvers
} // namespace mfem

#endif // MFEM_BP_SOLVER_HPP
