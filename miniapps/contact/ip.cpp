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

#include "ip.hpp"
#include "general/reducers.hpp"
#include "solver_utils.hpp"

namespace mfem
{

namespace
{

void EnableDevice(BlockVector &v)
{
   v.UseDevice(true);
   for (int i = 0; i < v.NumBlocks(); i++)
   {
      v.GetBlock(i).UseDevice(true);
   }
}

Array<real_t> &IPReduceWorkspace()
{
   static Array<real_t> workspace;
   return workspace;
}

}

IPSolver::IPSolver(OptContactProblem * problem_)
   : problem(problem_)
{
   abs_tol  = 1.e-2; // Tolerance of the optimizer
   max_iter = 20;   // Maximum iterations
   mu_k     = 1.0;  // Log-barrier penalty parameter

   /* The following constants follow that of
    * Wächter, Andreas, and Lorenz T. Biegler.
    * "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming."
    * Mathematical programming 106.1 (2006): 25-57.
    */

   /* line search constants */
   tauMin   =
      0.99;  // constant that controls the rate iterates approach boundary > 0, < 1
   eta      =
      1.e-4; // Armijo backtracking sufficient-decrease condition constant > 0, < 1
   thetaMin = 1.e-4; // allowed equality constraint violation
   delta    = 1.0;   // sufficient barrier objective progress constant > 0
   sTheta   = 1.1;   // sufficient barrier objective progress constant > 0
   sPhi     = 2.3;   // sufficient barrier objective progress constant >= 0
   gTheta = 1.e-5;   // sufficient constraint violation decrease constant > 0, < 1
   gPhi   = 1.e-5;   // sufficient barrier objective decrease constant > 0, < 1
   thetaMax = 1.e6; // maximum constraint violation (defines initial filter)

   /* interior-point continuation constants
    * \mu_new = max{abs_tol / 10, \min{ kMu * \mu_old, (\mu_old)^thetaMu}}
    * */
   kMu     = 0.2;
   thetaMu = 1.5;
   kEps   = 1.e1; // constant that determines when the log-barrier parameter is decreased

   kSig     = 1.e10; // primal-dual Hessian deviation constant

   /* The following constants follow that of
    * Chiang, Nai-Yuan, and Victor M. Zavala.
    * "An inertia-free filter line-search algorithm for large-scale nonlinear programming."
    * Computational Optimization and Applications 64.2 (2016): 327-354.
    * See Inertia-Free Regularizaton Algorithm (IFR).
    * We use the second variant (see Section 4.1 "Alternative Inertia-Free Tests")
    * which does not require solving for normal and tangential components.
    */

   /* inertia-regularization constants */
   alphaCurvatureTest = 1.e-11;
   deltaRegLast = 0.0;
   deltaRegMin = 1.e-20;
   deltaRegMax = 1.e40;
   deltaReg0 = 1.e-4;
   kRegMinus = 1. / 3.;
   kRegBarPlus = 1.e2;
   kRegPlus = 8.;

   dimU = problem->GetDimU();
   dimM = problem->GetDimM();
   dimC = problem->GetDimC();
   MFEM_VERIFY(dimM == dimC,
               "Expecting equal numbers of equality and inequality constraints");

   comm = problem->GetComm();
   Mcslump.UseDevice(true);
   Mvlump.UseDevice(true);
   problem->GetLumpedMassWeights(Mcslump, Mvlump);

   block_offsetsumlz.SetSize(5);
   block_offsetsuml.SetSize(4);
   block_offsetsx.SetSize(3);

   block_offsetsumlz[0] = 0;
   block_offsetsumlz[1] = dimU; // u
   block_offsetsumlz[2] = dimM; // m
   block_offsetsumlz[3] = dimC; // lambda
   block_offsetsumlz[4] = dimM; // zl
   block_offsetsumlz.PartialSum();

   Mlump.SetSize(dimM); Mlump.UseDevice(true); Mlump = 0.0;
   int dimG = 0; // number of gap constraints
   if (dimM < dimU)
   {
      dimG = dimM;
      constraint_offsets.SetSize(2);
      constraint_offsets[0] = 0;
      constraint_offsets[1] = dimM;
      Mlump.Set(1.0, Mcslump);
   }
   else
   {
      dimG = dimM - 2 * dimU;
      constraint_offsets.SetSize(4);
      constraint_offsets[0] = 0;
      constraint_offsets[1] = dimG;
      constraint_offsets[2] = dimU;
      constraint_offsets[3] = dimU;
      constraint_offsets.PartialSum();
      BlockVector Mlumpblock(constraint_offsets); EnableDevice(Mlumpblock);
      Mlumpblock = 0.0;
      Mlumpblock.GetBlock(0).Set(1.0, Mcslump);
      Mlumpblock.GetBlock(1).Set(1.0, Mvlump);
      Mlumpblock.GetBlock(2).Set(1.0, Mvlump);
      Mlumpblock.SyncFromBlocks();
      Mlump.Set(1.0, Mlumpblock);
   }

   for (int i = 0; i < block_offsetsuml.Size(); i++)
   {
      block_offsetsuml[i] = block_offsetsumlz[i];
   }
   for (int i = 0; i < block_offsetsx.Size(); i++)
   {
      block_offsetsx[i] = block_offsetsuml[i] ;
   }

   lk.SetSize(dimC);  lk.UseDevice(true);  lk  = 0.0;
   zlk.SetSize(dimM); zlk.UseDevice(true); zlk = 0.0;

   MPI_Comm_rank(comm, &myid);
}

void IPSolver::Mult(const Vector &x0, Vector &xf)
{
   BlockVector x0block(block_offsetsx); EnableDevice(x0block); x0block = 0.0;
   x0block.GetBlock(0).Set(1.0, x0);
   x0block.GetBlock(1) = 1.0;
   x0block.SyncFromBlocks();
   BlockVector xfblock(block_offsetsx); EnableDevice(xfblock); xfblock = 0.0;
   Mult(x0block, xfblock);
   xf.Set(1.0, xfblock.GetBlock(0));
}

void IPSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   converged = false;
   x0.SyncToBlocks();
   BlockVector xk(block_offsetsx), xhat(block_offsetsx);
   EnableDevice(xk); EnableDevice(xhat); xk = 0; xhat = 0.0;
   BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz);
   EnableDevice(Xk); EnableDevice(Xhat); Xk = 0.0;
   Xhat = 0.0;
   BlockVector Xhatuml(block_offsetsuml); EnableDevice(Xhatuml); Xhatuml = 0.0;
   Vector zlhat(dimM); zlhat.UseDevice(true); zlhat = 0.0;

   xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
   xk.GetBlock(1).Set(1.0, x0.GetBlock(1));
   // running estimate of the final values of the Lagrange multipliers
   lk  = 0.0;
   zlk = 0.0;

   xk.SyncFromBlocks();
   const int dim_u = dimU;
   const real_t mu = mu_k;
   const real_t *xk_d = xk.Read();
   real_t *zlk_d = zlk.Write();
   mfem::forall(dimM, [=] MFEM_HOST_DEVICE (int i)
   {
      zlk_d[i] = 1.e1 * mu / xk_d[i + dim_u];
   });

   Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
   Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
   Xk.GetBlock(2).Set(1.0, lk);
   Xk.GetBlock(3).Set(1.0, zlk);

   /* set theta0 = theta(x0)
    *     thetaMin
    *     thetaMax
    * when theta(xk) < thetaMin and the switching condition holds
    * then we ask for the Armijo sufficient decrease of the barrier
    * objective to be satisfied, in order to accept the trial step length alphakl
    *
    * thetaMax controls how the filter is initialized for each log-barrier subproblem
    * F0 = {(th, phi) s.t. th > thetaMax}
    * that is the filter does not allow for iterates where the constraint violation
    * is larger than that of thetaMax
    */
   real_t theta0 = GetTheta(xk);
   thetaMin = 1.e-4 * std::max(1.0, theta0);
   thetaMax = 1.e8  * thetaMin;

   real_t OptErrSubproblem, OptErr;

   int maxBarrierSolves = 10;
   for (int j = 0; j < max_iter; j++)
   {
      if (myid == 0 && print_level > 0)
      {
         mfem::out << "\n" << std::string(50,'-') << std::endl;
         mfem::out << "interior-point solve step # " << j << std::endl;
      }
      // Check convergence of optimization problem
      OptErr = OptimalityError(xk, lk, zlk);
      if (OptErr < abs_tol)
      {
         converged = true;
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "solved optimization problem\n";
         }
         break;
      }

      if (j > 0) { maxBarrierSolves = 1; }
      for (int i = 0; i < maxBarrierSolves; i++)
      {
         // Check convergence of the barrier subproblem
         OptErrSubproblem = OptimalityError(xk, lk, zlk, mu_k);
         if (OptErrSubproblem < kEps * mu_k)
         {
            if (myid == 0 && print_level > 0)
            {
               mfem::out << "solved mu = " << mu_k << " barrier subproblem\n";
            }
            UpdateBarrierSubProblem();
         }
         else
         {
            break;
         }
      }

      // Compute the search direction
      // solve for (uhat, mhat, lhat)
      zlhat = 0.0; Xhatuml = 0.0;

      bool passedCurvatureTest = false;
      IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, passedCurvatureTest, mu_k);
      if (!passedCurvatureTest)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "curvature test failed\n";
         }
         real_t deltaReg = 0.0;
         int maxCurvatureTests = 30;

         // inertia regularization initialization
         if (deltaRegLast < deltaRegMin)
         {
            deltaReg = deltaReg0;
         }
         else
         {
            deltaReg = fmax(deltaRegMin, kRegMinus * deltaRegLast);
         }

         // solve regularized IP-Newton linear system
         zlhat = 0.0; Xhatuml = 0.0;
         IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, passedCurvatureTest, mu_k, deltaReg);

         for (int numCurvatureTests = 0; numCurvatureTests < maxCurvatureTests;
              numCurvatureTests++)
         {
            if (myid == 0 && print_level > 0)
            {
               mfem::out << "deltaReg = " << deltaReg << std::endl;
            }
            if (passedCurvatureTest)
            {
               deltaRegLast = deltaReg;
               break;
            }
            else
            {
               if (deltaRegLast < deltaRegMin)
               {
                  if (myid == 0 && print_level > 0)
                  {
                     mfem::out << "delta *= " << kRegBarPlus << "\n";
                  }
                  deltaReg *= kRegBarPlus;
               }
               else
               {
                  deltaReg *= kRegPlus;
               }
            }
            // solve with regularization
            zlhat = 0.0; Xhatuml = 0.0;
            IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, passedCurvatureTest, mu_k, deltaReg);
         }
      }

      Xk = 0.0;
      Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
      Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
      Xk.GetBlock(2).Set(1.0, lk);
      Xk.GetBlock(3).Set(1.0, zlk);
      Xk.SyncFromBlocks();

      Xhat = 0.0;
      Xhat.SyncToBlocks();
      for (int i = 0; i < 3; i++)
      {
         Xhat.GetBlock(i).Set(1.0, Xhatuml.GetBlock(i));
      }
      Xhat.GetBlock(3).Set(1.0, zlhat);
      Xhat.SyncFromBlocks();

      if (myid == 0 && print_level > 0)
      {
         mfem::out << "mu = " << mu_k << std::endl;
      }
      LineSearch(Xk, Xhat, mu_k);

      if (lineSearchSuccess)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "lineSearch successful\n";
         }
         if (!switchCondition || !sufficientDecrease)
         {
            F1.Append( (1. - gTheta) * thx0);
            F2.Append( phx0 - gPhi * thx0);
         }
         // Accept the trial point
         xk.GetBlock(0).Add(alpha, Xhat.GetBlock(0));
         xk.GetBlock(1).Add(alpha, Xhat.GetBlock(1));
         lk.Add(alpha,   Xhat.GetBlock(2));
         zlk.Add(alphaz, Xhat.GetBlock(3));
         ProjectZ(xk, zlk, mu_k);
         xk.SyncFromBlocks();
      }
      else
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "lineSearch not successful\n";
         }
         converged = false;
         break;
      }
      if (j + 1 == max_iter && myid == 0 && print_level > 0)
      {
         mfem::out << "maximum optimization iterations\n";
      }
   }
   xf = 0.0;
   xf.SyncToBlocks();
   xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
   xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
   xf.SyncFromBlocks();
}

void IPSolver::FormIPNewtonMat(BlockVector & x, Vector & l,
                               Vector &zl,
                               BlockOperator &Ak, real_t delta)
{
   x.SyncFromBlocks();
   Huu = problem->Duuf(x);
   Hmm = problem->Dmmf(x);

   delete JuT;
   delete JmT;
   Ju = problem->Duc(x); JuT = Ju->Transpose();
   Jm = problem->Dmc(x); JmT = Jm->Transpose();

   Vector DiagLogBar(dimM); DiagLogBar.UseDevice(true); DiagLogBar = 0.0;
   const int dim_u = dimU;
   const real_t *Mlump_d = Mlump.Read();
   const real_t *zl_d = zl.Read();
   const real_t *x_d = x.Read();
   real_t *DiagLogBar_d = DiagLogBar.Write();
   mfem::forall(dimM, [=] MFEM_HOST_DEVICE (int i)
   {
      DiagLogBar_d[i] = (Mlump_d[i] * zl_d[i]) / x_d[i + dim_u]
                        + delta * Mlump_d[i];
   });

   delete Wmm;
   if (Hmm)
   {
      HypreParMatrix *D =
         MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumConstraints(),
                                    problem->GetConstraintsStarts(),
                                    DiagLogBar);
      Wmm = ParAdd(Hmm,D);
      delete D;
   }
   else
   {
      Wmm = MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumConstraints(),
                                       problem->GetConstraintsStarts(),
                                       DiagLogBar);
   }

   Vector deltaDiagVec(dimU); deltaDiagVec.UseDevice(true);
   deltaDiagVec = delta;
   deltaDiagVec *= Mvlump;

   delete Wuu;
   if (Huu)
   {
      HypreParMatrix *Duu =
         MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumDofs(),
                                    problem->GetDofStarts(), deltaDiagVec);
      Wuu = ParAdd(Huu, Duu);
      delete Duu;
   }
   else
   {
      Wuu = MakeDiagonalHypreParMatrix(comm, problem->GetGlobalNumDofs(),
                                       problem->GetDofStarts(), deltaDiagVec);
   }

   //         IP-Newton system matrix
   //    Ak = [[W_(u,u)  H_(u,m)   J_u^T]
   //          [H_(m,u)  W_(m,m)   J_m^T]
   //          [ J_u      J_m       0  ]]

   //    Ak = [[K+Dreg  0      Jᵀ ]   [u]    [bᵤ]
   //          [0      D+Dreg  -I ]   [m]  = [bₘ]
   //          [J      -I      0  ]]  [λ]  = [bₗ ]
   Ak.SetBlock(0, 0, Wuu);                         Ak.SetBlock(0, 2, JuT);
   Ak.SetBlock(1, 1, Wmm); Ak.SetBlock(1, 2, JmT);
   Ak.SetBlock(2, 0,  Ju); Ak.SetBlock(2, 1,  Jm);
}

// perturbed KKT system solve
// determine the search direction
void IPSolver::IPNewtonSolve(BlockVector &x, Vector &l,
                             Vector &zl, Vector &zlhat, BlockVector &Xhat, bool & passedCurvatureTest,
                             real_t mu,
                             real_t delta)
{
   iter++;
   // solve A x = b, where A is the IP-Newton matrix
   BlockOperator A(block_offsetsuml, block_offsetsuml);
   BlockVector b(block_offsetsuml); EnableDevice(b); b = 0.0;
   FormIPNewtonMat(x, l, zl, A, delta);

   //       [grad_u phi + Ju^T l]
   // b = - [grad_m phi + Jm^T l]
   //       [          c        ]
   BlockVector gradphi(block_offsetsx); EnableDevice(gradphi); gradphi = 0.0;
   GetDxphi(x, mu, gradphi);

   for (int i = 0; i < 2; i++)
   {
      b.GetBlock(i).Set(1.0, gradphi.GetBlock(i));
      A.GetBlock(i, 2).AddMult(l, b.GetBlock(i));
   }
   problem->c(x, b.GetBlock(2));
   b.SyncFromBlocks();
   b *= -1.0;
   b.SyncToBlocks();
   Xhat = 0.0;
   Xhat.SyncToBlocks();

   // form A = Huu + Ju^T D Ju, Wmm = D for contact
   HypreParMatrix *JuTDJu   = RAP(Wmm, Ju);     // Ju^T D Ju
   HypreParMatrix *Areduced = ParAdd(Huu, JuTDJu);  // Huu + Ju^T D Ju

   /* compute the reduced rhs */
   // breduced = bu + Ju^T (bm + Wmm bl)
   Vector breduced(dimU); breduced.UseDevice(true); breduced = 0.0;
   Vector tempVec(dimM); tempVec.UseDevice(true); tempVec = 0.0;
   Wmm->Mult(b.GetBlock(2), tempVec);
   tempVec.Add(1.0, b.GetBlock(1));
   JuT->Mult(tempVec, breduced);
   breduced.Add(1.0, b.GetBlock(0));

   // Solve the system with the given solver
   solver->SetOperator(*Areduced);
   solver->Mult(breduced, Xhat.GetBlock(0));
   auto itsolver = dynamic_cast<IterativeSolver *>(solver);
   int numit = (itsolver) ? itsolver->GetNumIterations() : -1;
   lin_solver_iterations.Append(numit);

   // now propagate solved uhat to obtain mhat and lhat
   // xm = Ju xu - bl
   Ju->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
   Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

   // xl = Wmm xm - bm
   Wmm->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
   Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));
   Xhat.SyncFromBlocks();

   delete JuTDJu;
   delete Areduced;

   passedCurvatureTest = CurvatureTest(A, Xhat, l, b, delta);

   /* backsolve to determine zlhat */
   x.SyncFromBlocks();
   const int dim_u = dimU;
   const real_t *zl_d = zl.Read();
   const real_t *Xhat_d = Xhat.Read();
   const real_t *x_d = x.Read();
   real_t *zlhat_d = zlhat.Write();
   mfem::forall(dimM, [=] MFEM_HOST_DEVICE (int i)
   {
      zlhat_d[i] = zl_d[i]
                   + (zl_d[i] * Xhat_d[i + dim_u] - mu) / x_d[i + dim_u];
   });
   zlhat *= -1.;
}

real_t IPSolver::GetMaxStepSize(Vector &x, Vector &xhat,
                                real_t tau)
{
   real_t alphaMaxloc = 1.0;
   const bool use_dev = x.UseDevice() || xhat.UseDevice();
   const real_t *x_d = x.Read(use_dev);
   const real_t *xhat_d = xhat.Read(use_dev);
   reduce(x.Size(), alphaMaxloc,
          [=] MFEM_HOST_DEVICE (int i, real_t &alpha)
   {
      if (xhat_d[i] < 0.0)
      {
         alpha = fmin(alpha, -tau * x_d[i] / xhat_d[i]);
      }
   },
   MinReducer<real_t> {}, use_dev, IPReduceWorkspace());

   real_t alphaMaxglb;
   MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MIN, comm);
   return alphaMaxglb;
}

/* line-search from X0 along direction Xhat for the log-barrier
 * subproblem with barrier parameter \mu
 */
void IPSolver::LineSearch(BlockVector& X0, BlockVector& Xhat,
                          real_t mu)
{
   int eval_err = 0;
   real_t tau  = std::max(tauMin, 1.0 - mu);
   Vector u0   = X0.GetBlock(0);
   Vector m0   = X0.GetBlock(1);
   Vector l0   = X0.GetBlock(2);
   Vector z0   = X0.GetBlock(3);
   Vector uhat = Xhat.GetBlock(0);
   Vector mhat = Xhat.GetBlock(1);
   Vector lhat = Xhat.GetBlock(2);
   Vector zhat = Xhat.GetBlock(3);
   real_t alphaMax  = GetMaxStepSize(m0, mhat, tau);
   real_t alphaMaxz = GetMaxStepSize(z0, zhat, tau);
   alphaz = alphaMaxz;

   BlockVector x0(block_offsetsx); EnableDevice(x0); x0 = 0.0;
   x0.GetBlock(0).Set(1.0, u0);
   x0.GetBlock(1).Set(1.0, m0);
   x0.SyncFromBlocks();

   BlockVector xhat(block_offsetsx); EnableDevice(xhat); xhat = 0.0;
   xhat.GetBlock(0).Set(1.0, uhat);
   xhat.GetBlock(1).Set(1.0, mhat);
   xhat.SyncFromBlocks();

   BlockVector xtrial(block_offsetsx); EnableDevice(xtrial); xtrial = 0.0;
   BlockVector Dxphi0(block_offsetsx); EnableDevice(Dxphi0); Dxphi0 = 0.0;
   int maxBacktrack = 20;
   alpha = alphaMax;

   GetDxphi(x0, mu, Dxphi0);

   Dxphi0.SyncFromBlocks();
   real_t Dxphi0_xhat = InnerProduct(comm, Dxphi0, xhat);
   bool descentDirection = (Dxphi0_xhat < 0.);


   if (myid == 0 && print_level > 0)
   {
      mfem::out << "is";
      if (!descentDirection)
      {
         mfem::out << " not";
      }
      mfem::out << " a descent direction for the log-barrier objective\n";
   }

   thx0 = GetTheta(x0);
   phx0 = GetPhi(x0, mu);

   lineSearchSuccess = false;
   for (int i = 0; i < maxBacktrack; i++)
   {
      if (myid == 0 && print_level > 0)
      {
         mfem::out << "\n--------- alpha = " << alpha << " ---------\n";
      }
      // ----- Compute trial point: xtrial = x0 + alpha * xhat
      xtrial.Set(1.0, x0);
      xtrial.Add(alpha, xhat);
      xtrial.SyncToBlocks();

      real_t thxtrial = GetTheta(xtrial);
      real_t phxtrial = GetPhi(xtrial, mu, eval_err);
      if (eval_err == 1)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "bad log-barrier objective eval, reducing step length\n";
         }
         alpha *= 0.5;
         continue;
      }

      auto inFilterRegion = FilterCheck(thxtrial, phxtrial);
      if (!inFilterRegion)
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "not in filter region\n";
         }
         if (!descentDirection)
         {
            switchCondition = false;
         }
         else
         {
            switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0,
                                                                                 sTheta));
         }
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "theta(x0) = "     << thx0     << ", thetaMin = "
                      <<
                      thetaMin             << std::endl;
            mfem::out << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = "
                      <<
                      (1. - gTheta) * thx0 << std::endl;
            mfem::out << "phi(xtrial) = "   << phxtrial << ", phi(x0) - gPhi *theta(x0) = "
                      <<
                      phx0 - gPhi * thx0   << std::endl;
         }
         if (thx0 <= thetaMin && switchCondition)
         {
            // sufficient decrease of the log-barrier objective
            sufficientDecrease = (phxtrial <= phx0 + eta * alpha * Dxphi0_xhat);
            if (sufficientDecrease)
            {
               if (myid == 0 && print_level > 0)
               {
                  mfem::out <<
                            "Line search successful: sufficient decrease in log-barrier objective.\n";
               }
               // accept the trial step
               lineSearchSuccess = true;
               break;
            }
         }
         else
         {
            if (thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0)
            {
               if (myid == 0 && print_level > 0)
               {
                  mfem::out <<
                            "Line search successful: infeasibility or log-barrier objective decreased.\n";
               }
               // accept the trial step
               lineSearchSuccess = true;
               break;
            }
         }
      }
      else
      {
         if (myid == 0 && print_level > 0)
         {
            mfem::out << "in filter region\n";
         }
      }
      alpha *= 0.5;
   }
}

bool IPSolver::FilterCheck(real_t thetax, real_t phix)
{
   bool inFilterRegion = false;
   if (thetax > thetaMax)
   {
      inFilterRegion = true;
   }
   else
   {
      for (int i = 0; i < F1.Size(); i++)
      {
         if (thetax >= F1[i] && phix >= F2[i])
         {
            inFilterRegion = true;
            break;
         }
      }
   }
   return inFilterRegion;
}

void IPSolver::ProjectZ(const Vector &x, Vector &z, real_t mu)
{
   const int dim_u = dimU;
   const real_t k_sig = kSig;
   const real_t *x_d = x.Read();
   real_t *z_d = z.ReadWrite();
   mfem::forall(dimM, [=] MFEM_HOST_DEVICE (int i)
   {
      const real_t zdual_i = z_d[i];
      const real_t zprimal_i = mu / x_d[i + dim_u];
      z_d[i] = fmax(fmin(zdual_i, k_sig * zprimal_i), zprimal_i / k_sig);
   });
}

real_t IPSolver::GetTheta(const BlockVector &x)
{
   x.SyncToBlocks();
   Vector cx(dimC); cx.UseDevice(true);
   problem->c(x, cx);
   Vector Mcx(dimC); Mcx.UseDevice(true);
   Mcx.Set(1.0, cx);
   Mcx *= Mlump;
   return sqrt(InnerProduct(comm, Mcx, cx));
}

// log-barrier objective
real_t IPSolver::GetPhi(const BlockVector &x, real_t mu,
                        int eval_err)
{
   x.SyncToBlocks();
   real_t fx = problem->CalcObjective(x, eval_err);
   real_t logBarrierLoc = 0.0;
   const int dim_u = dimU;
   const bool use_dev = x.UseDevice() || Mlump.UseDevice();
   const real_t *Mlump_d = Mlump.Read(use_dev);
   const real_t *x_d = x.Read(use_dev);
   reduce(dimM, logBarrierLoc,
          [=] MFEM_HOST_DEVICE (int i, real_t &sum)
   {
      sum += Mlump_d[i] * log(x_d[dim_u + i]);
   },
   SumReducer<real_t> {}, use_dev, IPReduceWorkspace());
   real_t logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   return fx - mu * logBarrierGlb;
}

// gradient of log-barrier objective with respect to x = (u, m)
void IPSolver::GetDxphi(const BlockVector &x, real_t mu,
                        BlockVector &y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   problem->CalcObjectiveGrad(x, y);
   Vector ytemp(dimM); ytemp.UseDevice(true); ytemp = 1.0;
   ytemp /= x.GetBlock(1);
   ytemp *= Mlump;
   y.GetBlock(1).Add(-mu, ytemp);
   y.SyncFromBlocks();
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
real_t IPSolver::EvalLagrangian(const BlockVector &x, const Vector &l,
                                const Vector &zl)
{
   x.SyncToBlocks();
   int eval_err = 0;
   real_t fx = problem->CalcObjective(x, eval_err);
   Vector cx(dimC); cx.UseDevice(true); problem->c(x, cx);
   Vector temp(dimM); temp.UseDevice(true); temp = 0.0;
   temp.Set(1.0, x.GetBlock(1));
   temp *= Mlump;
   return (fx + InnerProduct(comm, cx, l) - InnerProduct(comm, temp, zl));
}

// Gradient of the Lagrangian
// \nabla_x L = [ \nabla_u f + (\partial c / \partial u)^T l]
//              [ \nabla_m f + (\partial c / \partial m)^T l - zl]
void IPSolver::EvalLagrangianGradient(const BlockVector &x, const Vector &l,
                                      const Vector &zl, BlockVector &y)
{
   x.SyncToBlocks();
   y.SyncToBlocks();
   // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
   y.GetBlock(1).Set(-1.0, zl);
   y.GetBlock(1) *= Mlump;

   problem->Duc(x)->MultTranspose(l, y.GetBlock(0));
   problem->Dmc(x)->AddMultTranspose(l, y.GetBlock(1));

   BlockVector gradxf(block_offsetsx); EnableDevice(gradxf); gradxf = 0.0;
   problem->CalcObjectiveGrad(x, gradxf);
   y.SyncFromBlocks();
   gradxf.SyncFromBlocks();
   y.Add(1.0, gradxf);
   y.SyncToBlocks();
}


// curvature test
// dk^T Wk dk + max{ -(lk + lhat)^T ck, 0.0} >= alpha * dk^T dk
// see "An Inertia-Free Filter Line-search Algorithm for
// Large-scale Nonlinear Programming" by Nai-Yuan Chiang and
// Victor M Zavala, Computational Optimization and Applications (2016)
bool IPSolver::CurvatureTest(const BlockOperator & A,
                             const BlockVector & Xhat, const Vector & l, const BlockVector & b,
                             const real_t & delta)
{
   Xhat.SyncToBlocks();
   b.SyncToBlocks();
   Vector lplus(l.Size()); lplus.UseDevice(true);
   lplus.Set(1.0, l);
   lplus.Add(1.0, Xhat.GetBlock(2));


   real_t dWd = 0.0;
   real_t dd = 0.0;
   for (int i = 0; i < 2; i++)
   {
      for (int j = 0; j < 2; j++)
      {
         if (!A.IsZeroBlock(i, j))
         {
            Vector temp(A.GetBlock(i, j).Height()); temp.UseDevice(true); temp = 0.0;
            A.GetBlock(i, j).Mult(Xhat.GetBlock(j), temp);
            dWd += InnerProduct(comm, Xhat.GetBlock(i), temp);
         }
      }
      dd += InnerProduct(comm, Xhat.GetBlock(i), Xhat.GetBlock(i));
   }
   real_t lplusTck = -1.0 * InnerProduct(comm, lplus, b.GetBlock(2));

   bool passed = (dWd + fmax(-lplusTck,
                             0.0) >= alphaCurvatureTest * dd);
   return passed;
}


real_t IPSolver::OptimalityError(const BlockVector &x, const Vector &l,
                                 const Vector &zl, real_t mu)
{
   x.SyncToBlocks();
   /* stationarity, feasibility, and complementarity errors */
   real_t stationarityError, feasibilityError, complementarityError;
   real_t optimalityError;

   /* gradient of Lagrangian */
   BlockVector gradL(block_offsetsx); EnableDevice(gradL);
   gradL = 0.0;
   EvalLagrangianGradient(x, l, zl, gradL);

   /* constraint function c(u, m) = 0 */
   Vector cx(dimC); cx.UseDevice(true); cx = 0.0;
   problem->c(x, cx);

   /* regularized complementarity
    * |z_i * m_i - mu| */
   Vector comp(dimM); comp.UseDevice(true); comp = 0.0;
   const int dim_u = dimU;
   const real_t *x_d = x.Read();
   const real_t *zl_d = zl.Read();
   real_t *comp_d = comp.Write();
   mfem::forall(dimM, [=] MFEM_HOST_DEVICE (int i)
   {
      comp_d[i] = fabs(x_d[dim_u + i] * zl_d[i] - mu);
   });

   BlockVector MxinvgradL(block_offsetsx); EnableDevice(MxinvgradL);
   MxinvgradL = 0.0;
   MxinvgradL.Set(1.0, gradL);
   MxinvgradL.SyncToBlocks();
   MxinvgradL.GetBlock(0) /= Mvlump;
   MxinvgradL.GetBlock(1) /= Mlump;
   MxinvgradL.SyncFromBlocks();
   stationarityError = sqrt(InnerProduct(comm, gradL, MxinvgradL));
   feasibilityError = GlobalLpNorm(infinity(), cx.Normlinf(), comm);
   complementarityError = GlobalLpNorm(infinity(), comp.Normlinf(), comm);


   optimalityError = std::max(std::max(stationarityError, feasibilityError),
                              complementarityError);

   if (myid == 0 && print_level > 0)
   {
      mfem::out << "evaluating optimality error for mu = " << mu << std::endl;
      mfem::out << "stationarity error = " << stationarityError << std::endl;
      mfem::out << "feasibility error  = "    << feasibilityError << std::endl;
      mfem::out << "complimentarity error = " << complementarityError << std::endl;
      mfem::out << "optimality error = " << optimalityError << std::endl;
   }
   return optimalityError;
}

IPSolver::~IPSolver()
{
   if (iter > 0)
   {
      delete JuT;
      delete JmT;
      delete Wuu;
      delete Wmm;
   }
}

}
