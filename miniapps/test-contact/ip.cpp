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

#include "ip.hpp"
#include "../common/schwarz_solver.hpp"

namespace mfem
{

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

   Mlump.SetSize(dimM); Mlump = 0.0;
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
      BlockVector Mlumpblock(constraint_offsets); Mlumpblock = 0.0;
      Mlumpblock.GetBlock(0).Set(1.0, Mcslump);
      Mlumpblock.GetBlock(1).Set(1.0, Mvlump);
      Mlumpblock.GetBlock(2).Set(1.0, Mvlump);
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

   lk.SetSize(dimC);  lk  = 0.0;
   zlk.SetSize(dimM); zlk = 0.0;

   MPI_Comm_rank(comm, &myid);
}

void IPSolver::Mult(const Vector &x0, Vector &xf)
{
   BlockVector x0block(block_offsetsx); x0block = 0.0;
   x0block.GetBlock(0).Set(1.0, x0);
   x0block.GetBlock(1) = 1.0;
   BlockVector xfblock(block_offsetsx); xfblock = 0.0;
   Mult(x0block, xfblock);
   xf.Set(1.0, xfblock.GetBlock(0));
}

void IPSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   converged = false;
   lin_solver_iterations.DeleteAll();
   lin_solver_times.DeleteAll();
   BlockVector xk(block_offsetsx), xhat(block_offsetsx); xk = 0; xhat = 0.0;
   BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz); Xk = 0.0;
   Xhat = 0.0;
   BlockVector Xhatuml(block_offsetsuml); Xhatuml = 0.0;
   Vector zlhat(dimM); zlhat = 0.0;

   xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
   xk.GetBlock(1).Set(1.0, x0.GetBlock(1));
   // running estimate of the final values of the Lagrange multipliers
   lk  = 0.0;
   zlk = 0.0;

   for (int i = 0; i < dimM; i++)
   {
      zlk(i) = 1.e1 * mu_k / xk(i+dimU);
   }

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

      Xhat = 0.0;
      for (int i = 0; i < 3; i++)
      {
         Xhat.GetBlock(i).Set(1.0, Xhatuml.GetBlock(i));
      }
      Xhat.GetBlock(3).Set(1.0, zlhat);

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
   xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
   xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
}

void IPSolver::FormIPNewtonMat(BlockVector & x, Vector & l,
                               Vector &zl,
                               BlockOperator &Ak, real_t delta)
{
   Huu = problem->Duuf(x);
   Hmm = problem->Dmmf(x);

   delete JuT;
   delete JmT;
   Ju = problem->Duc(x); JuT = Ju->Transpose();
   Jm = problem->Dmc(x); JmT = Jm->Transpose();

   Vector DiagLogBar(dimM); DiagLogBar = 0.0;
   for (int i = 0; i < dimM; i++)
   {
      DiagLogBar(i) = (Mlump(i) * zl(i)) / x(i + dimU) + delta * Mlump(i);
   }

   delete Wmm;
   if (Hmm)
   {
      SparseMatrix * Ds = new SparseMatrix(DiagLogBar);
      HypreParMatrix * D = new HypreParMatrix(comm,
                                              problem->GetGlobalNumConstraints(), problem->GetConstraintsStarts(), Ds);
      HypreStealOwnership(*D,*Ds);
      delete Ds;
      Wmm = ParAdd(Hmm,D);
      delete D;
   }
   else
   {
      SparseMatrix * Ds = new SparseMatrix(DiagLogBar);
      Wmm = new HypreParMatrix(comm, problem->GetGlobalNumConstraints(),
                               problem->GetConstraintsStarts(), Ds);
      HypreStealOwnership(*Wmm,*Ds);
      delete Ds;
   }

   Vector deltaDiagVec(dimU);
   deltaDiagVec = delta;
   deltaDiagVec *= Mvlump;

   delete Wuu;
   if (Huu)
   {
      SparseMatrix * Duus = new SparseMatrix(deltaDiagVec);
      HypreParMatrix * Duu = new HypreParMatrix(comm, problem->GetGlobalNumDofs(),
                                                problem->GetDofStarts(), Duus);
      HypreStealOwnership(*Duu, *Duus);
      delete Duus;
      Wuu = ParAdd(Huu, Duu);
      delete Duu;
   }
   else
   {
      SparseMatrix * DuuS = new SparseMatrix(deltaDiagVec);
      Wuu = new HypreParMatrix(comm, problem->GetGlobalNumDofs(),
                               problem->GetDofStarts(), DuuS);
      HypreStealOwnership(*Wuu, *DuuS);
      delete DuuS;
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

// Build Schwarz subdomains from contact constraint Jacobian
void IPSolver::BuildSchwarzSubdomains(HypreParMatrix* Areduced, const BlockVector& x)
{
   if (!use_schwarz_subspace) return;

   // Get constraint Jacobian J and transfer operator P
   HypreParMatrix* J = problem->Duc(x);  // J: constraints × displacements
   HypreParMatrix* P = problem->GetContactSubspaceTransferOperator();  // P: subspace × displacements

   // Check if Schwarz solver is set
   if (!schwarz_solver) return;

   // Define D diagonal matrix in the same way as Wmm
   Vector D_diag(dimM);
   Wmm->GetDiag(D_diag);

   SparseMatrix * Ds = new SparseMatrix(D_diag);
   HypreParMatrix * D = new HypreParMatrix(comm,
                                           problem->GetGlobalNumConstraints(), problem->GetConstraintsStarts(), Ds);
   HypreStealOwnership(*D, *Ds);
   delete Ds;

   if (schwarz_examine_diagonal)
   {
      // Compute statistics for D diagonal
      double diag_min = D_diag.Min();
      double diag_max = D_diag.Max();
      double diag_sum = 0.0;
      for (int i = 0; i < D_diag.Size(); i++)
      {
         diag_sum += D_diag(i);
      }

      // Global reduction for statistics
      double global_min, global_max, global_sum;
      int local_size = D_diag.Size();
      int total_size;

      MPI_Allreduce(&diag_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
      MPI_Allreduce(&diag_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
      MPI_Allreduce(&diag_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
      MPI_Allreduce(&local_size, &total_size, 1, MPI_INT, MPI_SUM, comm);

      double global_avg = global_sum / total_size;

      if (myid == 0)
      {
         mfem::out << "\nDiagonal of D (Wmm) statistics:" << std::endl;
         mfem::out << "  Min: " << global_min << std::endl;
         mfem::out << "  Max: " << global_max << std::endl;
         mfem::out << "  Avg: " << global_avg << std::endl;
         mfem::out << "  Total size: " << total_size << std::endl;
         if (schwarz_min_diag_value > 0.0)
         {
            mfem::out << "  Using minimal D threshold: " << schwarz_min_diag_value << std::endl;
         }
         mfem::out << std::endl;
      }
   }

   // Compute projected operator P^T * Areduced * P
   HypreParMatrix* PTAP = RAP(Areduced, P);

   // Merge J matrix to access local rows
   SparseMatrix merged_J;
   J->MergeDiagAndOffd(merged_J);

   // Access PTAP diagonal for graph queries (if expanding subdomains)
   hypre_ParCSRMatrix* parPTAP = (hypre_ParCSRMatrix*)(*PTAP);
   hypre_CSRMatrix* PTAP_diag = hypre_ParCSRMatrixDiag(parPTAP);
   HYPRE_Int* PTAP_Ai = hypre_CSRMatrixI(PTAP_diag);
   HYPRE_Int* PTAP_Aj = hypre_CSRMatrixJ(PTAP_diag);
   HYPRE_Int num_dofs_subspace = hypre_CSRMatrixNumRows(PTAP_diag);

   int nranks;
   MPI_Comm_size(comm, &nranks);

   const HYPRE_BigInt local_ptap_row_start = PTAP->GetRowStarts()[0];

   struct DofMapping
   {
      HYPRE_BigInt global_dof;
      HYPRE_BigInt subspace_dof;
   };

   std::vector<DofMapping> local_mappings;

   // Get P's row/column partitions to convert local indices to global indices.
   const HYPRE_BigInt *p_row_starts = P->GetRowStarts();
   const HYPRE_BigInt p_local_row_start = p_row_starts[0];
   const HYPRE_BigInt p_local_col_start = P->GetColStarts()[0];
   const HYPRE_Int *P_diag_I = P->GetDiagMemoryI();
   const HYPRE_Int *P_diag_J = P->GetDiagMemoryJ();
   const HYPRE_Int *P_offd_I = P->GetOffdMemoryI();
   const HYPRE_Int *P_offd_J = P->GetOffdMemoryJ();
   HYPRE_BigInt *P_offd_cmap = nullptr;
   HYPRE_Int P_num_offd_cols = 0;
   P->GetOffdColMap(P_offd_cmap, P_num_offd_cols);

   for (int i = 0; i < P->GetNumRows(); i++)
   {
      const HYPRE_BigInt global_dof = p_local_row_start + i;
      for (int j = P_diag_I[i]; j < P_diag_I[i + 1]; ++j)
      {
         local_mappings.push_back({global_dof, p_local_col_start + P_diag_J[j]});
      }
      for (int j = P_offd_I[i]; j < P_offd_I[i + 1]; ++j)
      {
         local_mappings.push_back({global_dof, P_offd_cmap[P_offd_J[j]]});
      }
   }

   int local_count = local_mappings.size();
   std::vector<int> counts(nranks), displs(nranks + 1, 0);

   MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

   for (int r = 0; r < nranks; r++)
   {
      displs[r + 1] = displs[r] + counts[r];
   }

   int total_mappings = displs[nranks];
   std::vector<DofMapping> all_mappings(total_mappings);

   std::vector<int> byte_counts(nranks), byte_displs(nranks);
   for (int r = 0; r < nranks; r++)
   {
      byte_counts[r] = counts[r] * sizeof(DofMapping);
      byte_displs[r] = displs[r] * sizeof(DofMapping);
   }

   MPI_Allgatherv(local_mappings.data(), local_count * sizeof(DofMapping), MPI_BYTE,
                  all_mappings.data(), byte_counts.data(), byte_displs.data(),
                  MPI_BYTE, comm);

   std::map<HYPRE_BigInt, HYPRE_BigInt> global_to_subspace;
   for (const auto &mapping : all_mappings)
   {
      global_to_subspace[mapping.global_dof] = mapping.subspace_dof;
   }

   std::vector<std::vector<HYPRE_BigInt>> local_subdomains;
   {
      int local_J_rows = J->Height();
      for (int i = 0; i < local_J_rows; i++)
      {
         const int* cols = merged_J.GetRowColumns(i);
         int row_size = merged_J.RowSize(i);

         if (row_size == 0) { continue; }

         // Skip this constraint if D diagonal value is below threshold.
         if (schwarz_min_diag_value > 0.0 && i < D_diag.Size())
         {
            if (D_diag(i) < schwarz_min_diag_value)
            {
               continue;
            }
         }

         std::set<HYPRE_BigInt> subdomain_set;

         // Add all DoFs in this contact constraint.
         for (int j = 0; j < row_size; j++)
         {
            const HYPRE_BigInt global_dof = cols[j];
            auto it = global_to_subspace.find(global_dof);
            if (it != global_to_subspace.end())
            {
               subdomain_set.insert(it->second);
            }
         }

         if (schwarz_expand)
         {
            std::vector<HYPRE_BigInt> original_dofs(subdomain_set.begin(), subdomain_set.end());
            for (HYPRE_BigInt dof : original_dofs)
            {
               if (dof >= local_ptap_row_start &&
                   dof < local_ptap_row_start + num_dofs_subspace)
               {
                  const HYPRE_Int local_dof = static_cast<HYPRE_Int>(dof - local_ptap_row_start);
                  for (HYPRE_Int jj = PTAP_Ai[local_dof]; jj < PTAP_Ai[local_dof + 1]; jj++)
                  {
                     const HYPRE_BigInt neighbor = local_ptap_row_start + PTAP_Aj[jj];
                     if (neighbor != dof)
                     {
                        subdomain_set.insert(neighbor);
                     }
                  }
               }
            }
         }

         if (!subdomain_set.empty())
         {
            local_subdomains.emplace_back(subdomain_set.begin(), subdomain_set.end());
         }
      }
   } // end row-based subdomain selection

   // J rows may be unevenly distributed across ranks, so rebalance the
   // generated subdomains explicitly before building the Schwarz factors.
   const int local_num_subdomains = static_cast<int>(local_subdomains.size());
   std::vector<int> subdomain_counts(nranks, 0);
   MPI_Allgather(&local_num_subdomains, 1, MPI_INT,
                 subdomain_counts.data(), 1, MPI_INT, comm);

   std::vector<int> subdomain_displs(nranks, 0);
   int total_subdomains = 0;
   for (int r = 0; r < nranks; ++r)
   {
      subdomain_displs[r] = total_subdomains;
      total_subdomains += subdomain_counts[r];
   }

   std::vector<int> local_subdomain_sizes(local_num_subdomains, 0);
   int local_total_memberships = 0;
   for (int i = 0; i < local_num_subdomains; ++i)
   {
      local_subdomain_sizes[i] = static_cast<int>(local_subdomains[i].size());
      local_total_memberships += local_subdomain_sizes[i];
   }

   std::vector<int> all_subdomain_sizes(total_subdomains, 0);
   MPI_Allgatherv(local_subdomain_sizes.empty() ? nullptr : local_subdomain_sizes.data(),
                  local_num_subdomains, MPI_INT,
                  all_subdomain_sizes.empty() ? nullptr : all_subdomain_sizes.data(),
                  subdomain_counts.data(), subdomain_displs.data(), MPI_INT, comm);

   std::vector<int> membership_counts(nranks, 0);
   MPI_Allgather(&local_total_memberships, 1, MPI_INT,
                 membership_counts.data(), 1, MPI_INT, comm);

   std::vector<int> membership_displs(nranks, 0);
   int total_memberships = 0;
   for (int r = 0; r < nranks; ++r)
   {
      membership_displs[r] = total_memberships;
      total_memberships += membership_counts[r];
   }

   std::vector<HYPRE_BigInt> local_flat_subdomains;
   local_flat_subdomains.reserve(local_total_memberships);
   for (const auto &subdomain : local_subdomains)
   {
      local_flat_subdomains.insert(local_flat_subdomains.end(),
                                   subdomain.begin(), subdomain.end());
   }

   std::vector<HYPRE_BigInt> all_flat_subdomains(total_memberships);
   MPI_Allgatherv(local_flat_subdomains.empty() ? nullptr : local_flat_subdomains.data(),
                  local_total_memberships, HYPRE_MPI_BIG_INT,
                  all_flat_subdomains.empty() ? nullptr : all_flat_subdomains.data(),
                  membership_counts.data(), membership_displs.data(),
                  HYPRE_MPI_BIG_INT, comm);

   std::vector<int> subdomain_value_offsets(total_subdomains + 1, 0);
   for (int i = 0; i < total_subdomains; ++i)
   {
      subdomain_value_offsets[i + 1] = subdomain_value_offsets[i] + all_subdomain_sizes[i];
   }

   std::vector<std::vector<HYPRE_BigInt>> subdomains = local_subdomains;
   std::vector<int> assigned_subdomain_counts = subdomain_counts;
   std::vector<int> assigned_subdomain_sizes(nranks, 0);
   MPI_Allgather(&local_total_memberships, 1, MPI_INT,
                 assigned_subdomain_sizes.data(), 1, MPI_INT, comm);

   // Print diagnostic information about subdomains
   if (myid == 0 && (print_level > 1 || schwarz_examine_diagonal))
   {
      // Compute subdomain size statistics on the globally balanced set.
      int num_subdomains = total_subdomains;
      int total_size = 0;
      int min_size = num_subdomains > 0 ? all_subdomain_sizes[0] : 0;
      int max_size = 0;

      // Track covered DOFs
      std::set<HYPRE_BigInt> covered_dofs;
      for (int i = 0; i < num_subdomains; ++i)
      {
         int size = all_subdomain_sizes[i];
         total_size += size;
         if (size < min_size) min_size = size;
         if (size > max_size) max_size = size;

         for (int j = subdomain_value_offsets[i]; j < subdomain_value_offsets[i + 1]; ++j)
         {
            covered_dofs.insert(all_flat_subdomains[j]);
         }
      }

      double avg_size = num_subdomains > 0 ? (double)total_size / num_subdomains : 0.0;
      int num_covered = covered_dofs.size();
      int num_uncovered = PTAP->N() - num_covered;
      double overlap_ratio = num_covered > 0 ? (double)(total_size - num_covered) / num_covered : 0.0;

      mfem::out << "\nSchwarz subdomain statistics:" << std::endl;
      mfem::out << "  Total subspace size: " << PTAP->N() << std::endl;
      mfem::out << "  Number of subdomains: " << num_subdomains << std::endl;
      mfem::out << "  Assigned subdomains/rank: ";
      for (int r = 0; r < nranks; ++r)
      {
         mfem::out << assigned_subdomain_counts[r];
         if (r + 1 < nranks)
         {
            mfem::out << " ";
         }
      }
      mfem::out << std::endl;
      if (schwarz_min_diag_value > 0.0)
      {
         int total_constraints = J->Height();
         int num_filtered = total_constraints - num_subdomains;
         mfem::out << "  Total constraints: " << total_constraints << std::endl;
         mfem::out << "  Filtered (D < " << schwarz_min_diag_value << "): " << num_filtered << std::endl;
      }
      if (num_subdomains > 0)
      {
         mfem::out << "  Subdomain sizes - min: " << min_size
                  << ", max: " << max_size
                  << ", avg: " << avg_size << std::endl;
         mfem::out << "  DOF coverage: " << num_covered << "/" << PTAP->N()
                  << " (" << (100.0 * num_covered / PTAP->N()) << "%)" << std::endl;
         mfem::out << "  Uncovered DOFs: " << num_uncovered << std::endl;
         mfem::out << "  Total DOF instances (with overlaps): " << total_size << std::endl;
         mfem::out << "  Overlap ratio: " << overlap_ratio << std::endl;

         if (num_uncovered > 0)
         {
            mfem::out << "  WARNING: " << num_uncovered << " DOFs are not covered by any subdomain!" << std::endl;
            mfem::out << "           This may affect Schwarz preconditioner effectiveness." << std::endl;
         }
      }
      mfem::out << std::endl;
   }

   // Initialize HYPRE Schwarz if not already done
   if (!schwarz_solver->schwarz_solver)
   {
      HYPRE_SchwarzCreate(&schwarz_solver->schwarz_solver);
      HYPRE_SchwarzSetVariant(schwarz_solver->schwarz_solver, HypreSchwarz::RequiredVariant);
   }

   // Configure Schwarz with custom subdomains
   // use_nonsymm = 0 → Cholesky factorization (symmetric, appropriate for contact)
   HYPRE_Int use_nonsymm = 0;
   schwarz_solver->SetCustomSubdomains(subdomains, *PTAP, schwarz_relax_weight, use_nonsymm, schwarz_unweighted, schwarz_uniform_weight);

   delete PTAP;
   delete D;
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
   BlockVector b(block_offsetsuml); b = 0.0;
   FormIPNewtonMat(x, l, zl, A, delta);

   //       [grad_u phi + Ju^T l]
   // b = - [grad_m phi + Jm^T l]
   //       [          c        ]
   BlockVector gradphi(block_offsetsx); gradphi = 0.0;
   GetDxphi(x, mu, gradphi);

   for (int i = 0; i < 2; i++)
   {
      b.GetBlock(i).Set(1.0, gradphi.GetBlock(i));
      A.GetBlock(i, 2).AddMult(l, b.GetBlock(i));
   }
   problem->c(x, b.GetBlock(2));
   b *= -1.0;
   Xhat = 0.0;

   // form A = Huu + Ju^T D Ju, Wmm = D for contact
   HypreParMatrix *JuTDJu   = RAP(Wmm, Ju);     // Ju^T D Ju
   HypreParMatrix *Areduced = ParAdd(Huu, JuTDJu);  // Huu + Ju^T D Ju

   // Build Schwarz subdomains from current J matrix
   if (use_schwarz_subspace)
   {
      BuildSchwarzSubdomains(Areduced, x);
   }

   /* compute the reduced rhs */
   // breduced = bu + Ju^T (bm + Wmm bl)
   Vector breduced(dimU); breduced = 0.0;
   Vector tempVec(dimM); tempVec = 0.0;
   Wmm->Mult(b.GetBlock(2), tempVec);
   tempVec.Add(1.0, b.GetBlock(1));
   JuT->Mult(tempVec, breduced);
   breduced.Add(1.0, b.GetBlock(0));

   HypreParMatrix *Schur_system = nullptr;
   HypreParMatrix *P_free_T = nullptr;
   HypreParMatrix *P_free = nullptr;
   HypreParVector *b_schur = nullptr;
   HypreParVector *x_schur = nullptr;

   if (use_schur_complement)
   {
      // Get contact subspace transfer operator P_contact: (contact space) -> (full space)
      // P_contact maps from contact DOFs to full DOFs
      // The columns of P_contact tell us which full-space DOFs are contact DOFs
      HypreParMatrix *P_contact = problem->GetContactSubspaceTransferOperator();

      // Extract contact DOF indices from P_contact columns
      // P_contact is (num_contact_dofs x num_full_dofs)
      // We need to find which columns have nonzero entries
      Array<HYPRE_BigInt> contact_dofs;

      // Iterate through diagonal and off-diagonal blocks
      const HYPRE_Int *diag_I = P_contact->GetDiagMemoryI();
      const HYPRE_Int *diag_J = P_contact->GetDiagMemoryJ();
      const HYPRE_Int *offd_I = P_contact->GetOffdMemoryI();

      HYPRE_BigInt *offd_cmap = nullptr;
      HYPRE_Int num_offd_cols = 0;
      P_contact->GetOffdColMap(offd_cmap, num_offd_cols);

      const HYPRE_BigInt col_start = P_contact->GetColStarts()[0];

      for (int i = 0; i < P_contact->GetNumRows(); ++i)
      {
         // Diagonal entries
         for (int j = diag_I[i]; j < diag_I[i + 1]; ++j)
         {
            contact_dofs.Append(col_start + diag_J[j]);
         }

         // Off-diagonal entries
         for (int j = offd_I[i]; j < offd_I[i + 1]; ++j)
         {
            contact_dofs.Append(offd_cmap[j]);
         }
      }

      contact_dofs.Sort();
      contact_dofs.Unique();

      // Build set for fast lookup
      std::set<HYPRE_BigInt> contact_dofs_set(contact_dofs.GetData(),
                                               contact_dofs.GetData() + contact_dofs.Size());

      // Build projector onto free (non-contact) DOFs
      const HYPRE_BigInt local_row_start = Areduced->GetRowStarts()[0];
      const HYPRE_BigInt local_row_end = Areduced->GetRowStarts()[1];
      const int local_num_rows = Areduced->GetNumRows();
      const HYPRE_BigInt global_size = Areduced->N();

      int nrows_free = 0;
      for (int i = 0; i < local_num_rows; ++i)
      {
         if (contact_dofs_set.find(local_row_start + i) == contact_dofs_set.end())
         {
            nrows_free++;
         }
      }

      SparseMatrix Pf_local(nrows_free, global_size);
      int free_idx = 0;
      for (int i = 0; i < local_num_rows; ++i)
      {
         if (contact_dofs_set.find(local_row_start + i) == contact_dofs_set.end())
         {
            Pf_local.Set(free_idx++, local_row_start + i, 1.0);
         }
      }
      Pf_local.Finalize();

      // Create parallel projector
      HYPRE_BigInt rows_f[2];
      HYPRE_BigInt nrows_free_bigint = nrows_free;
      HYPRE_BigInt row_offset_free;
      MPI_Scan(&nrows_free_bigint, &row_offset_free, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);
      row_offset_free -= nrows_free_bigint;
      rows_f[0] = row_offset_free;
      rows_f[1] = row_offset_free + nrows_free;

      HYPRE_BigInt glob_nrows_free;
      MPI_Allreduce(&nrows_free_bigint, &glob_nrows_free, 1, HYPRE_MPI_BIG_INT, MPI_SUM, comm);

      HYPRE_BigInt *J_f;
#ifndef HYPRE_BIGINT
      J_f = Pf_local.GetJ();
#else
      J_f = new HYPRE_BigInt[Pf_local.NumNonZeroElems()];
      for (int i = 0; i < Pf_local.NumNonZeroElems(); i++)
      {
         J_f[i] = Pf_local.GetJ()[i];
      }
#endif

      P_free_T = new HypreParMatrix(comm,
                                     nrows_free, glob_nrows_free,
                                     global_size,
                                     Pf_local.GetI(), J_f, Pf_local.GetData(),
                                     rows_f, Areduced->GetColStarts());

      P_free = P_free_T->Transpose();

      // Form Schur complement: S = P_f^T * Areduced * P_f
      HypreParMatrix *AP_free = ParMult(Areduced, P_free);
      Schur_system = ParMult(P_free_T, AP_free);
      delete AP_free;

      // Project RHS: b_schur = P_f^T * breduced
      b_schur = new HypreParVector(comm, glob_nrows_free, rows_f);
      P_free_T->Mult(breduced, *b_schur);

      // Allocate solution vector
      x_schur = new HypreParVector(comm, glob_nrows_free, rows_f);
      *x_schur = 0.0;

      // Solve reduced system
      solver->SetOperator(*Schur_system);
      const double linear_solve_start = MPI_Wtime();
      solver->Mult(*b_schur, *x_schur);
      const double linear_solve_elapsed_local = MPI_Wtime() - linear_solve_start;
      double linear_solve_elapsed = 0.0;
      MPI_Allreduce(&linear_solve_elapsed_local, &linear_solve_elapsed, 1,
                    MPI_DOUBLE, MPI_MAX, comm);
      lin_solver_times.Append(linear_solve_elapsed);

      // Expand solution back: Xhat[0] = P_free * x_schur
      P_free->Mult(*x_schur, Xhat.GetBlock(0));

#ifdef HYPRE_BIGINT
      delete [] J_f;
#endif
   }
   else
   {
      // Standard solve on full system
      solver->SetOperator(*Areduced);
      const double linear_solve_start = MPI_Wtime();
      solver->Mult(breduced, Xhat.GetBlock(0));
      const double linear_solve_elapsed_local = MPI_Wtime() - linear_solve_start;
      double linear_solve_elapsed = 0.0;
      MPI_Allreduce(&linear_solve_elapsed_local, &linear_solve_elapsed, 1,
                    MPI_DOUBLE, MPI_MAX, comm);
      lin_solver_times.Append(linear_solve_elapsed);
   }

   if (lobpcg)
   {
      HypreParMatrix *minusAreduced = new HypreParMatrix(*Areduced);
      *minusAreduced *= -1.0;
      lobpcg->SetOperator(*minusAreduced);
      lobpcg->Solve();
      delete minusAreduced;
   }

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

   // Cleanup
   if (use_schur_complement)
   {
      delete x_schur;
      delete b_schur;
      delete Schur_system;
      delete P_free;
      delete P_free_T;
   }

   delete JuTDJu;
   delete Areduced;

   passedCurvatureTest = CurvatureTest(A, Xhat, l, b, delta);

   /* backsolve to determine zlhat */
   for (int i = 0; i < dimM; i++)
   {
      zlhat(i) = zl(i) + (zl(i) * Xhat(i + dimU) - mu) / x(i + dimU);
   }
   zlhat *= -1.;
}

real_t IPSolver::GetMaxStepSize(Vector &x, Vector &xhat,
                                real_t tau)
{
   real_t alphaMaxloc = 1.0;
   real_t alphaTmp;
   for (int i = 0; i < x.Size(); i++)
   {
      if ( xhat(i) < 0. )
      {
         alphaTmp = -1. * tau * x(i) / xhat(i);
         alphaMaxloc = std::min(alphaMaxloc, alphaTmp);
      }
   }

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

   BlockVector x0(block_offsetsx); x0 = 0.0;
   x0.GetBlock(0).Set(1.0, u0);
   x0.GetBlock(1).Set(1.0, m0);

   BlockVector xhat(block_offsetsx); xhat = 0.0;
   xhat.GetBlock(0).Set(1.0, uhat);
   xhat.GetBlock(1).Set(1.0, mhat);

   BlockVector xtrial(block_offsetsx); xtrial = 0.0;
   BlockVector Dxphi0(block_offsetsx); Dxphi0 = 0.0;
   int maxBacktrack = 20;
   alpha = alphaMax;

   GetDxphi(x0, mu, Dxphi0);

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
   real_t zdual_i;
   real_t zprimal_i;
   for (int i = 0; i < dimM; i++)
   {
      zdual_i = z(i);
      zprimal_i = mu / x(i + dimU);
      z(i) = std::max(std::min(zdual_i, kSig * zprimal_i), zprimal_i / kSig);
   }
}

real_t IPSolver::GetTheta(const BlockVector &x)
{
   Vector cx(dimC);
   problem->c(x, cx);
   Vector Mcx(dimC);
   Mcx.Set(1.0, cx);
   Mcx *= Mlump;
   return sqrt(InnerProduct(comm, Mcx, cx));
}

// log-barrier objective
real_t IPSolver::GetPhi(const BlockVector &x, real_t mu,
                        int eval_err)
{
   real_t fx = problem->CalcObjective(x, eval_err);
   real_t logBarrierLoc = 0.0;
   for (int i = 0; i < dimM; i++)
   {
      logBarrierLoc += Mlump(i) * log(x(dimU + i));
   }
   real_t logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, comm);
   return fx - mu * logBarrierGlb;
}

// gradient of log-barrier objective with respect to x = (u, m)
void IPSolver::GetDxphi(const BlockVector &x, real_t mu,
                        BlockVector &y)
{
   problem->CalcObjectiveGrad(x, y);
   Vector ytemp(dimM); ytemp = 1.0;
   ytemp /= x.GetBlock(1);
   ytemp *= Mlump;
   y.GetBlock(1).Add(-mu, ytemp);
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
real_t IPSolver::EvalLangrangian(const BlockVector &x, const Vector &l,
                                 const Vector &zl)
{
   int eval_err = 0;
   real_t fx = problem->CalcObjective(x, eval_err);
   Vector cx(dimC); problem->c(x, cx);
   Vector temp(dimM); temp = 0.0;
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
   // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
   y.GetBlock(1).Set(-1.0, zl);
   y.GetBlock(1) *= Mlump;

   problem->Duc(x)->MultTranspose(l, y.GetBlock(0));
   problem->Dmc(x)->AddMultTranspose(l, y.GetBlock(1));

   BlockVector gradxf(block_offsetsx); gradxf = 0.0;
   problem->CalcObjectiveGrad(x, gradxf);
   y.Add(1.0, gradxf);
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
   Vector lplus(l.Size());
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
            Vector temp(A.GetBlock(i, j).Height()); temp = 0.0;
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
   /* stationarity, feasibility, and complementarity errors */
   real_t stationarityError, feasibilityError, complementarityError;
   real_t optimalityError;

   /* gradient of Lagrangian */
   BlockVector gradL(block_offsetsx);
   gradL = 0.0;
   EvalLagrangianGradient(x, l, zl, gradL);

   /* constraint function c(u, m) = 0 */
   Vector cx(dimC); cx = 0.0;
   problem->c(x, cx);

   /* regularized complementarity
    * |z_i * m_i - mu| */
   Vector comp(dimM); comp = 0.0; // complementarity Z m - mu 1
   for (int i = 0; i < dimM; i++)
   {
      comp(i) = abs(x(dimU + i) * zl(i) - mu);
   }

   BlockVector MxinvgradL(block_offsetsx); MxinvgradL = 0.0;
   MxinvgradL.Set(1.0, gradL);
   MxinvgradL.GetBlock(0) /= Mvlump;
   MxinvgradL.GetBlock(1) /= Mlump;
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
