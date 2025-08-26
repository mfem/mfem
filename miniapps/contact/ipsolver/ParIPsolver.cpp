#include "mfem.hpp"
#include "ParIPsolver.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace mfem;


ParInteriorPointSolver::ParInteriorPointSolver(OptContactProblem * problem_) 
                     : problem(problem_)
{
   OptTol  = 1.e-2;
   max_iter = 20;
   mu_k     = 1.0;

   sMax     = 1.e2;
   kSig     = 1.e10;   // control deviation from primal Hessian
   tauMin   = 0.99;     // control rate at which iterates can approach the boundary
   eta      = 1.e-4;   // backtracking constant
   thetaMin = 1.e-4;   // allowed violation of the equality constraints

   // constants in line-step A-5.4
   delta    = 1.0;
   sTheta   = 1.1;
   sPhi     = 2.3;

   // control the rate at which the penalty parameter is decreased
   kMu     = 0.2;
   thetaMu = 1.5;

   thetaMax = 1.e6; // maximum constraint violation
   // data for the second order correction
   kSoc     = 0.99;

   // equation (18)
   gTheta = 1.e-5;
   gPhi   = 1.e-5;

   kEps   = 1.e1;

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
   MFEM_VERIFY(dimM == dimC, "only works for dimC = dimM");

   comm = problem->GetComm();
   problem->GetLumpedMassWeights(Mcslump, Mvlump);
   // MFEM_VERIFY(Mcslump.Size() == dimM, "does not work when the bound constraints are active");
   // MFEM_VERIFY(Mvlump.Size() == dimU, "size check failure");

   MPI_Allreduce(&dimU,&gdimU,1,MPI_INT,MPI_SUM,comm);
   MPI_Allreduce(&dimM,&gdimM,1,MPI_INT,MPI_SUM,comm);
   MPI_Allreduce(&dimC,&gdimC,1,MPI_INT,MPI_SUM,comm);

   ckSoc.SetSize(dimC);

   block_offsetsumlz.SetSize(5);
   block_offsetsuml.SetSize(4);
   block_offsetsx.SetSize(3);
  
   block_offsetsumlz[0] = 0;
   block_offsetsumlz[1] = dimU; // u
   block_offsetsumlz[2] = dimM; // m
   block_offsetsumlz[3] = dimC; // lambda
   block_offsetsumlz[4] = dimM; // zl
   block_offsetsumlz.PartialSum();

   if (dimM < dimU)
   {
      dimG = dimM;
      constraint_offsets.SetSize(2);
      constraint_offsets[0] = 0;
      constraint_offsets[1] = dimM;
   } 
   else
   {
      dimG = dimM - 2 * dimU;
      constraint_offsets.SetSize(4);
      constraint_offsets[0] = 0;
      constraint_offsets[1] = dimG;
      constraint_offsets[2] = dimU;
      constraint_offsets[3] = dimU;
   }
   constraint_offsets.PartialSum();


   for(int i = 0; i < block_offsetsuml.Size(); i++)  
   { 
      block_offsetsuml[i] = block_offsetsumlz[i]; 
   }
   for(int i = 0; i < block_offsetsx.Size(); i++)    
   { 
      block_offsetsx[i] = block_offsetsuml[i] ; 
   }

   ml = problem->Getml();
  
   lk.SetSize(dimC);  lk  = 0.0;
   zlk.SetSize(dimM); zlk = 0.0;

   
   MyRank = Mpi::WorldRank();
   iAmRoot = MyRank == 0 ? true : false;
}

// s>=0, s --> s* = O(h^2)
real_t ParInteriorPointSolver::MaxStepSize(Vector &x, Vector &xl, Vector &xhat, real_t tau)
{
   real_t alphaMaxloc = 1.0;
   real_t alphaTmp;
   for(int i = 0; i < x.Size(); i++)
   {   
      if( xhat(i) < 0. )
      {
         alphaTmp = -1. * tau * (x(i) - xl(i)) / xhat(i);
         alphaMaxloc = min(alphaMaxloc, alphaTmp);
      } 
   }

   // alphaMaxloc is the local maximum step size which is
   // distinct on each MPI process. Need to compute
   // the global maximum step size 
   real_t alphaMaxglb;
   MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPITypeMap<real_t>::mpi_type, MPI_MIN, MPI_COMM_WORLD);
   return alphaMaxglb;
}

real_t ParInteriorPointSolver::MaxStepSize(Vector &x, Vector &xhat, real_t tau)
{
   Vector zero(x.Size()); zero = 0.0;
   return MaxStepSize(x, zero, xhat, tau);
}


void ParInteriorPointSolver::Mult(const Vector &x0, Vector &xf)
{
   BlockVector x0block(block_offsetsx); x0block = 0.0;
   x0block.GetBlock(0).Set(1.0, x0);
   x0block.GetBlock(1) = 1.0;
   x0block.GetBlock(1).Add(1.0, ml);
   BlockVector xfblock(block_offsetsx); xfblock = 0.0;
   Mult(x0block, xfblock);
   xf.Set(1.0, xfblock.GetBlock(0));
}


void ParInteriorPointSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   converged = false;
   BlockVector xk(block_offsetsx), xhat(block_offsetsx); xk = 0; xhat = 0.0;
   BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz); Xk = 0.0; Xhat = 0.0;
   BlockVector Xhatuml(block_offsetsuml); Xhatuml = 0.0;
   Vector zlhat(dimM); zlhat = 0.0;

   xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
   xk.GetBlock(1).Set(1.0, x0.GetBlock(1));
   // running estimate of the final values of the Lagrange multipliers
   lk  = 0.0;
   zlk = 0.0;

   for (int i = 0; i < dimM; i++)
   {
      zlk(i) = 1.e1 * mu_k / (xk(i+dimU) - ml(i));
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
   real_t theta0 = theta(xk);
   thetaMin = 1.e-4 * max(1.0, theta0);
   thetaMax = 1.e8  * thetaMin; // 1.e4 * max(1.0, theta0)

   real_t Eeval, Eeval0, maxBarrierSolves, Eevalmu0;
   bool printOptimalityError; // control optimality error print to console for log-barrier subproblems
   
   maxBarrierSolves = 10;

   for(jOpt = 0; jOpt < max_iter; jOpt++)
   {
      if(iAmRoot)
      {
         std::cout << "\n" << std::string(50,'-') << endl;
         std::cout << "interior-point solve step " << jOpt << endl;
      }
      // A-2. Check convergence of overall optimization problem
      printOptimalityError = true;
      Eevalmu0 = E(xk, lk, zlk, printOptimalityError);
      if(Eevalmu0 < OptTol) // div Eeval0 for rel tol
      {
         converged = true;
         int numActiveConstraintsLoc = 0;
	 real_t zinfnorm = GlobalLpNorm(infinity(), zlk.Normlinf(), MPI_COMM_WORLD);
	 real_t uinfnorm = GlobalLpNorm(infinity(), Xk.GetBlock(0).Normlinf(), MPI_COMM_WORLD);
	 real_t sinfnorm = GlobalLpNorm(infinity(), Xk.GetBlock(1).Normlinf(), MPI_COMM_WORLD);
	 real_t linfnorm = GlobalLpNorm(infinity(), lk.Normlinf(), MPI_COMM_WORLD);
	 if (iAmRoot)
	 {
	    cout << "||u||_inf = " << uinfnorm << endl;
	    cout << "||s||_inf = " << sinfnorm << endl;
	    cout << "||z||_inf = " << zinfnorm << endl;
	    cout << "||l||_inf = " << linfnorm << endl;
	 }

	      int dimG = dimM;
	      if (dimM > dimU)
	      {
	         dimG = dimM - 2 * dimU;
	      }
	      MFEM_VERIFY(dimG >= 0, "error in determining num of constraints when bound constraints are active");
         for (int i = 0; i < dimG; i++)
         {
            // slack < Lagrange multiplier
            // min(s_i, z_i) \approx 0 for i = 1,2,...,dimM
            // at convergence point
            // s_i z_i \approx OptTol for each i
            if (zlk(i) > sqrt(mu_k) * zinfnorm)
            {
               numActiveConstraintsLoc += 1;
            }
         }
         MPI_Allreduce(&numActiveConstraintsLoc, &numActiveConstraints, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	 
	      if(iAmRoot)
         {
            cout << "solved optimization problem :)\n";
         }
         break;
      }
      
      if(jOpt > 0) { maxBarrierSolves = 1; }
      real_t Eeval_mu_0;
      for(int i = 0; i < maxBarrierSolves; i++)
      {
         // A-3. Check convergence of the barrier subproblem
         printOptimalityError = true;
         Eeval = E(xk, lk, zlk, mu_k, printOptimalityError);
	 if (i == 0)
	 {
	    Eeval_mu_0 = Eeval;
	 }
	 if(Eeval < kEps * mu_k)
         {
            if(iAmRoot)
            {
               cout << "solved barrier subproblem :), for mu = " << mu_k << endl;
            }
            // A-3.1. Recompute the barrier parameter
            mu_k  = max(OptTol / 10., min(kMu * mu_k, pow(mu_k, thetaMu)));
            // A-3.2. Re-initialize the filter
            F1.DeleteAll();
            F2.DeleteAll();
         }
         else
         {
            break;
         }
      }
    
      // A-4. Compute the search direction
      // solve for (uhat, mhat, lhat)
      if(iAmRoot)
      {
         cout << "\n** A-4. IP-Newton solve **\n";
      }
      zlhat = 0.0; Xhatuml = 0.0;
      
      
      bool passedCTest = false; 
      IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, passedCTest, mu_k, false); 
      if (!passedCTest)
      {
         if (iAmRoot) {	 cout << "curvature test failed\n"; }
         real_t deltaReg = 0.0;
	      int maxCTests = 30;

	      // choose appropriate initial inertia regularization
         if (deltaRegLast < deltaRegMin)
	      {
	         deltaReg = deltaReg0;       
	      }
	      else
	      {
	         // try a potentially smaller regularization value than the one that worked last time
	         deltaReg = fmax(deltaRegMin, kRegMinus * deltaRegLast); 
	      }
         // solve with regularization 
	      zlhat = 0.0; Xhatuml = 0.0;
         IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, passedCTest, mu_k, false, deltaReg);

         for (int numCTests = 0; numCTests < maxCTests; numCTests++)
         {
	         if (iAmRoot)
	         {
	            cout << "deltaReg = " << deltaReg << endl;
	         }
	         if (passedCTest)
	         {
	            deltaRegLast = deltaReg;
	            break;
	         }
	         else
	         {
	            if (deltaRegLast < deltaRegMin)
	            {
	               if (iAmRoot)
		            {
		               cout << "delta *= " << kRegBarPlus << "\n";
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
            IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, passedCTest, mu_k, false, deltaReg);
	      }
      }

      // assign data stack, X = (u, m, l, zl)
      Xk = 0.0;
      Xk.GetBlock(0).Set(1.0, xk.GetBlock(0));
      Xk.GetBlock(1).Set(1.0, xk.GetBlock(1));
      Xk.GetBlock(2).Set(1.0, lk);
      Xk.GetBlock(3).Set(1.0, zlk);

      // assign data stack, Xhat = (uhat, mhat, lhat, zlhat)
      Xhat = 0.0;
      for(int i = 0; i < 3; i++)
      {
         Xhat.GetBlock(i).Set(1.0, Xhatuml.GetBlock(i));
      }
      Xhat.GetBlock(3).Set(1.0, zlhat);

      // A-5. Backtracking line search.
      if(iAmRoot)
      {
         cout << "\n** A-5. Linesearch **\n";
         cout << "mu = " << mu_k << endl;
      }
      lineSearch(Xk, Xhat, mu_k);

      if(lineSearchSuccess)
      {
         if(iAmRoot)
         {
            cout << "lineSearch successful :)\n";
         }
         if(!switchCondition || !sufficientDecrease)
         {
            F1.Append( (1. - gTheta) * thx0);
            F2.Append( phx0 - gPhi * thx0);
         }
         // ----- A-6: Accept the trial point
         // print info regarding zl...
         xk.GetBlock(0).Add(alpha, Xhat.GetBlock(0));
         xk.GetBlock(1).Add(alpha, Xhat.GetBlock(1));
         lk.Add(alpha,   Xhat.GetBlock(2));
         zlk.Add(alphaz, Xhat.GetBlock(3));
         projectZ(xk, zlk, mu_k);
      }
      else
      {
         if(iAmRoot)
         {
            cout << "lineSearch not successful :(\n";
            cout << "attempting feasibility restoration with theta = " << thx0 << endl;
            cout << "no feasibility restoration implemented, exiting now \n";
         }
	 MFEM_ABORT("");
         break;
      }
      if(jOpt + 1 == max_iter && iAmRoot) 
      {  
         cout << "maximum optimization iterations :(\n";
      }
   }
   // done with optimization routine, just reassign data to xf reference so
   // that the application code has access to the optimal point
   xf = 0.0;
   xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
   xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
}

void ParInteriorPointSolver::FormIPNewtonMat(BlockVector & x, Vector & l, Vector &zl, 
                                             BlockOperator &Ak, real_t delta)
{
   // WARNING: Huu, Hum, Hmu, Hmm should all be Hessian terms of the Lagrangian, currently we 
   //          them by Hessian terms of the objective function and neglect the Hessian of l^T c

   Huu = problem->Duuf(x); 
   Hum = problem->Dumf(x);
   Hmu = problem->Dmuf(x);
   Hmm = problem->Dmmf(x);
   
   delete JuT;
   delete JmT;
   Ju = problem->Duc(x); JuT = Ju->Transpose();
   Jm = problem->Dmc(x); JmT = Jm->Transpose();

   Vector DiagLogBar(dimM); DiagLogBar = 0.0;
   if (useMassWeights)
   {
      if (constraint_offsets.Size() == 2)
      {
         for(int ii = 0; ii < dimM; ii++)
         {
            DiagLogBar(ii) = (Mcslump(ii) * zl(ii)) / (x(ii+dimU) - ml(ii)) + delta * Mcslump(ii);
         }
      }
      else if(constraint_offsets.Size() == 4)
      {
         for(int ii = 0; ii < dimG; ii++)
         {
            DiagLogBar(ii) = (Mcslump(ii) * zl(ii)) / (x(ii+dimU) - ml(ii)) + delta * Mcslump(ii);
         }
         for(int ii = dimG; ii < dimG + dimU; ii++)
         {
            DiagLogBar(ii) = (Mvlump(ii - dimG) * zl(ii)) / (x(ii+dimU) - ml(ii)) + delta * Mvlump(ii-dimG);
         }
         for(int ii = dimG + dimU; ii < dimM; ii++)
         {
            DiagLogBar(ii) = (Mvlump(ii - dimG - dimU) * zl(ii)) / (x(ii+dimU) - ml(ii)) + delta * Mvlump(ii - dimG - dimU);
         }
      }
   }
   else
   {
      for(int ii = 0; ii < dimM; ii++)
      {
         DiagLogBar(ii) = zl(ii) / (x(ii+dimU) - ml(ii)) + delta;
      }
   }
   
   delete Wmm;
   if(Hmm)
   {
      SparseMatrix * Ds = new SparseMatrix(DiagLogBar);
      HypreParMatrix * D = new HypreParMatrix(comm, problem->GetGlobalNumConstraints(), problem->GetConstraintsStarts(), Ds);
      HypreStealOwnership(*D,*Ds);
      delete Ds;
      Wmm = ParAdd(Hmm,D);
      delete D;
   }
   else
   {
      SparseMatrix * Ds = new SparseMatrix(DiagLogBar);
      Wmm = new HypreParMatrix(comm, problem->GetGlobalNumConstraints(), problem->GetConstraintsStarts(), Ds);
      HypreStealOwnership(*Wmm,*Ds);
      delete Ds;
   }
   
   Vector deltaDiagVec(dimU);
   deltaDiagVec = delta;
   if (useMassWeights)
   {
      deltaDiagVec *= Mvlump;
   }
   delete Wuu;
   if (Huu)
   {
      SparseMatrix * Duus = new SparseMatrix(deltaDiagVec);
      HypreParMatrix * Duu = new HypreParMatrix(comm, problem->GetGlobalNumDofs(), problem->GetDofStarts(), Duus);
      HypreStealOwnership(*Duu, *Duus);
      delete Duus;
      Wuu = ParAdd(Huu, Duu);
      delete Duu; 
   }
   else
   {
      SparseMatrix * DuuS = new SparseMatrix(deltaDiagVec);
      Wuu = new HypreParMatrix(comm, problem->GetGlobalNumDofs(), problem->GetDofStarts(), DuuS);
      HypreStealOwnership(*Wuu, *DuuS);
      delete DuuS;
   }
  
   //         IP-Newton system matrix
   //    Ak = [[H_(u,u)  H_(u,m)   J_u^T]
   //          [H_(m,u)  W_(m,m)   J_m^T]
   //          [ J_u      J_m       0  ]]

   //    Ak = [[K+d  0     Jᵀ ]   [u]    [bᵤ]
   //          [0    D+d    -I]   [m]  = [bₘ]      
   //          [J   -I     0  ]]  [λ]  = [bₗ ]
   Ak.SetBlock(0, 0, Wuu);                         Ak.SetBlock(0, 2, JuT);
                           Ak.SetBlock(1, 1, Wmm); Ak.SetBlock(1, 2, JmT);
   Ak.SetBlock(2, 0,  Ju); Ak.SetBlock(2, 1,  Jm);
   if (Hum)
   {
      Ak.SetBlock(0, 1, Hum);
      Ak.SetBlock(1, 0, Hmu);
   }
}

// perturbed KKT system solve
// determine the search direction
void ParInteriorPointSolver::IPNewtonSolve(BlockVector &x, Vector &l, Vector &zl, Vector &zlhat, BlockVector &Xhat, bool & passedCTest, real_t mu, bool socSolve, real_t delta)
{
   StopWatch chrono;
   chrono.Clear();
   iter++;
   // solve A x = b, where A is the IP-Newton matrix
   BlockOperator A(block_offsetsuml, block_offsetsuml); 
   BlockVector b(block_offsetsuml); b = 0.0;
   FormIPNewtonMat(x, l, zl, A, delta);

   //       [grad_u phi + Ju^T l]
   // b = - [grad_m phi + Jm^T l]
   //       [          c        ]
   BlockVector gradphi(block_offsetsx); gradphi = 0.0;
   BlockVector JTl(block_offsetsx); JTl = 0.0;
   Dxphi(x, mu, gradphi);
   
   (A.GetBlock(0,2)).Mult(l, JTl.GetBlock(0));
   (A.GetBlock(1,2)).Mult(l, JTl.GetBlock(1));

   for(int ii = 0; ii < 2; ii++)
   {
      b.GetBlock(ii).Set(1.0, gradphi.GetBlock(ii));
      b.GetBlock(ii).Add(1.0, JTl.GetBlock(ii));
   }
   if(!socSolve) 
   {
      problem->c(x, b.GetBlock(2));
   }
   else
   {
      b.GetBlock(2).Set(1.0, ckSoc);
   }
   b *= -1.0; 
   Xhat = 0.0;

   // form A = Huu + Ju^T D Ju, Wmm = D for contact
   HypreParMatrix * Wmmloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(1, 1)));
   HypreParMatrix * Huuloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 0)));
   HypreParMatrix * Juloc  = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(2, 0)));
      
   HypreParMatrix * JuTloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 2)));
   HypreParMatrix *JuTDJu   = RAP(Wmmloc, Juloc);     // Ju^T D Ju
   HypreParMatrix *Areduced = ParAdd(Huuloc, JuTDJu);  // Huu + Ju^T D Ju

   /* compute the reduced rhs */
   // breduced = bu + Ju^T (bm + Wmm bl)
   Vector breduced(dimU); breduced = 0.0;
   Vector tempVec(dimM); tempVec = 0.0;
   Wmmloc->Mult(b.GetBlock(2), tempVec);
   tempVec.Add(1.0, b.GetBlock(1));
   JuTloc->Mult(tempVec, breduced);
   breduced.Add(1.0, b.GetBlock(0));

   // Solver the system with the given solver   
   solver->SetOperator(*Areduced);
   solver->Mult(breduced, Xhat.GetBlock(0));
   auto itsolver = dynamic_cast<IterativeSolver *>(solver);
   int numit = (itsolver) ? itsolver->GetNumIterations() : -1;
   cgnum_iterations.Append(numit);

   // now propagate solved uhat to obtain mhat and lhat
   // xm = Ju xu - bl
   Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
   Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

   // xl = Wmm xm - bm
   Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
   Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

   delete JuTDJu;
   delete Areduced;

   passedCTest = CurvatureTest(A, Xhat, l, b, delta);

   /* backsolve to determine zlhat */
   for(int ii = 0; ii < dimM; ii++)
   {
      zlhat(ii) = -1.*(zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)) );
   }
}

// here Xhat, X will be BlockVectors w.r.t. the 4 partitioning X = (u, m, l, zl)

void ParInteriorPointSolver::lineSearch(BlockVector& X0, BlockVector& Xhat, real_t mu)
{
   real_t tau  = max(tauMin, 1.0 - mu);
   int eval_err = 0;
   Vector u0   = X0.GetBlock(0);
   Vector m0   = X0.GetBlock(1);
   Vector l0   = X0.GetBlock(2);
   Vector z0   = X0.GetBlock(3);
   Vector uhat = Xhat.GetBlock(0);
   Vector mhat = Xhat.GetBlock(1);
   Vector lhat = Xhat.GetBlock(2);
   Vector zhat = Xhat.GetBlock(3);
   real_t alphaMax  = MaxStepSize(m0, ml, mhat, tau);
   real_t alphaMaxz = MaxStepSize(z0, zhat, tau);
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

   Vector ck0(dimC); ck0 = 0.0;
   Vector zhatsoc(dimM); zhatsoc = 0.0;
   BlockVector Xhatumlsoc(block_offsetsuml); Xhatumlsoc = 0.0;
   BlockVector xhatsoc(block_offsetsx); xhatsoc = 0.0;
   Vector uhatsoc(dimU); uhatsoc = 0.0;
   Vector mhatsoc(dimM); mhatsoc = 0.0;

   Dxphi(x0, mu, Dxphi0);

   Dxphi0_xhat = InnerProduct(MPI_COMM_WORLD, Dxphi0, xhat);
   descentDirection = Dxphi0_xhat < 0. ? true : false;
   
   
   if (iAmRoot)
   {
      if(descentDirection)
      {
         cout << "is a descent direction for the log-barrier objective\n";
      }
      else
      {
         cout << "is not a descent direction for the log-barrier objective\n";
      }
   }

   thx0 = theta(x0);
   phx0 = phi(x0, mu);

   lineSearchSuccess = false;
   for(int i = 0; i < maxBacktrack; i++)
   {
      if (iAmRoot)
      {
         cout << "\n--------- alpha = " << alpha << " ---------\n";
      }
      // ----- A-5.2. Compute trial point: xtrial = x0 + alpha_i xhat
      xtrial.Set(1.0, x0);
      xtrial.Add(alpha, xhat);

      // ------ A-5.3. if not in filter region go to A.5.4 otherwise go to A-5.5.
      thxtrial = theta(xtrial);
      phxtrial = phi(xtrial, mu, eval_err);
      if (eval_err == 1)
      {
         if(iAmRoot)
	 {
	    cout << "bad step\n";
	 }
	 alpha *= 0.5;
	 continue;
      }
      filterCheck(thxtrial, phxtrial);    
      if(!inFilterRegion)
      {
         if (iAmRoot)
         {
            cout << "not in filter region :)\n";
         }
         // ------ A.5.4: Check sufficient decrease
         if(!descentDirection)
         {
            switchCondition = false;
         }
         else
         {
            switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0, sTheta)) ? true : false;
         }
         if (iAmRoot)
         {
            cout << "theta(x0) = "     << thx0     << ", thetaMin = "                  << thetaMin             << endl;
            cout << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = "     << (1. - gTheta) * thx0 << endl;
            cout << "phi(xtrial) = "   << phxtrial << ", phi(x0) - gPhi *theta(x0) = " << phx0 - gPhi * thx0   << endl;
         }      
         // Case I      
         if(thx0 <= thetaMin && switchCondition)
         {
            sufficientDecrease = (phxtrial <= phx0 + eta * alpha * Dxphi0_xhat) ? true : false;
            if(sufficientDecrease)
            {
               if(iAmRoot) { cout << "Line search successful: sufficient decrease in log-barrier objective.\n"; } 
               // accept the trial step
               lineSearchSuccess = true;
               break;
            }
         }
         else
         {
            if(thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0)
            {
               if(iAmRoot) { cout << "Line search successful: infeasibility or log-barrier objective decreased.\n"; } 
               // accept the trial step
               lineSearchSuccess = true;
               break;
            }
         }
         // A-5.5: Initialize the second-order correction
         if((!(thx0 < thxtrial)) && i == 0)
         {
            ////WARNING: not complete but currently solver isn't entering this region
         }
      }
      else
      {
         if (iAmRoot)
         {
            cout << "in filter region :(\n"; 
         }
      }
      // include more if needed
      alpha *= 0.5;
   } 
}

void ParInteriorPointSolver::projectZ(const Vector &x, Vector &z, real_t mu)
{
   real_t zi;
   real_t mudivmml;
   for(int i = 0; i < dimM; i++)
   {
      zi = z(i);
      mudivmml = mu / (x(i + dimU) - ml(i));
      z(i) = max(min(zi, kSig * mudivmml), mudivmml / kSig);
   }
}

void ParInteriorPointSolver::filterCheck(real_t th, real_t ph)
{
   inFilterRegion = false;
   if(th > thetaMax)
   {
      inFilterRegion = true;
   }
   else
   {
      for(int i = 0; i < F1.Size(); i++)
      {
         if(th >= F1[i] && ph >= F2[i])
         {
            inFilterRegion = true;
            break;
         }
      }
   }
}

// curvature test
// dk^T Wk dk + max{ -(lk + lhat)^T ck, 0.0} >= alpha * dk^T dk
// see "An Inertia-Free Filter Line-search Algorithm for
// Large-scale Nonlinear Programming" by Nai-Yuan Chiang and
// Victor M Zavala, Computational Optimization and Applications (2016)
bool ParInteriorPointSolver::CurvatureTest(const BlockOperator & A, const BlockVector & Xhat, const Vector & l, const BlockVector & b, const real_t & delta)
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
	    dWd += InnerProduct(MPI_COMM_WORLD, Xhat.GetBlock(i), temp);
	 }
      }
      dd += InnerProduct(MPI_COMM_WORLD, Xhat.GetBlock(i), Xhat.GetBlock(i));
   }
   real_t lplusTck = -1.0 * InnerProduct(MPI_COMM_WORLD, lplus, b.GetBlock(2));
 
   //if (iAmRoot)
   //{ 
   //   cout << "d^T W d + max{-(l+)^T c, 0} = " << dWd + fmax(-lplusTck, 0.0) << endl;
   //   cout << "d^T d = " << dd << endl;
   //   cout << "d^T W d / d^T d = " << dWd / dd << endl;
   //}
   bool passed = (dWd + fmax(-lplusTck, 0.0) >= alphaCurvatureTest * dd) ? true : false;
   return passed;
}


real_t ParInteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, real_t mu, bool printEeval)
{
   real_t E1, E2, E3; // stationarity, feasibility, and complementarity errors
   real_t optimalityError;
   real_t sc, sd;
   BlockVector gradL(block_offsetsx); gradL = 0.0; // stationarity grad L = grad f + J^T l - z
   Vector cx(dimC); cx = 0.0;     // feasibility c = c(x)
   Vector comp(dimM); comp = 0.0; // complementarity M Z - mu 1

   DxL(x, l, zl, gradL);

   problem->c(x, cx);


   for(int ii = 0; ii < dimM; ii++) 
   { 
      comp(ii) = abs((x(dimU + ii) - ml(ii)) * zl(ii) - mu);
   }
   
   if (!useMassWeights)
   {
      E1 = GlobalLpNorm(infinity(), gradL.Normlinf(), MPI_COMM_WORLD); 
      E2 = GlobalLpNorm(infinity(), cx.Normlinf(), MPI_COMM_WORLD); 
      E3 = GlobalLpNorm(infinity(), comp.Normlinf(), MPI_COMM_WORLD); 

      // compute norms of Lagrange multipliers
      // if they are growing large this is indicative of
      // poorly conditioned constraint Jacobians
      // and here we terminate the algorithm early
      real_t zl1 = GlobalLpNorm(1, zl.Norml1(), MPI_COMM_WORLD); 
      real_t ll1 = GlobalLpNorm(1, l.Norml1(), MPI_COMM_WORLD);
      sc = max(sMax, zl1 / (real_t(gdimM)) ) / sMax;
      sd = max(sMax, (ll1 + zl1) / (real_t(gdimC + gdimM))) / sMax;
   }
   else
   {
      BlockVector MxinvgradL(block_offsetsx); MxinvgradL = 0.0;
      MxinvgradL.Set(1.0, gradL);
      MxinvgradL.GetBlock(0) /= Mvlump; 
      BlockVector gradsL(constraint_offsets); gradsL = 0.0;
      gradsL.Set(1.0, MxinvgradL.GetBlock(1));
      gradsL.GetBlock(0) /= Mcslump;
      if (constraint_offsets.Size() == 4)
      {
         gradsL.GetBlock(1) /= Mvlump;
	      gradsL.GetBlock(2) /= Mvlump;
      }
      MxinvgradL.GetBlock(1).Set(1.0, gradsL);
      E1 = sqrt(InnerProduct(MPI_COMM_WORLD, gradL, MxinvgradL));
      E2 = GlobalLpNorm(infinity(), cx.Normlinf(), MPI_COMM_WORLD);
      E3 = GlobalLpNorm(infinity(), comp.Normlinf(), MPI_COMM_WORLD);
   }
   
   optimalityError = max(max(E1, E2), E3);

   if(iAmRoot && printEeval)
   {
      cout << "evaluating optimality error for mu = " << mu << endl;
      cout << "stationarity measure = " << E1 << endl;
      cout << "feasibility measure  = "    << E2      << endl;
      cout << "complimentarity measure = " << E3 << endl;
      cout << "optimality error = " << optimalityError << endl;
   }
   return optimalityError;
}

real_t ParInteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, bool printEeval)
{
  return E(x, l, zl, 0.0, printEeval);
}

real_t ParInteriorPointSolver::theta(const BlockVector &x)
{
   Vector cx(dimC);
   problem->c(x, cx);
   if (useMassWeights)
   {
      if (constraint_offsets.Size() == 2)
      {
         Vector Mcx(dimC);
         Mcx.Set(1.0, cx);
         Mcx *= Mcslump;
         return sqrt(InnerProduct(MPI_COMM_WORLD, Mcx, cx));
      }
      else
      {
         BlockVector Mcx(constraint_offsets);
         Mcx.Set(1.0, cx);
         Mcx.GetBlock(0) *= Mcslump;
         Mcx.GetBlock(1) *= Mvlump;
         Mcx.GetBlock(2) *= Mvlump;
         return sqrt(InnerProduct(MPI_COMM_WORLD, Mcx, cx));
      } 
   }
   else
   {
      return sqrt(InnerProduct(MPI_COMM_WORLD, cx, cx));
   }
}

// log-barrier objective
real_t ParInteriorPointSolver::phi(const BlockVector &x, real_t mu, int & eval_err)
{
   real_t fx = problem->CalcObjective(x, eval_err); 
   real_t logBarrierLoc = 0.0;
   if (useMassWeights)
   {
      for(int i = 0; i < dimG; i++) 
      { 
        logBarrierLoc += Mcslump(i) * log(x(dimU+i)-ml(i));
      }
      if (dimM != dimG)
      {
         for(int i = dimG; i < dimU + dimG; i++)
	      {    
            logBarrierLoc += Mvlump(i - dimG) * log(x(dimU+i)-ml(i));
	      }
         for(int i = dimG + dimU; i < dimM; i++)
         {
            logBarrierLoc += Mvlump(i - dimG - dimU) * log(x(dimU+i)-ml(i));
         }
      }
   }
   else
   {
      for(int i = 0; i < dimM; i++) 
      { 
        logBarrierLoc += log(x(dimU+i)-ml(i));
      }
   }
   real_t logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
   return fx - mu * logBarrierGlb;
}

real_t ParInteriorPointSolver::phi(const BlockVector & x, real_t mu)
{
   int eval_err = 0;
   return phi(x, mu, eval_err);
}


// gradient of log-barrier objective with respect to x = (u, m)
void ParInteriorPointSolver::Dxphi(const BlockVector &x, real_t mu, BlockVector &y)
{
   problem->CalcObjectiveGrad(x, y);
   //Vector ytemp(dimM); ytemp = 0.0;
   BlockVector ytemp(constraint_offsets); ytemp = 0.0;
   for (int i = 0; i < dimM; i++)
   {
      ytemp(i) = 1. / (x(dimU + i) - ml(i));
   }

   if (useMassWeights)
   {
      ytemp.GetBlock(0) *= Mcslump;
      if (constraint_offsets.Size() == 4)
      {
         ytemp.GetBlock(1) *= Mvlump;
	 ytemp.GetBlock(2) *= Mvlump;
      }
   }

   y.GetBlock(1).Add(-mu, ytemp);
  
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
real_t ParInteriorPointSolver::L(const BlockVector &x, const Vector &l, const Vector &zl)
{
   int eval_err = 0;
   real_t fx = problem->CalcObjective(x, eval_err);
   Vector cx(dimC); problem->c(x, cx);
   //Vector temp(dimM); temp = 0.0;
   BlockVector temp(constraint_offsets); temp = 0.0;
   temp.Set(1.0, x.GetBlock(1));
   temp.Add(-1.0, ml);
   if( useMassWeights)
   {
      temp.GetBlock(0) *= Mcslump;
      if (constraint_offsets.Size() == 4)
      {
         temp.GetBlock(1) *= Mvlump;
	 temp.GetBlock(2) *= Mvlump;
      }
   }
   return (fx + InnerProduct(MPI_COMM_WORLD, cx, l) - InnerProduct(MPI_COMM_WORLD, temp, zl));
}

void ParInteriorPointSolver::DxL(const BlockVector &x, const Vector &l, const Vector &zl, BlockVector &y)
{
   // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
   BlockVector gradxf(block_offsetsx); gradxf = 0.0;
   problem->CalcObjectiveGrad(x, gradxf);
   
   HypreParMatrix *Jacu, *Jacm, *JacuT, *JacmT;
   Jacu = problem->Duc(x); 
   Jacm = problem->Dmc(x);
   JacuT = Jacu->Transpose();
   JacmT = Jacm->Transpose();
   
   JacuT->Mult(l, y.GetBlock(0));
   JacmT->Mult(l, y.GetBlock(1));
   
   delete JacuT;
   delete JacmT;
   
   y.Add(1.0, gradxf);
   
   //Vector temp(dimM); temp = 0.0;
   BlockVector temp(constraint_offsets); temp = 0.0;
   temp.Set(1.0, zl);
   if (useMassWeights)
   {
      temp.GetBlock(0) *= Mcslump;
      if (constraint_offsets.Size() == 4)
      {
         temp.GetBlock(1) *= Mvlump;
	 temp.GetBlock(2) *= Mvlump;
      }
   }
   (y.GetBlock(1)).Add(-1.0, temp);
}

bool ParInteriorPointSolver::GetConverged() const
{
   return converged;
}

void ParInteriorPointSolver::SetTol(real_t Tol)
{
   OptTol = Tol;
}

void ParInteriorPointSolver::SetMaxIter(int max_it)
{
   max_iter = max_it;
}

void ParInteriorPointSolver::SetBarrierParameter(real_t mu_0)
{
   mu_k = mu_0;
}

void ParInteriorPointSolver::SetUsingMassWeights(bool useMassWeights_)
{
   useMassWeights = useMassWeights_;
}

void ParInteriorPointSolver::SaveLogBarrierHessianIterates(bool save)
{
   MFEM_ASSERT(MyRank == 0 || save == false, "currently can only save logbarrier hessian in serial codes");
   saveLogBarrierIterates = save;
}

ParInteriorPointSolver::~ParInteriorPointSolver() 
{
   delete JuT;
   delete JmT;
   delete Wuu;
   delete Wmm;
}
