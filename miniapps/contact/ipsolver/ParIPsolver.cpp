#include "mfem.hpp"
#include "ParIPsolver.hpp"
#include "two-level-solver.hpp"
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

   comm = problem->GetComm();

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

double ParInteriorPointSolver::MaxStepSize(Vector &x, Vector &xl, Vector &xhat, double tau)
{
   double alphaMaxloc = 1.0;
   double alphaTmp;
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
   double alphaMaxglb;
   MPI_Allreduce(&alphaMaxloc, &alphaMaxglb, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   return alphaMaxglb;
}

double ParInteriorPointSolver::MaxStepSize(Vector &x, Vector &xhat, double tau)
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
   double theta0 = theta(xk);
   thetaMin = 1.e-4 * max(1.0, theta0);
   thetaMax = 1.e8  * thetaMin; // 1.e4 * max(1.0, theta0)

   double Eeval, maxBarrierSolves, Eevalmu0;
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
      printOptimalityError = false;
      Eevalmu0 = E(xk, lk, zlk, printOptimalityError);
      if(Eevalmu0 < OptTol)
      {
         converged = true;
         if(iAmRoot)
         {
            cout << "solved optimization problem :)\n";
         }
         break;
      }
      
      if(jOpt > 0) { maxBarrierSolves = 1; }
      
      for(int i = 0; i < maxBarrierSolves; i++)
      {
         // A-3. Check convergence of the barrier subproblem
         printOptimalityError = true;
         Eeval = E(xk, lk, zlk, mu_k, printOptimalityError);
         if(iAmRoot)
         {
            cout << "E = " << Eeval << endl;
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
         cout << "curvature test failed\n";
         double deltaReg = 0.0;
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
                                             BlockOperator &Ak, double delta)
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
   for(int ii = 0; ii < dimM; ii++)
   {
      DiagLogBar(ii) = zl(ii) / (x(ii+dimU) - ml(ii)) + delta;
   }

   double dmax = (DiagLogBar.Size() > 0) ? DiagLogBar.Max() : -infinity();
   double dmin = (DiagLogBar.Size() > 0) ? DiagLogBar.Min() : infinity();

   MPI_Allreduce(MPI_IN_PLACE, &dmax,1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE, &dmin,1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

   dynamiclinSolver = linSolver;
   if (dynamicsolver && linSolver == 2 && dmax/dmin > 5e6)
   {
      dynamiclinSolver = 6;
   }
   dmaxmin_ratio.Append(dmax/dmin);


   if(saveLogBarrierIterates)
   {
      std::ofstream diagStream;
      char diagString[100];
      snprintf(diagString, 100, "logBarrierHessiandata/D%d.dat", jOpt);
      diagStream.open(diagString, ios::out | ios::trunc);
      for(int ii = 0; ii < dimM; ii++)
      {
         diagStream << setprecision(30) << DiagLogBar(ii) << endl;
         // mfem::out << DiagLogBar(ii) << endl;
      }
      diagStream.close();
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

   //    Ak = [[K    0     Jᵀ ]   [u]    [bᵤ]
   //          [0    D    -I  ]   [m]  = [bₘ]      
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
void ParInteriorPointSolver::IPNewtonSolve(BlockVector &x, Vector &l, Vector &zl, Vector &zlhat, BlockVector &Xhat, bool & passedCTest, double mu, bool socSolve, double delta)
{
   StopWatch chrono;
   chrono.Clear();
   iter++;
   // solve A x = b, where A is the IP-Newton matrix
   BlockOperator A(block_offsetsuml, block_offsetsuml); 
   BlockVector b(block_offsetsuml); b = 0.0;
   FormIPNewtonMat(x, l, zl, A);

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

   // Direct solver (default)
   if(dynamiclinSolver == 0)
   {
      Array2D<const HypreParMatrix *> ABlockMatrix(3,3);
      for(int ii = 0; ii < 3; ii++)
      {
         for(int jj = 0; jj < 3; jj++)
         {
            if(!A.IsZeroBlock(ii, jj))
            {
               ABlockMatrix(ii, jj) = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(ii, jj)));
	    }
	    else
	    {
	       ABlockMatrix(ii, jj) = nullptr;
	    }
         }  
      }
      
      HypreParMatrix * Ah = HypreParMatrixFromBlocks(ABlockMatrix);   

      /* direct solve of the 3x3 IP-Newton linear system */
      Solver * ASolver;
#ifdef MFEM_USE_MUMPS
      ASolver = new MUMPSSolver(MPI_COMM_WORLD);
      auto AmSolver = dynamic_cast<MUMPSSolver *>(ASolver);
      AmSolver->SetPrintLevel(0);
      AmSolver->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
#else 
#ifdef MFEM_USE_MKL_CPARDISO
      ASolver = new CPardisoSolver(MPI_COMM_WORLD);
      auto AcSolver = dynamic_cast<CPardisoSolver *>(ASolver);
      AcSolver->SetMatrixType(CPardisoSolver::MatType::REAL_NONSYMMETRIC);
#else
      MFEM_VERIFY(false, "linSolver 0 will not work unless compiled with MUMPS or MKL");
#endif
#endif
      ASolver->SetOperator(*Ah);
      ASolver->Mult(b, Xhat);
      delete ASolver;
      delete Ah;
   }
   else if(dynamiclinSolver >= 1)
   {
      // form A = Huu + Ju^T D Ju, Wmm = D for contact
      HypreParMatrix * Wmmloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(1, 1)));
      HypreParMatrix * Huuloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 0)));
      HypreParMatrix * Juloc  = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(2, 0)));
      
      HypreParMatrix * JuTloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 2)));
      HypreParMatrix *JuTDJu   = RAP(Wmmloc, Juloc);     // Ju^T D Ju
      HypreParMatrix *Areduced = ParAdd(Huuloc, JuTDJu);  // Huu + Ju^T D Ju

      Vector diag1(JuTDJu->Height());
      JuTDJu->GetDiag(diag1);
      Vector diag2(Areduced->Height());
      Areduced->GetDiag(diag2);

      double d1max = (diag1.Size() > 0) ? diag1.Max() : -infinity();
      // double d1min = (diag1.Size() > 0) ? diag1.Min() : infinity();
      double d1min = infinity();
      for (int i = 0; i< diag1.Size(); i++)
      {
         if (diag1[i] += 0.0)
         {
            d1min = min(d1min,diag1[i]);
         }
      }

      MPI_Allreduce(MPI_IN_PLACE, &d1max,1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &d1min,1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

      double d2max = (diag2.Size() > 0) ? diag2.Max() : -infinity();
      double d2min = (diag2.Size() > 0) ? diag2.Min() : infinity();

      MPI_Allreduce(MPI_IN_PLACE, &d2max,1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &d2min,1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

      jtdj_ratio.Append(d1max/d1min);
      Adiag_ratio.Append(d2max/d2min);


      /* prepare the reduced rhs */
      // breduced = bu + Ju^T (bm + Wmm bl)
      Vector breduced(dimU); breduced = 0.0;
      Vector tempVec(dimM); tempVec = 0.0;
      Wmmloc->Mult(b.GetBlock(2), tempVec);
      tempVec.Add(1.0, b.GetBlock(1));
      JuTloc->Mult(tempVec, breduced);
      breduced.Add(1.0, b.GetBlock(0));
      
      // Direct solver on the reduced system
      if(dynamiclinSolver == 1) 
      {
         // setup the solver for the reduced linear system
	 Solver * AreducedSolver;
#ifdef MFEM_USE_MUMPS
	 AreducedSolver = new MUMPSSolver(MPI_COMM_WORLD);
	 auto ArSolver = dynamic_cast<MUMPSSolver *>(AreducedSolver);
         ArSolver->SetPrintLevel(0);
         ArSolver->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
#else 
#ifdef MFEM_USE_MKL_CPARDISO
	 AreducedSolver = new CPardisoSolver(MPI_COMM_WORLD);
	 auto ArSolver = dynamic_cast<CPardisoSolver *>(AreducedSolver);
         ArSolver->SetMatrixType(CPardisoSolver::MatType::REAL_NONSYMMETRIC);
#else
         MFEM_VERIFY(false, "linSolver 1 will not work unless compiled with MUMPS or MKL");
#endif
#endif
         AreducedSolver->SetOperator(*Areduced);
         AreducedSolver->Mult(breduced, Xhat.GetBlock(0));
	 delete AreducedSolver;
      }
      // PCG-AMG solver on the reduced system
      else if (dynamiclinSolver == 2 || dynamiclinSolver == 5)
      {
         HypreBoomerAMG amg(*Areduced);
         // HypreBoomerAMG amg(*Huu);
         amg.SetPrintLevel(0);
         amg.SetRelaxType(relax_type);
         if (pfes)
         {
            amg.SetElasticityOptions(pfes);
         }
         else
         {
            amg.SetSystemsOptions(3,false);
         }
         int n;

         if (iAmRoot)
         {
            std::cout << "\n" << std::string(50,'-') << endl;
            mfem::out << std::string(20,' ') << "PCG SOLVER" << endl;
            std::cout << std::string(50,'-') << endl;
         }

         IterativeSolver * AreducedSolver = nullptr;

         if (dynamiclinSolver == 2) 
         {
            AreducedSolver = new CGSolver(MPI_COMM_WORLD);
            // AreducedSolver = new SLISolver(MPI_COMM_WORLD);
         }
         else
         {
            AreducedSolver = new SLISolver(MPI_COMM_WORLD);
         }
         AreducedSolver->SetRelTol(linSolveRelTol);
         AreducedSolver->SetMaxIter(50000);
         AreducedSolver->SetPrintLevel(3);
         AreducedSolver->SetOperator(*Areduced);
         GeneralSolutionMonitor * sol_monitor = nullptr;
         if (monitor)
         {
            AreducedSolver->SetMonitor(*sol_monitor);
         }
         AreducedSolver->SetPreconditioner(amg);
         chrono.Clear();
         chrono.Start();
         amg.Setup(breduced, Xhat.GetBlock(0));
         chrono.Stop();
         if (iAmRoot)
         {
            mfem::out << "AMG Setup time         = " << chrono.RealTime() << endl;
         }
         chrono.Clear();         
         chrono.Start();
         AreducedSolver->Mult(breduced, Xhat.GetBlock(0));
         chrono.Stop();
         n = AreducedSolver->GetNumIterations();
         if (monitor) 
         {  
            delete sol_monitor; 
            mfem::out << "Program paused. Press enter to continue...\n";
            cin.get();
         }
         if (iAmRoot)
         {
            mfem::out << "CG Mult total time     = " << chrono.RealTime() << endl;
            mfem::out << "CG Mult time/iteration = " << chrono.RealTime()/n << endl;
         }
         if (!AreducedSolver->GetConverged())
         {
            delete AreducedSolver;
            AreducedSolver = new GMRESSolver(MPI_COMM_WORLD);
            AreducedSolver->SetRelTol(linSolveRelTol);
            AreducedSolver->SetMaxIter(500);
            AreducedSolver->SetPrintLevel(3);
            AreducedSolver->SetOperator(*Areduced);
            AreducedSolver->SetPreconditioner(amg);
            AreducedSolver->Mult(breduced, Xhat.GetBlock(0));
            n = -AreducedSolver->GetNumIterations();
         }

         if (iAmRoot)
         {
            std::cout << std::string(50,'-') << "\n" << endl;
            if (!AreducedSolver->GetConverged())
            {
               if (iAmRoot)
               {
                  mfem::out << "CG interagtions = "; 
                  cgnum_iterations.Print(mfem::out, cgnum_iterations.Size());
               }
            }
         }
         MFEM_VERIFY(AreducedSolver->GetConverged(), "PCG solver did not converge");
         cgnum_iterations.Append(n);
         delete AreducedSolver;
      }
#ifdef MFEM_USE_MUMPS
      else if (dynamiclinSolver == 6) // Two level
      {
         if (iAmRoot)
         {
            std::cout << "\n" << std::string(50,'-') << endl;
            mfem::out << std::string(20,' ') << "PCG SOLVER" << endl;
            std::cout << std::string(50,'-') << endl;
         } 
         HypreParMatrix * Pb = problem->GetRestrictionToContactDofs();
         TwoLevelAMGSolver prec(*Areduced, *Pb);
         prec.SetAMGRelaxType(relax_type);
         CGSolver AreducedSolver(MPI_COMM_WORLD);
         // SLISolver AreducedSolver(MPI_COMM_WORLD);
         GeneralSolutionMonitor * sol_monitor = nullptr;
         if (monitor)
         {
            sol_monitor = new GeneralSolutionMonitor(problem->GetElasticityOperator()->GetFESpace(),  Areduced, breduced,1);
            AreducedSolver.SetMonitor(*sol_monitor);
         }    
         AreducedSolver.SetRelTol(linSolveRelTol);
         AreducedSolver.SetMaxIter(500);
         AreducedSolver.SetPrintLevel(3);
         AreducedSolver.SetOperator(*Areduced);
         AreducedSolver.SetPreconditioner(prec);
         chrono.Clear();
         chrono.Start();
         Xhat.GetBlock(0).Randomize();
         AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
         chrono.Stop();
         if (monitor) 
         {  
            delete sol_monitor;
            mfem::out << "Program paused. Press enter to continue...\n"; 
            cin.get();
         }
         int n = AreducedSolver.GetNumIterations();
         if (iAmRoot)
         {
            mfem::out << "CG Mult total time     = " << chrono.RealTime() << endl;
            mfem::out << "CG Mult time/iteration = " << chrono.RealTime()/n << endl;
         }
         if (iAmRoot)
         {
            std::cout << std::string(50,'-') << "\n" << endl;
            if (!AreducedSolver.GetConverged())
            {
               if (iAmRoot)
               {
                  mfem::out << "CG interagtions = "; 
                  cgnum_iterations.Print(mfem::out, cgnum_iterations.Size());
               }
            }
         }
         MFEM_VERIFY(AreducedSolver.GetConverged(), "PCG solver did not converge");
         cgnum_iterations.Append(n);
      }
      else // if linsolver == 3 or 4
      {
         // Extract interior and contact dofs
         HypreParMatrix * Pb = problem->GetRestrictionToContactDofs();
         HypreParMatrix * Pi = problem->GetRestrictionToInteriorDofs();

         HypreParMatrix * PitAPi = RAP(Areduced, Pi);
         HypreParMatrix * PbtAPb = RAP(Areduced, Pb);
         HypreParMatrix * PitAPb = RAP(Pi,Areduced,Pb);
         HypreParMatrix * PbtAPi = RAP(Pb,Areduced,Pi);

         Vector Xi(PitAPi->Height()); Pi->MultTranspose(Xhat.GetBlock(0),Xi); 
         Vector Xb(PbtAPb->Height()); Pb->MultTranspose(Xhat.GetBlock(0),Xb); 
         Vector bi(PitAPi->Height()); Pi->MultTranspose(breduced,bi); 
         Vector bb(PbtAPb->Height()); Pb->MultTranspose(breduced,bb); 
         Vector Xib, Bib;
         if (dynamiclinSolver == 3)
         {
            Array<int> blkoffs(3);
            blkoffs[0] = 0;
            blkoffs[1] = Xi.Size();
            blkoffs[2] = Xb.Size();
            blkoffs.PartialSum();

            BlockOperator blkA(blkoffs);
            blkA.SetBlock(0,0, PitAPi);
            blkA.SetBlock(0,1, PitAPb);
            blkA.SetBlock(1,0, PbtAPi);
            blkA.SetBlock(1,1, PbtAPb);

            Xib.SetSize(blkoffs.Last());
            Bib.SetSize(blkoffs.Last());

            Xib.SetVector(Xi,0);
            Xib.SetVector(Xb,blkoffs[1]);

            Bib.SetVector(bi,0);
            Bib.SetVector(bb,blkoffs[1]);

            HypreBoomerAMG amg_i(*PitAPi);
            amg_i.SetPrintLevel(0);
            amg_i.SetSystemsOptions(3,false);
            amg_i.SetRelaxType(relax_type);
            amg_i.SetOperator(*PitAPi);
            MUMPSSolver mumps_b(MPI_COMM_WORLD);
            mumps_b.SetPrintLevel(0);
            mumps_b.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
            mumps_b.SetOperator(*PbtAPb);


            // BlockDiagonalPreconditioner prec(blkoffs);
            BlockTriangularSymmetricPreconditioner prec(blkoffs);
            prec.SetOperator(blkA);
            prec.SetDiagonalBlock(0,&amg_i);
            prec.SetBlock(0,1,PitAPb);
            prec.SetBlock(1,0,PbtAPi);
            prec.SetDiagonalBlock(1,&mumps_b);

            CGSolver BlockSolver(MPI_COMM_WORLD);
            BlockSolver.SetRelTol(linSolveRelTol);
            BlockSolver.SetMaxIter(5000);
            BlockSolver.SetPrintLevel(3);
            BlockSolver.SetOperator(blkA);
            BlockSolver.SetPreconditioner(prec);
            BlockSolver.Mult(Bib, Xib);
            int m = BlockSolver.GetNumIterations();
            cgnum_iterations.Append(m);
            
            Xi.MakeRef(Xib,0);
            Xb.MakeRef(Xib,blkoffs[1]);

            Vector PiX(Xhat.GetBlock(0).Size());
            Vector PbX(Xhat.GetBlock(0).Size());
            Xhat.GetBlock(0) = 0.0;
            Pi->Mult(Xi,PiX);
            Pb->Mult(Xb,PbX);
            // recover original ordering for the solution;
            Xhat.GetBlock(0) += PiX;
            Xhat.GetBlock(0) += PbX;
         }
         else
         {
            MUMPSSolver * mumps_b = new MUMPSSolver(MPI_COMM_WORLD);
            mumps_b->SetPrintLevel(0);
            mumps_b->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
            mumps_b->SetOperator(*PbtAPb);
            // (Aᵢᵢ - Aᵢⱼ A⁻¹ⱼⱼ Aⱼᵢ) xᵢ = bᵢ - Aᵢⱼ A⁻¹ⱼⱼ bⱼ
            TripleProductOperator * S = new TripleProductOperator(PitAPb, mumps_b, PbtAPi,true,true,true); // Aᵢⱼ A⁻¹ⱼⱼ Aⱼᵢ
            SumOperator SumOp(PitAPi,1.0,S,-1.0,true,true); // Aᵢᵢ - Aᵢⱼ A⁻¹ⱼⱼ Aⱼᵢ

            Vector Yj(bb.Size()); mumps_b->Mult(bb,Yj); // A⁻¹ⱼⱼ bⱼ
            Vector Yi(bi.Size()); PitAPb->Mult(Yj,Yi); //Aᵢⱼ A⁻¹ⱼⱼ bⱼ
            bi -= Yi; // bᵢ - Aᵢⱼ A⁻¹ⱼⱼ bⱼ

            HypreBoomerAMG amg_i(*PitAPi);
            amg_i.SetPrintLevel(0);
            amg_i.SetSystemsOptions(3,false);
            amg_i.SetRelaxType(relax_type);

            CGSolver solver(MPI_COMM_WORLD);
            solver.SetRelTol(linSolveRelTol);
            solver.SetMaxIter(5000);
            solver.SetPrintLevel(3);
            solver.SetOperator(SumOp);
            solver.SetPreconditioner(amg_i);
            solver.Mult(bi, Xi);
            int m = solver.GetNumIterations();
            cgnum_iterations.Append(m);

            Vector Yb(bb.Size());
            PbtAPi->Mult(Xi,Yb);
            bb -= Yb;
            mumps_b->Mult(bb,Xb);

            Vector PiX(Xhat.GetBlock(0).Size());
            Vector PbX(Xhat.GetBlock(0).Size());
            Xhat.GetBlock(0) = 0.0;
            Pi->Mult(Xi,PiX);
            Pb->Mult(Xb,PbX);
            // recover original ordering for the solution;
            Xhat.GetBlock(0) += PiX;
            Xhat.GetBlock(0) += PbX;
         }
      }
#else
      else
      {
         MFEM_ABORT("This solver choice requires compiling with MUMPS");
      }
#endif      
      // now propagate solved uhat to obtain mhat and lhat
      // xm = Ju xu - bl
      Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
      Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

      // xl = Wmm xm - bm
      Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
      Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

      delete JuTDJu;
      delete Areduced;
   }

   passedCTest = CurvatureTest(A, Xhat, l, b, delta);


   /* backsolve to determine zlhat */
   for(int ii = 0; ii < dimM; ii++)
   {
      zlhat(ii) = -1.*(zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)) );
   }
}

// here Xhat, X will be BlockVectors w.r.t. the 4 partitioning X = (u, m, l, zl)

void ParInteriorPointSolver::lineSearch(BlockVector& X0, BlockVector& Xhat, double mu)
{
   // double tau  = max(tauMin, 1.0 - mu);
   double tau  = tauMin;
   Vector u0   = X0.GetBlock(0);
   Vector m0   = X0.GetBlock(1);
   Vector l0   = X0.GetBlock(2);
   Vector z0   = X0.GetBlock(3);
   Vector uhat = Xhat.GetBlock(0);
   Vector mhat = Xhat.GetBlock(1);
   Vector lhat = Xhat.GetBlock(2);
   Vector zhat = Xhat.GetBlock(3);
   double alphaMax  = MaxStepSize(m0, ml, mhat, tau);
   double alphaMaxz = MaxStepSize(z0, zhat, tau);
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
      phxtrial = phi(xtrial, mu);
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
            //if (iAmRoot)
            //{
            //   cout << "second order correction\n";
            //}
            //problem->c(xtrial, ckSoc);
            //problem->c(x0, ck0);
            //ckSoc.Add(alphaMax, ck0);
            //// A-5.6 Compute the second-order correction.
            //IPNewtonSolve(x0, l0, z0, zhatsoc, Xhatumlsoc, mu, true);
            //mhatsoc.Set(1.0, Xhatumlsoc.GetBlock(1));
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


void ParInteriorPointSolver::projectZ(const Vector &x, Vector &z, double mu)
{
   double zi;
   double mudivmml;
   for(int i = 0; i < dimM; i++)
   {
      zi = z(i);
      mudivmml = mu / (x(i + dimU) - ml(i));
      z(i) = max(min(zi, kSig * mudivmml), mudivmml / kSig);
   }
}

void ParInteriorPointSolver::filterCheck(double th, double ph)
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
bool ParInteriorPointSolver::CurvatureTest(const BlockOperator & A, const BlockVector & Xhat, const Vector & l, const BlockVector & b, const double & delta)
{
   Vector lplus(l.Size());
   lplus.Set(1.0, l);
   lplus.Add(1.0, Xhat.GetBlock(2));
   

   double dWd = 0.0;
   double dd = 0.0;
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
   double lplusTck = -1.0 * InnerProduct(MPI_COMM_WORLD, lplus, b.GetBlock(2));
 
   //if (iAmRoot)
   //{ 
   //   cout << "d^T W d + max{-(l+)^T c, 0} = " << dWd + fmax(-lplusTck, 0.0) << endl;
   //   cout << "d^T d = " << dd << endl;
   //   cout << "d^T W d / d^T d = " << dWd / dd << endl;
   //}
   bool passed = (dWd + fmax(-lplusTck, 0.0) >= alphaCurvatureTest * dd) ? true : false;
   return passed;
}






double ParInteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, double mu, bool printEeval)
{
   double E1, E2, E3;
   double sc, sd;
   BlockVector gradL(block_offsetsx); gradL = 0.0; // stationarity grad L = grad f + J^T l - z
   Vector cx(dimC); cx = 0.0;     // feasibility c = c(x)
   Vector comp(dimM); comp = 0.0; // complementarity M Z - mu 1

   DxL(x, l, zl, gradL);
   E1 = GlobalLpNorm(infinity(), gradL.Normlinf(), MPI_COMM_WORLD); 

   problem->c(x, cx);
   E2 = GlobalLpNorm(infinity(), cx.Normlinf(), MPI_COMM_WORLD); 


   for(int ii = 0; ii < dimM; ii++) 
   { 
      comp(ii) = x(dimU + ii) * zl(ii) - mu;
   }
   E3 = GlobalLpNorm(infinity(), comp.Normlinf(), MPI_COMM_WORLD); 

   double ll1, zl1;

   zl1 = GlobalLpNorm(1, zl.Norml1(), MPI_COMM_WORLD)/ double(gdimC + gdimM);; 
   ll1 = GlobalLpNorm(1, l.Norml1(), MPI_COMM_WORLD);
   sc = max(sMax, zl1 / (double(gdimM)) ) / sMax;
   sd = max(sMax, (ll1 + zl1) / (double(gdimC + gdimM))) / sMax;
   if(iAmRoot && printEeval)
   {
      cout << "evaluating optimality error for mu = " << mu << endl;
      cout << "stationarity measure = "    << E1 / sd << endl;
      cout << "feasibility measure  = "    << E2      << endl;
      cout << "complimentarity measure = " << E3 / sc << endl;
   }
   return max(max(E1 / sd, E2), E3 / sc);
}

double ParInteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, bool printEeval)
{
  return E(x, l, zl, 0.0, printEeval);
}

double ParInteriorPointSolver::theta(const BlockVector &x)
{
   Vector cx(dimC);
   problem->c(x, cx);
   return sqrt(InnerProduct(MPI_COMM_WORLD,cx, cx));
}

// log-barrier objective
double ParInteriorPointSolver::phi(const BlockVector &x, double mu)
{
   double fx = problem->CalcObjective(x); 
   double logBarrierLoc = 0.0;
   for(int i = 0; i < dimM; i++) 
   { 
     logBarrierLoc += log(x(dimU+i)-ml(i));
   }
   double logBarrierGlb;
   MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   return fx - mu * logBarrierGlb;
}

// gradient of log-barrier objective with respect to x = (u, m)
void ParInteriorPointSolver::Dxphi(const BlockVector &x, double mu, BlockVector &y)
{
   problem->CalcObjectiveGrad(x, y);
   
   for(int i = 0; i < dimM; i++) 
   { 
      y(dimU + i) -= mu / (x(dimU + i));
   } 
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
double ParInteriorPointSolver::L(const BlockVector &x, const Vector &l, const Vector &zl)
{
   double fx = problem->CalcObjective(x);
   Vector cx(dimC); problem->c(x, cx);
   return (fx + InnerProduct(MPI_COMM_WORLD,cx, l) - InnerProduct(MPI_COMM_WORLD, x.GetBlock(1), zl));
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
   (y.GetBlock(1)).Add(-1.0, zl);
}

bool ParInteriorPointSolver::GetConverged() const
{
   return converged;
}

void ParInteriorPointSolver::SetTol(double Tol)
{
   OptTol = Tol;
}

void ParInteriorPointSolver::SetMaxIter(int max_it)
{
   max_iter = max_it;
}

void ParInteriorPointSolver::SetBarrierParameter(double mu_0)
{
   mu_k = mu_0;
}

void ParInteriorPointSolver::SaveLogBarrierHessianIterates(bool save)
{
   MFEM_ASSERT(MyRank == 0 || save == false, "currently can only save logbarrier hessian in serial codes");
   saveLogBarrierIterates = save;
}

void ParInteriorPointSolver::SetLinearSolver(int LinSolver)
{
   linSolver = LinSolver;
}

void ParInteriorPointSolver::SetLinearSolveAbsTol(double Tol)
{
  linSolveAbsTol = Tol;
}
void ParInteriorPointSolver::SetLinearSolveRelTol(double Tol)
{
  linSolveRelTol = Tol;
}

void ParInteriorPointSolver::SetLinearSolveRelaxType(int relax_type_)
{
   relax_type = relax_type_;
}


ParInteriorPointSolver::~ParInteriorPointSolver() 
{
   delete JuT;
   delete JmT;
   delete Wuu;
   delete Wmm;
}
