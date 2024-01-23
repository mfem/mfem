#include "mfem.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace mfem;


InteriorPointSolver::InteriorPointSolver(GeneralOptProblem * problem_) 
                     : problem(problem_), 
                       block_offsetsumlz(5), block_offsetsuml(4), block_offsetsx(3),
                       saveLogBarrierIterates(false)
{
   OptTol  = 1.e-2;
   max_iter = 20;
   mu_k     = 1.0;

   sMax     = 1.e2;
   kSig     = 1.e10;   // control deviation from primal Hessian
   tauMin   = 0.8;     // control rate at which iterates can approach the boundary
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

   dimU = problem->GetDimU();
   dimM = problem->GetDimM();
   dimC = problem->GetDimC();
   ckSoc.SetSize(dimC);
  
   block_offsetsumlz[0] = 0;
   block_offsetsumlz[1] = dimU; // u
   block_offsetsumlz[2] = dimM; // m
   block_offsetsumlz[3] = dimC; // lambda
   block_offsetsumlz[4] = dimM; // zl
   block_offsetsumlz.PartialSum();
  
   #ifdef MFEM_USE_MPI 
      dimUGlb = problem->GetDimUGlb();
      dimMGlb = problem->GetDimMGlb();
      dimCGlb = problem->GetDimCGlb(); 
   #else
      dimUGlb = dimU;
      dimMGlb = dimM;
      dimCGlb = dimC;
   #endif

   for(int i = 0; i < block_offsetsuml.Size(); i++)  
   { 
      block_offsetsuml[i] = block_offsetsumlz[i]; 
   }
   for(int i = 0; i < block_offsetsx.Size(); i++)    
   { 
      block_offsetsx[i] = block_offsetsuml[i]; 
   }

  
   ml = problem->Getml();
  
   lk.SetSize(dimC);  lk  = 0.0;
   zlk.SetSize(dimM); zlk = 0.0;

   linSolver = 0;
   linSolveTol = 1.e-8;

   parallel = problem->IsParallel();
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MyRank = Mpi::WorldRank();
   }
   else
   {
#endif
   MyRank = 0;
#ifdef MFEM_USE_MPI
   }
#endif
   iAmRoot = MyRank == 0 ? true : false;


   Huu = nullptr;
   Hum = nullptr;
   Hmu = nullptr;
   Hmm = nullptr;
   Wmm = nullptr;
   D   = nullptr;
   Ju  = nullptr;
   Jm  = nullptr;
   JuT = nullptr;
   JmT = nullptr;
}

double InteriorPointSolver::MaxStepSize(Vector &x, Vector &xl, Vector &xhat, double tau)
{
   double alphaMaxLoc = 1.0;
   double alphaTmp;
   for(int i = 0; i < x.Size(); i++)
   {   
      if( xhat(i) < 0. )
      {
         alphaTmp = -1. * tau * (x(i) - xl(i)) / xhat(i);
         alphaMaxLoc = min(alphaMaxLoc, alphaTmp);
      } 
   }

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      double alphaMaxGlb;
      MPI_Allreduce(&alphaMaxLoc, &alphaMaxGlb, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      return alphaMaxGlb;
   }
   else
   {
#endif
   return alphaMaxLoc;
#ifdef MFEM_USE_MPI
      }
#endif
}

double InteriorPointSolver::MaxStepSize(Vector &x, Vector &xhat, double tau)
{
   Vector zero(x.Size()); zero = 0.0;
   return MaxStepSize(x, zero, xhat, tau);
}


void InteriorPointSolver::Mult(const Vector &x0, Vector &xf)
{
   BlockVector x0block(block_offsetsx); x0block = 0.0;
   x0block.GetBlock(0).Set(1.0, x0);
   if(dimM > 0 ) { x0block.GetBlock(1) = 100.;}
   x0block.GetBlock(1).Add(1.0, ml);
   BlockVector xfblock(block_offsetsx); xfblock = 0.0;
   Mult(x0block, xfblock);
   xf.Set(1.0, xfblock.GetBlock(0));
}


void InteriorPointSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
   converged = false;
   
   BlockVector xk(block_offsetsx), xhat(block_offsetsx); xk = 0; xhat = 0.0;
   BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz); Xk = 0.0; Xhat = 0.0;
   BlockVector Xhatuml(block_offsetsuml); Xhatuml = 0.0;
   Vector zlhat(dimM); zlhat = 0.0;

   xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
   xk.GetBlock(1).Set(1.0, x0.GetBlock(1));
   
   // running estimate of the final values of the Lagrange multipliers
   lk = 0.0;
   zlk = 0.0;

   for(int i = 0; i < dimM; i++)
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
   thetaMax = 1.e4  * max(1.0, theta0);

   double Eeval, maxBarrierSolves, Eevalmu0;
   bool printOptimalityError; // control optimality error print to console for log-barrier subproblems
   
   maxBarrierSolves = 10;

   for(jOpt = 0; jOpt < max_iter; jOpt++)
   {
      if(iAmRoot)
      {
         cout << "interior-point solve step " << jOpt << endl;
      }
      // A-2. Check convergence of overall optimization problem
      printOptimalityError = false;
      Eevalmu0 = E(xk, lk, zlk, printOptimalityError);
      if(Eevalmu0 < OptTol)
      {
         converged = true;
	 if(iAmRoot)
         {
            cout << "solved optimization problem!\n";
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
               cout << "solved barrier subproblem, for mu = " << mu_k << endl;
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
         cout << "\n** IP-Newton solve **\n";
      }
      zlhat = 0.0; Xhatuml = 0.0;
      IPNewtonSolve(xk, lk, zlk, zlhat, Xhatuml, mu_k, false); 

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
         cout << "\n** Linesearch **\n";
         cout << "mu = " << mu_k << endl;
      }
      lineSearch(Xk, Xhat, mu_k);

      if(lineSearchSuccess)
      {
         if(iAmRoot)
         {
            cout << "lineSearch success\n";
         }
         if(!switchCondition || !sufficientDecrease)
         {
            F1.Append( (1. - gTheta) * thx0);
            F2.Append( phx0 - gPhi * thx0);
         }
         // ----- A-6: Accept the trial point
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
            cout << "lineSearch not successful\n";
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

//         IP-Newton system matrix
//    Ak = [[H_(u,u)  H_(u,m)   J_u^T]
//          [H_(m,u)  W_(m,m)   J_m^T]
//          [ J_u      J_m       0  ]]
void InteriorPointSolver::FormIPNewtonMat(BlockVector & x, Vector & l, Vector &zl, BlockOperator &Ak)
{
   Vector DiagLogBar(dimM); DiagLogBar = 0.0;
   for(int ii = 0; ii < dimM; ii++)
   {
      DiagLogBar(ii) = zl(ii) / (x(ii+dimU) - ml(ii));
   }
   if(saveLogBarrierIterates)
   {
      std::ofstream diagStream;
      char diagString[100];
      snprintf(diagString, 100, "logBarrierHessiandata/D%d.dat", jOpt);
      diagStream.open(diagString, ios::out | ios::trunc);
      for(int ii = 0; ii < dimM; ii++)
      {
         diagStream << setprecision(30) << DiagLogBar(ii) << endl;
      }
      diagStream.close();
   } 

   #ifdef MFEM_USE_MPI
      if (parallel)
      {
	 SparseMatrix * DSparse = new SparseMatrix(DiagLogBar);
         Dh = new HypreParMatrix(MPI_COMM_WORLD, dimMGlb, problem->GetDofOffsetsM(), DSparse);
         HypreStealOwnership(*Dh, *DSparse);
	 Huuh = dynamic_cast<HypreParMatrix *>(problem->Duuf(x)); 
         Humh = dynamic_cast<HypreParMatrix *>(problem->Dumf(x));
         Hmuh = dynamic_cast<HypreParMatrix *>(problem->Dmuf(x));
         Hmmh = dynamic_cast<HypreParMatrix *>(problem->Dmmf(x));
         
	 if(Hmmh != nullptr)
	 {
	    Wmmh = Hmmh;
	    Wmmh->Add(1.0, *Dh);
	 }
	 else
	 {
	    Wmmh = Dh;
	 }
         Juh = dynamic_cast<HypreParMatrix *>(problem->Duc(x)); 
	 Jmh = dynamic_cast<HypreParMatrix *>(problem->Dmc(x));
         JuTh = Juh->Transpose();
         JmTh = Jmh->Transpose();
         Ak.SetBlock(0, 0, Huuh);                         Ak.SetBlock(0, 2, JuTh);
                                  Ak.SetBlock(1, 1, Wmmh); Ak.SetBlock(1, 2, JmTh);
         Ak.SetBlock(2, 0,  Juh); Ak.SetBlock(2, 1,  Jmh);

         if(Humh != nullptr) { Ak.SetBlock(0, 1, Humh); Ak.SetBlock(1, 0, Hmuh); }
      }
      else
      {
   #endif
   
   D = new SparseMatrix(DiagLogBar);
   
   Huu = dynamic_cast<SparseMatrix *>(problem->Duuf(x)); 
   Hum = dynamic_cast<SparseMatrix *>(problem->Dumf(x));
   Hmu = dynamic_cast<SparseMatrix *>(problem->Dmuf(x));
   Hmm = dynamic_cast<SparseMatrix *>(problem->Dmmf(x));
   
   if(Hmm != nullptr)
   {
      Wmm = Hmm;
      Wmm->Add(1.0, *D);
   }
   else
   {
      Wmm = D;
   }
   
   Ju = dynamic_cast<SparseMatrix *>(problem->Duc(x)); 
   Jm = dynamic_cast<SparseMatrix *>(problem->Dmc(x));
   JuT = Transpose(*Ju);
   JmT = Transpose(*Jm);
   
   Ak.SetBlock(0, 0, Huu);                         Ak.SetBlock(0, 2, JuT);
                           Ak.SetBlock(1, 1, Wmm); Ak.SetBlock(1, 2, JmT);
   Ak.SetBlock(2, 0,  Ju); Ak.SetBlock(2, 1,  Jm);

   if(Hum != nullptr) { Ak.SetBlock(0, 1, Hum); Ak.SetBlock(1, 0, Hmu); }

   #ifdef MFEM_USE_MPI
      }
   #endif
  

}

// perturbed KKT system solve
// determine the search direction
void InteriorPointSolver::IPNewtonSolve(BlockVector &x, Vector &l, Vector &zl, Vector &zlhat, BlockVector &Xhat, double mu, bool socSolve)
{
   // solve A x = b, where A is the IP-Newton matrix
   BlockOperator A(block_offsetsuml, block_offsetsuml); BlockVector b(block_offsetsuml); b = 0.0;
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

   #ifdef MFEM_USE_MPI
      // form A = Huu + Ju^T D Ju, Wmm = D for contact
      if (parallel)
      {
         HypreParMatrix * Huuloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 0)));
	 HypreParMatrix * Wmmloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(1, 1)));
         HypreParMatrix * Juloc  = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(2, 0)));
         HypreParMatrix * JuTloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 2)));
         
         HypreParMatrix *JuTDJu   = RAP(Wmmloc, Juloc);     // Ju^T D Ju
         HypreParMatrix *Areduced = ParAdd(Huuloc, JuTDJu);  // Huu + Ju^T D Ju
         /* prepare the reduced rhs */
         // breduced = bu + Ju^T (bm + Wmm bl)
         Vector breduced(dimU); breduced = 0.0;
         Vector tempVec(dimM); tempVec = 0.0;
         Wmmloc->Mult(b.GetBlock(2), tempVec);
         tempVec.Add(1.0, b.GetBlock(1));
         JuTloc->Mult(tempVec, breduced);
         breduced.Add(1.0, b.GetBlock(0));
         
         // CG-AMG on the "reduced" linear system
         HyprePCG AreducedSolver(MPI_COMM_WORLD);
         AreducedSolver.SetOperator(*Areduced);
         HypreBoomerAMG AreducedPrec;
         AreducedSolver.SetTol(linSolveTol);
         AreducedSolver.SetMaxIter(500);
         AreducedSolver.SetPreconditioner(AreducedPrec);
         AreducedSolver.SetResidualConvergenceOptions(); // convergence criteria based on residual norm
         AreducedSolver.SetPrintLevel(2);
         AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
         
         // back-propagate reduced-solve info to determine mhat and lhat
         // xm = Ju xu - bl
         Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
         Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

         // xl = Wmm xm - bm
         Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
         Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

         delete JuTDJu;
         delete Areduced;
      }
      else
      {
   #endif
   #ifdef MFEM_USE_SUITESPARSE
      BlockMatrix ABlockMatrix(block_offsetsuml, block_offsetsuml);
      for(int ii = 0; ii < 3; ii++)
      {
        for(int jj = 0; jj < 3; jj++)
        {
          if(!A.IsZeroBlock(ii, jj))
          {
            ABlockMatrix.SetBlock(ii, jj, dynamic_cast<SparseMatrix *>(&(A.GetBlock(ii, jj))));
          }
        }
      }
      /* direct solve of the 3x3 IP-Newton linear system */
      UMFPackSolver ASolver;
      SparseMatrix *ASparse = ABlockMatrix.CreateMonolithic();
      ASolver.SetOperator(*ASparse);
      ASolver.Mult(b, Xhat);
      delete ASparse;
   #else
      MFEM_VERIFY(false, "SUITESPARSE must be enabled for serial runs"); 
   #endif

   #ifdef MFEM_USE_MPI
      }
   #endif

   //// Direct solver (default)
   //if(linSolver == 0)
   //{
   //   Array2D<HypreParMatrix *> ABlockMatrix(3,3);
   //   for(int ii = 0; ii < 3; ii++)
   //   {
   //   for(int jj = 0; jj < 3; jj++)
   //   {
   //      if(!A.IsZeroBlock(ii, jj))
   //      {
   //         ABlockMatrix(ii, jj) = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(ii, jj)));
   //      }
   //      else
   //      {
   //         ABlockMatrix(ii, jj) = nullptr;
   //      }
   //   }
   //   }
   //   
   //   HypreParMatrix * Ah = HypreParMatrixFromBlocks(ABlockMatrix);   
   //   
   //   /* direct solve of the 3x3 IP-Newton linear system */
   //   #ifdef MFEM_USE_MUMPS
   //     MUMPSSolver ASolver;
   //     ASolver.SetPrintLevel(0);
   //     ASolver.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
   //     ASolver.SetOperator(*Ah);
   //     ASolver.Mult(b, Xhat);
   //   #else 
   //     #ifdef MFEM_USE_MKL_CPARDISO
   //       CPardisoSolver ASolver(MPI_COMM_WORLD);
   //       ASolver.SetOperator(*Ah);
   //       ASolver.Mult(b, Xhat);
   //     #else
   //       MFEM_VERIFY(false, "linSolver 0 will not work unless compiled with MUMPS or MKL");
   //     #endif
   //   #endif

   //   delete Ah;
   //}
   //else if(linSolver == 1 || linSolver == 2)
   //{
   //   // form A = Huu + Ju^T D Ju, Wmm = D for contact
   //   HypreParMatrix * Huuloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 0)));
   //   HypreParMatrix * Wmmloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(1, 1)));
   //   HypreParMatrix * Juloc  = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(2, 0)));
   //   HypreParMatrix * JuTloc = dynamic_cast<HypreParMatrix *>(&(A.GetBlock(0, 2)));
   //   
   //   
   //   HypreParMatrix *JuTDJu   = RAP(Wmmloc, Juloc);     // Ju^T D Ju
   //   HypreParMatrix *Areduced = ParAdd(Huuloc, JuTDJu);  // Huu + Ju^T D Ju
   //   /* prepare the reduced rhs */
   //   // breduced = bu + Ju^T (bm + Wmm bl)
   //   Vector breduced(dimU); breduced = 0.0;
   //   Vector tempVec(dimM); tempVec = 0.0;
   //   Wmmloc->Mult(b.GetBlock(2), tempVec);
   //   tempVec.Add(1.0, b.GetBlock(1));
   //   JuTloc->Mult(tempVec, breduced);
   //   breduced.Add(1.0, b.GetBlock(0));
   //   
   //   if(linSolver == 1)
   //   {
   //      // setup the solver for the reduced linear system
   //      #ifdef MFEM_USE_MUMPS
   //        MUMPSSolver AreducedSolver;   
   //        AreducedSolver.SetPrintLevel(0);
   //        AreducedSolver.SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_INDEFINITE);
   //        AreducedSolver.SetOperator(*Areduced);
   //        AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
   //      #else 
   //        #ifdef MFEM_USE_MKL_CPARDISO
   //          CPardisoSolver AreducedSolver(MPI_COMM_WORLD);
   //          AreducedSolver.SetOperator(*Areduced);
   //          AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
   //        #else
   //          MFEM_VERIFY(false, "linSolver 1 will not work unless compiled with MUMPS or MKL");
   //        #endif
   //      #endif
   //   }
   //   else
   //   {
   //      HyprePCG AreducedSolver(MPI_COMM_WORLD);
   //      AreducedSolver.SetOperator(*Areduced);
   //      HypreBoomerAMG AreducedPrec;
   //      AreducedSolver.SetTol(linSolveTol);
   //      AreducedSolver.SetMaxIter(500);
   //      AreducedSolver.SetPreconditioner(AreducedPrec);
   //      AreducedSolver.SetResidualConvergenceOptions(); // convergence criteria based on residual norm
   //      AreducedSolver.SetPrintLevel(2);
   //      AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
   //   }

   //   // now propagate solved uhat to obtain mhat and lhat
   //   // xm = Ju xu - bl
   //   Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
   //   Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

   //   // xl = Wmm xm - bm
   //   Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
   //   Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

   //   delete JuTDJu;
   //   delete Areduced;
   //}

   /* backsolve to determine zlhat */
   for(int ii = 0; ii < dimM; ii++)
   {
      zlhat(ii) = -1.*(zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)) );
   }

   // free memory
   #ifdef MFEM_USE_MPI
   if (parallel)
   {
      delete D;
      delete JuT;
      delete JmT;
      if(Hmm != nullptr)
      {
         delete Wmm;
      }
   }
   else
   {
   #endif
   delete D;
   delete JuT;
   delete JmT;
   if(Hmm != nullptr)
   {
      delete Wmm;
   }
   #ifdef MFEM_USE_MPI
      }
   #endif

   
}

// here Xhat, X will be BlockVectors w.r.t. the 4 partitioning X = (u, m, l, zl)

void InteriorPointSolver::lineSearch(BlockVector& X0, BlockVector& Xhat, double mu)
{
   double tau  = max(tauMin, 1.0 - mu);
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

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Dxphi0_xhat = InnerProduct(MPI_COMM_WORLD, Dxphi0, xhat);
   }
   else
   {
#endif
   Dxphi0_xhat = InnerProduct(Dxphi0, xhat);
#ifdef MFEM_USE_MPI
   }
#endif
   
   
   descentDirection = Dxphi0_xhat < 0. ? true : false;
   if(descentDirection)
   {
      if (iAmRoot)
      {
         cout << "is a descent direction for the log-barrier objective\n";
      }
   }
   else
   {
      cout << "is not a descent direction for the log-barrier objective\n";
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
      }
      else
      {
         if (iAmRoot)
         {
            cout << "in filter region :(\n"; 
         }
      }
      alpha *= 0.5;

   } 
}


void InteriorPointSolver::projectZ(const Vector &x, Vector &z, double mu)
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

void InteriorPointSolver::filterCheck(double th, double ph)
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

double InteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, double mu, bool printEeval)
{
   double E1, E2, E3;
   double sc, sd;
   BlockVector gradL(block_offsetsx); gradL = 0.0; // stationarity grad L = grad f + J^T l - z
   Vector cx(dimC); cx = 0.0;     // feasibility c = c(x)
   Vector comp(dimM); comp = 0.0; // complementarity M Z - mu 1

   // optimality residuals
   DxL(x, l, zl, gradL);
   problem->c(x, cx);
   for(int ii = 0; ii < dimM; ii++) 
   { 
      comp(ii) = x(dimU + ii) * zl(ii) - mu;
   }

   double ll1, zl1; // 1-norm of dual-variables

   // norms of optimality residuals and scalings
#ifdef MFEM_USE_MPI
   if (parallel)
   { 
      E1 = GlobalLpNorm(infinity(), gradL.Normlinf(), MPI_COMM_WORLD);
      E2 = GlobalLpNorm(infinity(), cx.Normlinf(), MPI_COMM_WORLD);
      E3 = GlobalLpNorm(infinity(), comp.Normlinf(), MPI_COMM_WORLD);
      ll1 = GlobalLpNorm(1,  l.Norml1(), MPI_COMM_WORLD);
      zl1 = GlobalLpNorm(1, zl.Norml1(), MPI_COMM_WORLD); 
   }
   else
   {
#endif
   E1 = gradL.Normlinf();
   E2 = cx.Normlinf();
   E3 = comp.Normlinf();
   ll1 = l.Norml1();
   zl1 = zl.Norml1();
#ifdef MFEM_USE_MPI
   }
#endif
   sc = max(sMax, zl1 / (double(dimMGlb)) ) / sMax;
   sd = max(sMax, (ll1 + zl1) / (double(dimCGlb + dimMGlb))) / sMax;
   if(iAmRoot && printEeval)
   {
      cout << "evaluating optimality error for mu = " << mu << endl;
      cout << "stationarity measure = "    << E1 / sd << endl;
      cout << "feasibility measure  = "    << E2      << endl;
      cout << "complimentarity measure = " << E3 / sc << endl;
   }
   return max(max(E1 / sd, E2), E3 / sc);
}

double InteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, bool printEeval)
{
  return E(x, l, zl, 0.0, printEeval);
}

double InteriorPointSolver::theta(const BlockVector &x)
{
  Vector cx(dimC); cx = 0.0;
  problem->c(x, cx);
#ifdef MFEM_USE_MPI
  if (parallel)
  {
     return GlobalLpNorm(2, cx.Norml2(), MPI_COMM_WORLD);
  }
  else
  {
#endif
  return cx.Norml2();
#ifdef MFEM_USE_MPI
     }
#endif
}

// log-barrier objective
double InteriorPointSolver::phi(const BlockVector &x, double mu)
{
   double fx = problem->CalcObjective(x); 
   double logBarrierLoc = 0.0;
   for(int i = 0; i < dimM; i++) 
   { 
     logBarrierLoc += log(x(dimU+i) - ml(i));
   }
#ifdef MFEM_USE_MPI
   if(parallel)
   {
      double logBarrierGlb;
      MPI_Allreduce(&logBarrierLoc, &logBarrierGlb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return fx - mu * logBarrierGlb;
   }
   else
   {
#endif
   return fx - mu * logBarrierLoc;
#ifdef MFEM_USE_MPI
   }
#endif
}

// gradient of log-barrier objective with respect to x = (u, m)
void InteriorPointSolver::Dxphi(const BlockVector &x, double mu, BlockVector &y)
{
   problem->CalcObjectiveGrad(x, y);
   
   for(int i = 0; i < dimM; i++) 
   { 
      y(dimU + i) -= mu / (x(dimU + i));
   } 
}

// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
double InteriorPointSolver::L(const BlockVector &x, const Vector &l, const Vector &zl)
{
   double fx = problem->CalcObjective(x);
   Vector cx(dimC); problem->c(x, cx);
#ifdef MFEM_USE_MPI
   if(parallel)
   {
      return (fx + InnerProduct(MPI_COMM_WORLD, cx, l) - InnerProduct(MPI_COMM_WORLD, x.GetBlock(1), zl));
   }
   else
   {
#endif
   return (fx + InnerProduct(cx, l) - InnerProduct(x.GetBlock(1), zl));
#ifdef MFEM_USE_MPI
   }
#endif
}

void InteriorPointSolver::DxL(const BlockVector &x, const Vector &l, const Vector &zl, BlockVector &y)
{
   // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
   BlockVector gradxf(block_offsetsx); gradxf = 0.0;
   problem->CalcObjectiveGrad(x, gradxf);
   
   #ifdef MFEM_USE_MPI
      if (parallel)
      {
         HypreParMatrix * Jacu, * Jacm, * JacuT, * JacmT;
	 Jacu = dynamic_cast<HypreParMatrix *>(problem->Duc(x));
	 Jacm = dynamic_cast<HypreParMatrix *>(problem->Dmc(x));
	 JacuT = Jacu->Transpose();
	 JacmT = Jacm->Transpose();
         JacuT->Mult(l, y.GetBlock(0));
         JacmT->Mult(l, y.GetBlock(1));
         
         delete JacuT;
         delete JacmT;
      }
      else
      {
   #endif
   SparseMatrix * Jacu, * Jacm, * JacuT, * JacmT;
   Jacu = dynamic_cast<SparseMatrix *>(problem->Duc(x));
   Jacm = dynamic_cast<SparseMatrix *>(problem->Dmc(x));
   JacuT = Transpose(*Jacu);
   JacmT = Transpose(*Jacm);
   JacuT->Mult(l, y.GetBlock(0));
   JacmT->Mult(l, y.GetBlock(1));
   delete JacuT;
   delete JacmT;
   #ifdef MFEM_USE_MPI
      }
   #endif 
   
   y.Add(1.0, gradxf);
   (y.GetBlock(1)).Add(-1.0, zl);
}

bool InteriorPointSolver::GetConverged() const
{
   return converged;
}

void InteriorPointSolver::SetTol(double Tol)
{
   OptTol = Tol;
}

void InteriorPointSolver::SetMaxIter(int max_it)
{
   max_iter = max_it;
}

void InteriorPointSolver::SetBarrierParameter(double mu_0)
{
   mu_k = mu_0;
}

void InteriorPointSolver::SaveLogBarrierHessianIterates(bool save)
{
   MFEM_ASSERT(MyRank == 0 || save == false, "currently can only save logbarrier hessian in serial codes");
   saveLogBarrierIterates = save;
}

void InteriorPointSolver::SetLinearSolver(int LinSolver)
{
   linSolver = LinSolver;
}

void InteriorPointSolver::SetLinearSolveTol(double Tol)
{
  linSolveTol = Tol;
}

void InteriorPointSolver::GetLagrangeMultiplier(Vector & y)
{
  y.SetSize(dimM); y = 0.;
  y.Set(1.0, zlk);
}



InteriorPointSolver::~InteriorPointSolver() 
{
   F1.DeleteAll();
   F2.DeleteAll();
   block_offsetsx.DeleteAll();
   block_offsetsumlz.DeleteAll();
   block_offsetsuml.DeleteAll();
   ml.SetSize(0);
}
