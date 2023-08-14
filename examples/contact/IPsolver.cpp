#include "mfem.hpp"
#include "IPsolver.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace mfem;



InteriorPointSolver::InteriorPointSolver(OptProblem * Problem, ParFiniteElementSpace *Vhin) : problem(Problem), block_offsetsumlz(5), block_offsetsuml(4), block_offsetsx(3),
Huu(nullptr), Hum(nullptr), Hmu(nullptr), Hmm(nullptr), Wmm(nullptr), D(nullptr), Ju(nullptr), Jm(nullptr), JuT(nullptr), JmT(nullptr), saveLogBarrierIterates(false), Vh(Vhin)
{
  tol  = 1.e-2;
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

  for(int i = 0; i < block_offsetsuml.Size(); i++)  { block_offsetsuml[i] = block_offsetsumlz[i]; }
  for(int i = 0; i < block_offsetsx.Size(); i++)    { block_offsetsx[i]   = block_offsetsuml[i] ; }

  // lower-bound for the inequality constraint m >= ml
  ml = problem->Getml();
  
  lk.SetSize(dimC);   lk  = 0.0;
  zlk.SetSize(dimM); zlk = 0.0;
  mf.SetSize(dimM);   mf = 0.0;

  linSolver = 0;
  MyRank = 0;
  iAmRoot = MyRank == 0 ? true : false;
}

double InteriorPointSolver::MaxStepSize(Vector &x, Vector &xl, Vector &xhat, double tau)
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
  alphaMaxglb = alphaMaxloc;
  return alphaMaxglb;
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
  // hard coded initialization :(
  x0block.GetBlock(1) = 1.e0;
  x0block.GetBlock(1).Add(1.0, ml);
  BlockVector xfblock(block_offsetsx); xfblock = 0.0;
  Mult(x0block, xfblock);
  xf.Set(1.0, xfblock.GetBlock(0));
  mf.Set(1.0, xfblock.GetBlock(1));
}

void InteriorPointSolver::Mult(const BlockVector &x0, BlockVector &xf)
{
  converged = false;
  IPNewtonKrylovIters.open("IPNewtonKrylovIters.dat", ios::out | ios::trunc);  
  BlockVector xk(block_offsetsx), xhat(block_offsetsx); xk = 0; xhat = 0.0;
  BlockVector Xk(block_offsetsumlz), Xhat(block_offsetsumlz); Xk = 0.0; Xhat = 0.0;
  BlockVector Xhatuml(block_offsetsuml); Xhatuml = 0.0;
  Vector zlhat(dimM); zlhat = 0.0;

  xk.GetBlock(0).Set(1.0, x0.GetBlock(0));
  xk.GetBlock(1).Set(1.0, x0.GetBlock(1));
  // running estimate of the final values of the Lagrange multipliers
  lk  = 0.0;
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
  thetaMax = 1.e8  * thetaMin;

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
    if(Eevalmu0 < tol)
    {
      converged = true;
      if(iAmRoot)
      {
	IPNewtonKrylovIters.close();
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
      if(Eeval < kEps * mu_k)
      {
        if(iAmRoot)
        {
          cout << "solved barrier subproblem :), for mu = " << mu_k << endl;
        }
        // A-3.1. Recompute the barrier parameter
        mu_k  = max(tol / 10., min(kMu * mu_k, pow(mu_k, thetaMu)));
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
    // why do we have Xhatuml ....???
    // TO DO: remove Xhatuml in favor of passing Xhat
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
      //cout << "feasibility restoration!!! :( :( :(\n";
      //problem->feasibilityRestoration(x, 1.e-12);
      //      break;
    }
    //
    if(jOpt + 1 == max_iter && iAmRoot) 
    { 
      cout << "maximum optimization iterations :(\n";
      IPNewtonKrylovIters.close();
    }
  }
  // done with optimization routine, just reassign data to xf reference so
  // that the application code has access to the optimal point
  xf = 0.0;
  xf.GetBlock(0).Set(1.0, xk.GetBlock(0));
  xf.GetBlock(1).Set(1.0, xk.GetBlock(1));
}

void InteriorPointSolver::FormIPNewtonMat(BlockVector & x, Vector & l, Vector &zl, BlockOperator &Ak)
{
  // WARNING: Huu, Hum, Hmu, Hmm should all be Hessian terms of the Lagrangian, currently we 
  //          them by Hessian terms of the objective function and neglect the Hessian of l^T c

  Huu = problem->Duuf(x); Hum = problem->Dumf(x);
  Hmu = problem->Dmuf(x); Hmm = problem->Dmmf(x);

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


  D = new SparseMatrix(DiagLogBar);
  
  if(Hmm != nullptr)
  {
    Wmm = Hmm;
    Wmm->Add(1.0, *D);
  }
  else
  {
    Wmm = D;
  }

  Ju = problem->Duc(x); JuT = Transpose(*Ju);
  Jm = problem->Dmc(x); JmT = Transpose(*Jm);
  
  //         IP-Newton system matrix
  //    Ak = [[H_(u,u)  H_(u,m)   J_u^T]
  //          [H_(m,u)  W_(m,m)   J_m^T]
  //          [ J_u      J_m       0  ]]

  Ak.SetBlock(0, 0, Huu);                         Ak.SetBlock(0, 2, JuT);
                          Ak.SetBlock(1, 1, Wmm); Ak.SetBlock(1, 2, JmT);
  Ak.SetBlock(2, 0,  Ju); Ak.SetBlock(2, 1,  Jm);

  if(Hum != nullptr) { Ak.SetBlock(0, 1, Hum); Ak.SetBlock(1, 0, Hmu); }
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


  #ifdef MFEM_USE_SUITESPARSE
    // Direct solve for IP-Newton saddle-point system
    // A = [ [ Huu   0    Ju^T]
    //       [  0    D     -I ]
    //       [ Ju   -I      0 ]]
    if(linSolver == 0)
    {
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

      Vector residual(Xhat.Size());
      ASparse->Mult(Xhat, residual);
      residual.Add(-1.0, b);
      delete ASparse;
    }
    else if(linSolver == 1)
    {
      // Direct solve for 0,0 Schur complement of IP-Newton system, Huu + Ju^T Wmm Ju,
      // where Wmm = D for contact problems
      SparseMatrix * Huuloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(0, 0))));
      SparseMatrix * Wmmloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(1, 1))));
      SparseMatrix * Juloc  = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(2, 0))));
      SparseMatrix * JuTloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(0, 2))));
      Vector D(dimM); D = 0.0;
      Vector one(dimM); one = 1.0;
      Wmmloc->Mult(one, D);
      SparseMatrix *JuTDJu = Mult_AtDA(*Juloc, D);     // Ju^T D Ju
      SparseMatrix *Areduced = Add(*Huuloc, *JuTDJu);  // Huu + Ju^T D Ju

      
      /* prepare the reduced rhs */
      // breduced = bu + Ju^T (bm + Wmm bl)
      Vector breduced(dimU); breduced = 0.0;
      Vector tempVec(dimM); tempVec = 0.0;
      Wmmloc->Mult(b.GetBlock(2), tempVec);
      tempVec.Add(1.0, b.GetBlock(1));
      JuTloc->Mult(tempVec, breduced);
      breduced.Add(1.0, b.GetBlock(0));

      // solve the reduced linear system
      UMFPackSolver AreducedSolver;
      AreducedSolver.SetOperator(*Areduced);
      AreducedSolver.Mult(breduced, Xhat.GetBlock(0));

      // now propagate solved uhat to obtain mhat and lhat
      // xm = Ju xu - bl
      Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
      Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

      // xl = Wmm xm - bm
      Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
      Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

      delete Wmmloc;
      delete Huuloc;
      delete JuTDJu;
      delete Juloc;
      delete Areduced;
    }
  #else
    MFEM_VERIFY(linSolver > 1, "linSolver = 0, 1 require MFEM_USE_SUITESPARSE=YES");
  #endif
  if (linSolver == 2 || linSolver == 3)
  { 
    // Iterative solve for 0,0 Schur complement of IP-Newton system, Huu + Ju^T Wmm Ju,
    // where Wmm = D for contact problems
    // here the iterative solver is a Jacobi-preconditioned CG-solve
    SparseMatrix * Huuloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(0, 0))));
    SparseMatrix * Wmmloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(1, 1))));
    SparseMatrix * Juloc  = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(2, 0))));
    SparseMatrix * JuTloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(0, 2))));
    Vector D(dimM); D = 0.0;
    Vector one(dimM); one = 1.0;
    Wmmloc->Mult(one, D);
    SparseMatrix *JuTDJu = Mult_AtDA(*Juloc, D);     // Ju^T D Ju
    SparseMatrix *Areduced = Add(*Huuloc, *JuTDJu);  // Huu + Ju^T D Ju

    /* prepare the reduced rhs */
    // breduced = bu + Ju^T (bm + Wmm bl)
    Vector breduced(dimU); breduced = 0.0;
    Vector tempVec(dimM); tempVec = 0.0;
    Wmmloc->Mult(b.GetBlock(2), tempVec);
    tempVec.Add(1.0, b.GetBlock(1));
    JuTloc->Mult(tempVec, breduced);
    breduced.Add(1.0, b.GetBlock(0));
    
    /* set up an iterative solver */
    int globalNumRows = dimU;
    HYPRE_BigInt rowStarts[2];
    rowStarts[0] = 0;
    rowStarts[1] = dimU;
    HypreParMatrix * Ahypre = new HypreParMatrix(MPI_COMM_WORLD, globalNumRows, rowStarts, Areduced);
    // CGSolver Asolver(MPI_COMM_WORLD);
    HyprePCG Asolver(MPI_COMM_WORLD);
    HypreBoomerAMG * Aprec = new HypreBoomerAMG(*Ahypre);
    Aprec->SetPrintLevel(0);
    if(linSolver == 3)
    {
      Aprec->SetElasticityOptions(Vh);
    }
    Aprec->SetSystemsOptions(3,false);

    Asolver.SetOperator(*Ahypre);
    Asolver.SetPrintLevel(1);
    Asolver.SetMaxIter(1000);
    // Asolver.SetResidualConvergenceOptions();
    Asolver.SetTol(1.e-6);
    Asolver.SetPreconditioner(*Aprec);
    Asolver.Mult(breduced, Xhat.GetBlock(0));
    int num_iterations;
    Asolver.GetNumIterations(num_iterations);
    cgnum_iterations.Append(num_iterations);
    int numNewtonKrylovIters = -1;
    // numNewtonKrylovIters = Asolver.GetNumIterations();
    // IPNewtonKrylovIters << numNewtonKrylovIters << endl;

    delete Aprec;
    delete Ahypre;
    
    // now propagate solved uhat to obtain mhat and lhat
    // xm = Ju xu - bl
    Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
    Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

    // xl = Wmm xm - bm
    Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
    Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

    delete Wmmloc;
    delete Huuloc;
    delete JuTDJu;
    delete Juloc;
    delete Areduced;
  }
  else if(linSolver > 2)
  { 
    // Iterative solve for 0,0 Schur complement of IP-Newton system, Huu + Ju^T Wmm Ju,
    // where Wmm = D for contact problems
    // here the iterative solver is a Jacobi-preconditioned CG-solve
    SparseMatrix * Huuloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(0, 0))));
    SparseMatrix * Wmmloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(1, 1))));
    SparseMatrix * Juloc  = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(2, 0))));
    SparseMatrix * JuTloc = new SparseMatrix(*dynamic_cast<SparseMatrix *>(&(A.GetBlock(0, 2))));
    Vector D(dimM); D = 0.0;
    Vector one(dimM); one = 1.0;
    Wmmloc->Mult(one, D);
    SparseMatrix *JuTDJu = Mult_AtDA(*Juloc, D);     // Ju^T D Ju
    SparseMatrix *Areduced = Add(*Huuloc, *JuTDJu);  // Huu + Ju^T D Ju

    /* prepare the reduced rhs */
    // breduced = bu + Ju^T (bm + Wmm bl)
    Vector breduced(dimU); breduced = 0.0;
    Vector tempVec(dimM); tempVec = 0.0;
    Wmmloc->Mult(b.GetBlock(2), tempVec);
    tempVec.Add(1.0, b.GetBlock(1));
    JuTloc->Mult(tempVec, breduced);
    breduced.Add(1.0, b.GetBlock(0));
    
    /* set up an iterative solver */
    GSSmoother AreducedPrec((SparseMatrix &)(*Areduced));
    GMRESSolver AreducedSolver;
    AreducedSolver.SetOperator(*Areduced);
    AreducedSolver.SetAbsTol(1.e-12);
    AreducedSolver.SetRelTol(1.e-8);
    AreducedSolver.SetMaxIter(500);
    AreducedSolver.SetPreconditioner(AreducedPrec);
    AreducedSolver.SetPrintLevel(1);
    AreducedSolver.Mult(breduced, Xhat.GetBlock(0));
    
    // now propagate solved uhat to obtain mhat and lhat
    // xm = Ju xu - bl
    Juloc->Mult(Xhat.GetBlock(0), Xhat.GetBlock(1));
    Xhat.GetBlock(1).Add(-1.0, b.GetBlock(2));

    // xl = Wmm xm - bm
    Wmmloc->Mult(Xhat.GetBlock(1), Xhat.GetBlock(2));
    Xhat.GetBlock(2).Add(-1.0, b.GetBlock(1));

    delete Wmmloc;
    delete Huuloc;
    delete JuTDJu;
    delete Juloc;
    delete Areduced;
  }
    
  
  /* backsolve to determine zlhat */
  for(int ii = 0; ii < dimM; ii++)
  {
    zlhat(ii) = -1.*(zl(ii) + (zl(ii) * Xhat(ii + dimU) - mu) / (x(ii + dimU) - ml(ii)) );
  }

  // free memory
  if(!(Hmm == nullptr))
  {
    delete Wmm;
  }
  delete Huu;
  delete Hum;
  delete Hmu;
  delete Hmm;
  delete Hum;
  delete D;
  delete Ju;
  delete Jm;
  delete JuT;
  delete JmT;
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
  Dxphi0_xhat = InnerProduct(Dxphi0, xhat);
  double xhat_L2norm = sqrt(InnerProduct(xhat, xhat));
  double Dxphi_L2norm = sqrt(InnerProduct(Dxphi0, Dxphi0));
  descentDirection = Dxphi0_xhat < 0. ? true : false;
  if(descentDirection)
  {
    cout << "is a descent direction for the log-barrier objective\n";
  }
  else
  {
    cout << "is not a descent direction for the log-barrier objective\n";
  }
  cout << "Dxphi^T xhat / (|| Dxphi ||_2 * || xhat ||_2) = " << Dxphi0_xhat / (xhat_L2norm * Dxphi_L2norm) << endl;
  thx0 = theta(x0);
  phx0 = phi(x0, mu);

  lineSearchSuccess = false;
  for(int i = 0; i < maxBacktrack; i++)
  {
    cout << "\n--------- alpha = " << alpha << " ---------\n";

    // ----- A-5.2. Compute trial point: xtrial = x0 + alpha_i xhat
    xtrial.Set(1.0, x0);
    xtrial.Add(alpha, xhat);

    // ------ A-5.3. if not in filter region go to A.5.4 otherwise go to A-5.5.
    thxtrial = theta(xtrial);
    phxtrial = phi(xtrial, mu);
    filterCheck(thxtrial, phxtrial);    
    if(!inFilterRegion)
    {
      cout << "not in filter region :)\n";
      // ------ A.5.4: Check sufficient decrease
      if(!descentDirection)
      {
        switchCondition = false;
      }
      else
      {
        switchCondition = (alpha * pow(abs(Dxphi0_xhat), sPhi) > delta * pow(thx0, sTheta)) ? true : false;
      }
      cout << "alpha |Dxphi(x0)^T xhat|^sPhi = " << alpha * pow(abs(Dxphi0_xhat), sPhi) << endl;
      cout << "delta * theta(x0)^sTheta = " << delta * pow(thx0, sTheta) << endl;
      cout << "theta(x0) = "     << thx0     << ", thetaMin = "                  << thetaMin             << endl;
      cout << "theta(xtrial) = " << thxtrial << ", (1-gTheta) *theta(x0) = "     << (1. - gTheta) * thx0 << endl;
      cout << "phi(xtrial) = "   << phxtrial << ", phi(x0) - gPhi *theta(x0) = " << phx0 - gPhi * thx0   << endl;
      
      // Case I      
      if(thx0 <= thetaMin && switchCondition)
      {
        sufficientDecrease = phxtrial <= phx0 + eta * alpha * Dxphi0_xhat ? true : false;
        if(sufficientDecrease)
        {
          if(iAmRoot) { cout << "A-5.4. Case I -- accepted step length.\n"; } 
          // accept the trial step
          lineSearchSuccess = true;
          break;
        }
      }
      else
      {
        if(thxtrial <= (1. - gTheta) * thx0 || phxtrial <= phx0 - gPhi * thx0)
        {
          if(iAmRoot) { cout << "A-5.4. Case II -- accepted step length.\n"; } 
          // accept the trial step
          lineSearchSuccess = true;
          break;
        }
      }
      // A-5.5: Initialize the second-order correction
      if((!(thx0 < thxtrial)) && i == 0)
      {
        cout << "second order correction\n";
        problem->c(xtrial, ckSoc);
        problem->c(x0, ck0);
        ckSoc.Add(alphaMax, ck0);
        // A-5.6 Compute the second-order correction.
        IPNewtonSolve(x0, l0, z0, zhatsoc, Xhatumlsoc, mu, true);
        mhatsoc.Set(1.0, Xhatumlsoc.GetBlock(1));
        // alphasoc = MaxStepSize(m0, ml, mhatsoc, tau);
        //WARNING: not complete but currently solver isn't entering this region
      }
    }
    else
    {
      cout << "in filter region :(\n"; 
    }

    // include more if needed
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

double InteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, double mu, bool print)
{
  double E1, E2, E3;
  double sc, sd;
  BlockVector gradL(block_offsetsx); gradL = 0.0; // stationarity grad L = grad f + J^T l - z
  Vector cx(dimC); cx = 0.0;     // feasibility c = c(x)
  Vector comp(dimM); comp = 0.0; // complementarity M Z - mu 1

  DxL(x, l, zl, gradL);
  E1 = gradL.Normlinf();

  problem->c(x, cx);
  E2 = cx.Normlinf();

  for(int ii = 0; ii < dimM; ii++) 
  { 
    comp(ii) = x(dimU + ii) * zl(ii) - mu;
  }
  E3 = comp.Normlinf();

  double ll1, zl1;
  zl1 = zl.Norml1() / double(dimC + dimM);
  ll1 =  l.Norml1();
  sc = max(sMax, zl1 / (double(dimM)) ) / sMax;
  sd = max(sMax, (ll1 + zl1) / (double(dimC + dimM))) / sMax;
  if(iAmRoot && print)
  {
    cout << "evaluating optimality error for mu = " << mu << endl;
    cout << "stationarity measure = "    << E1 / sd << endl;
    cout << "feasibility measure  = "    << E2      << endl;
    cout << "complimentarity measure = " << E3 / sc << endl;
  }
  return max(max(E1 / sd, E2), E3 / sc);
}

double InteriorPointSolver::E(const BlockVector &x, const Vector &l, const Vector &zl, bool print)
{
  return E(x, l, zl, 0.0, print);
}

double InteriorPointSolver::theta(const BlockVector &x)
{
  Vector cx(dimC); cx = 0.0;
  problem->c(x, cx);
  return sqrt(InnerProduct(cx, cx));
}

// log-barrier objective
double InteriorPointSolver::phi(const BlockVector &x, double mu)
{
  double fx = problem->CalcObjective(x); 
  double logBarrierLoc = 0.0;
  for(int i = 0; i < dimM; i++) 
  { 
    logBarrierLoc += log(x(dimU+i)-ml(i));
  }
  double logBarrierGlb = 0.0;
  logBarrierGlb = logBarrierLoc;
  return fx - mu * logBarrierGlb;
}




// gradient of log-barrier objective with respect to x = (u, m)
void InteriorPointSolver::Dxphi(const BlockVector &x, double mu, BlockVector &y)
{
  problem->CalcObjectiveGrad(x, y);
  for(int i = 0; i < dimM; i++) 
  { 
    y(dimU + i) -= mu / (x(dimU + i) - ml(i));
  } 
}


// Lagrangian function evaluation
// L(x, l, zl) = f(x) + l^T c(x) - zl^T m
double InteriorPointSolver::L(const BlockVector &x, const Vector &l, const Vector &zl)
{
  double fx = problem->CalcObjective(x);
  Vector cx(dimC); problem->c(x, cx);
  return (fx + InnerProduct(cx, l) - InnerProduct(x.GetBlock(1), zl));
}

void InteriorPointSolver::DxL(const BlockVector &x, const Vector &l, const Vector &zl, BlockVector &y)
{
  // evaluate the gradient of the objective with respect to the primal variables x = (u, m)
  BlockVector gradxf(block_offsetsx); gradxf = 0.0;
  problem->CalcObjectiveGrad(x, gradxf);
  
  SparseMatrix *Jacu, *Jacm, *JacuT, *JacmT;
  Jacu = problem->Duc(x); Jacm = problem->Dmc(x);
  JacuT = Transpose(*Jacu);
  JacmT = Transpose(*Jacm);
  JacuT->Mult(l, y.GetBlock(0));
  JacmT->Mult(l, y.GetBlock(1));
  delete Jacu; delete JacuT;
  delete Jacm; delete JacmT;
  y.Add(1.0, gradxf);
  (y.GetBlock(1)).Add(-1.0, zl);
}


bool InteriorPointSolver::GetConverged() const
{
  return converged;
}

void InteriorPointSolver::SetTol(double Tol)
{
  tol = Tol;
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



InteriorPointSolver::~InteriorPointSolver() 
{
  F1.DeleteAll();
  F2.DeleteAll();
  block_offsetsx.DeleteAll();
  block_offsetsumlz.DeleteAll();
  block_offsetsuml.DeleteAll();
  ml.SetSize(0);
}
