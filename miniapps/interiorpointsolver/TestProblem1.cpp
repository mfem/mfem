//                         Example Problem 1
//
//
// Compile with: make ParTestProblem1
//
// Sample runs: mpirun -np 4 ./ParTestProblem1
//
//
// Description: This example code demonstrates the use of the MFEM based
//              interior-point solver to solve the
//              bound-constrained minimization problem
//
//              minimize_(x \in R^n) 1/2 x^T x subject to x - xl ≥ 0 (component-wise).
//              
#include "mfem.hpp"
#include "Problem.hpp"
#include "IPsolver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;




// solve min 1/2 x^T K x s.t. J x - xl >= 0
// where K and J are identity matrices and xl
// has uniform random values in [-1, 1]
// for the Lagrangian L(x, s, l, z) = 1/2 x^T x + l^T (x - xl - s) - z^T s
// the optimal solution is x*_i = max{0, (xl)_i}, z*_i = x*_i


class ParEx1Problem : public ParOptProblem
{
protected:
   HypreParMatrix *K;
   HypreParMatrix *J;
   Vector xl;
   //HYPRE_BigInt * dofOffsets;
public:
   // create offsets internally only pass problem size
   //ParEx1Problem(HYPRE_BigInt * offsets);
   ParEx1Problem(int n);
   double E(const Vector &) const;
   void DdE(const Vector &, Vector &) const;
   HypreParMatrix* DddE(const Vector &);
   void g(const Vector &, Vector &) const;
   HypreParMatrix* Ddg(const Vector &);
   virtual ~ParEx1Problem();
};



void mfemIPSolve(ParGeneralOptProblem & problem, Vector &x, Vector &lambda)
{
   ParInteriorPointSolver IPoptimizer(&problem);

   int dimX = problem.GetDimU();
   Vector x0(dimX); x0 = 100.0;
   x.SetSize(dimX); x = 0.0;
   
   double OptTol = 1.e-6;
   double LinSolveTol = 1.e-10;
   int linSolveStrategy = 2;
   int MaxOptIter = 30; 
   IPoptimizer.SetTol(OptTol);
   IPoptimizer.SetLinearSolveTol(LinSolveTol);
   IPoptimizer.SetLinearSolver(linSolveStrategy);
   IPoptimizer.SetMaxIter(MaxOptIter);
   IPoptimizer.Mult(x0, x);
   
   int dimM = problem.GetDimM();
   lambda.SetSize(dimM);
   IPoptimizer.GetLagrangeMultiplier(lambda);
}






int main(int argc, char *argv[])
{
  // Initialize MPI
   Mpi::Init();
   Hypre::Init();   

   int n = 10;
   OptionsParser args(argc, argv);
   args.AddOption(&n, "-n", "--n", \
		   "Size of the optimization problem (dimension of primal variable)");
   args.ParseCheck();
   
   
   ParEx1Problem problem(n);
   
   Vector xOptimal, lambdaOptimal;
   mfemIPSolve(problem, xOptimal, lambdaOptimal);
   for(int i = 0; i < xOptimal.Size(); i++)
   {
     cout << "optimal (x, z)_" << i << " = (" << xOptimal(i) << ", " << lambdaOptimal(i) << ")\n";
   }

   Mpi::Finalize();
   return 0;
}


// Ex1Problem
// min 1/2 x^T K x such that J x - xl >= 0
// where K and J are identity matrices
ParEx1Problem::ParEx1Problem(int n) : ParOptProblem(), K(nullptr), J(nullptr)
{
  // generate the parallel partition of the 
  // variable x and the 
  int nprocs = Mpi::WorldSize();
  int myrank = Mpi::WorldRank();
  
  
  HYPRE_BigInt * dofOffsets = new HYPRE_BigInt[2];
  dofOffsets[0] = HYPRE_BigInt(myrank * n / nprocs);
  dofOffsets[1] = HYPRE_BigInt((myrank + 1) * n / nprocs);
  
  Init(dofOffsets, dofOffsets);

  Vector iDiag(dofOffsets[1] - dofOffsets[0]); iDiag = 1.0;
  
  K = GenerateHypreParMatrixFromDiagonal(dofOffsets, iDiag);

  J = GenerateHypreParMatrixFromDiagonal(dofOffsets, iDiag);

  xl.SetSize(dofOffsets[1] - dofOffsets[0]);
  xl.Randomize(myrank);
  xl *= 2.0;
  xl -= 1.0;
  delete[] dofOffsets;
}



double ParEx1Problem::E(const Vector & x) const
{
   Vector Kx(K->Height()); Kx = 0.0;
   MFEM_VERIFY(x.Size() == K->Width(), "ParEx1Problem::E - Inconsistent dimensions");
   K->Mult(x, Kx);
   return 0.5 * InnerProduct(MPI_COMM_WORLD, x, Kx);
}

void ParEx1Problem::DdE(const Vector &x, Vector &gradE) const
{
   gradE.SetSize(K->Height());
   MFEM_VERIFY(x.Size() == K->Width(), "ParEx1Problem::DdE - Inconsistent dimensions");
   K->Mult(x, gradE);
}

HypreParMatrix * ParEx1Problem::DddE(const Vector &x)
{
   return K; 
}

// g(x) = x - xl >= 0
void ParEx1Problem::g(const Vector &x, Vector &gx) const
{
   MFEM_VERIFY(x.Size() == J->Width(), "ParEx1Problem::g - Inconsistent dimensions");
   J->Mult(x, gx);
   MFEM_VERIFY(gx.Size() == J->Height(), "ParEx1Problem::g - Inconsistent dimensions");
   gx.Add(-1.0, xl);
}

HypreParMatrix * ParEx1Problem::Ddg(const Vector &)
{
  return J;
}

ParEx1Problem::~ParEx1Problem()
{
   delete K;
   delete J;
}

