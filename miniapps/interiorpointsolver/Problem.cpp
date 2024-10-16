#include "mfem.hpp"
#include "Problem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



ParGeneralOptProblem::ParGeneralOptProblem() : block_offsetsx(3) {}

void ParGeneralOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }
  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  
  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}


void ParGeneralOptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y) const
{
   Duf(x, y.GetBlock(0));
   Dmf(x, y.GetBlock(1));
}

ParGeneralOptProblem::~ParGeneralOptProblem()
{
   block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0
ParOptProblem::ParOptProblem() : ParGeneralOptProblem()
{
}

void ParOptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
{
  dofOffsetsU = new HYPRE_BigInt[2];
  dofOffsetsM = new HYPRE_BigInt[2];
  for(int i = 0; i < 2; i++)
  {
    dofOffsetsU[i] = dofOffsetsU_[i];
    dofOffsetsM[i] = dofOffsetsM_[i];
  }

  dimU = dofOffsetsU[1] - dofOffsetsU[0];
  dimM = dofOffsetsM[1] - dofOffsetsM[0];
  dimC = dimM;
  
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  
  MPI_Allreduce(&dimU, &dimUglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&dimM, &dimMglb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  ml.SetSize(dimM); ml = 0.0;
  Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  Ih = GenerateHypreParMatrixFromDiagonal(dofOffsetsM, negIdentDiag);
}


double ParOptProblem::CalcObjective(const BlockVector &x) const { return E(x.GetBlock(0)); }

void ParOptProblem::Duf(const BlockVector &x, Vector &y) const { DdE(x.GetBlock(0), y); }

void ParOptProblem::Dmf(const BlockVector &x, Vector &y) const { y = 0.0; }

HypreParMatrix * ParOptProblem::Duuf(const BlockVector &x) 
{ 
   return DddE(x.GetBlock(0)); 
}

HypreParMatrix * ParOptProblem::Dumf(const BlockVector &x) { return nullptr; }

HypreParMatrix * ParOptProblem::Dmuf(const BlockVector &x) { return nullptr; }

HypreParMatrix * ParOptProblem::Dmmf(const BlockVector &x) { return nullptr; }

void ParOptProblem::c(const BlockVector &x, Vector &y) const // c(u,m) = g(u) - m 
{
   g(x.GetBlock(0), y);
   y.Add(-1.0, x.GetBlock(1));  
}

HypreParMatrix * ParOptProblem::Duc(const BlockVector &x) 
{ 
   return Ddg(x.GetBlock(0)); 
}

HypreParMatrix * ParOptProblem::Dmc(const BlockVector &x) 
{ 
   return Ih;
} 

ParOptProblem::~ParOptProblem() 
{
  delete[] dofOffsetsU;
  delete[] dofOffsetsM;
  delete Ih;
}





// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ParObstacleProblem::ParObstacleProblem(ParFiniteElementSpace *Vh_, 
                                       double (*fSource)(const Vector &),
				       double (*obstacleSource)(const Vector &)) :
                                       ParOptProblem(), Vh(Vh_), J(nullptr) 
{
   Init(Vh->GetTrueDofOffsets(), Vh->GetTrueDofOffsets());
   cout << "dimU = " << dimU;
   f.SetSize(dimU); f = 0.0;
   psi.SetSize(dimU); psi = 0.0;


   Kform = new ParBilinearForm(Vh);
   Kform->AddDomainIntegrator(new MassIntegrator);
   Kform->AddDomainIntegrator(new DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);

   FunctionCoefficient fcoeff(fSource);
   fform = new ParLinearForm(Vh);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   fform->Assemble();
   Vector F(dimU);
   fform->ParallelAssemble(F);
   f.SetSize(dimU);
   f.Set(1.0, F);

   Vector iDiag(dimU); iDiag = 1.0;
   SparseMatrix * Jacg = new SparseMatrix(iDiag);
   
   J = new HypreParMatrix(MPI_COMM_WORLD, dimUglb, dofOffsetsU, Jacg);
   HypreStealOwnership(*J, *Jacg);
   delete Jacg;
}

// Obstacle Problem, essential boundary conditions enforced
// Hessian of energy term is K (stiffness)
ParObstacleProblem::ParObstacleProblem(ParFiniteElementSpace *Vh_, 
				       double (*fSource)(const Vector &),
				       double (*obstacleSource)(const Vector &),
				       Array<int> tdof_list, Vector &xDC) : ParOptProblem(), 
	                                                                    Vh(Vh_), J(nullptr)
{
   Init(Vh->GetTrueDofOffsets(), Vh->GetTrueDofOffsets());
   f.SetSize(dimU); f = 0.0;
   psi.SetSize(dimU); psi = 0.0;
   // elastic energy functional terms	
   ess_tdof_list = tdof_list;
   Kform = new ParBilinearForm(Vh);
   Kform->AddDomainIntegrator(new DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);

   FunctionCoefficient fcoeff(fSource);
   fform = new ParLinearForm(Vh);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   fform->Assemble();
   Vector F(dimU);
   fform->ParallelAssemble(F);
   f.SetSize(dimU);
   f.Set(1.0, F);
   Kform->EliminateVDofsInRHS(ess_tdof_list, xDC, f);
   
   // obstacle constraints --  
   Vector iDiag(dimU); iDiag = 1.0;
   for(int i = 0; i < ess_tdof_list.Size(); i++)
   {
     iDiag(ess_tdof_list[i]) = 0.0;
   }
   SparseMatrix * Jacg = new SparseMatrix(iDiag);

   J = new HypreParMatrix(MPI_COMM_WORLD, dimUglb, dofOffsetsU, Jacg);
   HypreStealOwnership(*J, *Jacg);
   delete Jacg;

   FunctionCoefficient psi_fc(obstacleSource);
   ParGridFunction psi_gf(Vh);
   psi_gf.ProjectCoefficient(psi_fc);
   psi.Set(1.0, (*psi_gf.GetTrueDofs()));
   for(int i = 0; i < ess_tdof_list.Size(); i++)
   {
     psi(ess_tdof_list[i]) = xDC(ess_tdof_list[i]) - 1.e-8;
   }
}



double ParObstacleProblem::E(const Vector &d) const
{
   Vector Kd(K.Height()); Kd = 0.0;
   MFEM_VERIFY(d.Size() == K.Width(), "ParObstacleProblem::E - Inconsistent dimensions");
   K.Mult(d, Kd);
   return 0.5 * InnerProduct(MPI_COMM_WORLD, d, Kd) - InnerProduct(MPI_COMM_WORLD, f, d);
}

void ParObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
   gradE.SetSize(K.Height());
   MFEM_VERIFY(d.Size() == K.Width(), "ParObstacleProblem::DdE - Inconsistent dimensions");
   K.Mult(d, gradE);
   MFEM_VERIFY(f.Size() == K.Height(), "ParObstacleProblem::DdE - Inconsistent dimensions");
   gradE.Add(-1.0, f);
}

HypreParMatrix * ParObstacleProblem::DddE(const Vector &d)
{
   return &K; 
}

// g(d) = d >= \psi
void ParObstacleProblem::g(const Vector &d, Vector &gd) const
{
   MFEM_VERIFY(d.Size() == J->Width(), "ParObstacleProblem::g - Inconsistent dimensions");
   J->Mult(d, gd);
   MFEM_VERIFY(gd.Size() == J->Height(), "ParObstacleProblem::g - Inconsistent dimensions");
   gd.Add(-1.0, psi);
}

HypreParMatrix * ParObstacleProblem::Ddg(const Vector &d)
{
   return J;
}

ParObstacleProblem::~ParObstacleProblem()
{
   delete Kform;
   delete fform;
   delete J;
}



ReducedProblem::ReducedProblem(ParOptProblem * problem_, HYPRE_Int * constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  int nprocs = Mpi::WorldSize();
  int myrank = Mpi::WorldRank();

  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);


  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsetsReduced_" << i << " = " << constraintOffsets_reduced[i] << ", (rank = " << myrank << ")\n";
  }

  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsets_" << i << " = " << constraintOffsets[i] << ", (rank = " << myrank << ")\n";
  }
  
  
  P = GenerateProjector(constraintOffsets, constraintOffsets_reduced, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

ReducedProblem::ReducedProblem(ParOptProblem * problem_, HypreParVector & constraintMask)
{
  problem = problem_;
  J = nullptr;
  P = nullptr;
  
  int nprocs = Mpi::WorldSize();
  int myrank = Mpi::WorldRank();

  HYPRE_BigInt * dofOffsets = problem->GetDofOffsetsU();

  // given a constraint mask, lets update the constraintOffsets
  // from the original problem
  int nLocConstraints = 0;
  int nProblemConstraints = problem->GetDimM();
  for (int i = 0; i < nProblemConstraints; i++)
  {
    if (constraintMask[i] == 1)
    {
      nLocConstraints += 1;
    }
  }
  cout << "nLocConstraints = " << nLocConstraints << ", (rank = " << myrank << ")\n";

  HYPRE_BigInt * constraintOffsets_reduced;
  constraintOffsets_reduced = offsetsFromLocalSizes(nLocConstraints);


  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsetsReduced_" << i << " = " << constraintOffsets_reduced[i] << ", (rank = " << myrank << ")\n";
  }

  HYPRE_BigInt * constraintOffsets;
  constraintOffsets = offsetsFromLocalSizes(nProblemConstraints);
  for (int i = 0; i < 2; i++)
  {
    cout << "constraintOffsets_" << i << " = " << constraintOffsets[i] << ", (rank = " << myrank << ")\n";
  }
  
  
  P = GenerateProjector(constraintOffsets, constraintOffsets_reduced, constraintMask);

  Init(dofOffsets, constraintOffsets_reduced);
  delete[] constraintOffsets_reduced;
  delete[] constraintOffsets;
}

// energy objective E(d)
double ReducedProblem::E(const Vector &d) const
{
  return problem->E(d);
}


// gradient of energy objective
void ReducedProblem::DdE(const Vector &d, Vector & gradE) const
{
  problem->DdE(d, gradE);
}


HypreParMatrix * ReducedProblem::DddE(const Vector &d)
{
  return problem->DddE(d);
}

void ReducedProblem::g(const Vector &d, Vector &gd) const
{
  Vector gdfull(problem->GetDimM()); gdfull = 0.0;
  problem->g(d, gdfull);
  P->Mult(gdfull, gd);
}


HypreParMatrix * ReducedProblem::Ddg(const Vector &d)
{
  HypreParMatrix * Jfull = problem->Ddg(d);
  if (J != nullptr)
  {
    delete J; J = nullptr;
  }
  J = ParMult(P, Jfull, true);
  return J;
}

ReducedProblem::~ReducedProblem()
{
  delete P;
  if (J != nullptr)
  {
    delete J;
  }
}


