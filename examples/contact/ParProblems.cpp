#include "mfem.hpp"
#include "ParProblems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

ParGeneralOptProblem::ParGeneralOptProblem(ParFiniteElementSpace * fesU_, ParFiniteElementSpace * fesM_) 
: fesU(fesU_), fesM(fesM_) 
{ 
   dimU = fesU->GetTrueVSize();
   dimM = fesM->GetTrueVSize();
   dimC = fesM->GetTrueVSize();
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
ParOptProblem::ParOptProblem(ParFiniteElementSpace * fesU_, 
                                     ParFiniteElementSpace * fesM_)
 : ParGeneralOptProblem(fesU_, fesM_), block_offsetsx(3)
{
   block_offsetsx[0] = 0;
   block_offsetsx[1] = dimU;
   block_offsetsx[2] = dimM;
   block_offsetsx.PartialSum();
   ml.SetSize(dimM); ml = 0.0;
   Vector negIdentDiag(dimM);
   negIdentDiag = -1.0;
   SparseMatrix * diag = new SparseMatrix(negIdentDiag);
   Ih = new HypreParMatrix(fesM->GetComm(), fesM->GlobalTrueVSize(),
                                             fesM->GetTrueDofOffsets(), diag);
   HypreStealOwnership(*Ih, *diag);
   delete diag;
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
   delete Ih;
}


// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ParObstacleProblem::ParObstacleProblem(ParFiniteElementSpace *fesU_, 
                                       ParFiniteElementSpace *fesM_, 
                                       double (*fSource)(const Vector &)) : 
                                       ParOptProblem(fesU_,fesM_), f(dimU), psi(dimU), J(nullptr)
{
   Kform = new ParBilinearForm(fesU);
   Kform->AddDomainIntegrator(new MassIntegrator);
   Kform->AddDomainIntegrator(new DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);

   FunctionCoefficient fcoeff(fSource);
   fform = new ParLinearForm(fesU);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   fform->Assemble();
   Vector F(dimU);
   fform->ParallelAssemble(F);
   f.SetSize(dimU);
   f.Set(1.0, F);

   psi = 0.0;
   
   Vector iDiag(dimU); iDiag = 1.0;
   SparseMatrix * Jacg = new SparseMatrix(iDiag);
   
   J = new HypreParMatrix(fesU->GetComm(),fesU->GlobalTrueVSize(),fesU->GetTrueDofOffsets(),Jacg);
   HypreStealOwnership(*J, *Jacg);
   delete Jacg;
}

// Obstacle Problem, essential boundary conditions enforced
// Hessian of energy term is K (stiffness)
ParObstacleProblem::ParObstacleProblem(ParFiniteElementSpace *fesU_, 
                                       ParFiniteElementSpace *fesM_, 
				       double (*fSource)(const Vector &),
				       double (*obstacleSource)(const Vector &),
				       Array<int> tdof_list, Vector &xDC) : ParOptProblem(fesU_,fesM_), f(dimU), psi(dimU), J(nullptr)
{
   // elastic energy functional terms	
   ess_tdof_list = tdof_list;
   Kform = new ParBilinearForm(fesU);
   Kform->AddDomainIntegrator(new DiffusionIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   Kform->FormSystemMatrix(ess_tdof_list, K);

   FunctionCoefficient fcoeff(fSource);
   fform = new ParLinearForm(fesU);
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

   J = new HypreParMatrix(fesU->GetComm(),fesU->GlobalTrueVSize(),fesU->GetTrueDofOffsets(),Jacg);
   HypreStealOwnership(*J, *Jacg);
   delete Jacg;

   FunctionCoefficient psi_fc(obstacleSource);
   ParGridFunction psi_gf(fesU);
   psi_gf.ProjectCoefficient(psi_fc);
   psi.Set(1.0, (*psi_gf.GetTrueDofs()));
   for(int i = 0; i < ess_tdof_list.Size(); i++)
   {
     psi(ess_tdof_list[i]) -= 1.e-8;
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
