//                             Problem classes, which contain
//                             needed functionality for an
//                             interior-point filter-line search solver   
//                               
//
//


#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



OptProblem::OptProblem() {}

void OptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y) const
{
  Duf(x, y.GetBlock(0));
  Dmf(x, y.GetBlock(1));
}

OptProblem::~OptProblem()
{
  block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0

ContactProblem::ContactProblem(int dimd) : OptProblem(), dimD(dimd), block_offsetsx(3)
{
  dimU    = dimD;
}

double ContactProblem::CalcObjective(const BlockVector &x) const { return E(x.GetBlock(0)); }

void ContactProblem::Duf(const BlockVector &x, Vector &y) const { DdE(x.GetBlock(0), y); }

void ContactProblem::Dmf(const BlockVector &x, Vector &y) const { y = 0.0; }

SparseMatrix* ContactProblem::Duuf(const BlockVector &x) { return DddE(x.GetBlock(0)); }

SparseMatrix* ContactProblem::Dumf(const BlockVector &x) { return nullptr; }

SparseMatrix* ContactProblem::Dmuf(const BlockVector &x) { return nullptr; }

SparseMatrix* ContactProblem::Dmmf(const BlockVector &x) { return nullptr; }

void ContactProblem::c(const BlockVector &x, Vector &y) const // c(u,m) = g(u) - m 
{
  g(x.GetBlock(0), y);
  y.Add(-1.0, x.GetBlock(1));  
}

SparseMatrix* ContactProblem::Duc(const BlockVector &x) { return Ddg(x.GetBlock(0)); }

SparseMatrix* ContactProblem::Dmc(const BlockVector &x) 
{ 
  Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  return new SparseMatrix(negIdentDiag);
} 

ContactProblem::~ContactProblem() {}




ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &)) : ContactProblem(fes->GetTrueVSize()), Vh(fes), f(dimD)
{
  Kform = new BilinearForm(Vh);
  Kform->AddDomainIntegrator(new MassIntegrator);
  Kform->AddDomainIntegrator(new DiffusionIntegrator);
  Kform->Assemble();
  Kform->Finalize();
  Kform->FormSystemMatrix(empty_tdof_list, K);

  FunctionCoefficient fcoeff(fSource);
  fform = new LinearForm(Vh);
  fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
  fform->Assemble();
  f.Set(1.0, *fform);

  dimS = dimD;
  
  /* ---- boiler plate code ---- */
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
  /* ---- end boiler plate ---- */
}

double ObstacleProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K.Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d);
}

void ObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
  K.Mult(d, gradE);
  gradE.Add(-1.0, f);
}

SparseMatrix* ObstacleProblem::DddE(const Vector &d)
{
  return new SparseMatrix(K); 
}

// g(d) = d >= 0
void ObstacleProblem::g(const Vector &d, Vector &gd) const
{
  gd.Set(1.0, d);
}

SparseMatrix* ObstacleProblem::Ddg(const Vector &d)
{
  Vector iDiag(dimD); iDiag = 1.0;
  return new SparseMatrix(iDiag);
}

ObstacleProblem::~ObstacleProblem()
{
  delete Kform;
  delete fform;
}



QPContactExample::QPContactExample(SparseMatrix *Kin, SparseMatrix *Jin, Vector *fin) : 
ContactProblem(Kin->Height()), K(Kin), J(Jin), f(fin) { 
  dimD = J->Width();
  dimS = J->Height();
  /* ---- boiler plate code ---- */
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
  /* ---- end boiler plate ---- */
}

double QPContactExample::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K->Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) + InnerProduct(*f, d);
}

void QPContactExample::DdE(const Vector &d, Vector &gradE) const
{
  K->Mult(d, gradE);
  gradE.Add(1.0, *f);
}

SparseMatrix* QPContactExample::DddE(const Vector &d)
{
  return new SparseMatrix(*K); 
}

// g(d) = J d >= 0
void QPContactExample::g(const Vector &d, Vector &gd) const
{
  J->Mult(d, gd);
}

SparseMatrix* QPContactExample::Ddg(const Vector &d)
{
  return new SparseMatrix(*J);
}

QPContactExample::~QPContactExample() {}


