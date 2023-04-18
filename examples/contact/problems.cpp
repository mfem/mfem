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

void ContactProblem::Duuf(const BlockVector &x, SparseMatrix *&y) { DddE(x.GetBlock(0), y); }

void ContactProblem::Dumf(const BlockVector &x, SparseMatrix *&y) { y = nullptr; }

void ContactProblem::Dmuf(const BlockVector &x, SparseMatrix *&y) { y = nullptr; }

void ContactProblem::Dmmf(const BlockVector &x, SparseMatrix *&y) { y = nullptr; }

void ContactProblem::c(const BlockVector &x, Vector &y) const // c(u,m) = g(u) - m 
{
  g(x.GetBlock(0), y);
  y.Add(-1.0, x.GetBlock(1));  
}

void ContactProblem::Duc(const BlockVector &x, SparseMatrix *&y) { Ddg(x.GetBlock(0), y); }

void ContactProblem::Dmc(const BlockVector &x, SparseMatrix *&y) 
{ 
  Vector negIdentDiag(dimM);
  negIdentDiag = -1.0;
  y = new SparseMatrix(negIdentDiag);
} 

ContactProblem::~ContactProblem() {}




ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes) : ContactProblem(fes->GetTrueVSize()), Vh(fes), f(dimD)
{
  Kform = new BilinearForm(Vh);
  Kform->AddDomainIntegrator(new MassIntegrator);
  Kform->AddDomainIntegrator(new DiffusionIntegrator);
  Kform->Assemble();
  Kform->Finalize();
  Kform->FormSystemMatrix(empty_tdof_list, K);

  FunctionCoefficient fcoeff(fRhs);
  fform = new LinearForm(Vh);
  fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
  fform->Assemble();
  f.Set(1.0, *fform);

  dimS = dimD;
  
  /* ---- boiling plate code ---- */
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;

  /* ---- end boiling plate ---- */
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

void ObstacleProblem::DddE(const Vector &d, SparseMatrix *&HessE)
{
  HessE = new SparseMatrix(K); 
}

// g(d) = d >= 0
void ObstacleProblem::g(const Vector &d, Vector &gd) const
{
  gd.Set(1.0, d);
}

void ObstacleProblem::Ddg(const Vector &d, SparseMatrix *&Jacg)
{
  Vector iDiag(dimD); iDiag = 1.0;
  Jacg = new SparseMatrix(iDiag);
}

// f(d) forcing term...
// this term is such that in the absence of bound-constraints then
// -div(grad(d)) + d = f + homogeneous Neumann conditions on the unit interval,
// for d(x) = cos(2 \pi x) + a0 + a3 (x^3 - 1.5 x^2), a2 = 0.2, a3 = -2
// see overleaf document Interior Point Progress II for more details
// for a similar function/where it has been used previously
double ObstacleProblem::fRhs(const Vector &x)
{
  double fx = 0.;
  //fx = (1. + pow(4.*M_PI,2))*cos(4.*M_PI*x(0)) + 1. * pow(x(0),2) - 0.85 * (pow(x(0),3) + 6. * x(0) - 3.);
  fx = 0.2 - 2.0 * (pow(x(0),3)- 1.5*pow(x(0),2.) - 6 * x(0) + 3.) + (1. + pow(2.*M_PI,2))*cos(2.*M_PI*x(0));
  return fx;
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
  cout << "dim(D) = " << dimD << endl;
  cout << "dim(S) = " << dimS << endl;
  /* ---- boiling plate code ---- */
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
  /* ---- end boiling plate ---- */
  cout << "dim(M) = " << dimM << endl;

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

void QPContactExample::DddE(const Vector &d, SparseMatrix *&HessE)
{
  HessE = new SparseMatrix(*K); 
}

// g(d) = J d >= 0
void QPContactExample::g(const Vector &d, Vector &gd) const
{
  J->Mult(d, gd);
}

void QPContactExample::Ddg(const Vector &d, SparseMatrix *&Jacg)
{
  Jacg = new SparseMatrix(*J);
}

QPContactExample::~QPContactExample() {}


#if 0
double ContactTNLP:E(const Vector &d) const
{
  double Eeval = 0.0;
  bool eval_E_flag;
  eval_E_flag = eval_f(dimD, d, true, Eeval);
  assert(eval_E_flag && "error in objective evaluation");
  return Eeval;
}

void ContactTNLP:DdE(const Vector &D, Vector &gradE) const
{
  bool eval_dE_flag;
  eval_dE_flag = eval_grad_f();
  assert(eval_dE_flag && "error in objective gradient computation");
}



#endif 

