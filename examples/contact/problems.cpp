//                             Problem classes, which contain
//                             needed functionality for an
//                             interior-point filter-line search solver   
//                               
//
//

#include <fstream>
#include <iostream>
#include <array>

#include "mfem.hpp"
#include "problems.hpp"

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

/*ContactProblem::ContactProblem(int dimd, int dimg) : OptProblem(), dimD(dimd), dimS(dimg), block_offsetsx(3)
{
  dimU = dimD;
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
}*/

ContactProblem::ContactProblem() : OptProblem(), block_offsetsx(3)
{
  /*dimU = dimD;
  dimM = dimS;
  dimC = dimS;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;*/
}

void ContactProblem::InitializeParentData(int dimd, int dims)
{
  dimU = dimd;
  dimM = dims;
  dimC = dims;
  block_offsetsx[0] = 0;
  block_offsetsx[1] = dimU;
  block_offsetsx[2] = dimM;
  block_offsetsx.PartialSum();
  ml.SetSize(dimM); ml = 0.0;
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




//ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &)) : ContactProblem(fes->GetTrueVSize(), fes->GetTrueVSize()), Vh(fes), f(dimD)
ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &)) : ContactProblem()
{
  Vh = fes;
  dimD = fes->GetTrueVSize();
  dimS = fes->GetTrueVSize();
  InitializeParentData(dimD, dimS);
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
  f.SetSize(dimD);
  f.Set(1.0, *fform);

  Vector iDiag(dimD); iDiag = 1.0;
  J = new SparseMatrix(iDiag);

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
  return new SparseMatrix(*J);
}

ObstacleProblem::~ObstacleProblem()
{
  delete Kform;
  delete fform;
  delete J;
}

//-------------------
DirichletObstacleProblem::DirichletObstacleProblem(FiniteElementSpace *fes, Vector &x0DC, double (*fSource)(const Vector &), 
		double (*obstacleSource)(const Vector &),
		Array<int> tdof_list) : ContactProblem()
{
  Vh = fes;
  ess_tdof_list = tdof_list; 
  dimD = fes->GetTrueVSize();
  dimS = dimD;
  InitializeParentData(dimD, dimS);

  xDC.SetSize(dimD);
  xDC.Set(1.0, x0DC);
  
  // define Hessian of energy objective
  // K = [[ \hat{K}   0]
  //      [ 0         I]]
  // where \hat{K} acts on dofs not constrained by the Dirichlet condition
  // I acts on dofs constrained by the Dirichlet condition
  Kform = new BilinearForm(Vh);
  Kform->AddDomainIntegrator(new DiffusionIntegrator);
  Kform->Assemble();
  Kform->EliminateVDofs(ess_tdof_list);
  Kform->Finalize();
  K = new SparseMatrix(Kform->SpMat());
  
  // define right hand side dual-vector
  // f_i = int fSource(x) \phi_i(x) dx, where {\phi_i}_i is the FE basis
  // f = f - K1 xDC, where K1 contains the eliminated part of K
  FunctionCoefficient fcoeff(fSource);
  fform = new LinearForm(Vh);
  fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
  fform->Assemble();
  f.SetSize(dimD);
  f.Set(1.0, *fform);
  Kform->EliminateVDofsInRHS(ess_tdof_list, xDC, f);

  // define obstacle function
  FunctionCoefficient psicoeff(obstacleSource);
  GridFunction psi_gf(Vh);
  psi_gf.ProjectCoefficient(psicoeff);
  psi.SetSize(dimS); psi = 0.0;
  psi.Set(1.0, psi_gf);
  
  // ------ construct dimS x dimD Jacobian with zero columns correspodning to Dirichlet dofs
  J = new SparseMatrix(dimS, dimD);
  bool freeDof;
  for(int j = 0; j < dimD; j++)
  {
    freeDof = true;
    for(int i = 0; i < ess_tdof_list.Size(); i++)
    {
      if( j == ess_tdof_list[i])
      {
        freeDof = false;
      }
    }
    
    if (freeDof)
    {
      Array<int> col_tmp; mfem::Vector v_tmp;
      col_tmp.SetSize(1); v_tmp.SetSize(1);
      col_tmp[0] = j; v_tmp(0) = 1.0;
      psi(j)   = psi_gf(j);
      J->SetRow(j, col_tmp, v_tmp);
    }
    else
    {
      psi(j) = psi_gf(j) - 1.e-8;
    }
  }
  J->Finalize();
}

double DirichletObstacleProblem::E(const Vector &d) const
{
  Vector Kd(dimD); Kd = 0.0;
  K->Mult(d, Kd);
  return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d);
}

void DirichletObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
  K->Mult(d, gradE);
  gradE.Add(-1.0, f);
}

SparseMatrix* DirichletObstacleProblem::DddE(const Vector &d)
{
  return new SparseMatrix(*K); 
}

// g(d) = d - \psi >= 0
//        d - \psi - s  = 0
//                   s >= 0
void DirichletObstacleProblem::g(const Vector &d, Vector &gd) const
{
  J->Mult(d, gd);
  gd.Add(-1., psi);
}

SparseMatrix* DirichletObstacleProblem::Ddg(const Vector &d)
{
  return new SparseMatrix(*J);
}

DirichletObstacleProblem::~DirichletObstacleProblem()
{
  delete Kform;
  delete fform;
  delete J;
  delete K;
}


