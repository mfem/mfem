#include "mfem.hpp"
#include "problems.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



GeneralOptProblem::GeneralOptProblem() : block_offsetsx(3) {}

#ifdef MFEM_USE_MPI
   void GeneralOptProblem::InitGeneral(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
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
     dimC = dimM; // true for contact problems
     
     block_offsetsx[0] = 0;
     block_offsetsx[1] = dimU;
     block_offsetsx[2] = dimM;
     block_offsetsx.PartialSum();
     
     MPI_Allreduce(&dimU, &dimUGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(&dimM, &dimMGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
     MPI_Allreduce(&dimC, &dimCGlb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   }
#else
   void GeneralOptProblem::InitGeneral(int dimU_, int dimM_)
   {
     dimU = dimU_;
     dimM = dimM_;
     dimC = dimM;
     dimUGlb = dimU;
     dimMGlb = dimM;
   }
#endif

void GeneralOptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y) const
{
   Duf(x, y.GetBlock(0));
   Dmf(x, y.GetBlock(1));
}

GeneralOptProblem::~GeneralOptProblem()
{
   block_offsetsx.DeleteAll();
}


// min E(d) s.t. g(d) >= 0
// min_(d,s) E(d) s.t. c(d,s) := g(d) - s = 0, s >= 0
OptProblem::OptProblem() : GeneralOptProblem()
{
}

#ifdef MFEM_USE_MPI
   void OptProblem::Init(HYPRE_BigInt * dofOffsetsU_, HYPRE_BigInt * dofOffsetsM_)
   {
      InitGeneral(dofOffsetsU_, dofOffsetsM_);
      
      ml.SetSize(dimM); ml = 0.0;
      Vector negOneDiag(dimM);
      negOneDiag = -1.0;
      SparseMatrix * ISparse = new SparseMatrix(negOneDiag);
      Ih = new HypreParMatrix(MPI_COMM_WORLD, dimUGlb, dofOffsetsU, ISparse);
      HypreStealOwnership(*Ih, *ISparse);
      delete ISparse;
   }
#else
   void OptProblem::Init(int dimU_, int dimM_)
   {
      InitGeneral(dimU_, dimM_);
      
      ml.SetSize(dimM); ml = 0.0;
      Vector negOneDiag(dimM);
      negOneDiag = -1.0;
      Ih = new SparseMatrix(negOneDiag);    
   }
#endif


double OptProblem::CalcObjective(const BlockVector &x) const { return E(x.GetBlock(0)); }

void OptProblem::Duf(const BlockVector &x, Vector &y) const { DdE(x.GetBlock(0), y); }

void OptProblem::Dmf(const BlockVector &x, Vector &y) const { y = 0.0; }


#ifdef MFEM_USE_MPI

   HypreParMatrix * OptProblem::Duuf(const BlockVector &x) { return DddE(x.GetBlock(0)); }
   
   HypreParMatrix * OptProblem::Dumf(const BlockVector &x) { return nullptr; }
   
   HypreParMatrix * OptProblem::Dmuf(const BlockVector &x) { return nullptr; }
   
   HypreParMatrix * OptProblem::Dmmf(const BlockVector &x) { return nullptr; }
   
   HypreParMatrix * OptProblem::Duc(const BlockVector &x) { return Ddg(x.GetBlock(0)); }
          
   HypreParMatrix * OptProblem::Dmc(const BlockVector &x) { return Ih; } 
#else
   SparseMatrix * OptProblem::Duuf(const BlockVector &x) { return DddE(x.GetBlock(0)); }
   
   SparseMatrix * OptProblem::Dumf(const BlockVector &x) { return nullptr; }
   
   SparseMatrix * OptProblem::Dmuf(const BlockVector &x) { return nullptr; }
   
   SparseMatrix * OptProblem::Dmmf(const BlockVector &x) { return nullptr; }
   
   SparseMatrix * OptProblem::Duc(const BlockVector &x) { return Ddg(x.GetBlock(0)); }
   
   SparseMatrix * OptProblem::Dmc(const BlockVector &x) { return Ih; } 
#endif


void OptProblem::c(const BlockVector &x, Vector &y) const // c(u,m) = g(u) - m 
{
   g(x.GetBlock(0), y);
   y.Add(-1.0, x.GetBlock(1));  
}


OptProblem::~OptProblem() 
{
   #ifdef MFEM_USE_MPI
      delete[] dofOffsetsU;
      delete[] dofOffsetsM;
   #endif
   delete Ih;
}





// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
#ifdef MFEM_USE_MPI
   ObstacleProblem::ObstacleProblem(ParFiniteElementSpace *Vh_, 
                                          double (*fSource)(const Vector &),
   				       double (*obstacleSource)(const Vector &)) :
                                          OptProblem(), Vh(Vh_), J(nullptr) 
   {
      Init(Vh->GetTrueDofOffsets(), Vh->GetTrueDofOffsets());
      f.SetSize(dimU); f = 0.0;
      psi.SetSize(dimU); psi = 0.0;
      FunctionCoefficient psi_fc(obstacleSource);
      ParGridFunction psi_gf(Vh);
      psi_gf.ProjectCoefficient(psi_fc);
      psi_gf.GetTrueDofs(psi);
   
   
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
      
      J = new HypreParMatrix(MPI_COMM_WORLD, dimUGlb, dofOffsetsU, Jacg);
      HypreStealOwnership(*J, *Jacg);
      delete Jacg;
   }
#else
   ObstacleProblem::ObstacleProblem(FiniteElementSpace *fes, double (*fSource)(const Vector &), 
   		double (*obstacleSource)(const Vector &)) : OptProblem(), Vh(fes)
   {
     int dimD = fes->GetTrueVSize();
     int dimS = dimD;
     Init(dimD, dimS);
   
     
     Kform = new BilinearForm(Vh);
     Kform->AddDomainIntegrator(new DiffusionIntegrator);
     Kform->AddDomainIntegrator(new MassIntegrator);
     Kform->Assemble();
     Kform->Finalize();
     K = Kform->SpMat();
   
     FunctionCoefficient fcoeff(fSource);
     fform = new LinearForm(Vh);
     fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
     fform->Assemble();
     f.SetSize(dimD);
     f.Set(1.0, *fform);
       
     // define obstacle function
     FunctionCoefficient psicoeff(obstacleSource);
     GridFunction psi_gf(Vh);
     psi_gf.ProjectCoefficient(psicoeff);
     //psi.SetSize(dimS); psi = 0.0;
     psi_gf.GetTrueDofs(psi);
     //psi.Set(1.0, psi_gf);
     
     // ------ construct dimS x dimD Jacobian with zero columns correspodning to Dirichlet dofs
     Vector one(dimD); one = 1.0;
     J = new SparseMatrix(one);
   }
#endif










#ifdef MFEM_USE_MPI
   // Obstacle Problem, essential boundary conditions enforced
   // Hessian of energy term is K (stiffness)
   ObstacleProblem::ObstacleProblem(ParFiniteElementSpace *Vh_, 
   				       double (*fSource)(const Vector &),
   				       double (*obstacleSource)(const Vector &),
   				       Array<int> tdof_list, Vector &xDC) : OptProblem(), 
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
   
      J = new HypreParMatrix(MPI_COMM_WORLD, dimUGlb, dofOffsetsU, Jacg);
      HypreStealOwnership(*J, *Jacg);
      delete Jacg;
   
      FunctionCoefficient psi_fc(obstacleSource);
      ParGridFunction psi_gf(Vh);
      psi_gf.ProjectCoefficient(psi_fc);
      psi_gf.GetTrueDofs(psi);
      for(int i = 0; i < ess_tdof_list.Size(); i++)
      {
        psi(ess_tdof_list[i]) = xDC(ess_tdof_list[i]) - 1.e-8;
      }
   }
#endif


double ObstacleProblem::E(const Vector &d) const
{
   Vector Kd(K.Height()); Kd = 0.0;
   MFEM_VERIFY(d.Size() == K.Width(), "ObstacleProblem::E - Inconsistent dimensions");
   K.Mult(d, Kd);
   #ifdef MFEM_USE_MPI
      return 0.5 * InnerProduct(MPI_COMM_WORLD, d, Kd) - InnerProduct(MPI_COMM_WORLD, f, d);
   #else
      return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d);
   #endif
}

void ObstacleProblem::DdE(const Vector &d, Vector &gradE) const
{
   gradE.SetSize(K.Height());
   MFEM_VERIFY(d.Size() == K.Width(), "ParObstacleProblem::DdE - Inconsistent dimensions");
   K.Mult(d, gradE);
   MFEM_VERIFY(f.Size() == K.Height(), "ParObstacleProblem::DdE - Inconsistent dimensions");
   gradE.Add(-1.0, f);
}

#ifdef MFEM_USE_MPI
   HypreParMatrix * ObstacleProblem::DddE(const Vector &d)
   {
      return &K; 
   }
   HypreParMatrix * ObstacleProblem::Ddg(const Vector &d)
   {
      return J; 
   }   
#else
   SparseMatrix * ObstacleProblem::DddE(const Vector &d)
   {
      return &K;
   }
   SparseMatrix * ObstacleProblem::Ddg(const Vector &d)
   {
      return J;
   }
#endif


// g(d) = d >= \psi
void ObstacleProblem::g(const Vector &d, Vector &gd) const
{
   MFEM_VERIFY(d.Size() == J->Width(), "ObstacleProblem::g - Inconsistent dimensions");
   J->Mult(d, gd);
   MFEM_VERIFY(gd.Size() == J->Height(), "ObstacleProblem::g - Inconsistent dimensions");
   gd.Add(-1.0, psi);
}

ObstacleProblem::~ObstacleProblem()
{
   delete Kform;
   delete fform;
   delete J;
}


