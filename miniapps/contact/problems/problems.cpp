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
     parallel = true;
   }
#endif
void GeneralOptProblem::InitGeneral(int dimU_, int dimM_)
{
  dimU = dimU_;
  dimM = dimM_;
  dimC = dimM;
  dimUGlb = dimU;
  dimMGlb = dimM;
  parallel = false;
}

void GeneralOptProblem::CalcObjectiveGrad(const BlockVector &x, BlockVector &y)
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
      SparseMatrix * Iloc = new SparseMatrix(negOneDiag);
      Ih = new HypreParMatrix(MPI_COMM_WORLD, dimMGlb, dofOffsetsM, Iloc);
      HypreStealOwnership(*Ih, *Iloc);
      delete Iloc;
      Isparse = nullptr;
   }
#endif
void OptProblem::Init(int dimU_, int dimM_)
{
   InitGeneral(dimU_, dimM_);
   
   ml.SetSize(dimM); ml = 0.0;
   Vector negOneDiag(dimM);
   negOneDiag = -1.0;
   Isparse = new SparseMatrix(negOneDiag);
   #ifdef MFEM_USE_MPI
      Ih = nullptr;
   #endif   
}


double OptProblem::CalcObjective(const BlockVector &x) { return E(x.GetBlock(0)); }

void OptProblem::Duf(const BlockVector &x, Vector &y) { DdE(x.GetBlock(0), y); }

void OptProblem::Dmf(const BlockVector &x, Vector &y) { y = 0.0; }

Operator * OptProblem::Duuf(const BlockVector &x) { return DddE(x.GetBlock(0)); }

Operator * OptProblem::Dumf(const BlockVector &x) { return nullptr; }

Operator * OptProblem::Dmuf(const BlockVector &x) { return nullptr; }

Operator * OptProblem::Dmmf(const BlockVector &x) { return nullptr; }

Operator * OptProblem::Duc(const BlockVector &x) { return Ddg(x.GetBlock(0)); }

Operator * OptProblem::Dmc(const BlockVector &x) 
{ 
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return Ih; 
   }
   else
   {
#endif
   return Isparse;
#ifdef MFEM_USE_MPI
   }
#endif
} 


void OptProblem::c(const BlockVector &x, Vector &y) // c(u,m) = g(u) - m 
{
   g(x.GetBlock(0), y);
   y.Add(-1.0, x.GetBlock(1));
}


OptProblem::~OptProblem() 
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      delete[] dofOffsetsU;
      delete[] dofOffsetsM;
      delete Ih;
   }
   else
   {
#endif
   delete Isparse;
#ifdef MFEM_USE_MPI
   }
#endif
}





// Obstacle Problem, no essential boundary conditions enforced
// Hessian of energy term is K + M (stiffness + mass)
ObstacleProblem::ObstacleProblem(FiniteElementSpace *Vh_, 
                                 double (*fSource)(const Vector &),
				                     double (*obstacleSource)(const Vector &)) 
                                 : OptProblem() 
{
   FunctionCoefficient fcoeff(fSource);
   FunctionCoefficient psi_fc(obstacleSource);
   
#ifdef MFEM_USE_MPI
   Vhp = dynamic_cast<ParFiniteElementSpace *>(Vh_);
   if (Vhp)
   {
      Init(Vhp->GetTrueDofOffsets(), Vhp->GetTrueDofOffsets());
   }
   else
   {
#endif
   
   Vh = Vh_;
   int dimD = Vh->GetTrueVSize();
   int dimS = dimD;
   Init(dimD, dimS);

#ifdef MFEM_USE_MPI
   }
#endif
   
   psi.SetSize(dimU); psi = 0.0;
   f.SetSize(dimU);   f   = 0.0;
   Vector one(dimU);  one = 1.0;
   
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Kform = new ParBilinearForm(Vhp);
      Kform->AddDomainIntegrator(new MassIntegrator);
      Kform->AddDomainIntegrator(new DiffusionIntegrator);
      Kform->Assemble();
      Kform->Finalize();
      Kform->FormSystemMatrix(ess_tdof_list, Kh);	 
         
      ParGridFunction psi_gf(Vhp);
      psi_gf.ProjectCoefficient(psi_fc);
      psi_gf.GetTrueDofs(psi);
         
      fformp = new ParLinearForm(Vhp);
      fformp->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
      fformp->Assemble();
      fformp->ParallelAssemble(f);

      SparseMatrix * Jacg = new SparseMatrix(one);
      Jh = new HypreParMatrix(MPI_COMM_WORLD, dimUGlb, dofOffsetsU, Jacg);
      HypreStealOwnership(*Jh, *Jacg);
      delete Jacg;
   }
   else
   {
#endif 
   Kform = new BilinearForm(Vh);
   Kform->AddDomainIntegrator(new DiffusionIntegrator);
   Kform->AddDomainIntegrator(new MassIntegrator);
   Kform->Assemble();
   Kform->Finalize();
   K = Kform->SpMat();
   
   GridFunction psi_gf(Vh);
   psi_gf.ProjectCoefficient(psi_fc);
   psi_gf.GetTrueDofs(psi);

   fform = new LinearForm(Vh);
   fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   fform->Assemble();
   f.Set(1.0, *fform);

   J = new SparseMatrix(one);
#ifdef MFEM_USE_MPI
   }
#endif
}




//#ifdef MFEM_USE_MPI
//   // Obstacle Problem, essential boundary conditions enforced
//   // Hessian of energy term is K (stiffness)
//   ObstacleProblem::ObstacleProblem(ParFiniteElementSpace *Vh_, 
//   				       double (*fSource)(const Vector &),
//   				       double (*obstacleSource)(const Vector &),
//   				       Array<int> tdof_list, Vector &xDC) : OptProblem(), 
//   	                                                                    Vh(Vh_), J(nullptr)
//   {
//      Init(Vh->GetTrueDofOffsets(), Vh->GetTrueDofOffsets());
//      f.SetSize(dimU); f = 0.0;
//      psi.SetSize(dimU); psi = 0.0;
//      // elastic energy functional terms	
//      ess_tdof_list = tdof_list;
//      Kform = new ParBilinearForm(Vh);
//      Kform->AddDomainIntegrator(new DiffusionIntegrator);
//      Kform->Assemble();
//      Kform->Finalize();
//      Kform->FormSystemMatrix(ess_tdof_list, K);
//   
//      FunctionCoefficient fcoeff(fSource);
//      fform = new ParLinearForm(Vh);
//      fform->AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
//      fform->Assemble();
//      Vector F(dimU);
//      fform->ParallelAssemble(F);
//      f.SetSize(dimU);
//      f.Set(1.0, F);
//      Kform->EliminateVDofsInRHS(ess_tdof_list, xDC, f);
//      
//      // obstacle constraints --  
//      Vector iDiag(dimU); iDiag = 1.0;
//      for(int i = 0; i < ess_tdof_list.Size(); i++)
//      {
//        iDiag(ess_tdof_list[i]) = 0.0;
//      }
//      SparseMatrix * Jacg = new SparseMatrix(iDiag);
//   
//      J = new HypreParMatrix(MPI_COMM_WORLD, dimUGlb, dofOffsetsU, Jacg);
//      HypreStealOwnership(*J, *Jacg);
//      delete Jacg;
//   
//      FunctionCoefficient psi_fc(obstacleSource);
//      ParGridFunction psi_gf(Vh);
//      psi_gf.ProjectCoefficient(psi_fc);
//      psi_gf.GetTrueDofs(psi);
//      for(int i = 0; i < ess_tdof_list.Size(); i++)
//      {
//        psi(ess_tdof_list[i]) = xDC(ess_tdof_list[i]) - 1.e-8;
//      }
//   }
//#endif


double ObstacleProblem::E(const Vector &d)
{
   Vector Kd(dimU); Kd = 0.0;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Kh.Mult(d, Kd);
      return 0.5 * InnerProduct(MPI_COMM_WORLD, d, Kd) - InnerProduct(MPI_COMM_WORLD, f, d);
   }
   else
   {
#endif
   K.Mult(d, Kd);
   return 0.5 * InnerProduct(d, Kd) - InnerProduct(f, d);
#ifdef MFEM_USE_MPI
   }
#endif
}

void ObstacleProblem::DdE(const Vector &d, Vector &gradE)
{
   gradE.SetSize(dimU); gradE = 0.0;
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Kh.Mult(d, gradE);
   }
   else
   {
#endif

   K.Mult(d, gradE);

#ifdef MFEM_USE_MPI
   }
#endif

   gradE.Add(-1.0, f);
}

Operator * ObstacleProblem::DddE(const Vector &d)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return &Kh;
   }
   else
   {
#endif
   return &K;
#ifdef MFEM_USE_MPI
   }
#endif 
}

Operator * ObstacleProblem::Ddg(const Vector &d)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return Jh;
   }
   else
   {
#endif
   return J;
#ifdef MFEM_USE_MPI
   }
#endif 
}   


// g(d) = d >= \psi
void ObstacleProblem::g(const Vector &d, Vector &gd)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Jh->Mult(d, gd);
   }
   else
   {
#endif

   J->Mult(d, gd);

#ifdef MFEM_USE_MPI
   }
#endif

   gd.Add(-1.0, psi);
}

ObstacleProblem::~ObstacleProblem()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      delete Kformp;
      delete fformp;
      delete Jh;
   }
   else
   {
#endif
   delete Kform;
   delete fform;
   delete J;
#ifdef MFEM_USE_MPI
   }
#endif
}


ElasticityProblem::ElasticityProblem(const char *mesh_file , int ref, int order_ = 1) 
                                     : order(order_)
{
   mesh = new Mesh(mesh_file, 1, 1);
   for (int i = 0; i < ref; i++)
   {
      mesh->UniformRefinement();
   }
   parallel = false;
   Init();
}

#ifdef MFEM_USE_MPI
ElasticityProblem::ElasticityProblem(MPI_Comm comm_, const char * mesh_file, 
                                         int sref, int pref, int order_ = 1) 
                                         : comm(comm_), order(order_) 
{
   own_mesh = true;
   mesh = new Mesh(mesh_file,1,1);
   for (int i = 0; i<sref; i++)
   {
      mesh->UniformRefinement();
   }
   pmesh = new ParMesh(comm,*mesh);
   MFEM_VERIFY(pmesh->GetNE(), "ElasticityProblem::Empty partition");
   delete mesh;
   for (int i = 0; i<pref; i++)
   {
      pmesh->UniformRefinement();
   }
   parallel = true;
   Init();
}

ElasticityProblem::ElasticityProblem(ParMesh * pmesh_, int order_ = 1) 
                                     :  pmesh(pmesh_), order(order_)
{
   own_mesh = false;
   comm = pmesh->GetComm();
   parallel = true;
   Init();
}
#endif


void ElasticityProblem::Init()
{
#ifdef MFEM_USE_MPI
   if(parallel)
   {
      // do something here for parallel Init
      int dim = pmesh->Dimension();
      fec    = new H1_FECollection(order, dim);
      fesp   = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
      ndofs  = fesp->GetVSize();
      ntdofs = fesp->GetTrueVSize();
      gndofs = fesp->GlobalTrueVSize();
      pmesh->SetNodalFESpace(fesp);
      if (pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      }
      ess_bdr = 0; ess_bdr[1] = 1;
      fesp->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      // Solution GridFunction
      xp.SetSpace(fesp);  x = 0.0;
      // RHS
      bp.Update(fesp);
   
      // Elasticity operator
      lambda.SetSize(pmesh->attributes.Max()); lambda = 57.6923076923;
      mu.SetSize(pmesh->attributes.Max()); mu = 38.4615384615;
   
      lambda_cf.UpdateConstants(lambda);
      mu_cf.UpdateConstants(mu);
   
      ap = new ParBilinearForm(fesp);
      ap->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));
   }
   else
   {
#endif
   int dim = mesh->Dimension();
   fec = new H1_FECollection(order, dim);
   fes = new FiniteElementSpace(mesh, fec, dim, Ordering::byVDIM);
   ndofs = fes->GetTrueVSize();
   ntdofs = ndofs;
   mesh->SetNodalFESpace(fes);
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
   }
   ess_bdr = 0; ess_bdr[1] = 1;
   fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);

   // Solution GridFunction
   x.SetSpace(fes);  x = 0.0;
   // RHS
   b.Update(fes);

   // Elasticity operator
   lambda.SetSize(mesh->attributes.Max()); lambda = 57.6923076923;
   mu.SetSize(mesh->attributes.Max()); mu = 38.4615384615;

   lambda_cf.UpdateConstants(lambda);
   mu_cf.UpdateConstants(mu);
   a = new BilinearForm(fes);
   a->SetDiagonalPolicy(Operator::DIAG_ONE);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));
#ifdef MFEM_USE_MPI
   }
#endif
}

Mesh * ElasticityProblem::GetMesh() 
{ 
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return pmesh;
   }
   else
   {
#endif
   return mesh;
#ifdef MFEM_USE_MPI
   }
#endif
}

FiniteElementSpace * ElasticityProblem::GetFESpace()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return fesp;
   }
   else
   {
#endif
   return fes;
#ifdef MFEM_USE_MPI
   }
#endif
}

Operator & ElasticityProblem::GetOperator() 
{ 
   MFEM_VERIFY(formsystem, "System not formed yet. Call FormLinearSystem()");
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return Ap;
   }
   else
   {
#endif
   return A;
#ifdef MFEM_USE_MPI
   }
#endif
}

Vector & ElasticityProblem::GetRHS() 
{ 
   MFEM_VERIFY(formsystem, "System not formed yet. Call FormLinearSystem()"); 
   return B; 
}

void ElasticityProblem::SetLambda(const Vector & lambda_) 
{ 
   lambda = lambda_; 
   lambda_cf.UpdateConstants(lambda);
}

void ElasticityProblem::SetMu(const Vector & mu_) 
{ 
   mu = mu_; 
   mu_cf.UpdateConstants(mu);
}

void ElasticityProblem::FormLinearSystem()
{
   if (!formsystem) 
   {
      formsystem = true;
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         bp.Assemble();
         ap->Assemble();
         ap->FormLinearSystem(ess_tdof_list, xp, bp, Ap, X, B);
      }
      else
      {
#endif
      b.Assemble();
      a->Assemble();
      a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
#ifdef MFEM_USE_MPI
      }
#endif
   }
}
   
void ElasticityProblem::UpdateLinearSystem()
{
   if (formsystem)
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         bp.Update();
         ap->Update();
      }
      else
      {
#endif
      b.Update();
      a->Update();
#ifdef MFEM_USE_MPI
      }
#endif
      formsystem = false;
   }
   FormLinearSystem();
}


void ElasticityProblem::SetDisplacementDirichletData(const Vector & delta) 
{
   VectorConstantCoefficient delta_cf(delta);
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      xp.ProjectBdrCoefficient(delta_cf,ess_bdr);
   }
   else
   {
#endif
   x.ProjectBdrCoefficient(delta_cf, ess_bdr);
#ifdef MFEM_USE_MPI
   }
#endif
}

GridFunction & ElasticityProblem::GetDisplacementGridFunction() 
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      MFEM_VERIFY(false, "ElasticityProblem (parallel) shouldn't call GetDisplacementGridFunction");
      return xp;
   }
   else
   {
#endif
   return x;
#ifdef MFEM_USE_MPI
   }
#endif
}

#ifdef MFEM_USE_MPI 
ParGridFunction & ElasticityProblem::GetDisplacementParGridFunction()
{
   return xp;
}
#endif


ElasticityProblem::~ElasticityProblem()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      delete ap;
      delete fesp;
      delete fec;
      if (own_mesh)
      {
         delete pmesh;
      }   
   }
   else
   {
#endif
   delete a;
   delete fes;
   delete fec;
   delete mesh;
#ifdef MFEM_USE_MPI
   }
#endif
}


ContactProblem::ContactProblem(ElasticityProblem * prob1_, ElasticityProblem * prob2_)
   : OptProblem(), prob1(prob1_), prob2(prob2_)
{
   Mesh * mesh1;
#ifdef MFEM_USE_MPI 
   parallel = prob1->IsParallel();
   ParMesh * pmesh1;
   if (parallel)
   {
      pmesh1 = dynamic_cast<ParMesh*>(prob1->GetMesh());
      comm = pmesh1->GetComm();
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numprocs);
   }
   else
   {
#endif
   mesh1 = prob1->GetMesh();
#ifdef MFEM_USE_MPI
   }
#endif
   
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      dim = pmesh1->Dimension();
      nodes0.SetSpace(pmesh1->GetNodes()->FESpace());
      nodes0 = *pmesh1->GetNodes();
      nodes1 =  pmesh1->GetNodes();
   }
   else
   {
#endif
   dim = mesh1->Dimension();
   nodes0.SetSpace(mesh1->GetNodes()->FESpace());
   nodes0 = *mesh1->GetNodes();
   nodes1 = mesh1->GetNodes();
#ifdef MFEM_USE_MPI
   }
#endif
   
   Vector delta1(dim);
   delta1 = 0.0; delta1[0] = 0.1;
   prob1->SetDisplacementDirichletData(delta1);
   prob1->FormLinearSystem();
   
   Vector delta2(dim);
   delta2 = 0.0; 
   prob2->SetDisplacementDirichletData(delta2);
   prob2->FormLinearSystem();
   
   int ndof1 = prob1->GetNumTDofs();
   int ndof2 = prob2->GetNumTDofs();
   
   tdof_offsets.SetSize(3);
   tdof_offsets[0] = 0;
   tdof_offsets[1] = ndof1;
   tdof_offsets[2] = ndof2;
   tdof_offsets.PartialSum();
   
#ifdef MFEM_USE_MPI
   if (parallel) 
   {
      Array2D<HypreParMatrix*> A(2,2);
      A(0,0) = dynamic_cast<HypreParMatrix *>(&prob1->GetOperator());
      A(1,1) = dynamic_cast<HypreParMatrix *>(&prob2->GetOperator());
      A(1,0) = nullptr;
      A(0,1) = nullptr;
      Kp = HypreParMatrixFromBlocks(A);
   }
   else
   {
#endif
   BlockMatrix A(tdof_offsets);
   SparseMatrix * A00 = dynamic_cast<SparseMatrix *>(&prob1->GetOperator());
   SparseMatrix * A11 = dynamic_cast<SparseMatrix *>(&prob2->GetOperator());
   A.SetBlock(0, 0, A00);
   A.SetBlock(1, 1, A11);

   K = A.CreateMonolithic();
   K->Threshold(0.0);
   K->SortColumnIndices();
#ifdef MFEM_USE_MPI
   }
#endif 
   
   B = new BlockVector(tdof_offsets);
   B->GetBlock(0).Set(1.0, prob1->GetRHS());
   B->GetBlock(1).Set(1.0, prob2->GetRHS());

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      ParComputeContactVertices();
   }
   else
   {
#endif
   ComputeContactVertices();
#ifdef MFEM_USE_MPI
   }
#endif

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      HYPRE_BigInt offsetU = 0;
      MPI_Scan(&tdof_offsets.Last(), &offsetU, 1, MPI_INT, MPI_SUM, pmesh1->GetComm());
      offsetU -= tdof_offsets.Last();
      
      
      HYPRE_BigInt * offsetsU_temp = new HYPRE_BigInt[2];
      offsetsU_temp[0] = offsetU;
      offsetsU_temp[1] = offsetU + tdof_offsets.Last();


      HYPRE_BigInt * offsetsM_temp = new HYPRE_BigInt[2];
      offsetsM_temp[0] = constraints_starts[0];
      offsetsM_temp[1] = constraints_starts[1];

      Init(offsetsU_temp, offsetsM_temp);
      delete[] offsetsU_temp;
      delete[] offsetsM_temp;
   }
   else
   {
#endif
   Init(GetNumDofs(), GetNumConstraints());
#ifdef MFEM_USE_MPI
   }
#endif  
 
}


int ContactProblem::GetNumDofs()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return Kp->Height();
   }
   else
   {
#endif
    return K->Height();
#ifdef MFEM_USE_MPI
   }
#endif
}

int ContactProblem::GetGlobalNumDofs() 
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return Kp->GetGlobalNumRows();   
   }
   else
   {
#endif
   return K->Height();
#ifdef MFEM_USE_MPI
   }
#endif
}

Operator * ContactProblem::GetJacobian()
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return Mp;   
   }
   else
   {
#endif
   return M;
#ifdef MFEM_USE_MPI
   }
#endif
}

Array<Operator *> ContactProblem::GetHessian() 
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      return dMp;
   }
   else
   {
#endif
   return dM;
#ifdef MFEM_USE_MPI
   }
#endif
}

double ContactProblem::E(const Vector & d)
{
   Vector Kd(GetNumDofs());
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Kp->Mult(d, Kd);
      return 0.5 * InnerProduct(comm, d, Kd) - InnerProduct(comm, d, *B);
   }
   else
   {
#endif
   K->Mult(d, Kd);
   return 0.5 * InnerProduct(d, Kd) - InnerProduct(d, *B);
#ifdef MFEM_USE_MPI
   }
#endif
}
   
void ContactProblem::DdE(const Vector &d, Vector &gradE)
{
   gradE.SetSize(GetNumDofs());
#ifdef MFEM_USE_MPI
   if (parallel)
   {
      Kp->Mult(d, gradE);
   }
   else
   {
#endif
   K->Mult(d, gradE);
#ifdef MFEM_USE_MPI
   }
#endif
   gradE.Add(-1.0, *B); 
}
   
Operator * ContactProblem::DddE(const Vector &d)
{
#ifdef MFEM_USE_MPI
   if (parallel)
   {   
      return Kp;
   }
   else
   {
#endif
   return K;
#ifdef MFEM_USE_MPI
   }
#endif 
}

void ContactProblem::g(const Vector &d, Vector &gd)//, bool compute_hessians_)
{
   compute_hessians = true;//compute_hessians_;
   int ndof1 = prob1->GetNumTDofs();
   int ndof2 = prob2->GetNumTDofs();
   double * data = d.GetData();
   Vector displ1(data,ndof1);
   Vector displ2(&data[ndof1],ndof2);
   
   if (recompute)
   {
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         ParComputeGapFunctionAndDerivatives(displ1, displ2);
      }
      else
      {
#endif 
      ComputeGapFunctionAndDerivatives(displ1, displ2);
#ifdef MFEM_USE_MPI
      }
#endif
      recompute = false;
   }
   
   gd = GetGapFunction();
}



Operator * ContactProblem::Ddg(const Vector &d)
{
   return GetJacobian();
}




#ifdef MFEM_USE_MPI
void ContactProblem::ParComputeContactVertices()
{
   if (gnpoints > 0) 
   {
      return;
   }
   ParMesh * pmesh1 = dynamic_cast<ParMesh *>(prob1->GetMesh());
   ParMesh * pmesh2 = dynamic_cast<ParMesh *>(prob2->GetMesh());
   dim = pmesh1->Dimension();
   
   vfes1p = new ParFiniteElementSpace(pmesh1, prob1->GetFECol());
   vfes2p = new ParFiniteElementSpace(pmesh2, prob2->GetFECol());
   
   int gnv1 = vfes1p->GlobalTrueVSize();
   int gnv2 = vfes2p->GlobalTrueVSize();
   gnv = gnv1+gnv2;
   int nv1 = vfes1p->GetTrueVSize();
   int nv2 = vfes2p->GetTrueVSize();
   nv = nv1+nv2;
   
   vertices1.SetSize(pmesh1->GetNV());
   vertices2.SetSize(pmesh2->GetNV());
   
   for (int i = 0; i<pmesh1->GetNV(); i++)
   {
      vertices1[i] = i;
   }
   pmesh1->GetGlobalVertexIndices(vertices1);
   
   for (int i = 0; i<pmesh2->GetNV(); i++)
   {
      vertices2[i] = i;
   }
   pmesh2->GetGlobalVertexIndices(vertices2);
   
   int voffset2 = vfes2p->GetMyTDofOffset();
   
   std::vector<int> vertex2_offsets;
   ComputeTdofOffsets(comm, voffset2, vertex2_offsets);
   
   Array<int> vert;
   for (int b=0; b<pmesh2->GetNBE(); b++)
   {
      if (pmesh2->GetBdrAttribute(b) == 3)
      {
         pmesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            if (myid != get_rank(vertices2[v],vertex2_offsets)) { continue; }
             contact_vertices.insert(v);
         }
      }
   }
   
   npoints = contact_vertices.size();
   
   MPI_Allreduce(&npoints, &gnpoints,1,MPI_INT,MPI_SUM,pmesh1->GetComm());
   int constrains_offset;
   MPI_Scan(&npoints,&constrains_offset,1,MPI_INT,MPI_SUM,pmesh1->GetComm());
   
   constrains_offset-=npoints;
   constraints_starts.SetSize(2);
   constraints_starts[0] = constrains_offset;
   constraints_starts[1] = constrains_offset+npoints;
  
   ComputeTdofOffsets(comm,constrains_offset, constraints_offsets);
}

void ContactProblem::ParComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector &displ2)
{
   ComputeContactVertices();
   ParMesh * pmesh1 = dynamic_cast<ParMesh *>(prob1->GetMesh());
   ParMesh * pmesh2 = dynamic_cast<ParMesh *>(prob2->GetMesh());

   vfes1p = dynamic_cast<ParFiniteElementSpace *>(prob1->GetFESpace());
   vfes2p = dynamic_cast<ParFiniteElementSpace *>(prob2->GetFESpace());
   ParGridFunction displ1_gf(vfes1p);
   ParGridFunction displ2_gf(vfes2p);

   displ1_gf.SetFromTrueDofs(displ1);
   displ2_gf.SetFromTrueDofs(displ2);

   Array<int> conn2(npoints); 
   Vector xyz(dim * npoints);

   int cnt = 0;
   for (auto v : contact_vertices)
   {
      for (int d = 0; d<dim; d++)
      {
         xyz(cnt*dim + d) = pmesh2->GetVertex(v)[d]+displ2_gf[v*dim+d];
      }
      conn2[cnt] = vertices2[v];
      cnt++;
   }

   MFEM_VERIFY(cnt == npoints, "");
   gapv.SetSize(npoints*dim); gapv = 0.0;
   // segment reference coordinates of the closest point
   Vector xi1(npoints*(dim-1));
   Array<int> conn1(npoints*4);
   DenseMatrix coordsm(npoints*4, dim);
   // add(nodes0, displ1_gf, *nodes1);
   FindPointsInMesh(*pmesh1, vertices1, conn2, displ1_gf, xyz, conn1, xi1, coordsm);
   if (Mp)
   {
      delete Mp;
      for (int i = 0; i < dMp.Size(); i++)
      {
         delete dMp[i];
      }
      dMp.SetSize(0);
   }

   int ndofs1 = vfes1p->GetTrueVSize();
   int ndofs2 = vfes2p->GetTrueVSize();
   int gndofs1 = vfes1p->GlobalTrueVSize();
   int gndofs2 = vfes2p->GlobalTrueVSize();
   
   Array<int> npts(numprocs);
   MPI_Allgather(&npoints,1,MPI_INT,&npts[0],1,MPI_INT,comm);
   npts.PartialSum(); npts.Prepend(0);

   SparseMatrix S1(gnpoints,gndofs1);
   SparseMatrix S2(gnpoints,gndofs2);
   Array<SparseMatrix *> dS11;
   Array<SparseMatrix *> dS12;
   Array<SparseMatrix *> dS21;
   Array<SparseMatrix *> dS22;

   // local to global map for constraints
   Array<int> points_map(npoints);
   cnt = 0;
   for (int i = 0; i<gnpoints; i++)
   {
      if (i >= npts[myid] && i< npts[myid+1])
      {
         points_map[cnt++] = i;
      }
   }
   if (compute_hessians)
   {
      dS11.SetSize(gnpoints);
      dS12.SetSize(gnpoints);
      dS21.SetSize(gnpoints);
      dS22.SetSize(gnpoints);
      for (int i = 0; i<gnpoints; i++)
      {
         if (i >= npts[myid] && i< npts[myid+1])
         {
            dS11[i] = new SparseMatrix(gndofs1,gndofs1);
            dS12[i] = new SparseMatrix(gndofs1,gndofs2);
            dS21[i] = new SparseMatrix(gndofs2,gndofs1);
            dS22[i] = new SparseMatrix(gndofs2,gndofs2);
         }
         else
         {
            dS11[i] = nullptr;
            dS12[i] = nullptr;
            dS21[i] = nullptr;
            dS22[i] = nullptr;
         }
      }
      Assemble_Contact(xyz, xi1, coordsm, conn2, conn1, gapv, S1,S2,
                      dS11,dS12,dS21,dS22);
   }
   else
   {
      Assemble_Contact(xyz, xi1, coordsm, conn2, conn1, gapv, S1,S2, points_map);
   }                   
                
   // --------------------------------------------------------------------
   // Redistribute the M block matrix [M1 M2]
   // --------------------------------------------------------------------
   int offset = constraints_offsets[myid];
   MPICommunicator Mcomm1(comm,offset,gnpoints);
   SparseMatrix localS1(npoints,gndofs1);
   Mcomm1.Communicate(S1,localS1);
   MPICommunicator Mcomm2(comm,offset,gnpoints);
   SparseMatrix localS2(npoints,gndofs2);
   Mcomm2.Communicate(S2,localS2);
   
   MFEM_VERIFY(HYPRE_AssumedPartitionCheck(), "Hypre_AssumedPartitionCheck is False");

   // Construct M row and col starts to construct HypreParMatrix
   int M1rows[2], M2rows[2]; 
   int M1cols[2], M2cols[2];
   M1rows[0] = constraints_starts[0];
   M1rows[1] = constraints_starts[1];

   M2rows[0] = constraints_starts[0];
   M2rows[1] = constraints_starts[1];

   M1cols[0] = vfes1p->GetTrueDofOffsets()[0];
   M1cols[1] = vfes1p->GetTrueDofOffsets()[1];

   M2cols[0] = vfes2p->GetTrueDofOffsets()[0];
   M2cols[1] = vfes2p->GetTrueDofOffsets()[1];

   Array2D<HypreParMatrix*> blockM(1,2);
   blockM(0,0) = new HypreParMatrix(comm,npoints,gnpoints,gndofs1,
                          localS1.GetI(), localS1.GetJ(),localS1.GetData(),
                          M1rows,M1cols);

   blockM(0,1) = new HypreParMatrix(comm,npoints,gnpoints,gndofs2,
                          localS2.GetI(), localS2.GetJ(),localS2.GetData(),
                          M2rows,M2cols);

   Mp = HypreParMatrixFromBlocks(blockM);
   delete blockM(0,0);
   delete blockM(0,1);
   blockM.DeleteAll();

   if (compute_hessians)
   {
      Array<SparseMatrix*> localdS11(gnpoints);
      Array<SparseMatrix*> localdS12(gnpoints);
      Array<SparseMatrix*> localdS21(gnpoints);
      Array<SparseMatrix*> localdS22(gnpoints);
      for (int k = 0; k<gnpoints; k++)
      {
         localdS11[k] = new SparseMatrix(ndofs1,gndofs1); 
         localdS12[k] = new SparseMatrix(ndofs1,gndofs2); 
         localdS21[k] = new SparseMatrix(ndofs2,gndofs1); 
         localdS22[k] = new SparseMatrix(ndofs2,gndofs2); 
      }

      int offset1 = vfes1p->GetMyTDofOffset();
      int offset2 = vfes2p->GetMyTDofOffset();

      MPICommunicator dmcomm11(comm, offset1, gndofs1);
      dmcomm11.Communicate(dS11,localdS11);
      for (int k = 0; k<gnpoints; k++) { delete dS11[k]; }

      MPICommunicator dmcomm12(comm, offset1, gndofs1);
      dmcomm12.Communicate(dS12,localdS12);
      for (int k = 0; k<gnpoints; k++) { delete dS12[k]; }

      MPICommunicator dmcomm21(comm, offset2, gndofs2);
      dmcomm21.Communicate(dS21,localdS21);
      for (int k = 0; k<gnpoints; k++) { delete dS21[k]; }

      MPICommunicator dmcomm22(comm, offset2, gndofs2);
      dmcomm22.Communicate(dS22,localdS22);
      for (int k = 0; k<gnpoints; k++) { delete dS22[k]; }

      // --------------------------------------------------------------------
      // Redistribute the block dM matrices [dM11 dM12; dM21 dM22]
      // --------------------------------------------------------------------

      // Construct dMi HypreParMatrix
      Array2D<HypreParMatrix *> dMs(2,2);
      dMp.SetSize(gnpoints);
      int * offs1 = vfes1p->GetTrueDofOffsets();
      int * offs2 = vfes2p->GetTrueDofOffsets();
      for (int i = 0; i<gnpoints; i++)
      {
         dMs(0,0) = new HypreParMatrix(comm, ndofs1, gndofs1, gndofs1, 
                                      localdS11[i]->GetI(), localdS11[i]->GetJ(), 
                                      localdS11[i]->GetData(),
                                      offs1,offs1);
         delete localdS11[i];                                 
         dMs(0,1) = new HypreParMatrix(comm, ndofs1, gndofs1, gndofs2, 
                                      localdS12[i]->GetI(), localdS12[i]->GetJ(), 
                                      localdS12[i]->GetData(),
                                      offs1,offs2);   
         delete localdS12[i];                                 
         dMs(1,0) = new HypreParMatrix(comm, ndofs2, gndofs2, gndofs1, 
                                      localdS21[i]->GetI(), localdS21[i]->GetJ(), 
                                      localdS21[i]->GetData(),
                                      offs2,offs1);
         delete localdS21[i];                                 
         dMs(1,1) = new HypreParMatrix(comm, ndofs2, gndofs2, gndofs2, 
                                      localdS22[i]->GetI(), localdS22[i]->GetJ(), 
                                      localdS22[i]->GetData(),
                                      offs2,offs2);    
         delete localdS22[i];                                 

         dMp[i] = HypreParMatrixFromBlocks(dMs);
         delete dMs(0,0);
         delete dMs(0,1);
         delete dMs(1,0);
         delete dMs(1,1);
      }
      dMs.DeleteAll();
   }
}





#endif 

void ContactProblem::ComputeContactVertices()                      
{                                                                   
   if (npoints>0) return;                                           
   Mesh * mesh2 = prob2->GetMesh();                                 
   Array<int> vert;                                                 
   for (int b=0; b<mesh2->GetNBE(); b++)                            
   {                                                                
      if (mesh2->GetBdrAttribute(b) == 3)                           
      {                                                             
         mesh2->GetBdrElementVertices(b, vert);                     
         for (auto v : vert)                                        
         {                                                          
            contact_vertices.insert(v);                             
         }                                                          
      }                                                             
   }                                                                
   npoints = contact_vertices.size();                               
}  


void ContactProblem::ComputeGapFunctionAndDerivatives(const Vector &displ1, 
                                                      const Vector & displ2)
{
   ComputeContactVertices();

   Mesh * mesh1 = prob1->GetMesh();
   int dim = mesh1->Dimension();
   Mesh * mesh2 = prob2->GetMesh();

   int ndof1 = prob1->GetNumDofs();
   int ndof2 = prob2->GetNumDofs();
   int ndofs = ndof1 + ndof2;

   int nv1 = mesh1->GetNV();
   // connectivity of the second mesh

   Array<int> conn2(npoints); 
   // mesh2->MoveNodes(displ2);
   Vector xyz(dim * npoints);

   int cnt = 0;
   for (auto v : contact_vertices)
   {
      for (int d = 0; d<dim; d++)
      {
         xyz(cnt*dim + d) = mesh2->GetVertex(v)[d]+displ2[v*dim+d];
      }
      conn2[cnt] = v + nv1; 
      cnt++;
   }
   
   MFEM_VERIFY(cnt == npoints, "");
   gapv.SetSize(npoints*dim);

   // segment reference coordinates of the closest point
   Vector xi1(npoints*(dim-1));
   Array<int> conn1(npoints*4);
   
   // add(nodes0, displ1, *nodes1);
   FindPointsInMesh(*mesh1, xyz, conn1, xi1);

   DenseMatrix coordsm(npoints*4, dim);
   for (int i=0; i<npoints; i++)
   {
      for (int j=0; j<4; j++)
      {
         for (int k=0; k<dim; k++)
         {
            coordsm(i*4+j,k) = mesh1->GetVertex(conn1[i*4+j])[k]+displ1[dim*conn1[i*4+j]+k];
         }
      }
   }

   if (M)
   {
      delete M;
      for (int i = 0; i<dM.Size(); i++)
      {
         delete dM[i];
      }
      dM.SetSize(0);
   }

   int h = npoints;
   M = new SparseMatrix(h,ndofs);
   dM.SetSize(npoints);
   for (int i = 0; i<npoints; i++)
   {
      dM[i] = new SparseMatrix(ndofs,ndofs);
   }
   Assemble_Contact(xyz, xi1, coordsm, conn2, conn1, gapv, *M, dM);
}

