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
      SparseMatrix * ISparse = new SparseMatrix(negOneDiag);
      Ih = new HypreParMatrix(MPI_COMM_WORLD, dimMGlb, dofOffsetsM, ISparse);
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


double OptProblem::CalcObjective(const BlockVector &x) { return E(x.GetBlock(0)); }

void OptProblem::Duf(const BlockVector &x, Vector &y) { DdE(x.GetBlock(0), y); }

void OptProblem::Dmf(const BlockVector &x, Vector &y) { y = 0.0; }


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


void OptProblem::c(const BlockVector &x, Vector &y) // c(u,m) = g(u) - m 
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


double ObstacleProblem::E(const Vector &d)
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

void ObstacleProblem::DdE(const Vector &d, Vector &gradE)
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
void ObstacleProblem::g(const Vector &d, Vector &gd)
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

#ifdef MFEM_USE_MPI
   void ParElasticityProblem::Init()
   {
      int dim = pmesh->Dimension();
      fec = new H1_FECollection(order,dim);
      fes = new ParFiniteElementSpace(pmesh,fec,dim,Ordering::byVDIM);
      ndofs = fes->GetVSize();
      ntdofs = fes->GetTrueVSize();
      gndofs = fes->GlobalTrueVSize();
      pmesh->SetNodalFESpace(fes);
      if (pmesh->bdr_attributes.Size())
      {
         ess_bdr.SetSize(pmesh->bdr_attributes.Max());
      }
      ess_bdr = 0; ess_bdr[1] = 1;
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list);
      // Solution GridFunction
      x.SetSpace(fes);  x = 0.0;
      // RHS
      b.Update(fes);
   
      // Elasticity operator
      lambda.SetSize(pmesh->attributes.Max()); lambda = 57.6923076923;
      mu.SetSize(pmesh->attributes.Max()); mu = 38.4615384615;
   
      lambda_cf.UpdateConstants(lambda);
      mu_cf.UpdateConstants(mu);
   
      a = new ParBilinearForm(fes);
      a->AddDomainIntegrator(new ElasticityIntegrator(lambda_cf,mu_cf));
   }
   
   void ParElasticityProblem::FormLinearSystem()
   {
      if (!formsystem) 
      {
         formsystem = true;
         b.Assemble();
         a->Assemble();
         a->FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      }
   }
   
   void ParElasticityProblem::UpdateLinearSystem()
   {
      if (formsystem)
      {
         b.Update();
         a->Update();
         formsystem = false;
      }
      FormLinearSystem();
   }


   ParContactProblem::ParContactProblem(ParElasticityProblem * prob1_, ParElasticityProblem * prob2_)
   : OptProblem(), prob1(prob1_), prob2(prob2_)
   {
      ParMesh* pmesh1 = prob1->GetMesh();
      comm = pmesh1->GetComm();
      MPI_Comm_rank(comm, &myid);
      MPI_Comm_size(comm, &numprocs);
    
      dim = pmesh1->Dimension();
      nodes0.SetSpace(pmesh1->GetNodes()->FESpace());
      nodes0 = *pmesh1->GetNodes();
      nodes1 = pmesh1->GetNodes();
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
   
      Array2D<HypreParMatrix*> A(2,2);
      A(0,0) = &prob1->GetOperator();
      A(1,1) = &prob2->GetOperator();
      A(1,0) = nullptr;
      A(0,1) = nullptr;
      K = HypreParMatrixFromBlocks(A);
   
      B = new BlockVector(tdof_offsets);
      B->GetBlock(0).Set(1.0, prob1->GetRHS());
      B->GetBlock(1).Set(1.0, prob2->GetRHS());
   
      ComputeContactVertices();

      // to do: 
      // use tdof_offsets and constraint_starts
      // to generate parallel partitioning info
      // needed to call Init
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
   
   void ParContactProblem::ComputeContactVertices()
   {
      if (gnpoints>0) return;
   
      ParMesh * pmesh1 = prob1->GetMesh();
      ParMesh * pmesh2 = prob2->GetMesh();
      dim = pmesh1->Dimension();
   
      vfes1 = new ParFiniteElementSpace(pmesh1, prob1->GetFECol());
      vfes2 = new ParFiniteElementSpace(pmesh2, prob2->GetFECol());
   
      int gnv1 = vfes1->GlobalTrueVSize();
      int gnv2 = vfes2->GlobalTrueVSize();
      gnv = gnv1+gnv2;
      int nv1 = vfes1->GetTrueVSize();
      int nv2 = vfes2->GetTrueVSize();
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
   
      int voffset2 = vfes2->GetMyTDofOffset();
   
      std::vector<int> vertex2_offsets;
      ComputeTdofOffsets(comm,voffset2, vertex2_offsets);
   
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
   
   void ParContactProblem::ComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector &displ2)
   {
      ComputeContactVertices();
      ParMesh * pmesh1 = prob1->GetMesh();
      ParMesh * pmesh2 = prob2->GetMesh();
   
      ParGridFunction displ1_gf(prob1->GetFESpace());
      ParGridFunction displ2_gf(prob2->GetFESpace());
   
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
      if (M)
      {
         delete M;
         for (int i = 0; i<dM.Size(); i++)
         {
            delete dM[i];
         }
         dM.SetSize(0);
      }
   
      int ndofs1 = prob1->GetFESpace()->GetTrueVSize();
      int ndofs2 = prob2->GetFESpace()->GetTrueVSize();
      int gndofs1 = prob1->GetFESpace()->GlobalTrueVSize();
      int gndofs2 = prob2->GetFESpace()->GlobalTrueVSize();
      
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
   
      M1cols[0] = prob1->GetFESpace()->GetTrueDofOffsets()[0];
      M1cols[1] = prob1->GetFESpace()->GetTrueDofOffsets()[1];
   
      M2cols[0] = prob2->GetFESpace()->GetTrueDofOffsets()[0];
      M2cols[1] = prob2->GetFESpace()->GetTrueDofOffsets()[1];
   
      Array2D<HypreParMatrix*> blockM(1,2);
      blockM(0,0) = new HypreParMatrix(comm,npoints,gnpoints,gndofs1,
                             localS1.GetI(), localS1.GetJ(),localS1.GetData(),
                             M1rows,M1cols);
   
      blockM(0,1) = new HypreParMatrix(comm,npoints,gnpoints,gndofs2,
                             localS2.GetI(), localS2.GetJ(),localS2.GetData(),
                             M2rows,M2cols);
   
      M = HypreParMatrixFromBlocks(blockM);
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
   
         int offset1 = prob1->GetFESpace()->GetMyTDofOffset();
         int offset2 = prob2->GetFESpace()->GetMyTDofOffset();
   
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
         dM.SetSize(gnpoints);
         int * offs1 = prob1->GetFESpace()->GetTrueDofOffsets();
         int * offs2 = prob2->GetFESpace()->GetTrueDofOffsets();
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
   
            dM[i] = HypreParMatrixFromBlocks(dMs);
            delete dMs(0,0);
            delete dMs(0,1);
            delete dMs(1,0);
            delete dMs(1,1);
         }
         dMs.DeleteAll();
      }
   }
   
   double ParContactProblem::E(const Vector & d)
   {
      Vector kd(K->Height());
      K->Mult(d,kd);
      return 0.5 * InnerProduct(comm,d, kd) - InnerProduct(comm,d, *B);
   }
   
   void ParContactProblem::DdE(const Vector &d, Vector &gradE)
   {
      gradE.SetSize(K->Height());
      K->Mult(d, gradE);
      gradE.Add(-1.0, *B); 
   }
   
   HypreParMatrix* ParContactProblem::DddE(const Vector &d)
   {
      return K; 
   }
   
   void ParContactProblem::g(const Vector &d, Vector &gd)//, bool compute_hessians_)
   {
      compute_hessians = true;//compute_hessians_;
      int ndof1 = prob1->GetNumTDofs();
      int ndof2 = prob2->GetNumTDofs();
      double * data = d.GetData();
      Vector displ1(data,ndof1);
      Vector displ2(&data[ndof1],ndof2);
   
      if (recompute)
      {
         ComputeGapFunctionAndDerivatives(displ1, displ2);
         recompute = false;
      }
   
      gd = GetGapFunction();
   }
   
   HypreParMatrix* ParContactProblem::Ddg(const Vector &d)
   {
     return GetJacobian();
   }
   
   HypreParMatrix* ParContactProblem::lDddg(const Vector &d, const Vector &l)
   {
      return nullptr; // for now
   }
#endif
