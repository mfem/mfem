#include "parproblems.hpp"

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
: prob1(prob1_), prob2(prob2_)
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

   blockK = new BlockOperator(tdof_offsets);
   blockK->SetBlock(0,0,&prob1->GetOperator());
   blockK->SetBlock(1,1,&prob2->GetOperator());
   blockK->owns_blocks=0;

   Array2D<HypreParMatrix*> A(2,2);
   A(0,0) = &prob1->GetOperator();
   A(1,1) = &prob2->GetOperator();
   A(1,0) = nullptr;
   A(0,1) = nullptr;
   K = HypreParMatrixFromBlocks(A);

   HypreBoomerAMG * amg1 = new HypreBoomerAMG(prob1->GetOperator());
   amg1->SetElasticityOptions(prob1->GetFESpace());
   amg1->SetPrintLevel(0);
   HypreBoomerAMG * amg2 = new HypreBoomerAMG(prob2->GetOperator());
   amg2->SetElasticityOptions(prob2->GetFESpace());
   amg2->SetPrintLevel(0);

   prec = new BlockDiagonalPreconditioner(tdof_offsets);
   prec->owns_blocks = 1;
   prec->SetDiagonalBlock(0,amg1);
   prec->SetDiagonalBlock(1,amg2);

   B = new BlockVector(tdof_offsets);
   B->GetBlock(0).Set(1.0, prob1->GetRHS());
   B->GetBlock(1).Set(1.0, prob2->GetRHS());

   ComputeContactVertices();
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
   add(nodes0, displ1_gf, *nodes1);
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

   Array<SparseMatrix *> dS11(gnpoints);
   Array<SparseMatrix *> dS12(gnpoints);
   Array<SparseMatrix *> dS21(gnpoints);
   Array<SparseMatrix *> dS22(gnpoints);
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

   int offset = constraints_offsets[myid];
   SparseMatrix S1(gnpoints,gndofs1);
   SparseMatrix S2(gnpoints,gndofs2);
   Assemble_Contact(xyz, xi1, coordsm, conn2, conn1, gapv, S1,S2,
                   dS11,dS12,dS21,dS22);

   // --------------------------------------------------------------------
   // Redistribute the M block matrix [M1 M2]
   // --------------------------------------------------------------------
   MPICommunicator Mcomm1(comm,offset,gnpoints);
   SparseMatrix localS1(npoints,gndofs1);
   Mcomm1.Communicate(S1,localS1);

   MPICommunicator Mcomm2(comm,offset,gnpoints);
   SparseMatrix localS2(npoints,gndofs2);
   Mcomm2.Communicate(S2,localS2);

   // --------------------------------------------------------------------
   // Redistribute the block dM matrices [dM11 dM12; dM21 dM22]
   // --------------------------------------------------------------------
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
   MPICommunicator dmcomm12(comm, offset1, gndofs1);
   MPICommunicator dmcomm21(comm, offset2, gndofs2);
   MPICommunicator dmcomm22(comm, offset2, gndofs2);
   dmcomm11.Communicate(dS11,localdS11);
   dmcomm12.Communicate(dS12,localdS12);
   dmcomm21.Communicate(dS21,localdS21);
   dmcomm22.Communicate(dS22,localdS22);

   for (int k = 0; k<gnpoints; k++)
   {
      delete dS11[k]; 
      delete dS12[k]; 
      delete dS21[k]; 
      delete dS22[k]; 
   }

   // --------------------------------------------------------------------

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

   M1 = new HypreParMatrix(comm,npoints,gnpoints,gndofs1,
                          localS1.GetI(), localS1.GetJ(),localS1.GetData(),
                          M1rows,M1cols);

   M2 = new HypreParMatrix(comm,npoints,gnpoints,gndofs2,
                          localS2.GetI(), localS2.GetJ(),localS2.GetData(),
                          M2rows,M2cols);

   Array2D<HypreParMatrix*> Marray(1,2);
   Marray(0,0) = M1; Marray(0,1) = M2;
   M = HypreParMatrixFromBlocks(Marray);

   Mrow_offsets.SetSize(2);
   Mrow_offsets[0] = 0;
   Mrow_offsets[1] = npoints;

   blockM = new BlockOperator(Mrow_offsets, tdof_offsets);
   blockM->SetBlock(0,0,M1);
   blockM->SetBlock(0,1,M2);
   blockM->owns_blocks = 0;
   
   // Construct dMi HypreParMatrix
   Array2D<HypreParMatrix *> dMs(2,2);
   dM11.SetSize(gnpoints);
   dM12.SetSize(gnpoints);
   dM21.SetSize(gnpoints);
   dM22.SetSize(gnpoints);
   dM.SetSize(gnpoints);
   int * offs1 = prob1->GetFESpace()->GetTrueDofOffsets();
   int * offs2 = prob2->GetFESpace()->GetTrueDofOffsets();
   blockdM.SetSize(gnpoints);
   for (int i = 0; i<gnpoints; i++)
   {
      dM11[i] = new HypreParMatrix(comm, ndofs1, gndofs1, gndofs1, 
                                 localdS11[i]->GetI(), localdS11[i]->GetJ(), localdS11[i]->GetData(),
                                 offs1,offs1);
      delete localdS11[i];                                 
      dM12[i] = new HypreParMatrix(comm, ndofs1, gndofs1, gndofs2, 
                                 localdS12[i]->GetI(), localdS12[i]->GetJ(), localdS12[i]->GetData(),
                                 offs1,offs2);   
      delete localdS12[i];                                 
      dM21[i] = new HypreParMatrix(comm, ndofs2, gndofs2, gndofs1, 
                                 localdS21[i]->GetI(), localdS21[i]->GetJ(), localdS21[i]->GetData(),
                                 offs2,offs1);
      delete localdS21[i];                                 
      dM22[i] = new HypreParMatrix(comm, ndofs2, gndofs2, gndofs2, 
                                 localdS22[i]->GetI(), localdS22[i]->GetJ(), localdS22[i]->GetData(),
                                 offs2,offs2);    
      delete localdS22[i];                                 
      blockdM[i] = new BlockOperator(tdof_offsets);
      blockdM[i]->SetBlock(0,0,dM11[i]);                                                                                                                                  
      blockdM[i]->SetBlock(0,1,dM12[i]);                                                                                                                                  
      blockdM[i]->SetBlock(1,0,dM21[i]);                                                                                                                                  
      blockdM[i]->SetBlock(1,1,dM22[i]);                                                                                                                                  
      blockdM[i]->owns_blocks = 0;                                                                                                                                  
      dMs(0,0) = dM11[i];
      dMs(0,1) = dM12[i];
      dMs(1,0) = dM21[i];
      dMs(1,1) = dM22[i];
      dM[i] = HypreParMatrixFromBlocks(dMs);
   }
}

double ParContactProblem::E(const Vector & d)
{
   Vector kd(K->Height());
   blockK->Mult(d,kd);
   return 0.5 * InnerProduct(comm,d, kd) - InnerProduct(comm,d, *B);
}

void ParContactProblem::DdE(const Vector &d, Vector &gradE)
{
   gradE.SetSize(K->Height());
   blockK->Mult(d, gradE);
   gradE.Add(-1.0, *B); 
}

HypreParMatrix* ParContactProblem::DddE(const Vector &d)
{
   return K; 
}
BlockOperator* ParContactProblem::blockDddE(const Vector &d)
{
   return blockK; 
}

void ParContactProblem::g(const Vector &d, Vector &gd)
{
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

BlockOperator* ParContactProblem::blockDdg(const Vector &d)
{
  return GetBlockJacobian();
}

HypreParMatrix* ParContactProblem::lDddg(const Vector &d, const Vector &l)
{
   return nullptr; // for now
}


QPOptParContactProblem::QPOptParContactProblem(ParContactProblem * problem_)
: problem(problem_)
{
   dimU = problem->GetNumDofs();
   dimM = problem->GetNumContraints();
   dimC = problem->GetNumContraints();
   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = dimU;
   block_offsets[2] = dimM;
   block_offsets.PartialSum();
   ml.SetSize(dimM); ml = 0.0;
   Vector negone(dimM); negone = -1.0;
   SparseMatrix diag(negone);

   int gsize = problem->GetGlobalNumConstraints();
   int * rows = problem->GetConstraintsStarts().GetData();

   NegId = new HypreParMatrix(problem->GetComm(),gsize, rows,&diag);
   HypreStealOwnership(*NegId, diag);
}

int QPOptParContactProblem::GetDimU() { return dimU; }

int QPOptParContactProblem::GetDimM() { return dimM; }

int QPOptParContactProblem::GetDimC() { return dimC; }

Vector & QPOptParContactProblem::Getml() { return ml; }

HypreParMatrix * QPOptParContactProblem::Duuf(const BlockVector & x)
{
   return problem->DddE(x.GetBlock(0));
}

BlockOperator * QPOptParContactProblem::blockDuuf(const BlockVector & x)
{
   return problem->blockDddE(x.GetBlock(0));
}

HypreParMatrix * QPOptParContactProblem::Dumf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblem::Dmuf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblem::Dmmf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblem::Duc(const BlockVector & x)
{
   return problem->Ddg(x.GetBlock(0));
}

BlockOperator * QPOptParContactProblem::blockDuc(const BlockVector & x)
{
   return problem->blockDdg(x.GetBlock(0));
}

HypreParMatrix * QPOptParContactProblem::Dmc(const BlockVector & x)
{
   return NegId;
}

HypreParMatrix * QPOptParContactProblem::lDuuc(const BlockVector & x, const Vector & l)
{
   return nullptr;
}

void QPOptParContactProblem::c(const BlockVector &x, Vector & y)
{
   Vector g0;
   problem->g(x.GetBlock(0),g0); // gap function
   g0.Add(-1.0, x.GetBlock(1));  
   problem->GetJacobian()->Mult(x.GetBlock(0),y);
   y.Add(1.0, g0);
}

double QPOptParContactProblem::CalcObjective(const BlockVector & x)
{
   return problem->E(x.GetBlock(0));
}

void QPOptParContactProblem::CalcObjectiveGrad(const BlockVector & x, BlockVector & y)
{
   problem->DdE(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
}

QPOptParContactProblem::~QPOptParContactProblem()
{
   delete NegId;
}
