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

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = ndof1;
   offsets[2] = ndof2;
   offsets.PartialSum();
   B = new BlockVector(offsets);
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

   Array<int> vertices1(pmesh1->GetNV());
   Array<int> vertices2(pmesh2->GetNV());

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

   int voffset1 = vfes1->GetMyTDofOffset();
   int voffset2 = vfes2->GetMyTDofOffset();
   int voffset = voffset1 + voffset2;

   std::vector<int> vertex1_offsets;
   ComputeTdofOffsets(comm,voffset1,vertex1_offsets);
   std::vector<int> vertex2_offsets;
   ComputeTdofOffsets(comm,voffset2, vertex2_offsets);
   ComputeTdofOffsets(comm,voffset, vertex_offsets);
   globalvertices1.SetSize(pmesh1->GetNV());
   globalvertices2.SetSize(pmesh2->GetNV());
   for (int i = 0; i<pmesh1->GetNV(); i++)
   {
      int rank = get_rank(vertices1[i],vertex1_offsets);
      globalvertices1[i] = vertices1[i] + vertex2_offsets[rank];
   }

   std::vector<int> vertex1_tdoffs;
   ComputeTdofOffsets(comm,nv1, vertex1_tdoffs);

   for (int i = 0; i<pmesh2->GetNV(); i++)
   {
      int rank = get_rank(vertices2[i],vertex2_offsets);
      globalvertices2[i] = vertices2[i] + vertex1_offsets[rank] + vertex1_tdoffs[rank];
   }

   Array<int> vert;
   for (int b=0; b<pmesh2->GetNBE(); b++)
   {
      if (pmesh2->GetBdrAttribute(b) == 3)
      {
         pmesh2->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            if (myid != get_rank(globalvertices2[v],vertex_offsets)) { continue; }
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

void ParContactProblem::ComputeGapFunctionAndDerivatives(const Vector & displ1, const Vector &displ2, bool reduced)
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
      conn2[cnt] = globalvertices2[v];
      cnt++;
   }

   MFEM_VERIFY(cnt == npoints, "");
   gapv.SetSize(npoints*dim); gapv = 0.0;

   // segment reference coordinates of the closest point
   Vector xi1(npoints*(dim-1));
   Array<int> conn1(npoints*4);
   DenseMatrix coordsm(npoints*4, dim);
   add(nodes0, displ1_gf, *nodes1);

   FindPointsInMesh(*pmesh1, globalvertices1, conn2, displ1_gf, xyz, conn1, xi1, coordsm);

   if (M)
   {
      delete M;
      for (int i = 0; i<dM.Size(); i++)
      {
         delete dM[i];
      }
      dM.SetSize(0);
   }

   int h = (reduced) ? npoints : nv;
   int gh = (reduced) ? gnpoints : gnv;

   int ndofs = prob1->GetFESpace()->GetTrueVSize() + prob2->GetFESpace()->GetTrueVSize();
   int gndofs = prob1->GetFESpace()->GlobalTrueVSize() + prob2->GetFESpace()->GlobalTrueVSize();
   
   SparseMatrix S(gh,gndofs);

   Array<int> npts(numprocs);
   MPI_Allgather(&npoints,1,MPI_INT,&npts[0],1,MPI_INT,MPI_COMM_WORLD);
   npts.PartialSum(); npts.Prepend(0);

   int gnpts = npts[numprocs];
   Array<SparseMatrix *> dS(gnpts);
   for (int i = 0; i<gnpts; i++)
   {
      if (i >= npts[myid] && i< npts[myid+1])
      {
         dS[i] = new SparseMatrix(gndofs,gndofs);
      }
      else
      {
         dS[i] = nullptr;
      }
   }

   int offset = reduced ? constraints_offsets[myid] : vertex_offsets[myid];
   Assemble_Contact(gnv, xyz, xi1, coordsm, conn2, conn1, gapv, S, dS, reduced, offset);

   // --------------------------------------------------------------------
   // Redistribute the M matrix
   // --------------------------------------------------------------------
   int glv = reduced ? gnpoints : gnv;
   MPICommunicator Mcomm(comm,offset,glv);
   SparseMatrix localS(h,K->GetGlobalNumCols());
   Mcomm.Communicate(S,localS);

   // --------------------------------------------------------------------
   // Redistribute the dM_i matrices
   // --------------------------------------------------------------------
   MPICommunicator dmcomm(K->GetComm(), K->RowPart()[0], gndofs);
   Array<SparseMatrix*> localdSs(gnpts);
   for (int k = 0; k<gnpts; k++)
   {
      localdSs[k] = new SparseMatrix(ndofs,gndofs); 
   }
   dmcomm.Communicate(dS,localdSs);
   for (int i = 0; i<gnpts; i++)
   {
      delete dS[i];
   }
   // --------------------------------------------------------------------

   MFEM_VERIFY(HYPRE_AssumedPartitionCheck(), "Hypre_AssumedPartitionCheck is False");

   // Construct M row and col starts to construct HypreParMatrix
   int Mrows[2]; 
   int Mcols[2]; 
   int nrows, gnrows;
   if (reduced)
   {
      Mrows[0] = constraints_starts[0];
      Mrows[1] = constraints_starts[1];
      nrows = npoints;
      gnrows = gnpoints;
   }
   else
   {
      Mrows[0] = vertex_offsets[myid];
      Mrows[1] = vertex_offsets[myid]+nv;
      nrows = nv;
      gnrows = gnv;
   }
   Mcols[0] = K->ColPart()[0]; 
   Mcols[1] = K->ColPart()[1]; 
   M = new HypreParMatrix(K->GetComm(),nrows,gnrows,gndofs,
                          localS.GetI(), localS.GetJ(),localS.GetData(),
                          Mrows,Mcols);
   HypreStealOwnership(*M, localS);

   // TODO Construct Mi HypreParMatrix
   // ...

   for (int k = 0; k<gnpts; k++)
   {
      delete localdSs[k]; 
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

void ParContactProblem::g(const Vector &d, Vector &gd, bool reduced)
{
   int ndof1 = prob1->GetNumTDofs();
   int ndof2 = prob2->GetNumTDofs();
   double * data = d.GetData();
   Vector displ1(data,ndof1);
   Vector displ2(&data[ndof1],ndof2);

   if (recompute)
   {
      ComputeGapFunctionAndDerivatives(displ1, displ2, reduced);
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
   problem->g(x.GetBlock(0),g0, true); // gap function
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
