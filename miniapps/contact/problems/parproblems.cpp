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

   Array2D<HypreParMatrix*> A(2,2);
   A(0,0) = &prob1->GetOperator();
   A(1,1) = &prob2->GetOperator();
   K = HypreParMatrixFromBlocks(A);

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

   Array<int> conn2(npoints); 
   // mesh2->MoveNodes(displ2);
   Vector xyz(dim * npoints);

   int cnt = 0;
   for (auto v : contact_vertices)
   {
      for (int d = 0; d<dim; d++)
      {
         xyz(cnt*dim + d) = pmesh2->GetVertex(v)[d]+displ2[v*dim+d];
      }
      conn2[cnt] = globalvertices2[v];
      cnt++;
   }

   MFEM_VERIFY(cnt == npoints, "");
   gapv.SetSize(npoints*dim);


   // segment reference coordinates of the closest point
   Vector xi1(npoints*(dim-1));
   Array<int> conn1(npoints*4);
   DenseMatrix coordsm(npoints*4, dim);
   
   add(nodes0, displ1, *nodes1);

   FindPointsInMesh(*pmesh1, globalvertices1, conn2, displ1, xyz, conn1, xi1, coordsm);


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

   Assemble_Contact(gnv, xyz, xi1, coordsm, conn2, conn1, gapv, S, dS, reduced);

   // --------------------------------------------------------------------
   // Redistribute the M matrix
   // --------------------------------------------------------------------
   int offset = reduced ? constraints_offsets[myid] : vertex_offsets[myid];
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
}


double ParContactProblem::E(const Vector & d)
{

}

void ParContactProblem::DdE(const Vector &d, Vector &gradE)
{

}

HypreParMatrix* ParContactProblem::DddE(const Vector &d)
{

}

void ParContactProblem::g(const Vector &d, Vector &gd, bool reduced)
{

}

HypreParMatrix* ParContactProblem::Ddg(const Vector &d)
{

}

HypreParMatrix* ParContactProblem::lDddg(const Vector &d, const Vector &l)
{

}


