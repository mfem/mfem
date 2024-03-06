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
   ess_bdr = 0; 
   Array<int> ess_tdof_list_temp;
   for (int i = 0; i < ess_bdr_attr.Size(); i++ )
   {
      ess_bdr[ess_bdr_attr[i]-1] = 1;
      fes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list_temp,ess_bdr_attr_comp[i]);
      ess_tdof_list.Append(ess_tdof_list_temp);
      ess_bdr[ess_bdr_attr[i]-1] = 0;
   }
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
      if (pmesh2->GetBdrAttribute(b) == 4)
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

void ParContactProblem::g(const Vector &d, Vector &gd, bool compute_hessians_)
{
   compute_hessians = compute_hessians_;
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


QPOptParContactProblem::QPOptParContactProblem(ParContactProblem * problem_)
: problem(problem_)
{
   dimU = problem->GetNumDofs();
   dimM = problem->GetNumContraints();
   dimC = problem->GetNumContraints();
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
   problem->g(x.GetBlock(0),g0, false); // gap function
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



#ifdef MFEM_USE_TRIBOL

ParContactProblemTribol::ParContactProblemTribol(ParElasticityProblem * prob_)
: prob(prob_)
{
   pmesh = prob->GetMesh();
   vfes = prob->GetFESpace();
   comm = pmesh->GetComm();
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &numprocs);
   prob->FormLinearSystem();
   K= new HypreParMatrix(prob->GetOperator());
   B = new Vector(prob->GetRHS());

   SetupTribol();

}

void ParContactProblemTribol::SetupTribol()
{
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());

   // plane of bottom block
   std::set<int> mortar_attrs({3});
   // plane of top block
   std::set<int> nonmortar_attrs({4});

   // Initialize Tribol contact library
   tribol::initialize(3, MPI_COMM_WORLD);

   int coupling_scheme_id = 0;
   int mesh1_id = 0;
   int mesh2_id = 1;
   coords = new ParGridFunction(vfes);
   pmesh->SetNodalGridFunction(coords);
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id, mesh1_id, mesh2_id,
      *pmesh, *coords, mortar_attrs, nonmortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_SLIDING,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   // Access Tribol's pressure grid function (on the contact surface)
   auto& pressure = tribol::getMfemPressure(coupling_scheme_id);
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of pressure unknowns: " <<
                pressure.ParFESpace()->GlobalTrueVSize() << std::endl;
   }

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Update contact gaps, forces, and tangent stiffness
   int cycle = 1;   // pseudo cycle
   double t = 1.0;  // pseudo time
   double dt = 1.0; // pseudo dt
   tribol::update(cycle, t, dt);

   // Return contact contribution to the tangent stiffness matrix
   auto A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);
   
   J = new HypreParMatrix(*(HypreParMatrix *)(&A_blk->GetBlock(1,0)));

   Vector gap;
   tribol::getMfemGap(coupling_scheme_id, gap);
   auto& P_submesh = *pressure.ParFESpace()->GetProlongationMatrix();
   gapv.SetSize(P_submesh.Width());
   P_submesh.MultTranspose(gap,gapv);
   constraints_starts.SetSize(2);
   constraints_starts[0] = J->RowPart()[0];
   constraints_starts[1] = J->RowPart()[1];
}

double ParContactProblemTribol::E(const Vector & d)
{
   Vector kd(K->Height());
   K->Mult(d,kd);
   return 0.5 * InnerProduct(comm,d, kd) - InnerProduct(comm,d, *B);
}

void ParContactProblemTribol::DdE(const Vector &d, Vector &gradE)
{
   gradE.SetSize(K->Height());
   K->Mult(d, gradE);
   gradE.Add(-1.0, *B); 
}

HypreParMatrix* ParContactProblemTribol::DddE()
{
   return K; 
}

void ParContactProblemTribol::g(Vector & gd)
{
   gd = GetGapFunction();
}

HypreParMatrix* ParContactProblemTribol::Ddg()
{
  return GetJacobian();
}


QPOptParContactProblemTribol::QPOptParContactProblemTribol(ParContactProblemTribol * problem_)
: problem(problem_)
{
   dimU = problem->GetNumDofs();
   dimM = problem->GetNumContraints();
   dimC = problem->GetNumContraints();
   ml.SetSize(dimM); ml = 0.0;
   Vector negone(dimM); negone = -1.0;
   SparseMatrix diag(negone);

   int gsize = problem->GetGlobalNumConstraints();
   int * rows = problem->GetConstraintsStarts().GetData();

   NegId = new HypreParMatrix(problem->GetComm(),gsize, rows,&diag);
   HypreStealOwnership(*NegId, diag);
}

int QPOptParContactProblemTribol::GetDimU() { return dimU; }

int QPOptParContactProblemTribol::GetDimM() { return dimM; }

int QPOptParContactProblemTribol::GetDimC() { return dimC; }

Vector & QPOptParContactProblemTribol::Getml() { return ml; }

HypreParMatrix * QPOptParContactProblemTribol::Duuf(const BlockVector & x)
{
   return problem->DddE();
}

HypreParMatrix * QPOptParContactProblemTribol::Dumf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblemTribol::Dmuf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblemTribol::Dmmf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblemTribol::Duc(const BlockVector & x)
{
   return problem->Ddg();
}

HypreParMatrix * QPOptParContactProblemTribol::Dmc(const BlockVector & x)
{
   return NegId;
}

HypreParMatrix * QPOptParContactProblemTribol::lDuuc(const BlockVector & x, const Vector & l)
{
   return nullptr;
}

void QPOptParContactProblemTribol::c(const BlockVector &x, Vector & y)
{
   Vector g0;
   problem->g(g0); // gap function
   g0.Add(-1.0, x.GetBlock(1));  
   HypreParMatrix *J = problem->GetJacobian();
   J->Mult(x.GetBlock(0),y);
   y.Add(1.0, g0);
}

double QPOptParContactProblemTribol::CalcObjective(const BlockVector & x)
{
   return problem->E(x.GetBlock(0));
}

void QPOptParContactProblemTribol::CalcObjectiveGrad(const BlockVector & x, BlockVector & y)
{
   problem->DdE(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
}

QPOptParContactProblemTribol::~QPOptParContactProblemTribol()
{
   delete NegId;
}

#endif


ParContactProblemSingleMesh::ParContactProblemSingleMesh(ParElasticityProblem * prob_, bool enable_tribol_)
: prob(prob_), enable_tribol(enable_tribol_)
{
   ParMesh* pmesh = prob->GetMesh();
   comm = pmesh->GetComm();
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &numprocs);
 
   dim = pmesh->Dimension();
   nodes0.SetSpace(pmesh->GetNodes()->FESpace());
   nodes0 = *pmesh->GetNodes();
   nodes1 = pmesh->GetNodes();
   
   prob->FormLinearSystem();
   K= new HypreParMatrix(prob->GetOperator());
   B = new Vector(prob->GetRHS());
   if (enable_tribol)
   {
      SetupTribol();
   }
   else
   {
      ComputeContactVertices();
   }
}

void ParContactProblemSingleMesh::ComputeContactVertices()
{
   if (gnpoints>0) return;

   ParMesh * pmesh = prob->GetMesh();
   dim = pmesh->Dimension();

   vfes = new ParFiniteElementSpace(pmesh, prob->GetFECol());

   vertices.SetSize(pmesh->GetNV());

   for (int i = 0; i<pmesh->GetNV(); i++)
   {
      vertices[i] = i;
   }
   pmesh->GetGlobalVertexIndices(vertices);

   int voffset = vfes->GetMyTDofOffset();

   std::vector<int> vertex_offsets;
   ComputeTdofOffsets(comm,voffset, vertex_offsets);

   Array<int> vert;
   for (int b=0; b<pmesh->GetNBE(); b++)
   {
      if (pmesh->GetBdrAttribute(b) == 4)
      {
         pmesh->GetBdrElementVertices(b, vert);
         for (auto v : vert)
         {
            if (myid != get_rank(vertices[v],vertex_offsets)) { continue; }
            contact_vertices.insert(v);
         }
      }
   }

   npoints = contact_vertices.size();
   MPI_Allreduce(&npoints, &gnpoints,1,MPI_INT,MPI_SUM,pmesh->GetComm());
   int constrains_offset;
   MPI_Scan(&npoints,&constrains_offset,1,MPI_INT,MPI_SUM,pmesh->GetComm());

   constrains_offset-=npoints;
   constraints_starts.SetSize(2);
   constraints_starts[0] = constrains_offset;
   constraints_starts[1] = constrains_offset+npoints;

   ComputeTdofOffsets(comm,constrains_offset, constraints_offsets);
}

void ParContactProblemSingleMesh::ComputeGapFunctionAndDerivatives(const Vector & displ)
{
   ComputeContactVertices();
   ParMesh * pmesh = prob->GetMesh();

   ParGridFunction displ_gf(prob->GetFESpace());

   displ_gf.SetFromTrueDofs(displ);

   Array<int> conn2(npoints); 
   Vector xyz(dim * npoints);

   int cnt = 0;
   for (auto v : contact_vertices)
   {
      for (int d = 0; d<dim; d++)
      {
         xyz(cnt*dim + d) = pmesh->GetVertex(v)[d]+displ_gf[v*dim+d];
      }
      conn2[cnt] = vertices[v];
      cnt++;
   }

   MFEM_VERIFY(cnt == npoints, "");
   gapv.SetSize(npoints*dim); gapv = 0.0;
   // segment reference coordinates of the closest point
   Vector xi1(npoints*(dim-1));
   Array<int> conn1(npoints*4);
   DenseMatrix coordsm(npoints*4, dim);
   // add(nodes0, displ1_gf, *nodes1);
   FindPointsInMesh(*pmesh, vertices, conn2, displ_gf, xyz, conn1, xi1, coordsm, true);
   if (M)
   {
      delete M;
      for (int i = 0; i<dM.Size(); i++)
      {
         delete dM[i];
      }
      dM.SetSize(0);
   }

   int gndofs = prob->GetFESpace()->GlobalTrueVSize();
   
   Array<int> npts(numprocs);
   MPI_Allgather(&npoints,1,MPI_INT,&npts[0],1,MPI_INT,comm);
   npts.PartialSum(); npts.Prepend(0);

   SparseMatrix S(gnpoints,gndofs);

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

   Assemble_Contact(xyz, xi1, coordsm, conn2, conn1, gapv, S, points_map);
                
   // --------------------------------------------------------------------
   // Redistribute the M block matrix [M1 M2]
   // --------------------------------------------------------------------
   int offset = constraints_offsets[myid];
   MPICommunicator Mcomm(comm,offset,gnpoints);
   SparseMatrix localS(npoints,gndofs);
   Mcomm.Communicate(S,localS);
   
   MFEM_VERIFY(HYPRE_AssumedPartitionCheck(), "Hypre_AssumedPartitionCheck is False");

   // Construct M row and col starts to construct HypreParMatrix
   int Mrows[2]; 
   int Mcols[2];
   Mrows[0] = constraints_starts[0];
   Mrows[1] = constraints_starts[1];

   Mcols[0] = prob->GetFESpace()->GetTrueDofOffsets()[0];
   Mcols[1] = prob->GetFESpace()->GetTrueDofOffsets()[1];

   M = new HypreParMatrix(comm,npoints,gnpoints,gndofs,
                          localS.GetI(), localS.GetJ(),localS.GetData(),
                          Mrows,Mcols);
}

void ParContactProblemSingleMesh::SetupTribol()
{
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());

   // plane of bottom block
   std::set<int> mortar_attrs({3});
   // plane of top block
   std::set<int> nonmortar_attrs({4});

   // Initialize Tribol contact library
   tribol::initialize(3, MPI_COMM_WORLD);

   int coupling_scheme_id = 0;
   int mesh1_id = 0;
   int mesh2_id = 1;
   vfes = prob->GetFESpace();
   ParGridFunction * coords = new ParGridFunction(vfes);
   ParMesh * pmesh = prob->GetMesh();
   pmesh->SetNodalGridFunction(coords);
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id, mesh1_id, mesh2_id,
      *pmesh, *coords, mortar_attrs, nonmortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_SLIDING,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   // Access Tribol's pressure grid function (on the contact surface)
   auto& pressure = tribol::getMfemPressure(coupling_scheme_id);
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of pressure unknowns: " <<
                pressure.ParFESpace()->GlobalTrueVSize() << std::endl;
   }

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Update contact gaps, forces, and tangent stiffness
   int cycle = 1;   // pseudo cycle
   double t = 1.0;  // pseudo time
   double dt = 1.0; // pseudo dt
   tribol::update(cycle, t, dt);

   // Return contact contribution to the tangent stiffness matrix
   auto A_blk = tribol::getMfemBlockJacobian(coupling_scheme_id);
   
   HypreParMatrix * Mfull = (HypreParMatrix *)(&A_blk->GetBlock(1,0));

   int h = Mfull->Height();
   SparseMatrix merged;
   Mfull->MergeDiagAndOffd(merged);
   Array<int> nonzero_rows;
   for (int i = 0; i<h; i++)
   {
      if (!merged.RowIsEmpty(i))
      {
         nonzero_rows.Append(i);
      }
   }

   int hnew = nonzero_rows.Size();
   SparseMatrix P(hnew,h);

   for (int i = 0; i<hnew; i++)
   {
      int col = nonzero_rows[i];
      P.Set(i,col,1.0);
   }
   P.Finalize();

   SparseMatrix * reduced_merged = Mult(P,merged);

   int rows[2];
   int cols[2];
   cols[0] = Mfull->ColPart()[0];
   cols[1] = Mfull->ColPart()[1];
   int nrows = reduced_merged->Height();

   int row_offset;
   MPI_Scan(&nrows,&row_offset,1,MPI_INT,MPI_SUM,Mfull->GetComm());

   row_offset-=nrows;
   rows[0] = row_offset;
   rows[1] = row_offset+nrows;
   int glob_nrows;
   MPI_Allreduce(&nrows, &glob_nrows,1,MPI_INT,MPI_SUM,Mfull->GetComm());


   int glob_ncols = reduced_merged->Width();
   M = new HypreParMatrix(Mfull->GetComm(), nrows, glob_nrows,
                          glob_ncols, reduced_merged->GetI(), reduced_merged->GetJ(),
                          reduced_merged->GetData(), rows,cols); 

   Vector gap;
   tribol::getMfemGap(coupling_scheme_id, gap);
   auto& P_submesh = *pressure.ParFESpace()->GetProlongationMatrix();
   Vector gap_true;
   gap_true.SetSize(P_submesh.Width());
   P_submesh.MultTranspose(gap,gap_true);

   gapv.SetSize(nrows);
   for (int i = 0; i<nrows; i++)
   {
      gapv[i] = gap_true[nonzero_rows[i]];
   }

   constraints_starts.SetSize(2);
   constraints_starts[0] = M->RowPart()[0];
   constraints_starts[1] = M->RowPart()[1];

   // find elast dofs in contact;
   HypreParMatrix * Jt = (HypreParMatrix *)(&A_blk->GetBlock(0,1));
   int hJt = Jt->Height();
   SparseMatrix mergedJt;
   Jt->MergeDiagAndOffd(mergedJt);

   Array<int> nonzerorows;
   Array<int> zerorows;
   for (int i = 0; i<hJt; i++)
   {
      if (!mergedJt.RowIsEmpty(i))
      {
         nonzerorows.Append(i);
      }
      else
      {
         zerorows.Append(i);
      }
   }

   int hb = nonzerorows.Size();
   SparseMatrix Pbt(hb,K->GetGlobalNumCols());

   for (int i = 0; i<hb; i++)
   {
      int col = nonzerorows[i]+prob->GetFESpace()->GetMyTDofOffset();
      Pbt.Set(i,col,1.0);
   }
   Pbt.Finalize();

   int rows_b[2];
   int cols_b[2];
   int nrows_b = Pbt.Height();

   int row_offset_b;
   MPI_Scan(&nrows_b,&row_offset_b,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   row_offset_b-=nrows_b;
   rows_b[0] = row_offset_b;
   rows_b[1] = row_offset_b+nrows_b;
   cols_b[0] = K->ColPart()[0];
   cols_b[1] = K->ColPart()[1];
   int glob_nrows_b;
   int glob_ncols_b = K->GetGlobalNumCols();
   MPI_Allreduce(&nrows_b, &glob_nrows_b,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   HypreParMatrix * P_bt = new HypreParMatrix(MPI_COMM_WORLD, nrows_b, glob_nrows_b,
                            glob_ncols_b, Pbt.GetI(), Pbt.GetJ(),
                            Pbt.GetData(), rows_b,cols_b); 

   Pb = P_bt->Transpose();
   delete P_bt;                         

   int hi = zerorows.Size();
   SparseMatrix Pit(hi,K->GetGlobalNumCols());

   for (int i = 0; i<hi; i++)
   {
      int col = zerorows[i]+prob->GetFESpace()->GetMyTDofOffset();
      Pit.Set(i,col,1.0);
   }
   Pit.Finalize();

   int rows_i[2];
   int cols_i[2];
   int nrows_i = Pit.Height();

   int row_offset_i;
   MPI_Scan(&nrows_i,&row_offset_i,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   row_offset_i-=nrows_i;
   rows_i[0] = row_offset_i;
   rows_i[1] = row_offset_i+nrows_i;
   cols_i[0] = K->ColPart()[0];
   cols_i[1] = K->ColPart()[1];
   int glob_nrows_i;
   int glob_ncols_i = K->GetGlobalNumCols();
   MPI_Allreduce(&nrows_i, &glob_nrows_i,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   HypreParMatrix * P_it = new HypreParMatrix(MPI_COMM_WORLD, nrows_i, glob_nrows_i,
                            glob_ncols_i, Pit.GetI(), Pit.GetJ(),
                            Pit.GetData(), rows_i,cols_i); 
   
   Pi = P_it->Transpose();
   delete P_it;
   

   

}

double ParContactProblemSingleMesh::E(const Vector & d)
{
   Vector kd(K->Height());
   K->Mult(d,kd);
   return 0.5 * InnerProduct(comm,d, kd) - InnerProduct(comm,d, *B);
}

void ParContactProblemSingleMesh::DdE(const Vector &d, Vector &gradE)
{
   gradE.SetSize(K->Height());
   K->Mult(d, gradE);
   gradE.Add(-1.0, *B); 
}

HypreParMatrix* ParContactProblemSingleMesh::DddE(const Vector &d)
{
   return K; 
}

void ParContactProblemSingleMesh::g(const Vector &d, Vector &gd)
{
   if (recompute)
   {
      if (!enable_tribol)
      {
         ComputeGapFunctionAndDerivatives(d);
      }
      recompute = false;
   }
   gd = GetGapFunction();
}

HypreParMatrix* ParContactProblemSingleMesh::Ddg(const Vector &d)
{
  return GetJacobian();
}

HypreParMatrix* ParContactProblemSingleMesh::lDddg(const Vector &d, const Vector &l)
{
   return nullptr; // for now
}


QPOptParContactProblemSingleMesh::QPOptParContactProblemSingleMesh(ParContactProblemSingleMesh * problem_)
: problem(problem_)
{
   dimU = problem->GetNumDofs();
   dimM = problem->GetNumContraints();
   dimC = problem->GetNumContraints();
   ml.SetSize(dimM); ml = 0.0;
   Vector negone(dimM); negone = -1.0;
   SparseMatrix diag(negone);

   int gsize = problem->GetGlobalNumConstraints();
   int * rows = problem->GetConstraintsStarts().GetData();

   NegId = new HypreParMatrix(problem->GetComm(),gsize, rows,&diag);
   HypreStealOwnership(*NegId, diag);
}

int QPOptParContactProblemSingleMesh::GetDimU() { return dimU; }

int QPOptParContactProblemSingleMesh::GetDimM() { return dimM; }

int QPOptParContactProblemSingleMesh::GetDimC() { return dimC; }

Vector & QPOptParContactProblemSingleMesh::Getml() { return ml; }

HypreParMatrix * QPOptParContactProblemSingleMesh::Duuf(const BlockVector & x)
{
   return problem->DddE(x.GetBlock(0));
}

HypreParMatrix * QPOptParContactProblemSingleMesh::Dumf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblemSingleMesh::Dmuf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblemSingleMesh::Dmmf(const BlockVector & x)
{
   return nullptr;
}

HypreParMatrix * QPOptParContactProblemSingleMesh::Duc(const BlockVector & x)
{
   return problem->Ddg(x.GetBlock(0));
}

HypreParMatrix * QPOptParContactProblemSingleMesh::Dmc(const BlockVector & x)
{
   return NegId;
}

HypreParMatrix * QPOptParContactProblemSingleMesh::lDuuc(const BlockVector & x, const Vector & l)
{
   return nullptr;
}

void QPOptParContactProblemSingleMesh::c(const BlockVector &x, Vector & y)
{
   Vector g0;
   problem->g(x.GetBlock(0),g0); // gap function
   g0.Add(-1.0, x.GetBlock(1));  
   problem->GetJacobian()->Mult(x.GetBlock(0),y);
   y.Add(1.0, g0);
}

double QPOptParContactProblemSingleMesh::CalcObjective(const BlockVector & x)
{
   return problem->E(x.GetBlock(0));
}

void QPOptParContactProblemSingleMesh::CalcObjectiveGrad(const BlockVector & x, BlockVector & y)
{
   problem->DdE(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
}

QPOptParContactProblemSingleMesh::~QPOptParContactProblemSingleMesh()
{
   delete NegId;
}