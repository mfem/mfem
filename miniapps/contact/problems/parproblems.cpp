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

// #ifdef MFEM_USE_TRIBOL



ParContactProblem::ParContactProblem(ParElasticityProblem * prob_, 
                                                         const std::set<int> & mortar_attrs_, 
                                                         const std::set<int> & nonmortar_attrs_ )
: prob(prob_), mortar_attrs(mortar_attrs_), nonmortar_attrs(nonmortar_attrs_)
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
   SetupTribol();
}

void ParContactProblem::SetupTribol()
{
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());

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
   Mfull->EliminateCols(prob->GetEssentialDofs());
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
   Jt->EliminateRows(prob->GetEssentialDofs());

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

void ParContactProblem::g(const Vector &d, Vector &gd)
{
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

// #endif
