#include "parproblems.hpp"



void ParNonlinearElasticityProblem::Init()
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
   b = new ParLinearForm(fes);
   B.SetSize(fes->TrueVSize()); B = 0.0;
   // Elasticity operator
   material_model = new NeoHookeanModel(shear_modulus, bulk_modulus);
   a = new ParNonlinearForm(fes);
   a->AddDomainIntegrator(new HyperelasticNLFIntegrator(material_model));
   a->SetEssentialTrueDofs(ess_tdof_list);
}

void ParNonlinearElasticityProblem::UpdateStep()
{
   //if (formsystem)
   //{
      delete b;
      b = new ParLinearForm(fes);
      delete material_model;
      material_model = new NeoHookeanModel(shear_modulus, bulk_modulus);
      delete a;
      a = new ParNonlinearForm(fes);
      a->AddDomainIntegrator(new HyperelasticNLFIntegrator(material_model));
      a->SetEssentialTrueDofs(ess_tdof_list);
      formsystem = false;
   //}
}


void ParNonlinearElasticityProblem::FormLinearSystem()
{
   if (!formsystem) 
   {
      formsystem = true;
      b->Assemble();
      b->ParallelAssemble(B);
      B.SetSubVector(ess_tdof_list, 0.0);

      a->SetEssentialTrueDofs(ess_tdof_list);
   }
}

void ParNonlinearElasticityProblem::UpdateLinearSystem()
{
   UpdateStep();
   FormLinearSystem();
}


real_t ParNonlinearElasticityProblem::E(const Vector & u) const
{
   real_t energy = 0.0;
   ParGridFunction u_gf(fes);
   Vector temp(u.Size()); temp = 0.0;
   temp.Set(1.0, xframe);
   temp.Add(1.0, u);
   u_gf.SetFromTrueDofs(temp);
   energy += a->GetEnergy(u_gf);
   energy -= InnerProduct(MPI_COMM_WORLD, B, u);
   return energy;
}

void ParNonlinearElasticityProblem::DuE(const Vector & u, Vector & gradE) const
{
   Vector temp(u.Size()); temp = 0.0;
   temp.Set(1.0, xframe);
   temp.Add(1.0, u);
   a->Mult(temp, gradE);
   gradE.Add(-1.0, B);
}

HypreParMatrix * ParNonlinearElasticityProblem::DuuE(const Vector & u)
{
   Vector temp(u.Size()); temp = 0.0;
   temp.Set(1.0, xframe);
   temp.Add(1.0, u);
   return dynamic_cast<HypreParMatrix *>(&a->GetGradient(temp));
}




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
   b = new ParLinearForm(fes);

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
      b->Assemble();
      a->Assemble();
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   }
}

void ParElasticityProblem::UpdateLinearSystem()
{
   UpdateStep();
   FormLinearSystem();
}

// #ifdef MFEM_USE_TRIBOL



ParContactProblem::ParContactProblem(ParElasticityProblem * prob_, 
                                                         const std::set<int> & mortar_attrs_, 
                                                         const std::set<int> & nonmortar_attrs_,
                                                         ParGridFunction * coords_,
                                                         bool doublepass_ )
: prob(prob_), mortar_attrs(mortar_attrs_), 
   nonmortar_attrs(nonmortar_attrs_), coords(coords_),
   doublepass(doublepass_)
{
   ParMesh* pmesh = prob->GetMesh();
   comm = pmesh->GetComm();
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &numprocs);
 
   dim = pmesh->Dimension();
   prob->FormLinearSystem();
   K = new HypreParMatrix(prob->GetOperator());
   B = new Vector(prob->GetRHS());
   if (doublepass)
   {
      SetupTribolDoublePass();
   }
   else
   {
      SetupTribol();
   }
}


ParContactProblem::ParContactProblem(ParNonlinearElasticityProblem * nlprob_,
	                             //ParElasticityProblem * prob_,	
                                                         const std::set<int> & mortar_attrs_, 
                                                         const std::set<int> & nonmortar_attrs_,
                                                         ParGridFunction * coords_,
                                                         bool doublepass_ )
: nlprob(nlprob_), //prob(prob_), 
   mortar_attrs(mortar_attrs_), 
   nonmortar_attrs(nonmortar_attrs_), coords(coords_),
   doublepass(doublepass_), nonlinearelasticity(true)
{
   ParMesh* pmesh = nlprob->GetMesh();
   comm = pmesh->GetComm();
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &numprocs);
   
    
   dim = pmesh->Dimension();
   //prob->FormLinearSystem();
   //K = new HypreParMatrix(prob->GetOperator());
   if (doublepass)
   {
      SetupTribolDoublePass();
   }
   else
   {
      SetupTribol();
   }

   nonlinearelasticity = true;
}





void ParContactProblem::SetupTribol()
{
   axom::slic::SimpleLogger logger;
   axom::slic::setIsRoot(mfem::Mpi::Root());

   // Initialize Tribol contact library
   tribol::initialize(3, MPI_COMM_WORLD);

   tribol::parameters_t::getInstance().gap_separation_ratio = 8;

   int coupling_scheme_id = 0;
   int mesh1_id = 0;
   int mesh2_id = 1;
   
   ParMesh * pmesh;
   if (!nonlinearelasticity)
   {
      vfes = prob->GetFESpace();
      pmesh = prob->GetMesh();
   }
   else
   {
      vfes = nlprob->GetFESpace();
      pmesh = nlprob->GetMesh();
   }
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
   int vsize = pressure.ParFESpace()->GlobalTrueVSize();
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of pressure unknowns: " <<
                vsize << std::endl;
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
   HypreParMatrix * Me;
   if (!nonlinearelasticity)
   {
      Mfull->EliminateCols(prob->GetEssentialDofs());
   }
   else
   {
      Mfull->EliminateCols(nlprob->GetEssentialDofs());
   }
   delete Me;
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
   delete reduced_merged;                          

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
}

void ParContactProblem::SetupTribolDoublePass()
{
   axom::slic::SimpleLogger logger1;
   axom::slic::setIsRoot(mfem::Mpi::Root());

   // Initialize Tribol contact library
   tribol::initialize(3, MPI_COMM_WORLD);

   int coupling_scheme_id1 = 0;
   int mesh1_id1 = 0;
   int mesh2_id1 = 1;
   if (!nonlinearelasticity)
   {
      vfes = prob->GetFESpace();
   }
   else
   {
      vfes = nlprob->GetFESpace();
   }
   ParGridFunction * coords1 = new ParGridFunction(vfes);
   ParMesh * pmesh1;
   if (!nonlinearelasticity)
   {
      pmesh1 = prob->GetMesh();
   }
   else
   {
      pmesh1 = nlprob->GetMesh();
   }
   pmesh1->SetNodalGridFunction(coords1);
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id1, mesh1_id1, mesh2_id1,
      *pmesh1, *coords1, mortar_attrs, nonmortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_SLIDING,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   // Access Tribol's pressure grid function (on the contact surface)
   auto& pressure1 = tribol::getMfemPressure(coupling_scheme_id1);
   int vsize1 = pressure1.ParFESpace()->GlobalTrueVSize();
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of pressure unknowns: " <<
                vsize1 << std::endl;
   }

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id1,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Update contact gaps, forces, and tangent stiffness
   int cycle1 = 1;   // pseudo cycle
   double t1 = 1.0;  // pseudo time
   double dt1 = 1.0; // pseudo dt
   tribol::update(cycle1, t1, dt1);

   // Return contact contribution to the tangent stiffness matrix
   auto A_blk1 = tribol::getMfemBlockJacobian(coupling_scheme_id1);
   
   HypreParMatrix * Mfull1 = (HypreParMatrix *)(&A_blk1->GetBlock(1,0));
   if (!nonlinearelasticity)
   {
      Mfull1->EliminateCols(prob->GetEssentialDofs());
   }
   else
   {
      Mfull1->EliminateCols(nlprob->GetEssentialDofs());
   }
   int h1 = Mfull1->Height();
   SparseMatrix merged1;
   Mfull1->MergeDiagAndOffd(merged1);
   Array<int> nonzero_rows1;
   for (int i = 0; i<h1; i++)
   {
      if (!merged1.RowIsEmpty(i))
      {
         nonzero_rows1.Append(i);
      }
   }

   int hnew1 = nonzero_rows1.Size();
   SparseMatrix P1(hnew1,h1);

   for (int i = 0; i<hnew1; i++)
   {
      int col = nonzero_rows1[i];
      P1.Set(i,col,1.0);
   }
   P1.Finalize();

   SparseMatrix * reduced_merged1 = Mult(P1,merged1);

   int rows1[2];
   int cols1[2];
   cols1[0] = Mfull1->ColPart()[0];
   cols1[1] = Mfull1->ColPart()[1];
   int nrows1 = reduced_merged1->Height();

   int row_offset1;
   MPI_Scan(&nrows1,&row_offset1,1,MPI_INT,MPI_SUM,Mfull1->GetComm());

   row_offset1-=nrows1;
   rows1[0] = row_offset1;
   rows1[1] = row_offset1+nrows1;
   int glob_nrows1;
   MPI_Allreduce(&nrows1, &glob_nrows1,1,MPI_INT,MPI_SUM,Mfull1->GetComm());


   int glob_ncols1 = reduced_merged1->Width();
   HypreParMatrix * M1 = new HypreParMatrix(Mfull1->GetComm(), nrows1, glob_nrows1,
                          glob_ncols1, reduced_merged1->GetI(), reduced_merged1->GetJ(),
                          reduced_merged1->GetData(), rows1,cols1); 

   Vector gap1;
   tribol::getMfemGap(coupling_scheme_id1, gap1);
   auto& P_submesh1 = *pressure1.ParFESpace()->GetProlongationMatrix();
   Vector gap_true1;
   gap_true1.SetSize(P_submesh1.Width());
   P_submesh1.MultTranspose(gap1,gap_true1);

   tribol::finalize();

   // ------------------------------ 
   // second pass
   // ------------------------------ 
   // Initialize Tribol contact library
   tribol::initialize(3, MPI_COMM_WORLD);

   int coupling_scheme_id2 = 0;
   int mesh1_id2 = 0;
   int mesh2_id2 = 1;
   ParGridFunction * coords2 = new ParGridFunction(vfes);
   ParMesh * pmesh2;
   if (!nonlinearelasticity)
   {
      pmesh2 = prob->GetMesh();
   }
   else
   {
      pmesh2 = nlprob->GetMesh();
   }
   pmesh2->SetNodalGridFunction(coords2);
   tribol::registerMfemCouplingScheme(
      coupling_scheme_id2, mesh1_id2, mesh2_id2,
      *pmesh2, *coords2, nonmortar_attrs, mortar_attrs,
      tribol::SURFACE_TO_SURFACE,
      tribol::NO_SLIDING,
      tribol::SINGLE_MORTAR,
      tribol::FRICTIONLESS,
      tribol::LAGRANGE_MULTIPLIER,
      tribol::BINNING_GRID
   );

   // Access Tribol's pressure grid function (on the contact surface)
   auto& pressure2 = tribol::getMfemPressure(coupling_scheme_id2);
   int vsize2 = pressure2.ParFESpace()->GlobalTrueVSize();
   if (mfem::Mpi::Root())
   {
      std::cout << "Number of pressure unknowns: " <<
                vsize2 << std::endl;
   }

   // Set Tribol options for Lagrange multiplier enforcement
   tribol::setLagrangeMultiplierOptions(
      coupling_scheme_id2,
      tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN
   );

   // Update contact mesh decomposition
   tribol::updateMfemParallelDecomposition();

   // Update contact gaps, forces, and tangent stiffness
   int cycle2 = 1;   // pseudo cycle
   double t2 = 1.0;  // pseudo time
   double dt2 = 1.0; // pseudo dt
   tribol::update(cycle2, t2, dt2);

   // Return contact contribution to the tangent stiffness matrix
   auto A_blk2 = tribol::getMfemBlockJacobian(coupling_scheme_id2);
   
   HypreParMatrix * Mfull2 = (HypreParMatrix *)(&A_blk2->GetBlock(1,0));
   if (!nonlinearelasticity)
   {
      Mfull2->EliminateCols(prob->GetEssentialDofs());
   }
   else
   {
      Mfull2->EliminateCols(nlprob->GetEssentialDofs());
   }
   int h2 = Mfull2->Height();
   SparseMatrix merged2;
   Mfull2->MergeDiagAndOffd(merged2);
   Array<int> nonzero_rows2;
   for (int i = 0; i<h2; i++)
   {
      if (!merged2.RowIsEmpty(i))
      {
         nonzero_rows2.Append(i);
      }
   }

   int hnew2 = nonzero_rows2.Size();
   SparseMatrix P2(hnew2,h2);

   for (int i = 0; i<hnew2; i++)
   {
      int col = nonzero_rows2[i];
      P2.Set(i,col,1.0);
   }
   P2.Finalize();

   SparseMatrix * reduced_merged2 = Mult(P2,merged2);

   int rows2[2];
   int cols2[2];
   cols2[0] = Mfull2->ColPart()[0];
   cols2[1] = Mfull2->ColPart()[1];
   int nrows2 = reduced_merged2->Height();

   int row_offset2;
   MPI_Scan(&nrows2,&row_offset2,1,MPI_INT,MPI_SUM,Mfull2->GetComm());

   row_offset2-=nrows2;
   rows2[0] = row_offset2;
   rows2[1] = row_offset2+nrows2;
   int glob_nrows2;
   MPI_Allreduce(&nrows2, &glob_nrows2,1,MPI_INT,MPI_SUM,Mfull2->GetComm());


   int glob_ncols2 = reduced_merged2->Width();
   HypreParMatrix * M2 = new HypreParMatrix(Mfull2->GetComm(), nrows2, glob_nrows2,
                          glob_ncols2, reduced_merged2->GetI(), reduced_merged2->GetJ(),
                          reduced_merged2->GetData(), rows2,cols2); 

   Vector gap2;
   tribol::getMfemGap(coupling_scheme_id2, gap2);
   auto& P_submesh2 = *pressure2.ParFESpace()->GetProlongationMatrix();
   Vector gap_true2;
   gap_true2.SetSize(P_submesh2.Width());
   P_submesh2.MultTranspose(gap2,gap_true2);

   tribol::finalize();


   gapv.SetSize(nrows1+nrows2);
   for (int i = 0; i<nrows1; i++)
   {
      gapv[i] = gap_true1[nonzero_rows1[i]];
   }
   for (int i = 0; i<nrows2; i++)
   {
      gapv[nrows1+i] = gap_true2[nonzero_rows2[i]];
   }

   Array2D<HypreParMatrix *> A_array(2,1);
   A_array(0,0) = M1;
   A_array(1,0) = M2;

   M = HypreParMatrixFromBlocks(A_array);

   delete M1;
   delete M2;

   constraints_starts.SetSize(2);
   constraints_starts[0] = M->RowPart()[0];
   constraints_starts[1] = M->RowPart()[1];

}


void ParContactProblem::ComputeRestrictionToContactDofs()
{
   // find elast dofs in contact;
   if (!Mt) 
   {
      Mt = M->Transpose();
      if (!nonlinearelasticity)
      {
         Mt->EliminateRows(prob->GetEssentialDofs());
      }
      else
      {
         Mt->EliminateRows(nlprob->GetEssentialDofs());
      }
   }

   int hJt = Mt->Height();
   SparseMatrix mergedJt;
   Mt->MergeDiagAndOffd(mergedJt);

   Array<int> nonzerorows;
   for (int i = 0; i<hJt; i++)
   {
      if (!mergedJt.RowIsEmpty(i))
      {
         nonzerorows.Append(i);
      }
   }

   int hb = nonzerorows.Size();
   SparseMatrix Pbt(hb,vfes->GlobalTrueVSize());//K->GetGlobalNumCols());

   for (int i = 0; i<hb; i++)
   {
      int col = nonzerorows[i]+vfes->GetMyTDofOffset();//prob->GetFESpace()->GetMyTDofOffset();
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
   for (int i = 0; i < 2; i++)
   {
      cols_b[i] = vfes->GetTrueDofOffsets()[i];
   }
   //cols_b[0] = K->ColPart()[0];
   //cols_b[1] = K->ColPart()[1];
   int glob_nrows_b;
   int glob_ncols_b = vfes->GlobalTrueVSize();//K->GetGlobalNumCols();
   MPI_Allreduce(&nrows_b, &glob_nrows_b,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   HypreParMatrix * P_bt = new HypreParMatrix(MPI_COMM_WORLD, nrows_b, glob_nrows_b,
                           glob_ncols_b, Pbt.GetI(), Pbt.GetJ(),
                           Pbt.GetData(), rows_b,cols_b); 

   Pb = P_bt->Transpose();
   delete P_bt;                         
}

void ParContactProblem::ComputeRestrictionToNonContactDofs()
{
   if (!Mt) 
   {
      Mt = M->Transpose();
      if (!nonlinearelasticity)
      {
         Mt->EliminateRows(prob->GetEssentialDofs());
      }
      else
      {
         Mt->EliminateRows(nlprob->GetEssentialDofs());
      }
   }

   int hJt = Mt->Height();
   SparseMatrix mergedJt;
   Mt->MergeDiagAndOffd(mergedJt);

   Array<int> zerorows;
   for (int i = 0; i<hJt; i++)
   {
      if (mergedJt.RowIsEmpty(i))
      {
         zerorows.Append(i);
      }
   }

   int hi = zerorows.Size();
   SparseMatrix Pit(hi,vfes->GlobalTrueVSize());//K->GetGlobalNumCols());

   for (int i = 0; i<hi; i++)
   {
      int col = zerorows[i]+vfes->GetMyTDofOffset();//prob->GetFESpace()->GetMyTDofOffset();
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
   for (int i = 0; i < 2; i++)
   {
      cols_i[i] = vfes->GetTrueDofOffsets()[i];
   }
   int glob_nrows_i;
   int glob_ncols_i = vfes->GlobalTrueVSize();//K->GetGlobalNumCols();
   MPI_Allreduce(&nrows_i, &glob_nrows_i,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   HypreParMatrix * P_it = new HypreParMatrix(MPI_COMM_WORLD, nrows_i, glob_nrows_i,
                           glob_ncols_i, Pit.GetI(), Pit.GetJ(),
                           Pit.GetData(), rows_i,cols_i); 
      
   Pi = P_it->Transpose();
   delete P_it;
}


double ParContactProblem::E(const Vector & d)
{
   if (nonlinearelasticity)
   {
      return nlprob->E(d);
   }
   else
   {
      Vector kd(K->Height());
      K->Mult(d,kd);
      return 0.5 * InnerProduct(comm,d, kd) - InnerProduct(comm,d, *B);
   }
}


void ParContactProblem::DdE(const Vector &d, Vector &gradE)
{
   if (nonlinearelasticity)
   {
      nlprob->DuE(d, gradE);
   }
   else
   {
      gradE.SetSize(K->Height());
      K->Mult(d, gradE);
      gradE.Add(-1.0, *B); 
   }
}

HypreParMatrix* ParContactProblem::DddE(const Vector &d)
{
   if (nonlinearelasticity)
   {
      return nlprob->DuuE(d);
   }
   else
   {
      return K; 
   }
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


QPOptParContactProblem::QPOptParContactProblem(ParContactProblem * problem_, const Vector & xref_)
: problem(problem_), xref(xref_)
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

   label = -1;

   nonlinearelasticity = problem->IsElasticityModelNonlinear();
   if (nonlinearelasticity)
   {
      KQP = new HypreParMatrix(*problem->DddE(xref));
      gradEQP.SetSize(dimU); gradEQP = 0.0;
      problem->DdE(xref, gradEQP);
      EQP = problem->E(xref);
   }
}


QPOptParContactProblem::QPOptParContactProblem(ParContactProblem * problem_, const Vector & xref_, const Vector & xframe_)
: problem(problem_), xref(xref_), xframe(xframe_)
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

   label = -1;

   nonlinearelasticity = problem->IsElasticityModelNonlinear();
   if (nonlinearelasticity)
   {
      Vector temp(dimU); temp = 0.0;
      temp.Set(1.0, xframe);
      temp.Add(1.0, xref);
      KQP = problem->DddE(temp);
      gradEQP.SetSize(dimU); gradEQP = 0.0;
      problem->DdE(temp, gradEQP);
      EQP = problem->E(temp);
   }
}




int QPOptParContactProblem::GetDimU() { return dimU; }

int QPOptParContactProblem::GetDimM() { return dimM; }

int QPOptParContactProblem::GetDimC() { return dimC; }

Vector & QPOptParContactProblem::Getml() { return ml; }

HypreParMatrix * QPOptParContactProblem::Duuf(const BlockVector & x)
{
   return DddE(x.GetBlock(0));
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

// J(dref) * (d - dref) + g(dref) - slack
void QPOptParContactProblem::c(const BlockVector &x, Vector & y)
{
   Vector g0; // g(dref) 
   problem->g(x.GetBlock(0), g0); // gap function
   Vector temp(x.GetBlock(0).Size()); temp = 0.0;
   temp.Set(1.0, x.GetBlock(0));  
   temp.Add(-1.0, xref); // displacement at previous time step  
   problem->GetJacobian()->Mult(temp, y); // J * (d - xref)
   y.Add(1.0, g0); // J * (d - xref) + g0 
   y.Add(-1.0, x.GetBlock(1)); // J * (d - xref) + g0 - s
}

double QPOptParContactProblem::CalcObjective(const BlockVector & x)
{
    return E(x.GetBlock(0));
}

// QPOptParContactProblem will take in a relative displacement
double QPOptParContactProblem::E(const Vector & d)
{
   if (nonlinearelasticity)
   {
      // TO DO: compute QP approximation of E
      // (d - xref)^T [ 1/2 K * (d - xref) + gradEQP] + EQP
      double energy = 0.0;
      Vector dx(dimU); dx = 0.0;
      Vector temp(dimU); temp = 0.0;
      dx.Set(1.0, d);
      dx.Add(-1.0, xref);
      KQP->Mult(dx, temp);
      temp *= 0.5;
      temp.Add(1.0, gradEQP);
      energy = InnerProduct(MPI_COMM_WORLD, dx, temp);
      energy += EQP;
      return energy;
   }
   else
   {
      return problem->E(d);
   }
}

void QPOptParContactProblem::DdE(const Vector & d, Vector & gradE)
{
   if (nonlinearelasticity)
   {
      // KQP * (d - xref) + gradEQP
      Vector dx(dimU); dx = 0.0;
      dx.Set(1.0, d);
      dx.Add(-1.0, xref);
      KQP->Mult(dx, gradE);
      gradE.Add(1.0, gradEQP);
   }
   else
   {
      problem->DdE(d, gradE);
   }
}

HypreParMatrix * QPOptParContactProblem::DddE(const Vector & d)
{
   if (nonlinearelasticity)
   {
      return KQP;
   }
   else
   {
      return problem->DddE(d);
   }
}




void QPOptParContactProblem::CalcObjectiveGrad(const BlockVector & x, BlockVector & y)
{
   DdE(x.GetBlock(0), y.GetBlock(0));
   y.GetBlock(1) = 0.0;
}

QPOptParContactProblem::~QPOptParContactProblem()
{
   delete NegId;
}

// #endif
