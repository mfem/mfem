#include "snavier_dg.hpp"

namespace mfem{

/// Constructor
SNavierPicardDGSolver::SNavierPicardDGSolver(ParMesh* mesh_,
											 int sigorder_,
                                             int uorder_,
                                             int porder_,
                                             double Re_,
											 double kappa_0_,
                                             bool verbose_):
pmesh(mesh_), sigorder(sigorder_), uorder(uorder_), porder(porder_), verbose(verbose_)
{
   // mesh
   dim  = pmesh->Dimension();

   // dimension of spaces
   vdim = dim;
   tdim = dim*dim;

   // FE collection and spaces for velocity and pressure
   sigfec = new DG_FECollection(sigorder, dim, BasisType::GaussLobatto);
   ufec   = new DG_FECollection(uorder,   dim, BasisType::GaussLobatto);
   pfec   = new DG_FECollection(porder,   dim, BasisType::GaussLobatto);

   sigfes = new ParFiniteElementSpace(pmesh, sigfec, tdim, Ordering::byNODES);
   ufes   = new ParFiniteElementSpace(pmesh, ufec,   vdim, Ordering::byNODES);
   pfes   = new ParFiniteElementSpace(pmesh, pfec);

   // initialize vectors of boundaies' attributes
   nbc_bdr.SetSize(pmesh->bdr_attributes.Max());      nbc_bdr=0;
   dbc_bdr.SetSize(pmesh->bdr_attributes.Max());      dbc_bdr=0;

   // initialize GridFunctions
   u_gf.SetSpace(ufes);  u_gf=0.0;
   uk_gf.SetSpace(ufes); uk_gf=0.0;
   p_gf.SetSpace(pfes);  p_gf=0.0;

   // setup GridFunctionCoefficients
   uk_coeff = new VectorGridFunctionCoefficient(&uk_gf);

   // initialize vectors & arrays
   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = ufes->GetVSize();
   block_offsets[2] = pfes->GetVSize();
   block_offsets.PartialSum();
   trueblock_offsets.SetSize(3);
   trueblock_offsets[0] = 0;
   trueblock_offsets[1] = ufes->GetTrueVSize();
   trueblock_offsets[2] = pfes->GetTrueVSize();
   trueblock_offsets.PartialSum();

   x_bvec = new BlockVector(block_offsets);
   rhs_bvec = new BlockVector(block_offsets);
   truex_bvec = new BlockVector(trueblock_offsets);
   truerhs_bvec = new BlockVector(trueblock_offsets);
   x_bvec->operator=(0.0);
   rhs_bvec->operator=(0.0);
   truex_bvec->operator=(0.0);
   truerhs_bvec->operator=(0.0);

   // set Reynold's number
   Re = Re_;

   // set stabilization parameter
   kappa_0 = kappa_0_;

   // Error computation setup
   err_u = err_p = 0;
   norm_u = norm_p = 0;
   int order_quad = std::max(2, 2*uorder+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
}

/// Public Interface
//TODO: copied from snavier_cg.cpp. Need to be checked later
void SNavierPicardDGSolver::AddVelDirichletBC(VectorFunctionCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( dbc_bdr[i] && attr[i]) == 0,
                  "Duplicate boundary definition detected.");
      if (attr[i] == 1)
    	  dbc_bdr[i] = 1;
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC (full) to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
         if (attr[i] == 1)
            mfem::out << i << " ";

      mfem::out << std::endl;
   }
}

//TODO: copied from snavier_cg.cpp. Need to be checked later
void SNavierPicardDGSolver::AddVelDirichletBC(VectorFunctionCoefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   tmp_bdr = 0;
   tmp_bdr[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, tmp_bdr);
}

//TODO: copied from snavier_cg.cpp. Need to be checked later
void SNavierPicardDGSolver::AddTractionBC(VectorFunctionCoefficient *coeff, Array<int> &attr)
{
   traction_bcs.emplace_back(attr, coeff);

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( dbc_bdr[i] && attr[i]) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
      if (attr[i] == 1)
    	  nbc_bdr[i] = 1;
   }

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Traction (Neumann) BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

//TODO: copied from snavier_cg.cpp. Need to be checked later
void SNavierPicardDGSolver::AddAccelTerm(VectorFunctionCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Acceleration term to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void SNavierPicardDGSolver::SetFixedPointSolver(SolverParams params)
{
   sParams_Picard = params;
}

void SNavierPicardDGSolver::SetLinearSolvers( SolverParams params)
{
   sParams_Lin = params;
}

void SNavierPicardDGSolver::SetInitialConditionVel(VectorCoefficient &u_in, VectorCoefficient &w_in)
{
	// Project coefficient onto velocity ParGridFunction
	u_gf.ProjectCoefficient(u_in);

	// Project coefficient onto (background) velocity ParGridFunction
	// Note: For regular Picard solve, u_in is equal to w_in. For solving the
	// specific Oseen equations (i.e. Picard with maxIter=1), w_in can be set to be
	// different from u_in.
	uk_gf.ProjectCoefficient(w_in);
}

void SNavierPicardDGSolver::Setup()
{
	/// 1. Setup and assemble bilinear forms
	ainv_form = new ParBilinearForm(sigfes);
	b_form    = new ParMixedBilinearForm(ufes, sigfes);
	d_form    = new ParMixedBilinearForm(pfes, ufes);

	ainv_form->AddDomainIntegrator(new InverseIntegrator(new TensorMassIntegrator(Re)));

	b_form->AddDomainIntegrator(new MixedVectorDivTensorIntegrator);
	b_form->AddInteriorFaceIntegrator( new DGVectorAvgNormalJumpIntegration(-1.0) );
	b_form->AddBdrFaceIntegrator( new DGVectorAvgNormalJumpIntegration(-1.0), nbc_bdr );

	ConstantCoefficient minusOne(-1.0);
	d_form->AddDomainIntegrator(new TransposeIntegrator(new VectorDivergenceIntegrator(minusOne)));
	d_form->AddInteriorFaceIntegrator( new DGAvgNormalJumpIntegrator );
	d_form->AddBdrFaceIntegrator( new DGAvgNormalJumpIntegrator, dbc_bdr);

	ainv_form->Assemble();  ainv_form->Finalize();
	b_form->Assemble();     b_form->Finalize();
	d_form->Assemble();     d_form->Finalize();

	Ainv_mat = ainv_form->ParallelAssemble();
	B_mat 	 = b_form->ParallelAssemble();
	Bt_mat	 = B_mat->Transpose();
	D_mat 	 = d_form->ParallelAssemble();
	nD_mat 	 = new HypreParMatrix(*D_mat);
	*nD_mat *= -1.0;
	zero_mat = NULL;

	BtAinv_mat    = ParMult(Bt_mat,Ainv_mat);
	BtAinvB_mat   = ParMult(BtAinv_mat,B_mat);

	/// 2. Setup and assemble linear form for rhs
	f_form    = new ParLinearForm(sigfes);
	h_form    = new ParLinearForm;

	f_form->AddBdrFaceIntegrator(new TensorDGDirichletLFIntegrator(*(vel_dbcs[0].coeff),1.0), dbc_bdr);
	h_form->AddBdrFaceIntegrator(new BoundaryNormalLFIntegrator_mod(*(vel_dbcs[0].coeff),-1.0), dbc_bdr);

	f_form->Update();
	f_form->Assemble();
	F_vec = new HypreParVector(sigfes);
	F_vec = f_form->ParallelAssemble();

	h_form->Update(pfes, rhs_bvec->GetBlock(1), 0);
	h_form->Assemble();
	h_form->ParallelAssemble(truerhs_bvec->GetBlock(1));

	BtAinvF_vec = new HypreParVector(ufes);
	BtAinv_mat->Mult(*F_vec,*BtAinvF_vec);
}

void SNavierPicardDGSolver::FSolve()
{
#ifdef MFEM_DEBUG
   PrintMatricesVectors( "setup", 0); // Export matrices/vectors before step
#endif

   PrintInfo();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "=========================================================="<< std::endl;
      mfem::out << "======      Picard Steady Navier-Stokes Solver      ======"<< std::endl;
      mfem::out << "=========================================================="<< std::endl;
   }

   timer.Clear();
   timer.Start();

   // Print header
//   mfem::out << std::endl;
//   mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8)
//             << "Res" << std::setw(12) << "AbsTol" << "\n";


   // Export initial solution
   //SaveResults( 0 );

   for (iter = 0; iter < sParams_Picard.maxIter; iter++)
   {
		// Assemble matrices
		//TODO

		// Evaluate the residual
		//TODO

		// Solve current iteration.
		Step();

		// Output results
		//SaveResults( iter+1 );

		// Compute errors.
		//ComputeError();

		// Update solution at previous iterate and gridfunction coefficients.
		//UpdateSolution();

		// Print results
		mfem::out << iter << "   " << std::setw(3)
				<< std::setprecision(2) << std::scientific << err_u
				<< "   " << sParams_Picard.atol << "\n";

		// Check Picard convergence.
		// TODO: criteria for convergence of Picard iteration
		if (err_u < sParams_Picard.atol)
		{
		 out << "Solver converged to steady state solution \n";
		 flag = 1;
		 break;
		}

   }

   timer.Stop();
}

void SNavierPicardDGSolver::SetupOutput( const char* folderPath, bool visit_, bool paraview_,
                                         DataCollection::Format par_format )
{
   visit    = visit_;
   paraview = paraview_;
   outfolder = folderPath;

   // Creating output directory if not existent
   if (mkdir(folderPath, 0777) == -1) {std::cerr << "Error :  " << strerror(errno) << std::endl;}

   // Create output data collections
   if( visit )
   {
      visit_dc = new VisItDataCollection("Results-VISit", pmesh);
      visit_dc->SetPrefixPath(folderPath);
      visit_dc->RegisterField("velocity", &u_gf);
      visit_dc->RegisterField("pressure", &p_gf);
      visit_dc->SetFormat(par_format);
   }

   if( paraview )
   {
      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh);
      paraview_dc->SetPrefixPath(folderPath);
      paraview_dc->SetLevelsOfDetail(uorder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->RegisterField("velocity",&u_gf);
      paraview_dc->RegisterField("pressure",&p_gf);
   }

}

/// Private Interface

void SNavierPicardDGSolver::Step()
{
	/// Assemble terms associated with new velocity uk and modify rhs for essential bcs.
	delete c_form; c_form = nullptr;
	c_form    = new ParBilinearForm(ufes);
	c_form->AddInteriorFaceIntegrator(new DGVectorNormalJumpIntegrator(1.0,kappa_0/Re));
	c_form->AddBdrFaceIntegrator(new DGVectorNormalJumpIntegrator(1.0,kappa_0/Re),dbc_bdr);
	c_form->AddDomainIntegrator(new VectorGradVectorIntegrator(*uk_coeff, -1.0));
	c_form->AddInteriorFaceIntegrator(new DGVectorUpwindJumpIntegrator(*uk_coeff, 1.0));
	c_form->AddBdrFaceIntegrator(new DGVectorUpwindJumpIntegrator(*uk_coeff, 1.0),nbc_bdr);
	c_form->Assemble();     c_form->Finalize();
	C_mat 	 = c_form->ParallelAssemble();
	BtAinvB_C_mat = ParAdd(BtAinvB_mat,C_mat);

	delete g_form; g_form = nullptr;
	g_form    = new ParLinearForm(ufes);

	g_form->AddBdrFaceIntegrator(new VectorDGDirichletLFIntegrator(*(vel_dbcs[0].coeff),1.0,kappa_0/Re), dbc_bdr);
	g_form->AddBdrFaceIntegrator(new VectorDGNeumannLFIntegrator(*(traction_bcs[0].coeff), 1.0), nbc_bdr);
	g_form->AddBdrFaceIntegrator(new VectorDGDirichletLFIntegrator(*uk_coeff, *(vel_dbcs[0].coeff), -1.0), dbc_bdr);
	g_form->AddDomainIntegrator(new VectorDomainLFIntegrator(*(accel_terms[0].coeff)));

	g_form->Update();
	g_form->Assemble();
	G_vec = new HypreParVector(ufes);
	G_vec = g_form->ParallelAssemble();

	G_vec->Add(1.0,*BtAinvF_vec);
	truerhs_bvec->GetBlock(0).Set(1.0, *G_vec); // the method is discussed in issue #103

	/// Form the global matrix from block matrices
	OseenOp.DeleteAll(); OseenOp.SetSize(2, 2);
	OseenOp(0,0) = BtAinvB_C_mat;       OseenOp(0,1) = D_mat;
	OseenOp(1,0) = nD_mat->Transpose(); OseenOp(1,1) = zero_mat;

	OseenMono = HypreParMatrixFromBlocks(OseenOp);

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "step", iter);  // Export matrices/vectors after assembly
#endif

	/// Solve
	// TODO: Currently, we call MUMPS from PETSc. May need to change to iterative solver later
#ifndef MFEM_USE_PETSC
	if (sParams_Lin.petsc && pmesh->GetMyRank() == 0)
		mfem_error("MFEM does not use PETSc. Need other parallel solver! (WIP)");
#endif

#ifdef MFEM_USE_PETSC
	MFEMInitializePetsc(NULL,NULL,sParams_Lin.petscrc_file,NULL);
	PetscLinearSolver *petsc_solver;
	PetscPreconditioner *petsc_precon= NULL;
	PetscParMatrix *M_pmat;
#endif

	if(sParams_Lin.petsc)
	{
#ifdef MFEM_USE_PETSC
		M_pmat = new PetscParMatrix(OseenMono);
		petsc_solver = new PetscLinearSolver(MPI_COMM_WORLD, "solver_");
		petsc_solver->SetOperator(*M_pmat);


		//petsc_solver->SetPreconditioner(*petsc_precon);
		petsc_solver->SetTol(sParams_Lin.atol);
		petsc_solver->SetAbsTol(0.0);
		petsc_solver->SetMaxIter(sParams_Lin.maxIter);
		petsc_solver->SetPrintLevel(sParams_Lin.pl);
		if (verbose && pmesh->GetMyRank() == 0){
			std::cout << std::endl;
			std::cout << "=========================================================="<< std::endl;
			std::cout << " Solve the linear system by PETSc" << std::endl;
			std::cout << "=========================================================="<< std::endl;
		}
		petsc_solver->Mult(*truerhs_bvec, *truex_bvec);
		if (verbose && pmesh->GetMyRank() == 0)
		{
		  if (petsc_solver->GetConverged())
			  std::cout << "Solver converged in " << petsc_solver->GetNumIterations()
			  << " iterations with a residual norm of " << petsc_solver->GetFinalNorm() << ".\n";
		  else
			  std::cout << "Solver did not converge in " << petsc_solver->GetNumIterations()
			  << " iterations. Residual norm is " << petsc_solver->GetFinalNorm() << ".\n";
		  std::cout << "=========================================================="<< std::endl;
		  std::cout << std::endl;
		}

		delete petsc_solver;
		delete petsc_precon;
		delete M_pmat;
#endif
	  }else{
		  if (verbose && pmesh->GetMyRank() == 0)
			  mfem_error("HYPRE solver are not available");
	  }

#ifdef MFEM_USE_PETSC
    MFEMFinalizePetsc();
#endif

	/// Update GridFunctions for solution.
	u_gf.MakeRef(ufes, x_bvec->GetBlock(0), 0);
	p_gf.MakeRef(pfes, x_bvec->GetBlock(1), 0);

	u_gf.Distribute(&(truex_bvec->GetBlock(0)));
	p_gf.Distribute(&(truex_bvec->GetBlock(1)));

	delete C_mat; C_mat=nullptr;
	delete BtAinvB_C_mat; BtAinvB_C_mat=nullptr;
	delete G_vec; G_vec=nullptr;
	delete OseenMono; OseenMono=nullptr;
}

void SNavierPicardDGSolver::ComputeError()
{
	//TODO: modify the function such that I can compute error
	// for Picard & single Oseen solve
//   err_u  = u_gf.ComputeL2Error(*uk_coeff);
//   norm_u = ComputeGlobalLpNorm(2., *uk_coeff, *pmesh, irs);
//   err_p  = p_gf.ComputeL2Error(*pk_c);
//   norm_p = ComputeGlobalLpNorm(2., *pk_c, *pmesh, irs);
//
//   if (verbose)
//   {
//      out << "|| v - v_k || / || v_k || = " << err_u / norm_u << "\n";
//      out << "|| p - p_k || / || p_k || = " << err_p / norm_p << "\n";
//   }
}

void SNavierPicardDGSolver::ComputeRes()
{
	/// TODO: we may want to use the residual of the momentum equation as
	//  the stopping criteria of ther Picard iteration.
}

void SNavierPicardDGSolver::UpdateSolution()
{
	*uk_gf = *u_gf;
}

void SNavierPicardDGSolver::SaveResults( int iter )
{

   if ( visit ) // Save to GLVis format if visit enabled
   {
      visit_dc->SetCycle(iter);
      visit_dc->SetTime(iter);
      visit_dc->Save();
   }
   if ( paraview ) // Save to Paraview format if visit enabled
   {
      paraview_dc->SetCycle(iter);
      paraview_dc->SetTime(iter);
      paraview_dc->Save();
   }
}

void SNavierPicardDGSolver::PrintInfo()
{
	HYPRE_BigInt fes_sizeVel = ufes->GlobalTrueVSize();
	HYPRE_BigInt fes_sizePres = pfes->GlobalTrueVSize();

	if (pmesh->GetMyRank() == 0)
	{
		mfem::out << std::endl;
		mfem::out << "NAVIER version: " << SNAVIER_DG_VERSION << std::endl
				<< "MFEM version: " << MFEM_VERSION << std::endl
				<< "MFEM GIT: " << MFEM_GIT_STRING << std::endl
				<< "Velocity #DOFs: " << fes_sizeVel << std::endl
				<< "Pressure #DOFs: " << fes_sizePres << std::endl
				<< "Total    #DOFs: " << fes_sizeVel + fes_sizePres << std::endl;
	}
}

void SNavierPicardDGSolver::PrintMatricesVectors( const char* id, int num )
{
	//TODO: May not need it. Complete it later.
}

/// Destructor
SNavierPicardDGSolver::~SNavierPicardDGSolver()
{
	//delete pmesh;

	delete sigfec;
	delete ufec;
	delete pfec;
	delete sigfes;
	delete ufes;
	delete pfes;

	delete uk_coeff;

	delete ainv_form;
	delete b_form;
	delete c_form;
	delete d_form;

	delete f_form;
	delete g_form;
	delete h_form;

	delete x_bvec;
	delete truex_bvec;
	delete rhs_bvec;
	delete truerhs_bvec;

	delete OseenMono;
	delete Ainv_mat;
	delete B_mat;
	delete Bt_mat;
	delete C_mat;
	delete D_mat;
	delete BtAinv_mat;
	delete BtAinvB_mat;
	delete BtAinvB_C_mat;
	delete zero_mat;

	delete F_vec;
	delete BtAinvF_vec;
	delete G_vec;

	delete fcoeff;
	delete traction;

	delete paraview_dc;
	delete visit_dc;
}

} // end of mfem namespace
