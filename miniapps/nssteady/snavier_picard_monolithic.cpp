#include "snavier_picard_monolithic.hpp"

namespace mfem{

/// Constructor
SNavierPicardMonolithicSolver::SNavierPicardMonolithicSolver(ParMesh* mesh,
                                             int vorder,
                                             int porder,
                                             double kin_vis_,
                                             bool verbose):
pmesh(mesh), vorder(vorder), porder(porder), verbose(verbose)
{
   // mesh
   dim=pmesh->Dimension();

   // FE collection and spaces for velocity and pressure
   vfec=new H1_FECollection(vorder,dim);
   pfec=new H1_FECollection(porder);
   vfes=new ParFiniteElementSpace(pmesh,vfec,dim);
   pfes=new ParFiniteElementSpace(pmesh,pfec,1);

   // determine spaces dimension
   vdim = vfes->GetTrueVSize();
   pdim = pfes->GetTrueVSize(); 
   
   block_offsets.SetSize(3);
   block_offsets[0] = 0;
   block_offsets[1] = vdim;
   block_offsets[2] = pdim;
   block_offsets.PartialSum();

   // initialize vectors of essential attributes
   vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());      vel_ess_attr=0;
   vel_ess_attr_x.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_x=0;
   vel_ess_attr_y.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_y=0;
   vel_ess_attr_z.SetSize(pmesh->bdr_attributes.Max());  vel_ess_attr_z=0;
   ess_attr_tmp.SetSize(pmesh->bdr_attributes.Max());      ess_attr_tmp=0;

   // initialize GridFunctions
   v_gf  = new ParGridFunction(vfes);  *v_gf = 0.0;
   vk_gf = new ParGridFunction(vfes); *vk_gf = 0.0;
   p_gf  = new ParGridFunction(pfes);  *p_gf = 0.0;
   pk_gf = new ParGridFunction(pfes); *pk_gf = 0.0;

   // initialize vectors
   x   = new BlockVector(block_offsets);  *x = 0.0;
   x_k = new BlockVector(block_offsets); *x_k = 0.0;
   rhs = new BlockVector(block_offsets); *rhs = 0.0; 
   fv  = new Vector(vdim); *fv = 0.0; 
   
   // setup GridFunctionCoefficients
   vk_vc = new VectorGridFunctionCoefficient(vk_gf);
   pk_c  = new GridFunctionCoefficient(pk_gf);

   // pressure lift
   lift = 0.0;

   // set default parameters alpha and gamma
   alpha0 = 1.0;
   gamma  = 1.0;

   // set kinematic viscosity
   kin_vis.constant = kin_vis_;

   // Error computation setup
   err_v = err_p = 0;
   norm_v = norm_p = 0;
   int order_quad = std::max(2, 2*vorder+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

}



/// Public Interface

// Boundary conditions
void SNavierPicardMonolithicSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   // Check for duplicate
   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                  "Duplicate boundary definition detected.");
      if (attr[i] == 1)
      {
         vel_ess_attr[i] = 1;
      }
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC (full) to attributes: ";
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

void SNavierPicardMonolithicSolver::AddVelDirichletBC(VecFunc func, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void SNavierPicardMonolithicSolver::AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir)
{
   // Add bc container to list of componentwise velocity bcs
   vel_dbcs_xyz.emplace_back(attr, coeff, dir);

   // Check for duplicate and add attributes for current bc to global list (for that specific component)
   for (int i = 0; i < attr.Size(); ++i)
   {
      switch (dir) {
            case 0: // x 
               dir_string = "x";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i]) && attr[i]) == 0,
                           "Duplicate boundary definition for x component detected.");
               if (attr[i] == 1){vel_ess_attr_x[i] = 1;}
               break;
            case 1: // y
               dir_string = "y";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_y[i]) && attr[i]) == 0,
                           "Duplicate boundary definition for y component detected.");
               if (attr[i] == 1){vel_ess_attr_y[i] = 1;}
               break;
            case 2: // z
               dir_string = "z";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                           "Duplicate boundary definition for z component detected.");
               if (attr[i] == 1){vel_ess_attr_z[i] = 1;}
               break;
            default:;
         }      
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC ( " << dir_string << " component) to attributes: " << std::endl;
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << ", ";
         }
      }
      mfem::out << std::endl;
   }
}


void SNavierPicardMonolithicSolver::AddVelDirichletBC(VectorCoefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, ess_attr_tmp);
}

void SNavierPicardMonolithicSolver::AddVelDirichletBC(VecFunc func, int &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void SNavierPicardMonolithicSolver::AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, ess_attr_tmp, dir);
}



void SNavierPicardMonolithicSolver::AddTractionBC(VectorCoefficient *coeff, Array<int> &attr)
{
   traction_bcs.emplace_back(attr, coeff);

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
   }

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Traction (Neumann) BC to attributes: ";
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

void SNavierPicardMonolithicSolver::AddTractionBC(VecFunc func, Array<int> &attr)
{
   AddTractionBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}

void SNavierPicardMonolithicSolver::AddTractionBC(VectorCoefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   trac_attr_tmp = 0;
   trac_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddTractionBC(coeff, trac_attr_tmp);
}

void SNavierPicardMonolithicSolver::AddTractionBC(VecFunc func, int &attr)
{
   AddTractionBC(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}


void SNavierPicardMonolithicSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Acceleration term to attributes: ";
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

void SNavierPicardMonolithicSolver::AddAccelTerm(VecFunc func, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), func), attr);
}


// Solver setup
void SNavierPicardMonolithicSolver::SetOuterSolver(SolverParams params)
{
   sParams = params;    
}

void SNavierPicardMonolithicSolver::SetGamma(double &gamma_)
{
   gamma = gamma_;
}

void SNavierPicardMonolithicSolver::SetAlpha(double &alpha_, const AlphaType &type_)
{
   alpha0    = alpha_;
   alphaType = type_;
}

void SNavierPicardMonolithicSolver::SetLift(double &lift_)
{
   lift    = lift_;
}

void SNavierPicardMonolithicSolver::SetInitialConditionVel(VectorCoefficient &v_in)
{
   // Project coefficient onto velocity ParGridFunction
   v_gf->ProjectCoefficient(v_in);
   v_gf->GetTrueDofs(x->GetBlock(0));
   //*x_k = *x;                         
   //vk_gf->SetFromTrueDofs(x_k->GetBlock(0));
}

void SNavierPicardMonolithicSolver::SetInitialConditionPres(Coefficient &p_in)
{
   // Project coefficient onto pressure ParGridFunction
   p_gf->ProjectCoefficient(p_in);
   p_gf->GetTrueDofs(x->GetBlock(1));
}

void SNavierPicardMonolithicSolver::Setup()
{
   /// 1. Setup and assemble bilinear forms 
   K_form = new ParBilinearForm(vfes);
   C_form = new ParBilinearForm(vfes);
   B_form = new ParMixedBilinearForm(vfes, pfes);
   K_form->AddDomainIntegrator(new VectorDiffusionIntegrator(kin_vis));
   B_form->AddDomainIntegrator(new VectorDivergenceIntegrator());
   K_form->Assemble();  K_form->Finalize();
   B_form->Assemble();  B_form->Finalize();
   K = K_form->ParallelAssemble();
   B = B_form->ParallelAssemble();


   /// 2. Assemble linear form for rhs
   f_form = new ParLinearForm(vfes);
   // Adding forcing terms
   for (auto &accel_term : accel_terms)
   {
      f_form->AddDomainIntegrator( new VectorDomainLFIntegrator( *(accel_term.coeff) ) );  // NOTE: need to modify this including domain attr, if having multiple domains
   }
   // Adding traction bcs
   for (auto &traction_bc : traction_bcs)
   {
      f_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator( *(traction_bc.coeff)) , traction_bc.attr);
   }
   f_form->Assemble(); 
   fv = f_form->ParallelAssemble(); 
   

   /// 3. Apply boundary conditions
   // Extract to list of true dofs
   vfes->GetEssentialTrueDofs(vel_ess_attr_x,vel_ess_tdof_x,0);
   vfes->GetEssentialTrueDofs(vel_ess_attr_y,vel_ess_tdof_y,1);
   vfes->GetEssentialTrueDofs(vel_ess_attr_z,vel_ess_tdof_z,2);
   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof_full);
   vel_ess_tdof.Append(vel_ess_tdof_x);
   vel_ess_tdof.Append(vel_ess_tdof_y);
   vel_ess_tdof.Append(vel_ess_tdof_z);
   vel_ess_tdof.Append(vel_ess_tdof_full);

   // Projection of coeffs (full velocity applied)
   for (auto &vel_dbc : vel_dbcs)
   {
      v_gf->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }
   v_gf->GetTrueDofs(x->GetBlock(0));

   // Projection of coeffs (velocity component applied)
   ParGridFunction tmp_gf(vfes);        // temporary velocity gf for projection
   Vector          tmp_vec(vdim);       // temporary velocity vector for projection
   Array<int>      tmp_tdofs;
   for (auto &vel_dbc : vel_dbcs_xyz)
   {
      VectorArrayCoefficient tmp_coeff(dim);                           // Set coefficient with right component
      tmp_coeff.Set(vel_dbc.dir, vel_dbc.coeff, false);
      tmp_gf.ProjectBdrCoefficient(tmp_coeff, vel_dbc.attr);           // Project on dummy gf
      tmp_gf.GetTrueDofs(tmp_vec);

      vfes->GetEssentialTrueDofs(vel_dbc.attr,tmp_tdofs,vel_dbc.dir);  // Update solution dofs
      for(int i=0;i<tmp_tdofs.Size();i++)
      {
         (x->GetBlock(0))[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
      }      
   }
   // Initialize solution gf with vector containing projected coefficients
   // and update grid function and vector for provisional velocity
   v_gf->SetFromTrueDofs(x->GetBlock(0));


   /// 4. Create Navier-Stokes block operator
   // Modify matrices for essential velocity bcs
   Be  = B->EliminateCols(vel_ess_tdof);
   G   = B->Transpose(); 
   (*G)  *= -1.0;

   // Modify rhs for pressure ( no pressure ess bcs)   fp = 0 - Be v
   Orthogonalize(rhs->GetBlock(1));
   Be->Mult(-1.0, x->GetBlock(0), 1.0, rhs->GetBlock(1)); // rhs_p -= Be*v
   
   // Flip signs of the second block part to make system positive definite.
   (*B) *= -1.0;
   rhs->GetBlock(1) *= -1.0;

   nsOp = new BlockOperator(block_offsets);
   nsOp->SetBlock(0, 1, G);
   nsOp->SetBlock(1, 0, B);



   /// 5. Construct the operators for the block preconditioner
   //
   //                 P = [  A           0        ]
   //                     [  0    (kin_vis)^-1 Mp ]
   //
   //     Here we use a single VCycle of AMG for the convective-diffusive partA =  K + alpha C,
   //     and G, and Jacobi preconditioner on pressure mass matrix Mp to approximate the 
   //     inverse of the pressure Schur Complement.
   Mp_form = new ParBilinearForm(pfes);
   Mp_form->AddDomainIntegrator(new MassIntegrator);
   Mp_form->Assemble(); Mp_form->Finalize();
   Mp = Mp_form->ParallelAssemble();
   *Mp *= 1.0/kin_vis.constant;
   //HypreDiagScale *invMp = new HypreDiagScale(*Mp);
   invMp = new HypreBoomerAMG(*Mp);
   invMp->SetPrintLevel(0);
   invMpOrtho = new OrthoSolver(pmesh->GetComm());
   invMpOrtho->SetSolver(*invMp);

   // Convection-diffusion (operator will be assigned inside loop)
   invA = new HypreBoomerAMG();
   invA->SetPrintLevel(0);
   invA->iterative_mode = false;

   nsPrec = new BlockDiagonalPreconditioner(block_offsets);
   nsPrec->SetDiagonalBlock(1, invMpOrtho);


   /// 6. Create GMRES solver (system is not symmetric).
   //
   solver = new GMRESSolver(pmesh->GetComm());
   solver->iterative_mode = false;
   solver->SetAbsTol(0.0);
   solver->SetRelTol(sParams.rtol);
   solver->SetMaxIter(sParams.maxIter);
   solver->SetOperator(*nsOp);
   solver->SetPreconditioner(*nsPrec);
   solver->SetPrintLevel(sParams.pl);
}

void SNavierPicardMonolithicSolver::FSolve()
{
#ifdef MFEM_DEBUG
   PrintMatricesVectors( "setup", 0); // Export matrices/vectors before step
#endif

   PrintInfo();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "================================================================"<< std::endl;
      mfem::out << "======    Picard Monolithic Steady Navier-Stokes Solver    ======"<< std::endl;
      mfem::out << "================================================================"<< std::endl;
   }

   timer.Clear();
   timer.Start();

   // Print header
   if ( pmesh->GetMyRank() == 0)
   {
      out << std::endl;
      out << std::setw(2) << "It" << std::setw(13) << "" << std::setw(10)
             << "Res" << std::setw(12) << "AbsTol" << "\n";
   }


   // Export initial solution
   SaveResults( 0 );

   for (iter = 0; iter < sParams.maxIter; iter++)
   {
      // Update parameter alpha
      UpdateAlpha();

      // Solve current iteration.
      Step();

      // Output results
      SaveResults( iter+1 );

      // Compute errors.
      ComputeError();

      // Update solution at previous iterate and gridfunction coefficients.
      UpdateSolution();

      // Print results
      if ( pmesh->GetMyRank() == 0 )
      {
         out << iter << "   " << "|| v - v_k ||  " << std::setw(3)
                << std::setprecision(2) << std::scientific << err_v
                << "   " << sParams.atol << std::endl;
         out << iter << "   " << "|| p - v_k ||  " << std::setw(3)
                  << std::setprecision(2) << std::scientific << err_p
                  << "   " << sParams.atol << std::endl;
         out << std::endl;
      }

      // Check convergence.
      if (err_v < sParams.atol)
      {
         if ( pmesh->GetMyRank() == 0 ) 
         { out << "Solver converged to steady state solution \n";}
         flag = 1;
         break;
      }

   }

   timer.Stop();
}


void SNavierPicardMonolithicSolver::SetupOutput( const char* folderPath, bool visit_, bool paraview_,
                                         DataCollection::Format par_format )
{
   visit    = visit_;
   paraview = paraview_;
   outfolder = folderPath;

   p_gf_out = new ParGridFunction(pfes); *p_gf_out=0.0;

   // Creating output directory if not existent
   if (mkdir(folderPath, 0777) == -1) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

   // Create output data collections   
   if( visit )
   {
      visit_dc = new VisItDataCollection("Results-VISit", pmesh);
      visit_dc->SetPrefixPath(folderPath);
      visit_dc->RegisterField("velocity", v_gf);
      visit_dc->RegisterField("pressure", p_gf_out);
      visit_dc->SetFormat(par_format);
   }

   if( paraview )
   {
      paraview_dc = new ParaViewDataCollection("Results-Paraview", pmesh);
      paraview_dc->SetPrefixPath(folderPath);
      paraview_dc->SetLevelsOfDetail(vorder);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->RegisterField("velocity",v_gf);
      paraview_dc->RegisterField("pressure",p_gf_out);
   }   

}

/// Private Interface

void SNavierPicardMonolithicSolver::Step()
{
   // Assemble convective term 
   delete C_form; C_form = nullptr;
   C_form = new ParBilinearForm(vfes);
   C_form->AddDomainIntegrator(new VectorConvectionIntegrator(*vk_vc, 1.0));
   C_form->Assemble(); C_form->Finalize();
   C = C_form->ParallelAssemble();  

   A = Add(1.0, *K, alpha, *C);                      // A = K + alpha C                     

   // Assemble RHS velocity block
   (rhs->GetBlock(0)).Set(1.0,*fv);                                   // rhs_v = fv
   C->AddMult(x_k->GetBlock(0),rhs->GetBlock(0), alpha-1.);  // rhs_v += (alpha - 1) * C * vk

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "prestep", iter);  // Export matrices/vectors after assembly of A, before modifications
#endif
  
   // Apply bcs
   Ae = A->EliminateRowsCols(vel_ess_tdof);
   A->EliminateBC(*Ae, vel_ess_tdof, x->GetBlock(0), rhs->GetBlock(0)); // rhs_v -= Ae*v

   // Update block operator and block preconditioner
   invA->SetOperator(*A);     
   nsOp->SetBlock(0, 0, A);
   nsPrec->SetDiagonalBlock(0, invA);

   // Solve current iteration.
   solver->Mult(*rhs, *x);

   // Remove nullspace by removing mean of the pressure solution (also ensured by the OrthoSolver wrapper)
   p_gf->SetFromTrueDofs(x->GetBlock(1));
   MeanZero(*p_gf);
   p_gf->GetTrueDofs(x->GetBlock(1));

   // Relaxation                 
   add(gamma,*x,(1-gamma),*x_k,*x);

   // Retrieve solution
   v_gf->SetFromTrueDofs(x->GetBlock(0));
   p_gf->SetFromTrueDofs(x->GetBlock(1));

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "step", iter);  // Export matrices/vectors after 1st step
#endif

   delete A; A = nullptr;
   delete Ae; Ae = nullptr;
   delete C; C = nullptr;
}

void SNavierPicardMonolithicSolver::ComputeError()
{
   err_v  = v_gf->ComputeL2Error(*vk_vc);
   norm_v = ComputeGlobalLpNorm(2., *vk_vc, *pmesh, irs);
   err_p  = p_gf->ComputeL2Error(*pk_c);
   norm_p = ComputeGlobalLpNorm(2., *pk_c, *pmesh, irs);
}

void SNavierPicardMonolithicSolver::UpdateSolution()
{
   *x_k = *x;
   vk_gf->SetFromTrueDofs(x_k->GetBlock(0));
   pk_gf->SetFromTrueDofs(x_k->GetBlock(1));  
}


void SNavierPicardMonolithicSolver::UpdateAlpha()
{
   if ( alphaType == AlphaType::CONSTANT) { alpha = alpha0;}
   else {  MFEM_ABORT("Error: SNavierPicardMonolithicSolver::UpdateAlpha does not implement"
                       "adaptive update of the segregation parameter yet!");} // NYI!
}


void SNavierPicardMonolithicSolver::MeanZero(ParGridFunction &p)
{
   // Make sure not to recompute the inner product linear form every
   // application.
   if (mass_lf == nullptr)
   {
      ConstantCoefficient onecoeff(1.0);
      mass_lf = new ParLinearForm(p.ParFESpace());
      auto *dlfi = new DomainLFIntegrator(onecoeff);
      mass_lf->AddDomainIntegrator(dlfi);
      mass_lf->Assemble();

      ParGridFunction one_gf(p.ParFESpace());
      one_gf.ProjectCoefficient(onecoeff);

      volume = mass_lf->operator()(one_gf);
   }

   double integ = mass_lf->operator()(p);

   p -= integ / volume;
}

void SNavierPicardMonolithicSolver::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pmesh->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}

void SNavierPicardMonolithicSolver::SaveResults( int iter )
{

   // Add lift to pressure solution
   p_gf_out->ProjectGridFunction(*p_gf);
   *p_gf_out += lift;

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

void SNavierPicardMonolithicSolver::PrintInfo()
{
   int fes_sizeVel = vfes->GlobalVSize();
   int fes_sizePres = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "NAVIER version: " << MFEM_SNAVIER_MONOLITHIC_VERSION << std::endl
                << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Velocity #DOFs: " << fes_sizeVel << std::endl
                << "Pressure #DOFs: " << fes_sizePres << std::endl;
   }
}

#ifdef MFEM_DEBUG

void SNavierPicardMonolithicSolver::PrintMatricesVectors( const char* id, int num )
{
   // Create folder
   std::string folderName(outfolder);
   folderName += "/MatVecs_iter";
   folderName += std::to_string(num);

   if (mkdir(folderName.c_str(), 0777) == -1) {} //{mfem::err << "Error :  " << strerror(errno) << std::endl;}

   //Create files
   std::ofstream K_file(std::string(folderName) + '/' + "K_" + std::string(id) + ".dat");
   std::ofstream C_file(std::string(folderName) + '/' + "C_" + std::string(id) + ".dat");
   std::ofstream A_file(std::string(folderName) + '/' + "A_" + std::string(id) + ".dat");
   std::ofstream S_file(std::string(folderName) + '/' + "S_" + std::string(id) + ".dat");
   std::ofstream B_file(std::string(folderName) + '/' + "B_" + std::string(id) + ".dat");
   std::ofstream G_file(std::string(folderName) + '/' + "G_" + std::string(id) + ".dat");

   std::ofstream Be_file(std::string(folderName) + '/' + "Be_" + std::string(id) + ".dat");
   std::ofstream Ae_file(std::string(folderName) + '/' + "Ae_" + std::string(id) + ".dat");

   std::ofstream fv_file(std::string(folderName) + '/' + "fv_" + std::string(id) + ".dat");
   std::ofstream fp_file(std::string(folderName) + '/' + "fp_" + std::string(id) + ".dat");
   std::ofstream rhs_file(std::string(folderName) + '/' + "rhs" + std::string(id) + ".dat");
   std::ofstream dofs_file(std::string(folderName) + '/' + "dofs_" + std::string(id) + ".dat");


   std::ofstream v_file(std::string(folderName) + '/' + "v_" + std::string(id) + ".dat");
   std::ofstream vk_file(std::string(folderName) + '/' + "vk_" + std::string(id) + ".dat");
   std::ofstream p_file(std::string(folderName) + '/' + "p_" + std::string(id) + ".dat");
   std::ofstream pk_file(std::string(folderName) + '/' + "pk_" + std::string(id) + ".dat");
   std::ofstream x_file(std::string(folderName) + '/' + "x_" + std::string(id) + ".dat");

   // Print matrices in matlab format
   K->PrintMatlab(K_file);
   if(C==nullptr)
   {
      C = new HypreParMatrix();
      C->PrintMatlab(C_file);
      delete C; C = nullptr;
   }
   else
   {
      C->PrintMatlab(C_file);
   }

   if(A==nullptr)
   {
      A = new HypreParMatrix();
      A->PrintMatlab(C_file);
      delete A; A = nullptr;
   }
   else
   {
      A->PrintMatlab(A_file);
   }

   B->PrintMatlab(B_file);
   G->PrintMatlab(G_file);

   if(Ae==nullptr)
   {
      Ae = new HypreParMatrix();
      Ae->PrintMatlab(Ae_file);
      delete Ae; Ae = nullptr;
   }
   else
   {
      Ae->PrintMatlab(Ae_file);
   }
   
   Be->PrintMatlab(Be_file);

   fv->Print(fv_file,1);
   (rhs->GetBlock(1)).Print(fp_file,1);
   rhs->Print(rhs_file,1);

   (x->GetBlock(0)).Print(v_file,1);
   (x_k->GetBlock(0)).Print(vk_file,1);
   (x->GetBlock(1)).Print(p_file,1);
   (x_k->GetBlock(1)).Print(pk_file,1);
   x->Print(x_file,1);


   for (int i = 0; i < vel_ess_tdof.Size(); ++i)
   {
      dofs_file << vel_ess_tdof[i] << std::endl;
   }
   dofs_file.close();

}

#endif

/// Destructor
SNavierPicardMonolithicSolver::~SNavierPicardMonolithicSolver()
{
   delete vfes; vfes = nullptr;
   delete vfec; vfec = nullptr;
   delete pfes; pfes = nullptr;
   delete pfec; pfec = nullptr;

   delete v_gf; v_gf = nullptr;
   delete vk_gf; vk_gf = nullptr;
   delete p_gf; p_gf = nullptr;
   delete pk_gf; pk_gf = nullptr;
   delete p_gf_out; p_gf_out = nullptr;

   delete K_form; 
   delete B_form;
   delete C_form; 
   delete f_form;
   delete Mp_form; 

   delete K;     K = nullptr;
   delete B;     B = nullptr;
   delete G;    G = nullptr;
   delete Be;    Be = nullptr;

   delete fv;    fv = nullptr;
   delete rhs;  rhs = nullptr;

   delete vk_vc; vk_vc = nullptr;
   delete pk_c;  pk_c = nullptr;

   delete fcoeff;  fcoeff = nullptr;
   delete traction; traction = nullptr;

   delete invMp;     invMp = nullptr;
   delete invMpOrtho;     invMpOrtho = nullptr;

   delete nsOp;  nsOp = nullptr;
   delete nsPrec;  nsPrec = nullptr;

   delete paraview_dc; paraview_dc = nullptr;
   delete visit_dc;    visit_dc = nullptr;

   delete mass_lf; mass_lf = nullptr;

}

}
