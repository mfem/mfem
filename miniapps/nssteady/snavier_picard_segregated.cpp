#include "snavier_picard_segregated.hpp"

namespace mfem{

/// Constructor
SNavierPicardCGSolver::SNavierPicardCGSolver(ParMesh* mesh,
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
   vfes=new ParFiniteElementSpace(pmesh,vfec,dim);
   pfec=new H1_FECollection(porder);
   pfes=new ParFiniteElementSpace(pmesh,pfec,1);

   // determine spaces dimension
   vdim = vfes->GetTrueVSize();
   pdim = pfes->GetTrueVSize(); 
   
   // initialize vectors of essential attributes
   vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());      vel_ess_attr=0;
   vel_ess_attr_x.SetSize(pmesh->bdr_attributes.Max());    vel_ess_attr_x=0;
   vel_ess_attr_y.SetSize(pmesh->bdr_attributes.Max());    vel_ess_attr_y=0;
   vel_ess_attr_z.SetSize(pmesh->bdr_attributes.Max());    vel_ess_attr_z=0;

   // initialize GridFunctions
   v_gf.SetSpace(vfes);  v_gf=0.0;
   vk_gf.SetSpace(vfes); vk_gf=0.0;
   z_gf.SetSpace(vfes);  z_gf=0.0;
   p_gf.SetSpace(pfes);  p_gf=0.0;
   pk_gf.SetSpace(pfes); pk_gf=0.0;

   // initialize vectors
   v    = new Vector(vdim); *v = 0.0;
   vk   = new Vector(vdim); *vk = 0.0; 
   z    = new Vector(vdim); *z = 0.0; 
   p    = new Vector(pdim); *p = 0.0; 
   pk   = new Vector(pdim); *pk = 0.0; 
   fv   = new Vector(vdim); *fv = 0.0; 
   fp   = new Vector(pdim); *fp = 0.0; 
   rhs1 = new Vector(vdim); *rhs1 = 0.0; 
   rhs2 = new Vector(pdim); *rhs2 = 0.0;
   rhs3 = new Vector(vdim); *rhs3 = 0.0; 
   tmp  = new Vector(vdim); *tmp = 0.0; 
   
   // setup GridFunctionCoefficients
   vk_vc = new VectorGridFunctionCoefficient(&vk_gf);
   pk_c  = new GridFunctionCoefficient(&pk_gf);

   // pressure lift
   lift = 0.0;

   // set default parameters alpha and gamma
   alpha0 = 1;
   gamma  = 1;

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

   // Variable to enable solution of Navier-Stokes problem
   ns_coeff = 1.0;

}



/// Public Interface
void SNavierPicardCGSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
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
      mfem::out << "Adding Velocity Dirichlet BC (full) to attributes ";
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

void SNavierPicardCGSolver::AddVelDirichletBC(Coefficient *coeff, Array<int> &attr, int &dir)
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

void SNavierPicardCGSolver::AddVelDirichletBC(VectorCoefficient *coeff, int &attr)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, ess_attr_tmp);
}

void SNavierPicardCGSolver::AddVelDirichletBC(Coefficient *coeff, int &attr, int &dir)
{
   // Create array for attributes and mark given mark given mesh boundary
   ess_attr_tmp = 0;
   ess_attr_tmp[ attr - 1] = 1;

   // Call AddVelDirichletBC accepting array of essential attributes
   AddVelDirichletBC(coeff, ess_attr_tmp, dir);
}

void SNavierPicardCGSolver::AddTractionBC(VectorCoefficient *coeff, Array<int> &attr)
{
   traction_bcs.emplace_back(attr, coeff);

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
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

void SNavierPicardCGSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
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

void SNavierPicardCGSolver::SetOuterSolver(SolverParams params)
{
   sParams = params;    
}

void SNavierPicardCGSolver::SetInnerSolvers( SolverParams params1,
                                            SolverParams params2,
                                            SolverParams params3)
{
   s1Params = params1;
   s2Params = params2;
   s3Params = params3;                          
}

void SNavierPicardCGSolver::SetGamma(double &gamma_)
{
   gamma = gamma_;
}

void SNavierPicardCGSolver::SetAlpha(double &alpha_, const AlphaType &type_)
{
   alpha0    = alpha_;
   alphaType = type_;
}

void SNavierPicardCGSolver::SetLift(double &lift_)
{
   lift    = lift_;
}

void SNavierPicardCGSolver::EnableStokes()
{
   ns_coeff = 0.0;
}

void SNavierPicardCGSolver::SetInitialConditionVel(VectorCoefficient &v_in)
{
   // Project coefficient onto velocity ParGridFunction
   v_gf.ProjectCoefficient(v_in);

   // Initialize provisional velocity and velocity at previous iteration
   v_gf.GetTrueDofs(*v);
   *z = *v;
   z_gf.SetFromTrueDofs(*z);
   //*vk = *v;                         // CHECK: do we need to initialize also vk?
   //vk_gf.SetFromTrueDofs(*vk);
}

void SNavierPicardCGSolver::SetInitialConditionPres(Coefficient &p_in)
{
   // Project coefficient onto pressure ParGridFunction
   p_gf.ProjectCoefficient(p_in);

   // Initialize pressure at previous iteration
   p_gf.GetTrueDofs(*p);
   //*pk = *p;                     // CHECK: do we need to initialize also pk?
   //pk_gf.SetFromTrueDofs(*pk);
}

void SNavierPicardCGSolver::Setup()
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


   /// 2. Setup and assemble linear form for rhs
   f_form = new ParLinearForm(vfes);
   // Adding forcing terms
   for (auto &accel_term : accel_terms)
   {
      f_form->AddDomainIntegrator( new VectorDomainLFIntegrator(*accel_term.coeff) );  // NOTE: need to modify this including domain attr, if having multiple domains
   }
   // Adding traction bcs
   for (auto &traction_bc : traction_bcs)
   {
      f_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*traction_bc.coeff), traction_bc.attr);
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
      v_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }
   v_gf.GetTrueDofs(*v);

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
         v[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
      }      
   }
   // Initialize solution gf with vector containing projected coefficients
   // and update grid function and vector for provisional velocity
   *z  = *v;
   v_gf.SetFromTrueDofs(*v);
   z_gf.SetFromTrueDofs(*z);


   /// 4. Apply transformation for essential bcs
   // NOTE: alternatively the following function performs the modification on both matrix and rhs
   //       EliminateRowsCols(const Array<int> &rows_cols, const HypreParVector &x, HypreParVector &b)
   Ke  = K->EliminateRowsCols(vel_ess_tdof);  // Remove rows/cols for ess tdofs
   Be  = B->EliminateCols(vel_ess_tdof);
   G   = B->Transpose(); 
   (*G)  *= -1.0;

   // Modify rhs for pressure ( no pressure ess bcs)   fp = 0 - Be v
   Be->Mult(-1.0, *v, 1.0, *fp); // rhs_p -= Be*v
 
   // Assemble the operator S
   HypreParVector* Kd = new HypreParVector(pmesh->GetComm(), K->GetGlobalNumRows(), K->GetRowStarts());
   HypreParMatrix* local = new HypreParMatrix(*G); // local = G
   K->GetDiag(*Kd);
   local->InvScaleRows(*Kd);  // local = Kd^{-1} G
   S = ParMult(B, local);     // S = B Kd^{-1} G
   *S *= -1.0;                // Change sign to make it SPD
   delete local; local=nullptr;
   delete Kd; Kd = nullptr;


   /// 5. Setup solvers and preconditioners
   //  5.1 Velocity prediction       A = K + alpha*C(uk)
   // solved with CGSolver preconditioned with HypreBoomerAMG (can enable elasticity version)
   invA_pc = new HypreBoomerAMG();
   invA_pc->SetSystemsOptions(dim);
   invA_pc->SetPrintLevel(0);
   //invA_pc->SetElasticityOptions(vfes);
   invA = new GMRESSolver(pmesh->GetComm());
   invA->iterative_mode = false;     // uses 2nd argument of mult as initial guess
   invA->SetPreconditioner(*invA_pc);
   invA->SetPrintLevel(s1Params.pl);
   invA->SetRelTol(s1Params.rtol);
   //invA->SetAbsTol(s1Params.atol);
   invA->SetMaxIter(s1Params.maxIter);


   // 5.2 Pressure correction       S = B K^{-1} G
   // solved with CGSolver preconditioned with HypreBoomerAMG 
   // NOTE: test different approaches to deal with Schur Complement:
   // * now using Jacobi, preconditioned with Mass matrix, but this may not be a good approximation when involving Brinkman Volume Penalization 
   // * we may solve the problem iteratively with preconditioned Richardson
   Mp_form = new ParBilinearForm(pfes);
   Mp_form->AddDomainIntegrator(new MassIntegrator);
   Mp_form->Assemble(); Mp_form->Finalize();
   Mp = Mp_form->ParallelAssemble();
   *Mp *= 1.0/kin_vis.constant;
   //HypreDiagScale *invMp = new HypreDiagScale(*Mp);
   invMp = new HypreBoomerAMG(*Mp);
   invMp->SetPrintLevel(0);

   invS_pc = new HypreBoomerAMG(*S);
   invS_pc->SetSystemsOptions(dim);
   invS_pc->SetPrintLevel(0);

   OrthoSolver* invS_pc_ortho = new OrthoSolver(pmesh->GetComm());
   invS_pc_ortho->SetSolver(*invS_pc);
   //invS_pc_ortho->SetSolver(*invMp);

   invS = new CGSolver(pmesh->GetComm());
   invS->iterative_mode = true;
   invS->SetOperator(*S);
   //invS->SetPreconditioner(*invS_pc);
   invS->SetPreconditioner(*invS_pc_ortho);
   invS->SetPrintLevel(s2Params.pl);
   invS->SetRelTol(s2Params.rtol);
   //invS->SetAbsTol(s2Params.atol);
   invS->SetMaxIter(s2Params.maxIter);
   delete B_form; B_form = nullptr;
   delete Mp_form; Mp_form = nullptr;


   // 5.3 Velocity correction
   // solved with CGSolver preconditioned with HypreBoomerAMG 
   invK_pc = new HypreBoomerAMG(*K);
   //invK_pc->SetSystemsOptions(dim);
   invK_pc->SetPrintLevel(0);
   invK = new CGSolver(pmesh->GetComm());
   invK->iterative_mode = true;
   invK->SetOperator(*K);
   invK->SetPreconditioner(*invK_pc);
   invK->SetPrintLevel(s3Params.pl);
   invK->SetRelTol(s3Params.rtol);
   //invK->SetAbsTol(s3Params.atol);
   invK->SetMaxIter(s3Params.maxIter);
   delete K_form; K_form = nullptr;
}

void SNavierPicardCGSolver::FSolve()
{
#ifdef MFEM_DEBUG
   PrintMatricesVectors( "setup", 0); // Export matrices/vectors before step
#endif

   PrintInfo();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "=========================================================="<< std::endl;
      mfem::out << "======    Picard-aCT Steady Navier-Stokes Solver    ======"<< std::endl;
      mfem::out << "=========================================================="<< std::endl;
   }

   timer.Clear();
   timer.Start();

   // Print header
   mfem::out << std::endl;
   mfem::out << std::setw(2) << "It" << std::setw(13) << "" << std::setw(10)
             << "Res" << std::setw(12) << "AbsTol" << "\n";


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
      mfem::out << iter << "   " << "|| v - v_k ||  " << std::setw(3)
                << std::setprecision(2) << std::scientific << err_v
                << "   " << sParams.atol << std::endl;
      mfem::out << iter << "   " << "|| p - v_k ||  " << std::setw(3)
                << std::setprecision(2) << std::scientific << err_p
                << "   " << sParams.atol << std::endl;

      // Check convergence.
      if (err_v < sParams.atol)
      {
         out << "Solver converged to steady state solution \n";
         flag = 1;
         break;
      }

   }

   timer.Stop();
}


void SNavierPicardCGSolver::SetupOutput( const char* folderPath, bool visit_, bool paraview_,
                                         DataCollection::Format par_format )
{
   visit    = visit_;
   paraview = paraview_;
   outfolder = folderPath;

   p_gf_out.SetSpace(pfes); p_gf_out=0.0;

   // Creating output directory if not existent
   if (mkdir(folderPath, 0777) == -1) {mfem::err << "Error :  " << strerror(errno) << std::endl;}

   // Create output data collections   
   if( visit )
   {
      visit_dc = new VisItDataCollection("Results-VISit", pmesh);
      visit_dc->SetPrefixPath(folderPath);
      visit_dc->RegisterField("velocity", &v_gf);
      visit_dc->RegisterField("intermediate velocity", &z_gf);
      visit_dc->RegisterField("pressure", &p_gf_out);
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
      paraview_dc->RegisterField("velocity",&v_gf);
      paraview_dc->RegisterField("intermediate velocity",&z_gf);
      paraview_dc->RegisterField("pressure",&p_gf_out);
   }   

}

/// Private Interface

void SNavierPicardCGSolver::Step()
{
   /// Assemble convective term with new velocity vk and modify matrix for essential bcs.
   delete C_form; C_form = nullptr;
   C_form = new ParBilinearForm(vfes);
   C_form->AddDomainIntegrator(new VectorConvectionIntegrator(*vk_vc, ns_coeff));
   C_form->Assemble(); C_form->Finalize();
   C = C_form->ParallelAssemble();              


   /// Solve.
   // 1: Velocity prediction      ( K + alpha*C(vk) ) z = f - (1-alpha)*C(uk)*k
   // Assemble rhs
   rhs1->Set(1.0,*fv);                         // rhs1 = fv
   C->AddMult(*vk,*rhs1,alpha-1.0);            // rhs1 += (alpha-1)*C*vk
   
   // Assemble operator
   A = Add(1.0, *K, alpha, *C);                // A = Km + alpha C                   
   A->Add(1.0, *Ke);                           // A = Km + Ke + alpha C 

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "prestep", iter);  // Export matrices/vectors after assembly of A, before modifications
#endif
  
   Ae = A->EliminateRowsCols(vel_ess_tdof);
   A->EliminateBC(*Ae, vel_ess_tdof, *z, *rhs1);   
   
   invA->SetOperator(*A);     
   invA_pc->SetOperator(*A);

   invA->Mult(*rhs1,*z);

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "step1", iter);  // Export matrices/vectors after 1st step
#endif

   // 2: Pressure correction                   B*K^-1*G p = B*z - fp
   rhs2->Set(-1.0,*fp);                        // rhs2 = -fp   
   B->AddMult(*z,*rhs2,1.0);                   // rhs2 += B z
   Orthogonalize(*rhs2);
   
   rhs2->Neg();  // Change sign as we are using S = - B K^-1 G   to make it SPD

   invS->Mult(*rhs2, *p);

   p_gf.SetFromTrueDofs(*p);
   MeanZero(p_gf);
   p_gf.GetTrueDofs(*p);

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "step2", iter);  // Export matrices/vectors after 2nd step
#endif

   // 3: Velocity correction         K u = K*z - G*p
   // NOTE: Could be more efficient storing and reusing rhs1, SparseMatrix -alpha C(uk)
   K->Mult(*z,*rhs3);               // rhs3 = K z
   G->AddMult(*p,*rhs3,-1.0);       // rhs3  -= G p
   K->EliminateBC(*Ke, vel_ess_tdof, *v, *rhs3);

   invK->Mult(*rhs3,*v);

#ifdef MFEM_DEBUG
   PrintMatricesVectors( "step3", iter);  // Export matrices/vectors after 3rd step
#endif

   // 4: Relaxation step    v = gamma * v + (1 - gamma) * v_k
   add(gamma,*v,(1-gamma),*vk,*v);

   /// Update GridFunctions for solution.
   v_gf.SetFromTrueDofs(*v);
   z_gf.SetFromTrueDofs(*z);

   delete A; A = nullptr;
   delete Ae; Ae = nullptr;
   delete C; C = nullptr;
}

void SNavierPicardCGSolver::ComputeError()
{
   err_v  = v_gf.ComputeL2Error(*vk_vc);
   norm_v = ComputeGlobalLpNorm(2., *vk_vc, *pmesh, irs);
   err_p  = p_gf.ComputeL2Error(*pk_c);
   norm_p = ComputeGlobalLpNorm(2., *pk_c, *pmesh, irs);

}

void SNavierPicardCGSolver::UpdateSolution()
{
   *vk = *v;
   *z  = *v;
   z_gf.SetFromTrueDofs(*z);
   vk_gf.SetFromTrueDofs(*vk);

   *pk = *p;
   pk_gf.SetFromTrueDofs(*pk);
}


void SNavierPicardCGSolver::UpdateAlpha()
{
   if ( alphaType == AlphaType::CONSTANT) { alpha = alpha0;}
   else {  MFEM_ABORT("Error: SNavierPicardCGSolver::UpdateAlpha does not implement"
                       "adaptive update of the segregation parameter yet!");} // NYI!
}


void SNavierPicardCGSolver::ModifyRHS(Array<int> &ess_tdof_list, HypreParMatrix* mat_e,
                                      Vector &sol, Vector &rhs, bool copy_sol)
{
   // Initialize temporary vector for solution
   Vector tmp(sol);
   tmp.SetSubVectorComplement(ess_tdof_list, 0.0);

   // Perform elimination
   mat_e->Mult(-1.0, tmp, 1.0, rhs); // rhs -= mat_e*sol
 
   // Set rhs equal to solution at essential tdofs
   if( copy_sol )
   {
      int idx;
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         idx = ess_tdof_list[i];
         rhs(idx) = sol(idx);
      }
   }

}

void SNavierPicardCGSolver::FullMult(HypreParMatrix* mat, HypreParMatrix* mat_e, Vector &x, Vector &y)
{
   mat->Mult(x, y);        // y =  mat x
   mat_e->AddMult(x, y);   // y += mat_e x
}

void SNavierPicardCGSolver::MeanZero(ParGridFunction &p)
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

void SNavierPicardCGSolver::Orthogonalize(Vector &v)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pmesh->GetComm());

   v -= global_sum / static_cast<double>(global_size);
}

void SNavierPicardCGSolver::SaveResults( int iter )
{

   // Add lift to pressure solution
   p_gf_out.ProjectGridFunction(p_gf);
   p_gf_out += lift;

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

void SNavierPicardCGSolver::PrintInfo()
{
   int fes_sizeVel = vfes->GlobalVSize();
   int fes_sizePres = pfes->GlobalVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::endl;
      mfem::out << "NAVIER version: " << SNAVIER_CG_VERSION << std::endl
                << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Velocity #DOFs: " << fes_sizeVel << std::endl
                << "Pressure #DOFs: " << fes_sizePres << std::endl;
   }
}

#ifdef MFEM_DEBUG

void SNavierPicardCGSolver::PrintMatricesVectors( const char* id, int num )
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

   std::ofstream Ke_file(std::string(folderName) + '/' + "Ke_" + std::string(id) + ".dat");
   std::ofstream Be_file(std::string(folderName) + '/' + "Be_" + std::string(id) + ".dat");
   std::ofstream Ge_file(std::string(folderName) + '/' + "Ge_" + std::string(id) + ".dat");
   std::ofstream Ae_file(std::string(folderName) + '/' + "Ae_" + std::string(id) + ".dat");

   std::ofstream fv_file(std::string(folderName) + '/' + "fv_" + std::string(id) + ".dat");
   std::ofstream fp_file(std::string(folderName) + '/' + "fp_" + std::string(id) + ".dat");
   std::ofstream rhs1_file(std::string(folderName) + '/' + "rhs1_" + std::string(id) + ".dat");
   std::ofstream rhs2_file(std::string(folderName) + '/' + "rhs2_" + std::string(id) + ".dat");
   std::ofstream rhs3_file(std::string(folderName) + '/' + "rhs3_" + std::string(id) + ".dat");
   std::ofstream dofs_file(std::string(folderName) + '/' + "dofs_" + std::string(id) + ".dat");

   std::ofstream v_file(std::string(folderName) + '/' + "v_" + std::string(id) + ".dat");
   std::ofstream vk_file(std::string(folderName) + '/' + "vk_" + std::string(id) + ".dat");
   std::ofstream p_file(std::string(folderName) + '/' + "p_" + std::string(id) + ".dat");
   std::ofstream pk_file(std::string(folderName) + '/' + "pk_" + std::string(id) + ".dat");
   std::ofstream z_file(std::string(folderName) + '/' + "z_" + std::string(id) + ".dat");

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

   S->PrintMatlab(S_file);
   B->PrintMatlab(B_file);
   G->PrintMatlab(G_file);

   Ke->PrintMatlab(Ke_file);

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
   fp->Print(fp_file,1);
   rhs1->Print(rhs1_file,1);
   rhs2->Print(rhs2_file,1);
   rhs3->Print(rhs3_file,1);

   v->Print(v_file,1);
   vk->Print(vk_file,1);
   p->Print(p_file,1);
   pk->Print(pk_file,1);
   z->Print(z_file,1);


   for (int i = 0; i < vel_ess_tdof.Size(); ++i)
   {
      dofs_file << vel_ess_tdof[i] << std::endl;
   }
   dofs_file.close();

}

#endif

/// Destructor
SNavierPicardCGSolver::~SNavierPicardCGSolver()
{
   delete vfes; vfes = nullptr;
   delete vfec; vfec = nullptr;
   delete pfes; pfes = nullptr;
   delete pfec; pfec = nullptr;

   delete K_form; 
   delete B_form;
   delete C_form; 
   delete f_form;

   delete K;     K = nullptr;
   delete B;     B = nullptr;
   delete G;    G = nullptr;
   //delete A;     A = nullptr;
   //delete C;     C = nullptr;
   delete S;     S = nullptr;
   delete Ke;    Ke = nullptr;
   delete Be;    Be = nullptr;
   //delete Ae;    Ae = nullptr;

   delete fv;    fv = nullptr;
   delete fp;    fp = nullptr;
   delete rhs1;  rhs1 = nullptr;
   delete rhs2;  rhs2 = nullptr;
   delete rhs3;  rhs3 = nullptr;
   delete tmp;   tmp = nullptr;

   delete vk_vc; vk_vc = nullptr;
   delete pk_c;  pk_c = nullptr;

   delete fcoeff;  fcoeff = nullptr;
   delete traction; traction = nullptr;

   delete invA;     invA = nullptr;
   delete invK;     invK = nullptr;
   delete invS;     invS = nullptr;

   delete invA_pc;  invA_pc = nullptr;
   delete invK_pc;  invK_pc = nullptr;
   delete invS_pc;  invS_pc = nullptr;
   delete invS_pc_ortho;  invS_pc_ortho = nullptr;

   delete paraview_dc; paraview_dc = nullptr;
   delete visit_dc;    visit_dc = nullptr;

   delete mass_lf; mass_lf = nullptr;

}

}
