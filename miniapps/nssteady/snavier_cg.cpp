#include "snavier_cg.hpp"

namespace mfem{

/// Constructor
SNavierPicardCGSolver::SNavierPicardCGSolver(ParMesh* mesh_,
                                             int vorder,
                                             int porder,
                                             double kin_vis_,
                                             bool verbose_)
{
   // mesh
   pmesh=mesh_;
   dim=pmesh->Dimension();

   // FE collection and spaces for velocity and pressure
   vfec=new H1_FECollection(vorder,dim);
   vfes=new ParFiniteElementSpace(pmesh,vfec,dim);
   pfec=new H1_FECollection(porder,dim);
   pfes=new ParFiniteElementSpace(pmesh,pfec,1);

   // determine spaces dimension
   int vdim = vfes->GetTrueVSize();
   int pdim = pfes->GetTrueVSize(); 
   
   // initialize GridFunctions
   v_gf.SetSpace(vfes);  v_gf=0.0;
   vk_gf.SetSpace(vfes); vk_gf=0.0;
   z_gf.SetSpace(vfes);  z_gf=0.0;
   p_gf.SetSpace(pfes);  p_gf=0.0;
   pk_gf.SetSpace(pfes); pk_gf=0.0;

   // initialize vectors
   v->SetSize(vdim);  *v=0.0;
   vk->SetSize(vdim); *vk=0.0;
   z->SetSize(vdim);  *z=0.0;
   p->SetSize(pdim);  *p=0.0;
   pk->SetSize(pdim); *pk=0.0;

   // setup GridFunctionCoefficients
   vk_vc->SetGridFunction(&vk_gf);
   pk_c->SetGridFunction(&pk_gf);

   // set kinematic viscosity
   kin_vis.constant = kin_vis_;

   // set verbosity level
   verbose=verbose_;

   // Error computation setup
   err_v = err_p = 0;
   norm_v = norm_p = 0;
   int order_quad = std::max(2, 2*vorder+1);
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

}

void SNavierPicardCGSolver::SetInitialCondition(VectorCoefficient &v_in)
{
   // Project coefficient onto velocity ParGridFunction
   v_gf.ProjectCoefficient(v_in);

   // Initialize provisional velocity and velocity at previous iteration
   v_gf.GetTrueDofs(*v);
   *z = *v;
   *vk = *v;
   z_gf.SetFromTrueDofs(*z);
   vk_gf.SetFromTrueDofs(*vk);
}


/// Methods 
void SNavierPicardCGSolver::Setup()
{
   /// 1. Setup and assemble bilinear forms 
   K_form = new ParBilinearForm(vfes);
   B_form = new ParMixedBilinearForm(vfes, pfes);
   C_form = new ParBilinearForm(vfes);

   K_form->AddDomainIntegrator(new VectorDiffusionIntegrator(kin_vis));
   K_form->Finalize();
   K = K_form->ParallelAssemble();

   B_form->AddDomainIntegrator(new VectorDivergenceIntegrator);
   B_form->Finalize();
   B = B_form->ParallelAssemble();

   C_form->AddDomainIntegrator(new VectorConvectionIntegrator(*vk_vc)); 
   
   
   /// 2. Setup and assemble linear form for rhs
   f_form = new ParLinearForm(vfes);
   // Adding forcing terms
   for (auto &accel_term : accel_terms)
   {
      f_form->AddDomainIntegrator( new VectorDomainLFIntegrator(*accel_term.coeff) );
   }
   // Adding traction bcs
   for (auto &traction_bc : traction_bcs)
   {
      f_form->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*traction_bc.coeff), traction_bc.attr);
   }
   
   f = f_form->ParallelAssemble(); 


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

   // Projection of coeff
   for (auto &vel_dbc : vel_dbcs)
   {
      v_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }

   VectorArrayCoefficient tmp_coeff(dim);
   for (auto &vel_dbc : vel_dbcs_xyz)
   {
      tmp_coeff.Set(vel_dbc.dir, vel_dbc.coeff, false);
      v_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }

   // Initialize solution vector with projected coefficients
   v_gf.GetTrueDofs(*v);

   /// 4. Apply transformation for essential bcs
   Ke = K->EliminateRowsCols(vel_ess_tdof);
   K->EliminateZeroRows();
   Bt  = B->Transpose(); 
   Be = B->EliminateCols(vel_ess_tdof);
   Bte  = Be->Transpose(); 

   K->EliminateBC(*Ke,vel_ess_tdof,*v,*f);
   // Alternatively
   //Ke.Mult(-1.0, *v, 1.0, *f);                  // rhs = f - Ke*x
   //for(int ii=0;ii<vel_ess_tdof.Size();ii++)    // rhs[tdof] = vd
   //{
   //   *f[vel_ess_tdof[ii]]=*v[vel_ess_tdof[ii]];
   //}

   // Update grid functions and vectors dependent on solution vector v
   UpdateSolution();      


   /// 5. Setup solvers and preconditioners
   //  5.1 Velocity prediction       A = K + alpha*C(uk)
   // solved with CGSolver preconditioned with HypreBoomerAMG (elasticity version)
   invA_pc = new HypreBoomerAMG();
   invA_pc->SetElasticityOptions(vfes);
   invA = new CGSolver(vfes->GetComm());
   invA->iterative_mode = false;
   invA->SetPrintLevel(s1Params.pl);
   invA->SetRelTol(s1Params.rtol);
   //invA->SetAbsTol(s1Params.atol);
   invA->SetMaxIter(s1Params.maxIter);

   // 5.2 Pressure correction       S = B K^{-1} Bt
   // solved with CGSolver preconditioned with HypreBoomerAMG 
   // NOTE: test different approaches to deal with Schur Complement:
   // * now using Jacobi, but this may not be a good approximation when involving Brinkman Volume Penalization
   // * alternative may be to use Multigrid to get better approximation
   auto Kd = new HypreParVector(MPI_COMM_WORLD, K->GetGlobalNumRows(), K->GetRowStarts());
   K->GetDiag(*Kd);
   S = B->Transpose();    // S = Bt
   S->InvScaleRows(*Kd);  // S = Kd^{-1} Bt
   S = ParMult(B, S);     // S = B Kd^{-1} Bt

   invS_pc = new HypreBoomerAMG(*S);
   invS_pc->SetSystemsOptions(dim);
   invS = new CGSolver(vfes->GetComm());
   invS->iterative_mode = false;
   invS->SetOperator(*S);
   invS->SetPreconditioner(*invS_pc);
   invS->SetPrintLevel(s2Params.pl);
   invS->SetRelTol(s2Params.rtol);
   //invS->SetAbsTol(s2Params.atol);
   invS->SetMaxIter(s2Params.maxIter);

   // 5.3 Velocity correction
   // solved with CGSolver preconditioned with HypreBoomerAMG 
   invK_pc = new HypreBoomerAMG(*K);
   invK_pc->SetSystemsOptions(dim);
   invK = new CGSolver(vfes->GetComm());
   invK->iterative_mode = false;
   invK->SetOperator(*K);
   invK->SetPreconditioner(*invK_pc);
   invK->SetPrintLevel(s3Params.pl);
   invK->SetRelTol(s3Params.rtol);
   //invK->SetAbsTol(s3Params.atol);
   invK->SetMaxIter(s3Params.maxIter);
}

void SNavierPicardCGSolver::SetFixedPointSolver(SolverParams params, double alpha_)
{
   sParams = params;    

   // set segregation parameter
   alpha = alpha_;
}

void SNavierPicardCGSolver::SetLinearSolvers( SolverParams params1,
                                            SolverParams params2,
                                            SolverParams params3)
{
   s1Params = params1;
   s2Params = params2;
   s3Params = params3;                          
}

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

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i] || vel_ess_attr_y[i] || vel_ess_attr_z[i]) && attr[i]) == 0,
                  "Trying to enforce traction bc on dirichlet boundary.");
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

void SNavierPicardCGSolver::FSolve()
{
   timer.Clear();
   timer.Start();

   for (iter = 0; iter < sParams.maxIter; iter++)
   {
      // Solve current iteration.
      Step();

      // Compute errors.
      ComputeError();

      // Update solution at previous iterate and gridfunction coefficients.
      UpdateSolution();

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


void SNavierPicardCGSolver::Step()
{
   /// Assemble convective term with new velocity vk and modify matrix for essential bcs.
   delete C_form;
   C_form = new ParBilinearForm(vfes);
   C_form->AddDomainIntegrator(new VectorConvectionIntegrator(*vk_vc, alpha)); 
   C = C_form->ParallelAssemble();
   Ce = C->EliminateRowsCols(vel_ess_tdof);
   //C->EliminateZeroRows(); // CHECK: We should leave it zeroed out otherwise we get 2 on diagonal of A

   /// Solve.
   // 1: Velocity prediction      ( K + alpha*C(vk) ) z = f - (1-alpha)*C(uk)*k
   A = Add(1,*K,1,*C);            
   invA->SetOperator(*A);     
   invA_pc->SetOperator(*A);

   C->Mult(*vk,*rhs1);    // C(vk)*vk 
   (*rhs1) *= (alpha-1);  // (alpha-1) C(vk)*vk 
   rhs1->Add(1,*f);       // fe + (alpha-1)*C(vk)*vk
   C->EliminateBC(*Ce,vel_ess_tdof,*v,*rhs1);   
   invA->Mult(*rhs1,*z);

   // 2: Pressure correction                    B*K^-1*B^T p = B*z   
   B->Mult(*z,*rhs2);
   invS->Mult(*rhs2, *p);

   // 3: Velocity correction         K u = K*z - B^T*p = f - (1-alpha)*C(uk)*uk - alpha C(uk) z - B^T*p  
   // NOTE: Could be more efficient storing and reusing rhs1, SparseMatrix -alpha C(uk)
   Bt->Mult(*p, *rhs3);     //  B^T p
   rhs3->Neg();             // -B^T p
   K->AddMult(*z,*rhs3,1);  //  K z - B^T p
   // B->MultTranspose(*p, *rhs3, -1.0);
   // K->AddMult(*z,*rhs3,1);
   K->EliminateBC(*Ke,vel_ess_tdof,*z,*rhs3);      
   invK->Mult(*rhs3,*v);

   /// Update GridFunctions for solution.
   v_gf.SetFromTrueDofs(*v);
   p_gf.SetFromTrueDofs(*p);
}

void SNavierPicardCGSolver::ComputeError()
{
   err_v  = v_gf.ComputeL2Error(*vk_vc);
   norm_v = ComputeGlobalLpNorm(2., *vk_vc, *pmesh, irs);
   err_p  = p_gf.ComputeL2Error(*pk_c);
   norm_p = ComputeGlobalLpNorm(2., *pk_c, *pmesh, irs);

   if (verbose)
   {
      out << "|| v - v_k || / || v_k || = " << err_v / norm_v << "\n";
      out << "|| p - p_k || / || p_k || = " << err_p / norm_p << "\n";
   }
}

void SNavierPicardCGSolver::UpdateSolution()
{
   *vk = *v;
   vk_gf.SetFromTrueDofs(*vk);

   *pk = *p;
   pk_gf.SetFromTrueDofs(*pk);
}


/// Destructor
SNavierPicardCGSolver::~SNavierPicardCGSolver()
{
   delete vfes;
   delete vfec;
   delete pfes;
   delete pfec;

   delete K_form;
   delete B_form;
   delete C_form; 
   delete f_form;

   delete K;
   delete B;
   delete Bt;
   delete C;
   delete A;
   delete S;
   delete Ke;
   delete Be;
   delete Bte;
   delete Ce;

   delete invA;     
   delete invK;     
   delete invS;     

   delete invA_pc;  
   delete invK_pc;  
   delete invS_pc;  
}

}
