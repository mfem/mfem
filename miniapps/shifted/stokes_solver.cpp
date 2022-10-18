#include "mfem.hpp"
#include "stokes_solver.hpp"
#include "p_divW.hpp"


namespace mfem {

  StokesSolver::StokesSolver(mfem::ParMesh* mesh_, int vorder, int porder, bool vis): pmesh(mesh_),  x(NULL), trueX(NULL), rhs(NULL), trueRhs(NULL), velocityOrder(vorder), pressureOrder(porder), visualization(vis), block_offsets(3), block_trueOffsets(3), fvelocity(NULL), fpressure(NULL) , volforce(NULL), stokesPr(NULL), ns(NULL), vfes(NULL), vfec(NULL), pfes(NULL), pfec(NULL), materials(new Viscosity(pmesh)), stokesOp(NULL), exactVelocity(NULL), exactPressure(NULL), ess_tdofv(pmesh->bdr_attributes.Max())
  {
    int dim=pmesh->Dimension();
    vfec=new H1_FECollection(vorder,dim);
    pfec=new H1_FECollection(porder,dim);
    
    vfes=new mfem::ParFiniteElementSpace(pmesh,vfec,dim);
    pfes=new mfem::ParFiniteElementSpace(pmesh,pfec);

    fvelocity = new ParGridFunction(vfes);
    fpressure = new ParGridFunction(pfes);

    *fvelocity=0.0;
    *fpressure=0.0;

    block_offsets[0] = 0;
    block_offsets[1] = vfes->GetVSize();
    block_offsets[2] = pfes->GetVSize();
    block_offsets.PartialSum();

    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = vfes->GetTrueVSize();
    block_trueOffsets[2] = pfes->GetTrueVSize();
    block_trueOffsets.PartialSum();

    x = new BlockVector(block_offsets);
    rhs = new BlockVector(block_offsets);
    trueX = new BlockVector(block_trueOffsets);
    trueRhs = new BlockVector(block_trueOffsets);

    *x = 0.0;
    *rhs = 0.0;
    *trueX = 0.0;
    *trueRhs = 0.0;
    
    stokesOp =  new BlockOperator(block_trueOffsets);
    ess_tdofv = 0; 

  }

  StokesSolver::~StokesSolver()
  {
    delete ns;
    delete stokesPr;

    delete vfes;
    delete vfec;
    delete pfes;
    delete pfec;

    delete x;
    delete trueX;
    delete rhs;
    delete trueRhs;
    
    delete materials;

    surf_loads.clear();
    velocity_BC.clear();
    
  }

  void StokesSolver::SetNewtonSolver(double rtol, double atol,int miter, int prt_level)
  {
    rel_tol=rtol;
    abs_tol=atol;
    max_iter=miter;
    print_level=prt_level;
  }

  void StokesSolver::AddVelocityBC(int id, mfem::VectorCoefficient &val)
  {
    velocity_BC[id] = &val;
  }

  void StokesSolver::SetVolForce(mfem::VectorCoefficient& fv)
  {
    volforce=&fv;
  }

  void StokesSolver::SetExactVelocitySolution(mfem::VectorCoefficient& fv)
  {
    exactVelocity=&fv;
  }

  void StokesSolver::SetExactPressureSolution(mfem::Coefficient& fv)
  {
    exactPressure=&fv;
  }

  void StokesSolver::FSolve()
  {
    // Set the BC
    int dim=pmesh->Dimension();
    ess_tdofv = 0;
    fvelocity->MakeRef(vfes, x->GetBlock(0), 0);
  
    for(auto it=velocity_BC.begin();it!=velocity_BC.end();it++)
      {
	ess_tdofv[it->first -1] = 1;
	mfem::Array<int> ess_bdr_tmp(pmesh->bdr_attributes.Max());
	ess_bdr_tmp = 0;
	ess_bdr_tmp[it->first-1] = 1;
	fvelocity->ProjectBdrCoefficient(*(it->second), ess_bdr_tmp);
      }
    fvelocity->ParallelAssemble();

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(vfes, rhs->GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorDomainLFIntegrator(*volforce));
    for(auto it=surf_loads.begin();it!=surf_loads.end();it++)
      {
	fform->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(*(it->second)),*(it->first));
      }
    fform->Assemble();
    fform->ParallelAssemble(trueRhs->GetBlock(0));
  
    ParLinearForm *gform(new ParLinearForm);
    gform->Update(pfes, rhs->GetBlock(1), 0);
    gform->Assemble();
    gform->ParallelAssemble(trueRhs->GetBlock(1));

    // 9. Assemble the finite element matrices for the Stokes operator
    //
    //                            D = [ M  B^T ]
    //                                [ B   0  ]
    //     where:
    //
    //     M = \int_\Omega 2*mu \epsilon(u_h) \cdot \epsilon(v_h) d\Omega   u_h, v_h \in V_h
    //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
    //    Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the linear elasticity integrator with piece-wise
    //    constants coefficient lambda and mu.
    Vector lambda(pmesh->attributes.Max());
    lambda = 0.0;
    PWConstCoefficient lambda_func(lambda);
  
    ParBilinearForm *mVarf(new ParBilinearForm(vfes));
    mVarf->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,*materials));
    mVarf->Assemble();
    mVarf->EliminateEssentialBC(ess_tdofv, x->GetBlock(0), rhs->GetBlock(0));
    mVarf->Finalize();
    mfem::HypreParMatrix *M;
    M = mVarf->ParallelAssemble();
   
    ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(vfes, pfes));
    bVarf->AddDomainIntegrator(new VectorDivergenceIntegrator());
    bVarf->Assemble();
    bVarf->EliminateTrialDofs(ess_tdofv, x->GetBlock(0), rhs->GetBlock(1));
    bVarf->Finalize();
    mfem::HypreParMatrix *B;
    B = bVarf->ParallelAssemble();
 
    ParMixedBilinearForm *btVarf(new ParMixedBilinearForm(pfes, vfes));
    btVarf->AddDomainIntegrator(new PDivWForceIntegrator());
    btVarf->Assemble();
    btVarf->EliminateTestDofs(ess_tdofv);
    btVarf->Finalize();
    mfem::HypreParMatrix *Bt;
    Bt = btVarf->ParallelAssemble();

    stokesOp->SetBlock(0,0, M);
    stokesOp->SetBlock(0,1, Bt);
    stokesOp->SetBlock(1,0, B);

    //allocate the solvers
    if(ns==nullptr)
      {
	ns=new mfem::GMRESSolver(pmesh->GetComm());
	stokesPr = new BlockDiagonalPreconditioner(block_trueOffsets);
      }

    // PRECONDITIONER
    ConstantCoefficient scale(1.0);
    ParBilinearForm *pMass = new ParBilinearForm(pfes);
    pMass->AddDomainIntegrator(new MassIntegrator(scale));
    pMass->Assemble();
    pMass->Finalize();
   
    ParBilinearForm *vMass(new ParBilinearForm(vfes));
    vMass->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,*materials));
    vMass->Assemble();
    vMass->Finalize();
   
    HypreParMatrix *velocity_Mass = NULL;
    HypreParMatrix *pressure_Mass = NULL;
    velocity_Mass = vMass->ParallelAssemble();
    pressure_Mass = pMass->ParallelAssemble();

    mfem::HypreBoomerAMG* Vamg = new HypreBoomerAMG(*velocity_Mass);
    Vamg->SetSystemsOptions(dim);
    Vamg->SetElasticityOptions(vfes);
    mfem::HypreBoomerAMG* Pamg = new HypreBoomerAMG(*pressure_Mass);
    stokesPr->SetDiagonalBlock(0,Vamg);
    stokesPr->SetDiagonalBlock(1,Pamg);
   
    ParGridFunction *x1(new ParGridFunction(vfes,x->GetBlock(0)));
    ParGridFunction *rhs1(new ParGridFunction(vfes,rhs->GetBlock(0)));
    x1->ParallelAssemble(trueX->GetBlock(0));
    rhs1->ParallelAssemble(trueRhs->GetBlock(0));
    ParGridFunction *x2(new ParGridFunction(pfes,x->GetBlock(1)));
    ParGridFunction *rhs2(new ParGridFunction(pfes,rhs->GetBlock(1)));
    x2->ParallelAssemble(trueX->GetBlock(1));
    rhs2->ParallelAssemble(trueRhs->GetBlock(1));

    //set the parameters
    ns->SetOperator(*stokesOp);
    ns->SetPrintLevel(print_level);
    ns->SetRelTol(rel_tol);
    ns->SetAbsTol(abs_tol);
    ns->SetPreconditioner(*stokesPr);   
    ns->SetMaxIter(max_iter);
    //solve the problem
    *trueX = 0.0;
    ns->Mult(*trueRhs, *trueX);

  }

  void StokesSolver::ComputeL2Errors(){

    fvelocity->MakeRef(vfes, x->GetBlock(0), 0);
    fpressure->MakeRef(pfes, x->GetBlock(1), 0);
    fvelocity->Distribute(&(trueX->GetBlock(0)));
    fpressure->Distribute(&(trueX->GetBlock(1)));
  
    // 12. Create the grid functions u and p. Compute the L2 error norms.
    //  p.MakeRef(pfes, x.GetBlock(1), 0);
    // 13. Compute the L2 error norms.
    int order_quad = max(2, 6*velocityOrder+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
      {
	irs[i] = &(IntRules.Get(i, order_quad));
      }

    double err_u  = fvelocity->ComputeL2Error(*exactVelocity,irs);
    double err_p  = fpressure->ComputeL2Error(*exactPressure, irs);
    MPI_Comm comm = pmesh->GetComm();

    int myid;
    MPI_Comm_rank(comm, &myid);

    if (myid == 0){
      std::cout << "|| u_h - u_ex || = " << err_u << "\n";
      std::cout << "|| p_h - p_ex || = " << err_p << "\n";
    }
  }

  void StokesSolver::VisualizeFields(){
    if (visualization){
      ParaViewDataCollection paraview_dc("Example5_pmesh3", pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(velocityOrder);
      paraview_dc.SetCycle(0);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetTime(0.0); // set the time
      paraview_dc.RegisterField("velocity",fvelocity);
      paraview_dc.RegisterField("pressure",fpressure);
      paraview_dc.Save();
    }
  }
  
}

