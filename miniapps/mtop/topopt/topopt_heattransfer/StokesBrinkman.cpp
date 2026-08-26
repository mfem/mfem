#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>


using namespace std;
using namespace mfem;

real_t f_natural(const Vector & x);

real_t simple_init_design(const Vector &x)    
{    
   //return 0.5;  
//    return 0.5 + 0.4 * std::sin(M_PI*x(0)) * std::cos(M_PI*x(1));  
   if (x(0) > 0.4 && x(0) < 0.6 && x(1) > 0.4 && x(1) < 0.6)  
   {    
      return 0.0;   
   }
   else
   { 
      return 1.0;
   }  
}
  
bool InitializeDesign(ParGridFunction &rho, real_t x_max, real_t y_max)       
{
   // GaussianDesignCoefficient gaussian(x_max/2.0, y_max/2.0,  
   //                                       0.25*x_max, 0.25*y_max,    
   //                                       0.10, 1.0); 
   FunctionCoefficient one(simple_init_design);  
   rho.ProjectCoefficient(one);   
   return true;
}

real_t inflow_function(const Vector &x)      
{
   return 0.0;  
}  

// Initial condition
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
//    Vector X(dim);
//    real_t bb_min = 0.0;
//    real_t bb_max = 1.0;
//    for (int i = 0; i < dim; i++)
//    {
//       real_t center = (1 + 0) * 0.5;
//       X(i) = 2 * (x(i) - center) / (1);
//    }
//    const real_t f = M_PI;
//    return sin(f*X(0))*sin(f*X(1));
    real_t x_center1 = 2.0/6.0;
    real_t x_center2 = 2.0/6.0;
    real_t x_center3 = 2.0/6.0;
    real_t x_center4 = 4.0 / 6.0;
    real_t x_center5 = 4.0 / 6.0;
    real_t x_center6 = 4.0 / 6.0;
    real_t x_center7 = 6.0/6.0;
    real_t x_center8 = 6.0/6.0;


    real_t y_center1 = 5.0/6.0;
    real_t y_center2 = 0.5;
    real_t y_center3 = 1.0/6.0;
    real_t y_center4 = 5.0/6.0;
    real_t y_center5 = 0.5;
    real_t y_center6 = 1.0/6.0;
    real_t y_center7 = 5.0/6.0;
    real_t y_center8 = 1.0/6.0;

    real_t sigma_x = 0.1;
    real_t sigma_y = 0.1;


    // Injection 1
    // Distance from center (normalized by sigma)
    real_t dx1 = (x(0) - x_center1) / sigma_x;
    real_t dy1 = (x(1) - y_center1) / sigma_y;
    real_t r_squared1 = dx1 * dx1 + dy1 * dy1;
    real_t gaussian1 = std::exp(-0.5 * r_squared1);

    // Injection 2
    // Distance from center (normalized by sigma)
    real_t dx2 = (x(0) - x_center2) / sigma_x;
    real_t dy2 = (x(1) - y_center2) / sigma_y;
    real_t r_squared2 = dx2 * dx2 + dy2 * dy2;
    real_t gaussian2 = std::exp(-0.5 * r_squared2);

    // Injection 3
    // Distance from center (normalized by sigma)
    real_t dx3 = (x(0) - x_center3) / sigma_x;
    real_t dy3 = (x(1) - y_center3) / sigma_y;
    real_t r_squared3 = dx3 * dx3 + dy3 * dy3;
    real_t gaussian3 = std::exp(-0.5 * r_squared3);

        // Injection 4
    // Distance from center (normalized by sigma)
    real_t dx4 = (x(0) - x_center4) / sigma_x;
    real_t dy4 = (x(1) - y_center4) / sigma_y;
    real_t r_squared4 = dx4 * dx4 + dy4 * dy4;
    real_t gaussian4 = std::exp(-0.5 * r_squared4);

        // Injection 5
    // Distance from center (normalized by sigma)
    real_t dx5 = (x(0) - x_center5) / sigma_x;
    real_t dy5 = (x(1) - y_center5) / sigma_y;
    real_t r_squared5 = dx5 * dx5 + dy5 * dy5;
    real_t gaussian5 = std::exp(-0.5 * r_squared5);

        // Injection 6
    // Distance from center (normalized by sigma)
    real_t dx6 = (x(0) - x_center6) / sigma_x;
    real_t dy6 = (x(1) - y_center6) / sigma_y;
    real_t r_squared6 = dx6 * dx6 + dy6 * dy6;
    real_t gaussian6 = std::exp(-0.5 * r_squared6);

            // Injection 7
    // Distance from center (normalized by sigma)
    real_t dx7 = (x(0) - x_center7) / sigma_x;
    real_t dy7 = (x(1) - y_center7) / sigma_y;
    real_t r_squared7 = dx7 * dx7 + dy7 * dy7;
    real_t gaussian7 = std::exp(-0.5 * r_squared7); 

        // Injection 8
    // Distance from center (normalized by sigma)
    real_t dx8 = (x(0) - x_center8) / sigma_x;
    real_t dy8 = (x(1) - y_center8) / sigma_y;  
    real_t r_squared8 = dx8 * dx8 + dy8 * dy8;
    real_t gaussian8 = std::exp(-0.5 * r_squared8);
 
    return 3*(gaussian1 + gaussian2 + gaussian3 + gaussian4 + gaussian6 + gaussian7 + gaussian8);
}


int main(int argc, char *argv[]) 
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();  
    int num_procs = Mpi::WorldSize();   
    const MPI_Comm comm = MPI_COMM_WORLD;     
    int myid = Mpi::WorldRank();                  
    Hypre::Init();  

    const char *mesh_file = "../../../../data/inline-quad.mesh";   
    int ser_ref_levels = 1;
    int par_ref_levels = 1;    
    int order = 2; 
    real_t dynamic_viscosity = 1.0;
    bool visualization = true;
    real_t t_final = 0.1;           
    real_t dt = 0.001;              
    real_t diffusion_term = 0.01;   
    int problem_type = 2; 
    int vis_steps = 1; 

    bool pv_vis = true; 
    int ode_solver_type = 4; // 1 - Forward Backward Euler 
    const char *device_config = "cpu";  
    OptionsParser args(argc, argv); 
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use."); 
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", 
                        "Number of times to refine the mesh uniformly in serial," 
                        " -1 for auto.");   
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",  
                        "Number of times to refine the mesh uniformly in parallel.");       
    args.AddOption(&order, "-o", "--order",
                        "Finite element order (polynomial degree) >= 0.");       
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",  
                        "--no-visualization", 
                        "Enable or disable Visualization"); 
    args.AddOption(&dynamic_viscosity, "-dv", "--dynamic-viscosity",
                        "Dynamic Viscosity of the Fluid.");  
                           args.AddOption(&pv_vis, "-vis", "--visualization", "-no-vis",  
                    "--no-visualization", 
                    "Enable or disable Paraview Visualization");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                        ODESolver::IMEXTypes.c_str()); 
    args.AddOption(&t_final, "-tf", "--t-final",   
                        "Final time; start time is 0.");   
    args.AddOption(&dt, "-dt", "--time-step",
                        "Time step.");
    args.AddOption(&diffusion_term, "-dc", "--diffusion-coeff",
                        "Diffusion coefficient in the PDE.");  
    args.AddOption(&vis_steps, "-vs", "--visualization-steps", 
                    "Visualize every n-th timestep.");   
    args.AddOption(&problem_type, "-pt", "--problem_type",                                  
                    "Select which problem solve.");
    args.AddOption(&device_config, "-d", "--device",
                    "Device configuration string, see Device::Configure().");         
    args.Parse(); 

    if (!args.Good()) 
    {
        if (Mpi::Root())
        {
            args.PrintUsage(cout);    
        }
        return 1;       
    }
    if (Mpi::Root())      
    {
        args.PrintOptions(cout); 
    }

    // 3. Read the meshfile  
    Mesh *mesh = new Mesh(mesh_file); 
    const int dim = mesh->Dimension();

    Device device(device_config);
    if (myid == 0) { device.Print(); }

    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement, where 'ref_levels' is a
    //    command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++) { mesh->UniformRefinement(); } 
    if (mesh->NURBSext)   
    {  
        mesh->SetCurvature(max(order, 1)); 
    }


    // 5. Define the parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.           
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)  
    {
        pmesh->UniformRefinement();
    }

    // 6. FE Collections for pressure and velocity spaces, using taylor hood elements for now
    FiniteElementCollection *v_coll(new H1_FECollection(order, dim));
    FiniteElementCollection *p_coll(new H1_FECollection(order-1, dim));

    
    ParFiniteElementSpace *V_space = new ParFiniteElementSpace(pmesh, v_coll, 2);
    ParFiniteElementSpace *P_space = new ParFiniteElementSpace(pmesh, p_coll);

    HYPRE_BigInt dimV = V_space->GlobalTrueVSize();
    HYPRE_BigInt dimP = P_space->GlobalTrueVSize();

    if(Mpi::Root)
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(V) = " << dimV << "\n";
        std::cout << "dim(P) = " << dimP << "\n";
        std::cout << "dim(V+P) = " << dimV + dimP << "\n";
        std::cout << "***********************************************************\n";
    }

    H1_FECollection filter_fec(order, dim); 
    L2_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace filter_fes(pmesh, &filter_fec);  
    ParFiniteElementSpace control_fes(pmesh, &control_fec);  
    
    ParGridFunction rho(&control_fes);  
    ParGridFunction rho_tilde(&filter_fes); 
    if (!InitializeDesign(rho, 1.0, 1.0))  
    { 
        if (myid == 0)
        {
            cerr << "Error: unknown -init value. Use uniform, solid, void, or gaussian.\n";        
        }
        return 1;
    } 

    toopt::PDEFilterOptions filter_opts;
    filter_opts.print_level = 0; 
    // filter_opts.solver_rtol = 1e-12;
    filter_opts.filter_radius = 0.05; 
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);     
    filter.Assemble();   
    filter.Mult(rho, rho_tilde);    
    rho_tilde.ExchangeFaceNbrData();

    BrinkmanCoefficient brink_coeff(&rho_tilde, 0.5);  
    ProductCoefficient b_coeff(10000.0, brink_coeff);


    
    // 7. Define the two BlockStructure of the problem.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = V_space->GetVSize();
    block_offsets[2] = P_space->GetVSize();
    block_offsets.PartialSum();

    Array<int> block_trueOffsets(3); // number of variables + 1
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = V_space->TrueVSize();
    block_trueOffsets[2] = P_space->TrueVSize();
    block_trueOffsets.PartialSum();

    // 8. coeffs
    ConstantCoefficient visc_coeff(dynamic_viscosity);
    ConstantCoefficient zero(0.0);
    ConstantCoefficient one(1.0);
    Vector vec_one(V_space->GetVSize());
    vec_one = 1.0;
    VectorConstantCoefficient vone(vec_one);

    FunctionCoefficient fnatcoeff(f_natural);


    Vector vec_zero(V_space->GetVSize());
    vec_zero = 0.0;
    VectorConstantCoefficient vzero(vec_zero);

    // 9. Block operator
    ParBilinearForm *a(new ParBilinearForm(V_space)); 
    ParMixedBilinearForm *b(new ParMixedBilinearForm(V_space, P_space));

    HypreParMatrix *A = NULL;
    HypreParMatrix *B = NULL;

    a->AddDomainIntegrator(new VectorMassIntegrator(b_coeff));
    a->AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
    a->Assemble();
    a->Finalize();


    b->AddDomainIntegrator(new VectorDivergenceIntegrator);
    b->Assemble();
    b->Finalize();


    BlockOperator *BrinkStokesOp = new BlockOperator(block_trueOffsets);

    Array<int> empty_tdof_list;  // empty
    OperatorPtr opA, opB;


    TransposeOperator *Bt = NULL;

    A = a->ParallelAssemble();
    B = b->ParallelAssemble();
    (*B) *= -1;
    Bt = new TransposeOperator(B);
    BrinkStokesOp->SetBlock(0,0, A);
    BrinkStokesOp->SetBlock(0,1, Bt);
    BrinkStokesOp->SetBlock(1,0, B);

    // 10. RHS
    MemoryType mt = device.GetMemoryType();
    BlockVector x(block_offsets, mt), rhs(block_offsets, mt);
    BlockVector trueX(block_trueOffsets, mt), trueRhs(block_trueOffsets, mt);

    ParLinearForm *fform(new ParLinearForm);
    fform->Update(V_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorDomainLFIntegrator(vzero));
    fform->AddBoundaryIntegrator(new VectorBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();
    fform->SyncAliasMemory(rhs);
    fform->ParallelAssemble(trueRhs.GetBlock(0));
    trueRhs.GetBlock(0).SyncAliasMemory(trueRhs);

    ParLinearForm *gform(new ParLinearForm);
    gform->Update(P_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(zero));
    gform->Assemble();
    gform->SyncAliasMemory(rhs);
    gform->ParallelAssemble(trueRhs.GetBlock(1));
    trueRhs.GetBlock(1).SyncAliasMemory(trueRhs);

    // 11. Construct the operators for preconditioner
    // HypreParMatrix *AinvBt = NULL;
    // HypreParVector *Ad = NULL;
    // HypreParMatrix *S = NULL; 
    // Solver *invA, *invS;

    // Ad = new HypreParVector(MPI_COMM_WORLD, A->GetGlobalNumRows(),
    //                         A->GetRowStarts());
    // A->GetDiag(*Ad);
    // AinvBt = B->Transpose();
    // AinvBt->InvScaleRows(*Ad);
    // S = ParMult(B, AinvBt);
    // invA = new HypreDiagScale(*A);
    // invS = new HypreBoomerAMG(*S);

    // invA->iterative_mode = false;
    // invS->iterative_mode = false;

    // BlockDiagonalPreconditioner *StokesBrinkmanPr = new BlockDiagonalPreconditioner(
    //     block_trueOffsets);
    // StokesBrinkmanPr->SetDiagonalBlock(0, invA);
    // StokesBrinkmanPr->SetDiagonalBlock(1, invS);

    // 12. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.
    int maxIter(2000);
    real_t rtol(1.e-6);
    real_t atol(1.e-10);
    MINRESSolver solver(MPI_COMM_WORLD);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(*BrinkStokesOp); 
    //solver.SetPreconditioner(*StokesBrinkmanPr);
    solver.SetPrintLevel(1);
    trueX = 0.0;
    solver.Mult(trueRhs, trueX);
    if (device.IsEnabled()) { trueX.HostRead(); }

    if(Mpi::Root())
    {
        if (solver.GetConverged())
        {
            std::cout << "MINRES converged in " << solver.GetNumIterations() << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
        }
        else
        {
            std::cout << "MINRES did not converge in " << solver.GetNumIterations() << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
        }
    }

    ParGridFunction *u(new ParGridFunction);
    ParGridFunction *p(new ParGridFunction);
    u->MakeRef(V_space, x.GetBlock(0), 0);
    p->MakeRef(P_space, x.GetBlock(1), 0);
    u->Distribute(&(trueX.GetBlock(0)));
    p->Distribute(&(trueX.GetBlock(1)));


    
   // 19. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << *u << "window_title 'Velocity'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      socketstream p_sock(vishost, visport);
      p_sock << "parallel " << num_procs << " " << myid << "\n";
      p_sock.precision(8);
      p_sock << "solution\n" << *pmesh << *p << "window_title 'Pressure'"
             << endl;
   }

    FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);                                                                
    HYPRE_BigInt global_vSize = fes->GlobalTrueVSize(); 

    // 8. Boundary Conditions    
    Array<int> ess_tdof_list;                              
    Array<int> ess_bdr(4);   
    ess_bdr = 0;   
    pmesh->MarkExternalBoundaries(ess_bdr);  
    fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);  
    Array<int> inflow_bdr(4); 
    inflow_bdr = 0;
    //inflow_bdr[1] = 1;    
        
    
    
    // 10. Define the Coefficients  
    SIMPCoefficient simp_stiff(&rho_tilde, 1e-6, 1.0, 3.0);  
    ConstantCoefficient cons_diff_coeff(diffusion_term);  
    ConstantCoefficient cons_dt_diff_coeff(dt*diffusion_term);    
    ProductCoefficient diff_coeff(cons_diff_coeff, simp_stiff);
    ProductCoefficient dt_diff_coeff(cons_dt_diff_coeff, simp_stiff); 
    FunctionCoefficient inflow(inflow_function);   
    FunctionCoefficient q0(u0_function); 
    VectorGridFunctionCoefficient u_cf(u); 
    real_t dt_diffusion_term = dt*diffusion_term; 
    
    // 11. Construct the Objective Function 
    RectangularIndicator indicator(0, 1, 0, 1); 
    ParGridFunction one_gf(fes);
    ConstantCoefficient one_cf(1.0);
    one_gf.ProjectCoefficient(one_cf);     
    TerminalTargetObjective obj_func(fes, indicator, one_gf, comm);           
    int n_steps = (int)ceil(t_final / dt);   
    
    const int n = control_fes.GetTrueVSize();      
    Vector rho_tv(n);
    rho.GetTrueDofs(rho_tv);

    Vector dJ_drho(rho_tv.Size());
    ParGridFunction q0_gf(fes); 
    q0_gf.ProjectCoefficient(q0);
    GridFunctionCoefficient q0_cf; 
    q0_cf.SetGridFunction(&q0_gf);
    DesignSolver design_solver(*fes,               
        filter_fes,  
        control_fes, 
        filter, 
        obj_func,
        u_cf, 
        diffusion_term, 
        dt_diffusion_term,
        inflow, 
        q0_cf, 
        n_steps, 
        dt, 
        t_final, 
        rho, rho_tilde, 
        simp_stiff, ode_solver_type, 
        vis_steps, problem_type, comm); 
    design_solver.FilterFSolve(rho_tv);              // forward filter:  rho -> rho_tilde
    const real_t J0 = design_solver.PhysicsFSolve(); // forward physics: -> J

    ParaViewDataCollection paraview_dc("density", pmesh); 
    if (pv_vis) {
        paraview_dc.SetPrefixPath("ParaView"); 
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &rho);
        paraview_dc.RegisterField("rho_filter", &rho_tilde); 
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.Save();
    }



    // Free the used memory.  
    // delete pd;
    delete fform;
    delete gform;
    delete u;
    delete p;
    delete BrinkStokesOp;
    // delete StokesBrinkmanPr;
    // delete invA;
    // delete invS;
    // delete S;
    // delete Ad;
    // delete AinvBt;
    delete Bt;
    delete B;
    delete A;
    delete a;
    delete b;
    delete P_space;
    delete V_space;
    delete p_coll;
    delete v_coll;
    delete pmesh;
    delete fes;  
    //delete pmesh;
    delete fec; 
    
    return 0; 
}


real_t f_natural(const Vector & x)
{
   return -exp(x(0))*sin(x(1));
}