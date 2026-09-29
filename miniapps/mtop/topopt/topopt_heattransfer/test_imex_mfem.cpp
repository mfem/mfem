#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>

using namespace std;
using namespace mfem;



// Initial condition for advection-diffusion heat transfer. 
real_t T0_func(const Vector &x)
{
    return 0.0;
} 

void velocity_func(const Vector &x, Vector &v)
{
    int dim = x.Size();
//     v(0) = std::sin(M_PI * x(1)) * std::cos(M_PI * x(0));
//    v(1) = -std::cos(M_PI * x(1)) * std::sin(M_PI * x(0));
    v(0) = 0.0;
    v(1) = 0.0;
}

void f_func(const Vector &x, Vector &v)
{
    int dim = x.Size();
    v(0) = 16.0*std::sin(M_PI * x(1)) * std::cos(M_PI * x(0)) - 16.0;
   v(1) = -16.0*std::cos(M_PI * x(1)) * std::sin(M_PI * x(0));
}

real_t Target_func(const Vector &x)
{
    if (x(0) < 1.0 && x(0) > 0.0 && x(1) > 0.0 && x(1) < 4.0)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

// Raw Volume Flux for injection
real_t inflow_flux_func(const Vector &x, real_t t)      
{
    real_t x_center1 = 4.0;
    real_t x_center2 = 8.5;
    real_t x_center3 = 6.5;
    real_t x_center4 = 15.0;
    real_t x_center5 = 12.0;
    real_t x_center6 = 10.0;


    real_t y_center1 = 1.5;
    real_t y_center2 = 3.5;
    real_t y_center3 = 2.0;
    real_t y_center4 = 0.5;
    real_t y_center5 = 1.5;
    real_t y_center6 = 3.0;
    real_t rad = 0.3;

    if (((x(0) - x_center1)*(x(0)-x_center1) + (x(1) - y_center1)*(x(1) - y_center1)) < rad*rad)
    {
        return 100.0;
    }
    else if (((x(0) - x_center2)*(x(0)-x_center2) + (x(1) - y_center2)*(x(1) - y_center2)) < rad*rad)
    {
        return 100.0;
    }
    else if (((x(0) - x_center3)*(x(0)-x_center3) + (x(1) - y_center3)*(x(1) - y_center3)) < rad*rad)
    {
        return 100.0;
    }
    else if (((x(0) - x_center4)*(x(0)-x_center4) + (x(1) - y_center4)*(x(1) - y_center4)) < rad*rad)
    {
        return 100.0;
    }
        else if (((x(0) - x_center5)*(x(0)-x_center5) + (x(1) - y_center5)*(x(1) - y_center5)) < rad*rad)
    {
        return 100.0;
    }
    else if (((x(0) - x_center6)*(x(0)-x_center6) + (x(1) - y_center6)*(x(1) - y_center6)) < rad*rad)
    {
        return 100.0;
    }
    else
    { 
        return 0.0;
    }
}  

// Initial design density for the optimizer
real_t init_design_func(const Vector &x)    
{    
    real_t x_center1 = 4.0;
    real_t x_center2 = 8.5;
    real_t x_center3 = 6.5;
    real_t x_center4 = 15.0;
    real_t x_center5 = 12.0;
    real_t x_center6 = 10.0;


    real_t y_center1 = 1.5;
    real_t y_center2 = 3.5;
    real_t y_center3 = 2.0;
    real_t y_center4 = 0.5;
    real_t y_center5 = 1.5;
    real_t y_center6 = 3.0;
    real_t rad = 0.05;

    if (((x(0) - x_center1)*(x(0)-x_center1) + (x(1) - y_center1)*(x(1) - y_center1)) < rad*rad)
    {
        return 1.0;
    }
    else if (((x(0) - x_center2)*(x(0)-x_center2) + (x(1) - y_center2)*(x(1) - y_center2)) < rad*rad)
    {
        return 1.0;
    }
    else if (((x(0) - x_center3)*(x(0)-x_center3) + (x(1) - y_center3)*(x(1) - y_center3)) < rad*rad)
    {
        return 1.0;
    }
    else if (((x(0) - x_center4)*(x(0)-x_center4) + (x(1) - y_center4)*(x(1) - y_center4)) < rad*rad)
    {
        return 1.0;
    }
        else if (((x(0) - x_center5)*(x(0)-x_center5) + (x(1) - y_center5)*(x(1) - y_center5)) < rad*rad)
    {
        return 1.0;
    }
    else if (((x(0) - x_center6)*(x(0)-x_center6) + (x(1) - y_center6)*(x(1) - y_center6)) < rad*rad)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}


// Function which initializes the design. If not using Gaussian, x_max and y_max are irrelevant. Boolean to 
// tell code that the design has in fact been initialized without error.
bool InitializeDesign(ParGridFunction &rho, real_t x_max=0.0, real_t y_max=0.0)       
{
   // GaussianDesignCoefficient gaussian(x_max/2.0, y_max/2.0,  
   //                                       0.25*x_max, 0.25*y_max,    
   //                                       0.10, 1.0); 
   FunctionCoefficient init_design_cf(init_design_func);  
   rho.ProjectCoefficient(init_design_cf);   
   return true;
}

int main(int argc, char *argv[]) 
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();  
    int num_procs = Mpi::WorldSize();   
    MPI_Comm comm = MPI_COMM_WORLD;     
    int myid = Mpi::WorldRank();                  
    Hypre::Init();   

    int ser_ref_levels = 0;
    int par_ref_levels = 1;

    const char *mesh_file = "mfem2.mesh";   
    int order_fluid_heat = 2;
  
    real_t kf = 0.05; // thermal conductivity of fluid
    
    int ode_solver_type = 5; 
    real_t t_final = 2.0;           
    real_t dt = 0.001;                 
    int vis_steps = 100; 
    bool pv_vis = true; 
    bool density_pv = true;
    int maxit = 100;
    const char *device_config = "cpu"; 

    OptionsParser args(argc, argv); 
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use."); 
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", 
                        "Number of times to refine the mesh uniformly in serial,"  
                        " -1 for auto.");   
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",  
                        "Number of times to refine the mesh uniformly in parallel.");       
    args.AddOption(&order_fluid_heat, "-ofh", "--order_fluid_heat",
                        "Finite element order for fluid heat (polynomial degree) >= 0."); 
    args.AddOption(&kf, "-kf", "--kf",
                        "Thermal Conductivity of the Fluid.");                            
    args.AddOption(&pv_vis, "-vis", "--visualization", "-no-vis",    
                    "--no-visualization", 
                    "Enable or disable Paraview Visualization");
    args.AddOption(&density_pv, "-dvis", "--density-visualization", "-no-dvis",    
                    "--no-density-visualization", 
                    "Enable or disable Paraview Visualization for density");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                        ODESolver::IMEXTypes.c_str()); 
    args.AddOption(&t_final, "-tf", "--t-final",   
                        "Final time; start time is 0.");   
    args.AddOption(&dt, "-dt", "--time-step",
                        "Time step."); 
    args.AddOption(&vis_steps, "-vs", "--visualization-steps", 
                    "Visualize every n-th timestep.");   
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
    // Also the solid and fluid advection diffusion region fes
    FiniteElementCollection *fluid_heat_fec = new DG_FECollection(order_fluid_heat, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace *fluid_heat_fes = new ParFiniteElementSpace(pmesh, fluid_heat_fec);   

    H1_FECollection filter_fec(2, dim);  
    L2_FECollection control_fec(2, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace filter_fes(pmesh, &filter_fec);  
    ParFiniteElementSpace control_fes(pmesh, &control_fec); 
    
    // 7. Initialize rho and the filter
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
    filter_opts.filter_radius = 0.02; 
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);     
    filter.Assemble();   
    filter.Mult(rho, rho_tilde);     
    rho_tilde.ExchangeFaceNbrData();
    GridFunctionCoefficient rho_cf(&rho);

        //Solve stokes problem to get the velocity field
    int order = 2;
    const int pressure_order = order - 1;
    H1_FECollection vfec(order, dim, BasisType::GaussLobatto);
    H1_FECollection pfec(pressure_order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace vspace(pmesh, &vfec, dim, Ordering::byNODES);
    ParFiniteElementSpace pspace(pmesh, &pfec);
    BrinkmanStokesSolver brinkman_stokes_solver(vspace, pspace);
    BrinkmanCoefficient brink_cf(&rho_tilde, 0.5, 1e-12);
    ParGridFunction brinkman_gf(&filter_fes);
    brinkman_gf.ProjectCoefficient(brink_cf);
    // const int n = control_fes.GetTrueVSize();      
    // Vector rho_tv(n);
    // rho.GetTrueDofs(rho_tv);
    brinkman_stokes_solver.SetSolverType(StokesSolver::KrylovSolver::MINRES);
    brinkman_stokes_solver.SetVelocityPreconditionerType(StokesSolver::VelocityPreconditioner::AMG);
    brinkman_stokes_solver.SetPressurePreconditionerType(StokesSolver::PressurePreconditioner::CAHOUET_CHABARD);
    brinkman_stokes_solver.SetCCDiffusionSolverType(StokesSolver::CCDiffusionSolver::GMRES);
    brinkman_stokes_solver.SetLSCVelocityOperatorType(StokesSolver::LSCVelocityOperator::ASSEMBLED);
    brinkman_stokes_solver.SetLSCDiagonalOperatorType(StokesSolver::LSCDiagonalOperator::MATCH_VELOCITY);
    brinkman_stokes_solver.SetLSCQPreconditionerType(StokesSolver::LSCQPreconditioner::OPERATOR_JACOBI);
    brinkman_stokes_solver.SetRelTol(1e-6);
    brinkman_stokes_solver.SetAbsTol(1e-8);
    brinkman_stokes_solver.SetMaxIter(20000);
    brinkman_stokes_solver.SetVelocityAMGElasticityNearNullspace(false);
    brinkman_stokes_solver.SetVelocityPreconditionerCGRelTol(1.0e-5); 
    brinkman_stokes_solver.SetVelocityPreconditionerCGAbsTol(1e-10);
    brinkman_stokes_solver.SetVelocityPreconditionerCGMaxIter(2000);
    brinkman_stokes_solver.SetPressurePreconditionerCGRelTol(1e-8);
    brinkman_stokes_solver.SetPressurePreconditionerCGAbsTol(1e-12);
    brinkman_stokes_solver.SetPressurePreconditionerCGMaxIter(2000);
    brinkman_stokes_solver.SetKDim(50);
    brinkman_stokes_solver.SetPrintLevel(0);
    ConstantCoefficient viscosity_cf(1e-5);
    brinkman_stokes_solver.SetViscosity(viscosity_cf);
    brinkman_stokes_solver.SetBrinkmanPenalization(brinkman_gf);
    int max_bdr_attr = GlobalMax(comm, vspace.GetParMesh()->bdr_attributes.Size() ? vspace.GetParMesh()->bdr_attributes.Max() : 0);

    int max_attr = GlobalMax(comm, vspace.GetParMesh()->attributes.Size() ? vspace.GetParMesh()->attributes.Max() : 0);
    VectorFunctionCoefficient inlet_cf(dim, velocity_func);
    VectorFunctionCoefficient accel_cf(dim, f_func);
    for (int attr = 1; attr <= max_bdr_attr; attr++)
    {
        brinkman_stokes_solver.VelocityBoundary().Add(attr, inlet_cf);
    }

    for (int attr = 1; attr <= max_attr; attr++)
    {
        brinkman_stokes_solver.VelocityBoundary().Add(attr, accel_cf);
    }


//     real_t out_pressure = 0.0;

//    for (int attr = 1; attr <= max_bdr_attr; attr++)
//    {
//         brinkman_stokes_solver.AddPressureBoundaryID(attr);
//         brinkman_stokes_solver.PressureBoundary().Add(attr, out_pressure);
//    }

   BlockVector x;
   x.Update(brinkman_stokes_solver.GetBlockOffsets());
   x = 0.0;
   brinkman_stokes_solver.Solve(x);
   ParGridFunction u_gf(&vspace);
   u_gf.SetFromTrueDofs(x.GetBlock(0));
   u_gf.ExchangeFaceNbrData();

   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock << "parallel " << num_procs << " " << myid << "\n";
      u_sock.precision(8);
      u_sock << "solution\n" << *pmesh << u_gf << "window_title 'Velocity'"
             << endl;
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
   }

   VectorGridFunctionCoefficient u_cf(&u_gf);

    
// 9. Define the Coefficients 
    FunctionCoefficient inflow(inflow_flux_func);   
    FunctionCoefficient q0_f(T0_func);   

    ParGridFunction q0_f_gf(fluid_heat_fes); 
    q0_f_gf.ProjectCoefficient(q0_f); 
    GridFunctionCoefficient q0_f_cf;  
    q0_f_cf.SetGridFunction(&q0_f_gf);  


    
    // 11. Construct the Objective Function 
   RectangularIndicator indicator(0.0, 16.0, 0.0, 4.0); 
   ParGridFunction target_gf(fluid_heat_fes);
   FunctionCoefficient target_cf(Target_func); 
   target_gf.ProjectCoefficient(target_cf);    
   TerminalTargetObjective obj_func(fluid_heat_fes, indicator, target_gf, comm);           
   int n_steps = (int)ceil(t_final / dt);   

    // Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
    ConstantCoefficient one(1.0);
    ParLinearForm vol_form(&control_fes);
    vol_form.AddDomainIntegrator(new DomainLFIntegrator(one));
    vol_form.Assemble();
    std::unique_ptr<HypreParVector> vol_w(vol_form.ParallelAssemble());
    real_t domain_volume;
    real_t loc = vol_w->Sum();
    MPI_Allreduce(&loc, &domain_volume, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM, MPI_COMM_WORLD);
    const real_t Vstar = 1.0 * domain_volume;
    const int num_constraints = 1; // volume constraint

    // 12. Set up vectorized design field. Initialize the gradient.
    const int control_fes_size = control_fes.GetTrueVSize();
    Vector rho_tv(control_fes_size);
    Vector rho_old(control_fes_size);
    rho.GetTrueDofs(rho_tv);
    Vector dJ_drho(rho_tv.Size());
    //ParGridFunction phys_density(&filter_fes);
    ParaViewDataCollection paraview_dc("density", pmesh); 
    if (density_pv) {
        paraview_dc.SetPrefixPath("ParaView"); 
        paraview_dc.SetLevelsOfDetail(order_fluid_heat);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &rho);
        paraview_dc.RegisterField("rho_filter", &rho_tilde); 
        paraview_dc.SetCycle(0);
        paraview_dc.SetTime(0.0);
        paraview_dc.Save();
    }

   //12. Operator setup
   real_t dtkf = dt*kf;

    std::unique_ptr<MixedMultiPhysicsOperator> oper = std::make_unique<AdvectionDiffusionMixedMultiPhysicsOperator>(
        *fluid_heat_fes,
        q0_f_cf,
        &rho_tilde,
        u_cf,
        kf,
        dtkf,
        dt,
        t_final,
        inflow,
        comm);

    DesignSolver design_solver(*fluid_heat_fes,                 
    filter_fes,  
    control_fes, 
    oper,
    filter, 
    obj_func,
    q0_f_cf, 
    n_steps,   
    dt, 
    t_final,  
    rho, rho_tilde, ode_solver_type,  
    vis_steps, comm, pv_vis);

    // 14. Set up the MMA optimizer
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, control_fes_size, num_constraints, rho_tv);
    Vector tx_min(control_fes_size), tx_max(control_fes_size);
    Vector dvol(control_fes_size);                     // volume constraint gradient is constant:  vol_w/Vstar 
    dvol = *vol_w;  dvol /= Vstar;
    Vector dfidx[num_constraints];  dfidx[0] = dvol; 
    Vector fival(num_constraints);
    real_t initial_vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
    if (myid == 0){std::cout<<"Initial volume = " << initial_vol <<std::endl;}


        // 15. Optimization loop.
    real_t iterationError = 1.0;
    real_t tol = 1e-4;
    for (int k = 0; k < maxit && iterationError > tol; k++)
    {

        design_solver.FilterFSolve(rho_tv);              // forward filter:  rho -> rho_tilde
        const real_t J0 = design_solver.PhysicsFSolve(); // forward physics: -> J
        design_solver.PhysicsASolve();                      // adjoint physics: -> dJ/drho_tilde 
        design_solver.FilterASolve(dJ_drho); 

        // MMA update
        rho.GetTrueDofs(rho_tv);
        rho_old = rho_tv;
        // box constraints:  rho ∈ [0,1],  α_i ∈ [alpha_min, alpha_max]  (move limits)
        real_t move = 10.0;
        for (int i = 0; i < control_fes_size; i++)
        {
            tx_min[i] = std::max(real_t(0.0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(1.0), rho_tv[i] + move);
        }

        // volume constraint
        // Vector rho_v(rho_tv.Size());
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar - 0.0007; 


        mma.Update(rho_tv, dJ_drho, J0, fival, dfidx, tx_min, tx_max);
        rho.SetFromTrueDofs(rho_tv);

        // measure iteration error
        ParGridFunction rho_old_gf(&control_fes);
        rho_old_gf.SetFromTrueDofs(rho_old);
        // Vector iterationErr_vec(control_fes_size);
        // iterationErr_vec = rho_tv;
        // iterationErr_vec -= rho_old;
        iterationError = rho_old_gf.ComputeL2Error(rho_cf);
        real_t gradnorm = sqrt(InnerProduct(comm,dJ_drho, dJ_drho)); 
        // iterationError = iterationErr_vec.Norml2();

        if (myid == 0)
        {
            mfem::out << "it " << setw(3) << k + 1
                    << "   J = " << scientific << setprecision(6) << J0
                    << "   iterErr = " << setprecision(4) << iterationError
                    << "   Volume = "  << setprecision(4) << vol
                    << "   dJ/drho = " << setprecision(4) << gradnorm << endl;
        }

        // physical density r(rho~) for both GLVis and the ParaView archive
        // phys_density.ProjectCoefficient(simp_cf);


        if (density_pv)
        {
            paraview_dc.SetCycle(k + 1);
            paraview_dc.SetTime(k + 1);
            paraview_dc.Save();
        }
    }
 
 
    // Free the used memory.  
    delete pmesh;
    delete fluid_heat_fec;
    delete fluid_heat_fes;

    
    return 0; 
}