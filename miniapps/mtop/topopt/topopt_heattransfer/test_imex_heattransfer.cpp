#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>

using namespace std;
using namespace mfem;


// Pre-defined velocity flow. In future versions, will use a stokes solver for this.
// Note that this is the raw velocity field over the fluid region. Over solid regions, the velocity is zero.
void velocity_func(const Vector &x, Vector &v)
{
    int dim = x.Size();
    // v(0) = 1.0;
    // v(1) = 0.0;
    v(0) = 5.0*sin(M_PI*3.0*x(0))*cos(M_PI*3.0*x(1)) + 1.0;
    v(1) = -5.0*cos(M_PI*3.0*x(0))*sin(M_PI*3.0*x(1)); 
}

// Initial condition for advection-diffusion heat transfer. 
real_t T0_func(const Vector &x)
{
    // int dim = x.Size(); 
    // return sin(M_PI*x(0)) * cos(M_PI*x(1));  
        // return 0.5; 
    // if ((((x(1) > 0.3 && x(1) < 0.4) ||  (x(1) > 0.5 && x(1) < 0.6)) && (x(0) < 0.2 && x(0) > 0.1)) || ((x(1) > 0.7 && x(1) < 0.8) && (x(0) < 0.8 && x(0) > 0.9)))
    // {
    //     return 0.0;
    // }
    // else
    // {
    //     return 0.0;
    // }
    return 0.0;
} 

// Raw Volume Flux for injection
real_t inflow_flux_func(const Vector &x, real_t t)      
{
    real_t rad = 0.1;
    if (t < 5.0)
    {
        if ((fabs(x(0) - 2.0/6.0) < rad || fabs(x(0) - 4.0 / 6.0) < rad || fabs(x(0) - 6.0/6.0) < rad) && (fabs(x(1) - 5.0/6.0) < rad || fabs(x(1) - 0.5) < rad || fabs(x(1) - 1.0/6.0) < rad) && !(fabs(x(1) - 0.5) < rad && fabs(x(0) - 6.0/6.0) < rad) && !(fabs(x(1) - 0.5) < rad && fabs(x(0) - 4.0/6.0) < rad))
        {
            return 100.0;
        }
        else
        {
            return 0.0;
        }
    }
    else{
        return 0.0;
    }
}  

// Initial design density for the optimizer
real_t init_design_func(const Vector &x)    
{    
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

    real_t sigma_x = 0.01;
    real_t sigma_y = 0.01;


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

real_t target_func(const Vector &x)
{

    // real_t x_center1 = 5.0/6.0;
    // real_t y_center1 = 0.5;
    // if ((x(0) < (5.0 / 6.0) + 0.1 && x(0) > (5.0 / 6.0) - 0.1) && (x(1) < 0.55 && x(1) > 0.45))
    // {
    //     return 10.0;
    // }
    // else
    // {
    //     return 0.0;
    // }
    // real_t sigma_x = 0.05;
    // real_t sigma_y = 0.05;
    // real_t dx1 = (x(0) - x_center1) / sigma_x;
    // real_t dy1 = (x(1) - y_center1) / sigma_y;
    // real_t r_squared1 = dx1 * dx1 + dy1 * dy1;
    // real_t gaussian1 = std::exp(-0.5 * r_squared1);
    // return 10*gaussian1;
    return 10.0;
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
    const MPI_Comm comm = MPI_COMM_WORLD;     
    int myid = Mpi::WorldRank();          
    Hypre::Init();  

    // 2. Parse command-line options.
    const char *mesh_file = "../../../../data/inline-quad.mesh";       
    int ser_ref_levels = 1;
    int par_ref_levels = 1;    
    int order = 2; 
    bool pv_vis = true; 
    bool density_pv = true;
    int ode_solver_type = 1; // 1 - Only Forward-Backward Euler Available currently
    real_t t_final = 3.0;         
    real_t dt = 0.01; 
    real_t fl_diff_const = 0.1;     
    real_t s_diff_const = 1.0;
    real_t filter_rad = 0.06;
    real_t SIMP_exp = 3.0;
    int vis_steps = 10;  
    const char *device_config = "cpu";  
    int problem_type = 1;
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
    args.AddOption(&pv_vis, "-vis", "--visualization", "-no-vis",  
                        "--no-visualization", 
                        "Enable or disable Paraview Visualization");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                        ODESolver::IMEXTypes.c_str()); 
    args.AddOption(&t_final, "-tf", "--t-final", 
                        "Final time; start time is 0.");   
    args.AddOption(&dt, "-dt", "--time-step",
                        "Time step.");
    args.AddOption(&fl_diff_const, "-fdc", "--fl_diff_const",
                        "Diffusion coefficient in fluid domain.");  
    args.AddOption(&s_diff_const, "-sdc", "--s_diff_const",
                        "Diffusion coefficient in solid regions."); 
    args.AddOption(&filter_rad, "-fr", "--filter_rad",
                        "PDE filter radius."); 
    args.AddOption(&SIMP_exp, "-se", "--SIMP_exp",
                        "Exponent in SIMP interpolation."); 
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

    // 6. Define the discontinuous DG finite element space of the given
    //    polynomial order on the refined mesh.
    FiniteElementCollection *state_fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace *state_fes = new ParFiniteElementSpace(pmesh, state_fec);
    HYPRE_BigInt global_vSize = state_fes->GlobalTrueVSize(); 
    
    H1_FECollection filter_fec(order+2, dim); 
    L2_FECollection control_fec(order+2, dim, BasisType::GaussLobatto); 
    ParFiniteElementSpace filter_fes(pmesh, &filter_fec);  
    ParFiniteElementSpace control_fes(pmesh, &control_fec);  
    
    ParGridFunction rho(&control_fes);   
    ParGridFunction rho_tilde(&filter_fes); 

    // 7. Initialize the Design Variable, rho
    if (!InitializeDesign(rho, 1.0, 1.0))  
    { 
        if (myid == 0)
        {
            cerr << "Error: unknown -init value. Use uniform, solid, void, or gaussian.\n"; 
        }
        return 1;
    } 

    // 8. Boundary Conditions    
    // Essential Dofs
    Array<int> ess_tdof_list;  
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());  
    ess_bdr = 0;   
    // ess_bdr[0] = 1;
    pmesh->MarkExternalBoundaries(ess_bdr);  
    state_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);  

    // Inflow Dofs
    Array<int> inflow_bdr(pmesh->bdr_attributes.Max()); 
    inflow_bdr = 0;
    // inflow_bdr[2] = 1;   

    // 9. Construct the Objective Function
    FunctionCoefficient target_cf(target_func);
    ParGridFunction target_gf(state_fes);
    target_gf.ProjectCoefficient(target_cf); 
    RectangularIndicator indicator( (5.0 / 6.0) - 0.1,  (5.0 / 6.0) + 0.1, 0.45, 0.55); 
    TimeIntegratedL2TargetObjective obj_func(state_fes, indicator, target_gf, comm);           
    int n_steps = (int)ceil(t_final / dt);  

    // 10. PDE Filter
    toopt::PDEFilterOptions filter_opts; 
    filter_opts.print_level = 0;
    filter_opts.filter_radius = filter_rad; 
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);   
    filter.Assemble();   
    filter.Mult(rho, rho_tilde);   
    rho_tilde.ExchangeFaceNbrData();
    GridFunctionCoefficient rho_cf(&rho);

    // 11. Define the Coefficients  
    ConstantCoefficient one(1.0); 
    SIMPCoefficient simp_cf(&rho_tilde, 1e-6, 3.0, SIMP_exp);   
    //Velocity Field  
    VectorFunctionCoefficient raw_velocity_cf(dim, velocity_func);  
    ScalarVectorProductCoefficient velocity_cf(one, raw_velocity_cf);   
    //Fluid Diffusion Coefficient
    ConstantCoefficient fl_diff_cf(fl_diff_const);  
    ConstantCoefficient fl_dt_diff_cf(dt*fl_diff_const);
    ProductCoefficient fl_simp_diff_cf(fl_diff_cf, one); // DesignSolver expects a product coefficient
    ProductCoefficient fl_simp_dt_diff_cf(fl_dt_diff_cf, one);  
    // Inflow 
    FunctionCoefficient inflow_cf(inflow_flux_func);
    ProductCoefficient inflow_rho_cf(inflow_cf, simp_cf);  
    // Initial and Dirichlet Boundary  
    FunctionCoefficient T0_cf(T0_func);
    // FunctionCoefficient T_d_cf(T_d_func); 

    // Volume constraint data:  g(rho) = (1, rho)/Vstar - 1.
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
        paraview_dc.SetLevelsOfDetail(order);
        paraview_dc.SetDataFormat(VTKFormat::BINARY);
        paraview_dc.SetHighOrderOutput(true);
        paraview_dc.RegisterField("density", &rho);
        paraview_dc.RegisterField("rho_filter", &rho_tilde); 
    }

    // 13. Set up the solver
    ParGridFunction T0_gf(state_fes);  
    T0_gf.ProjectCoefficient(T0_cf); 
    GridFunctionCoefficient T0_gcf; 
    T0_gcf.SetGridFunction(&T0_gf);

    real_t dt_fl_diff_const = dt*fl_diff_const;
    DesignSolver design_solver(*state_fes, 
        filter_fes, 
        control_fes, 
        filter, 
        ess_bdr, 
        inflow_bdr, 
        obj_func, 
        raw_velocity_cf, 
        fl_diff_const, 
        dt_fl_diff_const, 
        inflow_cf, 
        T0_gcf,
        n_steps, dt, t_final, 
        rho, rho_tilde, 
        simp_cf, ode_solver_type,
        vis_steps, problem_type, 
        comm);    


    // 14. Set up the MMA optimizer
    mfem_mma::MMAOptimizerParallel mma(MPI_COMM_WORLD, control_fes_size, num_constraints, rho_tv);
    Vector tx_min(control_fes_size), tx_max(control_fes_size);
    Vector dvol(control_fes_size);                     // volume constraint gradient is constant:  vol_w/Vstar 
    dvol = *vol_w;  dvol /= Vstar;
    Vector dfidx[num_constraints];  dfidx[0] = dvol; 
    Vector fival(num_constraints);
    real_t initial_vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
    if (myid == 0){std::cout<<"Initial volume = " << initial_vol <<std::endl;}

    //    Visualize the velocity field
    FiniteElementCollection *vel_fec;
    ParFiniteElementSpace *vel_fespace;
    vel_fec = new H1_FECollection(order, 2);
    vel_fespace = new ParFiniteElementSpace(pmesh, vel_fec, 2);
    ParGridFunction v_gf(vel_fespace); 
    v_gf.ProjectCoefficient(raw_velocity_cf);
    {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *pmesh << v_gf << flush;
    }


    // 15. Optimization loop.
    real_t iterationError = 1.0;
    int max_it = 100;
    real_t tol = 1e-5;
    for (int k = 0; k < max_it && iterationError > tol; k++)
    {

        design_solver.FilterFSolve(rho_tv);              // forward filter:  rho -> rho_tilde
        const real_t J0 = design_solver.PhysicsFSolve(); // forward physics: -> J
        design_solver.PhysicsASolve();                      // adjoint physics: -> dJ/drho_tilde 
        design_solver.FilterASolve(dJ_drho); 

        // MMA update
        rho.GetTrueDofs(rho_tv);
        rho_old = rho_tv;
        // box constraints:  rho ∈ [0,1],  α_i ∈ [alpha_min, alpha_max]  (move limits)
        real_t move = 20.0;
        for (int i = 0; i < control_fes_size; i++)
        {
            tx_min[i] = std::max(real_t(0), rho_tv[i] - move);
            tx_max[i] = std::min(real_t(100), rho_tv[i] + move);
        }

        // volume constraint
        real_t vol = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / domain_volume;
        fival(0) = InnerProduct(MPI_COMM_WORLD, *vol_w, rho_tv) / Vstar -  1.0;


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



    return 0;
}