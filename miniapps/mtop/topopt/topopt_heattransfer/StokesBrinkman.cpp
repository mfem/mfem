#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>


using namespace std;
using namespace mfem;


real_t simple_init_design(const Vector &x)    
{    
    real_t x_center1 = 0.6;
    real_t x_center2 = 1.2;
    real_t x_center3 = 1.8;
    real_t x_center4 = 2.4;


    real_t y_center1 = 0.2;
    real_t y_center2 = 0.4;
    real_t y_center3 = 0.6;
    real_t y_center4 = 0.8;
    real_t rad = 0.25;

    if (((x(0) - x_center1)*(x(0)-x_center1) + (x(1) - y_center1)*(x(1) - y_center1)) < rad*rad)
    {
        return 0.0;
    }
    else if (((x(0) - x_center2)*(x(0)-x_center2) + (x(1) - y_center2)*(x(1) - y_center2)) < rad*rad)
    {
        return 0.0;
    }
    else if (((x(0) - x_center3)*(x(0)-x_center3) + (x(1) - y_center3)*(x(1) - y_center3)) < rad*rad)
    {
        return 0.0;
    }
    else if (((x(0) - x_center4)*(x(0)-x_center4) + (x(1) - y_center4)*(x(1) - y_center4)) < rad*rad)
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }

    // real_t sigma_x = 0.1;
    // real_t sigma_y = 0.1;

    // // Injection 1
    // // Distance from center (normalized by sigma)
    // real_t dx1 = (x(0) - x_center1) / sigma_x;
    // real_t dy1 = (x(1) - y_center1) / sigma_y;
    // real_t r_squared1 = dx1 * dx1 + dy1 * dy1;
    // real_t gaussian1 = std::exp(-0.5 * r_squared1);


    // // Injection 3
    // // Distance from center (normalized by sigma)
    // real_t dx3 = (x(0) - x_center3) / sigma_x;
    // real_t dy3 = (x(1) - y_center3) / sigma_y;
    // real_t r_squared3 = dx3 * dx3 + dy3 * dy3;
    // real_t gaussian3 = std::exp(-0.5 * r_squared3);

    //     // Injection 4
    // // Distance from center (normalized by sigma)
    // real_t dx4 = (x(0) - x_center4) / sigma_x;
    // real_t dy4 = (x(1) - y_center4) / sigma_y;
    // real_t r_squared4 = dx4 * dx4 + dy4 * dy4;
    // real_t gaussian4 = std::exp(-0.5 * r_squared4);
 
    // return 1 - (gaussian1 + gaussian3 + gaussian4);
}

bool InitializeDesign(ParGridFunction &rho, real_t x_max, real_t y_max)       
{
   // GaussianDesignCoefficient gaussian(x_max/2.0, y_max/2.0,  
   //                                       0.25*x_max, 0.25*y_max,    
   //                                       0.10, 1.0); 
   FunctionCoefficient sid(simple_init_design);  
   rho.ProjectCoefficient(sid);   
   return true;
}

void inlet_vel_func(const Vector &x, Vector &v) 
{
   int dim = x.Size();
   v(0) = 2.0;  
   v(1) = 0.0;   
} 
  

real_t inflow_function(const Vector &x)      
{
   return 20.0 + 273.15;  
}  

// Initial condition
real_t q0_s_function(const Vector &x)
{
    return 60.0 + 273.15;
    // return 20.0 + 273.15;
}

real_t q0_f_function(const Vector &x)
{
    real_t x_center1 = 0.6;
    real_t x_center3 = 1.5;
    real_t x_center4 = 2.4;


    real_t y_center1 = 0.2;
    real_t y_center3 = 0.5;
    real_t y_center4 = 0.8;

    real_t sigma_x = 0.1;
    real_t sigma_y = 0.1;

    // Injection 1
    // Distance from center (normalized by sigma)
    real_t dx1 = (x(0) - x_center1) / sigma_x;
    real_t dy1 = (x(1) - y_center1) / sigma_y;
    real_t r_squared1 = dx1 * dx1 + dy1 * dy1;
    real_t gaussian1 = std::exp(-0.5 * r_squared1);


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


 
    //return 20.0*(gaussian1 + gaussian3 + gaussian4) + 30.0 + 273.15;
    return 50.0 + 273.15;
}


int main(int argc, char *argv[]) 
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();  
    int num_procs = Mpi::WorldSize();   
    MPI_Comm comm = MPI_COMM_WORLD;     
    int myid = Mpi::WorldRank();                  
    Hypre::Init();  

    const char *mesh_file = "rect-quad.mesh";   
    int ser_ref_levels = 0;
    int par_ref_levels = 2;    
    int order_solid_heat = 1; 
    int order_fluid_heat = 2;
    int order_stokes = 2;

    // real_t dynamic_viscosity = 1.94e-5; // 1.94e-5;     
    // real_t kf = 0.024; // thermal conductivity of fluid
    // real_t ks = 400.0; // thermal conductivity of solid;
    // real_t hs = 2e5; 
    // real_t hf = 50.0;
    // real_t Q_prod = 0.175; // heat production rate;
    // real_t cf = 1006.0; // fluid heat capacity
    // real_t rf = 1.204; // fluid density
    // real_t finn_height = 8.0;
    // real_t plate_thickness = 0.2;

    real_t dynamic_viscosity = 1.94e-5;     
    real_t kf = 0.024; // thermal conductivity of fluid
    real_t ks = 400.0; // thermal conductivity of solid;
    real_t hs = 0.2; 
    real_t hf = 5.0e-5;
    real_t Q_prod = 0.175; // heat production rate;
    real_t cf = 1006.0; // fluid heat capacity
    real_t rf = 1.204; // fluid density
    real_t finn_height = 8.0;
    real_t plate_thickness = 0.2;
    real_t pressure_drop = 0.1;
    

    int ode_solver_type = 1; 
    real_t t_final = 5.0;           
    real_t dt = 0.001;                 
    int vis_steps = 50; 
    real_t b_term = 1.0;
    bool pv_vis = false; 
    const char *device_config = "cpu"; 

    OptionsParser args(argc, argv); 
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use."); 
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial", 
                        "Number of times to refine the mesh uniformly in serial,"  
                        " -1 for auto.");   
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",  
                        "Number of times to refine the mesh uniformly in parallel.");       
    args.AddOption(&order_solid_heat, "-osh", "--order_solid_heat",
                        "Finite element order for solid heat (polynomial degree) >= 0."); 
    args.AddOption(&order_fluid_heat, "-ofh", "--order_fluid_heat",
                        "Finite element order for fluid heat (polynomial degree) >= 0."); 
    args.AddOption(&order_stokes, "-os", "--order_stokes",
                        "Finite element order for stokes velocity (polynomial degree) >= 0."); 

    args.AddOption(&dynamic_viscosity, "-dv", "--dynamic-viscosity",
                        "Dynamic Viscosity of the Fluid."); 
    args.AddOption(&kf, "-kf", "--kf",
                        "Thermal Conductivity of the Fluid.");                         
    args.AddOption(&ks, "-ks", "--ks",
                        "Thermal Conductivity of the Solid."); 
    args.AddOption(&hf, "-hf", "--hf",
                        "Transfer Param fluid.");                         
    args.AddOption(&hs, "-hs", "--hs",
                        "Transfer Param solid."); 
    args.AddOption(&Q_prod, "-q", "--Q_prod",
                        "Heat Production Rate.");                         
    args.AddOption(&cf, "-cf", "--cf",
                        "Fluid Heat Capacity."); 
    args.AddOption(&rf, "-rf", "--rf",
                        "Fluid Density.");  
    args.AddOption(&finn_height, "-fh", "--finn_height",
                        "Height of the Fins."); 
    args.AddOption(&plate_thickness, "-pt", "--plate_thickness",
                        "Thickness of base plate. ");    

    args.AddOption(&pv_vis, "-vis", "--visualization", "-no-vis",    
                    "--no-visualization", 
                    "Enable or disable Paraview Visualization");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                        ODESolver::IMEXTypes.c_str()); 
    args.AddOption(&t_final, "-tf", "--t-final",   
                        "Final time; start time is 0.");   
    args.AddOption(&dt, "-dt", "--time-step",
                        "Time step."); 
    args.AddOption(&vis_steps, "-vs", "--visualization-steps", 
                    "Visualize every n-th timestep.");   
    args.AddOption(&b_term, "-b", "--brinkman-term",                                  
                    "Brinkman Scaling.");
    args.AddOption(&pressure_drop, "-pr", "--pressure_drop",                                  
                    "Pressure Drop.");
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
    FiniteElementCollection *v_coll(new H1_FECollection(order_stokes, dim, BasisType::GaussLobatto));  
    FiniteElementCollection *p_coll(new H1_FECollection(order_stokes-1, dim, BasisType::GaussLobatto));
    ParFiniteElementSpace *V_space = new ParFiniteElementSpace(pmesh, v_coll, dim, Ordering::byNODES);
    ParFiniteElementSpace *P_space = new ParFiniteElementSpace(pmesh, p_coll);
    FiniteElementCollection *solid_heat_fec = new DG_FECollection(order_solid_heat, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace *solid_heat_fes = new ParFiniteElementSpace(pmesh, solid_heat_fec);   
    FiniteElementCollection *fluid_heat_fec = new DG_FECollection(order_fluid_heat, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace *fluid_heat_fes = new ParFiniteElementSpace(pmesh, fluid_heat_fec);   

    HYPRE_BigInt dimV = V_space->GlobalTrueVSize(); 
    HYPRE_BigInt dimP = P_space->GlobalTrueVSize();
    
    HYPRE_BigInt dimS = solid_heat_fes->GlobalTrueVSize(); 
    HYPRE_BigInt dimF = fluid_heat_fes->GlobalTrueVSize();

    if(Mpi::Root())
    {
        std::cout << "***********************************************************\n";
        std::cout << "dim(V) = " << dimV << "\n";
        std::cout << "dim(P) = " << dimP << "\n";
        std::cout << "dim(S) = " << dimS << "\n";
        std::cout << "dim(F) = " << dimF << "\n";
        std::cout << "dim(V+P+S+F) = " << dimV + dimP  + dimS + dimF << "\n";
        std::cout << "***********************************************************\n";  
    }

    H1_FECollection filter_fec(order_stokes, dim);  
    L2_FECollection control_fec(order_stokes-1, dim, BasisType::GaussLobatto);
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
    const int n = control_fes.GetTrueVSize();      
    Vector rho_tv(n);
    rho.GetTrueDofs(rho_tv);

    // 8. Boundary conditions
    int local_max_bdr = pmesh->bdr_attributes.Size() ? pmesh->bdr_attributes.Max() : 0;
    int max_bdr = 0;
    MPI_Allreduce(&local_max_bdr, &max_bdr, 1, MPI_INT, MPI_MAX, pmesh->GetComm());

    // Separate arrays for projection 
    Array<int> inlet_bdr(max_bdr);    inlet_bdr = 0;
    Array<int> noslip_bdr(max_bdr);   noslip_bdr = 0;

    // Combined array for matrix elimination
    Array<int> all_ess_bdr(max_bdr);  all_ess_bdr = 0;

    if (max_bdr >= 4)
    {
        // Set Left (Attr 4) as Inlet
        inlet_bdr[3] = 1;
        all_ess_bdr[3] = 1;
    }

    if (max_bdr >= 3)
    {
        // Set Top (Attr 3) and Bottom (Attr 1) as No-Slip
        noslip_bdr[2] = 1; 
        noslip_bdr[0] = 1;
        all_ess_bdr[2] = 1; 
        all_ess_bdr[0] = 1;
    }
    
    // 9. Define the Coefficients 
    BrinkmanCoefficient b_coeff(&rho_tilde, 1.0, b_term); 
    ConstantCoefficient visc_cf(dynamic_viscosity);     
    ConstantCoefficient inflow(30.0 + 273.15);   
    FunctionCoefficient q0_f(q0_f_function);   
    FunctionCoefficient q0_s(q0_s_function);   
    VectorFunctionCoefficient inlet_cf(dim, inlet_vel_func); 

    ParGridFunction q0_s_gf(solid_heat_fes); 
    q0_s_gf.ProjectCoefficient(q0_s); 
    GridFunctionCoefficient q0_s_cf;  
    q0_s_cf.SetGridFunction(&q0_s_gf); 

    ParGridFunction q0_f_gf(fluid_heat_fes); 
    q0_f_gf.ProjectCoefficient(q0_f); 
    GridFunctionCoefficient q0_f_cf;  
    q0_f_cf.SetGridFunction(&q0_f_gf);  

    
    // 11. Construct the Objective Function 
   RectangularIndicator indicator(0, 1, 0, 1); 
   ParGridFunction one_gf(solid_heat_fes);
   ConstantCoefficient one_cf(1.0);
   one_gf.ProjectCoefficient(one_cf);     
   TerminalL2Objective obj_func(solid_heat_fes, indicator, comm);           
   int n_steps = (int)ceil(t_final / dt);   

   //12. Solve the Forward Stuff
   real_t dtkf = dt*kf;
   real_t dtks = dt*ks;

    std::unique_ptr<MixedMultiPhysicsOperator> oper = std::make_unique<Pseudo3DStokesOperator>(
    *V_space, 
    *P_space, 
    *fluid_heat_fes,
    *solid_heat_fes,
    b_coeff,
    visc_cf,
    inlet_cf,
    inflow,
    q0_f_cf,
    q0_s_cf,
    &rho_tilde,
    inlet_bdr,  
    noslip_bdr, 
    all_ess_bdr, 
    kf, ks, dtkf, dtks,
    cf, rf,
    Q_prod,
    plate_thickness,
    finn_height,
    hs, hf,
    dt,
    t_final,
    pressure_drop, 
    comm);

    DesignSolver design_solver(*solid_heat_fes,                 
    filter_fes,  
    control_fes, 
    oper,
    filter, 
    obj_func,
    q0_s_cf, 
    n_steps,   
    dt, 
    t_final,  
    rho, rho_tilde, ode_solver_type,  
    vis_steps, comm, pv_vis);
    
    design_solver.FilterFSolve(rho_tv);              // forward filter:  rho -> rho_tilde
    const real_t J0 = design_solver.PhysicsFSolve();

 
 
    // Free the used memory.  

    delete P_space;
    delete V_space;   
    delete p_coll;
    delete v_coll; 
    delete pmesh;
    delete solid_heat_fec;
    delete solid_heat_fes;
    delete fluid_heat_fec;
    delete fluid_heat_fes;

    
    return 0; 
}
