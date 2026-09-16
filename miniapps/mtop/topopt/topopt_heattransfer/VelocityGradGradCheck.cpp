#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>

using namespace std;
using namespace mfem;
 
static std::string fmtRate(double v);

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size(); 
   v(0) = sin(M_PI*x(0));  
   v(1) = cos(M_PI*x(1));    
}    

void inlet_vel_func(const Vector &x, Vector &v)  
{
   int dim = x.Size();    
   v(0) = 1.0;     
   v(1) = 0.0;       
}   
  
real_t q0_function(const Vector &x)   
{  
   int dim = x.Size(); 
   return sin(M_PI*x(0)) * cos(M_PI*x(1));      
   // return x(0)*x(0) + x(1)*x(1);    
   // return 2.0;       
} 

real_t inflow_function(const Vector &x)      
{
   return 0.0;   
}  
 
real_t simple_init_design(const Vector &x)    
{    
   return 0.5;  
   // return std::sin(M_PI*x(0)) * std::cos(M_PI*x(1));  
   // if (x(0) > 0.4 && x(0) < 0.6 && x(1) > 0.4 && x(1) < 0.6)  
   // {    
   //    return 0.0;   
   // }
   // else
   // { 
   //    return 1.0; 
   // }   
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
   
int main(int argc, char *argv[]) 
{
    // 1. Initialize MPI and HYPRE.
    Mpi::Init();  
    int num_procs = Mpi::WorldSize();   
    MPI_Comm comm = MPI_COMM_WORLD;             
    int myid = Mpi::WorldRank();                  
    Hypre::Init();  
  
    const char *mesh_file = "../../../../data/inline-quad.mesh";    
    int ser_ref_levels = 1;
    int par_ref_levels = 1;    
    int order = 2; 
    real_t dynamic_viscosity = 0.1;      
    bool visualization = true;  
    real_t t_final = 0.1;            
    real_t dt = 0.001;                                       
    real_t diffusion_term = 0.01;     
    int problem_type = 2;  
    int vis_steps = 1;   
    real_t b_term = 100.0; 
   
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
    args.AddOption(&b_term, "-b", "--brinkman-term",                                  
                    "Brinkman Scaling.");
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
    //    'ref_levels' of uniform refinement, where 'ref_levels'  is a 
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
    FiniteElementCollection *v_coll(new H1_FECollection(order, dim, BasisType::GaussLobatto));  
    FiniteElementCollection *p_coll(new H1_FECollection(order-1, dim, BasisType::GaussLobatto)); 

    
    ParFiniteElementSpace *V_space = new ParFiniteElementSpace(pmesh, v_coll, 2, Ordering::byNODES);
    ParFiniteElementSpace *P_space = new ParFiniteElementSpace(pmesh, p_coll);  

    HYPRE_BigInt dimV = V_space->GlobalTrueVSize();  
    HYPRE_BigInt dimP = P_space->GlobalTrueVSize(); 

    if(Mpi::Root())
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
    filter_opts.filter_radius = 0.01; 
    toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);     
    filter.Assemble();       
    filter.Mult(rho, rho_tilde);      
    rho_tilde.ExchangeFaceNbrData();  
 
    BrinkmanCoefficient b_coeff(&rho_tilde, 0.5, b_term);    
    // ProductCoefficient b_coeff(100000.0, brink_coeff); 

 
      
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
      //   noslip_bdr[2] = 1; 
      //   noslip_bdr[0] = 1;
      //   all_ess_bdr[2] = 1; 
      //   all_ess_bdr[0] = 1;
    }

 
 
   // 6. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);                                                                
   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize(); 
  
   // 10. Define the Coefficients   
   SIMPCoefficient simp_stiff(&rho_tilde, 1e-6, 1.0, 3.0);   
   // VectorFunctionCoefficient raw_velocity(dim, velocity_function); 
   // ScalarVectorProductCoefficient velocity(simp_stiff, raw_velocity);  
   ConstantCoefficient cons_diff_coeff(diffusion_term);  
   ConstantCoefficient cons_dt_diff_coeff(dt*diffusion_term);    
   ProductCoefficient diff_coeff(cons_diff_coeff, simp_stiff);
   ProductCoefficient dt_diff_coeff(cons_dt_diff_coeff, simp_stiff);  
   FunctionCoefficient inflow(inflow_function);   
   FunctionCoefficient q0(q0_function);   
   ConstantCoefficient visc_cf(dynamic_viscosity);  
   VectorFunctionCoefficient inlet_cf(dim, inlet_vel_func); 
   real_t dt_diffusion_term = dt*diffusion_term; 

   constexpr real_t alpha = -1.0;
   ParBilinearForm *K;
   ParBilinearForm *Kp;
   ParBilinearForm *Km;
   




   
   // 11. Construct the Objective Function  
   ParGridFunction one_gf(fes);
   ConstantCoefficient one_cf(1.0);
   one_gf.ProjectCoefficient(one_cf);     
   Vector one_vec(fes->GetTrueVSize());
   one_gf.GetTrueDofs(one_vec);         
   int n_steps = (int)ceil(t_final / dt);   
  
   const int n = control_fes.GetTrueVSize();      
   Vector rho_tv(n);
   rho.GetTrueDofs(rho_tv);
   double worst_best_fd_rel = 0.0;  

   Vector q0_vec(fes->GetTrueVSize()), l0_vec(fes->GetTrueVSize()), h(V_space->GetTrueVSize());
   Vector dJ_drho(V_space->GetTrueVSize()), v_plus(V_space->GetTrueVSize()), v_minus(V_space->GetTrueVSize()); 

   VectorFunctionCoefficient raw_velocity(dim, velocity_function); 
   ScalarVectorProductCoefficient velocity(simp_stiff, raw_velocity);  

   int seed3 = 24;
   int seed2 = 44;

   q0_vec.Randomize(seed3);
   l0_vec.Randomize(seed2);
   
   ParGridFunction q0_gf(fes); 
   //q0_gf.ProjectCoefficient(q0); 
   q0_gf.SetFromTrueDofs(q0_vec);

   ParGridFunction l0_gf(fes); 
   //q0_gf.ProjectCoefficient(q0); 
   l0_gf.SetFromTrueDofs(l0_vec);


   std::unique_ptr<HypreParMatrix> K_mat;
   std::unique_ptr<HypreParMatrix> K_matp;
   std::unique_ptr<HypreParMatrix> K_matm;

   Vector v_vec(V_space->GetTrueVSize());

   VectorGridFunctionCoefficient v_cf;

   Vector place_holder(fes->GetTrueVSize());

   real_t eps = 1.0;  
   real_t tolerance = 1e-3; 
   int ntrials = 5; 
   int nscales = 15;   
   for(int trial = 0; trial < ntrials; trial++)  
   { 
      int seed1 = 50 + 3*trial;    
      h.Randomize(seed1);  
      real_t h_norm = sqrt(InnerProduct(comm, h, h));   
      h /= h_norm; 

      ParGridFunction v_gf(V_space);
      v_gf.ProjectCoefficient(velocity);
      
      v_gf.GetTrueDofs(v_vec);
  
      K = new ParBilinearForm(fes);  
      K->AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
      K->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));                                   
      K->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha), inlet_bdr);
      K->Assemble();
      K->Finalize();

      K_mat.reset(K->ParallelAssemble());

      
      K_mat->Mult(q0_vec, place_holder); 
      const real_t J0 = l0_vec*place_holder;

      ParLinearForm adv_lf(V_space);
      adv_lf.AddDomainIntegrator(new StokesVelocityGradientLFIntegrator(rho_tilde, q0_gf, l0_gf, simp_stiff, raw_velocity));
      adv_lf.AddBdrFaceIntegrator(new StokesVelocityGradientLFIntegrator(rho_tilde, q0_gf, l0_gf, simp_stiff, raw_velocity), inlet_bdr);
      adv_lf.AddInteriorFaceIntegrator(new StokesVelocityGradientLFIntegrator(rho_tilde, q0_gf, l0_gf, simp_stiff, raw_velocity));
      adv_lf.Assemble();
      std::unique_ptr<HypreParVector> adv_vec(adv_lf.ParallelAssemble());
      
      const real_t projected_grad = InnerProduct(comm, h, *adv_vec);     
      real_t gradnorm = sqrt(InnerProduct(comm, *adv_vec, *adv_vec));    

      if (Mpi::Root()) 
      {
         mfem::out << "\nDesign Taylor trial " << trial    
                   << ": J0=" << setprecision(16) << J0  
                   << ", <dJ/drho,p>=" << projected_grad   
                   << ", ||dJ/drho||="<< gradnorm << '\n';   
      }
 

      real_t scale = 100000.0;  
      double previous_remainder = -1.0;  
      double trial_best_fd_rel = numeric_limits<double>::infinity();     
      bool trial_has_quadratic_drop = false;  
   
      for (int s = 0; s < nscales; s++)   
      {
         v_plus = v_vec; 
         v_minus = v_vec;  
         v_plus.Add(scale, h);     
         v_minus.Add(-scale, h);   

         v_gf.SetFromTrueDofs(v_plus);
         v_cf.SetGridFunction(&v_gf);
         ScalarVectorProductCoefficient velocity_pcf(simp_stiff, v_cf);  
         Kp = new ParBilinearForm(fes);
         Kp->AddDomainIntegrator(new ConvectionIntegrator(velocity_pcf, alpha));
         Kp->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_pcf, alpha));                                   
         Kp->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_pcf, alpha), inlet_bdr);
         Kp->Assemble();
         Kp->Finalize();
         K_matp.reset(Kp->ParallelAssemble());
         K_matp->Mult(q0_vec, place_holder);
         const real_t Jp = l0_vec*place_holder;

         v_gf.SetFromTrueDofs(v_minus);
         v_cf.SetGridFunction(&v_gf);
         ScalarVectorProductCoefficient velocity_mcf(simp_stiff, v_cf);  
         Km = new ParBilinearForm(fes); 
         Km->AddDomainIntegrator(new ConvectionIntegrator(velocity_mcf, alpha));
         Km->AddInteriorFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_mcf, alpha));                                   
         Km->AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity_mcf, alpha), inlet_bdr);
         Km->Assemble();
         Km->Finalize();
         K_matm.reset(Km->ParallelAssemble());       
         K_matm->Mult(q0_vec, place_holder);
         const real_t Jm = l0_vec*place_holder;    
 
         const real_t fd = (Jp - Jm) / (2.0 * scale);     
 
         const double derivative_scale = max(max(fabs(static_cast<double>(fd)), fabs(static_cast<double>(projected_grad))), 1e-30);
         const double fd_rel = fabs(static_cast<double>(fd - projected_grad))
                               / derivative_scale;   
         const double fd_abs = fabs(static_cast<double>(fd - projected_grad));      
         trial_best_fd_rel = min(trial_best_fd_rel, fd_rel);     

         const real_t first_order_remainder = 
            fabs(Jp - J0 - scale * projected_grad); 
         const double remainder_ratio = 
            (previous_remainder > 0.0) ?   
            previous_remainder / first_order_remainder : 0.0;       

         if (Mpi::Root())
         {
            mfem::out << "  scale=" << scientific << setprecision(3) << scale
                      << "  FD=" << setprecision(12) << fd
                      << "  Jp= " << Jp  
                      << "  Jm= " << Jm
                      << "  rel_err=" << fd_rel
                      << "  abs_err=" << fd_abs
                      << "  first_order_rem=" << first_order_remainder;
            if (previous_remainder > 0.0)
            {
               mfem::out << "  rem_ratio=" << remainder_ratio;
            }
            mfem::out << '\n';
         }  
         if (previous_remainder > 0.0 && remainder_ratio > 50.0)  
         {
            trial_has_quadratic_drop = true;
         }
         previous_remainder = first_order_remainder;   
         scale *= 0.1; 
      }
      worst_best_fd_rel = max(worst_best_fd_rel, trial_best_fd_rel);  
      if(Mpi::Root())
      {
         MFEM_VERIFY(trial_best_fd_rel < tolerance,
                  "Raw design Taylor check did not find an accurate scale."); 
         MFEM_VERIFY(trial_has_quadratic_drop,
                  "Raw design Taylor check did not show quadratic remainder decay.");  
      }
   }
 

 
   // Free the used memory.  
   // delete pd; 
   delete fes;   
   delete pmesh;
   delete fec; 
   delete P_space;
   delete V_space;    
   delete p_coll;
   delete v_coll; 
  
   return 0; 
}


static std::string fmtRate(double v)
{
    if (std::isnan(v)) return "  ---";
    std::ostringstream s; s << std::fixed << std::setprecision(3) << v;        
    return s.str();
}