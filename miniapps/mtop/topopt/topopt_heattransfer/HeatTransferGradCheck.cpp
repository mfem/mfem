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
   v(0) = 0.0;  
   v(1) = 1.0;   
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
   return 1.0;  
}  
 
real_t simple_init_design(const Vector &x)    
{    
   return 0.5;  
   // return 0.5 + 0.4 * std::sin(M_PI*x(0)) * std::cos(M_PI*x(1));  
   // if (x(0) > 0.1 && x(0) < 0.3 && x(1) > 0.4 && x(1) < 0.6)  
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
   const MPI_Comm comm = MPI_COMM_WORLD;     
   int myid = Mpi::WorldRank();                  
   Hypre::Init();  

   // 2. Parse command-line options.
   const char *mesh_file = "../../../../data/inline-quad.mesh";      
   int ser_ref_levels = 1;
   int par_ref_levels = 1;    
   int order = 1; 
   bool pv_vis = false; 
   int ode_solver_type = 4; // 1 - Forward Backward Euler  
   real_t t_final = 0.1;           
   real_t dt = 0.01;              
   real_t diffusion_term = 0.1;   
   int problem_type = 2  ; 
   int vis_steps = 10;                   
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
   FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);                                                                
   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize(); 
 
   H1_FECollection filter_fec(order, dim); 
   L2_FECollection control_fec(order-1, dim, BasisType::GaussLobatto);
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
 
    
   // 9. PDE Filter
   toopt::PDEFilterOptions filter_opts;
   filter_opts.print_level = 0; 
   // filter_opts.solver_rtol = 1e-12;
   filter_opts.filter_radius = 0.05; 
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);     
   filter.Assemble();   
   filter.Mult(rho, rho_tilde);    
   rho_tilde.ExchangeFaceNbrData();
 
  
   // 10. Define the Coefficients  
   SIMPCoefficient simp_stiff(&rho_tilde, 1e-6, 1.0, 3.0);  
   VectorFunctionCoefficient raw_velocity(dim, velocity_function); 
   ScalarVectorProductCoefficient velocity(simp_stiff, raw_velocity);  
   ConstantCoefficient cons_diff_coeff(diffusion_term);  
   ConstantCoefficient cons_dt_diff_coeff(dt*diffusion_term);    
   ProductCoefficient diff_coeff(cons_diff_coeff, simp_stiff);
   ProductCoefficient dt_diff_coeff(cons_dt_diff_coeff, simp_stiff); 
   FunctionCoefficient inflow(inflow_function);   
   FunctionCoefficient q0(q0_function); 
   real_t dt_diffusion_term = dt*diffusion_term; 
   
   // 11. Construct the Objective Function  
   RectangularIndicator indicator(0, 1, 0, 1); 
   ParGridFunction one_gf(fes);
   ConstantCoefficient one_cf(1.0);
   one_gf.ProjectCoefficient(one_cf);     
   TimeIntegratedL2TargetObjective obj_func(fes, indicator, one_gf, comm);           
   int n_steps = (int)ceil(t_final / dt);   
  
   const int n = control_fes.GetTrueVSize();      
   Vector rho_tv(n);
   rho.GetTrueDofs(rho_tv);
   double worst_best_fd_rel = 0.0; 

   Vector q0_vec(fes->GetTrueVSize()), h(control_fes.GetTrueVSize());
   Vector dJ_drho(rho_tv.Size()), rho_plus(rho_tv.Size()), rho_minus(rho_tv.Size()); 
   real_t eps = 1.0;  
   real_t tolerance = 1e-3; 
   int ntrials = 10;
   int nscales = 7;  
   for(int trial = 0; trial < ntrials; trial++)  
   { 
      int seed1 = 50 + 3*trial;  
      int seed2 = 51 + 3*trial; 
      // int seed3 = 102 + 3*trial;
      q0_vec.Randomize(seed1);
      h.Randomize(seed2);
      real_t h_norm = sqrt(InnerProduct(comm, h, h)); 
      h /= h_norm; 
      ParGridFunction q0_gf(fes); 
      q0_gf.SetFromTrueDofs(q0_vec); 
      GridFunctionCoefficient q0_cf; 
      q0_cf.SetGridFunction(&q0_gf);
      DesignSolver design_solver(*fes,               
         filter_fes,  
         control_fes, 
         filter, 
         obj_func,
         raw_velocity, 
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
      design_solver.PhysicsASolve();                      // adjoint physics: -> dJ/drho_tilde 
      design_solver.FilterASolve(dJ_drho);
      
      const real_t projected_grad = InnerProduct(comm, h, dJ_drho);     
      real_t gradnorm = sqrt(InnerProduct(comm, dJ_drho, dJ_drho));  

      if (Mpi::Root())
      {
         mfem::out << "\nDesign Taylor trial " << trial   
                   << ": J0=" << setprecision(16) << J0 
                   << ", <dJ/drho,p>=" << projected_grad   
                   << ", ||dJ/drho||="<< gradnorm << '\n';   
      }
 

      real_t scale = 1.0;  
      double previous_remainder = -1.0;  
      double trial_best_fd_rel = numeric_limits<double>::infinity();     
      bool trial_has_quadratic_drop = false;  
  
      for (int s = 0; s < nscales; s++) 
      {
         rho_plus = rho_tv; 
         rho_minus = rho_tv;  
         rho_plus.Add(scale, h); 
         rho_minus.Add(-scale, h);



         design_solver.FilterFSolve(rho_plus);                // forward filter:  rho -> rho_tilde
         const real_t Jp = design_solver.PhysicsFSolve();  // forward physics: -> J

         design_solver.FilterFSolve(rho_minus);              // forward filter:  rho -> rho_tilde
         const real_t Jm = design_solver.PhysicsFSolve(); // forward physics: -> J 
 
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
  
   return 0; 
}


static std::string fmtRate(double v)
{
    if (std::isnan(v)) return "  ---";
    std::ostringstream s; s << std::fixed << std::setprecision(3) << v;  
    return s.str();
}