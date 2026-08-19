#include "mfem.hpp"
#include "TopOptDesignSolvers.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>


using namespace std;
using namespace mfem;

// Mesh bounding box
Vector bb_min, bb_max;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   real_t bb_min = 0.0;
   real_t bb_max = 1.0;
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min + bb_max) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max - bb_min);
   }
   // Clockwise twisting rotation in 2D around the origin
   const real_t w = M_PI/2;
   real_t d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
   d = d*d;
   switch (dim)
   {
      case 1: v(0) = 1.0; break;
      case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
      case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
   }
}


// Initial condition
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   real_t bb_min = 0.0;
   real_t bb_max = 1.0;
   for (int i = 0; i < dim; i++)
   {
      real_t center = (1 + 0) * 0.5;
      X(i) = 2 * (x(i) - center) / (1);
   }
   const real_t f = M_PI;
   return sin(f*X(0))*sin(f*X(1));
}

real_t inflow_function(const Vector &x)      
{
   return 1.0;  
}  
 
real_t simple_init_design(const Vector &x)    
{    
   return 1.0;  
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
   int order = 2; 
   bool pv_vis = true; 
   int ode_solver_type = 4; // 1 - Forward Backward Euler  
   real_t t_final = 0.1;           
   real_t dt = 0.01;              
   real_t diffusion_term = 0.1;   
   int problem_type = 2; 
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
 
   // 8. Boundary Conditions    
   Array<int> ess_tdof_list;                              
   Array<int> ess_bdr(4);   
   ess_bdr = 0;   
   pmesh->MarkExternalBoundaries(ess_bdr);  
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);  
   Array<int> inflow_bdr(4); 
   inflow_bdr = 0;
   //inflow_bdr[1] = 1;    
    
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
   FunctionCoefficient q0(u0_function); 
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
      ess_bdr,  
      inflow_bdr, 
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
  // Free the used memory.  
   // delete pd;
   delete fes;  
   delete pmesh;
   delete fec; 
  
   return 0; 
}
