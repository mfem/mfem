#include "mfem.hpp"
#include "HeatTransferTopOpt.hpp"
#include "../../mma/MMA_MFEM.hpp"
#include "../../pde_filter.hpp"
#include "../../mtop_solvers.hpp"
#include <memory>

using namespace std;
using namespace mfem;

void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();
   v(0) = 1.0;
   v(1) = 0.0;
}

real_t q0_function(const Vector &x)
{
   int dim = x.Size();
   // real_t cx = 0.01; 
   // real_t cy1 = 0.8;
   // real_t cy2 = 0.5;
   // real_t cy3 = 0.1;
   // real_t rx = 0.1, ry = 0.1, w = 10.;
   // real_t e1 = std::erfc(w*(x(0)-cx-rx))*std::erfc(-w*(x(0)-cx+rx)) * std::erfc(w*(x(1)-cy1-ry))*std::erfc(-w*(x(1)-cy1+ry));
   // real_t e2 = std::erfc(w*(x(0)-cx-rx))*std::erfc(-w*(x(0)-cx+rx)) * std::erfc(w*(x(1)-cy2-ry))*std::erfc(-w*(x(1)-cy2+ry));
   // real_t e3 = std::erfc(w*(x(0)-cx-rx))*std::erfc(-w*(x(0)-cx+rx)) * std::erfc(w*(x(1)-cy3-ry))*std::erfc(-w*(x(1)-cy3+ry));
   // return e1;
   return 2.0;
}

real_t inflow_function(const Vector &x)
{
   return 1.0;
}

real_t simple_init_design(const Vector &x)
{
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
   int par_ref_levels = 2;
   int order = 2;
   bool pv_vis = true;
   int ode_solver_type = 61; // 61 - Forward Backward Euler
   // 62 - IMEXRK2(2,2,2)
   // 63 - IMEXRK2(2,3,2)
   // 64 - IMEXRK3(3,4,3)
   real_t t_final = 1.0;        
   real_t dt = 0.001;
   real_t diffusion_term = 0.01;
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
   FiniteElementCollection *fec = new DG_FECollection(order, dim, BasisType::GaussLobatto);;
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();

   H1_FECollection filter_fec(1, dim);
   L2_FECollection control_fec(1, dim, BasisType::GaussLobatto);
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
   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0; 
   pmesh->MarkExternalBoundaries(ess_bdr);
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   Array<int> inflow_bdr(pmesh->bdr_attributes.Max());
   inflow_bdr = 0;
   inflow_bdr[1] = 1;  

   // 9. PDE Filter
   toopt::PDEFilterOptions filter_opts;
   filter_opts.filter_radius = 0.02;
   toopt::PDEFilter filter(filter_fes, control_fes, filter_opts);
   filter.Assemble();  
   filter.Mult(rho, rho_tilde);   
 
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
 
   // 11. Construct the Objective Function
   RectangularIndicator indicator(0, 1, 0, 1); 
   TerminalL2Objective obj_func(fes, indicator, comm);       
   int n_steps = (int)ceil(t_final / dt);

   const int n = control_fes.GetTrueVSize();
   Vector rho_tv(n);
   rho.GetTrueDofs(rho_tv);
   Vector dJ_drho(n);

   // 12. Set up visualization for rho 
   ParaViewDataCollection paraview_dc("rho", pmesh);
   if (pv_vis)
   {
      paraview_dc.SetPrefixPath("ParaView");
      paraview_dc.SetLevelsOfDetail(2);
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.RegisterField("rho", &rho); 
      paraview_dc.RegisterField("rho_tilde", &rho_tilde);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0); 
      paraview_dc.Save(); 
   }

   // 13. Construct the Solver
   // unique_ptr<ODESolver> ode_solver = ODESolver::SelectIMEX(ode_solver_type);
   ParGridFunction q0_gf(fes);
   q0_gf.ProjectCoefficient(q0);
   GridFunctionCoefficient q0_cf(&q0_gf);

   DesignSolver design_solver( 
      *fes, filter_fes, control_fes, filter, ess_bdr, inflow_bdr, obj_func, velocity, raw_velocity, diffusion_term,
      diff_coeff, dt_diff_coeff, inflow, q0_cf, n_steps, dt, t_final, rho, rho_tilde, ode_solver_type, comm);

   // 14. Optimization iteratio for test
   design_solver.FilterFSolve(rho_tv);              // forward filter:  rho -> rho_tilde
   const real_t J = design_solver.PhysicsFSolve(); // forward physics: -> J
   design_solver.PhysicsASolve();                      // adjoint physics: -> dJ/drho_tilde
   design_solver.FilterASolve(dJ_drho);

   // // 11. Set up and compute the cost function at the terminal time
   // ConstantCoefficient zero(0.0);
   // RectangularIndicator indicator(0, 1, 0, 1);
   // TerminalL2Objective obj_func(fes, indicator, comm); 
   // obj_func.ComputeObjective(q_gf); 
   // real_t loss = obj_func.GetObjective();
   // std::cout << "Cost at terminal time is: " << loss << std::endl;
   // std::cout << "l2 norm at terminal time is " << q_gf.ComputeL2Error(zero) << std::endl;  
   // ParLinearForm grad_form(fes);
   // obj_func.ComputeObjectiveGradient(q_gf, grad_form);
   // std::unique_ptr<HypreParVector> grad_vec(grad_form.ParallelAssemble());
   // std::cout << "l2 norm of gradient at terminal time is " << grad_vec->Norml2() << std::endl;  
   // std::cout << "gradient size " << grad_vec->Size() << std::endl; 
   // *q_vec -= *grad_vec;
   std::cout << "cost  = " << J << std::endl;
   std::cout << "l2 norm of design gradient = " << dJ_drho.Norml2() << std::endl;  


   // Free the used memory.
   // delete pd;
   delete fes;  
   delete pmesh;
   delete fec; 
  
   return 0; 
}

