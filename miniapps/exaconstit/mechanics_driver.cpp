//***********************************************************************
//
//   ExaConstit Proxy App for AM Constitutive Properties Applications
//   Author: Steven R. Wopschall
//           wopschall1@llnl.gov
//   Date: August 6, 2017
//
//   Description: The purpose of this proxy app is to determine bulk 
//                constitutive properties for 3D printed material. This 
//                is the first-pass implementation of this app. The app
//                is a quasi-static, implicit solid mechanics code built 
//                on the MFEM library.
//                
//                Currently, only Dirichlet boundary conditions and a 
//                Neo-Hookean hyperelastic material model are 
//                implemented. Neumann (traction) boundary conditions and 
//                a body force are not implemented. This code uses 
//                psuedo-time stepping and Dirichlet boundary conditions 
//                are prescribed as either fully fixed, or along a 
//                prescribed direction. The latter, nonzero Dirichlet 
//                boundary conditions, are hard coded at the moment, and 
//                applied in the negative Z-direction as a function of 
//                the timestep. Until further implementation, the 
//                simulation's "final time" and "time step" size should 
//                be fixed (see running the app comments below). 
//
//                To run this code:
//
//                srun -np mechanics_driver -m ../../data/cube-hex.mesh -tf 1.0 -dt 0.2
//
//                where "np" is number of processors, the mesh is a 
//                simple cube mesh containing 8 elements, "tf" is the 
//                final simulation time, and dt is the timestep. Keep 
//                the time step >= 0.2. This has to do with how the 
//                nonzero Dirichlet BCs are applied.
//
//                Remark:
//                In principle, the mesh may be refined automatically, but 
//                this has not been thoroughly tested and has some problems
//                with -rp or -rs specified as anything but what is shown 
//                in the command line arguments above.
//
//                The finite element order defaults to 1 (linear), but 
//                may be increased by passing the argument at the 
//                command line "-o #" where "#" is the interpolation 
//                order (e.g. -o 2 for quadratic elements).
//
//                The mesh configuration is output for each time step
//                in mesh files (e.g. mesh.000001_1), which is per 
//                timestep, per processor. Visualization is using 
//                glvis. Currently I have not experimented with parallel 
//                mesh generation, so testing and debugging using glvis 
//                has been done in serial. An example call to glvis is
//                as follows
//
//                glvs -m mesh.000001_1
//
//                Lastly, if modifications are made to this file, one 
//                must type "make" in the command line to compile the 
//                code. 
//
//   Future Implemenations Notes:
//                
//                -Visco-plasticity constitutive model
//                -enhanced user control of Dirichlet BCs
//                -debug ability to read different mesh formats
//
//***********************************************************************
#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

class NonlinearMechOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fe_space;

   ParNonlinearForm *Hform;
   mutable Operator *Jacobian;
   const Vector *x;

   /// Newton solver for the hyperelastic operator
   NewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian
   Solver *J_prec;
   /// Specific cardiac hyperelastic model
   HyperelasticModel *model;

public:
   NonlinearMechOperator(ParFiniteElementSpace &fes, Array<int> &ess_bdr, 
                         double rel_tol, double abs_tol, int iter);

   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Driver for the newton solver
   void Solve(Vector &x) const;

   virtual ~NonlinearMechOperator();
};

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

void ReferenceConfiguration(const Vector &x, Vector &y);
void InitialDeformation(const Vector &x, Vector &y);
void NonZeroBdrFunc(const Vector &x, double t, Vector &y);
void InitGridFunction(const Vector &x, Vector &y);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Parse command-line options.
   const char *mesh_file = "../../data/beam-hex.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 0;
   int order = 1;
   double t_final = 300.0;
   double dt = 3.0;
   bool visualization = true;
   int vis_steps = 1;
   bool cubit = false;
   double newton_rel_tol = 1.0e-12;
   double newton_abs_tol = 1.0e-12;
   int newton_iter = 500;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&cubit, "-mcub", "--cubit", "-no-mcub", "--no-cubit",
                  "Read in a cubit mesh.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&newton_rel_tol, "-rel", "--relative-tolerance",
                  "Relative tolerance for the Newton solve.");
   args.AddOption(&newton_abs_tol, "-abs", "--absolute-tolerance",
                  "Absolute tolerance for the Newton solve.");
   args.AddOption(&newton_iter, "-it", "--newton-iterations",
                  "Maximum iterations for the Newton solve.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Open the mesh
   Mesh *mesh;
   if (cubit) {
      //named_ifgzstream imesh(mesh_file);
      mesh = new Mesh(mesh_file, 1, 1);
   }
   if (!cubit) {
      ifstream imesh(mesh_file);
      if (!imesh)
      {
         if (myid == 0)
         {
            cerr << "\nCan not open mesh file: " << mesh_file << '\n' << endl;
         }
         MPI_Finalize();
         return 2;
      }
      mesh = new Mesh(imesh, 1, 1);
      imesh.close();
   }
   ParMesh *pmesh = NULL;
   
   for (int lev = 0; lev < ser_ref_levels; lev++)
      {
         mesh->UniformRefinement();
      }
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   for (int lev = 0; lev < par_ref_levels; lev++)
      {
         pmesh->UniformRefinement();
      }

   delete mesh;
   int dim = pmesh->Dimension();

   // Definie the finite element spaces for displacement
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fe_space(pmesh, &fe_coll, dim);

   HYPRE_Int glob_size = fe_space.GlobalTrueVSize();

   // Print the mesh statistics
   if (myid == 0)
   {
      std::cout << "***********************************************************\n";
      std::cout << "dim(u) = " << glob_size << "\n";
      std::cout << "***********************************************************\n";
   }

   // Define grid functions for the current configuration, incremental reference configuration,
   // final deformation, and global reference configuration
   ParGridFunction x_gf(&fe_space);
   ParGridFunction x_ref(&fe_space);
   ParGridFunction x_def(&fe_space);

   // Project the initial and reference configuration functions onto the appropriate grid functions
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   VectorFunctionCoefficient refconfig(dim, ReferenceConfiguration);
  
   // deform is a function that populates x_gf with an initial guess
   x_gf.ProjectCoefficient(deform);
   x_ref.ProjectCoefficient(refconfig);

   // Define grid function for the nonzero Dirichlet boundary conditions
   ParGridFunction x_non_zero_ess(&fe_space);

   // Define grid function for the current configuration grid function
   // WITH nonzero Dirichlet BCs
   ParGridFunction x_bar_gf(&fe_space);

   // define a time dependent vector valued function for nonzero Dirichlet 
   // boundary conditions and an initialization function for the nonzero
   // Dirichlet BC grid function
   VectorFunctionCoefficient non_zero_ess_func(dim, NonZeroBdrFunc);
   VectorFunctionCoefficient init_grid_func(dim, InitGridFunction);

   // initialize the nonzero Dirichlet BC grid function
   x_non_zero_ess.ProjectCoefficient(init_grid_func);
   x_bar_gf.ProjectCoefficient(init_grid_func);

   // define a boundary attribute array and initialize to 0
   Array<int> ess_bdr;
   ess_bdr.SetSize(fe_space.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   
   // set nonzero Dirichlet boundary attributes to 1
   ess_bdr[1] = 1;

   // reset ALL Dirichlet BCs, i.e. boundary attribute = 1
   ess_bdr = 0;
   ess_bdr[0] = 1;
   ess_bdr[1] = 1;

   // Initialize the nonlinear mechanics operator
   NonlinearMechOperator oper(fe_space, ess_bdr, 
                              newton_rel_tol, newton_abs_tol, newton_iter);


   // declare solution vector
   Vector x(fe_space.TrueVSize());

   // initialize the vectors
   for (int k=0; k<x.Size(); ++k) 
   {
      x[k] = 0.;
   }

   // initialize visualization if requested 
   socketstream vis_u, vis_p;
   if (visualization) {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_u.open(vishost, visport);
      vis_u.precision(8);
      visualize(vis_u, pmesh, &x_gf, &x_def, "Deformation", true);
      // Make sure all ranks have sent their 'u' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
   }

   // add time loop
   double t = 0.0;
   oper.SetTime(t); 

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {

      // compute time step (this calculation is pulled from ex10p.cpp)
      double dt_real = min(dt, t_final - t);

      // compute current time
      t = t + dt_real;

      // set the time for the nonzero Dirichlet BC function evaluation
      non_zero_ess_func.SetTime(t);

      //project nonzero Dirchlet BC function onto grid function. Reset
      //boundary attribute list to do so.
      ess_bdr = 0;
      ess_bdr[1] = 1;
      x_non_zero_ess.ProjectBdrCoefficient(non_zero_ess_func, ess_bdr);

      // reset boundary attribute array for homogeneous Dirichlet BCs
      // prior to solve
      ess_bdr[0] = 1;

      // add the current configuration grid function and the nonzero 
      // Dirichlet BC grid function into x_bar_gf and then populate 
      // solution vector, x, with entries
      add(x_non_zero_ess, x_gf, x_bar_gf);
      x_bar_gf.GetTrueDofs(x);
 
      // Solve the Newton system 
      oper.Solve(x);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {

         // distribute the solution vector to the current configuration 
         // grid function
         x_gf.Distribute(x);

         // Set the end-step deformation 
         subtract(x_gf, x_ref, x_def);

         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << endl;
         }

      }

      if (!last_step) 
      {
         // distribute the solution vector to the current configuration 
         // grid function 
         x_gf.Distribute(x);

         // Set the end-step deformation 
         subtract(x_gf, x_ref, x_def);

         // set the new reference configuration to the end step current 
         // configuration
         x_ref = x_gf; 

      }
         
      {

      // Save the displaced mesh, the final deformation

         GridFunction *nodes = &x_gf;
         int owns_nodes = 0;
         pmesh->SwapNodes(nodes, owns_nodes);

         ostringstream mesh_name, deformation_name;
         mesh_name << "mesh." << setfill('0') << setw(6) << myid << "_" << ti;
         deformation_name << "deformation." << setfill('0') << setw(6) << myid << "_" << ti;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);
       
         ofstream deformation_ofs(deformation_name.str().c_str());
         deformation_ofs.precision(8);
         x_def.Save(deformation_ofs);
      }

   }
      
   // Free the used memory.
   delete pmesh;

   MPI_Finalize();

   return 0;
}

NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             double rel_tol,
                                             double abs_tol,
                                             int iter)
   : TimeDependentOperator(fes.TrueVSize()), fe_space(fes),
     newton_solver(fes.GetComm())
{
   Vector * rhs;
   rhs = NULL;

   // Initialize the neo-Hookean model
   model = new NeoHookeanModel(0.25, 5.0);

   // Define the mixed nonlinear form 
   Hform = new ParNonlinearForm(&fes);

   // Add the passive stress integrator
   Hform->AddDomainIntegrator(new HyperelasticNLFIntegrator(model));

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, rhs);

   HypreSmoother *J_hypreSmoother = new HypreSmoother;
   J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
   J_hypreSmoother->SetPositiveDiagonal(true);
   J_prec = J_hypreSmoother;

   MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
   J_minres->SetRelTol(rel_tol);
   J_minres->SetAbsTol(0.0);
   J_minres->SetMaxIter(300);
   J_minres->SetPrintLevel(-1);
   J_minres->SetPreconditioner(*J_prec);
   J_solver = J_minres;

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1); 
   newton_solver.SetRelTol(rel_tol);
   newton_solver.SetAbsTol(abs_tol);
   newton_solver.SetMaxIter(iter);
}

// Solve the Newton system
void NonlinearMechOperator::Solve(Vector &x) const
{
   Vector zero;
   newton_solver.Mult(zero, x);

   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{

   // Apply the nonlinear form
   Hform->Mult(k, y);

}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);

   return *Jacobian;
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
}

// In line visualization
void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
   if (!out)
   {  
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
   out << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      out << "window_size 800 800\n";
      out << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         out << "view 0 0\n"; // view from top
         out << "keys jl\n";  // turn off perspective and light
      }
      out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << flush;
}

void ReferenceConfiguration(const Vector &x, Vector &y)
{
   // set the reference, stress
   // free, configuration
   y = x;
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set initial configuration to be the reference 
   // configuration
   y = x;
}

void NonZeroBdrFunc(const Vector &x, double t, Vector &y)
{
   // we don't have the final time of the simulation, so it 
   // seems like we have to make an assumption about that. If 
   // we assume that the final time for a quasi-static implicit
   // simulation is always 1, and dt is what varies, then we can 
   // construct a BC curve based on the ratio of current time to 
   // final time

   y = 0.;
   // specify the displacement BC increment
   y[2] = -.1 ;
}

void InitGridFunction(const Vector &x, Vector &y)
{
   y = 0.;
}
