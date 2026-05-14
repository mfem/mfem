//                          MFEM-SAMRAI Example 16
//
// Compile with: make ex16
//
// Sample runs:  mpirun -np 1 ex16 -m ../data/inline-quad.mesh -i samrai_input.2d
//               mpirun -np 1 ex16 -m ../data/star.mesh -i samrai_input.3d
//               mpirun -np 4 ex16 -m ../data/inline-quad.mesh -i samrai_input.2d
//
// Description:  This example demonstrates basic interoperability between MFEM
//               and SAMRAI by running both solvers side-by-side in an
//               alternating time loop.
//
//               MFEM solves a time dependent nonlinear heat equation:
//               du/dt = C(u), with C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u
//
//               SAMRAI solves linear advection on structured AMR grid:
//               du/dt + div(a*u) = 0
//
//               Both advance in an alternating time loop with no data sharing.
//               This demonstrates that MFEM and SAMRAI can coexist and provides
//               a foundation for future coupling with data exchange.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

// SAMRAI includes
#include "SAMRAI/SAMRAI_config.h"
#include "LinAdv.h"
#include "SAMRAI/appu/VisItDataWriter.h"
#include "SAMRAI/mesh/BergerRigoutsos.h"
#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/algs/HyperbolicLevelIntegrator.h"
#include "SAMRAI/mesh/CascadePartitioner.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"
#include "SAMRAI/algs/TimeRefinementIntegrator.h"
#include "SAMRAI/tbox/SAMRAIManager.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/InputDatabase.h"
#include "SAMRAI/tbox/InputManager.h"
#include "SAMRAI/tbox/SAMRAI_MPI.h"

using namespace std;
using namespace mfem;
using namespace SAMRAI;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   BilinearForm *M;
   BilinearForm *K;

   SparseMatrix Mmat, Kmat;
   SparseMatrix *T; // T = M + dt K
   real_t current_dt;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   DSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + dt K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   real_t alpha, kappa;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(FiniteElementSpace &f, real_t alpha, real_t kappa,
                      const Vector &u);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

   ~ConductionOperator() override;
};

real_t InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   // Initialize MPI and SAMRAI
   tbox::SAMRAI_MPI::init(&argc, &argv);
   tbox::SAMRAIManager::initialize();
   tbox::SAMRAIManager::startup();
   const tbox::SAMRAI_MPI& mpi(tbox::SAMRAI_MPI::getSAMRAIWorld());

   // 1. Parse command-line options for MFEM
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int order = 2;

   int ode_solver_type = 23;  // SDIRK33Solver
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t alpha = 1.0e-2;
   real_t kappa = 0.5;

   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;
   bool solve_implicit_state = false;

   // SAMRAI options
   const char *samrai_input_file = "samrai_input.3d";
   int samrai_vis_dump_interval = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&solve_implicit_state, "-imp-state", "--implicit-state",
                  "-imp-slope", "--implicit-slope",
                  "Implicitly solve for stage state or slope.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&samrai_input_file, "-i", "--samrai-input",
                  "SAMRAI input file.");
   args.AddOption(&samrai_vis_dump_interval, "-svi", "--samrai-vis-interval",
                  "SAMRAI visualization dump interval.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      tbox::SAMRAIManager::shutdown();
      tbox::SAMRAIManager::finalize();
      tbox::SAMRAI_MPI::finalize();
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize MFEM: Read mesh
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   // 4. Refine the mesh
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the finite element space
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;

   GridFunction u_gf(&fespace);

   // 6. Set initial conditions for MFEM
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // 7. Initialize the conduction operator
   ConductionOperator oper(fespace, alpha, kappa, u);
   using ImplicitVariableType = ConductionOperator::ImplicitVariableType;
   ImplicitVariableType imp_var = solve_implicit_state ?
                                  ImplicitVariableType::STATE
                                  : ImplicitVariableType::SLOPE;
   oper.SetImplicitVariableType(imp_var);

   u_gf.SetFromTrueDofs(u);
   {
      ofstream omesh("ex16.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex16-init.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   VisItDataCollection visit_dc("Example16", mesh);
   visit_dc.RegisterField("temperature", &u_gf);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u_gf;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // Initialize MFEM time integrator
   ode_solver->Init(oper);
   real_t mfem_time = 0.0;

   // 8. Initialize SAMRAI
   cout << "\n=== Initializing SAMRAI ===" << endl;

   // Parse SAMRAI input file
   tbox::InputManager::getManager()->parseInputFile(samrai_input_file);
   std::shared_ptr<tbox::Database> input_db(
      tbox::InputManager::getManager()->getInputDatabase());

   std::shared_ptr<tbox::Database> main_db = input_db->getDatabase("Main");
   std::shared_ptr<tbox::Database> linadv_db = input_db->getDatabase("LinAdv");
   std::shared_ptr<tbox::Database> cart_geom_db =
      input_db->getDatabase("CartesianGeometry");
   std::shared_ptr<tbox::Database> hier_db =
      input_db->getDatabase("PatchHierarchy");
   std::shared_ptr<tbox::Database> berger_db =
      input_db->getDatabase("BergerRigoutsos");
   std::shared_ptr<tbox::Database> gridding_db =
      input_db->getDatabase("GriddingAlgorithm");
   std::shared_ptr<tbox::Database> std_tag_db =
      input_db->getDatabase("StandardTagAndInitialize");
   std::shared_ptr<tbox::Database> hyperbolic_db =
      input_db->getDatabase("HyperbolicLevelIntegrator");
   std::shared_ptr<tbox::Database> time_refine_db =
      input_db->getDatabase("TimeRefinementIntegrator");
   std::shared_ptr<tbox::Database> load_balancer_db =
      input_db->getDatabase("LoadBalancer");

   const tbox::Dimension samrai_dim(static_cast<unsigned short>(main_db->getInteger("dim")));
   const string samrai_base_name = main_db->getStringWithDefault("base_name", "unnamed");
   const string samrai_viz_dump_dirname =
      main_db->getStringWithDefault("viz_dump_dirname", samrai_base_name);

   // Create SAMRAI objects
   std::shared_ptr<geom::CartesianGridGeometry> grid_geometry(
      new geom::CartesianGridGeometry(
         samrai_dim,
         "CartesianGeometry",
         cart_geom_db));

   std::shared_ptr<hier::PatchHierarchy> patch_hierarchy(
      new hier::PatchHierarchy(
         "PatchHierarchy",
         grid_geometry,
         hier_db));

   std::shared_ptr<LinAdv> linadv_model(
      new LinAdv(
         "LinAdv",
         samrai_dim,
         linadv_db,
         grid_geometry));

   std::shared_ptr<algs::HyperbolicLevelIntegrator> hyp_level_integrator(
      new algs::HyperbolicLevelIntegrator(
         "HyperbolicLevelIntegrator",
         hyperbolic_db,
         linadv_model.get(),
         true));

   std::shared_ptr<mesh::StandardTagAndInitialize> error_detector(
      new mesh::StandardTagAndInitialize(
         "StandardTagAndInitialize",
         hyp_level_integrator.get(),
         std_tag_db));

   std::shared_ptr<mesh::BergerRigoutsos> box_generator(
      new mesh::BergerRigoutsos(
         samrai_dim,
         berger_db));

   std::shared_ptr<mesh::CascadePartitioner> load_balancer(
      new mesh::CascadePartitioner(
         samrai_dim,
         "CascadePartitioner",
         load_balancer_db));

   std::shared_ptr<mesh::GriddingAlgorithm> gridding_algorithm(
      new mesh::GriddingAlgorithm(
         patch_hierarchy,
         "GriddingAlgorithm",
         gridding_db,
         error_detector,
         box_generator,
         load_balancer));

   std::shared_ptr<algs::TimeRefinementIntegrator> time_integrator(
      new algs::TimeRefinementIntegrator(
         "TimeRefinementIntegrator",
         time_refine_db,
         patch_hierarchy,
         hyp_level_integrator,
         gridding_algorithm));

#ifdef HAVE_HDF5
   std::shared_ptr<appu::VisItDataWriter> visit_data_writer(
      new appu::VisItDataWriter(
         samrai_dim,
         "LinAdv VisIt Writer",
         samrai_viz_dump_dirname));
   linadv_model->registerVisItDataWriter(visit_data_writer);
#endif

   // Initialize SAMRAI hierarchy
   double samrai_loop_time = time_integrator->getIntegratorTime();
   double samrai_loop_time_end = time_integrator->getEndTime();
   double samrai_dt = 0.0;  // Will be computed after initialization

   time_integrator->initializeHierarchy();

   cout << "SAMRAI hierarchy initialized" << endl;
   cout << "SAMRAI start time: " << samrai_loop_time << endl;
   cout << "SAMRAI end time: " << samrai_loop_time_end << endl;
   cout << "SAMRAI initial dt: " << samrai_dt << endl;

   // 9. Alternating time loop
   cout << "\n=== Starting alternating time loop ===" << endl;

   real_t final_time = min(t_final, static_cast<real_t>(samrai_loop_time_end));
   int ti = 1;
   bool last_step = false;

   while (mfem_time < final_time && samrai_loop_time < samrai_loop_time_end)
   {
      if (mfem_time + dt >= final_time - dt/2 ||
          samrai_loop_time + samrai_dt >= samrai_loop_time_end - samrai_dt/2)
      {
         last_step = true;
      }

      // MFEM heat equation step
      ode_solver->Step(u, mfem_time, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "MFEM step " << ti << ", t = " << mfem_time << endl;

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(mfem_time);
            visit_dc.Save();
         }
      }
      oper.SetParameters(u);

      // SAMRAI advection step
      bool rebalance_hierarchy = false;
      double samrai_dt_new = time_integrator->advanceHierarchy(samrai_dt, rebalance_hierarchy);
      samrai_loop_time += samrai_dt;
      samrai_dt = samrai_dt_new;

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "SAMRAI step " << ti << ", t = " << samrai_loop_time
              << ", dt = " << samrai_dt << endl;
      }

      ti++;

      if (last_step)
      {
         break;
      }
   }

   cout << "\n=== Time integration completed ===" << endl;
   cout << "MFEM final time: " << mfem_time << endl;
   cout << "SAMRAI final time: " << samrai_loop_time << endl;

   // 10. Save final solution
   {
      ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   cout << "\nMFEM solution saved to: ex16.mesh, ex16-final.gf" << endl;
#ifdef HAVE_HDF5
   cout << "SAMRAI solution saved to: " << samrai_viz_dump_dirname << endl;
#endif

   // 11. Cleanup
   delete mesh;

   tbox::SAMRAIManager::shutdown();
   tbox::SAMRAIManager::finalize();
   tbox::SAMRAI_MPI::finalize();

   return 0;
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, real_t al,
                                       real_t kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0), fespace(f),
     M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height)
{
   const real_t rel_tol = 1e-8;

   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   alpha = al;
   kappa = kap;

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   SetParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   // Compute:
   //    du_dt = M^{-1}*-Ku
   // for du_dt, where K is linearized by using u from the previous timestep
   Kmat.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const real_t dt,
                                       const Vector &u, Vector &k)
{
   // Solve the equation:
   //    M*k = -K(u + dt*k) for k = du/dt, if solving for stage-slope
   // or
   //    M*k = -dt*K(k) + M*u for k = u_s, if solving for stage-state
   // where K is linearized by using u from the previous timestep, and
   // the stage-state and slope relation: du/dt = (u_s - u)/dt.
   if (!T)
   {
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_solver.SetOperator(*T);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt

   // Construct current right-hand side for stage state vs. slope solve
   if (ImplicitVarTypeIsState())
   {
      // k, on return, is the stage value u_s
      Mmat.Mult(u, z);
   }
   else
   {
      // k, on return, is the stage slope du/dt
      Kmat.Mult(u, z);
      z.Neg();
   }
   T_solver.Mult(z, k);
}

void ConductionOperator::SetParameters(const Vector &u)
{
   GridFunction u_alpha_gf(&fespace);
   u_alpha_gf.SetFromTrueDofs(u);
   for (int i = 0; i < u_alpha_gf.Size(); i++)
   {
      u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
   }

   delete K;
   K = new BilinearForm(&fespace);

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
}

real_t InitialTemperature(const Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
