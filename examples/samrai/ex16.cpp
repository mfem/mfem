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

#include "MeshOps.hpp"

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

using namespace mfem;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with scalar diffusivity \kappa
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   std::unique_ptr<ParBilinearForm> M;
   std::unique_ptr<ParBilinearForm> K;

   HypreParMatrix Mmat, Kmat;
   std::unique_ptr<HypreParMatrix> Tmat; // T = M + dt K
   real_t current_dt;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + dt K
   HypreSmoother T_prec;  // Preconditioner for the implicit solver

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(ParFiniteElementSpace* fes, real_t kappa);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;
};

real_t InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{
   const int precision = 8;
   std::cout.precision(precision);

   int ode_solver_type = 23;  // SDIRK33Solver
   real_t kappa = 0.01;

   bool visualization = true;
   bool visit = false;
   int vis_steps = 5;
   bool solve_implicit_state = false;

   // Define command line argument defaults for SAMRAI
   const char *samrai_input_file = "samrai_input.3d";

   // Parse command line arguments
   OptionsParser args(argc, argv);
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&kappa, "-k", "--kappa",
                  "Diffusion coefficient.");
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
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   // Initialize SAMRAI
   SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
   SAMRAI::tbox::SAMRAIManager::initialize();
   SAMRAI::tbox::SAMRAIManager::startup();
   const SAMRAI::tbox::SAMRAI_MPI& mpi(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

   /************************* Create SAMRAI objects ***************************/

   // Parse SAMRAI input file
   SAMRAI::tbox::InputManager::getManager()->parseInputFile(samrai_input_file);
   std::shared_ptr<SAMRAI::tbox::Database> input_db =
      SAMRAI::tbox::InputManager::getManager()->getInputDatabase();

   const SAMRAI::tbox::Dimension samrai_dim(
      static_cast<unsigned short>(input_db->getDatabase("Main")->getInteger("dim")));
   auto grid_geometry = std::make_shared<SAMRAI::geom::CartesianGridGeometry>(
      samrai_dim, "CartesianGeometry", input_db->getDatabase("CartesianGeometry"));
   auto patch_hierarchy = std::make_shared<SAMRAI::hier::PatchHierarchy>(
      "PatchHierarchy", grid_geometry, input_db->getDatabase("PatchHierarchy"));
   auto linadv_model = std::make_shared<LinAdv>("LinAdv", samrai_dim,
      input_db->getDatabase("LinAdv"), grid_geometry);
   auto hyp_level_integrator = std::make_shared<SAMRAI::algs::HyperbolicLevelIntegrator>(
      "HyperbolicLevelIntegrator", input_db->getDatabase("HyperbolicLevelIntegrator"),
      linadv_model.get(), true);
   auto error_detector = std::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
      "StandardTagAndInitialize", hyp_level_integrator.get(),
      input_db->getDatabase("StandardTagAndInitialize"));
   auto box_generator = std::make_shared<SAMRAI::mesh::BergerRigoutsos>(samrai_dim,
      input_db->getDatabase("BergerRigoutsos"));
   auto load_balancer = std::make_shared<SAMRAI::mesh::CascadePartitioner>(samrai_dim,
      "CascadePartitioner", input_db->getDatabase("LoadBalancer"));
   auto gridding_algorithm = std::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
      patch_hierarchy, "GriddingAlgorithm",
      input_db->getDatabase("GriddingAlgorithm"), error_detector, box_generator,
      load_balancer);
   auto samrai_time_integrator = std::make_unique<SAMRAI::algs::TimeRefinementIntegrator>(
      "TimeRefinementIntegrator", input_db->getDatabase("TimeRefinementIntegrator"),
      patch_hierarchy, hyp_level_integrator, gridding_algorithm);

   double dt = samrai_time_integrator->initializeHierarchy();
   const int samrai_position_id = linadv_model->getPositionId();
   const int samrai_state_id = linadv_model->getStateId();

   /************************** Create MFEM objects ****************************/

   // Create MFEM mesh to match SAMRAI mesh using the SAMRAI communicator
   MeshOps mesh_ops(patch_hierarchy, mpi.getCommunicator());
   const ParMesh& mesh = mesh_ops.getMesh();

   // Transfer initial SAMRAI state (cell averages) to an MFEM grid function
   // (piecewise-constant representation)
   std::unique_ptr<ParGridFunction> uavg_gf;
   {
      std::vector<std::unique_ptr<ParGridFunction>> gfs =
         mesh_ops.transferToMFEM(samrai_position_id, {}, {samrai_state_id});
      uavg_gf = std::move(gfs[0]);
   }

   // Create an MFEM grid function and obtain initial condition by
   // reconstructing from the piece-constant representation
   const int order = 1;
   H1_FECollection u_fecollection(order, mesh.Dimension());
   std::unique_ptr<ParFiniteElementSpace> u_fespace =
      mesh_ops.createFESpace(u_fecollection);
   std::cout << "Number of temperature unknowns: " << u_fespace->GlobalTrueVSize()
             << std::endl;
   ParGridFunction u_gf(u_fespace.get());
   Vector u;
   u_gf.GetTrueDofs(u);
   {
      OperatorPtr M_op;
      ParBilinearForm M(u_fespace.get());
      M.AddDomainIntegrator(new MassIntegrator());
      M.Assemble();
      M.FormSystemMatrix({}, M_op);

      OperatorPtr Rhs_op;
      Vector uavg;
      Vector rhs(u.Size());
      ParMixedBilinearForm Rhs(uavg_gf->ParFESpace(), u_fespace.get());
      Rhs.AddDomainIntegrator(new MassIntegrator());
      Rhs.Assemble();
      Rhs.FormRectangularSystemMatrix({}, {}, Rhs_op);
      uavg_gf->GetTrueDofs(uavg);
      Rhs_op->Mult(uavg, rhs);

      CGSolver solver(mesh.GetComm());
      solver.iterative_mode = false;
      solver.SetRelTol(1e-5);
      solver.SetAbsTol(0.0);
      solver.SetMaxIter(30);
      solver.SetPrintLevel(0);
      HypreSmoother prec;
      //prec.SetType(HypreSmoother::Jacobi);
      //solver.SetPreconditioner(prec);
      solver.SetOperator(*M_op);
      solver.Mult(rhs, u);
      MFEM_VERIFY(solver.GetConverged(), "Solver did not converge.");
   }
   u = 1.0;
   u_gf.SetFromTrueDofs(u);

/*
   // Create the conduction operator
   ConductionOperator conduction(u_fespace, kappa);
   using ImplicitVariableType = ConductionOperator::ImplicitVariableType;
   ImplicitVariableType imp_var = solve_implicit_state ?
                                  ImplicitVariableType::STATE
                                  : ImplicitVariableType::SLOPE;
   conduction.SetImplicitVariableType(imp_var);

   // Create the ODE solver used for time integration
   std::unique_ptr<ODESolver> mfem_ode_solver = ODESolver::Select(ode_solver_type);
   mfem_ode_solver->Init(conduction);
*/
   // Write out the mesh and initial condition
   {
      std::ofstream omesh("ex16.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      std::ofstream osol("ex16-init.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // Optionally, create the VisIt output object and save the initial condition
   std::unique_ptr<VisItDataCollection> visit_dc;
   if (visit)
   {
      //visit_dc = std::make_unique<VisItDataCollection>("Example16", &mesh);
      visit_dc->RegisterField("temperature", &u_gf);
      visit_dc->SetCycle(0);
      visit_dc->SetTime(0.0);
      visit_dc->Save();
   }

   // Optionally, create the visualization stream and visualize initial condition
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         std::cout << "Unable to connect to GLVis server at "
                   << vishost << ':' << visport << std::endl;
         visualization = false;
         std::cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << mesh << u_gf;
         sout << "pause\n";
         sout << std::flush;
         std::cout << "GLVis visualization paused."
                   << " Press space (in the GLVis window) to resume it.\n";
      }
   }
   // Create the conduction operator
   ConductionOperator conduction(u_fespace.get(), kappa);
   using ImplicitVariableType = ConductionOperator::ImplicitVariableType;
   ImplicitVariableType imp_var = solve_implicit_state ?
                                  ImplicitVariableType::STATE
                                  : ImplicitVariableType::SLOPE;
   conduction.SetImplicitVariableType(imp_var);

   // Create the ODE solver used for time integration
   std::unique_ptr<ODESolver> mfem_ode_solver = ODESolver::Select(ode_solver_type);
   mfem_ode_solver->Init(conduction);



   /*************************** Advance Solutions *****************************/

   const double final_time = samrai_time_integrator->getEndTime();
   int ti = 1;
   bool last_step = false;
   double time = samrai_time_integrator->getIntegratorTime();
   while (time < final_time)
   {
      if (time + dt >=  final_time - 0.5*dt)
      {
         last_step = true;
      }

      // SAMRAI advection step
      const double dt_new = samrai_time_integrator->advanceHierarchy(dt);

      // MFEM heat equation step
      mfem_ode_solver->Step(u, time, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         std::cout << "step " << ti << ", t = " << time << std::endl;

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            sout << "solution\n" << mesh << u_gf << std::flush;
         }

         if (visit)
         {
            visit_dc->SetCycle(ti);
            visit_dc->SetTime(time);
            visit_dc->Save();
         }
      }

      ti++;
      dt = dt_new;

      if (last_step)
      {
         break;
      }
   }

   // 10. Save final solution
   {
      std::ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   std::cout << "\nMFEM solution saved to: ex16.mesh, ex16-final.gf" << std::endl;

   SAMRAI::tbox::SAMRAIManager::shutdown();
   SAMRAI::tbox::SAMRAIManager::finalize();
   SAMRAI::tbox::SAMRAI_MPI::finalize();

   return 0;
}

ConductionOperator::ConductionOperator(ParFiniteElementSpace *fespace,
   real_t kappa) : TimeDependentOperator(fespace->GetTrueVSize(), (real_t) 0.0),
   current_dt(0.0), M_solver(fespace->GetComm()), T_solver(fespace->GetComm()),
   z(height)
{
   const real_t rel_tol = 1e-8;

   M = std::make_unique<ParBilinearForm>(fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble(0);
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   K = std::make_unique<ParBilinearForm>(fespace);
   ConstantCoefficient kappa_coeff(kappa);
   K->AddDomainIntegrator(new DiffusionIntegrator(kappa_coeff));
   K->Assemble(0);
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(1000);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);
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
   Tmat = std::unique_ptr<HypreParMatrix>(Add(1.0, Mmat, dt, Kmat));
   T_solver.SetOperator(*Tmat);

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

real_t InitialTemperature(const Vector &x)
{
   Vector c({15.0, 10.0});
   c -= x;
   if (c.Norml2() < 5.0)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
