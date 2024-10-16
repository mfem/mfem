//                       MFEM Example 10 - Parallel Version
//                             SUNDIALS Modification
//
// Compile with:
//    make ex10p            (GNU make)
//    make sundials_ex10p   (CMake)
//
// Sample runs:
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh -rp 1 -o 2 -s 12 -dt 0.15 -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-tri.mesh  -rp 1 -o 2 -s 16 -dt 0.25 -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-hex.mesh  -rp 0 -o 2 -s 12 -dt 0.15 -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-tri.mesh  -rp 1 -o 2 -s  2 -dt 3 -nls 1
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh -rp 1 -o 2 -s  2 -dt 3 -nls 2
//    mpirun -np 4 ex10p -m ../../data/beam-hex.mesh  -rs 1 -o 2 -s  2 -dt 3 -nls 4
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh -rp 1 -o 2 -s 14 -dt 0.15 -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-tri.mesh  -rp 1 -o 2 -s 17 -dt 5e-3 -vs 60
//    mpirun -np 4 ex10p -m ../../data/beam-hex.mesh  -rp 0 -o 2 -s 14 -dt 0.15 -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-quad-amr.mesh -rp 1 -o 2 -s 12 -dt 0.15 -vs 10
//
// Description:  This examples solves a time dependent nonlinear elasticity
//               problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
//               hyperelastic model and S is a viscosity operator of Laplacian
//               type. The geometry of the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |
//                    attribute 1  |                     |
//                    (fixed)      +---------------------+
//
//               The example demonstrates the use of nonlinear operators (the
//               class HyperelasticOperator defining H(x)), as well as their
//               implicit time integration using a Newton method for solving an
//               associated reduced backward-Euler type nonlinear equation
//               (class ReducedSystemOperator). Each Newton step requires the
//               inversion of a Jacobian matrix, which is done through a
//               (preconditioned) inner solver. Note that implementing the
//               method HyperelasticOperator::ImplicitSolve is the only
//               requirement for high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <map>

#ifndef MFEM_USE_SUNDIALS
#error This example requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

using namespace std;
using namespace mfem;

class ReducedSystemOperator;

/** After spatial discretization, the hyperelastic model can be written as a
 *  system of ODEs:
 *     dv/dt = -M^{-1}*(H(x) + S*v)
 *     dx/dt = v,
 *  where x is the vector representing the deformation, v is the velocity field,
 *  M is the mass matrix, S is the viscosity matrix, and H(x) is the nonlinear
 *  hyperelastic operator.
 *
 *  Class HyperelasticOperator represents the right-hand side of the above
 *  system of ODEs. */
class HyperelasticOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   ParBilinearForm M, S;
   ParNonlinearForm H;
   double viscosity;
   HyperelasticModel *model;

   HypreParMatrix *Mmat; // Mass matrix from ParallelAssemble()
   CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
   HypreSmoother M_prec; // Preconditioner for the mass matrix M

   /** Nonlinear operator defining the reduced backward Euler equation for the
       velocity. Used in the implementation of method ImplicitSolve. */
   ReducedSystemOperator *reduced_oper;

   /// Newton solver for the reduced backward Euler equation
   NewtonSolver *newton_solver;

   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian solve in the Newton method
   Solver *J_prec;

   mutable Vector z; // auxiliary vector

   const SparseMatrix *local_grad_H;
   HypreParMatrix *Jacobian;

   double saved_gamma; // saved gamma value from implicit setup

public:

   HyperelasticOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc, double mu, double K,
                        int kinsol_nls_type = -1, double kinsol_damping = 0.0,
                        int kinsol_aa_n = 0);

   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);


   /// Custom Jacobian system solver for the SUNDIALS time integrators.
   /** For the ODE system represented by HyperelasticOperator

          M dv/dt = -(H(x) + S*v)
          dx/dt = v,

       this class facilitates the solution of linear systems of the form

           (M + γS) yv + γJ yx = M bv,   J=(dH/dx)(x)
                - γ yv +    yx =   bx

       for given bv, bx, x, and γ = GetTimeStep(). */

   /** Linear solve applicable to the SUNDIALS format.
       Solves (Mass - dt J) y = Mass b, where in our case:
       Mass = | M  0 |  J = | -S  -grad_H |  y = | v_hat |  b = | b_v |
              | 0  I |      |  I     0    |      | x_hat |      | b_x |
       The result replaces the rhs b.
       We substitute x_hat = b_x + dt v_hat and solve
       (M + dt S + dt^2 grad_H) v_hat = M b_v - dt grad_H b_x. */

   /** Setup the linear system. This method is used by the implicit
       SUNDIALS solvers. */
   virtual int SUNImplicitSetup(const Vector &y, const Vector &fy,
                                int jok, int *jcur, double gamma);

   /** Solve the linear system. This method is used by the implicit
       SUNDIALS solvers. */
   virtual int SUNImplicitSolve(const Vector &b, Vector &x, double tol);

   double ElasticEnergy(const ParGridFunction &x) const;
   double KineticEnergy(const ParGridFunction &v) const;
   void GetElasticEnergyDensity(const ParGridFunction &x,
                                ParGridFunction &w) const;

   virtual ~HyperelasticOperator();
};

/** Nonlinear operator of the form:
    k --> (M + dt*S)*k + H(x + dt*v + dt^2*k) + S*v,
    where M and S are given BilinearForms, H is a given NonlinearForm, v and x
    are given vectors, and dt is a scalar. */
class ReducedSystemOperator : public Operator
{
private:
   ParBilinearForm *M, *S;
   ParNonlinearForm *H;
   mutable HypreParMatrix *Jacobian;
   double dt;
   const Vector *v, *x;
   mutable Vector w, z;
   const Array<int> &ess_tdof_list;

public:
   ReducedSystemOperator(ParBilinearForm *M_, ParBilinearForm *S_,
                         ParNonlinearForm *H_, const Array<int> &ess_tdof_list);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *v_, const Vector *x_);

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();
};


/** Function representing the elastic energy density for the given hyperelastic
    model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient : public Coefficient
{
private:
   HyperelasticModel     &model;
   const ParGridFunction &x;
   DenseMatrix            J;

public:
   ElasticEnergyCoefficient(HyperelasticModel &m, const ParGridFunction &x_)
      : model(m), x(x_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ElasticEnergyCoefficient() { }
};

void InitialDeformation(const Vector &x, Vector &y);

void InitialVelocity(const Vector &x, Vector &v);

void visualize(ostream &os, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI, HYPRE, and SUNDIALS.
   Mpi::Init(argc, argv);
   int myid = Mpi::WorldRank();
   Hypre::Init();
   Sundials::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../../data/beam-quad.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 2;
   int ode_solver_type = 3;
   double t_final = 300.0;
   double dt = 3.0;
   double visc = 1e-2;
   double mu = 0.25;
   double K = 5.0;
   bool visualization = true;
   int nonlinear_solver_type = 0;
   int vis_steps = 1;
   double kinsol_damping = 0.0;
   int kinsol_aa_n = -1;

   // Relative and absolute tolerances for CVODE and ARKODE.
   const double reltol = 1e-1, abstol = 1e-1;
   // Since this example uses the loose tolerances defined above, it is
   // necessary to lower the linear solver tolerance for CVODE which is relative
   // to the above tolerances.
   const double cvode_eps_lin = 1e-4;
   // Similarly, the nonlinear tolerance for ARKODE needs to be tightened.
   const double arkode_eps_nonlin = 1e-6;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver:\n\t"
                  "1  - Backward Euler,\n\t"
                  "2  - SDIRK2, L-stable\n\t"
                  "3  - SDIRK3, L-stable\n\t"
                  "4  - Implicit Midpoint,\n\t"
                  "5  - SDIRK2, A-stable,\n\t"
                  "6  - SDIRK3, A-stable,\n\t"
                  "7  - Forward Euler,\n\t"
                  "8  - RK2,\n\t"
                  "9  - RK3 SSP,\n\t"
                  "10 - RK4,\n\t"
                  "11 - CVODE implicit BDF, approximate Jacobian,\n\t"
                  "12 - CVODE implicit BDF, specified Jacobian,\n\t"
                  "13 - CVODE implicit ADAMS, approximate Jacobian,\n\t"
                  "14 - CVODE implicit ADAMS, specified Jacobian,\n\t"
                  "15 - ARKODE implicit, approximate Jacobian,\n\t"
                  "16 - ARKODE implicit, specified Jacobian,\n\t"
                  "17 - ARKODE explicit, 4th order.");
   args.AddOption(&nonlinear_solver_type, "-nls", "--nonlinear-solver",
                  "Nonlinear system solver:\n\t"
                  "0  - MFEM Newton method,\n\t"
                  "1  - KINSOL Newton method,\n\t"
                  "2  - KINSOL Newton method with globalization,\n\t"
                  "3  - KINSOL fixed-point method (with or without AA),\n\t"
                  "4  - KINSOL Picard method (with or without AA).");
   args.AddOption(&kinsol_damping, "-damp", "--kinsol-damping",
                  "Picard or Fixed-Point damping parameter (only valid with KINSOL): "
                  "0 < d <= 1.0");
   args.AddOption(&kinsol_aa_n, "-aan", "--anderson-subspace",
                  "Anderson Acceleration subspace size (only valid with KINSOL)");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visc, "-v", "--viscosity",
                  "Viscosity coefficient.");
   args.AddOption(&mu, "-mu", "--shear-modulus",
                  "Shear modulus in the Neo-Hookean hyperelastic model.");
   args.AddOption(&K, "-K", "--bulk-modulus",
                  "Bulk modulus in the Neo-Hookean hyperelastic model.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // check for valid ODE solver option
   if (ode_solver_type < 1 || ode_solver_type > 17)
   {
      if (myid == 0)
      {
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      }
      return 1;
   }

   // check for valid nonlinear solver options
   if (nonlinear_solver_type < 0 || nonlinear_solver_type > 4)
   {
      if (myid == 0)
      {
         cout << "Unknown nonlinear solver type: " << nonlinear_solver_type
              << "\n";
      }
      return 1;
   }
   if (kinsol_damping > 0.0 &&
       !(nonlinear_solver_type == 3 || nonlinear_solver_type == 4))
   {
      if (myid == 0)
      {
         cout << "Only KINSOL fixed-point and Picard methods can use damping\n";
      }
      return 1;
   }
   if (kinsol_aa_n > 0 &&
       !(nonlinear_solver_type == 3 || nonlinear_solver_type == 4))
   {
      if (myid == 0)
      {
         cout << "Only KINSOL fixed-point and Picard methods can use AA\n";
      }
      return 1;
   }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 6. Define the parallel vector finite element spaces representing the mesh
   //    deformation x_gf, the velocity v_gf, and the initial configuration,
   //    x_ref. Define also the elastic energy density, w_gf, which is in a
   //    discontinuous higher-order space. Since x and v are integrated in time
   //    as a system, we group them together in block vector vx, on the unique
   //    parallel degrees of freedom, with offsets given by array true_offset.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll, dim);

   HYPRE_BigInt glob_size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of velocity/deformation unknowns: " << glob_size << endl;
   }
   int true_size = fespace.TrueVSize();
   Array<int> true_offset(3);
   true_offset[0] = 0;
   true_offset[1] = true_size;
   true_offset[2] = 2*true_size;

   BlockVector vx(true_offset);
   ParGridFunction v_gf, x_gf;
   v_gf.MakeTRef(&fespace, vx, true_offset[0]);
   x_gf.MakeTRef(&fespace, vx, true_offset[1]);

   ParGridFunction x_ref(&fespace);
   pmesh->GetNodes(x_ref);

   L2_FECollection w_fec(order + 1, dim);
   ParFiniteElementSpace w_fespace(pmesh, &w_fec);
   ParGridFunction w_gf(&w_fespace);

   // 7. Set the initial conditions for v_gf, x_gf and vx, and define the
   //    boundary conditions on a beam-like mesh (see description above).
   VectorFunctionCoefficient velo(dim, InitialVelocity);
   v_gf.ProjectCoefficient(velo);
   v_gf.SetTrueVector();
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   x_gf.ProjectCoefficient(deform);
   x_gf.SetTrueVector();

   v_gf.SetFromTrueVector(); x_gf.SetFromTrueVector();

   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

   // 8. Initialize the hyperelastic operator, the GLVis visualization and print
   //    the initial energies.
   std::unique_ptr<HyperelasticOperator> oper;
   if (nonlinear_solver_type == 0)
      oper = std::make_unique<HyperelasticOperator>(fespace, ess_bdr, visc, mu,
                                                    K);
   else
   {
      switch (nonlinear_solver_type)
      {
         case 1:
            oper = std::make_unique<HyperelasticOperator>(fespace, ess_bdr,
                                                          visc, mu, K, KIN_NONE);
            break;
         case 2:
            oper = std::make_unique<HyperelasticOperator>(fespace, ess_bdr,
                                                          visc, mu, K, KIN_LINESEARCH);
            break;
         case 3:
            oper = std::make_unique<HyperelasticOperator>(fespace, ess_bdr,
                                                          visc, mu, K, KIN_FP, kinsol_damping, kinsol_aa_n);
            break;
         case 4:
            oper = std::make_unique<HyperelasticOperator>(fespace, ess_bdr,
                                                          visc, mu, K, KIN_PICARD, kinsol_damping, kinsol_aa_n);
            break;
         default:
            cout << "Unknown type of nonlinear solver: "
                 << nonlinear_solver_type << endl;
            return 4;
      }
   }

   socketstream vis_v, vis_w;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_v.open(vishost, visport);
      vis_v.precision(8);
      visualize(vis_v, pmesh, &x_gf, &v_gf, "Velocity", true);
      // Make sure all ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_w.open(vishost, visport);
      if (vis_w)
      {
         oper->GetElasticEnergyDensity(x_gf, w_gf);
         vis_w.precision(8);
         visualize(vis_w, pmesh, &x_gf, &w_gf, "Elastic energy density", true);
      }
   }

   double ee0 = oper->ElasticEnergy(x_gf);
   double ke0 = oper->KineticEnergy(v_gf);
   if (myid == 0)
   {
      cout << "initial elastic energy (EE) = " << ee0 << endl;
      cout << "initial kinetic energy (KE) = " << ke0 << endl;
      cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;
   }

   // 9. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   double t = 0.0;
   oper->SetTime(t);

   ODESolver *ode_solver = NULL;
   CVODESolver *cvode = NULL;
   ARKStepSolver *arkode = NULL;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 4:  ode_solver = new ImplicitMidpointSolver; break;
      case 5:  ode_solver = new SDIRK23Solver; break;
      case 6:  ode_solver = new SDIRK34Solver; break;
      // Explicit methods
      case 7:  ode_solver = new ForwardEulerSolver; break;
      case 8:  ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 9:  ode_solver = new RK3SSPSolver; break;
      case 10: ode_solver = new RK4Solver; break;
      // CVODE BDF
      case 11:
      case 12:
         cvode = new CVODESolver(MPI_COMM_WORLD, CV_BDF);
         cvode->Init(*oper);
         cvode->SetSStolerances(reltol, abstol);
         CVodeSetEpsLin(cvode->GetMem(), cvode_eps_lin);
         cvode->SetMaxStep(dt);
         if (ode_solver_type == 11)
         {
            cvode->UseSundialsLinearSolver();
         }
         ode_solver = cvode; break;
      // CVODE Adams
      case 13:
      case 14:
         cvode = new CVODESolver(MPI_COMM_WORLD, CV_ADAMS);
         cvode->Init(*oper);
         cvode->SetSStolerances(reltol, abstol);
         CVodeSetEpsLin(cvode->GetMem(), cvode_eps_lin);
         cvode->SetMaxStep(dt);
         if (ode_solver_type == 13)
         {
            cvode->UseSundialsLinearSolver();
         }
         ode_solver = cvode; break;
      // ARKStep Implicit methods
      case 15:
      case 16:
         arkode = new ARKStepSolver(MPI_COMM_WORLD, ARKStepSolver::IMPLICIT);
         arkode->Init(*oper);
         arkode->SetSStolerances(reltol, abstol);
         ARKStepSetNonlinConvCoef(arkode->GetMem(), arkode_eps_nonlin);
         arkode->SetMaxStep(dt);
         if (ode_solver_type == 15)
         {
            arkode->UseSundialsLinearSolver();
         }
         ode_solver = arkode; break;
      // ARKStep Explicit methods
      case 17:
         arkode = new ARKStepSolver(MPI_COMM_WORLD, ARKStepSolver::EXPLICIT);
         arkode->Init(*oper);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         ode_solver = arkode; break;
   }

   // Initialize MFEM integrators, SUNDIALS integrators are initialized above
   if (ode_solver_type < 11) { ode_solver->Init(*oper); }

   // 10. Perform time-integration
   //     (looping over the time iterations, ti, with a time-step dt).
   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(vx, t, dt_real);

      last_step = (t >= t_final - 1e-8*dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         v_gf.SetFromTrueVector(); x_gf.SetFromTrueVector();

         double ee = oper->ElasticEnergy(x_gf);
         double ke = oper->KineticEnergy(v_gf);

         if (myid == 0)
         {
            cout << "step " << ti << ", t = " << t << ", EE = " << ee
                 << ", KE = " << ke << ", ΔTE = " << (ee+ke)-(ee0+ke0) << endl;

            if (cvode) { cvode->PrintInfo(); }
            else if (arkode) { arkode->PrintInfo(); }
         }

         if (visualization)
         {
            visualize(vis_v, pmesh, &x_gf, &v_gf);
            if (vis_w)
            {
               oper->GetElasticEnergyDensity(x_gf, w_gf);
               visualize(vis_w, pmesh, &x_gf, &w_gf);
            }
         }
      }
   }

   // 11. Save the displaced mesh, the velocity and elastic energy.
   {
      v_gf.SetFromTrueVector(); x_gf.SetFromTrueVector();
      GridFunction *nodes = &x_gf;
      int owns_nodes = 0;
      pmesh->SwapNodes(nodes, owns_nodes);

      ostringstream mesh_name, velo_name, ee_name;
      mesh_name << "deformed." << setfill('0') << setw(6) << myid;
      velo_name << "velocity." << setfill('0') << setw(6) << myid;
      ee_name << "elastic_energy." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);
      pmesh->SwapNodes(nodes, owns_nodes);
      ofstream velo_ofs(velo_name.str().c_str());
      velo_ofs.precision(8);
      v_gf.Save(velo_ofs);
      ofstream ee_ofs(ee_name.str().c_str());
      ee_ofs.precision(8);
      oper->GetElasticEnergyDensity(x_gf, w_gf);
      w_gf.Save(ee_ofs);
   }

   // 12. Free the used memory.
   delete ode_solver;
   delete pmesh;

   return 0;
}

void visualize(ostream &os, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
   if (!os)
   {
      return;
   }

   GridFunction *nodes = deformed_nodes;
   int owns_nodes = 0;

   mesh->SwapNodes(nodes, owns_nodes);

   os << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
   os << "solution\n" << *mesh << *field;

   mesh->SwapNodes(nodes, owns_nodes);

   if (init_vis)
   {
      os << "window_size 800 800\n";
      os << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         os << "view 0 0\n"; // view from top
         os << "keys jl\n";  // turn off perspective and light
      }
      os << "keys cm\n";         // show colorbar and mesh
      os << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      os << "pause\n";
   }
   os << flush;
}


ReducedSystemOperator::ReducedSystemOperator(
   ParBilinearForm *M_, ParBilinearForm *S_, ParNonlinearForm *H_,
   const Array<int> &ess_tdof_list_)
   : Operator(M_->ParFESpace()->TrueVSize()), M(M_), S(S_), H(H_),
     Jacobian(NULL), dt(0.0), v(NULL), x(NULL), w(height), z(height),
     ess_tdof_list(ess_tdof_list_)
{ }

void ReducedSystemOperator::SetParameters(double dt_, const Vector *v_,
                                          const Vector *x_)
{
   dt = dt_;  v = v_;  x = x_;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   // compute: y = H(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   H->Mult(z, y);
   M->TrueAddMult(k, y);
   S->TrueAddMult(w, y);
   y.SetSubVector(ess_tdof_list, 0.0);
}

Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   delete Jacobian;
   SparseMatrix *localJ = Add(1.0, M->SpMat(), dt, S->SpMat());
   add(*v, dt, k, w);
   add(*x, dt, w, z);
   localJ->Add(dt*dt, H->GetLocalGradient(z));
   Jacobian = M->ParallelAssemble(localJ);
   delete localJ;
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(ess_tdof_list);
   delete Je;
   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Jacobian;
}


HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace &f,
                                           Array<int> &ess_bdr, double visc,
                                           double mu, double K,
                                           int kinsol_nls_type,
                                           double kinsol_damping,
                                           int kinsol_aa_n)

   : TimeDependentOperator(2*f.TrueVSize(), 0.0), fespace(f),
     M(&fespace), S(&fespace), H(&fespace),
     viscosity(visc), M_solver(f.GetComm()), z(height/2),
     local_grad_H(NULL), Jacobian(NULL)
{
   const double rel_tol = 1e-8;
   const int skip_zero_entries = 0;

   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);
   M.AddDomainIntegrator(new VectorMassIntegrator(rho0));
   M.Assemble(skip_zero_entries);
   M.Finalize(skip_zero_entries);
   Mmat = M.ParallelAssemble();
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   HypreParMatrix *Me = Mmat->EliminateRowsCols(ess_tdof_list);
   delete Me;

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(*Mmat);

   model = new NeoHookeanModel(mu, K);
   H.AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
   H.SetEssentialTrueDofs(ess_tdof_list);

   ConstantCoefficient visc_coeff(viscosity);
   S.AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
   S.Assemble(skip_zero_entries);
   S.Finalize(skip_zero_entries);

   reduced_oper = new ReducedSystemOperator(&M, &S, &H, ess_tdof_list);

   HypreSmoother *J_hypreSmoother = new HypreSmoother;
   J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
   J_hypreSmoother->SetPositiveDiagonal(true);
   J_prec = J_hypreSmoother;

   MINRESSolver *J_minres = new MINRESSolver(f.GetComm());
   J_minres->SetRelTol(rel_tol);
   J_minres->SetAbsTol(0.0);
   J_minres->SetMaxIter(300);
   J_minres->SetPrintLevel(-1);
   J_minres->SetPreconditioner(*J_prec);
   J_solver = J_minres;

   if (kinsol_nls_type > 0)
   {
      KINSolver *kinsolver = new KINSolver(f.GetComm(), kinsol_nls_type, true);
      if (kinsol_nls_type != KIN_PICARD)
      {
         kinsolver->SetJFNK(true);
         kinsolver->SetLSMaxIter(100);
      }
      if (kinsol_aa_n > 0)
      {
         kinsolver->EnableAndersonAcc(kinsol_aa_n);
      }
      newton_solver = kinsolver;
      newton_solver->SetOperator(*reduced_oper);
      newton_solver->SetMaxIter(200);
      newton_solver->SetRelTol(rel_tol);
      newton_solver->SetPrintLevel(0);
      kinsolver->SetMaxSetupCalls(4);
      if (kinsol_damping > 0.0)
      {
         kinsolver->SetDamping(kinsol_damping);
      }
   }
   else
   {
      newton_solver = new NewtonSolver(f.GetComm());
      newton_solver->SetOperator(*reduced_oper);
      newton_solver->SetMaxIter(10);
      newton_solver->SetRelTol(rel_tol);
      newton_solver->SetPrintLevel(-1);
   }
   newton_solver->SetSolver(*J_solver);
   newton_solver->iterative_mode = false;
}

void HyperelasticOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
   // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   H.Mult(x, z);
   if (viscosity != 0.0)
   {
      S.TrueAddMult(v, z);
      z.SetSubVector(ess_tdof_list, 0.0);
   }
   z.Neg(); // z = -z
   M_solver.Mult(z, dv_dt);

   dx_dt = v;
}

void HyperelasticOperator::ImplicitSolve(const double dt,
                                         const Vector &vx, Vector &dvx_dt)
{
   int sc = height/2;
   Vector v(vx.GetData() +  0, sc);
   Vector x(vx.GetData() + sc, sc);
   Vector dv_dt(dvx_dt.GetData() +  0, sc);
   Vector dx_dt(dvx_dt.GetData() + sc, sc);

   // By eliminating kx from the coupled system:
   //    kv = -M^{-1}*[H(x + dt*kx) + S*(v + dt*kv)]
   //    kx = v + dt*kv
   // we reduce it to a nonlinear equation for kv, represented by the
   // reduced_oper. This equation is solved with the newton_solver
   // object (using J_solver and J_prec internally).
   reduced_oper->SetParameters(dt, &v, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton_solver->Mult(zero, dv_dt);
   MFEM_VERIFY(newton_solver->GetConverged(),
               "Nonlinear solver did not converge.");
#ifdef MFEM_DEBUG
   if (fespace.GetMyRank() == 0)
   {
      cout << "  num nonlin sol iters = " << newton_solver->GetNumIterations()
           << ", final norm = " << newton_solver->GetFinalNorm() << '\n';
   }
#endif
   add(v, dt, dv_dt, dx_dt);
}

int HyperelasticOperator::SUNImplicitSetup(const Vector &y,
                                           const Vector &fy, int jok, int *jcur,
                                           double gamma)
{
   int sc = y.Size() / 2;
   const Vector x(y.GetData() + sc, sc);

   // J = M + dt*(S + dt*grad(H))
   if (Jacobian) { delete Jacobian; }
   SparseMatrix *localJ = Add(1.0, M.SpMat(), gamma, S.SpMat());
   local_grad_H = &H.GetLocalGradient(x);
   localJ->Add(gamma*gamma, *local_grad_H);
   Jacobian = M.ParallelAssemble(localJ);
   delete localJ;
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(ess_tdof_list);
   delete Je;

   // Set Jacobian solve operator
   J_solver->SetOperator(*Jacobian);

   // Indicate that the Jacobian was updated
   *jcur = 1;

   // Save gamma for use in solve
   saved_gamma = gamma;

   // Return success
   return 0;
}

int HyperelasticOperator::SUNImplicitSolve(const Vector &b, Vector &x,
                                           double tol)
{
   int sc = b.Size() / 2;
   ParFiniteElementSpace *fes = H.ParFESpace();
   Vector b_v(b.GetData() +  0, sc);
   Vector b_x(b.GetData() + sc, sc);
   Vector x_v(x.GetData() +  0, sc);
   Vector x_x(x.GetData() + sc, sc);
   Vector rhs(sc);

   // We can assume that b_v and b_x have zeros at essential tdofs.

   // rhs = M b_v - dt*grad(H) b_x
   ParGridFunction lb_x(fes), lrhs(fes);
   lb_x.Distribute(b_x);
   local_grad_H->Mult(lb_x, lrhs);
   lrhs.ParallelAssemble(rhs);
   rhs *= -saved_gamma;
   M.TrueAddMult(b_v, rhs);
   rhs.SetSubVector(ess_tdof_list, 0.0);

   J_solver->iterative_mode = false;
   J_solver->Mult(rhs, x_v);

   add(b_x, saved_gamma, x_v, x_x);

   return 0;
}

double HyperelasticOperator::ElasticEnergy(const ParGridFunction &x) const
{
   return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const ParGridFunction &v) const
{
   double energy = 0.5*M.ParInnerProduct(v, v);
   return energy;
}

void HyperelasticOperator::GetElasticEnergyDensity(
   const ParGridFunction &x, ParGridFunction &w) const
{
   ElasticEnergyCoefficient w_coeff(*model, x);
   w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
   delete Jacobian;
   delete newton_solver;
   delete J_solver;
   delete J_prec;
   delete reduced_oper;
   delete model;
   delete Mmat;
}


double ElasticEnergyCoefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
   model.SetTransformation(T);
   x.GetVectorGradient(T, J);
   // return model.EvalW(J);  // in reference configuration
   return model.EvalW(J)/J.Det(); // in deformed configuration
}


void InitialDeformation(const Vector &x, Vector &y)
{
   // set the initial configuration to be the same as the reference, stress
   // free, configuration
   y = x;
}

void InitialVelocity(const Vector &x, Vector &v)
{
   const int dim = x.Size();
   const double s = 0.1/64.;

   v = 0.0;
   v(dim-1) = s*x(0)*x(0)*(8.0-x(0));
   v(0) = -s*x(0)*x(0);
}
