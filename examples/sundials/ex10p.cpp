//                       MFEM Example 10 - Parallel Version
//                             SUNDIALS Modification
//
// Compile with: make ex10p
//
// Sample runs:
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh -rp 1 -o 2 -s  5 -dt 0.15 -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-tri.mesh  -rp 1 -o 2 -s  7 -dt 0.25  -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-hex.mesh  -rp 0 -o 2 -s  5 -dt 0.15  -vs 10
//    mpirun -np 4 ex10p -m ../../data/beam-tri.mesh  -rp 1 -o 2 -s  2 -dt 3 -nls kinsol
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh -rp 1 -o 2 -s  2 -dt 3 -nls kinsol
//    mpirun -np 4 ex10p -m ../../data/beam-hex.mesh  -rs 1 -o 2 -s  2 -dt 3 -nls kinsol
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh -rp 1 -o 2 -s 15 -dt 3e-3 -vs 120
//    mpirun -np 4 ex10p -m ../../data/beam-tri.mesh  -rp 1 -o 2 -s 16 -dt 5e-3 -vs 60
//    mpirun -np 4 ex10p -m ../../data/beam-hex.mesh  -rp 0 -o 2 -s 15 -dt 5e-3 -vs 60
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
class SundialsJacSolver;

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

public:
   /// Solver type to use in the ImplicitSolve() method, used by SDIRK methods.
   enum NonlinearSolverType
   {
      NEWTON = 0, ///< Use MFEM's plain NewtonSolver
      KINSOL = 1  ///< Use SUNDIALS' KINSOL (through MFEM's class KinSolver)
   };

   HyperelasticOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc, double mu, double K,
                        NonlinearSolverType nls_type);

   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;
   /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   /** Connect the Jacobian linear system solver (SundialsJacSolver) used by
       SUNDIALS' CVODE and ARKODE time integrators to the internal objects
       created by HyperelasticOperator. This method is called by the InitSystem
       method of SundialsJacSolver. */
   void InitSundialsJacSolver(SundialsJacSolver &sjsolv);

   double ElasticEnergy(ParGridFunction &x) const;
   double KineticEnergy(ParGridFunction &v) const;
   void GetElasticEnergyDensity(ParGridFunction &x, ParGridFunction &w) const;

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

public:
   ReducedSystemOperator(ParBilinearForm *M_, ParBilinearForm *S_,
                         ParNonlinearForm *H_);

   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *v_, const Vector *x_);

   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();
};

/// Custom Jacobian system solver for the SUNDIALS time integrators.
/** For the ODE system represented by HyperelasticOperator

        M dv/dt = -(H(x) + S*v)
          dx/dt = v,

    this class facilitates the solution of linear systems of the form

        (M + γS) yv + γJ yx = M bv,   J=(dH/dx)(x)
             - γ yv +    yx =   bx

    for given bv, bx, x, and γ = GetTimeStep(). */
class SundialsJacSolver : public SundialsODELinearSolver
{
private:
   ParBilinearForm *M, *S;
   ParNonlinearForm *H;
   const SparseMatrix *local_grad_H;
   HypreParMatrix *Jacobian;
   Solver *J_solver;

public:
   SundialsJacSolver()
      : M(), S(), H(), local_grad_H(), Jacobian(), J_solver() { }

   /// Connect the solver to the objects created inside HyperelasticOperator.
   void SetOperators(ParBilinearForm &M_, ParBilinearForm &S_,
                     ParNonlinearForm &H_, Solver &solver)
   {
      M = &M_; S = &S_; H = &H_; J_solver = &solver;
   }

   /** Linear solve applicable to the SUNDIALS format.
       Solves (Mass - dt J) y = Mass b, where in our case:
       Mass = | M  0 |  J = | -S  -grad_H |  y = | v_hat |  b = | b_v |
              | 0  I |      |  I     0    |      | x_hat |      | b_x |
       The result replaces the rhs b.
       We substitute x_hat = b_x + dt v_hat and solve
       (M + dt S + dt^2 grad_H) v_hat = M b_v - dt grad_H b_x. */
   int InitSystem(void *sundials_mem);
   int SetupSystem(void *sundials_mem, int conv_fail,
                   const Vector &y_pred, const Vector &f_pred, int &jac_cur,
                   Vector &v_temp1, Vector &v_temp2, Vector &v_temp3);
   int SolveSystem(void *sundials_mem, Vector &b, const Vector &weight,
                   const Vector &y_cur, const Vector &f_cur);
   int FreeSystem(void *sundials_mem);
};


/** Function representing the elastic energy density for the given hyperelastic
    model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient : public Coefficient
{
private:
   HyperelasticModel &model;
   ParGridFunction   &x;
   DenseMatrix        J;

public:
   ElasticEnergyCoefficient(HyperelasticModel &m, ParGridFunction &x_)
      : model(m), x(x_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
   virtual ~ElasticEnergyCoefficient() { }
};

void InitialDeformation(const Vector &x, Vector &y);

void InitialVelocity(const Vector &x, Vector &v);

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

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
   const char *nls = "newton";
   int vis_steps = 1;

   // Relative and absolute tolerances for CVODE and ARKODE.
   const double reltol = 1e-1, abstol = 1e-1;

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
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "            4 - CVODE implicit, approximate Jacobian,\n\t"
                  "            5 - CVODE implicit, specified Jacobian,\n\t"
                  "            6 - ARKODE implicit, approximate Jacobian,\n\t"
                  "            7 - ARKODE implicit, specified Jacobian,\n\t"
                  "            11 - Forward Euler, 12 - RK2,\n\t"
                  "            13 - RK3 SSP, 14 - RK4,\n\t"
                  "            15 - CVODE (adaptive order) explicit,\n\t"
                  "            16 - ARKODE default (4th order) explicit.");
   args.AddOption(&nls, "-nls", "--nonlinear-solver",
                  "Nonlinear systems solver: "
                  "\"newton\" (plain Newton) or \"kinsol\" (KINSOL).");
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
      MPI_Finalize();
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   CVODESolver *cvode = NULL;
   ARKODESolver *arkode = NULL;
   SundialsJacSolver *sjsolver = NULL;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1: ode_solver = new BackwardEulerSolver; break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK33Solver; break;
      case 4:
      case 5:
         cvode = new CVODESolver(MPI_COMM_WORLD, CV_BDF, CV_NEWTON);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         if (ode_solver_type == 5)
         {
            sjsolver = new SundialsJacSolver;
            cvode->SetLinearSolver(*sjsolver); // Custom Jacobian inversion.
         }
         ode_solver = cvode; break;
      case 6:
      case 7:
         arkode = new ARKODESolver(MPI_COMM_WORLD, ARKODESolver::IMPLICIT);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         if (ode_solver_type == 7)
         {
            sjsolver = new SundialsJacSolver;
            arkode->SetLinearSolver(*sjsolver); // Custom Jacobian inversion.
         }
         ode_solver = arkode; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      case 15:
         cvode = new CVODESolver(MPI_COMM_WORLD, CV_ADAMS, CV_FUNCTIONAL);
         cvode->SetSStolerances(reltol, abstol);
         cvode->SetMaxStep(dt);
         ode_solver = cvode; break;
      case 16:
         arkode = new ARKODESolver(MPI_COMM_WORLD, ARKODESolver::EXPLICIT);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         ode_solver = arkode; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         MPI_Finalize();
         return 3;
   }

   map<string,HyperelasticOperator::NonlinearSolverType> nls_map;
   nls_map["newton"] = HyperelasticOperator::NEWTON;
   nls_map["kinsol"] = HyperelasticOperator::KINSOL;
   if (nls_map.find(nls) == nls_map.end())
   {
      if (myid == 0)
      {
         cout << "Unknown type of nonlinear solver: " << nls << endl;
      }
      delete ode_solver;
      delete mesh;
      MPI_Finalize();
      return 4;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the parallel vector finite element spaces representing the mesh
   //    deformation x_gf, the velocity v_gf, and the initial configuration,
   //    x_ref. Define also the elastic energy density, w_gf, which is in a
   //    discontinuous higher-order space. Since x and v are integrated in time
   //    as a system, we group them together in block vector vx, on the unique
   //    parallel degrees of freedom, with offsets given by array true_offset.
   H1_FECollection fe_coll(order, dim);
   ParFiniteElementSpace fespace(pmesh, &fe_coll, dim);

   HYPRE_Int glob_size = fespace.GlobalTrueVSize();
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
   ParGridFunction v_gf(&fespace), x_gf(&fespace);

   ParGridFunction x_ref(&fespace);
   pmesh->GetNodes(x_ref);

   L2_FECollection w_fec(order + 1, dim);
   ParFiniteElementSpace w_fespace(pmesh, &w_fec);
   ParGridFunction w_gf(&w_fespace);

   // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
   //    boundary conditions on a beam-like mesh (see description above).
   VectorFunctionCoefficient velo(dim, InitialVelocity);
   v_gf.ProjectCoefficient(velo);
   VectorFunctionCoefficient deform(dim, InitialDeformation);
   x_gf.ProjectCoefficient(deform);

   v_gf.GetTrueDofs(vx.GetBlock(0));
   x_gf.GetTrueDofs(vx.GetBlock(1));

   Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

   // 9. Initialize the hyperelastic operator, the GLVis visualization and print
   //    the initial energies.
   HyperelasticOperator oper(fespace, ess_bdr, visc, mu, K, nls_map[nls]);

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
         oper.GetElasticEnergyDensity(x_gf, w_gf);
         vis_w.precision(8);
         visualize(vis_w, pmesh, &x_gf, &w_gf, "Elastic energy density", true);
      }
   }

   double ee0 = oper.ElasticEnergy(x_gf);
   double ke0 = oper.KineticEnergy(v_gf);
   if (myid == 0)
   {
      cout << "initial elastic energy (EE) = " << ee0 << endl;
      cout << "initial kinetic energy (KE) = " << ke0 << endl;
      cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;
   }

   double t = 0.0;
   oper.SetTime(t);
   ode_solver->Init(oper);

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
         v_gf.Distribute(vx.GetBlock(0));
         x_gf.Distribute(vx.GetBlock(1));

         double ee = oper.ElasticEnergy(x_gf);
         double ke = oper.KineticEnergy(v_gf);

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
               oper.GetElasticEnergyDensity(x_gf, w_gf);
               visualize(vis_w, pmesh, &x_gf, &w_gf);
            }
         }
      }
   }

   // 11. Save the displaced mesh, the velocity and elastic energy.
   {
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
      oper.GetElasticEnergyDensity(x_gf, w_gf);
      w_gf.Save(ee_ofs);
   }

   // 12. Free the used memory.
   delete ode_solver;
   delete sjsolver;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

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


ReducedSystemOperator::ReducedSystemOperator(
   ParBilinearForm *M_, ParBilinearForm *S_, ParNonlinearForm *H_)
   : Operator(M_->ParFESpace()->TrueVSize()), M(M_), S(S_), H(H_),
     Jacobian(NULL), dt(0.0), v(NULL), x(NULL), w(height), z(height)
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
   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Jacobian;
}


int SundialsJacSolver::InitSystem(void *sundials_mem)
{
   TimeDependentOperator *td_oper = GetTimeDependentOperator(sundials_mem);
   HyperelasticOperator *he_oper;

   // During development, we use dynamic_cast<> to ensure the setup is correct:
   he_oper = dynamic_cast<HyperelasticOperator*>(td_oper);
   MFEM_VERIFY(he_oper, "operator is not HyperelasticOperator");

   // When the implementation is finalized, we can switch to static_cast<>:
   // he_oper = static_cast<HyperelasticOperator*>(td_oper);

   he_oper->InitSundialsJacSolver(*this);
   return 0;
}

int SundialsJacSolver::SetupSystem(void *sundials_mem, int conv_fail,
                                   const Vector &y_pred, const Vector &f_pred,
                                   int &jac_cur, Vector &v_temp1,
                                   Vector &v_temp2, Vector &v_temp3)
{
   int sc = y_pred.Size() / 2;
   const Vector x(y_pred.GetData() + sc, sc);
   double dt = GetTimeStep(sundials_mem);

   // J = M + dt*(S + dt*grad(H))
   delete Jacobian;
   SparseMatrix *localJ = Add(1.0, M->SpMat(), dt, S->SpMat());
   local_grad_H = &H->GetLocalGradient(x);
   localJ->Add(dt*dt, *local_grad_H);
   Jacobian = M->ParallelAssemble(localJ);
   delete localJ;

   J_solver->SetOperator(*Jacobian);

   jac_cur = 1;
   return 0;
}

int SundialsJacSolver::SolveSystem(void *sundials_mem, Vector &b,
                                   const Vector &weight, const Vector &y_cur,
                                   const Vector &f_cur)
{
   int sc = b.Size() / 2;
   ParFiniteElementSpace *fes = H->ParFESpace();
   // Vector x(y_cur.GetData() + sc, sc);
   Vector b_v(b.GetData() +  0, sc);
   Vector b_x(b.GetData() + sc, sc);
   Vector rhs(sc);
   double dt = GetTimeStep(sundials_mem);

   // rhs = M b_v - dt*grad(H) b_x
   ParGridFunction lb_x(fes), lrhs(fes);
   lb_x.Distribute(b_x);
   local_grad_H->Mult(lb_x, lrhs);
   lrhs.ParallelAssemble(rhs);
   rhs *= -dt;
   M->TrueAddMult(b_v, rhs);

   J_solver->iterative_mode = false;
   J_solver->Mult(rhs, b_v);

   b_x.Add(dt, b_v);

   return 0;
}

int SundialsJacSolver::FreeSystem(void *sundials_mem)
{
   delete Jacobian;
   return 0;
}


HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace &f,
                                           Array<int> &ess_bdr, double visc,
                                           double mu, double K,
                                           NonlinearSolverType nls_type)
   : TimeDependentOperator(2*f.TrueVSize(), 0.0), fespace(f),
     M(&fespace), S(&fespace), H(&fespace),
     viscosity(visc), M_solver(f.GetComm()), z(height/2)
{
   const double rel_tol = 1e-8;
   const int skip_zero_entries = 0;

   const double ref_density = 1.0; // density in the reference configuration
   ConstantCoefficient rho0(ref_density);
   M.AddDomainIntegrator(new VectorMassIntegrator(rho0));
   M.Assemble(skip_zero_entries);
   M.EliminateEssentialBC(ess_bdr);
   M.Finalize(skip_zero_entries);
   Mmat = M.ParallelAssemble();

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
   H.SetEssentialBC(ess_bdr);

   ConstantCoefficient visc_coeff(viscosity);
   S.AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
   S.Assemble(skip_zero_entries);
   S.EliminateEssentialBC(ess_bdr);
   S.Finalize(skip_zero_entries);

   reduced_oper = new ReducedSystemOperator(&M, &S, &H);

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

   if (nls_type == KINSOL)
   {
      KinSolver *kinsolver = new KinSolver(f.GetComm(), KIN_NONE, true);
      kinsolver->SetMaxSetupCalls(4);
      newton_solver = kinsolver;
      newton_solver->SetMaxIter(200);
      newton_solver->SetRelTol(rel_tol);
      newton_solver->SetPrintLevel(0);
   }
   else
   {
      newton_solver = new NewtonSolver(f.GetComm());
      newton_solver->SetMaxIter(10);
      newton_solver->SetRelTol(rel_tol);
      newton_solver->SetPrintLevel(-1);
   }
   newton_solver->SetSolver(*J_solver);
   newton_solver->iterative_mode = false;
   newton_solver->SetOperator(*reduced_oper);
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

void HyperelasticOperator::InitSundialsJacSolver(SundialsJacSolver &sjsolv)
{
   sjsolv.SetOperators(M, S, H, *J_solver);
}

double HyperelasticOperator::ElasticEnergy(ParGridFunction &x) const
{
   return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(ParGridFunction &v) const
{
   double loc_energy = 0.5*M.InnerProduct(v, v);
   double energy;
   MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM,
                 fespace.GetComm());
   return energy;
}

void HyperelasticOperator::GetElasticEnergyDensity(
   ParGridFunction &x, ParGridFunction &w) const
{
   ElasticEnergyCoefficient w_coeff(*model, x);
   w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
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
