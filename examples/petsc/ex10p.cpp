//                       MFEM Example 10 - Parallel Version
//                              PETSc Modification
//
// Compile with: make ex10p
//
// Sample runs:
//    mpirun -np 4 ex10p -m ../../data/beam-quad.mesh --petscopts rc_ex10p -s 3 -rs 2 -dt 3
//    mpirun -np 4 ex10p -m ../../data/beam-quad-amr.mesh --petscopts rc_ex10p -s 3 -rs 2 -dt 3
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
//               If using PETSc to solve the nonlinear problem, use the option
//               file provided (rc_ex10p) that customizes the
//               Newton-Krylov method.
//
//               We recommend viewing examples 2 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
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
   NewtonSolver newton_solver;

   /// Newton solver for the reduced backward Euler equation (PETSc SNES)
   PetscNonlinearSolver* pnewton_solver;

   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian solve in the Newton method
   Solver *J_prec;

   mutable Vector z; // auxiliary vector

public:
   HyperelasticOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr,
                        double visc, double mu, double K, bool use_petsc);

   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;
   /** Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

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
   int vis_steps = 1;
   bool use_petsc = true;
   const char *petscrc_file = "";

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
                  "            11 - Forward Euler, 12 - RK2,\n\t"
                  "            13 - RK3 SSP, 14 - RK4.");
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
   args.AddOption(&use_petsc, "-usepetsc", "--usepetsc", "-no-petsc",
                  "--no-petsc",
                  "Use or not PETSc to solve the nonlinear system.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");
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

   // 2b. We initialize PETSc
   if (use_petsc)
   {
      MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);
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
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         MPI_Finalize();
         return 3;
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
   ParGridFunction v_gf, x_gf;
   v_gf.MakeTRef(&fespace, vx, true_offset[0]);
   x_gf.MakeTRef(&fespace, vx, true_offset[1]);

   ParGridFunction x_ref(&fespace);
   pmesh->GetNodes(x_ref);

   L2_FECollection w_fec(order + 1, dim);
   ParFiniteElementSpace w_fespace(pmesh, &w_fec);
   ParGridFunction w_gf(&w_fespace);

   // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
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

   // 9. Initialize the hyperelastic operator, the GLVis visualization and print
   //    the initial energies.
   HyperelasticOperator *oper = new HyperelasticOperator(fespace, ess_bdr, visc,
                                                         mu, K, use_petsc);

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

   double t = 0.0;
   oper->SetTime(t);
   ode_solver->Init(*oper);

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
   delete oper;

   // We finalize PETSc
   if (use_petsc) { MFEMFinalizePetsc(); }

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
   // if we are using PETSc, the HypreParCSR jacobian will be converted to
   // PETSc's AIJ on the fly
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
                                           double mu, double K, bool use_petsc)
   : TimeDependentOperator(2*f.TrueVSize(), 0.0), fespace(f),
     M(&fespace), S(&fespace), H(&fespace),
     viscosity(visc), M_solver(f.GetComm()),
     newton_solver(f.GetComm()), pnewton_solver(NULL), z(height/2)
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
   if (!use_petsc)
   {
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

      newton_solver.iterative_mode = false;
      newton_solver.SetSolver(*J_solver);
      newton_solver.SetOperator(*reduced_oper);
      newton_solver.SetPrintLevel(1); // print Newton iterations
      newton_solver.SetRelTol(rel_tol);
      newton_solver.SetAbsTol(0.0);
      newton_solver.SetMaxIter(10);
   }
   else
   {
      // if using PETSc, we create the same solver (NEWTON+MINRES+Jacobi)
      // by command line options (see rc_ex10p)
      J_solver = NULL;
      J_prec = NULL;
      pnewton_solver = new PetscNonlinearSolver(f.GetComm(),
                                                *reduced_oper);
      pnewton_solver->SetPrintLevel(1); // print Newton iterations
      pnewton_solver->SetRelTol(rel_tol);
      pnewton_solver->SetAbsTol(0.0);
      pnewton_solver->SetMaxIter(10);
   }
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
   if (!pnewton_solver)
   {
      newton_solver.Mult(zero, dv_dt);
      MFEM_VERIFY(newton_solver.GetConverged(),
                  "Newton solver did not converge.");
   }
   else
   {
      pnewton_solver->Mult(zero, dv_dt);
      MFEM_VERIFY(pnewton_solver->GetConverged(),
                  "Newton solver did not converge.");
   }
   add(v, dt, dv_dt, dx_dt);
}

double HyperelasticOperator::ElasticEnergy(const ParGridFunction &x) const
{
   return H.GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const ParGridFunction &v) const
{
   double loc_energy = 0.5*M.InnerProduct(v, v);
   double energy;
   MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM,
                 fespace.GetComm());
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
   delete J_solver;
   delete J_prec;
   delete reduced_oper;
   delete model;
   delete Mmat;
   delete pnewton_solver;
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
