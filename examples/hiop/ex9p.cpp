//                       MFEM Example 9 - Parallel Version
//               Nonlinear Constrained Optimization Modification
//
// Compile with: make ex9p
//
// Sample runs:
//    mpirun -np 4 ex9p -m ../../data/periodic-segment.mesh -rs 3 -p 0 -o 2 -dt 0.002 -opt 1
//    mpirun -np 4 ex9p -m ../../data/periodic-segment.mesh -rs 3 -p 0 -o 2 -dt 0.002 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/periodic-square.mesh -p 0 -rs 2 -dt 0.01 -tf 10 -opt 1
//    mpirun -np 4 ex9p -m ../../data/periodic-square.mesh -p 0 -rs 2 -dt 0.01 -tf 10 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/periodic-square.mesh -p 1 -rs 2 -dt 0.005 -tf 9 -opt 1
//    mpirun -np 4 ex9p -m ../../data/periodic-square.mesh -p 1 -rs 2 -dt 0.005 -tf 9 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/amr-quad.mesh -p 1 -rs 1 -dt 0.002 -tf 9 -opt 1
//    mpirun -np 4 ex9p -m ../../data/amr-quad.mesh -p 1 -rs 1 -dt 0.002 -tf 9 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/disc-nurbs.mesh -p 1 -rs 2 -dt 0.005 -tf 9 -opt 1
//    mpirun -np 4 ex9p -m ../../data/disc-nurbs.mesh -p 1 -rs 2 -dt 0.005 -tf 9 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/disc-nurbs.mesh -p 2 -rs 2 -dt 0.01 -tf 9 -opt 1
//    mpirun -np 4 ex9p -m ../../data/disc-nurbs.mesh -p 2 -rs 2 -dt 0.01 -tf 9 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/periodic-square.mesh -p 3 -rs 3 -dt 0.0025 -tf 9 -opt 1
//    mpirun -np 4 ex9p -m ../../data/periodic-square.mesh -p 3 -rs 3 -dt 0.0025 -tf 9 -opt 2
//
//    mpirun -np 4 ex9p -m ../../data/periodic-cube.mesh -p 0 -rs 2 -o 2 -dt 0.02 -tf 8 -opt 1
//    mpirun -np 4 ex9p -m ../../data/periodic-cube.mesh -p 0 -rs 2 -o 2 -dt 0.02 -tf 8 -opt 2

// Description:  This example modifies the standard MFEM ex9 by adding nonlinear
//               constrained optimization capabilities through the SLBQP and
//               HIOP solvers. It demonstrates how a user can define a custom
//               class OptimizationProblem that includes linear/nonlinear
//               equality/inequality constraints. This optimization is applied
//               as post-processing to the solution of the transport equation.
//
//               Description of ex9:
//               This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of explicit
//               ODE time integrators, the definition of periodic boundary
//               conditions through periodic meshes, as well as the use of GLVis
//               for persistent visualization of a time-evolving solution. The
//               saving of time-dependent data files for external visualization
//               with VisIt (visit.llnl.gov) is also illustrated.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#ifndef MFEM_USE_HIOP
#error This example requires that MFEM is built with MFEM_USE_HIOP=YES
#endif

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Nonlinear optimizer.
int optimizer_type;

// Velocity coefficient
bool invert_velocity = false;
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

/// Computes C(x) = sum w_i x_i, where w is a given Vector.
class LinearScaleOperator : public Operator
{
private:
   ParFiniteElementSpace &pfes;
   // Local weights.
   const Vector &w;
   // Gradient for the tdofs.
   mutable DenseMatrix grad;

public:
   LinearScaleOperator(ParFiniteElementSpace &space, const Vector &weight)
      : Operator(1, space.TrueVSize()),
        pfes(space), w(weight), grad(1, width)
   {
      Vector w_glob(width);
      pfes.Dof_TrueDof_Matrix()->MultTranspose(w, w_glob);
      w_glob.HostReadWrite(); // read+write -> can use w_glob(i) (non-const)
      for (int i = 0; i < width; i++) { grad(0, i) = w_glob(i); }
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      Vector x_loc(w.Size());
      pfes.GetProlongationMatrix()->Mult(x, x_loc);
      const double loc_res = w * x_loc;
      MPI_Allreduce(&loc_res, &y(0), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   }

   virtual Operator &GetGradient(const Vector &x) const
   {
      return grad;
   }
};

/// Nonlinear monotone bounded operator to test nonlinear inequality constraints
/// Computes D(x) = tanh(sum(x_i)).
class TanhSumOperator : public Operator
{
private:
   // Gradient for the tdofs.
   mutable DenseMatrix grad;

public:
   TanhSumOperator(ParFiniteElementSpace &space)
      : Operator(1, space.TrueVSize()), grad(1, width) { }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      double sum_loc = x.Sum();
      MPI_Allreduce(&sum_loc, &y(0), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      y(0) = std::tanh(y(0));
   }

   virtual Operator &GetGradient(const Vector &x) const
   {
      double sum_loc = x.Sum();
      double dtanh;
      MPI_Allreduce(&sum_loc, &dtanh, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      dtanh = 1.0 - pow(std::tanh(dtanh), 2);

      for (int i = 0; i < width; i++) { grad(0, i) = dtanh; }
      return grad;
   }
};


/** Monotone and conservative a posteriori correction for transport solutions:
 *  Find x that minimizes 0.5 || x - x_HO ||^2, subject to
 *  sum w_i x_i = mass,
 *  tanh(sum(x_i_min)) <= tanh(sum(x_i)) <= tanh(sum(x_i_max)),
 *  x_i_min <= x_i <= x_i_max,
 */
class OptimizedTransportProblem : public OptimizationProblem
{
private:
   const Vector &x_HO;
   Vector massvec, d_lo, d_hi;
   const LinearScaleOperator LSoper;
   const TanhSumOperator TSoper;

public:
   OptimizedTransportProblem(ParFiniteElementSpace &space,
                             const Vector &xho, const Vector &w, double mass,
                             const Vector &xmin, const Vector &xmax)
      : OptimizationProblem(xho.Size(), NULL, NULL),
        x_HO(xho), massvec(1), d_lo(1), d_hi(1),
        LSoper(space, w), TSoper(space)
   {
      C = &LSoper;
      massvec(0) = mass;
      SetEqualityConstraint(massvec);

      D = &TSoper;
      double lsums[2], gsums[2];
      lsums[0] = xmin.Sum();
      lsums[1] = xmax.Sum();
      MPI_Allreduce(lsums, gsums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      d_lo(0) = std::tanh(gsums[0]);
      d_hi(0) = std::tanh(gsums[1]);
      MFEM_ASSERT(d_lo(0) < d_hi(0),
                  "The bounds produce an infeasible optimization problem");
      SetInequalityConstraint(d_lo, d_hi);

      SetSolutionBounds(xmin, xmax);
   }

   virtual double CalcObjective(const Vector &x) const
   {
      double loc_res = 0.0;
      for (int i = 0; i < input_size; i++)
      {
         const double d = x(i) - x_HO(i);
         loc_res += d * d;
      }
      loc_res *= 0.5;
      double res;
      MPI_Allreduce(&loc_res, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return res;
   }

   virtual void CalcObjectiveGrad(const Vector &x, Vector &grad) const
   {
      for (int i = 0; i < input_size; i++) { grad(i) = x(i) - x_HO(i); }
   }
};


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &K;
   const Vector &b;
   HypreSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;

   double dt;
   ParBilinearForm &pbf;
   Vector &M_rowsums;

public:
   FE_Evolution(HypreParMatrix &M_, HypreParMatrix &K_,
                const Vector &b_, ParBilinearForm &pbf_, Vector &M_rs);

   void SetTimeStep(double dt_) { dt = dt_; }
   void SetK(HypreParMatrix &K_) { K = K_; }
   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 0;
   optimizer_type = 2;
   const char *mesh_file = "../../data/periodic-hexagon.mesh";
   int ser_ref_levels = 2;
   int par_ref_levels = 0;
   int order = 3;
   int ode_solver_type = 3;
   double t_final = 1.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&optimizer_type, "-opt", "--optimizer",
                  "Nonlinear optimizer: 1 - SLBQP,\n\t"
                  "                     2 - HIOP.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) { args.PrintUsage(cout); }
      return 1;
   }
   if (myid == 0) { args.PrintOptions(cout); }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle geometrically periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete mesh;
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter. If the mesh is of NURBS type, we convert it
   //    to a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the parallel discontinuous DG finite element space on the
   //    parallel refined mesh of the given polynomial order.
   DG_FECollection fec(order, dim, BasisType::Positive);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec);

   HYPRE_BigInt global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   // 8. Set up and assemble the parallel bilinear and linear forms (and the
   //    parallel hypre matrices) corresponding to the DG discretization. The
   //    DGTraceIntegrator involves integrals over mesh interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   ParBilinearForm *m = new ParBilinearForm(fes);
   m->AddDomainIntegrator(new MassIntegrator);
   ParBilinearForm *k = new ParBilinearForm(fes);
   k->AddDomainIntegrator(new ConvectionIntegrator(velocity, -1.0));
   k->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));
   k->AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, 1.0, -0.5)));

   ParLinearForm *b = new ParLinearForm(fes);
   b->AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, -1.0, -0.5));

   m->Assemble();
   m->Finalize();
   int skip_zeros = 0;
   k->Assemble(skip_zeros);
   k->Finalize(skip_zeros);
   b->Assemble();

   HypreParMatrix *M = m->ParallelAssemble();
   HypreParMatrix *K = k->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();

   // 9. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   ParGridFunction *u = new ParGridFunction(fes);
   u->ProjectCoefficient(u0);
   HypreParVector *U = u->GetTrueDofs();

   {
      ostringstream mesh_name, sol_name;
      mesh_name << "ex9-mesh." << setfill('0') << setw(6) << myid;
      sol_name << "ex9-init." << setfill('0') << setw(6) << myid;
      ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9-Parallel", pmesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9-Parallel", pmesh);
         dc->SetPrecision(precision);
         // To save the mesh using MFEM's parallel mesh format:
         // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
      }
      dc->RegisterField("solution", u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         if (myid == 0)
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         visualization = false;
         if (myid == 0)
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         sout << "parallel " << num_procs << " " << myid << "\n";
         sout.precision(precision);
         sout << "solution\n" << *pmesh << *u;
         sout << "pause\n";
         sout << flush;
         if (myid == 0)
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   Vector M_rowsums(m->Size());
   m->SpMat().GetRowSums(M_rowsums);

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   FE_Evolution adv(*M, *K, *B, *k, M_rowsums);

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   *u = *U;

   // Compute initial volume.
   const double vol0_loc = M_rowsums * (*u);
   double vol0;
   MPI_Allreduce(&vol0_loc, &vol0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      adv.SetTimeStep(dt_real);
      ode_solver->Step(*U, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         if (myid == 0)
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }

         // 11. Extract the parallel grid function corresponding to the finite
         //     element approximation U (the local solution on each processor).
         *u = *U;

         if (visualization)
         {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout << "solution\n" << *pmesh << *u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      }
   }

   // Print the error vs exact solution.
   const double max_error = u->ComputeMaxError(u0),
                l1_error  = u->ComputeL1Error(u0),
                l2_error  = u->ComputeL2Error(u0);
   if (myid == 0)
   {
      std::cout << "Linf error = " << max_error << endl
                << "L1   error = " << l1_error << endl
                << "L2   error = " << l2_error << endl;
   }

   // Print error in volume.
   const double vol_loc = M_rowsums * (*u);
   double vol;
   MPI_Allreduce(&vol_loc, &vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (myid == 0)
   {
      std::cout << "Vol  error = " << vol - vol0 << endl;
   }

   // 12. Save the final solution in parallel. This output can be viewed later
   //     using GLVis: "glvis -np <np> -m ex9-mesh -g ex9-final".
   {
      *u = *U;
      ostringstream sol_name;
      sol_name << "ex9-final." << setfill('0') << setw(6) << myid;
      ofstream osol(sol_name.str().c_str());
      osol.precision(precision);
      u->Save(osol);
   }

   // 13. Free the used memory.
   delete U;
   delete u;
   delete B;
   delete b;
   delete K;
   delete k;
   delete M;
   delete m;
   delete fes;
   delete pmesh;
   delete ode_solver;
   delete dc;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &M_, HypreParMatrix &K_,
                           const Vector &b_, ParBilinearForm &pbf_,
                           Vector &M_rs)
   : TimeDependentOperator(M_.Height()),
     M(M_), K(K_), b(b_), M_solver(M.GetComm()), z(M_.Height()),
     pbf(pbf_), M_rowsums(M_rs)
{
   M_prec.SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // Get values on the ldofs.
   ParFiniteElementSpace *pfes = pbf.ParFESpace();
   ParGridFunction x_gf(pfes);
   pfes->GetProlongationMatrix()->Mult(x, x_gf);

   // Compute bounds y_min, y_max for y from from x on the ldofs.
   const int ldofs = x_gf.Size();
   Vector y_min(ldofs), y_max(ldofs);
   x_gf.ExchangeFaceNbrData();
   Vector &x_nd = x_gf.FaceNbrData();
   const int *In = pbf.SpMat().GetI(), *Jn = pbf.SpMat().GetJ();
   for (int i = 0, k = 0; i < ldofs; i++)
   {
      double x_i_min = +std::numeric_limits<double>::infinity();
      double x_i_max = -std::numeric_limits<double>::infinity();
      for (int end = In[i+1]; k < end; k++)
      {
         const int j = Jn[k];
         const double x_j = (j < ldofs) ? x(j): x_nd(j-ldofs);

         if (x_j > x_i_max) { x_i_max = x_j; }
         if (x_j < x_i_min) { x_i_min = x_j; }
      }
      y_min(i) = x_i_min;
      y_max(i) = x_i_max;
   }
   for (int i = 0; i < ldofs; i++)
   {
      y_min(i) = (y_min(i) - x_gf(i) ) / dt;
      y_max(i) = (y_max(i) - x_gf(i) ) / dt;
   }
   Vector y_min_tdofs(y.Size()), y_max_tdofs(y.Size());
   // Move the bounds to the tdofs.
   pfes->GetRestrictionMatrix()->Mult(y_min, y_min_tdofs);
   pfes->GetRestrictionMatrix()->Mult(y_max, y_max_tdofs);

   // Compute the high-order solution y = M^{-1} (K x + b) on the tdofs.
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);

   // The solution y is an increment; it should not introduce new mass.
   const double mass_y = 0.0;

   // Perform optimization on the tdofs.
   Vector y_out(y.Size());
   const int max_iter = 500;
   const double rtol = 1.e-7;
   double atol = 1.e-7;

   OptimizationSolver* optsolver = NULL;
   if (optimizer_type == 2)
   {
#ifdef MFEM_USE_HIOP
      HiopNlpOptimizer *tmp_opt_ptr = new HiopNlpOptimizer(MPI_COMM_WORLD);
      optsolver = tmp_opt_ptr;
#else
      MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
   }
   else
   {
      SLBQPOptimizer *slbqp = new SLBQPOptimizer(MPI_COMM_WORLD);
      slbqp->SetBounds(y_min_tdofs, y_max_tdofs);
      slbqp->SetLinearConstraint(M_rowsums, mass_y);
      atol = 1.e-15;
      optsolver = slbqp;
   }

   OptimizedTransportProblem ot_prob(*pfes, y, M_rowsums, mass_y,
                                     y_min_tdofs, y_max_tdofs);
   optsolver->SetOptimizationProblem(ot_prob);

   optsolver->SetMaxIter(max_iter);
   optsolver->SetAbsTol(atol);
   optsolver->SetRelTol(rtol);
   optsolver->SetPrintLevel(0);
   optsolver->Mult(y, y_out);

   y = y_out;

   delete optsolver;
}


// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = (invert_velocity) ? -1.0 : 1.0; break;
            case 2: v(0) = sqrt(2./3.); v(1) = sqrt(1./3.); break;
            case 3: v(0) = sqrt(3./6.); v(1) = sqrt(2./6.); v(2) = sqrt(1./6.);
               break;
         }
         break;
      }
      case 1:
      case 2:
      {
         // Clockwise rotation in 2D around the origin
         const double w = M_PI/2;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = w*X(1); v(1) = -w*X(0); break;
            case 3: v(0) = w*X(1); v(1) = -w*X(0); v(2) = 0.0; break;
         }
         break;
      }
      case 3:
      {
         // Clockwise twisting rotation in 2D around the origin
         const double w = M_PI/2;
         double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
         d = d*d;
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
            case 2: v(0) = d*w*X(1); v(1) = -d*w*X(0); break;
            case 3: v(0) = d*w*X(1); v(1) = -d*w*X(0); v(2) = 0.0; break;
         }
         break;
      }
   }
}

// Initial condition
double u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      double center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
               return (X(0) > -0.15 && X(0) < 0.15) ? 1.0 : 0.0;
            //return exp(-40.*pow(X(0)-0.0,2));
            case 2:
            case 3:
            {
               double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                        erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         double x_ = X(0), y_ = X(1), rho, phi;
         rho = hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const double f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   switch (problem)
   {
      case 0:
      case 1:
      case 2:
      case 3: return 0.0;
   }
   return 0.0;
}
