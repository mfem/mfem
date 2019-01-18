//                         MFEM Example 18 - Parallel Version
//
// Compile with: make ex18
//
// Sample runs:
//
//       mpirun -np 4 ex18p -p 1 -rs 2 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 3 -s 4
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 5 -s 6
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Classes FE_Evolution, RiemannSolver, DomainIntegrator and FaceIntegrator
// shared between the serial and parallel version of the example.
//#include "ex18_diff.hpp"

// Choice for the problem setup. See InitialCondition in ex18.hpp.
static int problem;

// Equation constant parameters.
const int num_equation = 4;
const double specific_heat_ratio = 1.4;
const double gas_constant = 1.0;
static double diffusion_constant = 0.1;

// Maximum characteristic speed (updated by integrators)
static double max_char_speed;

// Initial condition
void InitialCondition(const Vector &x, Vector &y);
/*
// Specialized Coefficient to compute the pressure from the state variables
class PressureCoefficient : public Coefficient
{
private:
   const ParGridFunction &density_;
   const ParGridFunction &momentum_;
   const ParGridFunction &energy_;

   mutable Vector momVec_;

   double specific_heat_ratio_;

public:
   PressureCoefficient(const ParGridFunction &density,
             const ParGridFunction &momentum,
             const ParGridFunction &energy,
             double specific_heat_ratio)
     : density_(density),
       momentum_(momentum),
       energy_(energy),
       momVec_(momentum.VectorDim()),
       specific_heat_ratio_(specific_heat_ratio)
   {}

   double Eval(ElementTransformation &T,
          const IntegrationPoint &ip)
   {
      double density = density_.GetValue(T.ElementNo, T.GetIntPoint());
      momentum_.GetVectorValue(T.ElementNo, T.GetIntPoint(), momVec_);
      double energy = energy_.GetValue(T.ElementNo, T.GetIntPoint());

      return (specific_heat_ratio_ - 1.0) *
   (energy - 0.5 * (momVec_ * momVec_) / density);
   }
};
*/
/*
// Specialized VectorCoefficient to compute one component of the momentum flux
class MomentumFluxCoefficient : public VectorCoefficient
{
private:
   const ParGridFunction &density_;
   const ParGridFunction &momentum_;
   const ParGridFunction &pressure_;

   mutable Vector momVec_;
   int comp_;

public:
   MomentumFluxCoefficient(const ParGridFunction &density,
            const ParGridFunction &momentum,
            const ParGridFunction &pressure)
     : VectorCoefficient(momentum.VectorDim()),
       density_(density),
       momentum_(momentum),
       pressure_(pressure),
       momVec_(momentum.VectorDim()),
       comp_(0)
   {}

   void SetComponent(int comp) { comp_ = comp; }

   void Eval(Vector & V, ElementTransformation &T,
        const IntegrationPoint &ip)
   {
      double density  = density_.GetValue(T.ElementNo, T.GetIntPoint());
      momentum_.GetVectorValue(T.ElementNo, T.GetIntPoint(), momVec_);
      double pressure = pressure_.GetValue(T.ElementNo, T.GetIntPoint());

      V = momVec_;
      V *= momVec_[comp_] / density;
      V[comp_] += pressure;
   }
};
*/
/*
// Specialized VectorCoefficient to compute the energy flux
class EnergyFluxCoefficient : public VectorCoefficient
{
private:
   const ParGridFunction &density_;
   const ParGridFunction &momentum_;
   const ParGridFunction &energy_;
   const ParGridFunction &pressure_;

public:
   EnergyFluxCoefficient(const ParGridFunction &density,
          const ParGridFunction &momentum,
          const ParGridFunction &energy,
          const ParGridFunction &pressure)
     : VectorCoefficient(momentum.VectorDim()),
       density_(density),
       momentum_(momentum),
       energy_(energy),
       pressure_(pressure)
   {}

   void Eval(Vector & V, ElementTransformation &T,
        const IntegrationPoint &ip)
   {
      double density  = density_.GetValue(T.ElementNo, T.GetIntPoint());
      momentum_.GetVectorValue(T.ElementNo, T.GetIntPoint(), V);
      double energy   = energy_.GetValue(T.ElementNo, T.GetIntPoint());
      double pressure = pressure_.GetValue(T.ElementNo, T.GetIntPoint());

      V *= (energy + pressure) / density;
   }
};
*/
// Specialized VectorCoefficient to compute one component of the momentum flux
class DiffusionFluxCoefficient : public VectorCoefficient
{
private:
   const Coefficient & nuCoef_;
   GradientGridFunctionCoefficient gradCoef_;

public:
   DiffusionFluxCoefficient(int dim, const Coefficient & nuCoef,
                            ParGridFunction *momentum_comp)
      : VectorCoefficient(dim),
        nuCoef_(nuCoef),
        gradCoef_(momentum_comp)
   {}

   void Eval(Vector & V, ElementTransformation &T,
             const IntegrationPoint &ip)
   {
      const_cast<GradientGridFunctionCoefficient&>(gradCoef_).Eval(V, T, ip);
      V *= -1.0 * const_cast<Coefficient&>(nuCoef_).Eval(T, ip);
   }
};

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class FE_Evolution : public TimeDependentOperator
{
private:
   const int dim_;

   ParFiniteElementSpace &fes_;
   ParFiniteElementSpace &dfes_;
   ParFiniteElementSpace &vfes_;
   Operator &A_;
   SparseMatrix &Aflux_;
   // SparseMatrix &Diff;
   DenseTensor Me_inv_;
   DenseTensor Me_;
   DenseTensor De_;
   DenseMatrix Ae_;
   DenseMatrixInverse inv_;

   ConstantCoefficient nuCoef_;
   // PressureCoefficient PCoef_; // Computes pressure from state variables

   mutable ParGridFunction P_; // Pressure
   // mutable vector<ParGridFunction> momentum_comp_;
   mutable vector<ParGridFunction> gradMom_;

   mutable Vector state_;
   mutable DenseMatrix f_;
   mutable DenseTensor flux_;
   mutable Vector z_;

   void GetFlux(const DenseMatrix &state, DenseTensor &flux) const;

public:
   FE_Evolution(ParFiniteElementSpace &_fes, ParFiniteElementSpace &_dfes,
                ParFiniteElementSpace &_vfes,
                Operator &_A, SparseMatrix &_Aflux/*, SparseMatrix &_Diff*/);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y);

   virtual ~FE_Evolution() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   Vector flux1;
   Vector flux2;

public:
   RiemannSolver();
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};


// Constant (in time) mixed bilinear form multiplying the flux grid function.
// The form is (vec(v), grad(w)) where the trial space = vector L2 space (mesh
// dim) and test space = scalar L2 space.
class DomainIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape;
   DenseMatrix dshapedr;
   DenseMatrix dshapedx;

public:
   DomainIntegrator();

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Tr,
                                       DenseMatrix &elmat);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   RiemannSolver rsolver_;
   Vector shape1_;
   Vector shape2_;
   DenseMatrix dshape1_;
   DenseMatrix dshape2_;
   Vector state1_;
   Vector state2_;
   Vector funval1_;
   Vector funval2_;
   DenseMatrix dfunval1_;
   DenseMatrix dfunval2_;
   Vector nor_;
   Vector fluxN_;
   IntegrationPoint eip1_;
   IntegrationPoint eip2_;
   int dim_;

public:
   FaceIntegrator(RiemannSolver &rsolver, const int dim);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi(argc, argv);

   // 2. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 3;
   int ode_solver_type = 3;
   double t_final = -1.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: Implicit L-stable methods\n\t"
                  "            1 - Backward Euler,\n\t"
                  "            2 - SDIRK23, 3 - SDIRK33,\n\t"
                  "            Implicit A-stable methods (not L-stable)\n\t"
                  "            22 - ImplicitMidPointSolver,\n\t"
                  "            23 - SDIRK23, 34 - SDIRK34.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&diffusion_constant, "-nu", "--diffusion-constant",
                  "Diffusion constant used in momentum equation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   if (t_final < 0.0)
   {
      if (strcmp(mesh_file, "../data/periodic-hexagon.mesh") == 0)
      {
         t_final = 3.0;
      }
      else if (strcmp(mesh_file, "../data/periodic-square.mesh") == 0)
      {
         t_final = 2.0;
      }
      else
      {
         t_final = 1.0;
      }
   }

   // 3. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");

   // 4. Define the ODE solver used for time integration. Several implicit
   //    methods are available, including singly diagonal implicit Runge-Kutta
   //    (SDIRK).
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1:  ode_solver = new BackwardEulerSolver; break;
      case 2:  ode_solver = new SDIRK23Solver(2); break;
      case 3:  ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 34: ode_solver = new SDIRK34Solver; break;
      // Explicit mthods
      case 4: ode_solver = new RK4Solver; break;
      default:
         if (mpi.Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 7. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   ParFiniteElementSpace dfes(&pmesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   ParFiniteElementSpace vfes(&pmesh, &fec, num_equation, Ordering::byNODES);

   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   HYPRE_Int glob_size = vfes.GlobalTrueVSize();
   if (mpi.Root()) { cout << "Number of unknowns: " << glob_size << endl; }

   // 8. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * vfes.GetNDofs(); }
   BlockVector u_block(offsets);

   // Momentum grid function on dfes for visualization.
   ParGridFunction mom(&dfes, u_block.GetData() + offsets[1]);

   // Initialize the state.
   VectorFunctionCoefficient u0(num_equation, InitialCondition);
   ParGridFunction sol(&vfes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "vortex-mesh." << setfill('0') << setw(6) << mpi.WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equation; k++)
      {
         ParGridFunction uk(&fes, u_block.GetBlock(k));
         ostringstream sol_name;
         sol_name << "vortex-" << k << "-init."
                  << setfill('0') << setw(6) << mpi.WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 9. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   // ConstantCoefficient nuCoef(diffusion_constant);

   MixedBilinearForm Aflux(&dfes, &fes);
   Aflux.AddDomainIntegrator(new DomainIntegrator);
   Aflux.Assemble();
   /*
   BilinearForm Diff(&fes);
   Diff.AddDomainIntegrator(new DiffusionIntegrator(nuCoef));
   Diff.Assemble();
   */
   ParNonlinearForm A(&vfes);
   RiemannSolver rsolver;
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));

   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   FE_Evolution euler(fes, dfes, vfes, A, Aflux.SpMat()/*, Diff.SpMat()*/);

   // Visualize the density
   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      MPI_Barrier(pmesh.GetComm());
      sout.open(vishost, visport);
      if (!sout)
      {
         if (mpi.Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (mpi.Root()) { cout << "GLVis visualization disabled.\n"; }
      }
      else
      {
         sout << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank() << "\n";
         sout.precision(precision);
         sout << "solution\n" << pmesh << mom;
         sout << "pause\n";
         sout << flush;
         if (mpi.Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // Determine the minimum element size.
   double hmin;
   if (cfl > 0)
   {
      double my_hmin = pmesh.GetElementSize(0, 1);
      for (int i = 1; i < pmesh.GetNE(); i++)
      {
         my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
      }
      // Reduce to find the global minimum element size
      MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   euler.SetTime(t);
   ode_solver->Init(euler);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      max_char_speed = 0.;
      Vector z(sol.Size());
      A.Mult(sol, z);
      // Reduce to find the global maximum wave speed
      {
         double all_max_char_speed;
         MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                       1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         max_char_speed = all_max_char_speed;
      }
      dt = cfl * hmin / max_char_speed / (2*order+1);
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0)
      {
         // Reduce to find the global maximum wave speed
         {
            double all_max_char_speed;
            MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                          1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
            max_char_speed = all_max_char_speed;
         }
         dt = cfl * hmin / max_char_speed / (2*order+1);
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         if (mpi.Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }
         if (visualization)
         {
            MPI_Barrier(pmesh.GetComm());
            sout << "parallel " << mpi.WorldSize() << " " << mpi.WorldRank() << "\n";
            sout << "solution\n" << pmesh << mom << flush;
         }
      }
   }

   tic_toc.Stop();
   if (mpi.Root()) { cout << " done, " << tic_toc.RealTime() << "s." << endl; }

   // 11. Save the final solution. This output can be viewed later using GLVis:
   //     "glvis -np 4 -m vortex-mesh -g vortex-1-final".
   for (int k = 0; k < num_equation; k++)
   {
      ParGridFunction uk(&fes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "vortex-" << k << "-final."
               << setfill('0') << setw(6) << mpi.WorldRank();
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 12. Compute the L2 solution error summed for all components.
   if ((t_final == 2.0 &&
        strcmp(mesh_file, "../data/periodic-square.mesh") == 0) ||
       (t_final == 3.0 &&
        strcmp(mesh_file, "../data/periodic-hexagon.mesh") == 0))
   {
      const double error = sol.ComputeLpError(2, u0);
      if (mpi.Root()) { cout << "Solution error: " << error << endl; }
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(ParFiniteElementSpace &_fes,
                           ParFiniteElementSpace &_dfes,
                           ParFiniteElementSpace &_vfes,
                           Operator &_A, SparseMatrix &_Aflux/*,
                            SparseMatrix &_Diff*/)
   : TimeDependentOperator(_A.Height(), 0.0, IMPLICIT),
     dim_(_vfes.GetFE(0)->GetDim()),
     fes_(_fes),
     dfes_(_dfes),
     vfes_(_vfes),
     A_(_A),
     Aflux_(_Aflux),
     // Diff(_Diff),
     Me_inv_(vfes_.GetFE(0)->GetDof(), vfes_.GetFE(0)->GetDof(), vfes_.GetNE()),
     Me_(vfes_.GetFE(0)->GetDof(), vfes_.GetFE(0)->GetDof(), vfes_.GetNE()),
     De_(vfes_.GetFE(0)->GetDof(), vfes_.GetFE(0)->GetDof(), vfes_.GetNE()),
     nuCoef_(diffusion_constant),
     P_(&fes_),
     gradMom_(dim_),
     state_(num_equation + dim_ * dim_),
     f_(num_equation, dim_),
     flux_(vfes_.GetNDofs(), dim_, num_equation),
     z_(A_.Height())
{
   for (int d=0; d<dim_; d++)
   {
      gradMom_[d].SetSpace(&dfes_);
   }

   // Standard local assembly and inversion for energy mass matrices.
   MassIntegrator mi;
   DiffusionIntegrator di(nuCoef_);
   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      di.AssembleElementMatrix(*vfes_.GetFE(i),
                               *vfes_.GetElementTransformation(i), De_(i));

      mi.AssembleElementMatrix(*vfes_.GetFE(i),
                               *vfes_.GetElementTransformation(i), Me_(i));

      inv_.Factor(Me_(i));
      inv_.GetInverseMatrix(Me_inv_(i));
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A_.Mult(x, z_);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes_.GetNDofs(), num_equation);
   GetFlux(xmat, flux_);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux_(k).GetData(), dim_ * vfes_.GetNDofs());
      Vector zk(z_.GetData() + k * vfes_.GetNDofs(), vfes_.GetNDofs());
      Aflux_.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes_.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes_.GetElementVDofs(i, vdofs);
      z_.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      mfem::Mult(Me_inv_(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &y)
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A_.Mult(x, z_);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes_.GetNDofs(), num_equation);
   GetFlux(xmat, flux_);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux_(k).GetData(), dim_ * vfes_.GetNDofs());
      Vector zk(z_.GetData() + k * vfes_.GetNDofs(), vfes_.GetNDofs());
      Aflux_.AddMult(fk, zk);
      /*
      if (k > 0 && k < num_equation - 1)
      {
      Vector xk(x.GetData() + k * vfes.GetNDofs(), vfes.GetNDofs());
      Diff.AddMult(xk, zk);
      }
      */
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes_.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int i = 0; i < vfes_.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes_.GetElementVDofs(i, vdofs);
      z_.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      // mfem::Mult(Me_inv(i), zmat, ymat);
      Me_inv_(i).Mult(zmat.GetColumn(0), ymat.GetColumn(0));

      Ae_.SetSize(Me_(i).Width());
      Add(Me_(i), De_(i), dt, Ae_);
      inv_.Factor(Ae_);
      inv_.Mult(zmat.GetColumn(1), ymat.GetColumn(1));
      inv_.Mult(zmat.GetColumn(2), ymat.GetColumn(2));

      Me_inv_(i).Mult(zmat.GetColumn(3), ymat.GetColumn(3));

      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim);

// Pressure (EOS) computation
inline double ComputePressure(const Vector &state, int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   return (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);
}

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);
   const DenseMatrix dmom(const_cast<double*>(&state[dim + 2]), dim, dim);

   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   for (int d = 0; d < dim; d++)
   {
      flux(0, d) = den_vel(d);
      for (int i = 0; i < dim; i++)
      {
         flux(1+i, d) = den_vel(i) * den_vel(d) / den + dmom(i, d);
      }
      flux(1+d, d) += pres;
   }

   const double H = (den_energy + pres) / den;
   for (int d = 0; d < dim; d++)
   {
      flux(1+dim, d) = den_vel(d) * H;
   }
}

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);
   const DenseMatrix dmom(const_cast<double*>(&state[dim + 2]), dim, dim);

   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, dim);

   double den_velN = 0;
   for (int d = 0; d < dim; d++) { den_velN += den_vel(d) * nor(d); }

   fluxN(0) = den_velN;
   for (int d = 0; d < dim; d++)
   {
      fluxN(1+d) = den_velN * den_vel(d) / den + pres * nor(d);
      for (int i=0; i<dim; i++)
      {
         fluxN(1+d) += dmom(i, d) * nor(i);
      }
   }

   const double H = (den_energy + pres) / den;
   fluxN(1 + dim) = den_velN * H;
}

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   const double pres = ComputePressure(state, dim);
   const double sound = sqrt(specific_heat_ratio * pres / den);
   const double vel = sqrt(den_vel2 / den);

   return vel + sound;
}

// Compute the flux at solution nodes.
void FE_Evolution::GetFlux(const DenseMatrix &x, DenseTensor &flux) const
{
   const int dof = flux.SizeI();
   const int dim = flux.SizeJ();

   for (int d=0; d<dim_; d++)
   {
      ParGridFunction
      momentum_comp(&fes_, const_cast<double*>(x.GetColumn(1)) + d * dof);
      DiffusionFluxCoefficient diffFluxCoef(dim_, nuCoef_, &momentum_comp);
      gradMom_[d].ProjectCoefficient(diffFluxCoef);
   }

   for (int i = 0; i < dof; i++)
   {
      for (int k = 0; k < num_equation; k++) { state_(k) = x(i, k); }
      for (int di=0; di<dim_; di++)
      {
         for (int dj=0; dj<dim_; dj++)
         {
            state_[num_equation + di * dim_ + dj] = gradMom_[di][dj * dof + i];
         }
      }
      ComputeFlux(state_, dim, f_);

      for (int d = 0; d < dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux(i, d, k) = f_(k, d);
         }
      }

      // Update max char speed
      const double mcs = ComputeMaxCharSpeed(state_, dim);
      if (mcs > max_char_speed) { max_char_speed = mcs; }
   }
   /*
    // Create ParGridFunctions around each component of x
    ParGridFunction density(&fes_, const_cast<double*>(x.GetColumn(0)));
    ParGridFunction momentum(&dfes_, const_cast<double*>(x.GetColumn(1)));
    ParGridFunction energy(&fes_, const_cast<double*>(x.GetColumn(dim_ + 1)));

    // Update the pressure field
    PressureCoefficient PCoef(density, momentum, energy, specific_heat_ratio);
    P_.ProjectCoefficient(PCoef);

    // Create ParGridFunctions around each component of flux
    ParGridFunction denFlux(&dfes_, flux.GetData(0));
    vector<ParGridFunction> momFlux(dim_);
    for (int d=0; d<dim_; d++)
    {
      momFlux[d].MakeRef(&dfes_, flux.GetData(d + 1));
    }
    ParGridFunction engFlux(&dfes_, flux.GetData(dim_ + 1));

    // Update the density flux using the momentum
    denFlux = momentum;

    // Update the momentum flux using specialized coefficients
    MomentumFluxCoefficient MCoef(density, momentum, P_);
    for (int d=0; d<dim_; d++)
    {
       MCoef.SetComponent(d);
       momFlux[d].ProjectCoefficient(MCoef);
    }

    // Update the energy flux using a specialized coefficient
    EnergyFluxCoefficient EFCoef(density, momentum, energy, P_);
    engFlux.ProjectCoefficient(EFCoef);
   */
}

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver() :
   flux1(num_equation),
   flux2(num_equation) { }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   MFEM_ASSERT(StateIsPhysical(state1, dim), "");
   MFEM_ASSERT(StateIsPhysical(state2, dim), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, flux1);
   ComputeFluxDotN(state2, nor, flux2);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equation; i++)
   {
      flux(i) = 0.5 * (flux1(i) + flux2(i))
                - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

// Implementation of class DomainIntegrator
DomainIntegrator::DomainIntegrator() { }

void DomainIntegrator::AssembleElementMatrix2(const FiniteElement &trial_fe,
                                              const FiniteElement &test_fe,
                                              ElementTransformation &Tr,
                                              DenseMatrix &elmat)
{
   // Assemble the form (vec(v), grad(w))

   // Trial space = vector L2 space (mesh dim)
   // Test space  = scalar L2 space

   const int dof_trial = trial_fe.GetDof();
   const int dof_test = test_fe.GetDof();
   const int dim = trial_fe.GetDim();

   shape.SetSize(dof_trial);
   dshapedr.SetSize(dof_test, dim);
   dshapedx.SetSize(dof_test, dim);

   elmat.SetSize(dof_test, dof_trial * dim);
   elmat = 0.0;

   const int maxorder = max(trial_fe.GetOrder(), test_fe.GetOrder());
   const int intorder = 2 * maxorder;
   const IntegrationRule *ir = &IntRules.Get(trial_fe.GetGeomType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      // Calculate the shape functions
      trial_fe.CalcShape(ip, shape);
      shape *= ip.weight;

      // Compute the physical gradients of the test functions
      Tr.SetIntPoint(&ip);
      test_fe.CalcDShape(ip, dshapedr);
      Mult(dshapedr, Tr.AdjugateJacobian(), dshapedx);

      for (int d = 0; d < dim; d++)
      {
         for (int j = 0; j < dof_test; j++)
         {
            for (int k = 0; k < dof_trial; k++)
            {
               elmat(j, k + d * dof_trial) += shape(k) * dshapedx(j, d);
            }
         }
      }
   }
}

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver, const int dim) :
   rsolver_(rsolver),
   state1_(num_equation + dim * dim),
   state2_(num_equation + dim * dim),
   funval1_(num_equation),
   funval2_(num_equation),
   dfunval1_(num_equation, dim),
   dfunval2_(num_equation, dim),
   nor_(dim),
   fluxN_(num_equation),
   dim_(dim)
{ }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1_.SetSize(dof1);
   shape2_.SetSize(dof2);

   dshape1_.SetSize(dof1, dim_);
   dshape2_.SetSize(dof2, dim_);

   elvect.SetSize((dof1 + dof2) * num_equation);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation, dof2,
                          num_equation);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation, dof2,
                           num_equation);

   // Integration order calculation from DGTraceIntegrator
   int intorder;
   if (Tr.Elem2No >= 0)
      intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
   else
   {
      intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      intorder++;
   }
   const IntegrationRule *ir = &IntRules.Get(Tr.FaceGeom, intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.Loc1.Transform(ip, eip1_);
      Tr.Loc2.Transform(ip, eip2_);

      // Calculate basis functions on both elements at the face
      el1.CalcShape(eip1_, shape1_);
      el2.CalcShape(eip2_, shape2_);

      el1.CalcDShape(eip1_, dshape1_);
      el2.CalcDShape(eip2_, dshape2_);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1_, funval1_);
      elfun2_mat.MultTranspose(shape2_, funval2_);

      MultAtB(elfun1_mat, dshape1_, dfunval1_);
      MultAtB(elfun2_mat, dshape2_, dfunval2_);

      Tr.Face->SetIntPoint(&ip);

      // Copy values to state vectors
      for (int j=0; j<num_equation; j++)
      {
         state1_[j] = funval1_[j];
         state2_[j] = funval2_[j];
      }
      for (int di=0; di<dim_; di++)
      {
         for (int dj=0; dj<dim_; dj++)
         {
            state1_[num_equation + di * dim_ + dj] = dfunval1_(di + 1, dj);
            state2_[num_equation + di * dim_ + dj] = dfunval2_(di + 1, dj);
         }
      }

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Face->Jacobian(), nor_);
      const double mcs = rsolver_.Eval(state1_, state2_, nor_, fluxN_);

      // Update max char speed
      if (mcs > max_char_speed) { max_char_speed = mcs; }

      fluxN_ *= ip.weight;
      for (int k = 0; k < num_equation; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN_(k) * shape1_(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN_(k) * shape2_(s);
         }
      }
   }
}

// Check that the state is physical - enabled in debug mode
bool StateIsPhysical(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   if (den < 0)
   {
      cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   if (den_energy <= 0)
   {
      cout << "Negative energy: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }

   double den_vel2 = 0;
   for (int i = 0; i < dim; i++) { den_vel2 += den_vel(i) * den_vel(i); }
   den_vel2 /= den;

   const double internal_energy = (den_energy - 0.5 * den_vel2);
   const double pres = (specific_heat_ratio - 1.0) * internal_energy;

   if (pres <= 0)
   {
      cout << "Negative pressure: " << pres << ", state: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   return true;
}

// Initial condition
void InitialCondition(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() == 2, "");

   double radius = 0, Minf = 0, beta = 0;
   if (problem == 1)
   {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
   }
   else if (problem == 2)
   {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
   }
   else
   {
      mfem_error("Cannot recognize problem."
                 "Options are: 1 - fast vortex, 2 - slow vortex");
   }

   const double xc = 0.0, yc = 0.0;

   // Nice units
   const double vel_inf = 1.;
   const double den_inf = 1.;

   // Derive remainder of background state from this and Minf
   const double pres_inf = (den_inf / specific_heat_ratio) * (vel_inf / Minf) *
                           (vel_inf / Minf);
   const double temp_inf = pres_inf / (den_inf * gas_constant);

   double r2rad = 0.0;
   r2rad += (x(0) - xc) * (x(0) - xc);
   r2rad += (x(1) - yc) * (x(1) - yc);
   r2rad /= (radius * radius);

   const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

   const double velX = vel_inf * (1 - beta * (x(1) - yc) / radius * exp(
                                     -0.5 * r2rad));
   const double velY = vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
   const double vel2 = velX * velX + velY * velY;

   const double specific_heat = gas_constant * specific_heat_ratio * shrinv1;
   const double temp = temp_inf - 0.5 * (vel_inf * beta) *
                       (vel_inf * beta) / specific_heat * exp(-r2rad);

   const double den = den_inf * pow(temp/temp_inf, shrinv1);
   const double pres = den * gas_constant * temp;
   const double energy = shrinv1 * pres / den + 0.5 * vel2;

   y(0) = den;
   y(1) = den * velX;
   y(2) = den * velY;
   y(3) = den * energy;
}
