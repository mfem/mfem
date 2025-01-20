//                                MFEM Example 9
//
// Compile with: make ex9
//
// DG sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.02 -s 13 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ex9 -m ../data/periodic-square.msh -p 0 -r 2 -dt 0.005 -tf 2
//    ex9 -m ../data/periodic-cube.msh -p 0 -r 1 -o 2 -tf 2
//
// CG sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 5 -dt 0.001 -sc 11 -vs 50 -o 1 -s 2
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 5 -dt 0.001 -sc 12 -vs 50 -o 1 -s 2
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 5 -dt 0.001 -sc 13 -vs 50 -o 1 -s 2
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 3 -dt 0.01 -tf 10 -sc 11 -o 2 -s 3 -vs 20
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 3 -dt 0.01 -tf 10 -sc 12 -vs 20
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 4 -dt 0.002 -tf 9 -sc 11 -o 1 -s 2 -vs 20
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.002 -tf 9 -sc 11 -vs 20
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 4 -dt 0.002 -tf 9 -sc 13 -o 1 -s 2 -vs 20
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 4 -dt 0.004 -tf 9 -vs 20 -sc 11 -o 1 -s 2
//    ex9 -m ../data/star-q3.mesh -p 1 -r 4 -dt 0.004 -tf 9 -vs 20 -sc 11 -o 1 -s 2
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9 -sc 11
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 4 -dt 0.005 -tf 9 -sc 12 -o 2 -s 3
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.005 -tf 9 -vs 20 -sc 11 -o 2 -s 3
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -dt 0.02 -tf 8 -sc 11 -o 2 -s 3
//    ex9 -m ../data/periodic-cube.msh -p 0 -r 2 -o 2 -s 3 -tf 2 -sc 11
//
// Device sample runs:
//    ex9 -pa
//    ex9 -ea
//    ex9 -fa
//    ex9 -pa -m ../data/periodic-cube.mesh
//    ex9 -pa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -ea -m ../data/periodic-cube.mesh -d cuda
//    ex9 -fa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -pa -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9 -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. The saving of time-dependent data files for external
//               visualization with VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) is also illustrated.
//               Additionally, the example showcases the implementation of an
//               element-based Clip & Scale limiter for continuous finite
//               elements, which is designed to be bound-preserving.
//               For more detail, see https://doi.org/10.1142/13466.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
real_t u0_function(const Vector &x);

// Inflow boundary condition
real_t inflow_function(const Vector &x);

// Function f = 1 for lumped boundary operator
real_t one(const Vector &x) {return 1.0;}

// Mesh bounding box
Vector bb_min, bb_max;

class DG_Solver : public Solver
{
private:
   SparseMatrix &M, &K, A;
   GMRESSolver linear_solver;
   BlockILU prec;
   real_t dt;
public:
   DG_Solver(SparseMatrix &M_, SparseMatrix &K_, const FiniteElementSpace &fes)
      : M(M_),
        K(K_),
        prec(fes.GetFE(0)->GetDof(),
             BlockILU::Reordering::MINIMUM_DISCARDED_FILL),
        dt(-1.0)
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-9);
      linear_solver.SetAbsTol(0.0);
      linear_solver.SetMaxIter(100);
      linear_solver.SetPrintLevel(0);
      linear_solver.SetPreconditioner(prec);
   }

   void SetTimeStep(real_t dt_)
   {
      if (dt_ != dt)
      {
         dt = dt_;
         // Form operator A = M - dt*K
         A = K;
         A *= -dt;
         A += M;

         // this will also call SetOperator on the preconditioner
         linear_solver.SetOperator(A);
      }
   }

   void SetOperator(const Operator &op)
   {
      linear_solver.SetOperator(op);
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      linear_solver.Mult(x, y);
   }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class DG_FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;

   mutable Vector z;

public:
   DG_FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const real_t dt, const Vector &x, Vector &k);

   virtual ~DG_FE_Evolution();
};

/** Abstract base class for evaluating the time-dependent operator in the ODE
    formulation. The continuous Galerkin (CG) strong form of the advection
    equation du/dt = -v.grad(u) is given by M du/dt = -K u + b, where M and K
    are the mass and advection matrices, respectively, and b represents the
    boundary flow contribution.

    The ODE can be reformulated as:
    du/dt = M_L^{-1}((-K + D) u + F^*(u) + b),
    where M_L is the lumped mass matrix, D is a low-order stabilization term,
    and F^*(u) represents the limited anti-diffusive fluxes.
    Here, F^* is a limited version of F, which recover the high-order target
    scheme. The limited anti-diffusive fluxes F^* are the sum of the limited
    element contributions of the original flux F to enforce local bounds.

    Additional to the limiter we implement the low-order scheme and
    high-order target scheme by chosing:
    - F^* = 0 for the bound-preserving low-order scheme.
    - F^* = F for the high-order target scheme which is not bound-preserving.

    This abstract class provides a framework for evaluating the right-hand side
    of the ODE and is intended to be inherited by classes that implement
    the three schemes:
    - The ClipAndScale class, which employes the limiter to enforces local
      bounds
    - The HighOrderTargetScheme class, which employs the raw anti-diffusive
      fluxes F
    - The LowOrderScheme class, which employs F = 0 and has low accuracy, but
      is bound-preserving */
class CG_FE_Evolution : public TimeDependentOperator
{
protected:
   const Vector &lumpedmassmatrix;
   FiniteElementSpace &fes;
   int *I, *J;
   LinearForm b_lumped;
   GridFunction u_inflow;

   mutable DenseMatrix Ke, Me;
   mutable Vector ue, re, udote, fe, fe_star, gammae;
   mutable ConvectionIntegrator conv_int;
   mutable MassIntegrator mass_int;
   mutable Vector z;

   virtual void ComputeLOTimeDerivatives(const Vector &u, Vector &udot) const;

public:
   CG_FE_Evolution(FiniteElementSpace &fes_,
                   const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                   VectorFunctionCoefficient &velocity,
                   BilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const = 0;

   virtual ~CG_FE_Evolution();
};

// Clip and Scale limiter class
class ClipAndScale : public CG_FE_Evolution
{
private:
   mutable Array<real_t> umin, umax;
   mutable Vector udot;

   virtual void ComputeBounds(const Vector &u, Array<real_t> &u_min,
                              Array<real_t> &u_max) const;

public:
   ClipAndScale(FiniteElementSpace &fes_,
                const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                VectorFunctionCoefficient &velocity, BilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~ClipAndScale();
};

// High-order target scheme class
class HighOrderTargetScheme : public CG_FE_Evolution
{
private:
   mutable Vector udot;

public:
   HighOrderTargetScheme(FiniteElementSpace &fes_,
                         const Vector &lumpedmassmatrix_,
                         FunctionCoefficient &inflow,
                         VectorFunctionCoefficient &velocity, BilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~HighOrderTargetScheme();
};

// Low-order scheme class
class LowOrderScheme : public CG_FE_Evolution
{
public:
   LowOrderScheme(FiniteElementSpace &fes_,
                  const Vector &lumpedmassmatrix_, FunctionCoefficient &inflow,
                  VectorFunctionCoefficient &velocity, BilinearForm &M);

   virtual void Mult(const Vector &x, Vector &y) const override;

   virtual ~LowOrderScheme();
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 4;
   int scheme = 1;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                  "--no-element-assembly", "Enable Element Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            11 - Backward Euler,\n\t"
                  "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                  "            22 - Implicit Midpoint Method,\n\t"
                  "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
   args.AddOption(&scheme, "-sc", "--scheme",
                  "FE scheme: 1 - Standard DG,\n\t"
                  "           11 - Clip and Scale Limiter for CG,\n\t"
                  "           12 - High-order target schme for CG,\n\t"
                  "           13 - Low-order schme for CG.");
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
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                  "--ascii-datafiles",
                  "Use binary (Sidre) or ascii format for VisIt data files.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   const bool DG = (scheme < 11);

   // Limiter is only implemented to run on cpu.
   if (!DG && strcmp(device_config, "cuda") == 0)
   {
      cout << "Cuda not supported for this CG implementation" << endl;
      return 2;
   }
   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // Nonconforming meshes are not feasible for continuous elements
   if (!DG && !mesh.Conforming())
   {
      cout << "CG needs a conforming mesh." << endl;
      return 3;
   }

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   //    The CG Limiter is only implemented for explicit time-stepping methods.
   if (!DG && ode_solver_type > 10)
   {
      cout << "The stabilized CG method is only implemented ";
      cout << "for explicit Runge-Kutta methods."
           << endl;
      return 4;
   }
   // Limiter and low order scheme are only provably bound preserving
   // when employing SSP-RK time-stepping methods
   else if ((scheme == 11 || scheme == 13) && ode_solver_type > 3)
   {
      MFEM_WARNING("Non-SSP-RK mehod! Bounds might be violated.");
   }
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      // Implicit (L-stable) methods
      case 11: ode_solver = new BackwardEulerSolver; break;
      case 12: ode_solver = new SDIRK23Solver(2); break;
      case 13: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;

      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 5;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec_DG(order, dim, BasisType::GaussLobatto);
   H1_FECollection fec_CG(order, dim, BasisType::Positive);

   FiniteElementSpace *fes = NULL;
   switch (scheme)
   {
      case 1:
         fes = new FiniteElementSpace(&mesh, &fec_DG);
         break;
      case 11:
      case 12:
      case 13:
         fes = new FiniteElementSpace(&mesh, &fec_CG);
         break;
      default:
         cout << "Unknown scheme: " << scheme << '\n';
         return 6;
   }

   cout << "Number of unknowns: " << fes->GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);

   BilinearForm m(fes);
   BilinearForm k(fes);
   if (DG)
   {
      if (pa)
      {
         m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
         k.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      else if (ea)
      {
         m.SetAssemblyLevel(AssemblyLevel::ELEMENT);
         k.SetAssemblyLevel(AssemblyLevel::ELEMENT);
      }
      else if (fa)
      {
         m.SetAssemblyLevel(AssemblyLevel::FULL);
         k.SetAssemblyLevel(AssemblyLevel::FULL);
      }
   }
   else if (scheme == 11 && (pa || ea))
   {
      cout << "The CG Limiter needs full assembly of the ";
      cout << "mass matrix to obtain the local stencil via ";
      cout << "its sparsity pattern. " << endl;
      delete fes;
      return 7;
   }

   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();
   m.Finalize();

   constexpr real_t alpha = -1.0;
   int skip_zeros = 0;
   Vector lumpedmassmatrix(m.Height());

   // The convective bilinear form is not needed in the CG case.
   if (DG)
   {
      k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
      k.AddInteriorFaceIntegrator(
         new NonconservativeDGTraceIntegrator(velocity, alpha));
      k.AddBdrFaceIntegrator(
         new NonconservativeDGTraceIntegrator(velocity, alpha));

      k.Assemble(skip_zeros);
      k.Finalize(skip_zeros);
   }
   // lumped mass matrix not needed in the DG case
   else
   {
      BilinearForm mL(fes);
      mL.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
      mL.Assemble();
      mL.Finalize();
      mL.SpMat().GetDiag(lumpedmassmatrix);
   }

   LinearForm b(fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha));
   b.Assemble();

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(fes);
   u.ProjectCoefficient(u0);

   {
      ofstream omesh("ex9.mesh");
      omesh.precision(precision);
      mesh.Print(omesh);
      ofstream osol("ex9-init.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
   DataCollection *dc = NULL;
   if (visit)
   {
      if (binary)
      {
#ifdef MFEM_USE_SIDRE
         dc = new SidreDataCollection("Example9", &mesh);
#else
         MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
      }
      else
      {
         dc = new VisItDataCollection("Example9", &mesh);
         dc->SetPrecision(precision);
      }
      dc->RegisterField("solution", &u);
      dc->SetCycle(0);
      dc->SetTime(0.0);
      dc->Save();
   }

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example9", &mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", &u);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
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
         sout << "solution\n" << mesh << u;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   //DG_FE_Evolution adv(m, k, b);
   TimeDependentOperator *adv = NULL;
   switch (scheme)
   {
      case 1: adv = new DG_FE_Evolution(m, k, b); break;
      case 11: adv = new ClipAndScale(*fes, lumpedmassmatrix, inflow,
                                         velocity, m); break;
      case 12: adv = new HighOrderTargetScheme(*fes, lumpedmassmatrix, inflow,
                                                  velocity, m); break;
      case 13: adv = new LowOrderScheme(*fes, lumpedmassmatrix, inflow,
                                           velocity, m); break;
   }

   real_t t = 0.0;
   adv->SetTime(t);
   ode_solver->Init(*adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete pd;
   delete dc;
   delete fes;
   delete adv;

   return 0;
}


// Implementation of class DG_FE_Evolution
DG_FE_Evolution::DG_FE_Evolution(BilinearForm &M_, BilinearForm &K_,
                                 const Vector &b_)
   : TimeDependentOperator(M_.FESpace()->GetTrueVSize()),
     M(M_), K(K_), b(b_), z(height)
{
   Array<int> ess_tdof_list;
   if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY)
   {
      M_prec = new DSmoother(M.SpMat());
      M_solver.SetOperator(M.SpMat());
      dg_solver = new DG_Solver(M.SpMat(), K.SpMat(), *M.FESpace());
   }
   else
   {
      M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
      M_solver.SetOperator(M);
      dg_solver = NULL;
   }
   M_solver.SetPreconditioner(*M_prec);
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void DG_FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void DG_FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
   MFEM_VERIFY(dg_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   K.Mult(x, z);
   z += b;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
}

DG_FE_Evolution::~DG_FE_Evolution()
{
   delete M_prec;
   delete dg_solver;
}

// Implementation of class CG_FE_Evolution
CG_FE_Evolution::CG_FE_Evolution(FiniteElementSpace &fes_,
                                 const Vector &lumpedmassmatrix_,
                                 FunctionCoefficient &inflow,
                                 VectorFunctionCoefficient &velocity,
                                 BilinearForm &M) :
   TimeDependentOperator(lumpedmassmatrix_.Size()),
   lumpedmassmatrix(lumpedmassmatrix_), fes(fes_),
   I(M.SpMat().GetI()), J(M.SpMat().GetJ()), b_lumped(&fes), u_inflow(&fes),
   conv_int(velocity), mass_int()
{
   u_inflow.ProjectCoefficient(inflow);

   // For bound preservation the boundary condition \hat{u} is enforced
   // via a lumped approximation to < (u_h - u_inflow) * min(v * n, 0 ), w >,
   // i.e., (u_i - (u_inflow)_i) * \int_F \varphi_i * min(v * n, 0).
   // The integral can be implemented as follows:
   FunctionCoefficient one_coeff(one);
   b_lumped.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(one_coeff, velocity, 1.0));
   b_lumped.Assemble();

   z.SetSize(lumpedmassmatrix.Size());
}

void CG_FE_Evolution::ComputeLOTimeDerivatives(const Vector &u,
                                               Vector &udot) const
{
   udot = 0.0;
   const int nE = fes.GetNE();
   Array<int> dofs;

   for (int e = 0; e < nE; e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element matrix of convection operator
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      u.GetSubVector(dofs, ue);
      re.SetSize(dofs.Size());
      re = 0.0;

      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add low-order stabilization with discrete upwinding
            real_t dije = max(max(Ke(i,j), Ke(j,i)), real_t(0.0));
            real_t diffusion = dije * (ue(j) - ue(i));

            re(i) += diffusion;
            re(j) -= diffusion;
         }
      }
      // Add -K_e u_e to obtain (-K_e + D_e) u_e and add element contribution
      // to global vector
      Ke.AddMult(ue, re, -1.0);
      udot.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b.
   // This is under the assumption that b_lumped has been updated
   subtract(u, u_inflow, z);
   z *= b_lumped;
   udot += z;

   // apply inverse lumped mass matrix
   udot /= lumpedmassmatrix;
}

CG_FE_Evolution::~CG_FE_Evolution()
{ }

// Implementation of class ClipAndScale
ClipAndScale::ClipAndScale(FiniteElementSpace &fes_,
                           const Vector &lumpedmassmatrix_,
                           FunctionCoefficient &inflow,
                           VectorFunctionCoefficient &velocity,
                           BilinearForm &M):
   CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
{
   umin.SetSize(lumpedmassmatrix.Size());
   umax.SetSize(lumpedmassmatrix.Size());
   udot.SetSize(lumpedmassmatrix.Size());
}

void ClipAndScale::ComputeBounds(const Vector &u, Array<real_t> &u_min,
                                 Array<real_t> &u_max) const
{
   // iterate over local number of dofs on this processor
   // and compute maximum and minimum over local stencil
   for (int i = 0; i < fes.GetVSize(); i++)
   {
      umin[i] = u(i);
      umax[i] = u(i);

      for (int k = I[i]; k < I[i+1]; k++)
      {
         int j = J[k];
         umin[i] = min(umin[i], u(j));
         umax[i] = max(umax[i], u(j));
      }
   }
}

void ClipAndScale::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;

   // compute low-order time derivative for high-order
   // stabilization and local bounds
   ComputeLOTimeDerivatives(x, udot);
   ComputeBounds(x, umin, umax);

   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element mass and convection matrices
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      re.SetSize(dofs.Size());
      udote.SetSize(dofs.Size());
      fe.SetSize(dofs.Size());
      fe_star.SetSize(dofs.Size());
      gammae.SetSize(dofs.Size());

      x.GetSubVector(dofs, ue);
      udot.GetSubVector(dofs, udote);

      re = 0.0;
      fe = 0.0;
      gammae = 0.0;
      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add low-order diffusion
            real_t dije = max(max(Ke(i,j), Ke(j,i)), real_t(0.0));
            real_t diffusion = dije * (ue(j) - ue(i));

            re(i) += diffusion;
            re(j) -= diffusion;

            // for bounding fluxes
            gammae(i) += dije;
            gammae(j) += dije;

            // assemble raw antidifussive fluxes
            // note that fije = - fjie
            real_t fije = Me(i,j) * (udote(i) - udote(j)) - diffusion;
            fe(i) += fije;
            fe(j) -= fije;
         }
      }

      // add convective term
      Ke.AddMult(ue, re, -1.0);

      gammae *= 2.0;

      real_t P_plus = 0.0;
      real_t P_minus = 0.0;

      //Clip
      for (int i = 0; i < dofs.Size(); i++)
      {
         // bounding fluxes to enforce u_i = u_i_min implies du/dt >= 0
         // and u_i = u_i_max implies du/dt <= 0
         real_t fie_max = gammae(i) * (umax[dofs[i]] - ue(i));
         real_t fie_min = gammae(i) * (umin[dofs[i]] - ue(i));

         fe_star(i) = min(max(fie_min, fe(i)), fie_max);

         // track positive and negative contributions s
         P_plus += max(fe_star(i), real_t(0.0));
         P_minus += min(fe_star(i), real_t(0.0));
      }
      const real_t P = P_minus + P_plus;

      //and Scale for the sum of fe_star to be 0, i.e., mass conservation
      for (int i = 0; i < dofs.Size(); i++)
      {
         if (fe_star(i) > 0.0 && P > 0.0)
         {
            fe_star(i) *= - P_minus / P_plus;
         }
         else if (fe_star(i) < 0.0 && P < 0.0)
         {
            fe_star(i) *= - P_plus / P_minus;
         }
      }
      // add limited antidiffusive fluxes to element contribution
      // and add to global vector
      re += fe_star;
      y.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b
   subtract(x, u_inflow, z);
   z *= b_lumped;
   y += z;

   // apply inverse lumped mass matrix
   y /= lumpedmassmatrix;
}


ClipAndScale::~ClipAndScale()
{ }

// Implementation of class HighOrderTargetScheme
HighOrderTargetScheme::HighOrderTargetScheme(FiniteElementSpace &fes_,
                                             const Vector &lumpedmassmatrix_,
                                             FunctionCoefficient &inflow,
                                             VectorFunctionCoefficient &velocity,
                                             BilinearForm &M):
   CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
{
   udot.SetSize(lumpedmassmatrix.Size());
}

void HighOrderTargetScheme::Mult(const Vector &x, Vector &y) const
{
   y = 0.0;

   // compute low-order time derivative for high-order stabilization
   ComputeLOTimeDerivatives(x, udot);

   Array<int> dofs;
   for (int e = 0; e < fes.GetNE(); e++)
   {
      auto element = fes.GetFE(e);
      auto eltrans = fes.GetElementTransformation(e);

      // assemble element mass and convection matrices
      conv_int.AssembleElementMatrix(*element, *eltrans, Ke);
      mass_int.AssembleElementMatrix(*element, *eltrans, Me);

      fes.GetElementDofs(e, dofs);
      ue.SetSize(dofs.Size());
      re.SetSize(dofs.Size());
      udote.SetSize(dofs.Size());

      x.GetSubVector(dofs, ue);
      udot.GetSubVector(dofs, udote);

      re = 0.0;
      for (int i = 0; i < dofs.Size(); i++)
      {
         for (int j = 0; j < i; j++)
         {
            // add high-order stabilization without correction for low-order
            // stabilization
            real_t fije = Me(i,j) * (udote(i) - udote(j));
            re(i) += fije;
            re(j) -= fije;
         }
      }

      // add convective term and add to global vector
      Ke.AddMult(ue, re, -1.0);
      y.AddElementVector(dofs, re);
   }

   // add boundary condition (u - u_inflow) * b (u - u_inflow) * b
   subtract(x, u_inflow, z);
   z *= b_lumped;
   y += z;

   // apply inverse lumped mass matrix
   y /= lumpedmassmatrix;
}

HighOrderTargetScheme::~HighOrderTargetScheme()
{ }

// Implementation of Class LowOrderScheme
LowOrderScheme::LowOrderScheme(FiniteElementSpace &fes_,
                               const Vector &lumpedmassmatrix_,
                               FunctionCoefficient &inflow,
                               VectorFunctionCoefficient &velocity,
                               BilinearForm &M):
   CG_FE_Evolution(fes_, lumpedmassmatrix_, inflow, velocity, M)
{ }

void LowOrderScheme::Mult(const Vector &x, Vector &y) const
{
   ComputeLOTimeDerivatives(x, y);
}

LowOrderScheme::~LowOrderScheme()
{ }

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
      X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
   }

   switch (problem)
   {
      case 0:
      {
         // Translations in 1D, 2D, and 3D
         switch (dim)
         {
            case 1: v(0) = 1.0; break;
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
         const real_t w = M_PI/2;
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
         const real_t w = M_PI/2;
         real_t d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
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
real_t u0_function(const Vector &x)
{
   int dim = x.Size();

   // map to the reference [-1,1] domain
   Vector X(dim);
   for (int i = 0; i < dim; i++)
   {
      real_t center = (bb_min[i] + bb_max[i]) * 0.5;
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
               return exp(-40.*pow(X(0)-0.5,2));
            case 2:
            case 3:
            {
               real_t rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
               if (dim == 3)
               {
                  const real_t s = (1. + 0.25*cos(2*M_PI*X(2)));
                  rx *= s;
                  ry *= s;
               }
               return ( std::erfc(w*(X(0)-cx-rx))*std::erfc(-w*(X(0)-cx+rx)) *
                        std::erfc(w*(X(1)-cy-ry))*std::erfc(-w*(X(1)-cy+ry)) )/16;
            }
         }
      }
      case 2:
      {
         real_t x_ = X(0), y_ = X(1), rho, phi;
         rho = std::hypot(x_, y_);
         phi = atan2(y_, x_);
         return pow(sin(M_PI*rho),2)*sin(3*phi);
      }
      case 3:
      {
         const real_t f = M_PI;
         return sin(f*X(0))*sin(f*X(1));
      }
   }
   return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
real_t inflow_function(const Vector &x)
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
