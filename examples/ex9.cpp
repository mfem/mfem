//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.02 -s 23 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ex9 -m ../data/periodic-square.msh -p 0 -r 2 -dt 0.005 -tf 2
//    ex9 -m ../data/periodic-cube.msh -p 0 -r 1 -o 2 -tf 2
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

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

bool HasNegative(const Vector &u);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
real_t u0_function(const Vector &x);

// Initial bounds
Vector bounds(const int dim);

// Inflow boundary condition
real_t inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

// Element Neighbor function for localizing min/max bounds
void GetElementNeighbors(const Mesh &mesh, int elem_id, Array<int> &neighbors);

void Limiter(GridFunction &u, const FiniteElementSpace &fes, const Vector &bds);

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
        prec(fes.GetTypicalFE()->GetDof(),
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

   void SetOperator(const Operator &op) override
   {
      linear_solver.SetOperator(op);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      linear_solver.Mult(x, y);
   }
};

/// Abstract class for an ODESolver that can use a bounds-preserving limiter dependent on FEspace information in Step(). 
/// Still retains ODESolver::Step if limiter does not need to be applied. 
class LimitedODESolver : virtual public ODESolver
{
protected:
   const FiniteElementSpace fes;
   const Vector bds;

public:
   LimitedODESolver(const FiniteElementSpace &fes_, const Vector &bds_) : ODESolver() , fes(fes_) , bds(bds_) { };

   using ODESolver::Step;

   virtual void Step(GridFunction &x, real_t &t, real_t &dt, const FiniteElementSpace &fes, const Vector &bds) = 0;
};

class LimitedForwardEulerSolver : public ForwardEulerSolver, public LimitedODESolver //ODESolver
{
public:
   LimitedForwardEulerSolver(const FiniteElementSpace &fes_, const Vector &bds_) : LimitedODESolver(fes_, bds_) { }

   void Step(Vector &x, real_t &t, real_t &dt) override 
   {
        ForwardEulerSolver::Step(x, t, dt); // Only necessary if we want to keep override ODESolver::Step
   }

   void Step(GridFunction &x, real_t &t, real_t &dt, const FiniteElementSpace &fes, const Vector &bds) override
   {
      f->SetTime(t);
      f->Mult(x, dxdt);
      x.Add(dt, dxdt);
      Limiter(x, fes, bds);
      t += dt;
   }
};

class LimitedRK2Solver : public RK2Solver, public LimitedODESolver //ODESolver
{
public:
   LimitedRK2Solver(const FiniteElementSpace &fes_, const Vector bds_, const real_t a_ = 2./3.) : RK2Solver(a_) , LimitedODESolver(fes_, bds_) { }

   void Step(Vector &x, real_t &t, real_t &dt) override 
   {
        RK2Solver::Step(x, t, dt); // Only necessary if we want to keep override ODESolver::Step
   }

   void Step(GridFunction &x, real_t &t, real_t &dt, const FiniteElementSpace &fes, const Vector &bds) 
   {
      
      //  0 |
      //  a |  a
      // ---+--------
      //    | 1-b  b      b = 1/(2a)

      const real_t b = 0.5/a;
      
      f->SetTime(t);
      f->Mult(x, dxdt);
      add(x, (1. - b)*dt, dxdt, x1);
      x.Add(a*dt, dxdt);
      Limiter(x, fes, bds);

      f->SetTime(t + a*dt);
      f->Mult(x, dxdt);
      add(x1, b*dt, dxdt, x);
      Limiter(x, fes, bds);
      t += dt;
   }
};

class LimitedRK3SSPSolver : public RK3SSPSolver, public LimitedODESolver
{
private:
   GridFunction gy;
   GridFunction gk;

public:
   LimitedRK3SSPSolver(const FiniteElementSpace &fes_, const Vector &bds_) 
      : LimitedODESolver(fes_, bds_) , 
        gy(const_cast<mfem::FiniteElementSpace*>(&fes)) , 
        gk(const_cast<mfem::FiniteElementSpace*>(&fes)) { }

   void Step(Vector &x, real_t &t, real_t &dt) override 
   {
        RK3SSPSolver::Step(x, t, dt); // Only necessary if we want to keep override ODESolver::Step
   }

   void Step(GridFunction &x, real_t &t, real_t &dt, const FiniteElementSpace &fes, const Vector &bds) 
   {
      // x0 = x, t0 = t, k0 = dt*f(t0, x0)
      f->SetTime(t);
      f->Mult(x, gk);

      // x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
      add(x, dt, gk, gy);
      Limiter(gy, fes, bds);
      f->SetTime(t + dt);
      f->Mult(gy, gk);

      // x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
      gy.Add(dt, gk);
      add(3./4, x, 1./4, gy, gy);
      Limiter(gy, fes, bds); 
      f->SetTime(t + dt/2);
      f->Mult(gy, gk);

      // x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt
      gy.Add(dt, gk);
      add(1./3, x, 2./3, gy, x);
      Limiter(x, fes, bds);
      t += dt;
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
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;

   mutable Vector z;

public:
   FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);

   void Mult(const Vector &x, Vector &y) const override;
   void ImplicitSolve(const real_t dt, const Vector &x, Vector &k) override;

   ~FE_Evolution() override;
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   // const char *mesh_file = "../data/periodic-segment.mesh";
   int ref_levels = 0;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 3;
   real_t t_final = 10.0;
   real_t dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 5;
   bool limiter = false;

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
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&limiter, "-l", "--limiter", "-no-l", "--no-limiter",
                  "Apply bounds-preserving limiter.");
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

   Device device(device_config);
   device.Print();

   cout << "MFEM version: "
      << MFEM_VERSION_MAJOR << "."
      << MFEM_VERSION_MINOR << "." 
      << MFEM_VERSION_PATCH << std::endl;

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

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
   // DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   DG_FECollection fec(order, dim, BasisType::Positive);

   FiniteElementSpace fes(&mesh, &fec);

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, velocity_function);
   FunctionCoefficient inflow(inflow_function);
   FunctionCoefficient u0(u0_function);
   Vector bds = bounds(dim);
   bds.Print();

   int quadorder = 2*order;
   const FiniteElement *fe = fes.GetFE(0);
   IntegrationRules irGL(0, Quadrature1D::GaussLobatto);
   IntegrationRule ir = irGL.Get(fe->GetGeomType(), quadorder);

   BilinearForm m(&fes);
   BilinearForm k(&fes);
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
   m.AddDomainIntegrator(new MassIntegrator(&ir));
   // m.AddDomainIntegrator(new MassIntegrator);
   constexpr real_t alpha = -1.0;
   auto convecInt = new ConvectionIntegrator(velocity, alpha);
   // const IntegrationRule* intrul = convecInt->GetIntegrationRule();
   // cout << "int rule order is default: " << intrul->GetNPoints() << endl;
   convecInt->SetIntRule(&ir);     
   k.AddDomainIntegrator(convecInt);
   // k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
   k.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
   k.AddBdrFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));

   LinearForm b(&fes);
   b.AddBdrFaceIntegrator(
      new BoundaryFlowIntegrator(inflow, velocity, alpha));

   m.Assemble();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   b.Assemble();
   m.Finalize();
   k.Finalize(skip_zeros);

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.

   LimitedODESolver* ode_solver = nullptr;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new LimitedForwardEulerSolver(fes, bds); break;
      case 2: ode_solver = new LimitedRK2Solver(fes, bds, 0.5); break; // midpoint method (SSP)
      case 3: ode_solver = new LimitedRK3SSPSolver(fes, bds); break;
   }

   // 7. Define the initial conditions, save the corresponding grid function to
   //    a file and (optionally) save data in the VisIt format and initialize
   //    GLVis visualization.
   GridFunction u(&fes);
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
         // sout << "autoscale on\n"; 
         // sout << "autoscale off\n";  // Turn off autoscale for colorscare
         // sout << "scale 0.25 1.75\n";  // Sets colorscale 
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m, k, b);

   real_t t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   int offset = 0;
   double Pmin = bds(1), Pmax = bds(0);
   for (int ti = 0; !done; )
   {
      real_t dt_real = min(dt, t_final - t);

      if (limiter) { ode_solver->Step(u, t, dt_real, fes, bds); }
      else { ode_solver->Step(u, t, dt_real); }

      ti++;
      
      Vector nodes(u.Size());
      Vector node_coords(dim);
      Vector P(u.Size());
      offset = 0;
      for (int el = 0; el < mesh.GetNE(); el++)
      {
         const FiniteElement *fe = fes.GetFE(el);
         ElementTransformation *trans = mesh.GetElementTransformation(el);

         int edof = fe->GetDof();
         const IntegrationRule &ir = fe->GetNodes(); // Node locations in reference element

         for (int i = 0; i < edof; i++)
         {
            const IntegrationPoint &ip = ir.IntPoint(i);

            // Map reference node to physical coordinates (optional, for info)
            trans->Transform(ip, node_coords);

            // Evaluate GridFunction at this node
            real_t value = u.GetValue(el, ip);
            P[offset+i] = value;

            // cout << "Element " << el << ", node " << i << " at "; node_coords.Print(); cout << ": value = " << value << endl;
         }
         offset += edof;
      }
      // cout << "u coef values: " << u.Min() << " and " << u.Max() << endl;
      // cout << "u phys values: " << P.Min() << " and " << P.Max() << endl;
      Pmin = min(Pmin, P.Min());
      Pmax = max(Pmax, P.Max());
      
      if (HasNegative(P)) {
         cout << "P has negative value at step " << ti << endl;
      }
      // P.Print();
      // cout << "\n";

      
      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;
         // u.Print();

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
   cout << "Minimum and maximum value of P over all time is " << Pmin << " and " << Pmax << endl;

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex9.mesh -g ex9-final.gf".
   {
      ofstream osol("ex9-final.gf");
      osol.precision(precision);
      u.Save(osol);
   }

   // 10. Free the used memory.
   delete pd;
   delete dc;

   return 0;
}


bool HasNegative(const Vector &u)
{
    for (int i = 0; i < u.Size(); i++)
    {
        if (u(i) < 0.0)
        {
            return true; // Found a negative value
        }
    }
    return false; // No negative values
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
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

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}

void FE_Evolution::ImplicitSolve(const real_t dt, const Vector &x, Vector &k)
{
   MFEM_VERIFY(dg_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   K.Mult(x, z);
   z += b;
   dg_solver->SetTimeStep(dt);
   dg_solver->Mult(z, k);
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
   delete dg_solver;
}

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
               // return 0.5*cos(M_PI*X[0])+0.5;
               // return -(X[0]-1)*(X[0]+1);
               // return 1.0;
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

// Initial bounds 
Vector bounds(const int dim)
{
   switch (problem)
   {
      case 0:
      case 1:
      {
         switch (dim)
         {
            case 1:
            {
               Vector bds(2);
               bds(0) = 0.0; bds(1) = 1.0;
               return bds;
            }
               
         }
      }
   }
   Vector bds;
   return bds;
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

void GetElementNeighbors(const mfem::Mesh &mesh, int elem_id, mfem::Array<int> &neighbors)
{
    neighbors.SetSize(0);

    int num_faces = mesh.GetNumFaces();
   //  cout << "num faces " << num_faces << endl;

    for (int i = 0; i < num_faces; i++)
    {
        int elem1, elem2;
        mesh.GetFaceElements(i, &elem1, &elem2);
      //   cout << "elem1 " << elem1 << " and " << "elem2 " << elem2 << endl;

        if (elem1 == elem_id && elem2 != -1)
        {
            neighbors.Append(elem2);
            // cout << "neigh size " << neighbors.Size() << endl;
        }
        else if (elem2 == elem_id && elem1 != -1)
        {
            neighbors.Append(elem1);
            // cout << "neigh size " << neighbors.Size() << endl;
        }
    }
}

void Limiter(GridFunction &u, const FiniteElementSpace &fes, const Vector &bds)
{
   Mesh* mesh = fes.GetMesh(); 

   const FiniteElement *fe = nullptr;
   const IntegrationRule *ir = nullptr;
   int order = 0;
   Vector u_local, u_vals;
   DenseMatrix elfun;

   // Loop over elements
   for (int i = 0; i < fes.GetNE(); i++)
   {
      Array<int> dofs;
      fes.GetElementDofs(i, dofs);
      fe = fes.GetFE(i);
      // order = fe->GetOrder(); cout << "order of fe is " << endl;
      ElementTransformation *trans = mesh->GetElementTransformation(i);

      // Use a suitable integration rule (e.g., order = 2*fe->GetOrder())
      int intorder = floor((order+3)/2.0);
      IntegrationRules irGL(0, Quadrature1D::GaussLobatto);
      ir = &irGL.Get(fe->GetGeomType(), intorder);
      // ir = &IntRules.Get(fe->GetGeomType(), intorder);

      u_local.SetSize(dofs.Size());
      u.GetSubVector(dofs, u_local);

      // Evaluate solution at quadrature points
      u_vals.SetSize(ir->GetNPoints());
      // fe->GetValues(*ir, u_local, u_vals);

      // Compute cell average via quadrature
      double int_val = 0.0, int_w = 0.0;
      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         trans->SetIntPoint(&ip);

         double w = ip.weight * trans->Weight();
         u_vals(q) = u.GetValue(i, ip);

         int_val += w * u_vals(q);
         int_w += w;
      }
      double cell_avg = int_val / int_w;
      // cout << cell_avg << endl;

      double m_i = bds(0), M_i = bds(1);
      // Compute theta for limiting
      double theta = 1.0;
      for (int k = 0; k < u_local.Size(); k++)
      {
         double diff = u_local[k] - cell_avg;
         if (diff > 0.0)
         {
               if (cell_avg < M_i)
                  theta = std::min(theta, (M_i - cell_avg) / diff);
               else
                  theta = 0.0;
         }
         else if (diff < 0.0)
         {
               if (cell_avg > m_i)
                  theta = std::min(theta, (m_i - cell_avg) / diff);
               else
                  theta = 0.0;
         }
      }
      // cout << "theta is " << theta << endl;

      // Apply limiter
      for (int k = 0; k < u_local.Size(); k++)
         u_local(k) = theta * (u_local(k) - cell_avg) + cell_avg;
         // if (hold < bds(0) && abs(u_local(k)) < (bds(0)+1e-14) ) u_local(k) = bds(0);
      u.SetSubVector(dofs, u_local);
   }

}
