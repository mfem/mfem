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

// Velocity coefficient
std::function<void(const Vector&, Vector&)> getVelocityFunction(int problem);
// Initial condition
std::function<double(const Vector&)> getInflowFunction(int problem);

// Inflow boundary condition
std::function<double(const Vector&)> getInitialCondition(int problem);

// Mesh bounding box
Vector bb_min, bb_max;

#ifdef MFEM_USE_LAPACK
/// Integrator that inverts the matrix assembled by another integrator via Pseudo-inverse.
class PseudoInverseIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrator;
   BilinearFormIntegrator *integrator;

public:
   PseudoInverseIntegrator(BilinearFormIntegrator *integ, int own_integ = 1)
   { integrator = integ; own_integrator = own_integ; }

   virtual void SetIntRule(const IntegrationRule *ir);

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~PseudoInverseIntegrator() { if (own_integrator) { delete integrator; } }
};
#endif

class DG_Solver : public Solver
{
private:
   SparseMatrix &M, &K, A;
   GMRESSolver linear_solver;
   BlockILU prec;
   double dt;
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

   void SetTimeStep(double dt_)
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
class FE_Evolution : public TimeDependentOperator
{
private:
   BilinearForm &M, &K;
   const Vector &b;
   Solver *M_prec;
   CGSolver M_solver;
   DG_Solver *dg_solver;
   FiniteElementSpace *fes;

   Array<int> offsets;
   BlockVector sol;
   Vector ktemp;
   GridFunction latent;
   GridFunction latent_k;
   GridFunction xnew;

   GridFunctionCoefficient latent_cf;
   GridFunctionCoefficient latent_k_cf;
   GridFunctionCoefficient xnew_cf;
   Vector delta_k, delta_latent, delta_latent_inner;

   TransformedCoefficient expLatent_cf;
   SumCoefficient diff_latent_cf;
   TransformedCoefficient expResidual_cf;

   LinearForm delta_latent_form;
   NonlinearForm invExpLatentForm;
   LinearForm newtonRHS;


   mutable Vector z;

public:
   FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

   virtual ~FE_Evolution();
   bool postprocess = false;
   bool initialized;
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int problem = 1;
   const char *mesh_file = "../data/periodic-square-4x4.mesh";
   int ref_levels = 2;
   int order = 3;
   bool pa = false;
   bool ea = false;
   bool fa = false;
   const char *device_config = "cpu";
   int ode_solver_type = 11;
   double t_final = 2;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool paraview = false;
   bool binary = false;
   int vis_steps = 1;
   bool applyPostprocess = false;

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

   Device device(device_config);
   device.Print();

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 3. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
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
         return 3;
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
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   VectorFunctionCoefficient velocity(dim, getVelocityFunction(problem));
   FunctionCoefficient inflow(getInflowFunction(problem));
   FunctionCoefficient u0(getInitialCondition(problem));

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
   m.AddDomainIntegrator(new MassIntegrator);
   constexpr double alpha = -1.0;
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
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
         // sout << "pause\n";
         sout << flush;
         // cout << "GLVis visualization paused."
         //      << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   FE_Evolution adv(m, k, b);
   adv.postprocess = applyPostprocess;

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
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
   out << "L2 error: " << u.ComputeL2Error(u0) << std::endl;

   // 10. Free the used memory.
   delete ode_solver;
   delete pd;
   delete dc;

   return 0;
}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
   : TimeDependentOperator(M_.Height()), M(M_), K(K_), b(b_), fes(M.FESpace()),
     offsets({0, fes->GetNDofs(), fes->GetNDofs()*2}), sol(offsets),
ktemp(sol.GetBlock(0).GetData(), fes->GetNDofs()),
latent(fes, sol.GetBlock(1).GetData()), latent_k(fes), xnew(fes),
latent_cf(&latent),
latent_k_cf(&latent_k), xnew_cf(&xnew), delta_k(xnew.Size()),
delta_latent(latent.Size()), delta_latent_inner(latent.Size()),
expLatent_cf(&latent_cf, [](double x) {return exp(x);}),
diff_latent_cf(latent_cf, latent_k_cf, -1.0, 1.0),
expResidual_cf(&latent_cf, [](double psi) {return exp(psi)*(1.0 - psi);}),
delta_latent_form(fes), invExpLatentForm(fes), newtonRHS(fes), z(M_.Height()),
initialized(false)
{

   delta_latent_form.AddDomainIntegrator(new DomainLFIntegrator(diff_latent_cf));
   // since DG is local space, we can perform inverse element-wise.
   invExpLatentForm.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(
                                                                 expLatent_cf)));
   newtonRHS.AddDomainIntegrator(new DomainLFIntegrator(expResidual_cf));
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

void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   if (!initialized)
   {
      ktemp = 0.0;
      latent = x;
      // latent.ApplyMap([](double x) {return log(x); });
      latent.ApplyMap([](double x) {return x > 1e-09 ? log(x) : -20.0; });
      initialized = true;
   }
   int printlevel = 2;
   bool debug = true;
   MFEM_VERIFY(dg_solver != NULL,
               "Implicit time integration is not supported with partial assembly");
   double alpha_prev = 1.0;
   // auto alphamaker = [&alpha_prev](int i) {return min(max(1.0, pow(1.5, pow(1.5, i-1)) - alpha_prev), 1e10);};
   auto alphamaker = [](int i) {return (double)(i*i);};
   GridFunction zero_gf(fes);
   zero_gf = 0.0;

   BlockOperator op(offsets);
   SparseMatrix A(x.Size());
   A = K.SpMat();
   A *= -dt;
   A += M.SpMat();
   op.SetBlock(0, 0, &A);

   SparseMatrix offdiag(x.Size());
   offdiag = M.SpMat();
   offdiag *= dt;

   op.SetBlock(1, 0, &offdiag);
   op.SetBlock(0, 1, &offdiag);

   BlockVector rhs(offsets);
   z.SetDataAndSize(rhs.GetBlock(0).GetData(), x.Size());
   Vector z2(rhs.GetBlock(1).GetData(), x.Size());


   BlockDiagonalPreconditioner prec(offsets);
   BilinearForm invM(fes);
   invM.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
   invM.Assemble();
   prec.SetDiagonalBlock(0, &invM.SpMat());

   int maxIter(100000);
   double rtol(1.e-6);
   double atol(1.e-10);
   GMRESSolver solver;
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(op);
   solver.SetPreconditioner(prec);
   solver.SetPrintLevel(0);
   solver.iterative_mode = true;
   bool converged = false;
   for (int i=1; i<1e04; i++)
   {
      latent_k = latent;
      if (i > 1)
      {
         double alpha = alphamaker(i);
         double dalpha = alpha / alpha_prev;
         alpha_prev = alpha;

         A *= dalpha;
         invM.SpMat().operator*=(dalpha);
      }
      bool converged_inner = false;
      for (int j=1; j<=10; j++)
      {
         delta_k = ktemp;
         rhs = 0.0;

         K.Mult(x, z);
         z += b;
         z *= alphamaker(i);
         M.AddMult(latent, z, -dt);
         M.AddMult(latent_k, z, dt);

         BilinearForm expM(fes);
         expM.AddDomainIntegrator(new MassIntegrator(expLatent_cf));
         expM.Assemble();
         expM.SpMat().operator*=(-1.0);

         op.SetBlock(1, 1, &expM.SpMat());

         if (debug)
         {
            out << latent.Min() << ", " << latent.Max() << std::endl;
            if (latent.CheckFinite())
            {
               for (auto val : latent) {if (!IsFinite(val)) out << val << " .. ";}
               out << std::endl;
               mfem_error("latent variable is not finite.");
            }
            GridFunction dummy(fes);
            dummy.ProjectCoefficient(expLatent_cf);
            if (dummy.CheckFinite())
            {
               for (auto val : dummy) {if (!IsFinite(val)) out << val << " .. ";}
               out << std::endl;
               mfem_error("Mapped latent variable is not finite.");
            }
         }
         BilinearForm expInvM(fes);

         expInvM.AddDomainIntegrator(new PseudoInverseIntegrator(new MassIntegrator(
                                                                    expLatent_cf)));
         expInvM.Assemble();
         expInvM.SpMat().operator*=(-1.0);
         prec.SetDiagonalBlock(1, &expInvM.SpMat());

         M.Mult(x, z2);
         z2.Neg();
         newtonRHS.Assemble();
         z2 += newtonRHS;

         solver.Mult(rhs, sol);
         if (!solver.GetConverged())
         {
            mfem_warning("LinearSolver failed to converge.");
         }

         delta_k -= ktemp;
         if (printlevel > 1) { out << "\t\tInner Step " << j << ", " << delta_k.Normlinf() << std::endl; }
         if (delta_k.Normlinf() < 1e-07)
         {
            if (printlevel > 0) { out << "\t\tInnerloop convereged in " << j << " steps."<< std::endl; }
            converged_inner = true;
            break;
         }
      }
      if (!converged_inner) {mfem_warning("Subproblem failed to converge."); }
      TransformedCoefficient latent_diff(&latent_k_cf, &latent_cf, [](double x,
      double xnew) {return exp(x)*(x-xnew);});
      double residual = zero_gf.ComputeLpError(infinity(), latent_diff);
      if (printlevel > 1) { out << "\tOuter Step " << i << ", " << residual << std::endl; }
      if (zero_gf.ComputeLpError(infinity(), latent_diff) < 1e-07)
      {
         converged = true;
         if (printlevel > 0) out << "\tOuterloop convereged in " << i <<
                                    " steps.\n---------------------------------------------------"<< std::endl;
         break;
      }
   }
   if (!converged) {mfem_warning("Main loop failed to converge."); }
   if (postprocess)
   {
      k = latent;
      k.ApplyMap([](double x) {return exp(x); });
      k -= x;
      k *= dt;
   }
   else
   {
      k = ktemp;
   }
}

FE_Evolution::~FE_Evolution()
{
   delete M_prec;
   delete dg_solver;
}

// Velocity coefficient
std::function<void(const Vector&, Vector &)> getVelocityFunction(int problem)
{
   switch (problem)
   {
      case 0:
         return [](const Vector &x, Vector &y) { y = 1.0; };
      case 1:
         return [](const Vector &x, Vector &y) { y = 1.0; };
      default:
         mfem_error("Undefined problem");
         return [](const Vector &x, Vector &y) { };
   }

}

// Initial condition
std::function<double(const Vector &)> getInitialCondition(int problem)
{
   switch (problem)
   {
      case 0:
         return [](const Vector &x)
         {
            double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
            return ( erfc(w*(x(0)-cx-rx))*erfc(-w*(x(0)-cx+rx)) *
                     erfc(w*(x(1)-cy-ry))*erfc(-w*(x(1)-cy+ry)) )/16;
         };
      case 1:
         return [](const Vector &x)
         {
            double y = 1.0;
            for (auto v:x) {y *= sin(M_PI*v); }
            y += 1.0;
            return y;
         };
      default:
         mfem_error("Undefined problem");
         return [](const Vector &x) { return 0.0; };
   }
}

// Inflow boundary condition (zero for the problems considered in this example)
std::function<double(const Vector&)> getInflowFunction(int problem)
{
   switch (problem)
   {
      case 0:
         return [](const Vector &x) {return 0.0; };
      case 1:
         return [](const Vector &x) {return 0.0; };
      default:
         mfem_error("Undefined problem");
         return [](const Vector &x) { return 0.0; };
   }
}


#ifdef MFEM_USE_LAPACK
void PseudoInverseIntegrator::SetIntRule(const IntegrationRule *ir)
{
   IntRule = ir;
   integrator->SetIntRule(ir);
}

void PseudoInverseIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   integrator->AssembleElementMatrix(el, Trans, elmat);
   DenseMatrixSVD svd(elmat,true,true);
   svd.Eval(elmat);
   Vector &sigma = svd.Singularvalues();
   DenseMatrix &U = svd.LeftSingularvectors();
   DenseMatrix &V = svd.RightSingularvectors();
   sigma.ApplyMap([](double x) {return x > 1e-08 ? 1.0 / x : 0; });
   DenseMatrix Vt(V); Vt.Transpose();
   MultADBt(U,sigma,Vt,elmat);
}
#endif