//                                MFEM Example 23
//
// Compile with: make ex23
//
// Sample runs:  ex23
//               ex23 -o 4 -tf 5
//               ex23 -m ../data/square-disc.mesh -o 2 -tf 2 --neumann
//               ex23 -m ../data/disc-nurbs.mesh -r 3 -o 4 -tf 2
//               ex23 -m ../data/inline-hex.mesh -o 1 -tf 2 --neumann
//               ex23 -m ../data/inline-tet.mesh -o 1 -tf 2 --neumann
//
// Description:  This example solves the wave equation problem of the form:
//
//                               d^2u/dt^2 = c^2 \Delta u.
//
//               The example demonstrates the use of time dependent operators,
//               implicit solvers and second order time integration.
//
//               We recommend viewing examples 9 and 10 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** After spatial discretization, the wave model can be written as:
 *
 *     d^2u/dt^2 = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass,
 *  and K is the stiffness matrix.
 *
 *  Class WaveOperator represents the right-hand side of the above ODE.
 */
class WaveOperator : public SecondOrderTimeDependentOperator
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

   CGSolver T_solver; // Implicit solver for T = M + fac0*K
   DSmoother T_prec;  // Preconditioner for the implicit solver

   Coefficient *c2;
   mutable Vector z; // auxiliary vector

public:
   WaveOperator(FiniteElementSpace &f, Array<int> &ess_bdr, real_t speed);

   using SecondOrderTimeDependentOperator::Mult;
   void Mult(const Vector &u, const Vector &du_dt,
             Vector &d2udt2) const override;

   /** Solve the Backward-Euler equation:
       d2udt2 = f(u + fac0*d2udt2,dudt + fac1*d2udt2, t),
       for the unknown d2udt2. */
   using SecondOrderTimeDependentOperator::ImplicitSolve;
   void ImplicitSolve(const real_t fac0, const real_t fac1,
                      const Vector &u, const Vector &dudt, Vector &d2udt2) override;

   ///
   void SetParameters(const Vector &u);

   ~WaveOperator() override;
};


WaveOperator::WaveOperator(FiniteElementSpace &f,
                           Array<int> &ess_bdr, real_t speed)
   : SecondOrderTimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0),
     fespace(f), M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height)
{
   // Assemble Laplace matrix
   c2 = new ConstantCoefficient(speed*speed);
   K = new BilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator(*c2));
   K->Assemble();

   // Assemble Mass matrix
   M = new BilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();

   // Apply Bcs
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   K->FormSystemMatrix(ess_tdof_list, Kmat);
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   // Configure preconditioner
   const real_t rel_tol = 1e-8;
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(Mmat);

   // Configure solver
   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(100);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);
}

void WaveOperator::Mult(const Vector &u, const Vector &du_dt,
                        Vector &d2udt2)  const
{
   // Compute:
   //    d2udt2 = M^{-1}*-K(u)
   // for d2udt2
   K->FullMult(u, z);
   z.Neg(); // z = -z
   z.SetSubVector(ess_tdof_list, 0.0);
   M_solver.Mult(z, d2udt2);
   d2udt2.SetSubVector(ess_tdof_list, 0.0);
}

void WaveOperator::ImplicitSolve(const real_t fac0, const real_t fac1,
                                 const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   // Solve the equation:
   //    d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
   // for d2udt2
   if (!T)
   {
      T = Add(1.0, Mmat, fac0, Kmat);
      T_solver.SetOperator(*T);
   }
   K->FullMult(u, z);
   z.Neg();
   z.SetSubVector(ess_tdof_list, 0.0);
   T_solver.Mult(z, d2udt2);
   d2udt2.SetSubVector(ess_tdof_list, 0.0);
}

void WaveOperator::SetParameters(const Vector &u)
{
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
}

WaveOperator::~WaveOperator()
{
   delete T;
   delete M;
   delete K;
   delete c2;
}

real_t InitialSolution(const Vector &x)
{
   return exp(-x.Norml2()*x.Norml2()*30);
}

real_t InitialRate(const Vector &x)
{
   return 0.0;
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   const char *ref_dir  = "";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 10;
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t speed = 1.0;
   bool visualization = true;
   bool visit = true;
   bool dirichlet = true;
   int vis_steps = 5;

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
                  "ODE solver: [0--10] - GeneralizedAlpha(0.1 * s),\n\t"
                  "\t   11 - Average Acceleration, 12 - Linear Acceleration\n"
                  "\t   13 - CentralDifference, 14 - FoxGoodwin");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&speed, "-c", "--speed",
                  "Wave speed.");
   args.AddOption(&dirichlet, "-dir", "--dirichlet", "-neu",
                  "--neumann",
                  "BC switch.");
   args.AddOption(&ref_dir, "-r", "--ref",
                  "Reference directory for checking final solution.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several second order
   //    time integrators are available.
   SecondOrderODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit methods
      case 0: ode_solver = new GeneralizedAlpha2Solver(0.0); break;
      case 1: ode_solver = new GeneralizedAlpha2Solver(0.1); break;
      case 2: ode_solver = new GeneralizedAlpha2Solver(0.2); break;
      case 3: ode_solver = new GeneralizedAlpha2Solver(0.3); break;
      case 4: ode_solver = new GeneralizedAlpha2Solver(0.4); break;
      case 5: ode_solver = new GeneralizedAlpha2Solver(0.5); break;
      case 6: ode_solver = new GeneralizedAlpha2Solver(0.6); break;
      case 7: ode_solver = new GeneralizedAlpha2Solver(0.7); break;
      case 8: ode_solver = new GeneralizedAlpha2Solver(0.8); break;
      case 9: ode_solver = new GeneralizedAlpha2Solver(0.9); break;
      case 10: ode_solver = new GeneralizedAlpha2Solver(1.0); break;

      case 11: ode_solver = new AverageAccelerationSolver(); break;
      case 12: ode_solver = new LinearAccelerationSolver(); break;
      case 13: ode_solver = new CentralDifferenceSolver(); break;
      case 14: ode_solver = new FoxGoodwinSolver(); break;

      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         delete mesh;
         return 3;
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;

   GridFunction u_gf(&fespace);
   GridFunction dudt_gf(&fespace);

   // 6. Set the initial conditions for u. All boundaries are considered
   //    natural.
   FunctionCoefficient u_0(InitialSolution);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   FunctionCoefficient dudt_0(InitialRate);
   dudt_gf.ProjectCoefficient(dudt_0);
   Vector dudt;
   dudt_gf.GetTrueDofs(dudt);

   // 7. Initialize the wave operator and the visualization.
   Array<int> ess_bdr;
   if (mesh->bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());

      if (dirichlet)
      {
         ess_bdr = 1;
      }
      else
      {
         ess_bdr = 0;
      }
   }
   WaveOperator oper(fespace, ess_bdr, speed);

   u_gf.SetFromTrueDofs(u);
   {
      ofstream omesh("ex23.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex23-init.gf");
      osol.precision(precision);
      u_gf.Save(osol);
      dudt_gf.Save(osol);
   }

   VisItDataCollection visit_dc("Example23", mesh);
   visit_dc.RegisterField("solution", &u_gf);
   visit_dc.RegisterField("rate", &dudt_gf);
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

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ode_solver->Init(oper);
   real_t t = 0.0;

   bool last_step = false;
   for (int ti = 1; !last_step; ti++)
   {

      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, dudt, t, dt);

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         dudt_gf.SetFromTrueDofs(dudt);
         if (visualization)
         {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
      oper.SetParameters(u);
   }

   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex23.mesh -g ex23-final.gf".
   {
      ofstream osol("ex23-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
      dudt_gf.Save(osol);
   }

   // 10. Free the used memory.
   delete ode_solver;
   delete mesh;

   return 0;
}
