//                                MFEM Example 9
//                             SUNDIALS Modification
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../../data/periodic-segment.mesh -p 0 -r 2 -s 11 -dt 0.005
//    ex9 -m ../../data/periodic-square.mesh  -p 1 -r 2 -s 12 -dt 0.005 -tf 9
//    ex9 -m ../../data/periodic-hexagon.mesh -p 0 -r 2 -s 11 -dt 0.0018 -vs 25
//    ex9 -m ../../data/periodic-hexagon.mesh -p 0 -r 2 -s 13 -dt 0.01 -vs 15
//    ex9 -m ../../data/amr-quad.mesh         -p 1 -r 2 -s 13 -dt 0.002 -tf 9
//    ex9 -m ../../data/star-q3.mesh          -p 1 -r 2 -s 13 -dt 0.005 -tf 9
//    ex9 -m ../../data/disc-nurbs.mesh       -p 1 -r 3 -s 11 -dt 0.005 -tf 9
//    ex9 -m ../../data/periodic-cube.mesh    -p 0 -r 2 -s 12 -dt 0.02 -tf 8 -o 2
//
// Description:  This example code solves the time-dependent advection equation
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
#include <algorithm>

#ifndef MFEM_USE_SUNDIALS
#error This example requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Mesh bounding box
Vector bb_min, bb_max;


/** Reimplement Roberts problem here */
class RobertsSUNDIALS : public TimeDependentAdjointOperator
{
public:
   RobertsSUNDIALS(int dim, Vector p) :
      TimeDependentAdjointOperator(dim, 3),
      p_(p),
      adjointMatrix(NULL)
   {}

   virtual void Mult(const Vector &x, Vector &y) const;
   virtual void QuadratureIntegration(const Vector &x, Vector &y) const;
   virtual void AdjointRateMult(const Vector &y, Vector &yB, Vector &yBdot) const;
   virtual void ObjectiveSensitivityMult(const Vector &y, const Vector &yB,
                                         Vector &qbdot) const;
   virtual int SUNImplicitSetupB(const double t, const Vector &y, const Vector &yB,
                                 const Vector &fyB, int jokB, int *jcurB, double gammaB);
   virtual int SUNImplicitSolveB(Vector &x, const Vector &b, double tol);



   ~RobertsSUNDIALS()
   {
      delete adjointMatrix;
   }

protected:
   Vector p_;

   // Solvers
   GMRESSolver adjointSolver;
   SparseMatrix* adjointMatrix;
};

// class SundialsJacSolverB : public SundialsLinearSolver
// {
// public:
//   SundialsJacSolverB(TimeDependentAdjointOperator &oper_) : oper(&oper_) {}

//   virtual int ODELinSysB(double t, Vector y, Vector yB, Vector fyB, int jokB, int *jcurB,
//           double gammaB)
//   {
//     return oper->ImplicitSetupB(t, y, yB, fyB, jokB, jcurB, gammaB);
//   }

//   virtual int Solve(Vector &x, Vector b) {
//     double ignored = 0.0;
//     return oper->ImplicitSolveB(x, b, ignored);
//   }

// private:
//   TimeDependentAdjointOperator *oper;

// };


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 4e7;
   double dt = 0.01;
   bool visualization = true;
   bool visit = false;
   bool binary = false;
   int vis_steps = 5;

   // Relative and absolute tolerances for CVODE and ARKODE.
   const double reltol = 1e-4, abstol = 1e-6;

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
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - CVODE (adaptive order) implicit Adams,\n\t"
                  "            2 - ARKODE default (4th order) explicit,\n\t"
                  "            3 - ARKODE, \n\t"
                  "            4 - CVODES for adjoint sensitivities");
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
      args.PrintUsage(cout);
      return 1;
   }
   // check for vaild ODE solver option
   if (ode_solver_type < 1 || ode_solver_type > 4)
   {
      cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
      return 3;
   }
   args.PrintOptions(cout);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(1,1,1,Element::HEXAHEDRON);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   if (mesh->NURBSext)
   {
      mesh->SetCurvature(max(order, 1));
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 7. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and define the ODE solver used for time integration.

   Vector p(3);
   p[0] = 0.04;
   p[1] = 1.0e4;
   p[2] = 3.0e7;
   // 3 is the size of the solution vector
   RobertsSUNDIALS adv(3, p);

   double t = 0.0;
   adv.SetTime(t);

   // Create the time integrator
   ODESolver *ode_solver = NULL;
   CVODESolver *cvode = NULL;
   CVODESSolver *cvodes = NULL;
   ARKStepSolver *arkode = NULL;

   Vector u(3);
   u = 0.;
   u[0] = 1.;

   Vector abstol_v(3);
   abstol_v[0] = 1.0e-8;
   abstol_v[1] = 1.0e-14;
   abstol_v[2] = 1.0e-6;

   Vector q(1);
   q = 0.;

   switch (ode_solver_type)
   {
      case 1:
         cvode = new CVODESolver(CV_BDF);
         cvode->Init(adv);
         cvode->SetSVtolerances(reltol, abstol_v);
         cvode->SetMaxStep(dt);
         cvode->UseSundialsLinearSolver();
         ode_solver = cvode;
         break;
      case 2:
      case 3:
         arkode = new ARKStepSolver(ARKStepSolver::EXPLICIT);
         arkode->Init(adv);
         arkode->SetSStolerances(reltol, abstol);
         arkode->SetMaxStep(dt);
         if (ode_solver_type == 3) { arkode->SetERKTableNum(FEHLBERG_13_7_8); }
         ode_solver = arkode; break;
      case 4:
         cvodes = new CVODESSolver(CV_BDF);
         cvodes->Init(adv);
         cvodes->SetWFTolerances([reltol, abstol_v]
                                 (Vector y, Vector w, CVODESolver * self)
         {
            for (int i = 0; i < y.Size(); i++)
            {
               double ww = reltol * abs(y[i]) + abstol_v[i];
               if (ww <= 0.) { return -1; }
               w[i] = 1./ww;
            }
            return 0;
         }
                                );
         cvodes->SetSVtolerances(reltol, abstol_v);
         cvodes->UseSundialsLinearSolver();
         cvodes->InitQuadIntegration(q, 1.e-6, 1.e-6);
         cvodes->InitAdjointSolve(150, CV_HERMITE);
         ode_solver = cvodes; break;
   }

   // 8. Perform time-integration (looping over the time iterations, ti,
   //    with a time-step dt).
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = max(dt, t_final - t);
      ode_solver->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         //         cout << "time step: " << ti << ", time: " << t << endl;
         if (cvode) { cvode->PrintInfo(); }
         if (arkode) { arkode->PrintInfo(); }
         if (cvodes) { cvodes->PrintInfo(); }

      }
   }

   cout << "Final Solution: " << t << endl;
   u.Print();

   if (cvodes)
   {
      Vector q;
      cout << " Final Quadrature " << endl;
      cvodes->EvalQuadIntegration(t, q);
      q.Print();
   }

   // backward portion
   Vector w(3);
   w=0.;
   double TBout1 = 40.;
   Vector dG_dp(3);
   dG_dp=0.;
   if (cvodes)
   {
      t = t_final;
      adv.SetTime(t);
      cvodes->InitB(adv);
      cvodes->InitQuadIntegrationB(dG_dp, 1.e-6, 1.e-6);

      //     SundialsJacSolverB jacB(adv);
      //     cvodes->SetLinearSolverB(jacB);

      // Results at time TBout1
      double dt_real = max(dt, t - TBout1);
      cvodes->StepB(w, t, dt_real);
      cout << "t: " << t << endl;
      cout << "w:" << endl;
      w.Print();

      cvodes->GetForwardSolution(t, u);
      cout << "u:" << endl;
      u.Print();

      // Results at T0
      dt_real = max(dt, t - 0.);
      cvodes->StepB(w, t, dt_real);
      cout << "t: " << t << endl;
      cout << "w:" << endl;

      cvodes->GetForwardSolution(t, u);
      w.Print();
      cout << "u:" << endl;
      u.Print();

      // Evaluate Sensitivity
      cvodes->EvalQuadIntegrationB(t, dG_dp);
      cout << "dG/dp:" << endl;
      dG_dp.Print();

   }

   // 10. Free the used memory.
   delete ode_solver;

   return 0;
}

// Roberts Implementation
void RobertsSUNDIALS::Mult(const Vector &x, Vector &y) const
{
   y[0] = -p_[0]*x[0] + p_[1]*x[1]*x[2];
   y[2] = p_[2]*x[1]*x[1];
   y[1] = -y[0] - y[2];
}


void RobertsSUNDIALS::QuadratureIntegration(const Vector &y, Vector &qdot) const
{
   qdot[0] = y[2];
}


void RobertsSUNDIALS::AdjointRateMult(const Vector &y, Vector & yB,
                                      Vector &yBdot) const
{
   double l21 = (yB[1]-yB[0]);
   double l32 = (yB[2]-yB[1]);
   double p1 = p_[0];
   double p2 = p_[1];
   double p3 = p_[2];
   yBdot[0] = -p1 * l21;
   yBdot[1] = p2 * y[2] * l21 - 2. * p3 * y[1] * l32;
   yBdot[2] = p2 * y[1] * l21 - 1.0;
}

void RobertsSUNDIALS::ObjectiveSensitivityMult(const Vector &y,
                                               const Vector &yB, Vector &qBdot) const
{
   double l21 = (yB[1]-yB[0]);
   double l32 = (yB[2]-yB[1]);
   double y23 = y[1] * y[2];

   qBdot[0] = y[0] * l21;
   qBdot[1] = -y23 * l21;
   qBdot[2] = y[1]*y[1]*l32;
}

int RobertsSUNDIALS::SUNImplicitSetupB(const double t, const Vector &y,
                                       const Vector &yB,
                                       const Vector &fyB, int jokB, int *jcurB, double gammaB)
{

   // M = I- gamma J
   // J = dfB/dyB
   // fB
   // Let's create a SparseMatrix and fill in the entries since this example doesn't contain finite elements

   delete adjointMatrix;
   adjointMatrix = new SparseMatrix(y.Size(), yB.Size());
   for (int j = 0; j < y.Size(); j++)
   {
      Vector JacBj(yB.Size());
      Vector yBone(yB.Size());
      yBone = 0.;
      yBone[j] = 1.;
      AdjointRateMult(y, yBone, JacBj);
      JacBj[2] += 1.;
      for (int i = 0; i < y.Size(); i++)
      {
         adjointMatrix->Set(i,j, (i == j ? 1.0 : 0.) - gammaB * JacBj[i]);
      }
   }

   *jcurB = 1;
   adjointMatrix->Finalize();
   //  adjointMatrix->PrintMatlab();
   //  y.Print();
   adjointSolver.SetOperator(*adjointMatrix);

   return 0;
}

// Is b = -fB ?
// is tol reltol or abstol?
int RobertsSUNDIALS::SUNImplicitSolveB(Vector &x, const Vector &b, double tol)
{
   adjointSolver.SetRelTol(1e-14);
   adjointSolver.Mult(b, x);
   return (0);
}
