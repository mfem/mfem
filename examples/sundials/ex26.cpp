//                                MFEM Example 26
//                             SUNDIALS Example in MFEM
//
// Compile with: make ex9
//
// Sample runs:
//    sundials_ex26 -dt 0.01
//    sundials_ex26 -dt 0.005
//
// Description:  This example is a port of cvodes/serial/cvsRoberts_ASAi_dns example that is part of SUNDIALS.
//               The goal is to demonstrate how to use the adjoint SUNDIALS CVODES interface from MFEM.
//               The following is an exerpt description from the aforementioned file.
//
//
// * Adjoint sensitivity example problem.
// * The following is a simple example problem, with the coding
// * needed for its solution by CVODES. The problem is from chemical
// * kinetics, and consists of the following three rate equations.
// *    dy1/dt = -p1*y1 + p2*y2*y3
// *    dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2
// *    dy3/dt =  p3*(y2)^2
// * on the interval from t = 0.0 to t = 4.e10, with initial
// * conditions: y1 = 1.0, y2 = y3 = 0. The reaction rates are:
// * p1=0.04, p2=1e4, and p3=3e7. The problem is stiff.
// * This program solves the problem with the BDF method, Newton
// * iteration with the DENSE linear solver, and a user-supplied
// * Jacobian routine.
// * It uses a scalar relative tolerance and a vector absolute
// * tolerance.
// * Output is printed in decades from t = .4 to t = 4.e10.
// * Run statistics (optional outputs) are printed at the end.
// * 
// * Optionally, CVODES can compute sensitivities with respect to
// * the problem parameters p1, p2, and p3 of the following quantity:
// *   G = int_t0^t1 g(t,p,y) dt
// * where
// *   g(t,p,y) = y3
// *        
// * The gradient dG/dp is obtained as:
// *   dG/dp = int_t0^t1 (g_p - lambda^T f_p ) dt - lambda^T(t0)*y0_p
// *         = - xi^T(t0) - lambda^T(t0)*y0_p
// * where lambda and xi are solutions of:
// *   d(lambda)/dt = - (f_y)^T * lambda - (g_y)^T
// *   lambda(t1) = 0
// * and
// *   d(xi)/dt = - (f_p)^T * lambda + (g_p)^T
// *   xi(t1) = 0
// * 
// * During the backward integration, CVODES also evaluates G as
// *   G = - phi(t0)
// * where
// *   d(phi)/dt = g(t,y,p)
// *   phi(t1) = 0




#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#ifndef MFEM_USE_SUNDIALS
#error This example requires that MFEM is built with MFEM_USE_SUNDIALS=YES
#endif

using namespace std;
using namespace mfem;


/// We create a TimeDependentAdjointOperator implementation of the rate equations to recreate the cvsRoberts_ASAi_dns problem
class RobertsSUNDIALS : public TimeDependentAdjointOperator
{
public:
   RobertsSUNDIALS(int dim, Vector p) :
      TimeDependentAdjointOperator(dim, 3),
      p_(p),
      adjointMatrix(NULL)
   {}

   /// Rate equation for forward problem
   virtual void Mult(const Vector &x, Vector &y) const;

   /// Quadrature integration for G
   virtual void QuadratureIntegration(const Vector &x, Vector &y) const;

   /// Adjoint rate equation corresponding to d(lambda)/dt
   virtual void AdjointRateMult(const Vector &y, Vector &yB, Vector &yBdot) const;

   /// Quadrature sensitivity equations corresponding to dG/dp
   virtual void QuadratureSensitivityMult(const Vector &y, const Vector &yB,
                                         Vector &qbdot) const;

   /// Setup custom MFEM solvers using GMRES since the Jacobian matrix is not symmetric
   virtual int SUNImplicitSetupB(const double t, const Vector &y, const Vector &yB,
                                 const Vector &fyB, int jokB, int *jcurB, double gammaB);

   /// Setup custom MFEM solve
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

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   double t_final = 4e7;
   double dt = 0.01;

   // Relative and absolute tolerances for CVODE and ARKODE.
   const double reltol = 1e-4, abstol = 1e-6;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   
   args.PrintOptions(cout);

   // 2. The original cvsRoberts_ASAi_dns problem is a fixed sized problem of size 3. Define the solution vector.
   Vector u(3);
   u = 0.;
   u[0] = 1.;

   // 3. Define the TimeDependentAdjointOperator which implements the various rate equations: rate, quadrature rate, adjoint rate, and quadrature sensitivity rate equation

   // Define material parameters p
   Vector p(3);
   p[0] = 0.04;
   p[1] = 1.0e4;
   p[2] = 3.0e7;
   // 3 is the size of the adjoint solution vector
   RobertsSUNDIALS adv(3, p);

   // 4. Set the inital time
   double t = 0.0;
   adv.SetTime(t);

   // 5. Create the CVODES solver and set the various tolerances

   // Set absolute tolerances for the solution
   Vector abstol_v(3);
   abstol_v[0] = 1.0e-8;
   abstol_v[1] = 1.0e-14;
   abstol_v[2] = 1.0e-6;

   // Initialize the quadrature result
   Vector q(1);
   q = 0.;

   // Create the solver
   CVODESSolver *cvodes = new CVODESSolver(CV_BDF);

   // Initialize the forward problem (this must be done first)
   cvodes->Init(adv);

   // Set error control function
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

   // Set weighted tolerances
   cvodes->SetSVtolerances(reltol, abstol_v);

   // Use the builtin sundials solver for the forward solver
   cvodes->UseSundialsLinearSolver();

   // Initialize the quadrature integration and set the tolerances
   cvodes->InitQuadIntegration(q, 1.e-6, 1.e-6);

   // Initialize the adjoint solve
   cvodes->InitAdjointSolve(150, CV_HERMITE);

   // 6. Perform time-integration (looping over the time iterations, ti,
   //    with a time-step dt).
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = max(dt, t_final - t);
      cvodes->Step(u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done)
      {
	cvodes->PrintInfo();
      }
   }

   cout << "Final Solution: " << t << endl;
   u.Print();

   q = 0.;
   cout << " Final Quadrature " << endl;
   cvodes->EvalQuadIntegration(t, q);
   q.Print();

   // 7. Solve the adjoint problem at different points in time

   // Create the adjoint solution vector
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

   // 8. Free the used memory.
   delete cvodes;

   return 0;
}

// cvsRoberts_ASAi_dns rate equation
void RobertsSUNDIALS::Mult(const Vector &x, Vector &y) const
{
   y[0] = -p_[0]*x[0] + p_[1]*x[1]*x[2];
   y[2] = p_[2]*x[1]*x[1];
   y[1] = -y[0] - y[2];
}

// cvsRoberts_ASAi_dns quadrature rate equation
void RobertsSUNDIALS::QuadratureIntegration(const Vector &y, Vector &qdot) const
{
   qdot[0] = y[2];
}

// cvsRoberts_ASAi_dns adjoint rate equation
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

// cvsRoberts_ASAi_dns quadrature sensitivity rate equation
void RobertsSUNDIALS::QuadratureSensitivityMult(const Vector &y,
                                               const Vector &yB, Vector &qBdot) const
{
   double l21 = (yB[1]-yB[0]);
   double l32 = (yB[2]-yB[1]);
   double y23 = y[1] * y[2];

   qBdot[0] = y[0] * l21;
   qBdot[1] = -y23 * l21;
   qBdot[2] = y[1]*y[1]*l32;
}

// cvsRoberts_ASAi_dns implicit solve setup for adjoint
int RobertsSUNDIALS::SUNImplicitSetupB(const double t, const Vector &y,
                                       const Vector &yB,
                                       const Vector &fyB, int jokB, int *jcurB, double gammaB)
{

   // M = I- gamma J
   // J = dfB/dyB
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
   adjointSolver.SetOperator(*adjointMatrix);

   return 0;
}

// cvsRoberts_ASAi_dns implicit solve for adjoint 
int RobertsSUNDIALS::SUNImplicitSolveB(Vector &x, const Vector &b, double tol)
{
  // The tolerance is ignored in this example as we're trying to replicate CVODES cvsRoberts_ASAi_dns
   adjointSolver.SetRelTol(1e-14);
   adjointSolver.Mult(b, x);
   return (0);
}
