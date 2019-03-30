//                                MFEM Example 2x
//
// Compile with: make ex2x
//
// Sample runs:  ex2x
//
// Dexcription:
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/*
class InitialCondition : public VectorCoefficient
{
private:
  int problem_;
  double beta_;

public:
  InitialCondition(int problem);

  void Eval(Vector &V, ElementTransformation &T,
       const IntegrationPoint &ip);
};
*/
void InitialCondition(int problem, Vector &y);

class ExampleTDO : public TimeDependentOperator
{
private:
   int problem_;
   double beta_;

public:
   ExampleTDO(int problem);
   void Mult(const Vector &y, Vector &dydt) const;
};

int main(int argc, char *argv[])
{
   int prob = 1;
   int ode_solver_type = 3;
   int ode_msr_type = 1;
   int ode_acc_type = 2;
   int ode_rej_type = 1;
   int ode_lim_type = 1;

   double t_init  = 0.0;
   double t_final = 1.0;;
   double dt = -1.0;
   double tol = 1e-2;
   double rho = 1.2;

   OptionsParser args(argc, argv);
   args.AddOption(&prob, "-p", "--problem-type",
                  "Problem Type From Gustafsson 1988:  1 - 7");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Initial time step size.");
   args.AddOption(&tol, "-tol", "--tolerance",
                  "Tolerance.");
   args.AddOption(&rho, "-rho", "--rejection",
                  "Rejection tolerance.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   ODEController ode_controller;

   ODESolver                * ode_solver   = NULL;
   ODEErrorMeasure          * ode_err_msr  = NULL;
   ODEStepAdjustmentFactor  * ode_step_acc = NULL;
   ODEStepAdjustmentFactor  * ode_step_rej = NULL;
   ODEStepAdjustmentLimiter * ode_step_lim = NULL;

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
      case 15: ode_solver = new GeneralizedAlphaSolver(0.5); break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   ExampleTDO tdo(prob);
   ode_solver->Init(tdo);

   ode_controller.Init(*ode_solver, *ode_err_msr,
                       *ode_step_acc, *ode_step_rej, *ode_step_lim);

   ode_controller.SetTimeStep(dt);
   ode_controller.SetTolerance(tol);
   ode_controller.SetRejectionLimit(rho);

   Vector y;
   InitialCondition(prob, y);

   ode_controller.Run(y, t_init, t_final);

   delete ode_solver;
   delete ode_err_msr;
   delete ode_step_acc;
   delete ode_step_rej;
   delete ode_step_lim;
}

/*
InitialCondition::InitialCondition(int problem)
  : VectorCoefficient(0),
    problem_(problem),
    beta_(0.0)
{
  switch(problem_)
  {
  case 1:
  case 2:
    vdim = 4;
    break;
  case 3:
  case 6:
    vdim = 3;
    break;
  case 4:
  case 5:
  case 7:
    vdim = 2;
    beta_ = 3.0;
    break;
  case 8:
    vdim = 2;
    beta_ = 8.533;
    break;
  default:
    vdim = 0;
    break;
  }
}

void InitialCondition::Eval(Vector &v, ElementTransformation &T,
             const IntegrationPoint &ip)
{
  v.SetSize(vdim);

  switch(problem_)
  {
  case 1:
    v(0) = v(2) = 1.0;
    v(1) = v(3) = 0.0;
    break;
  case 2:
    v = 1.0;
    break;
  case 3:
    v(0) = 1.0; v(1) = v(2) = 0.0;
    break;
  case 4:
  case 5:
    v(0) = 2.0; v(1) = 0.0;
    break;
  case 6:
    v(0) = v(1) = 1.0; v(2) = 0.0;
    break;
  case 7:
  case 8:
    v(0) = 1.3; v(1) = beta_;
    break;
  default:
    v = 0.0;
    break;
  }
}
*/

void InitialCondition(int problem, Vector &y)
{
   switch (problem)
   {
      case 1:
         y.SetSize(4);
         y(0) = y(2) = 1.0;
         y(1) = y(3) = 0.0;
         break;
      case 2:
         y.SetSize(4);
         y = 1.0;
         break;
      case 3:
         y.SetSize(3);
         y(0) = 1.0; y(1) = y(2) = 0.0;
         break;
      case 4:
      case 5:
         y.SetSize(2);
         y(0) = 2.0; y(1) = 0.0;
         break;
      case 6:
         y.SetSize(3);
         y(0) = y(1) = 1.0; y(2) = 0.0;
         break;
      case 7:
         y.SetSize(2);
         y(0) = 1.3; y(1) = 3.0;
         break;
      case 8:
         y.SetSize(2);
         y(0) = 1.3; y(1) = 8.533;
         break;
      default:
         y = 0.0;
         break;
   }
}

ExampleTDO::ExampleTDO(int problem)
   : problem_(problem)
{
   switch (problem_)
   {
      case 1:
         height = width = 4;
         break;
      case 2:
         height = width = 4;
         beta_ = 0.1;
         break;
      case 3:
         height = width = 3;
         break;
      case 4:
         height = width = 2;
         break;
      case 5:
         height = width = 2;
         break;
      case 6:
         height = width = 3;
         break;
      case 7:
         height = width = 2;
         beta_ = 3.0;
         break;
      case 8:
         height = width = 2;
         beta_ = 8.533;
         break;
      default:
         beta_ = 0.0;
   }
}

void ExampleTDO::Mult(const Vector &y, Vector & dydt) const
{
   dydt.SetSize(y.Size());

   switch (problem_)
   {
      case 1:
         dydt(0) =     -1.0 * y(0) +         y(1);
         dydt(1) =   -100.0 * y(0) -         y(1);
         dydt(2) =   -100.0 * y(2) +         y(3);
         dydt(3) = -10000.0 * y(2) - 100.0 * y(3);
         break;
      case 2:
         dydt(0) =   -1.0 * y(0) + 2.0;
         dydt(1) =  -10.0 * y(1) + beta_ * y(0) * y(0);
         dydt(2) =  -40.0 * y(2) + 4.0 * beta_ * (y(0) * y(0) + y(1) * y(1));
         dydt(3) = -100.0 * y(3) +
                   10.0 * beta_ * (y(0) * y(0) + y(1) * y(1) + y(2) * y(2));
         break;
      case 3:
         dydt(0) =  -0.04 * y(0) + 0.01 * y(1) * y(2);
         dydt(1) = 400.0  * y(0) - 100.0 * y(1) * (y(2) + 30.0 * y(1));
         dydt(2) =  30.0  * y(1) * y(1);
         break;
      case 4:
         dydt(0) = y(1);
         dydt(1) = (1.0 - y(0) * y(0)) * y(1) - y(0);
         break;
      case 5:
         dydt(0) = y(1);
         dydt(1) = 50.0 * (1.0 - y(0) * y(0)) * y(1) - 10.0 * y(0);
         break;
      case 6:
         dydt(0) = -(55.0 + y(2)) * y(0) + 65.0 * y(1);
         dydt(1) = 0.0785 * (y(0) - y(1));
         dydt(2) =    0.1 * y(0);
         break;
      case 7:
      case 8:
         dydt(0) = 1.0 + y(0) * y(0) * y(1) - (beta_ + 1.0) * y(0);
         dydt(1) = y(0) * (beta_ - y(0) * y(1));
         break;
      default:
         dydt = 0.0;
         break;
   }
}

