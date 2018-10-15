// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
//
//      ----------------------------------------------------------------
//      Check timeintegrator Miniapp:
//      ----------------------------------------------------------------
//
//
// Compile with: make check-timeintegrator
//
// Sample runs:  check-timeintegrator
//               check-timeintegrator -s 10
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** Class for simple linear second order ODE.
 *
 *    du2/dt^2 + b du/dt  + a u = 0
 *
 */
class ODE2 : public TimeDependent2Operator
{
protected:
   double a,b;

public:
   ODE2(double a_, double b_) : TimeDependent2Operator(1, 0.0),  a(a_), b(b_) {};

   virtual void ExplicitSolve(const Vector &u, const Vector &dudt,
                              Vector &d2udt2) const;

   virtual void ImplicitSolve(const double fac0, const double fac1,
                              const Vector &u, const Vector &dudt, Vector &d2udt2);

   virtual ~ODE2() {};
};

void ODE2::ExplicitSolve(const Vector &u, const Vector &dudt,
                         Vector &d2udt2)  const
{
   d2udt2[0] = -a*u[0] - b*dudt[0];
}

void ODE2::ImplicitSolve(const double fac0, const double fac1,
                         const Vector &u, const Vector &dudt, Vector &d2udt2)
{
   double T = 1.0 + a*fac0 + fac1*b;
   d2udt2[0] = (-a*u[0] - b*dudt[0])/T;
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ode_solver_type = 10;
   double t_final = 0.5;
   double dt = 1.0e-2;
   double a = 1.0;
   double b = 0.0;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 11/21 - Average Acceleration, 12/22 Linear Acceleration,\n\t"
                  "13/23 - Central Difference, 14/24 - Fox-Goodwin,\n\t"
                  "0 -- 10 Generalized-alpha,\n\t"
                  "30 -- 40 Hilber-Hughes-Taylor-alpha,\n\t"
                  "50 -- 60 Wood-Bossak-Zienkiewicz-alpha");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&a, "-a", "--stiffness",
                  "Coefficient.");
   args.AddOption(&b, "-b", "--damping",
                  "Coefficient.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODE2Solver *ode_solver;
   switch (ode_solver_type)
   {
      // Generalized-alpha solvers
      case 0:  ode_solver = new GeneralizedAlpha2Solver(0.0); break;
      case 1:  ode_solver = new GeneralizedAlpha2Solver(0.1); break;
      case 2:  ode_solver = new GeneralizedAlpha2Solver(0.2); break;
      case 3:  ode_solver = new GeneralizedAlpha2Solver(0.3); break;
      case 4:  ode_solver = new GeneralizedAlpha2Solver(0.4); break;
      case 5:  ode_solver = new GeneralizedAlpha2Solver(0.5); break;
      case 6:  ode_solver = new GeneralizedAlpha2Solver(0.6); break;
      case 7:  ode_solver = new GeneralizedAlpha2Solver(0.7); break;
      case 8:  ode_solver = new GeneralizedAlpha2Solver(0.8); break;
      case 9:  ode_solver = new GeneralizedAlpha2Solver(0.9); break;
      case 10: ode_solver = new GeneralizedAlpha2Solver(1.0); break;

      // Newmark solvers
      case 11: ode_solver = new AverageAccelerationSolver(); break;
      case 12: ode_solver = new LinearAccelerationSolver();  break;
      case 13: ode_solver = new CentralDifferenceSolver();   break;
      case 14: ode_solver = new FoxGoodwinSolver();          break;

      // Newmark solvers --> as special case of gen-alpha
      case 21: ode_solver = new NewmarkSolver(0.25,     0.5); break;
      case 22: ode_solver = new NewmarkSolver(1.0/6.0,  0.5); break;
      case 23: ode_solver = new NewmarkSolver(0.0,      0.5); break;
      case 24: ode_solver = new NewmarkSolver(1.0/12.0, 0.5); break;

      // Hilber-Hughes-Taylor solvers --> as special case of gen-alpha
      case 30:  ode_solver = new HHTAlphaSolver(0.0); break;
      case 31:  ode_solver = new HHTAlphaSolver(0.1); break;
      case 32:  ode_solver = new HHTAlphaSolver(0.2); break;
      case 33:  ode_solver = new HHTAlphaSolver(0.3); break;
      case 34:  ode_solver = new HHTAlphaSolver(0.4); break;
      case 35:  ode_solver = new HHTAlphaSolver(0.5); break;
      case 36:  ode_solver = new HHTAlphaSolver(0.6); break;
      case 37:  ode_solver = new HHTAlphaSolver(0.7); break;
      case 38:  ode_solver = new HHTAlphaSolver(0.8); break;
      case 39:  ode_solver = new HHTAlphaSolver(0.9); break;
      case 40:  ode_solver = new HHTAlphaSolver(1.0); break;

      // Wood-Bossak-Zienkiewicz solvers --> as special case of gen-alpha
      case 50:  ode_solver = new WBZAlphaSolver(0.0); break;
      case 51:  ode_solver = new WBZAlphaSolver(0.1); break;
      case 52:  ode_solver = new WBZAlphaSolver(0.2); break;
      case 53:  ode_solver = new WBZAlphaSolver(0.3); break;
      case 54:  ode_solver = new WBZAlphaSolver(0.4); break;
      case 55:  ode_solver = new WBZAlphaSolver(0.5); break;
      case 56:  ode_solver = new WBZAlphaSolver(0.6); break;
      case 57:  ode_solver = new WBZAlphaSolver(0.7); break;
      case 58:  ode_solver = new WBZAlphaSolver(0.8); break;
      case 59:  ode_solver = new WBZAlphaSolver(0.9); break;
      case 60:  ode_solver = new WBZAlphaSolver(1.0); break;

      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }
   ode_solver->PrintProperties();



   // 3. Set the initial conditions for u.
   Vector u(1);
   Vector dudt(1);

   u    = 1.0;
   dudt = 0.0;

   // 4. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ODE2 oper(a,b);
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   ofstream output("output.dat");
   output<<t<<" "<<u[0]<<" "<<dudt[0]<<endl;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, dudt, t, dt);
      output<<t<<" "<<u[0]<<" "<<dudt[0]<<endl;
   }
   output.close();

   // 5. Free the used memory.
   delete ode_solver;

   return 0;
}

