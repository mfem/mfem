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


/** Class for simple linear first order ODE.
 *
 *     du/dt  + a u = f
 *
 */
class ODE : public TimeDependentOperator
{
protected:
   double a;

public:
   ODE(double a_) :  TimeDependentOperator(1, 0.0), a(a_) {};

   virtual void Mult(const Vector &u, Vector &dudt) const;

   virtual void ImplicitSolve(const double dt,const Vector &u, Vector &dudt);

   virtual ~ODE() {};
};

void ODE::Mult(const Vector &u, Vector &dudt)  const
{
   // f
   dudt[0] = -a*u[0];
}

void ODE::ImplicitSolve(const double dt, const Vector &u, Vector &dudt)
{
   // f
   double T = 1.0 + a*dt;
   dudt[0] = -a*u[0]/T;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   int ode_solver_type = 11;
   double t_final = 0.5;
   double dt = 1.0e-2;
   double a = 1.0;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4, \n"
                  "\t   99 - Generalized alpha");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&a, "-a", "--stiffness",
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
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1: ode_solver = new BackwardEulerSolver; break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK33Solver; break;
      // Explicit methods
      case 11: ode_solver = new ForwardEulerSolver; break;
      case 12: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 13: ode_solver = new RK3SSPSolver; break;
      case 14: ode_solver = new RK4Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

   // 3. Set the initial conditions for u.
   Vector u(1);
   u    = 1.0;

   // 4. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ODE oper(a);
   ode_solver->Init(oper);
   double t = 0.0;

   bool last_step = false;
   ofstream output("output.dat");
   output<<t<<" "<<u[0]<<endl;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, t, dt);
      output<<t<<" "<<u[0]<<endl;
   }
   output.close();

   // 5. Free the used memory.
   delete ode_solver;

   return 0;
}

