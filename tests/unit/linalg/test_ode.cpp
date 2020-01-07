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

#include "mfem.hpp"
#include "catch.hpp"
#include <cmath>

using namespace mfem;

TEST_CASE("First order ODE methods",
          "[ODE1]")
{
    double tol = 0.1;

   /** Class for simple linear first order ODE.
    *
    *     du/dt  + a u = 0
    *
    */
   class ODE : public TimeDependentOperator
   {
   protected:
      DenseMatrix A;
   public:
      ODE(double a00,double a01,double a10,double a11) : TimeDependentOperator(2, 0.0)
      {
         A.SetSize(2,2);
         A(0,0) = a00;
         A(0,1) = a01;
         A(1,0) = a10;
         A(1,1) = a11;
      };

      virtual void Mult(const Vector &u, Vector &dudt)  const
      {
         A.Mult(u,dudt);
         dudt.Neg();
      }

      virtual void ImplicitSolve(const double dt, const Vector &u, Vector &dudt)
      {
         DenseMatrix I(2,2),T(2,2);
         I =  0.0;
         I(0,0) = I(1,1) = 1.0;
         Add(I,A,dt,T);
         T.Invert();

         Vector r(2);
         A.Mult(u,r);
         r.Neg();
         T.Mult(r,dudt);
      }

      virtual ~ODE() {};
   };

   /** Class for checking order of convergence of first order ODE.
    */
   class CheckODE
   {
   protected:
      int ti_steps,levels;
      Vector u0;
      double t_final,dt;
      ODE *oper;
   public:
      CheckODE()
      {
         oper = new ODE(0.0, 1.0, -1.0, 0.0);
         ti_steps = 100;
         levels   = 6;

         u0.SetSize(2);
         u0    = 1.0;

         t_final = M_PI;
         dt = t_final/double(ti_steps);
      };

      double order(ODESolver* ode_solver)
      {
         ode_solver->Init(*oper);

         double dt,t;
         Vector u(2);
         Vector err(levels);
         int steps = ti_steps;

         t = 0.0;
         dt = t_final/double(steps);
         u = u0;
         for (int ti = 0; ti< steps; ti++)
         {
            ode_solver->Step(u, t, dt);
         }
         u +=u0;
         err[0] = u.Norml2();

         std::cout<<std::setw(12)<<"Error"
                  <<std::setw(12)<<"Ratio"
                  <<std::setw(12)<<"Order"<<std::endl;
         std::cout<<std::setw(12)<<err[0]<<std::endl;

         for (int l = 1; l< levels; l++)
         {
            t = 0.0;
            steps *=2;
            dt = t_final/double(steps);
            u = u0;
            for (int ti = 0; ti< steps; ti++)
            {
               ode_solver->Step(u, t, dt);
            }
            u +=u0;
            err[l] = u.Norml2();
            std::cout<<std::setw(12)<<err[l]
                     <<std::setw(12)<<err[l-1]/err[l]
                     <<std::setw(12)<<log(err[l-1]/err[l])/log(2) <<std::endl;
         }
         delete ode_solver;

         return log(err[levels-2]/err[levels-1])/log(2);
      }
      virtual ~CheckODE() {delete oper;};
   };
   CheckODE check;

   // Implicit L-stable methods
   SECTION("BackwardEuler")
   {
      std::cout <<"\nTesting BackwardEuler" << std::endl;
      REQUIRE(check.order(new BackwardEulerSolver) + tol > 1.0 );
   }

   SECTION("SDIRK23Solver(2)")
   {
      std::cout <<"\nTesting SDIRK23Solver(2)" << std::endl;
      REQUIRE(check.order(new SDIRK23Solver(2)) + tol > 2.0 );
   }

   SECTION("SDIRK33Solver")
   {
      std::cout <<"\nTesting SDIRK33Solver" << std::endl;
      REQUIRE(check.order(new SDIRK33Solver) + tol > 3.0 );
   }

   SECTION("ForwardEulerSolver")
   {
      std::cout <<"\nTesting ForwardEulerSolver" << std::endl;
      REQUIRE(check.order(new ForwardEulerSolver) + tol > 1.0 );
   }

   SECTION("RK2Solver(0.5)")
   {
      std::cout <<"\nTesting RK2Solver(0.5)" << std::endl;
      REQUIRE(check.order(new RK2Solver(0.5))+ tol > 2.0 );
   }

   SECTION("RK3SSPSolver")
   {
      std::cout <<"\nTesting RK3SSPSolver" << std::endl;
      REQUIRE(check.order(new RK3SSPSolver) + tol > 3.0 );
   }

   SECTION("RK4Solver")
   {
      std::cout <<"\nTesting RK4Solver" << std::endl;
      REQUIRE(check.order(new RK4Solver) + tol > 4.0 );
   }

   SECTION("ImplicitMidpointSolver")
   {
      std::cout <<"\nTesting ImplicitMidpointSolver" << std::endl;
      REQUIRE(check.order(new ImplicitMidpointSolver) + tol > 2.0 );
   }

   SECTION("SDIRK23Solver")
   {
      std::cout <<"\nTesting SDIRK23Solver" << std::endl;
      REQUIRE(check.order(new SDIRK23Solver) + tol > 3.0 );
   }

   SECTION("SDIRK34Solver")
   {
      std::cout <<"\nTesting SDIRK34Solver" << std::endl;
      REQUIRE(check.order(new SDIRK34Solver) + tol > 4.0 );
   }

   /*SECTION("GeneralizedAlphaSolver(0.0)")
   {
      std::cout <<"\nTesting GeneralizedAlphaSolver(0.0)" << std::endl;
      REQUIRE(check.order(new GeneralizedAlphaSolver(0.0)) + tol > 2.0 );
   }*/

  /* SECTION("GeneralizedAlphaSolver(0.5)")
   {
      std::cout <<"\nTesting GeneralizedAlphaSolver(0.5)" << std::endl;
      REQUIRE(check.order(new GeneralizedAlphaSolver(0.5)) + tol > 2.0 );
   }*/

   SECTION("GeneralizedAlphaSolver(1.0)")
   {
      std::cout <<"\nTesting GeneralizedAlphaSolver(1.0)" << std::endl;
      REQUIRE(check.order(new GeneralizedAlphaSolver(1.0)) + tol > 2.0 );
   }

   SECTION("AdamsBashforthSolver(1)")
   {
      std::cout <<"\nTesting AdamsBashforthSolver(1)" << std::endl;
      REQUIRE(check.order(new AdamsBashforthSolver(1)) + tol > 1.0 );
   }

   SECTION("AdamsBashforthSolver(2)")
   {
      std::cout <<"\nTesting AdamsBashforthSolver(2)" << std::endl;
      REQUIRE(check.order(new AdamsBashforthSolver(2)) + tol > 2.0 );
   }

   SECTION("AdamsBashforthSolver(3)")
   {
      std::cout <<"\nTesting AdamsBashforthSolver(3)" << std::endl;
      REQUIRE(check.order(new AdamsBashforthSolver(3)) + tol > 3.0 );
   }

   SECTION("AdamsBashforthSolver(4)")
   {
      std::cout <<"\nTesting AdamsBashforthSolver(4)" << std::endl;
      REQUIRE(check.order(new AdamsBashforthSolver(4)) + tol > 4.0 );
   }

   SECTION("AdamsBashforthSolver(5)")
   {
      std::cout <<"\nTesting AdamsBashforthSolver(5)" << std::endl;
      REQUIRE(check.order(new AdamsBashforthSolver(5)) + tol > 5.0 );
   }

   SECTION("AdamsMoultonSolver(1)")
   {
      std::cout <<"\nTesting AdamsMoultonSolver(1)" << std::endl;
      REQUIRE(check.order(new AdamsMoultonSolver(1)) + tol > 1.0 );
   }

   SECTION("AdamsMoultonSolver(2)")
   {
      std::cout <<"\nTesting AdamsMoultonSolver(2)" << std::endl;
      REQUIRE(check.order(new AdamsMoultonSolver(2)) + tol > 2.0 );
   }

   SECTION("AdamsMoultonSolver(3)")
   {
      std::cout <<"\nTesting AdamsMoultonSolver(3)" << std::endl;
      REQUIRE(check.order(new AdamsMoultonSolver(3)) + tol > 3.0 );
   }

   SECTION("AdamsMoultonSolver(4)")
   {
      std::cout <<"\nTesting AdamsMoultonSolver(4)" << std::endl;
      REQUIRE(check.order(new AdamsMoultonSolver(4)) + tol > 4.0 );
   }
}

