// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "catch.hpp"
#include <cmath>

using namespace mfem;

TEST_CASE("Second order ODE methods",
          "[ODE2]")
{
   double tol = 0.1;

   /** Class for simple linear second order ODE.
    *
    *    du2/dt^2 + b du/dt  + a u = 0
    *
    */
   class ODE2 : public SecondOrderTimeDependentOperator
   {
   protected:
      double a, b;

   public:
      ODE2(double a, double b) :
         SecondOrderTimeDependentOperator(1, 0.0), a(a), b(b) {};

      using SecondOrderTimeDependentOperator::Mult;
      virtual void Mult(const Vector &u, const Vector &dudt,
                        Vector &d2udt2) const
      {
         d2udt2[0] = -a*u[0] - b*dudt[0];
      }

      using SecondOrderTimeDependentOperator::ImplicitSolve;
      virtual void ImplicitSolve(const double fac0, const double fac1,
                                 const Vector &u, const Vector &dudt,
                                 Vector &d2udt2)
      {
         double T = 1.0 + a*fac0 + fac1*b;
         d2udt2[0] = (-a*u[0] - b*dudt[0])/T;
      }

      virtual ~ODE2() {};
   };

   // Class for checking order of convergence of second order ODE.
   class CheckODE2
   {
   protected:
      int ti_steps,levels;
      Vector u0;
      Vector dudt0;
      double t_final,dt;
      ODE2 *oper;
   public:
      CheckODE2()
      {
         oper = new ODE2(1.0, 0.0);
         ti_steps = 20;
         levels   = 5;

         u0.SetSize(1);
         u0    = 1.0;

         dudt0.SetSize(1);
         dudt0  = 1.0;

         t_final = 2*M_PI;
         dt = t_final/double(ti_steps);
      };

      double order(SecondOrderODESolver* ode_solver)
      {
         double dt,t;
         Vector u(1);
         Vector du(1);
         Vector err_u(levels);
         Vector err_du(levels);
         int steps = ti_steps;

         t = 0.0;
         dt = t_final/double(steps);
         u = u0;
         du = dudt0;
         ode_solver->Init(*oper);
         for (int ti = 0; ti< steps; ti++)
         {
            ode_solver->Step(u, du, t, dt);
         }
         u -= u0;
         du -= dudt0;

         err_u[0] = u.Norml2();
         err_du[0] = du.Norml2();

         std::cout<<std::setw(12)<<"Error u"
                  <<std::setw(12)<<"Error du"
                  <<std::setw(12)<<"Ratio u"
                  <<std::setw(12)<<"Ratio du"
                  <<std::setw(12)<<"Order u"
                  <<std::setw(12)<<"Order du"<<std::endl;
         std::cout<<std::setw(12)<<err_u[0]
                  <<std::setw(12)<<err_du[0]<<std::endl;

         for (int l = 1; l< levels; l++)
         {
            t = 0.0;
            steps *=2;
            dt = t_final/double(steps);
            u = u0;
            du = dudt0;
            ode_solver->Init(*oper);
            for (int ti = 0; ti< steps; ti++)
            {
               ode_solver->Step(u, du, t, dt);
            }
            u -= u0;
            du -= dudt0;
            err_u[l] = u.Norml2();
            err_du[l] = du.Norml2();
            std::cout<<std::setw(12)<<err_u[l]
                     <<std::setw(12)<<err_du[l]
                     <<std::setw(12)<<err_u[l-1]/err_u[l]
                     <<std::setw(12)<<err_du[l-1]/err_du[l]
                     <<std::setw(12)<<log(err_u[l-1]/err_u[l])/log(2)
                     <<std::setw(12)<<log(err_du[l-1]/err_du[l])/log(2) <<std::endl;
         }
         delete ode_solver;

         return log(err_u[levels-2]/err_u[levels-1])/log(2);
      }
      virtual ~CheckODE2() {delete oper;};
   };
   CheckODE2 check;

   // Newmark-based solvers
   SECTION("Newmark")
   {
      std::cout <<"\nTesting NewmarkSolver" << std::endl;
      REQUIRE(check.order(new NewmarkSolver) + tol > 2.0 );
   }

   SECTION("LinearAcceleration")
   {
      std::cout <<"\nLinearAccelerationSolver" << std::endl;
      REQUIRE(check.order(new LinearAccelerationSolver) + tol > 2.0 );
   }

   SECTION("CentralDifference")
   {
      std::cout <<"\nTesting CentralDifference" << std::endl;
      REQUIRE(check.order(new CentralDifferenceSolver) + tol > 2.0 );
   }

   SECTION("FoxGoodwin")
   {
      std::cout <<"\nTesting FoxGoodwin" << std::endl;
      REQUIRE(check.order(new FoxGoodwinSolver) + tol > 4.0 );
   }

   // Generalized-alpha based solvers
   SECTION("GeneralizedAlpha(0.0)")
   {
      std::cout <<"\nTesting GeneralizedAlpha(0.0)" << std::endl;
      REQUIRE(check.order(new GeneralizedAlpha2Solver(0.0)) + tol > 2.0 );
   }

   SECTION("GeneralizedAlpha(0.5)")
   {
      std::cout <<"\nTesting GeneralizedAlpha(0.5)" << std::endl;
      REQUIRE(check.order(new GeneralizedAlpha2Solver(0.5)) + tol > 2.0 );
   }

   SECTION("GeneralizedAlpha(1.0)")
   {
      std::cout <<"\nTesting GeneralizedAlpha(1.0)" << std::endl;
      REQUIRE(check.order(new GeneralizedAlpha2Solver(1.0)) + tol > 2.0 );
   }


   SECTION("AverageAcceleration")
   {
      std::cout <<"\nTesting AverageAcceleration" << std::endl;
      REQUIRE(check.order(new AverageAccelerationSolver) + tol > 2.0 );
   }

   SECTION("HHTAlpha(2/3)")
   {
      std::cout <<"\nTesting HHTAlpha(2/3)" << std::endl;
      REQUIRE(check.order(new HHTAlphaSolver(2.0/3.0)) + tol > 2.0 );
   }

   SECTION("HHTAlpha(0.75)")
   {
      std::cout <<"\nTesting HHTAlpha(0.75)" << std::endl;
      REQUIRE(check.order(new HHTAlphaSolver(0.75)) + tol > 2.0 );
   }

   SECTION("HHTAlpha(1.0)")
   {
      std::cout <<"\nTesting HHTAlpha(1.0)" << std::endl;
      REQUIRE(check.order(new HHTAlphaSolver(1.0)) + tol > 2.0 );
   }

   SECTION("WBZAlpha(0.0)")
   {
      std::cout <<"\nTesting WBZAlpha(0.0)" << std::endl;
      REQUIRE(check.order(new WBZAlphaSolver(0.0)) + tol > 2.0 );
   }

   SECTION("WBZAlpha(0.5)")
   {
      std::cout <<"\nTesting WBZAlpha(0.5)" << std::endl;
      REQUIRE(check.order(new WBZAlphaSolver(0.5)) + tol > 2.0 );
   }

   SECTION("WBZAlpha(1.0)")
   {
      std::cout <<"\nTesting WBZAlpha(1.0)" << std::endl;
      REQUIRE(check.order(new WBZAlphaSolver(1.0)) + tol > 2.0 );
   }
}
