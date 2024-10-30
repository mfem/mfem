// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
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
#include "unit_tests.hpp"
#include <cmath>

using namespace mfem;

TEST_CASE("Second order ODE methods", "[ODE]")
{
   real_t tol = 0.1;

   /** Class for simple linear second order ODE.
    *
    *    du2/dt^2 + b du/dt  + a u = 0
    *
    */
   class ODE2 : public SecondOrderTimeDependentOperator
   {
   protected:
      real_t a, b;

   public:
      ODE2(real_t a, real_t b) :
         SecondOrderTimeDependentOperator(1, (real_t) 0.0), a(a), b(b) {};

      using SecondOrderTimeDependentOperator::Mult;
      void Mult(const Vector &u, const Vector &dudt,
                Vector &d2udt2) const override
      {
         d2udt2[0] = -a*u[0] - b*dudt[0];
      }

      using SecondOrderTimeDependentOperator::ImplicitSolve;
      void ImplicitSolve(const real_t fac0, const real_t fac1,
                         const Vector &u, const Vector &dudt,
                         Vector &d2udt2) override
      {
         real_t T = 1.0 + a*fac0 + fac1*b;
         d2udt2[0] = (-a*u[0] - b*dudt[0])/T;
      }

      ~ODE2() override {};
   };

   // Class for checking order of convergence of second order ODE.
   class CheckODE2
   {
   protected:
      int ti_steps,levels;
      Vector u0;
      Vector dudt0;
      real_t t_final,dt;
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
         dt = t_final/real_t(ti_steps);
      };

      void init_hist(SecondOrderODESolver* ode_solver,real_t dt_)
      {
         int nstate = ode_solver->GetState().Size();

         for (int s = 0; s< nstate; s++)
         {
            real_t t = -(s)*dt_;
            Vector uh(1);
            uh[0] = -cos(t) - sin(t);
            ode_solver->GetState().Set(s,uh);
         }
      }

      real_t order(SecondOrderODESolver* ode_solver, bool init_hist_ = false)
      {
         real_t dt_order,t;
         Vector u(1);
         Vector du(1);
         Vector err_u(levels);
         Vector err_du(levels);
         int steps = ti_steps;

         t = 0.0;
         dt_order = t_final/real_t(steps);
         u = u0;
         du = dudt0;
         ode_solver->Init(*oper);
         if (init_hist_) { init_hist(ode_solver,dt_order); }
         ode_solver->Run(u, du, t, dt_order, t_final - 1e-12);

         u -= u0;
         du -= dudt0;

         err_u[0] = u.Norml2();
         err_du[0] = du.Norml2();

         mfem::out<<std::setw(12)<<"Error u"
                  <<std::setw(12)<<"Error du"
                  <<std::setw(12)<<"Ratio u"
                  <<std::setw(12)<<"Ratio du"
                  <<std::setw(12)<<"Order u"
                  <<std::setw(12)<<"Order du"<<std::endl;
         mfem::out<<std::setw(12)<<err_u[0]
                  <<std::setw(12)<<err_du[0]<<std::endl;

         std::vector<Vector> uh(ode_solver->GetState().MaxSize());
         for (int l = 1; l< levels; l++)
         {
            int lvl = static_cast<int>(pow(2,l));
            t = 0.0;
            dt_order *= 0.5;
            u = u0;
            du = dudt0;
            ode_solver->Init(*oper);
            if (init_hist_) { init_hist(ode_solver,dt_order); }

            // Instead of single run command:
            // ode_solver->Run(u, du, t, dt_order, t_final - 1e-12);
            // Chop-up sequence with Get/Set in between
            // in order to test these routines
            for (int ti = 0; ti < steps; ti++)
            {
               ode_solver->Step(u, du, t, dt_order);
            }

            int nstate = ode_solver->GetState().Size();
            for (int s = 0; s < nstate; s++)
            {
               uh[s] = ode_solver->GetState().Get(s);
            }

            for (int ll = 1; ll < lvl; ll++)
            {
               for (int s = 0; s < nstate; s++)
               {
                  ode_solver->GetState().Set(s,uh[s]);
               }
               for (int ti = 0; ti < steps; ti++)
               {
                  ode_solver->Step(u, du, t, dt_order);
               }
               nstate = ode_solver->GetState().Size();
               for (int s = 0; s< nstate; s++)
               {
                  uh[s] = ode_solver->GetState().Get(s);
               }
            }

            u -= u0;
            du -= dudt0;
            err_u[l] = u.Norml2();
            err_du[l] = du.Norml2();
            mfem::out<<std::setw(12)<<err_u[l]
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
      real_t conv_rate = check.order(new NewmarkSolver);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("LinearAcceleration")
   {
      real_t conv_rate = check.order(new LinearAccelerationSolver);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("CentralDifference")
   {
      real_t conv_rate = check.order(new CentralDifferenceSolver);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("FoxGoodwin")
   {
      real_t conv_rate = check.order(new FoxGoodwinSolver);
      REQUIRE(conv_rate + tol > 4.0);
   }

   // Generalized-alpha based solvers
   SECTION("GeneralizedAlpha(0.0)")
   {
      real_t conv_rate = check.order(new GeneralizedAlpha2Solver(0.0));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("GeneralizedAlpha(0.5)")
   {
      real_t conv_rate = check.order(new GeneralizedAlpha2Solver(0.5));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("GeneralizedAlpha(0.5) - restart")
   {
      real_t conv_rate = check.order(new GeneralizedAlpha2Solver(0.5),true);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("GeneralizedAlpha(1.0)")
   {
      real_t conv_rate = check.order(new GeneralizedAlpha2Solver(1.0));
      REQUIRE(conv_rate + tol > 2.0);
   }


   SECTION("AverageAcceleration")
   {
      real_t conv_rate = check.order(new AverageAccelerationSolver);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("HHTAlpha(2/3)")
   {
      real_t conv_rate = check.order(new HHTAlphaSolver(2.0/3.0));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("HHTAlpha(0.75)")
   {
      real_t conv_rate = check.order(new HHTAlphaSolver(0.75));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("HHTAlpha(1.0)")
   {
      real_t conv_rate = check.order(new HHTAlphaSolver(1.0));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("WBZAlpha(0.0)")
   {
      real_t conv_rate = check.order(new WBZAlphaSolver(0.0));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("WBZAlpha(0.5)")
   {
      real_t conv_rate = check.order(new WBZAlphaSolver(0.5));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("WBZAlpha(1.0)")
   {
      real_t conv_rate = check.order(new WBZAlphaSolver(1.0));
      REQUIRE(conv_rate + tol > 2.0);
   }
}
