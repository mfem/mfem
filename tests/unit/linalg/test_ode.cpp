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

TEST_CASE("First order ODE methods", "[ODE]")
{
   real_t tol = 0.1;

   // Class for simple linear first order ODE.
   //    du/dt  + a u = 0
   class ODE : public TimeDependentOperator
   {
   protected:
      DenseMatrix A,I,T;
      Vector r;
   public:
      ODE(real_t a00,real_t a01,real_t a10,real_t a11) :
         TimeDependentOperator(2, (real_t) 0.0)
      {
         A.SetSize(2,2);
         I.SetSize(2,2);
         T.SetSize(2,2);
         r.SetSize(2);

         A(0,0) = a00;
         A(0,1) = a01;
         A(1,0) = a10;
         A(1,1) = a11;

         I =  0.0;
         I(0,0) = I(1,1) = 1.0;
      };

      void Mult(const Vector &u, Vector &dudt)  const override
      {
         A.Mult(u,dudt);
         dudt.Neg();
      }

      void ImplicitSolve(const real_t dt, const Vector &u, Vector &dudt) override
      {
         // Residual
         A.Mult(u,r);
         r.Neg();

         // Jacobian
         Add(I,A,dt,T);

         // Solve
         T.Invert();

         T.Mult(r,dudt);
      }

      ~ODE() override {};
   };

   // Class for checking order of convergence of first order ODE.
   class CheckODE
   {
   protected:
      int ti_steps,levels;
      Vector u0;
      real_t t_final,dt;
      ODE *oper;
   public:
      CheckODE()
      {
         oper = new ODE(0.0, 1.0, -1.0, 0.0);
         ti_steps = 2;
         levels   = 6;

         u0.SetSize(2);
         u0    = 1.0;

         t_final = M_PI;
         dt = t_final/real_t(ti_steps);
      };

      void init_hist(ODESolver* ode_solver,real_t dt_)
      {
         int nstate = ode_solver->GetStateSize();

         for (int s = 0; s< nstate; s++)
         {
            real_t t = -(s)*dt_;
            Vector uh(2);
            uh[0] = -cos(t) - sin(t);
            uh[1] =  cos(t) - sin(t);
            ode_solver->SetStateVector(s,uh);
         }
      }

      real_t order(ODESolver* ode_solver, bool init_hist_ = false)
      {
         real_t dt_order,t;
         Vector u(2);
         Vector error(levels);
         int steps = ti_steps;

         t = 0.0;
         dt_order = t_final/real_t(steps);
         u = u0;
         ode_solver->Init(*oper);
         if (init_hist_) { init_hist(ode_solver,dt_order); }
         ode_solver->Run(u, t, dt_order, t_final - 1e-12);
         u +=u0;
         error[0] = u.Norml2();

         mfem::out<<std::setw(12)<<"Error"
                  <<std::setw(12)<<"Ratio"
                  <<std::setw(12)<<"Order"<<std::endl;
         mfem::out<<std::setw(12)<<error[0]<<std::endl;

         std::vector<Vector> uh(ode_solver->GetMaxStateSize());
         for (int l = 1; l < levels; l++)
         {
            int lvl = static_cast<int>(pow(2,l));
            t = 0.0;
            dt_order *= 0.5;
            u = u0;
            ode_solver->Init(*oper);
            if (init_hist_)
            {
               init_hist(ode_solver,dt_order);

               // Instead of single run command
               // Chop-up sequence with Get/Set in between
               // in order to test these routines
               for (int ti = 0; ti < steps; ti++)
               {
                  ode_solver->Step(u, t, dt_order);
               }
               int nstate = ode_solver->GetStateSize();
               for (int s = 0; s < nstate; s++)
               {
                  ode_solver->GetStateVector(s,uh[s]);
               }

               for (int ll = 1; ll < lvl; ll++)
               {
                  for (int s = 0; s < nstate; s++)
                  {
                     ode_solver->SetStateVector(s,uh[s]);
                  }
                  for (int ti = 0; ti < steps; ti++)
                  {
                     ode_solver->Step(u, t, dt_order);
                  }
                  nstate = ode_solver->GetStateSize();
                  for (int s = 0; s< nstate; s++)
                  {
                     uh[s] = ode_solver->GetStateVector(s);
                  }
               }
            }
            else
            {
               ode_solver->Run(u, t, dt_order, t_final-1e-12);
            }

            u += u0;
            error[l] = u.Norml2();
            mfem::out<<std::setw(12)<<error[l]
                     <<std::setw(12)<<error[l-1]/error[l]
                     <<std::setw(12)<<log(error[l-1]/error[l])/log(2)
                     <<std::endl;
         }
         delete ode_solver;

         return log(error[levels-2]/error[levels-1])/log(2);
      }
      virtual ~CheckODE() {delete oper;};
   };
   CheckODE check;

   // Implicit L-stable methods
   SECTION("BackwardEuler")
   {
      real_t conv_rate = check.order(new BackwardEulerSolver);
      REQUIRE(conv_rate + tol > 1.0);
   }

   SECTION("SDIRK23Solver(2)")
   {
      real_t conv_rate = check.order(new SDIRK23Solver(2));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("SDIRK33Solver")
   {
      real_t conv_rate = check.order(new SDIRK33Solver);
      REQUIRE(conv_rate + tol > 3.0);
   }

   SECTION("ForwardEulerSolver")
   {
      real_t conv_rate = check.order(new ForwardEulerSolver);
      REQUIRE(conv_rate + tol > 1.0);
   }

   SECTION("RK2Solver(0.5)")
   {
      real_t conv_rate = check.order(new RK2Solver(0.5));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("RK3SSPSolver")
   {
      real_t conv_rate = check.order(new RK3SSPSolver);
      REQUIRE(conv_rate + tol > 3.0);
   }

   SECTION("RK4Solver")
   {
      real_t conv_rate = check.order(new RK4Solver);
      REQUIRE(conv_rate + tol > 4.0);
   }

   SECTION("ImplicitMidpointSolver")
   {
      real_t conv_rate = check.order(new ImplicitMidpointSolver);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("SDIRK23Solver")
   {
      real_t conv_rate = check.order(new SDIRK23Solver);
      REQUIRE(conv_rate + tol > 3.0);
   }

   SECTION("SDIRK34Solver")
   {
      real_t conv_rate = check.order(new SDIRK34Solver);
      REQUIRE(conv_rate + tol > 4.0);
   }

   SECTION("TrapezoidalRuleSolver")
   {
      REQUIRE(check.order(new TrapezoidalRuleSolver) + tol > 2.0 );
   }

   SECTION("ESDIRK32Solver")
   {
      REQUIRE(check.order(new ESDIRK32Solver) + tol > 2.0 );
   }

   SECTION("ESDIRK33Solver")
   {
      REQUIRE(check.order(new ESDIRK33Solver) + tol > 3.0 );
   }

   // Generalized-alpha
   SECTION("GeneralizedAlphaSolver(1.0)")
   {
      real_t conv_rate = check.order(new GeneralizedAlphaSolver(1.0));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("GeneralizedAlphaSolver(0.5)")
   {
      real_t conv_rate = check.order(new GeneralizedAlphaSolver(0.5));
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("GeneralizedAlphaSolver(0.5) - restart")
   {
      real_t conv_rate = check.order(new GeneralizedAlphaSolver(0.5), true);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("GeneralizedAlphaSolver(0.0)")
   {
      real_t conv_rate = check.order(new GeneralizedAlphaSolver(0.0));
      REQUIRE(conv_rate + tol > 2.0);
   }

   // Adams-Bashforth
   SECTION("AB1Solver()")
   {
      real_t conv_rate = check.order(new AB1Solver());
      REQUIRE(conv_rate + tol > 1.0);
   }

   SECTION("AB1Solver() - restart")
   {
      real_t conv_rate = check.order(new AB1Solver(), true);
      REQUIRE(conv_rate + tol > 1.0);
   }

   SECTION("AB2Solver()")
   {
      real_t conv_rate = check.order(new AB2Solver());
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("AB2Solver() - restart")
   {
      real_t conv_rate = check.order(new AB2Solver(), true);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("AB3Solver()")
   {
      real_t conv_rate = check.order(new AB3Solver());
      REQUIRE(conv_rate + tol > 3.0);
   }

   SECTION("AB4Solver()")
   {
      real_t conv_rate = check.order(new AB4Solver());
      REQUIRE(conv_rate + tol > 4.0);
   }

   SECTION("AB5Solver()")
   {
      real_t conv_rate = check.order(new AB5Solver());
      REQUIRE(conv_rate + tol > 5.0);
   }

   SECTION("AB5Solver() - restart")
   {
      real_t conv_rate = check.order(new AB5Solver(), true);
      REQUIRE(conv_rate + tol > 5.0);
   }

   // Adams-Moulton
   SECTION("AM0Solver()")
   {
      real_t conv_rate = check.order(new AM0Solver());
      REQUIRE(conv_rate + tol > 1.0);
   }

   SECTION("AM1Solver()")
   {
      real_t conv_rate = check.order(new AM1Solver());
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("AM1Solver() - restart")
   {
      real_t conv_rate = check.order(new AM1Solver(), true);
      REQUIRE(conv_rate + tol > 2.0);
   }

   SECTION("AM2Solver()")
   {
      real_t conv_rate = check.order(new AM2Solver());
      REQUIRE(conv_rate + tol > 3.0);
   }

   SECTION("AM2Solver() - restart")
   {
      real_t conv_rate = check.order(new AM2Solver(), true);
      REQUIRE(conv_rate + tol > 1.0);
   }

   SECTION("AM3Solver()")
   {
      real_t conv_rate = check.order(new AM3Solver());
      REQUIRE(conv_rate + tol > 4.0);
   }

   SECTION("AM4Solver()")
   {
      real_t conv_rate = check.order(new AM4Solver());
      REQUIRE(conv_rate + tol > 5.0);
   }

   SECTION("AM4Solver() - restart")
   {
      real_t conv_rate = check.order(new AM4Solver(),true);
      REQUIRE(conv_rate + tol > 5.0);
   }
}
