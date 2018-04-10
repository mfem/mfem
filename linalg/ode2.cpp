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

#include "operator.hpp"
#include "ode2.hpp"

namespace mfem
{

void NewmarkSolver::Init(TimeDependent2Operator &_f)
{
   ODE2Solver::Init(_f);
   d2xdt2.SetSize(f->Width());
   d2xdt2 = 0.0;
   first = true;
}


void NewmarkSolver::PrintProperties(std::ostream &out)
{
   out << "Newmark time integrator:" << std::endl;
   out << "beta    = " << beta  << std::endl;
   out << "gamma   = " << gamma << std::endl;

   if (gamma == 0.5)
   {
      out<<"Second order"<<" and ";
   }
   else
   {
      out<<"First order"<<" and ";
   }

   if ((gamma >= 0.5) && (beta >= (gamma + 0.5)*(gamma + 0.5)/4))
   {
      out<<"A-Stable"<<std::endl;
   }
   else if ((gamma >= 0.5) && (beta >= 0.5*gamma)
   {
      out<<"Conditionally stable"<<std::endl;
   }
   else
   {
      out<<"Unstable"<<std::endl;
   }
}


void NewmarkSolver::Step(Vector &x, Vector &dxdt,  double &t, double &dt)
{
   double fac0 = 0.5 - beta;
   double fac1 = 1.0 - gamma;
   double fac2 = beta;
   double fac3 = gamma;

   // In the first pass d2xdt2 is not yet computed. If parameter choices requires
   // d2xdt2  backward euler is used instead for the first step only.
   if (first && !(fac0*fac1 == 0.0))
   {
      fac0 = 0.0;
      fac1 = 0.0;
      fac2 = 0.5;
      fac3 = 1.0;
      first = false;
   }

   f->SetTime(t + dt);

   x.Add(dt, dxdt);
   x.Add(fac0*dt*dt, d2xdt2);
   dxdt.Add(fac1*dt,  d2xdt2);

   f->ImplicitSolve(fac2*dt*dt,fac3*dt, x, dxdt, d2xdt2);
   x.Add(fac2*dt*dt, d2xdt2);
   dxdt.Add(fac3*dt, d2xdt2);
   t += dt;
}


void GeneralizedAlphaSolver::Init(TimeDependent2Operator &_f)
{
   ODE2Solver::Init(_f);
   k.SetSize(f->Width());
   y.SetSize(f->Width());
   dxdt.SetSize(f->Width());
   dxdt = 0.0;
   first = true;
}

void GeneralizedAlphaSolver::SetRhoInf(double rho_inf)
{
   rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
   rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

   alpha_m = 0.5*(3.0 - rho_inf)/(1.0 + rho_inf);
   alpha_f = 1.0/(1.0 + rho_inf);
   gamma = 0.5 + alpha_m - alpha_f;
}

void GeneralizedAlphaSolver::PrintProperties(std::ostream &out)
{
   out << "Generalized alpha time integrator:" << std::endl;
   out << "alpha_m = " << alpha_m << std::endl;
   out << "alpha_f = " << alpha_f << std::endl;
   out << "gamma   = " << gamma   << std::endl;

   if (gamma == 0.5 + alpha_m - alpha_f)
   {
      out<<"Second order"<<" and ";
   }
   else
   {
      out<<"First order"<<" and ";
   }

   if ((alpha_m >= alpha_f)&&(alpha_f >= 0.5))
   {
      out<<"Stable"<<std::endl;
   }
   else
   {
      out<<"Unstable"<<std::endl;
   }
}

// This routine assumes xdot is initialized.
void GeneralizedAlphaSolver::Step(Vector &x, Vector &dxdt,  double &t,
                                  double &dt)
{
   double dt_fac1 = alpha_f*(1.0 - gamma/alpha_m);
   double dt_fac2 = alpha_f*gamma/alpha_m;
   double dt_fac3 = 1.0/alpha_m;

   // In the first pass xdot is not yet computed. If parameter choices requires
   // xdot midpoint rule is used instead for the first step only.
   if (first && (dt_fac1 != 0.0))
   {
      dt_fac1 = 0.0;
      dt_fac2 = 0.5;
      dt_fac3 = 2.0;
      first = false;
   }

   /*   add(x, dt_fac1*dt, xdot, y);
      f->SetTime(t + dt_fac2*dt);
      f->ImplicitSolve(dt_fac2*dt, y, k);

      add(y, dt_fac2*dt, k, x);
      k.Add(-1.0, xdot);
      xdot.Add(dt_fac3, k);
   */
   t += dt;
}

}
