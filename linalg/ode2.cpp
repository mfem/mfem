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
#include <iomanip>

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
   else if ((gamma >= 0.5) && (beta >= 0.5*gamma))
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
   double fac2 = 1.0 - gamma;
   double fac3 = beta;
   double fac4 = gamma;

   // In the first pass d2xdt2 is not yet computed. If parameter choices requires
   // d2xdt2  backward euler is used instead for the first step only.
   if (first && !(fac0*fac2 == 0.0))
   {
      fac0 = 0.0;
      fac2 = 0.0;
      fac3 = 0.5;
      fac4 = 1.0;
      first = false;
   }

   f->SetTime(t + dt);

   x.Add(dt, dxdt);
   x.Add(fac0*dt*dt, d2xdt2);
   dxdt.Add(fac2*dt,  d2xdt2);

   f->SetTime(t + dt);
   f->ImplicitSolve(fac3*dt*dt,fac4*dt, x, dxdt, d2xdt2);

   x.Add(fac3*dt*dt, d2xdt2);
   dxdt.Add(fac4*dt, d2xdt2);
   t += dt;
}

void GeneralizedAlpha2Solver::Init(TimeDependent2Operator &_f)
{
   ODE2Solver::Init(_f);
   k.SetSize(f->Width());
   d2xdt2.SetSize(f->Width());
   d2xdt2 = 0.0;
   first = true;
}

void GeneralizedAlpha2Solver::PrintProperties(std::ostream &out)
{
   out << "Generalized alpha time integrator:" << std::endl;
   out << "alpha_m = " << alpha_m << std::endl;
   out << "alpha_f = " << alpha_f << std::endl;
   out << "beta    = " << beta    << std::endl;
   out << "gamma   = " << gamma   << std::endl;

   if (gamma == 0.5 + alpha_m - alpha_f)
   {
      out<<"Second order"<<" and ";
   }
   else
   {
      out<<"First order"<<" and ";
   }

   if ((alpha_m >= alpha_f)&&
       (alpha_f >= 0.5) &&
       (beta >= 0.25 + 0.5*(alpha_m - alpha_f)))
   {
      out<<"Stable"<<std::endl;
   }
   else
   {
      out<<"Unstable"<<std::endl;
   }
}

void GeneralizedAlpha2Solver::Step(Vector &x, Vector &dxdt,  double &t,
                                   double &dt)
{
   double fac0 = (0.5 - (beta/alpha_m));
   double fac1 = alpha_f;
   double fac2 = alpha_f*(1.0 - (gamma/alpha_m));
   double fac3 = beta*alpha_f/alpha_m;
   double fac4 = gamma*alpha_f/alpha_m;
   double fac5 = 1.0/alpha_m;

   // In the first pass d2xdt2 is not yet computed. If parameter choices requires
   // d2xdt2 then backward Eeuler is used instead for the first step only.
   if (first)// && !(fac0*fac2 == 0.0))
   {
      fac0 = 0.0;
      fac1 = 1.0;
      fac2 = 0.0;
      fac3 = 0.5;
      fac4 = 1.0;
      fac5 = 1.0;
      first = false;
   }

   dxdt.Add(fac0*dt, d2xdt2);
   x.Add(fac1*dt, dxdt);
   dxdt.Add((fac2-fac0)*dt, d2xdt2);
   d2xdt2 *= 1.0 - fac5;

   f->SetTime(t + dt);
   f->ImplicitSolve(fac3*dt*dt,fac4*dt, x, dxdt, k);

   x     .Add(fac3*dt*dt, k);
   dxdt  .Add(fac4*dt,    k);
   d2xdt2.Add(fac5,       k);
   t += dt;

}

}
