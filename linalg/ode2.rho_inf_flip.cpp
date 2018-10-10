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

   f->ImplicitSolve(fac3*dt*dt,fac4*dt, x, dxdt, d2xdt2);
   std::cout<<dt<<" "<<fac3*dt*dt<<" "<<fac4*dt<<"  -->  ";
   std::cout<<x[0]<<" "<<dxdt[0]<<" "<<d2xdt2[0]<<std::endl;
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

void GeneralizedAlpha2Solver::SetRhoInf(double rho_inf)
{
   rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
   rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;
   std::cout<<rho_inf<<std::endl;
   alpha_m = 0.5*(3.0 - rho_inf)/(1.0 + rho_inf);
   alpha_f = 1.0/(1.0 + rho_inf);

   beta    = 0.25*pow(1.0 + alpha_m - alpha_f,2);
   gamma   = 0.5 + alpha_m - alpha_f;

   alpha_m = (2*rho_inf-1.0)/(1.0 + rho_inf);
   alpha_f = rho_inf/(1.0 + rho_inf);

   beta    = 0.25*pow(1.0 - alpha_m + alpha_f,2);
   gamma   = 0.5 -alpha_m + alpha_f;

   alpha_m = 1.0 - alpha_m;
   alpha_f = 1.0 - alpha_f;

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
   // d2xdt2 then Midpoint (rho_inf = 1) is used instead for the first step only.
   if (first && !(fac0*fac2 == 0.0))
   {
      fac0 = 0.0;
      fac1 = 0.5;
      fac2 = 0.0;
      fac3 = 0.25;
      fac4 = 0.5;
      fac5 = 0.5;

      fac0 = 0.0;
      fac1 = 1.0;
      fac2 = 0.0;
      fac3 = 0.5;
      fac4 = 1.0;
      fac5 = 1.0;

      first = false;
   }
   std::cout<<fac0<<" "<<fac1<<" "<<fac2<<" "<<fac3<<" "<<fac4<<" "<<fac5<<std::endl;


   f->SetTime(t + fac1*dt);

   dxdt.Add(fac0*dt, d2xdt2);
   x.Add(fac1*dt, dxdt);
   dxdt.Add((fac2-fac0)*dt, d2xdt2);

   f->ImplicitSolve(fac3*dt*dt,fac4*dt, x, dxdt, k);
   std::cout<<dt<<" "<<fac3*dt*dt<<" "<<fac4*dt<<"  -->  ";
   std::cout<<x[0]<<" "<<dxdt[0]<<" "<<k[0]<<std::endl;
   x.Add(fac3*dt*dt, k);
   dxdt.Add(fac4*dt, k);

   k -= d2xdt2;
   d2xdt2.Add(fac5,k);

   t += dt;

}

}
