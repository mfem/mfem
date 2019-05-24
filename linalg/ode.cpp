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
#include "ode.hpp"

namespace mfem
{

void ODESolver::Init(TimeDependentOperator &f)
{
   this->f = &f;
   mem_type = GetMemoryType(f.GetMemoryClass());
}

void ForwardEulerSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   dxdt.SetSize(f->Width(), mem_type);
}

void ForwardEulerSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t);
   f->Mult(x, dxdt);
   x.Add(dt, dxdt);
   t += dt;
}


void RK2Solver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   dxdt.SetSize(n, mem_type);
   x1.SetSize(n, mem_type);
}

void RK2Solver::Step(Vector &x, double &t, double &dt)
{
   //  0 |
   //  a |  a
   // ---+--------
   //    | 1-b  b      b = 1/(2a)

   const double b = 0.5/a;

   f->SetTime(t);
   f->Mult(x, dxdt);
   add(x, (1. - b)*dt, dxdt, x1);
   x.Add(a*dt, dxdt);

   f->SetTime(t + a*dt);
   f->Mult(x, dxdt);
   add(x1, b*dt, dxdt, x);
   t += dt;
}


void RK3SSPSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   y.SetSize(n, mem_type);
   k.SetSize(n, mem_type);
}

void RK3SSPSolver::Step(Vector &x, double &t, double &dt)
{
   // x0 = x, t0 = t, k0 = dt*f(t0, x0)
   f->SetTime(t);
   f->Mult(x, k);

   // x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
   add(x, dt, k, y);
   f->SetTime(t + dt);
   f->Mult(y, k);

   // x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
   y.Add(dt, k);
   add(3./4, x, 1./4, y, y);
   f->SetTime(t + dt/2);
   f->Mult(y, k);

   // x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt
   y.Add(dt, k);
   add(1./3, x, 2./3, y, x);
   t += dt;
}


void RK4Solver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   y.SetSize(n, mem_type);
   k.SetSize(n, mem_type);
   z.SetSize(n, mem_type);
}

void RK4Solver::Step(Vector &x, double &t, double &dt)
{
   //   0  |
   //  1/2 | 1/2
   //  1/2 |  0   1/2
   //   1  |  0    0    1
   // -----+-------------------
   //      | 1/6  1/3  1/3  1/6

   f->SetTime(t);
   f->Mult(x, k); // k1
   add(x, dt/2, k, y);
   add(x, dt/6, k, z);

   f->SetTime(t + dt/2);
   f->Mult(y, k); // k2
   add(x, dt/2, k, y);
   z.Add(dt/3, k);

   f->Mult(y, k); // k3
   add(x, dt, k, y);
   z.Add(dt/3, k);

   f->SetTime(t + dt);
   f->Mult(y, k); // k4
   add(z, dt/6, k, x);
   t += dt;
}

ExplicitRKSolver::ExplicitRKSolver(int _s, const double *_a, const double *_b,
                                   const double *_c)
{
   s = _s;
   a = _a;
   b = _b;
   c = _c;
   k = new Vector[s];
}

void ExplicitRKSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   y.SetSize(n, mem_type);
   for (int i = 0; i < s; i++)
   {
      k[i].SetSize(n, mem_type);
   }
}

void ExplicitRKSolver::Step(Vector &x, double &t, double &dt)
{
   //   0     |
   //  c[0]   | a[0]
   //  c[1]   | a[1] a[2]
   //  ...    |    ...
   //  c[s-2] | ...   a[s(s-1)/2-1]
   // --------+---------------------
   //         | b[0] b[1] ... b[s-1]

   f->SetTime(t);
   f->Mult(x, k[0]);
   for (int l = 0, i = 1; i < s; i++)
   {
      add(x, a[l++]*dt, k[0], y);
      for (int j = 1; j < i; j++)
      {
         y.Add(a[l++]*dt, k[j]);
      }

      f->SetTime(t + c[i-1]*dt);
      f->Mult(y, k[i]);
   }
   for (int i = 0; i < s; i++)
   {
      x.Add(b[i]*dt, k[i]);
   }
   t += dt;
}

ExplicitRKSolver::~ExplicitRKSolver()
{
   delete [] k;
}

const double RK6Solver::a[] =
{
   .6e-1,
   .1923996296296296296296296296296296296296e-1,
   .7669337037037037037037037037037037037037e-1,
   .35975e-1,
   0.,
   .107925,
   1.318683415233148260919747276431735612861,
   0.,
   -5.042058063628562225427761634715637693344,
   4.220674648395413964508014358283902080483,
   -41.87259166432751461803757780644346812905,
   0.,
   159.4325621631374917700365669070346830453,
   -122.1192135650100309202516203389242140663,
   5.531743066200053768252631238332999150076,
   -54.43015693531650433250642051294142461271,
   0.,
   207.0672513650184644273657173866509835987,
   -158.6108137845899991828742424365058599469,
   6.991816585950242321992597280791793907096,
   -.1859723106220323397765171799549294623692e-1,
   -54.66374178728197680241215648050386959351,
   0.,
   207.9528062553893734515824816699834244238,
   -159.2889574744995071508959805871426654216,
   7.018743740796944434698170760964252490817,
   -.1833878590504572306472782005141738268361e-1,
   -.5119484997882099077875432497245168395840e-3
};
const double RK6Solver::b[] =
{
   .3438957868357036009278820124728322386520e-1,
   0.,
   0.,
   .2582624555633503404659558098586120858767,
   .4209371189673537150642551514069801967032,
   4.405396469669310170148836816197095664891,
   -176.4831190242986576151740942499002125029,
   172.3641334014150730294022582711902413315
};
const double RK6Solver::c[] =
{
   .6e-1,
   .9593333333333333333333333333333333333333e-1,
   .1439,
   .4973,
   .9725,
   .9995,
   1.,
};

const double RK8Solver::a[] =
{
   .5e-1,
   -.69931640625e-2,
   .1135556640625,
   .399609375e-1,
   0.,
   .1198828125,
   .3613975628004575124052940721184028345129,
   0.,
   -1.341524066700492771819987788202715834917,
   1.370126503900035259414693716084313000404,
   .490472027972027972027972027972027972028e-1,
   0.,
   0.,
   .2350972042214404739862988335493427143122,
   .180855592981356728810903963653454488485,
   .6169289044289044289044289044289044289044e-1,
   0.,
   0.,
   .1123656831464027662262557035130015442303,
   -.3885046071451366767049048108111244567456e-1,
   .1979188712522045855379188712522045855379e-1,
   -1.767630240222326875735597119572145586714,
   0.,
   0.,
   -62.5,
   -6.061889377376669100821361459659331999758,
   5.650823198222763138561298030600840174201,
   65.62169641937623283799566054863063741227,
   -1.180945066554970799825116282628297957882,
   0.,
   0.,
   -41.50473441114320841606641502701994225874,
   -4.434438319103725011225169229846100211776,
   4.260408188586133024812193710744693240761,
   43.75364022446171584987676829438379303004,
   .787142548991231068744647504422630755086e-2,
   -1.281405999441488405459510291182054246266,
   0.,
   0.,
   -45.04713996013986630220754257136007322267,
   -4.731362069449576477311464265491282810943,
   4.514967016593807841185851584597240996214,
   47.44909557172985134869022392235929015114,
   .1059228297111661135687393955516542875228e-1,
   -.5746842263844616254432318478286296232021e-2,
   -1.724470134262485191756709817484481861731,
   0.,
   0.,
   -60.92349008483054016518434619253765246063,
   -5.95151837622239245520283276706185486829,
   5.556523730698456235979791650843592496839,
   63.98301198033305336837536378635995939281,
   .1464202825041496159275921391759452676003e-1,
   .6460408772358203603621865144977650714892e-1,
   -.7930323169008878984024452548693373291447e-1,
   -3.301622667747079016353994789790983625569,
   0.,
   0.,
   -118.011272359752508566692330395789886851,
   -10.14142238845611248642783916034510897595,
   9.139311332232057923544012273556827000619,
   123.3759428284042683684847180986501894364,
   4.623244378874580474839807625067630924792,
   -3.383277738068201923652550971536811240814,
   4.527592100324618189451265339351129035325,
   -5.828495485811622963193088019162985703755
};
const double RK8Solver::b[] =
{
   .4427989419007951074716746668098518862111e-1,
   0.,
   0.,
   0.,
   0.,
   .3541049391724448744815552028733568354121,
   .2479692154956437828667629415370663023884,
   -15.69420203883808405099207034271191213468,
   25.08406496555856261343930031237186278518,
   -31.73836778626027646833156112007297739997,
   22.93828327398878395231483560344797018313,
   -.2361324633071542145259900641263517600737
};
const double RK8Solver::c[] =
{
   .5e-1,
   .1065625,
   .15984375,
   .39,
   .465,
   .155,
   .943,
   .901802041735856958259707940678372149956,
   .909,
   .94,
   1.,
};


void BackwardEulerSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
}

void BackwardEulerSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt);
   f->ImplicitSolve(dt, x, k); // solve for k: k = f(x + dt*k, t + dt)
   x.Add(dt, k);
   t += dt;
}


void ImplicitMidpointSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
}

void ImplicitMidpointSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt/2);
   f->ImplicitSolve(dt/2, x, k);
   x.Add(dt, k);
   t += dt;
}


SDIRK23Solver::SDIRK23Solver(int gamma_opt)
{
   if (gamma_opt == 0)
   {
      gamma = (3. - sqrt(3.))/6.;   // not A-stable, order 3
   }
   else if (gamma_opt == 2)
   {
      gamma = (2. - sqrt(2.))/2.;   // L-stable, order 2
   }
   else if (gamma_opt == 3)
   {
      gamma = (2. + sqrt(2.))/2.;   // L-stable, order 2
   }
   else
   {
      gamma = (3. + sqrt(3.))/6.;   // A-stable, order 3
   }
}

void SDIRK23Solver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
}

void SDIRK23Solver::Step(Vector &x, double &t, double &dt)
{
   // with a = gamma:
   //   a   |   a
   //  1-a  |  1-2a  a
   // ------+-----------
   //       |  1/2  1/2
   // note: with gamma_opt=3, both solve are outside [t,t+dt] since a>1
   f->SetTime(t + gamma*dt);
   f->ImplicitSolve(gamma*dt, x, k);
   add(x, (1.-2.*gamma)*dt, k, y); // y = x + (1-2*gamma)*dt*k
   x.Add(dt/2, k);

   f->SetTime(t + (1.-gamma)*dt);
   f->ImplicitSolve(gamma*dt, y, k);
   x.Add(dt/2, k);
   t += dt;
}


void SDIRK34Solver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
   z.SetSize(f->Width(), mem_type);
}

void SDIRK34Solver::Step(Vector &x, double &t, double &dt)
{
   //   a   |    a
   //  1/2  |  1/2-a    a
   //  1-a  |   2a    1-4a   a
   // ------+--------------------
   //       |    b    1-2b   b
   // note: two solves are outside [t,t+dt] since c1=a>1, c3=1-a<0
   const double a = 1./sqrt(3.)*cos(M_PI/18.) + 0.5;
   const double b = 1./(6.*(2.*a-1.)*(2.*a-1.));

   f->SetTime(t + a*dt);
   f->ImplicitSolve(a*dt, x, k);
   add(x, (0.5-a)*dt, k, y);
   add(x,  (2.*a)*dt, k, z);
   x.Add(b*dt, k);

   f->SetTime(t + dt/2);
   f->ImplicitSolve(a*dt, y, k);
   z.Add((1.-4.*a)*dt, k);
   x.Add((1.-2.*b)*dt, k);

   f->SetTime(t + (1.-a)*dt);
   f->ImplicitSolve(a*dt, z, k);
   x.Add(b*dt, k);
   t += dt;
}


void SDIRK33Solver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
}

void SDIRK33Solver::Step(Vector &x, double &t, double &dt)
{
   //   a  |   a
   //   c  |  c-a    a
   //   1  |   b   1-a-b  a
   // -----+----------------
   //      |   b   1-a-b  a
   const double a = 0.435866521508458999416019;
   const double b = 1.20849664917601007033648;
   const double c = 0.717933260754229499708010;

   f->SetTime(t + a*dt);
   f->ImplicitSolve(a*dt, x, k);
   add(x, (c-a)*dt, k, y);
   x.Add(b*dt, k);

   f->SetTime(t + c*dt);
   f->ImplicitSolve(a*dt, y, k);
   x.Add((1.-a-b)*dt, k);

   f->SetTime(t + dt);
   f->ImplicitSolve(a*dt, x, k);
   x.Add(a*dt, k);
   t += dt;
}


void GeneralizedAlphaSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
   xdot.SetSize(f->Width(), mem_type);
   xdot = 0.0;
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
void GeneralizedAlphaSolver::Step(Vector &x, double &t, double &dt)
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

   add(x, dt_fac1*dt, xdot, y);
   f->SetTime(t + dt_fac2*dt);
   f->ImplicitSolve(dt_fac2*dt, y, k);

   add(y, dt_fac2*dt, k, x);
   k.Add(-1.0, xdot);
   xdot.Add(dt_fac3, k);

   t += dt;
}


void
SIASolver::Init(Operator &P, TimeDependentOperator & F)
{
   P_ = &P; F_ = &F;

   dp_.SetSize(F_->Height());
   dq_.SetSize(P_->Height());
}

void
SIA1Solver::Step(Vector &q, Vector &p, double &t, double &dt)
{
   F_->SetTime(t);
   F_->Mult(q,dp_);
   p.Add(dt,dp_);

   P_->Mult(p,dq_);
   q.Add(dt,dq_);

   t += dt;
}

void
SIA2Solver::Step(Vector &q, Vector &p, double &t, double &dt)
{
   P_->Mult(p,dq_);
   q.Add(0.5*dt,dq_);

   F_->SetTime(t+0.5*dt);
   F_->Mult(q,dp_);
   p.Add(dt,dp_);

   P_->Mult(p,dq_);
   q.Add(0.5*dt,dq_);

   t += dt;
}

SIAVSolver::SIAVSolver(int order)
   : order_(order)
{
   a_.SetSize(order);
   b_.SetSize(order);

   switch (order_)
   {
      case 1:
         a_[0] = 1.0;
         b_[0] = 1.0;
         break;
      case 2:
         a_[0] = 0.5;
         a_[1] = 0.5;
         b_[0] = 0.0;
         b_[1] = 1.0;
         break;
      case 3:
         a_[0] =  2.0/3.0;
         a_[1] = -2.0/3.0;
         a_[2] =  1.0;
         b_[0] =  7.0/24.0;
         b_[1] =  0.75;
         b_[2] = -1.0/24.0;
         break;
      case 4:
         a_[0] = (2.0+pow(2.0,1.0/3.0)+pow(2.0,-1.0/3.0))/6.0;
         a_[1] = (1.0-pow(2.0,1.0/3.0)-pow(2.0,-1.0/3.0))/6.0;
         a_[2] = a_[1];
         a_[3] = a_[0];
         b_[0] = 0.0;
         b_[1] = 1.0/(2.0-pow(2.0,1.0/3.0));
         b_[2] = 1.0/(1.0-pow(2.0,2.0/3.0));
         b_[3] = b_[1];
         break;
      default:
         MFEM_ASSERT(false, "Unsupported order in SIAVSolver");
   };
}

void
SIAVSolver::Step(Vector &q, Vector &p, double &t, double &dt)
{
   for (int i=0; i<order_; i++)
   {
      if ( b_[i] != 0.0 )
      {
         F_->SetTime(t);
         if ( F_->isExplicit() )
         {
            F_->Mult(q, dp_);
         }
         else
         {
            F_->ImplicitSolve(b_[i] * dt, q, dp_);
         }
         p.Add(b_[i] * dt, dp_);
      }

      P_->Mult(p, dq_);
      q.Add(a_[i] * dt, dq_);

      t += a_[i] * dt;
   }
}

}
