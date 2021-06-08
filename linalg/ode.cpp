// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "operator.hpp"
#include "ode.hpp"

namespace mfem
{

void ODESolver::Init(TimeDependentOperator &f_)
{
   this->f = &f_;
   mem_type = GetMemoryType(f_.GetMemoryClass());
}

void ForwardEulerSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   dxdt.SetSize(f->Width(), mem_type);
}

void ForwardEulerSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t);
   f->Mult(x, dxdt);
   x.Add(dt, dxdt);
   t += dt;
}


void RK2Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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


void RK3SSPSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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


void RK4Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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

ExplicitRKSolver::ExplicitRKSolver(int s_, const double *a_, const double *b_,
                                   const double *c_)
{
   s = s_;
   a = a_;
   b = b_;
   c = c_;
   k = new Vector[s];
}

void ExplicitRKSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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


AdamsBashforthSolver::AdamsBashforthSolver(int s_, const double *a_)
{
   smax = std::min(s_,5);
   a = a_;
   k = new Vector[5];

   if (smax <= 2)
   {
      RKsolver = new RK2Solver();
   }
   else if (smax == 3)
   {
      RKsolver = new RK3SSPSolver();
   }
   else
   {
      RKsolver = new RK4Solver();
   }
}

void AdamsBashforthSolver::GetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i >= 0) && ( i < s ),
                " AdamsBashforthSolver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);

   state = k[idx[i]];
}

const Vector &AdamsBashforthSolver::GetStateVector(int i)
{
   MFEM_ASSERT( (i >= 0) && ( i < s ),
                " AdamsBashforthSolver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);

   return k[idx[i]];
}


void AdamsBashforthSolver::SetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i >= 0) && ( i < smax ),
                " AdamsBashforthSolver::SetStateVector \n" <<
                " - Tried to set non-existent state "<<i);
   k[idx[i]] = state;
   s = std::max(i,s);
}

void AdamsBashforthSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   RKsolver->Init(f_);
   idx.SetSize(smax);
   for (int i = 0; i < smax; i++)
   {
      idx[i] = (smax-i)%smax;
      k[i].SetSize(f->Width());
   }
   s = 0;
}

void AdamsBashforthSolver::Step(Vector &x, double &t, double &dt)
{
   s++;
   s = std::min(s, smax);
   if (s == smax)
   {
      f->SetTime(t);
      f->Mult(x, k[idx[0]]);
      for (int i = 0; i < s; i++)
      {
         x.Add(a[i]*dt, k[idx[i]]);
      }
   }
   else
   {
      f->Mult(x,k[idx[0]]);
      RKsolver->Step(x,t,dt);
   }
   t += dt;

   // Shift the index
   for (int i = 0; i < smax; i++) { idx[i] = ++idx[i]%smax; }
}

const double AB1Solver::a[] =
{1.0};
const double AB2Solver::a[] =
{1.5,-0.5};
const double AB3Solver::a[] =
{23.0/12.0,-4.0/3.0, 5.0/12.0};
const double AB4Solver::a[] =
{55.0/24.0,-59.0/24.0, 37.0/24.0,-9.0/24.0};
const double AB5Solver::a[] =
{1901.0/720.0,-2774.0/720.0, 2616.0/720.0,-1274.0/720.0, 251.0/720.0};

AdamsMoultonSolver::AdamsMoultonSolver(int s_, const double *a_)
{
   s = 0;
   smax = std::min(s_+1,5);
   a = a_;
   k = new Vector[5];

   if (smax <= 3)
   {
      RKsolver = new SDIRK23Solver();
   }
   else
   {
      RKsolver = new SDIRK34Solver();
   }
}

const Vector &AdamsMoultonSolver::GetStateVector(int i)
{
   MFEM_ASSERT( (i >= 0) && ( i < s ),
                " AdamsMoultonSolver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);
   return k[idx[i+1]];
}

void AdamsMoultonSolver::GetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i >= 0) && ( i < s ),
                " AdamsMoultonSolver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);
   state = k[idx[i+1]];
}

void AdamsMoultonSolver::SetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i >= 0) && ( i < smax ),
                " AdamsMoultonSolver::SetStateVector \n" <<
                " - Tried to set non-existent state "<<i);
   k[idx[i+1]] = state;
   s = std::max(i,s);
}

void AdamsMoultonSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   RKsolver->Init(f_);
   int n = f->Width();
   idx.SetSize(smax);
   for (int i = 0; i < smax; i++)
   {
      idx[i] = (smax-i)%smax;
      k[i].SetSize(n);
   }
   s = 0;
}

void AdamsMoultonSolver::Step(Vector &x, double &t, double &dt)
{
   if ((s == 0)&&(smax>1))
   {
      f->Mult(x,k[idx[1]]);
   }
   s++;
   s = std::min(s, smax);

   if (s >= smax-1)
   {
      f->SetTime(t);
      for (int i = 1; i < smax; i++)
      {
         x.Add(a[i]*dt, k[idx[i]]);
      }
      f->ImplicitSolve(a[0]*dt, x, k[idx[0]]);
      x.Add(a[0]*dt, k[idx[0]]);
   }
   else
   {
      RKsolver->Step(x,t,dt);
      f->Mult(x,k[idx[0]]);
   }
   t += dt;

   // Shift the index
   for (int i = 0; i < smax; i++) { idx[i] = ++idx[i]%smax; }
}

const double AM0Solver::a[] =
{1.0};
const double AM1Solver::a[] =
{0.5, 0.5};
const double AM2Solver::a[] =
{5.0/12.0, 2.0/3.0, -1.0/12.0};
const double AM3Solver::a[] =
{3.0/8.0, 19.0/24.0,-5.0/24.0, 1.0/24.0};
const double AM4Solver::a[] =
{251.0/720.0,646.0/720.0,-264.0/720.0, 106.0/720.0, -19.0/720.0};


void BackwardEulerSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   k.SetSize(f->Width(), mem_type);
}

void BackwardEulerSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt);
   f->ImplicitSolve(dt, x, k); // solve for k: k = f(x + dt*k, t + dt)
   x.Add(dt, k);
   t += dt;
}


void ImplicitMidpointSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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

void SDIRK23Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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


void SDIRK34Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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


void SDIRK33Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
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
   x.Add((1.0-a-b)*dt, k);

   f->SetTime(t + dt);
   f->ImplicitSolve(a*dt, x, k);
   x.Add(a*dt, k);
   t += dt;
}

void TrapezoidalRuleSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
}

void TrapezoidalRuleSolver::Step(Vector &x, double &t, double &dt)
{
   //   0   |   0    0
   //   1   |  1/2  1/2
   // ------+-----------
   //       |  1/2  1/2
   f->SetTime(t);
   f->Mult(x,k);
   add(x, dt/2.0, k, y);
   x.Add(dt/2.0, k);

   f->SetTime(t + dt);
   f->ImplicitSolve(dt/2.0, y, k);
   x.Add(dt/2.0, k);
   t += dt;
}

void ESDIRK32Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
   z.SetSize(f->Width(), mem_type);
}

void ESDIRK32Solver::Step(Vector &x, double &t, double &dt)
{
   //   0   |    0      0    0
   //   2a  |    a      a    0
   //   1   |  1-b-a    b    a
   // ------+--------------------
   //       |  1-b-a    b    a
   const double a = (2.0 - sqrt(2.0)) / 2.0;
   const double b = (1.0 - 2.0*a) / (4.0*a);

   f->SetTime(t);
   f->Mult(x,k);
   add(x, a*dt, k, y);
   add(x, (1.0-b-a)*dt, k, z);
   x.Add((1.0-b-a)*dt, k);

   f->SetTime(t + (2.0*a)*dt);
   f->ImplicitSolve(a*dt, y, k);
   z.Add(b*dt, k);
   x.Add(b*dt, k);

   f->SetTime(t + dt);
   f->ImplicitSolve(a*dt, z, k);
   x.Add(a*dt, k);
   t += dt;
}

void ESDIRK33Solver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
   z.SetSize(f->Width(), mem_type);
}

void ESDIRK33Solver::Step(Vector &x, double &t, double &dt)
{
   //   0   |      0          0        0
   //   2a  |      a          a        0
   //   1   |    1-b-a        b        a
   // ------+----------------------------
   //       |  1-b_2-b_3     b_2      b_3
   const double a   = (3.0 + sqrt(3.0)) / 6.0;
   const double b   = (1.0 - 2.0*a) / (4.0*a);
   const double b_2 = 1.0 / ( 12.0*a*(1.0 - 2.0*a) );
   const double b_3 = (1.0 - 3.0*a) / ( 3.0*(1.0 - 2.0*a) );

   f->SetTime(t);
   f->Mult(x,k);
   add(x, a*dt, k, y);
   add(x, (1.0-b-a)*dt, k, z);
   x.Add((1.0-b_2-b_3)*dt, k);

   f->SetTime(t + (2.0*a)*dt);
   f->ImplicitSolve(a*dt, y, k);
   z.Add(b*dt, k);
   x.Add(b_2*dt, k);

   f->SetTime(t + dt);
   f->ImplicitSolve(a*dt, z, k);
   x.Add(b_3*dt, k);
   t += dt;
}

void GeneralizedAlphaSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   k.SetSize(f->Width(), mem_type);
   y.SetSize(f->Width(), mem_type);
   xdot.SetSize(f->Width(), mem_type);
   xdot = 0.0;
   nstate = 0;
}

const Vector &GeneralizedAlphaSolver::GetStateVector(int i)
{
   MFEM_ASSERT( (i == 0) && (nstate == 1),
                "GeneralizedAlphaSolver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);
   return xdot;
}

void GeneralizedAlphaSolver::GetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i == 0) && (nstate == 1),
                "GeneralizedAlphaSolver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);
   state = xdot;
}

void GeneralizedAlphaSolver::SetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i == 0),
                "GeneralizedAlphaSolver::SetStateVector \n" <<
                " - Tried to set non-existent state "<<i);
   xdot = state;
   nstate = 1;
}

void GeneralizedAlphaSolver::SetRhoInf(double rho_inf)
{
   rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
   rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

   // According to Jansen
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
   if (nstate == 0)
   {
      f->Mult(x,xdot);
      nstate = 1;
   }

   // Set y = x + alpha_f*(1.0 - (gamma/alpha_m))*dt*xdot
   add(x, alpha_f*(1.0 - (gamma/alpha_m))*dt, xdot, y);

   // Solve k = f(y + dt_eff*k)
   double dt_eff = (gamma*alpha_f/alpha_m)*dt;
   f->SetTime(t + alpha_f*dt);
   f->ImplicitSolve(dt_eff, y, k);

   // Update x and xdot
   x.Add((1.0 - (gamma/alpha_m))*dt, xdot);
   x.Add(       (gamma/alpha_m) *dt, k);

   xdot *= (1.0-(1.0/alpha_m));
   xdot.Add((1.0/alpha_m),k);

   t += dt;
}


void IMEX_BE_FE::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k_imp.SetSize(f->Width());
   y.SetSize(f->Width());
   k_exp.SetSize(f->Width());
}

void IMEX_BE_FE::Step(Vector &x, double &t, double &dt)
{
   f->ExplicitMult(x, k_exp);
   add(x, dt, k_exp, y);

   f->SetTime(t + dt);
   f->ImplicitSolve(dt, y, k_imp);

   x.Add(dt, k_exp);
   x.Add(dt, k_imp);
   t += dt;
}


void IMEXRK2::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   f = ODESolver::f;
   k_imp.SetSize(f->Width());
   k_exp.SetSize(f->Width());
   y.SetSize(f->Width());
   z.SetSize(f->Width());
}

void IMEXRK2::Step(Vector &x, double &t, double &dt)
{
   double gamma = 1 - sqrt(2)/2;
   double delta = -2*sqrt(2)/3;

   // The method is given by
   // k1_exp = f(u)
   // k1_imp = g(u + gamma*dt*k1_exp + gamma*dt*k1_imp)
   // k2_exp = f(u + gamma*dt*k1_exp + gamma*dt*k1_imp)
   // k2_imp = g(u + delta*dt*k1_exp + (1-gamma)*dt*k1_imp
   //              + (1-delta)*dt*k2_exp + gamma*dt*k2_imp)
   // k3_exp = f(u + delta*dt*k1_exp + (1-gamma)*dt*k1_imp
   //              + (1-delta)*dt*k2_exp + gamma*dt*k2_imp)
   // u_new = u + dt*((1-gamma)*k1_imp + (1-gamma)*k2_exp
   //                 + gamma*k2_imp + gamma*k3_exp)

   // Take first explicit step
   // k1_exp = f(u)
   f->ExplicitMult(x, k_exp);
   // b corresponding to this stage is zero, so don't add to solution

   // Solve first implicit step
   // y = u + gamma*dt*k1_exp
   add(x, gamma*dt, k_exp, y);
   // Solve x1_imp = g(u + gamma*dt*k1_exp + gamma*dt*k1_imp)
   f->SetTime(t + gamma*dt);
   f->ImplicitSolve(gamma*dt, y, k_imp);
   // x = u + (1-gamma)*dt*k1_imp
   x.Add((1-gamma)*dt, k_imp);

   // Begin setting up rhs for second solve
   // z = u + (1-gamma)*dt*k_imp + delta*dt*k_exp
   add(x, delta*dt, k_exp, z);

   // Take second explicit step
   // y = x + gamma*dt*k1_exp + gamma*dt*k1_imp
   y.Add(gamma*dt, k_imp);
   // k2_exp = f(x + gamma*dt*k1_exp + gamma*dt*k1_imp)
   f->ExplicitMult(y, k_exp);
   // x = u + (1-gamma)*dt*k1_imp + (1-gamma)*dt*k2_exp
   x.Add((1-gamma)*dt, k_exp);

   // Finish formoing rhs
   // z = x + (1-gamma)*dt*k1_imp + delta*dt*k1_exp + (1-delta)*dt*k2_exp
   z.Add((1-delta)*dt, k_exp);

   // Solve second implicit step for k2_imp
   f->SetTime(t + dt);
   f->ImplicitSolve(gamma*dt, z, k_imp);
   // x = u + (1-gamma)*dt*k1_imp + (1-gamma)*dt*k2_exp + gamma*dt*k2_imp
   x.Add(gamma*dt, k_imp);

   // Take final explicit step for k3_exp
   z.Add(gamma*dt, k_imp);
   f->ExplicitMult(z, k_exp);

   // x = u + (1-gamma)*dt*k1_imp + (1-gamma)*dt*k2_exp + gamma*dt*k2_imp
   //       + gamma*dt*k3_exp
   x.Add(gamma*dt, k_exp);

   t += dt;
}

EmbeddedRKSolver::EmbeddedRKSolver(int _s, const double *_a,
                                   const double *_b, const double *_b_embedded,
                                   const double *_c)
{
   s = _s;
   a = _a;
   b = _b;
   b_em = _b_embedded;
   c = _c;
   k = new Vector[s];
}

void EmbeddedRKSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   y.SetSize(n, mem_type);
   for (int i = 0; i < s; i++)
   {
      k[i].SetSize(n, mem_type);
   }
}

void EmbeddedRKSolver::Step(Vector &x, double &t, double &dt)
{
   //   0     |
   //  c[0]   | a[0]
   //  c[1]   | a[1]  a[2]
   //  ...    |    ...
   //  c[s-2] | ...    a[s(s-1)/2-1]
   // --------+---------------------
   //         | b[0]  b[1]  ... b[s-1]
   //         | b*[0] b*[1] ... b*[s-1]

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

void EmbeddedRKSolver::Step(Vector &x, Vector &e, double &t, double &dt)
{
   //   0     |
   //  c[0]   | a[0]
   //  c[1]   | a[1]  a[2]
   //  ...    |    ...
   //  c[s-2] | ...    a[s(s-1)/2-1]
   // --------+---------------------
   //         | b[0]  b[1]  ... b[s-1]
   //         | b*[0] b*[1] ... b*[s-1]

   if (e.Size() != x.Size()) { e.SetSize(x.Size()); }
   e = 0.0;

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
      e.Add((b_em[i]-b[i])*dt, k[i]);
   }
   t += dt;
}

EmbeddedRKSolver::~EmbeddedRKSolver()
{
   delete [] k;
}

const double HeunEulerSolver::a[] = {1.0};
const double HeunEulerSolver::b[] = {0.5, 0.5};
const double HeunEulerSolver::b_star[] = {1.0, 0.0};
const double HeunEulerSolver::c[] = {1.0};

const double FehlbergRK12Solver::a[] = {0.5, 0.00390625, 0.99609375};
const double FehlbergRK12Solver::b[] = {0.00390625, 0.99609375, 0.0};
const double FehlbergRK12Solver::b_star[] =
{
   0.001953125, 0.99609375, 0.001953125
};
const double FehlbergRK12Solver::c[] = {0.5, 1.0};

const double BogackiShampineSolver::a[] =
{
   0.5,
   0.0,
   0.75,
   0.2222222222222222222222222222222222222222,
   0.3333333333333333333333333333333333333333,
   0.4444444444444444444444444444444444444444
};
const double BogackiShampineSolver::b[] =
{
   0.2222222222222222222222222222222222222222,
   0.3333333333333333333333333333333333333333,
   0.4444444444444444444444444444444444444444,
   0.0
};
const double BogackiShampineSolver::b_star[] =
{
   0.2916666666666666666666666666666666666667,
   0.25,
   0.3333333333333333333333333333333333333333,
   0.125
};
const double BogackiShampineSolver::c[] = {0.5, 0.75, 1.0};

const double FehlbergRK45Solver::a[] =
{
   0.25,
   0.09375,
   0.28125,
   0.8793809740555302685480200273099681383705,
   -3.277196176604460628129267182521620391443,
   3.320892125625853436504324078288575329995,
   2.032407407407407407407407407407407407407,
   -8.0,
   7.173489278752436647173489278752436647173,
   -0.2058966861598440545808966861598440545809,
   -0.2962962962962962962962962962962962962963,
   2.0,
   -1.381676413255360623781676413255360623782,
   0.4529727095516569200779727095516569200780,
   -0.275
};
const double FehlbergRK45Solver::b[] =
{
   0.1185185185185185185185185185185185185185,
   0.0,
   0.5189863547758284600389863547758284600390,
   0.5061314903420166578061314903420166578061,
   -0.18,
   0.03636363636363636363636363636363636363636
};
const double FehlbergRK45Solver::b_star[] =
{
   0.1157407407407407407407407407407407407407,
   0.0,
   0.5489278752436647173489278752436647173489,
   0.5353313840155945419103313840155945419103,
   -0.2,
   0.0
};
const double FehlbergRK45Solver::c[] =
{
   0.25,
   0.375,
   0.9230769230769230769230769230769230769231,
   1.0,
   0.5
};

const double CashKarpSolver::a[] =
{
   0.2,
   0.075,
   0.225,
   0.3,
   -0.9,
   1.2,
   0.2037037037037037037037037037037037037037,
   2.5,
   -2.592592592592592592592592592592592592593,
   1.296296296296296296296296296296296296296,
   0.02949580439814814814814814814814814814815,
   0.341796875,
   0.04159432870370370370370370370370370370370,
   0.4003454137731481481481481481481481481481,
   0.061767578125
};
const double CashKarpSolver::b[] =
{
   0.09788359788359788359788359788359788359788,
   0.0,
   0.4025764895330112721417069243156199677939,
   0.2104377104377104377104377104377104377104,
   0.0,
   0.2891022021456804065499717673630717108978
};
const double CashKarpSolver::b_star[] =
{
   0.1021773726851851851851851851851851851852,
   0.0,
   0.3839079034391534391534391534391534391534,
   0.2445927372685185185185185185185185185185,
   0.01932198660714285714285714285714285714286,
   0.25
};
const double CashKarpSolver::c[] =
{
   0.2,
   0.3,
   0.6,
   1.0,
   0.875
};

const double DormandPrinceSolver::a[] =
{
   0.2,
   0.075,
   0.225,
   0.9777777777777777777777777777777777777778,
   -3.733333333333333333333333333333333333333,
   3.555555555555555555555555555555555555556,
   2.952598689224203627495808565767413504039,
   -11.59579332418838591678097850937357110197,
   9.822892851699436061575979271452522481329,
   -0.2908093278463648834019204389574759945130,
   2.846275252525252525252525252525252525253,
   -10.75757575757575757575757575757575757576,
   8.906422717743472460453592529064227177435,
   0.2784090909090909090909090909090909090909,
   -0.2735313036020583190394511149228130360206,
   0.09114583333333333333333333333333333333333,
   0.0,
   0.4492362982929020664869721473495058400719,
   0.6510416666666666666666666666666666666667,
   -0.3223761792452830188679245283018867924528,
   0.1309523809523809523809523809523809523810
};
const double DormandPrinceSolver::b[] =
{
   0.09114583333333333333333333333333333333333,
   0.0,
   0.4492362982929020664869721473495058400719,
   0.6510416666666666666666666666666666666667,
   -0.3223761792452830188679245283018867924528,
   0.1309523809523809523809523809523809523810,
   0.0
};
const double DormandPrinceSolver::b_star[] =
{
   0.08991319444444444444444444444444444444444,
   0.0,
   0.4534890685834082060497154836777478286912,
   0.6140625,
   -0.2715123820754716981132075471698113207547,
   0.08904761904761904761904761904761904761905,
   0.025
};
const double DormandPrinceSolver::c[] =
{
   0.2,
   0.3,
   0.8,
   0.8888888888888888888888888888888888888889,
   1.0,
   1.0
};

EmbeddedSDIRKSolver::EmbeddedSDIRKSolver(int _s, const double *_a,
                                         const double *_b,
                                         const double *_b_embedded,
                                         const double *_c)
{
   s = _s;
   a = _a;
   b = _b;
   b_em = _b_embedded;
   c = _c;
   k = new Vector[s];
}

void EmbeddedSDIRKSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   int n = f->Width();
   y.SetSize(n, mem_type);
   for (int i = 0; i < s; i++)
   {
      k[i].SetSize(n, mem_type);
   }
}

void EmbeddedSDIRKSolver::Step(Vector &x, double &t, double &dt)
{
   //  c[0]   | a[0]
   //  c[1]   | a[1]  a[2]
   //  ...    |    ...
   //  c[s-2] | ...    a[s(s-1)/2-1]
   // --------+---------------------
   //         | b[0]  b[1]  ... b[s-1]
   //         | b*[0] b*[1] ... b*[s-1]

   f->SetTime(t + dt * c[0]);
   f->ImplicitSolve(dt * a[0], x, k[0]);
   for (int l = 1, i = 1; i < s; i++)
   {
      add(x, dt * a[l++], k[0], y);
      for (int j = 1; j < i; j++)
      {
         y.Add(dt * a[l++], k[j]);
      }

      f->SetTime(t + dt * c[i]);
      f->ImplicitSolve(dt * a[l++], y, k[i]);
   }
   for (int i = 0; i < s; i++)
   {
      x.Add(dt * b[i], k[i]);
   }
   t += dt;
}

void EmbeddedSDIRKSolver::Step(Vector &x, Vector &e, double &t, double &dt)
{
   //  c[0]   | a[0]
   //  c[1]   | a[1]  a[2]
   //  ...    |    ...
   //  c[s-2] | ...    a[s(s-1)/2-1]
   // --------+---------------------
   //         | b[0]  b[1]  ... b[s-1]
   //         | b*[0] b*[1] ... b*[s-1]

   if (e.Size() != x.Size()) { e.SetSize(x.Size()); }
   e = 0.0;

   f->SetTime(t + dt * c[0]);
   f->ImplicitSolve(dt * a[0], x, k[0]);
   for (int l = 1, i = 1; i < s; i++)
   {
      add(x, dt * a[l++], k[0], y);
      for (int j = 1; j < i; j++)
      {
         y.Add(dt * a[l++], k[j]);
      }

      f->SetTime(t + dt * c[i]);
      f->ImplicitSolve(dt * a[l++], y, k[i]);
   }
   for (int i = 0; i < s; i++)
   {
      x.Add(dt * b[i], k[i]);
      e.Add(dt * (b_em[i] - b[i]), k[i]);
   }
   t += dt;
}

EmbeddedSDIRKSolver::~EmbeddedSDIRKSolver()
{
   delete [] k;
}

const double SDIRK212Solver::a[]      = {1.0, -1.0, 1.0};
const double SDIRK212Solver::b[]      = {0.5, 0.5};
const double SDIRK212Solver::b_star[] = {1.0, 0.0};
const double SDIRK212Solver::c[]      = {1.0, 0.0};

const double SDIRK534Solver::a[] =
{
   0.25,
   0.5,
   0.25,
   0.34,
   -0.04,
   0.25,
   0.2727941176470588235294117647058823529412,
   -0.05036764705882352941176470588235294117647,
   0.02757352941176470588235294117647058823529,
   0.25,
   1.041666666666666666666666666666666666667,
   -1.020833333333333333333333333333333333333,
   7.8125,
   -7.083333333333333333333333333333333333333,
   0.25
};
const double SDIRK534Solver::b[] =
{
   1.041666666666666666666666666666666666667,
   -1.020833333333333333333333333333333333333,
   7.8125,
   -7.083333333333333333333333333333333333333,
   0.25
};
const double SDIRK534Solver::b_star[] =
{
   1.229166666666666666666666666666666666667,
   -0.1770833333333333333333333333333333333333,
   7.03125,
   -7.083333333333333333333333333333333333333,
   0.0
};
const double SDIRK534Solver::c[] = {0.25, 0.75, 0.55, 0.5, 1.0};

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

void ODEController::SetTolerance(double tol)
{
   this->tol = tol;

   if (this->acc) { this->acc->SetTolerance(tol); }
   if (this->rej) { this->rej->SetTolerance(tol); }
}

void ODEController::Step(Vector &x, double &t, double delta_t)
{
   bool accept = false;
   dt = delta_t;
   while (!accept && nrejs < max_nrejs)
   {
      next_x = x;
      double next_t = t;

      sol->Step(next_x, error, next_t, dt);

      double r = msr->Eval(next_x, error);
      double a = -1.0;

      if (epus) { r /= dt; }

      if (r <= rho * tol)
      {
         accept = true;
         nrejs  = 0;

         curr_r = r;
         x = next_x;
         t = next_t;
         a = (*acc)(r);
         rej->Reset();
      }
      else
      {
         nrejs++;
         nrejs_tot++;
         a = (*rej)(r);
         acc->Reset();
      }
      dt = std::max(min_dt, (*lim)(a) * dt); // Prevent bottomless descent
   }
}

void ODEController::Run(Vector &x, double &t, double tf)
{
   while (t < tf)
   {
      this->Step(x, t, std::min(dt, std::abs(tf - t)));

      nsteps++;

      if (out)
      {
         *out << t << '\t'
              << nsteps + nrejs_tot << '\t'
              << curr_r << '\t'
              << dt;
         if (log_sol_)
         {
            for (int i=0; i<x.Size(); i++)
            {
               *out << '\t' << x(i);
            }
         }
         *out << std::endl;
      }
      if (ofreq > 0 && nsteps % ofreq == 0)
      {
         // Return control to calling routine for visulization etc.
         break;
      }
   }
}

double MaxAbsRelDiffMeasure::Eval(Vector &x, Vector &e)
{
   MFEM_ASSERT(x.Size() == e.Size(), "Incompatible vector sizes: "
               << x.Size() << " and " << e.Size());
   if (etaVec)
   {
      MFEM_ASSERT(x.Size() == etaVec->Size(), "Incorrect scaling vector size: "
                  << etaVec->Size() << " should match " << x.Size());
   }

   double max = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      double eta = (etaVec) ? (*etaVec)(i) : etaConst;

      max = std::max(std::abs((e(i)) / (eta + std::abs(x(i)))), max);
   }

   return max;
}

double L2AbsRelDiffMeasure::Eval(Vector &x, Vector &e)
{
   MFEM_ASSERT(x.Size() == e.Size(), "Incompatible vector sizes: "
               << x.Size() << " and " << e.Size());
   if (etaVec)
   {
      MFEM_ASSERT(x.Size() == etaVec->Size(), "Incorrect scaling vector size: "
                  << etaVec->Size() << " should match " << x.Size());
   }

   double scale = 0.0;
   double sum = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      double eta = (etaVec) ? (*etaVec)(i) : etaConst;

      if (e(i) != 0.0)
      {
         const double absdata = std::abs(e(i)) / (eta + std::abs(x(i)));
         if (scale <= absdata)
         {
            const double sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         } // end if scale <= absdata
         const double sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg); // else scale > absdata
      } // end if e(i) != 0.0
   }

   return scale * std::sqrt(sum);
}

#ifdef MFEM_USE_MPI

double ParMaxAbsRelDiffMeasure::Eval(Vector &x, Vector &e)
{
   MFEM_ASSERT(x.Size() == e.Size(), "Incompatible vector sizes: "
               << x.Size() << " and " << e.Size());
   if (etaVec)
   {
      MFEM_ASSERT(x.Size() == etaVec->Size(), "Incorrect scaling vector size: "
                  << etaVec->Size() << " should match " << x.Size());
   }

   double max = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      double eta = (etaVec) ? (*etaVec)(i) : etaConst;

      max = std::max(std::abs(e(i)) / (eta + std::abs(x(i))), max);
   }

   double glb_max = 0.0;
   MPI_Allreduce(&max, &glb_max, 1, MPI_DOUBLE, MPI_MAX, comm);
   return glb_max;
}

double ParL2AbsRelDiffMeasure::Eval(Vector &x, Vector &e)
{
   MFEM_ASSERT(x.Size() == e.Size(), "Incompatible vector sizes: "
               << x.Size() << " and " << e.Size());
   if (etaVec)
   {
      MFEM_ASSERT(x.Size() == etaVec->Size(), "Incorrect scaling vector size: "
                  << etaVec->Size() << " should match " << x.Size());
   }

   double scale = 0.0;
   double sum = 0.0;

   for (int i = 0; i < x.Size(); i++)
   {
      double eta = (etaVec) ? (*etaVec)(i) : etaConst;

      if (e(i) != 0.0)
      {
         const double absdata = std::abs(e(i)) / (eta + std::abs(x(i)));
         if (scale <= absdata)
         {
            const double sqr_arg = scale / absdata;
            sum = 1.0 + sum * (sqr_arg * sqr_arg);
            scale = absdata;
            continue;
         } // end if scale <= absdata
         const double sqr_arg = absdata / scale;
         sum += (sqr_arg * sqr_arg); // else scale > absdata
      } // end if e(i) != 0.0
   }

   double loc_sum = scale * scale * sum;
   double glb_sum = 0.0;
   MPI_Allreduce(&loc_sum, &glb_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
   return glb_sum;
}
#endif

double StdAdjFactor::operator()(double err) const
{
   if (err == 0.0) { return 1.0; }

   return gamma * pow(tol / err, kI);
}

double PIAdjFactor::operator()(double err) const
{
   if (err == 0.0) { return 1.0; }

   double theta = pow(tol / err, kI) *
                  ((prev_err > 0.0) ?
                   pow(prev_err / err, kP) : 1.0);
   prev_err = err;

   return theta;
}

double PIDAdjFactor::operator()(double err) const
{
   if (err == 0.0) { return 1.0; }

   double theta = pow(tol / err, kI) *
                  ((prev_err1 > 0.0) ?
                   pow(prev_err1 / err, kP) : 1.0) *
                  ((prev_err2 > 0.0) ?
                   pow(prev_err1 * prev_err1 / (err * prev_err2), kD) : 1.0);

   prev_err2 = prev_err1;
   prev_err1 = err;

   return theta;
}

void SecondOrderODESolver::Init(SecondOrderTimeDependentOperator &f)
{
   this->f = &f;
   mem_type = GetMemoryType(f.GetMemoryClass());
}

void NewmarkSolver::Init(SecondOrderTimeDependentOperator &f_)
{
   SecondOrderODESolver::Init(f_);
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

void NewmarkSolver::Step(Vector &x, Vector &dxdt, double &t, double &dt)
{
   double fac0 = 0.5 - beta;
   double fac2 = 1.0 - gamma;
   double fac3 = beta;
   double fac4 = gamma;

   // In the first pass compute d2xdt2 directy from operator.
   if (first)
   {
      f->Mult(x, dxdt, d2xdt2);
      first = false;
   }
   f->SetTime(t + dt);

   x.Add(dt, dxdt);
   x.Add(fac0*dt*dt, d2xdt2);
   dxdt.Add(fac2*dt, d2xdt2);

   f->SetTime(t + dt);
   f->ImplicitSolve(fac3*dt*dt, fac4*dt, x, dxdt, d2xdt2);

   x   .Add(fac3*dt*dt, d2xdt2);
   dxdt.Add(fac4*dt,    d2xdt2);
   t += dt;
}

void GeneralizedAlpha2Solver::Init(SecondOrderTimeDependentOperator &f_)
{
   SecondOrderODESolver::Init(f_);
   xa.SetSize(f->Width());
   va.SetSize(f->Width());
   aa.SetSize(f->Width());
   d2xdt2.SetSize(f->Width());
   d2xdt2 = 0.0;
   nstate = 0;
}

const Vector &GeneralizedAlpha2Solver::GetStateVector(int i)
{
   MFEM_ASSERT( (i == 0) && (nstate == 1),
                "GeneralizedAlpha2Solver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);
   return d2xdt2;
}


void GeneralizedAlpha2Solver::GetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i == 0) && (nstate == 1),
                "GeneralizedAlpha2Solver::GetStateVector \n" <<
                " - Tried to get non-existent state "<<i);
   state = d2xdt2;
}

void GeneralizedAlpha2Solver::SetStateVector(int i, Vector &state)
{
   MFEM_ASSERT( (i == 0),
                "GeneralizedAlpha2Solver::SetStateVector \n" <<
                " - Tried to set non-existent state "<<i);
   d2xdt2 = state;
   nstate = 1;
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

void GeneralizedAlpha2Solver::Step(Vector &x, Vector &dxdt,
                                   double &t, double &dt)
{
   double fac0 = (0.5 - (beta/alpha_m));
   double fac1 = alpha_f;
   double fac2 = alpha_f*(1.0 - (gamma/alpha_m));
   double fac3 = beta*alpha_f/alpha_m;
   double fac4 = gamma*alpha_f/alpha_m;
   double fac5 = alpha_m;

   // In the first pass compute d2xdt2 directy from operator.
   if (nstate == 0)
   {
      f->Mult(x, dxdt, d2xdt2);
      nstate = 1;
   }

   // Predict alpha levels
   add(dxdt, fac0*dt, d2xdt2, va);
   add(x, fac1*dt, va, xa);
   add(dxdt, fac2*dt, d2xdt2, va);

   // Solve alpha levels
   f->SetTime(t + dt);
   f->ImplicitSolve(fac3*dt*dt, fac4*dt, xa, va, aa);

   // Correct alpha levels
   xa.Add(fac3*dt*dt, aa);
   va.Add(fac4*dt,    aa);

   // Extrapolate
   x *= 1.0 - 1.0/fac1;
   x.Add (1.0/fac1, xa);

   dxdt *= 1.0 - 1.0/fac1;
   dxdt.Add (1.0/fac1, va);

   d2xdt2 *= 1.0 - 1.0/fac5;
   d2xdt2.Add (1.0/fac5, aa);

   t += dt;
}

}
