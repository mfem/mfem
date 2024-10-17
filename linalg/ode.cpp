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

#include "../general/communication.hpp"
#include "operator.hpp"
#include "ode.hpp"

namespace mfem
{

std::string ODESolver::ExplicitTypes =
   "\n\tExplicit solver: \n\t"
   "        RK      :  1 - Forward Euler, 2 - RK2(0.5), 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
   "        AB      : 11 - AB1, 12 - AB2, 13 - AB3, 14 - AB4, 15 - AB5\n";

std::string ODESolver::ImplicitTypes  =
   "\n\tImplicit solver: \n\t"
   "        (L-Stab): 21 - Backward Euler, 22 - SDIRK23(2), 23 - SDIRK33,\n\t"
   "        (A-Stab): 32 - Implicit Midpoint, 33 - SDIRK23, 34 - SDIRK34,\n\t"
   "        GA      : 40 -- 50  - Generalized-alpha,\n\t"
   "        AM      : 51 - AM1, 52 - AM2, 53 - AM3, 54 - AM4\n";

std::string ODESolver::Types = ODESolver::ExplicitTypes +
                               ODESolver::ImplicitTypes;

std::unique_ptr<ODESolver> ODESolver::Select(int ode_solver_type)
{
   if (ode_solver_type < 20)
   {
      return SelectExplicit(ode_solver_type);
   }
   else
   {
      return SelectImplicit(ode_solver_type);
   }
}

std::unique_ptr<ODESolver> ODESolver::SelectExplicit(int ode_solver_type)
{
   using ode_ptr = std::unique_ptr<ODESolver>;
   switch (ode_solver_type)
   {
      // Explicit RK methods
      case 1: return ode_ptr(new ForwardEulerSolver);
      case 2: return ode_ptr(new RK2Solver(0.5)); // midpoint method
      case 3: return ode_ptr(new RK3SSPSolver);
      case 4: return ode_ptr(new RK4Solver);
      case 6: return ode_ptr(new RK6Solver);

      // Explicit AB methods
      case 11: return ode_ptr(new AB1Solver);
      case 12: return ode_ptr(new AB2Solver);
      case 13: return ode_ptr(new AB3Solver);
      case 14: return ode_ptr(new AB4Solver);
      case 15: return ode_ptr(new AB5Solver);

      default:
         MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type);
   }
}

std::unique_ptr<ODESolver> ODESolver::SelectImplicit(int ode_solver_type)
{
   using ode_ptr = std::unique_ptr<ODESolver>;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 21: return ode_ptr(new BackwardEulerSolver);
      case 22: return ode_ptr(new SDIRK23Solver(2));
      case 23: return ode_ptr(new SDIRK33Solver);

      // Implicit A-stable methods (not L-stable)
      case 32: return ode_ptr(new ImplicitMidpointSolver);
      case 33: return ode_ptr(new SDIRK23Solver);
      case 34: return ode_ptr(new SDIRK34Solver);

      // Implicit generalized alpha
      case 40:  return ode_ptr(new GeneralizedAlphaSolver(0.0));
      case 41:  return ode_ptr(new GeneralizedAlphaSolver(0.1));
      case 42:  return ode_ptr(new GeneralizedAlphaSolver(0.2));
      case 43:  return ode_ptr(new GeneralizedAlphaSolver(0.3));
      case 44:  return ode_ptr(new GeneralizedAlphaSolver(0.4));
      case 45:  return ode_ptr(new GeneralizedAlphaSolver(0.5));
      case 46:  return ode_ptr(new GeneralizedAlphaSolver(0.6));
      case 47:  return ode_ptr(new GeneralizedAlphaSolver(0.7));
      case 48:  return ode_ptr(new GeneralizedAlphaSolver(0.8));
      case 49:  return ode_ptr(new GeneralizedAlphaSolver(0.9));
      case 50:  return ode_ptr(new GeneralizedAlphaSolver(1.0));

      // Implicit AM methods
      case 51: return ode_ptr(new AM1Solver);
      case 52: return ode_ptr(new AM2Solver);
      case 53: return ode_ptr(new AM3Solver);
      case 54: return ode_ptr(new AM4Solver);

      default:
         MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type );
   }
}


void ODEStateDataVector::SetSize( int vsize, MemoryType m_t)
{
   mem_type = m_t;
   for (int i = 0; i < smax; i++)
   {
      idx[i] = smax - i - 1;
      data[i].SetSize(vsize, mem_type);
   }

   ss = 0;
}

const Vector &ODEStateDataVector::Get(int i) const
{
   MFEM_ASSERT_INDEX_IN_RANGE(i,0,ss);
   return data[idx[i]];
}

Vector &ODEStateDataVector::Get(int i)
{
   MFEM_ASSERT_INDEX_IN_RANGE(i,0,ss);
   return data[idx[i]];
}

void ODEStateDataVector::Get(int i, Vector &vec) const
{
   MFEM_ASSERT_INDEX_IN_RANGE(i,0,ss);
   vec = data[idx[i]];
}

void ODEStateDataVector::Set(int i, Vector &state)
{
   MFEM_ASSERT_INDEX_IN_RANGE(i,0,smax);
   data[idx[i]] = state;
}

void ODEStateDataVector::Append(Vector &state)
{
   ShiftStages();
   data[idx[0]] = state;
   Increment();
}

void ODEStateDataVector::Print(std::ostream &os) const
{
   os << ss <<"/" <<smax<<std::endl;
   idx.Print(os);
   for (int i = 0; i < ss; i++) { data[idx[i]].Print(os); }
}


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

void ForwardEulerSolver::Step(Vector &x, real_t &t, real_t &dt)
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

void RK2Solver::Step(Vector &x, real_t &t, real_t &dt)
{
   //  0 |
   //  a |  a
   // ---+--------
   //    | 1-b  b      b = 1/(2a)

   const real_t b = 0.5/a;

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

void RK3SSPSolver::Step(Vector &x, real_t &t, real_t &dt)
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

void RK4Solver::Step(Vector &x, real_t &t, real_t &dt)
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

ExplicitRKSolver::ExplicitRKSolver(int s_, const real_t *a_, const real_t *b_,
                                   const real_t *c_)
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

void ExplicitRKSolver::Step(Vector &x, real_t &t, real_t &dt)
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

const real_t RK6Solver::a[] =
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
const real_t RK6Solver::b[] =
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
const real_t RK6Solver::c[] =
{
   .6e-1,
   .9593333333333333333333333333333333333333e-1,
   .1439,
   .4973,
   .9725,
   .9995,
   1.,
};

const real_t RK8Solver::a[] =
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
const real_t RK8Solver::b[] =
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
const real_t RK8Solver::c[] =
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


AdamsBashforthSolver::AdamsBashforthSolver(int s_, const real_t *a_):
   stages(s_), state(s_)
{
   a = a_;
}

void AdamsBashforthSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   if (RKsolver) { RKsolver->Init(f_); }
   state.SetSize(f->Width(), mem_type);
   dt_ = -1.0;
}

void AdamsBashforthSolver::Step(Vector &x, real_t &t, real_t &dt)
{
   CheckTimestep(dt);

   if (state.Size() >= stages -1)
   {
      f->SetTime(t);
      f->Mult(x, state[0]);
      state.Increment();
      for (int i = 0; i < stages; i++)
      {
         x.Add(a[i]*dt, state[i]);
      }
      t += dt;
   }
   else
   {
      f->Mult(x,state[0]);
      RKsolver->Step(x,t,dt);
      state.Increment();
   }

   state.ShiftStages();
}

void AdamsBashforthSolver::CheckTimestep(real_t dt)
{
   if (dt_ < 0.0)
   {
      dt_ = dt;
      return;
   }
   else if (fabs(dt-dt_) >10*std::numeric_limits<real_t>::epsilon())
   {
      state.Reset();
      dt_ = dt;

      if (print())
      {
         mfem::out << "WARNING:" << std::endl;
         mfem::out << " - Time step changed" << std::endl;
         mfem::out << " - Purging time stepping history" << std::endl;
         mfem::out << " - Will run Runge-Kutta to rebuild history" << std::endl;
      }
   }
}

const real_t AB1Solver::a[] =
{1.0};
const real_t AB2Solver::a[] =
{1.5,-0.5};
const real_t AB3Solver::a[] =
{23.0/12.0,-4.0/3.0, 5.0/12.0};
const real_t AB4Solver::a[] =
{55.0/24.0,-59.0/24.0, 37.0/24.0,-9.0/24.0};
const real_t AB5Solver::a[] =
{1901.0/720.0,-2774.0/720.0, 2616.0/720.0,-1274.0/720.0, 251.0/720.0};


AdamsMoultonSolver::AdamsMoultonSolver(int s_, const real_t *a_):
   stages(s_), state(s_)
{
   a = a_;
}

void AdamsMoultonSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   if (RKsolver) { RKsolver->Init(f_); }
   state.SetSize(f->Width(), mem_type);
   dt_ = -1.0;
}

void AdamsMoultonSolver::Step(Vector &x, real_t &t, real_t &dt)
{
   if (dt_ < 0.0)
   {
      dt_ = dt;
   }
   else if (fabs(dt-dt_) > 10*std::numeric_limits<real_t>::epsilon())
   {
      state.Reset();
      dt_ = dt;

      if (print())
      {
         mfem::out << "WARNING:" << std::endl;
         mfem::out << " - Time step changed" << std::endl;
         mfem::out << " - Purging time stepping history" << std::endl;
         mfem::out << " - Will run Runge-Kutta to rebuild history" << std::endl;
      }
   }

   if ((state.Size() == 0)&&(stages>1))
   {
      f->Mult(x,state[0]);
      state.Increment();
   }

   if (state.Size() >= stages )
   {
      f->SetTime(t);
      for (int i = 0; i < stages; i++)
      {
         x.Add(a[i+1]*dt, state[i]);
      }
      state.ShiftStages();
      f->ImplicitSolve(a[0]*dt, x, state[0]);
      x.Add(a[0]*dt, state[0]);
      t += dt;
   }
   else
   {
      state.ShiftStages();
      RKsolver->Step(x,t,dt);
      f->Mult(x,state[0]);
      state.Increment();
   }
}

const real_t AM1Solver::a[] =
{0.5, 0.5};
const real_t AM2Solver::a[] =
{5.0/12.0, 2.0/3.0, -1.0/12.0};
const real_t AM3Solver::a[] =
{3.0/8.0, 19.0/24.0,-5.0/24.0, 1.0/24.0};
const real_t AM4Solver::a[] =
{251.0/720.0,646.0/720.0,-264.0/720.0, 106.0/720.0, -19.0/720.0};


void BackwardEulerSolver::Init(TimeDependentOperator &f_)
{
   ODESolver::Init(f_);
   k.SetSize(f->Width(), mem_type);
}

void BackwardEulerSolver::Step(Vector &x, real_t &t, real_t &dt)
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

void ImplicitMidpointSolver::Step(Vector &x, real_t &t, real_t &dt)
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

void SDIRK23Solver::Step(Vector &x, real_t &t, real_t &dt)
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

void SDIRK34Solver::Step(Vector &x, real_t &t, real_t &dt)
{
   //   a   |    a
   //  1/2  |  1/2-a    a
   //  1-a  |   2a    1-4a   a
   // ------+--------------------
   //       |    b    1-2b   b
   // note: two solves are outside [t,t+dt] since c1=a>1, c3=1-a<0
   const real_t a = 1./sqrt(3.)*cos(M_PI/18.) + 0.5;
   const real_t b = 1./(6.*(2.*a-1.)*(2.*a-1.));

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

void SDIRK33Solver::Step(Vector &x, real_t &t, real_t &dt)
{
   //   a  |   a
   //   c  |  c-a    a
   //   1  |   b   1-a-b  a
   // -----+----------------
   //      |   b   1-a-b  a
   const real_t a = 0.435866521508458999416019;
   const real_t b = 1.20849664917601007033648;
   const real_t c = 0.717933260754229499708010;

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

void TrapezoidalRuleSolver::Step(Vector &x, real_t &t, real_t &dt)
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

void ESDIRK32Solver::Step(Vector &x, real_t &t, real_t &dt)
{
   //   0   |    0      0    0
   //   2a  |    a      a    0
   //   1   |  1-b-a    b    a
   // ------+--------------------
   //       |  1-b-a    b    a
   const real_t a = (2.0 - sqrt(2.0)) / 2.0;
   const real_t b = (1.0 - 2.0*a) / (4.0*a);

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

void ESDIRK33Solver::Step(Vector &x, real_t &t, real_t &dt)
{
   //   0   |      0          0        0
   //   2a  |      a          a        0
   //   1   |    1-b-a        b        a
   // ------+----------------------------
   //       |  1-b_2-b_3     b_2      b_3
   const real_t a   = (3.0 + sqrt(3.0)) / 6.0;
   const real_t b   = (1.0 - 2.0*a) / (4.0*a);
   const real_t b_2 = 1.0 / ( 12.0*a*(1.0 - 2.0*a) );
   const real_t b_3 = (1.0 - 3.0*a) / ( 3.0*(1.0 - 2.0*a) );

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
   state.SetSize(f->Width(), mem_type);
}

void GeneralizedAlphaSolver::SetRhoInf(real_t rho_inf)
{
   rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
   rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

   // According to Jansen
   alpha_m = 0.5*(3.0 - rho_inf)/(1.0 + rho_inf);
   alpha_f = 1.0/(1.0 + rho_inf);
   gamma = 0.5 + alpha_m - alpha_f;
}

void GeneralizedAlphaSolver::PrintProperties(std::ostream &os)
{
   os << "Generalized alpha time integrator:" << std::endl;
   os << "alpha_m = " << alpha_m << std::endl;
   os << "alpha_f = " << alpha_f << std::endl;
   os << "gamma   = " << gamma   << std::endl;

   if (gamma == 0.5 + alpha_m - alpha_f)
   {
      os<<"Second order"<<" and ";
   }
   else
   {
      os<<"First order"<<" and ";
   }

   if ((alpha_m >= alpha_f)&&(alpha_f >= 0.5))
   {
      os<<"Stable"<<std::endl;
   }
   else
   {
      os<<"Unstable"<<std::endl;
   }
}

// This routine state[0] represents xdot
void GeneralizedAlphaSolver::Step(Vector &x, real_t &t, real_t &dt)
{
   if (state.Size() == 0)
   {
      f->Mult(x,state[0]);
      state.Increment();
   }

   // Set y = x + alpha_f*(1.0 - (gamma/alpha_m))*dt*xdot
   add(x, alpha_f*(1.0 - (gamma/alpha_m))*dt, state[0], y);

   // Solve k = f(y + dt_eff*k)
   real_t dt_eff = (gamma*alpha_f/alpha_m)*dt;
   f->SetTime(t + alpha_f*dt);
   f->ImplicitSolve(dt_eff, y, k);

   // Update x and xdot
   x.Add((1.0 - (gamma/alpha_m))*dt, state[0]);
   x.Add(       (gamma/alpha_m) *dt, k);

   state[0] *= (1.0-(1.0/alpha_m));
   state[0].Add((1.0/alpha_m),k);

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
SIA1Solver::Step(Vector &q, Vector &p, real_t &t, real_t &dt)
{
   F_->SetTime(t);
   F_->Mult(q,dp_);
   p.Add(dt,dp_);

   P_->Mult(p,dq_);
   q.Add(dt,dq_);

   t += dt;
}

void
SIA2Solver::Step(Vector &q, Vector &p, real_t &t, real_t &dt)
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
SIAVSolver::Step(Vector &q, Vector &p, real_t &t, real_t &dt)
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

std::string SecondOrderODESolver::Types =
   "ODE solver: \n\t"
   "  [0--10] - GeneralizedAlpha(0.1 * s),\n\t"
   "  11 - Average Acceleration, 12 - Linear Acceleration\n\t"
   "  13 - CentralDifference, 14 - FoxGoodwin";

SecondOrderODESolver* SecondOrderODESolver::Select(int ode_solver_type)
{
   SecondOrderODESolver*  ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Implicit methods
      case 0: ode_solver = new GeneralizedAlpha2Solver(0.0); break;
      case 1: ode_solver = new GeneralizedAlpha2Solver(0.1); break;
      case 2: ode_solver = new GeneralizedAlpha2Solver(0.2); break;
      case 3: ode_solver = new GeneralizedAlpha2Solver(0.3); break;
      case 4: ode_solver = new GeneralizedAlpha2Solver(0.4); break;
      case 5: ode_solver = new GeneralizedAlpha2Solver(0.5); break;
      case 6: ode_solver = new GeneralizedAlpha2Solver(0.6); break;
      case 7: ode_solver = new GeneralizedAlpha2Solver(0.7); break;
      case 8: ode_solver = new GeneralizedAlpha2Solver(0.8); break;
      case 9: ode_solver = new GeneralizedAlpha2Solver(0.9); break;
      case 10: ode_solver = new GeneralizedAlpha2Solver(1.0); break;

      case 11: ode_solver = new AverageAccelerationSolver(); break;
      case 12: ode_solver = new LinearAccelerationSolver(); break;
      case 13: ode_solver = new CentralDifferenceSolver(); break;
      case 14: ode_solver = new FoxGoodwinSolver(); break;

      default:
         MFEM_ABORT("Unknown ODE solver type: " << ode_solver_type);
   }
   return ode_solver;
}

// In this routine state[0] represents d2xdt2
void SecondOrderODESolver::EulerStep(Vector &x, Vector &dxdt, real_t &t,
                                     real_t &dt)
{
   x.Add(dt, dxdt);

   f->SetTime(t + dt);
   f->ImplicitSolve(0.5*dt*dt, dt, x, dxdt, state[0]);

   x   .Add(0.5*dt*dt, state[0]);
   dxdt.Add(dt,    state[0]);
   t += dt;
}

// In this routine state[0] represents d2xdt2
void SecondOrderODESolver::MidPointStep(Vector &x, Vector &dxdt, real_t &t,
                                        real_t &dt)
{
   x.Add(0.5*dt, dxdt);

   f->SetTime(t + dt);
   f->ImplicitSolve(0.25*dt*dt, 0.5*dt, x, dxdt, state[0]);

   x.Add(0.5*dt, dxdt);
   x.Add(0.5*dt*dt, state[0]);
   dxdt.Add(dt, state[0]);
   t += dt;
}

void SecondOrderODESolver::Init(SecondOrderTimeDependentOperator &f_)
{
   this->f = &f_;
   mem_type = GetMemoryType(f_.GetMemoryClass());
   state.SetSize(f->Width(), mem_type);
}

void NewmarkSolver::PrintProperties(std::ostream &os)
{
   os << "Newmark time integrator:" << std::endl;
   os << "beta    = " << beta  << std::endl;
   os << "gamma   = " << gamma << std::endl;

   if (gamma == 0.5)
   {
      os<<"Second order"<<" and ";
   }
   else
   {
      os<<"First order"<<" and ";
   }

   if ((gamma >= 0.5) && (beta >= (gamma + 0.5)*(gamma + 0.5)/4))
   {
      os<<"A-Stable"<<std::endl;
   }
   else if ((gamma >= 0.5) && (beta >= 0.5*gamma))
   {
      os<<"Conditionally stable"<<std::endl;
   }
   else
   {
      os<<"Unstable"<<std::endl;
   }
}

// In this routine state[0] represents d2xdt2
void NewmarkSolver::Step(Vector &x, Vector &dxdt, real_t &t, real_t &dt)
{
   real_t fac0 = 0.5 - beta;
   real_t fac2 = 1.0 - gamma;
   real_t fac3 = beta;
   real_t fac4 = gamma;

   // In the first pass compute d2xdt2 directly from operator.
   if (state.Size() == 0)
   {
      if (no_mult)
      {
         MidPointStep(x, dxdt, t, dt);
         return;
      }
      else
      {
         f->Mult(x, dxdt, state[0]);
      }
   }
   f->SetTime(t + dt);

   x.Add(dt, dxdt);
   x.Add(fac0*dt*dt, state[0]);
   dxdt.Add(fac2*dt, state[0]);

   f->SetTime(t + dt);
   f->ImplicitSolve(fac3*dt*dt, fac4*dt, x, dxdt, state[0]);

   x   .Add(fac3*dt*dt, state[0]);
   dxdt.Add(fac4*dt,    state[0]);
   t += dt;
}

void GeneralizedAlpha2Solver::Init(SecondOrderTimeDependentOperator &f_)
{
   SecondOrderODESolver::Init(f_);
   xa.SetSize(f->Width(), mem_type);
   va.SetSize(f->Width(), mem_type);
   aa.SetSize(f->Width(), mem_type);
}

void GeneralizedAlpha2Solver::PrintProperties(std::ostream &os)
{
   os << "Generalized alpha time integrator:" << std::endl;
   os << "alpha_m = " << alpha_m << std::endl;
   os << "alpha_f = " << alpha_f << std::endl;
   os << "beta    = " << beta    << std::endl;
   os << "gamma   = " << gamma   << std::endl;

   if (gamma == 0.5 + alpha_m - alpha_f)
   {
      os<<"Second order"<<" and ";
   }
   else
   {
      os<<"First order"<<" and ";
   }

   if ((alpha_m >= alpha_f)&&
       (alpha_f >= 0.5) &&
       (beta >= 0.25 + 0.5*(alpha_m - alpha_f)))
   {
      os<<"Stable"<<std::endl;
   }
   else
   {
      os<<"Unstable"<<std::endl;
   }
}

// In this routine state[0] represents d2xdt2
void GeneralizedAlpha2Solver::Step(Vector &x, Vector &dxdt,
                                   real_t &t, real_t &dt)
{
   real_t fac0 = (0.5 - (beta/alpha_m));
   real_t fac1 = alpha_f;
   real_t fac2 = alpha_f*(1.0 - (gamma/alpha_m));
   real_t fac3 = beta*alpha_f/alpha_m;
   real_t fac4 = gamma*alpha_f/alpha_m;
   real_t fac5 = alpha_m;

   // In the first pass compute d2xdt2 directly from operator.
   if (state.Size() == 0)
   {
      if (no_mult)
      {
         MidPointStep(x, dxdt, t, dt);
         return;
      }
      else
      {
         f->Mult(x, dxdt, state[0]);
      }
      state.Increment();
   }

   // Predict alpha levels
   add(dxdt, fac0*dt, state[0], va);
   add(x, fac1*dt, va, xa);
   add(dxdt, fac2*dt, state[0], va);

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

   state[0] *= 1.0 - 1.0/fac5;
   state[0].Add (1.0/fac5, aa);

   t += dt;
}

}
