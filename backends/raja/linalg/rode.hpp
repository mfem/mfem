// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef LAGHOS_RAJA_ODE
#define LAGHOS_RAJA_ODE

namespace mfem {

  // ***************************************************************************
  class RajaODESolver{
  protected:
    RajaTimeDependentOperator *f;
  public:
    RajaODESolver() : f(NULL){}
    virtual ~RajaODESolver(){}
    virtual void Init(RajaTimeDependentOperator &f) { this->f = &f; }
    virtual void Step(RajaVector &x, double &t, double &dt) =0;
  };

  // ***************************************************************************
  class RajaForwardEulerSolver : public RajaODESolver{
  private:
    RajaVector dxdt;
  public:
    void Init(RajaTimeDependentOperator &_f){
      f = &_f;
      dxdt.SetSize(f->Width());
    }
    void Step(RajaVector &x, double &t, double &dt){
      push(SkyBlue);
      f->SetTime(t);
      f->Mult(x, dxdt);
      x.Add(dt, dxdt);
      t += dt;
      pop();
    }
  };

  // ***************************************************************************
  class RajaRK2Solver : public RajaODESolver{
  private:
    double a;
    RajaVector dxdt, x1;
  public:
    RajaRK2Solver(const double _a = 2./3.) : a(_a) { }
    void Init(RajaTimeDependentOperator &_f){
      f = &_f;
      int n = f->Width();
      dxdt.SetSize(n);
      x1.SetSize(n);
    }
    void Step(RajaVector &x, double &t, double &dt){
      push(SkyBlue);
      const double b = 0.5/a;
      f->SetTime(t);
      f->Mult(x, dxdt);
      add(x, (1. - b)*dt, dxdt, x1);
      x.Add(a*dt, dxdt);
      f->SetTime(t + a*dt);
      f->Mult(x, dxdt);
      add(x1, b*dt, dxdt, x);
      t += dt;
      pop();
    }
  };

  // ***************************************************************************
 class RajaRK3SSPSolver : public RajaODESolver{
  private:
    RajaVector y, k;
 public:
   void Init(RajaTimeDependentOperator &_f){
     f = &_f;
     int n = f->Width();
     y.SetSize(n);
     k.SetSize(n);
   }
   void Step(RajaVector &x, double &t, double &dt){
      push(SkyBlue);
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
     pop();
   }
 };

  // ***************************************************************************
  class RajaRK4Solver : public RajaODESolver{
  private:
    RajaVector y, k, z;
  public:
    void Init(RajaTimeDependentOperator &_f){
      f = &_f;
      int n = RajaODESolver::f->Width();
      y.SetSize(n);
      k.SetSize(n);
      z.SetSize(n);
    }
    
    void Step(RajaVector &x, double &t, double &dt){
      push(SkyBlue);
      f->SetTime(t);
      push(k1,SkyBlue);
      f->Mult(x, k); // k1
      pop();
      
      push(addxx,SkyBlue);
      add(x, dt/2, k, y);
      add(x, dt/6, k, z);pop();
      
      f->SetTime(t + dt/2);
      
      push(k2,SkyBlue);
      f->Mult(y, k); // k2
      pop();
      
      push(addxz1,SkyBlue);
      add(x, dt/2, k, y);
      z.Add(dt/3, k);pop();
      
      push(k3,SkyBlue);
      f->Mult(y, k); // k3
      pop();

      push(addxz2,SkyBlue);
      add(x, dt, k, y);
      z.Add(dt/3, k);
      f->SetTime(t + dt);pop();
      
      push(k4,SkyBlue);
      f->Mult(y, k); // k4
      pop();
      
      push(addz,SkyBlue);
      add(z, dt/6, k, x);pop();
      
      t += dt;
      pop();
    }
  };

  // ***************************************************************************
  class RajaExplicitRKSolver : public RajaODESolver{
  private:
    int s;
    const double *a, *b, *c;
    RajaVector y, *k;
  public:
    RajaExplicitRKSolver(int _s, const double *_a,
                         const double *_b, const double *_c){
      s = _s;
      a = _a;
      b = _b;
      c = _c;
      k = new RajaVector[s];
    }
    void Init(RajaTimeDependentOperator &_f){
      f = &_f;
      int n = f->Width();
      y.SetSize(n);
      for (int i = 0; i < s; i++)    {
        k[i].SetSize(n);
      }
    }
    void Step(RajaVector &x, double &t, double &dt){
      push(SkyBlue);
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
      pop();
    }
    ~RajaExplicitRKSolver(){
      delete [] k;
    }
  };

  // ***************************************************************************
  // ***************************************************************************
  static const double RK6_a[28] = {
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
  
  static const double RK6_b[8] = {
    .3438957868357036009278820124728322386520e-1,
    0.,
    0.,
    .2582624555633503404659558098586120858767,
    .4209371189673537150642551514069801967032,
    4.405396469669310170148836816197095664891,
    -176.4831190242986576151740942499002125029,
    172.3641334014150730294022582711902413315
  };

  static const double RK6_c[7] = {
    .6e-1,
    .9593333333333333333333333333333333333333e-1,
    .1439,
    .4973,
    .9725,
    .9995,
    1.,
  };

  class RajaRK6Solver : public RajaExplicitRKSolver{
  public:
    RajaRK6Solver() : RajaExplicitRKSolver(8, RK6_a, RK6_b, RK6_c) { }
  };

} // mfem

#endif // LAGHOS_RAJA_ODE
