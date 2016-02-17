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

#ifndef MFEM_ODE
#define MFEM_ODE

#include "../config/config.hpp"
#include "operator.hpp"

namespace mfem
{

/// Abstract class for solving systems of ODEs: dx/dt = f(x,t)
class ODESolver
{
protected:
   TimeDependentOperator *f; // f(.,t) : R^n --> R^n

public:
   ODESolver() : f(NULL) { }

   virtual void Init(TimeDependentOperator &_f) { f = &_f; }

   virtual void Step(Vector &x, double &t, double &dt) = 0;

   virtual ~ODESolver() { }
};


/// The classical forward Euler method
class ForwardEulerSolver : public ODESolver
{
private:
   Vector dxdt;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/** A family of explicit second-order RK2 methods. Some choices for the
    parameter 'a' are:
    a = 1/2 - the midpoint method
    a =  1  - Heun's method
    a = 2/3 - default, has minimal truncation error. */
class RK2Solver : public ODESolver
{
private:
   double a;
   Vector dxdt, x1;

public:
   RK2Solver(const double _a = 2./3.) : a(_a) { }

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/// Third-order, strong stability preserving (SSP) Runge-Kutta method
class RK3SSPSolver : public ODESolver
{
private:
   Vector y, k;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/// The classical explicit forth-order Runge-Kutta method, RK4
class RK4Solver : public ODESolver
{
private:
   Vector y, k, z;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/** An explicit Runge-Kutta method corresponding to a general Butcher tableau
    +--------+----------------------+
    | c[0]   | a[0]                 |
    | c[1]   | a[1] a[2]            |
    | ...    |    ...               |
    | c[s-2] | ...   a[s(s-1)/2-1]  |
    +--------+----------------------+
    |        | b[0] b[1] ... b[s-1] |
    +--------+----------------------+ */
class ExplicitRKSolver : public ODESolver
{
private:
   int s;
   const double *a, *b, *c;
   Vector y, *k;

public:
   ExplicitRKSolver(int _s, const double *_a, const double *_b,
                    const double *_c);

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);

   virtual ~ExplicitRKSolver();
};


/** An 8-stage, 6th order RK method. From Verner's "efficient" 9-stage 6(5)
    pair. */
class RK6Solver : public ExplicitRKSolver
{
private:
   static const double a[28], b[8], c[7];

public:
   RK6Solver() : ExplicitRKSolver(8, a, b, c) { }
};


/** A 12-stage, 8th order RK method. From Verner's "efficient" 13-stage 8(7)
    pair. */
class RK8Solver : public ExplicitRKSolver
{
private:
   static const double a[66], b[12], c[11];

public:
   RK8Solver() : ExplicitRKSolver(12, a, b, c) { }
};


/// Backward Euler ODE solver. L-stable.
class BackwardEulerSolver : public ODESolver
{
protected:
   Vector k;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/// Implicit midpoint method. A-stable, not L-stable.
class ImplicitMidpointSolver : public ODESolver
{
protected:
   Vector k;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/** Two stage, singly diagonal implicit Runge-Kutta (SDIRK) methods;
    the choices for gamma_opt are:
    0 - 3rd order method, not A-stable
    1 - 3rd order method, A-stable, not L-stable (default)
    2 - 2nd order method, L-stable
    3 - 2nd order method, L-stable (has solves outside [t,t+dt]). */
class SDIRK23Solver : public ODESolver
{
protected:
   double gamma;
   Vector k, y;

public:
   SDIRK23Solver(int gamma_opt = 1);

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/** Three stage, singly diagonal implicit Runge-Kutta (SDIRK) method of
    order 4. A-stable, not L-stable. */
class SDIRK34Solver : public ODESolver
{
protected:
   Vector k, y, z;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/** Three stage, singly diagonal implicit Runge-Kutta (SDIRK) method of
    order 3. L-stable. */
class SDIRK33Solver : public ODESolver
{
protected:
   Vector k, y;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};

}

#endif
