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
   /// Pointer to the associated TimeDependentOperator.
   TimeDependentOperator *f;  // f(.,t) : R^n --> R^n

public:
   ODESolver() : f(NULL) { }

   /// Associate a TimeDependentOperator with the ODE solver.
   /** This method has to be called:
       - Before the first call to Step().
       - When the dimensions of the associated TimeDependentOperator change.
       - When a time stepping sequence has to be restarted.
       - To change the associated TimeDependentOperator. */
   virtual void Init(TimeDependentOperator &f)
   {
      this->f = &f;
   }

   /** @brief Perform a time step from time @a t [in] to time @a t [out] based
       on the requested step size @a dt [in]. */
   /** @param[in,out] x   Approximate solution.
       @param[in,out] t   Time associated with the approximate solution @a x.
       @param[in,out] dt  Time step size.

       The following rules describe the common behavior of the method:
       - The input @a x [in] is the approximate solution for the input time
         @a t [in].
       - The input @a dt [in] is the desired time step size, defining the desired
         target time: t [target] = @a t [in] + @a dt [in].
       - The output @a x [out] is the approximate solution for the output time
         @a t [out].
       - The output @a dt [out] is the last time step taken by the method which
         may be smaller or larger than the input @a dt [in] value, e.g. because
         of time step control.
       - The method may perform more than one time step internally; in this case
         @a dt [out] is the last internal time step size.
       - The output value of @a t [out] may be smaller or larger than
         t [target], however, it is not smaller than @a t [in] + @a dt [out], if
         at least one internal time step was performed.
       - The value @a x [out] may be obtained by interpolation using internally
         stored data.
       - In some cases, the contents of @a x [in] may not be used, e.g. when
         @a x [out] from a previous Step() call was obtained by interpolation.
       - In consecutive calls to this method, the output @a t [out] of one
         Step() call has to be the same as the input @a t [in] to the next
         Step() call.
       - If the previous rule has to be broken, e.g. to restart a time stepping
         sequence, then the ODE solver must be re-initialized by calling Init()
         between the two Step() calls. */
   virtual void Step(Vector &x, double &t, double &dt) = 0;

   /// Perform time integration from time @a t [in] to time @a tf [in].
   /** @param[in,out] x   Approximate solution.
       @param[in,out] t   Time associated with the approximate solution @a x.
       @param[in,out] dt  Time step size.
       @param[in]     tf  Requested final time.

       The default implementation makes consecutive calls to Step() until
       reaching @a tf.
       The following rules describe the common behavior of the method:
       - The input @a x [in] is the approximate solution for the input time
         @a t [in].
       - The input @a dt [in] is the initial time step size.
       - The output @a dt [out] is the last time step taken by the method which
         may be smaller or larger than the input @a dt [in] value, e.g. because
         of time step control.
       - The output value of @a t [out] is not smaller than @a tf [in]. */
   virtual void Run(Vector &x, double &t, double &dt, double tf)
   {
      while (t < tf) { Step(x, t, dt); }
   }

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


/// Generalized-alpha ODE solver from "A generalized-α method for integrating
/// the filtered Navier–Stokes equations with a stabilized finite element
/// method" by K.E. Jansen, C.H. Whiting and G.M. Hulbert.
class GeneralizedAlphaSolver : public ODESolver
{
protected:
   Vector xdot,k,y;
   double alpha_f, alpha_m, gamma;
   bool first;

   void SetRhoInf(double rho_inf);
   void PrintProperties(std::ostream &out = mfem::out);
public:

   GeneralizedAlphaSolver(double rho = 1.0) { SetRhoInf(rho); };

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(Vector &x, double &t, double &dt);
};


/// The SIASolver class is based on the Symplectic Integration Algorithm
/// described in "A Symplectic Integration Algorithm for Separable Hamiltonian
/// Functions" by J. Candy and W. Rozmus, Journal of Computational Physics,
/// Vol. 92, pages 230-256 (1991).

/** The Symplectic Integration Algorithm (SIA) is designed for systems of first
    order ODEs derived from a Hamiltonian.
       H(q,p,t) = T(p) + V(q,t)
    Which leads to the equations:
       dq/dt = dT/dp
       dp/dt = -dV/dq
    In the integrator the operators P and F are defined to be:
       P = dT/dp
       F = -dV/dq
 */
class SIASolver
{
public:
   SIASolver() : F_(NULL), P_(NULL) {}

   virtual void Init(Operator &P, TimeDependentOperator & F);

   virtual void Step(Vector &q, Vector &p, double &t, double &dt) = 0;

   virtual void Run(Vector &q, Vector &p, double &t, double &dt, double tf)
   {
      while (t < tf) { Step(q, p, t, dt); }
   }

   virtual ~SIASolver() {}

protected:
   TimeDependentOperator * F_; // p_{i+1} = p_{i} + dt F(q_{i})
   Operator              * P_; // q_{i+1} = q_{i} + dt P(p_{i+1})

   mutable Vector dp_;
   mutable Vector dq_;
};

// First Order Symplectic Integration Algorithm
class SIA1Solver : public SIASolver
{
public:
   SIA1Solver() {}
   void Step(Vector &q, Vector &p, double &t, double &dt);
};

// Second Order Symplectic Integration Algorithm
class SIA2Solver : public SIASolver
{
public:
   SIA2Solver() {}
   void Step(Vector &q, Vector &p, double &t, double &dt);
};

// Variable order Symplectic Integration Algorithm (orders 1-4)
class SIAVSolver : public SIASolver
{
public:
   SIAVSolver(int order);
   void Step(Vector &q, Vector &p, double &t, double &dt);

private:
   int order_;

   Array<double> a_;
   Array<double> b_;
};

}

#endif
