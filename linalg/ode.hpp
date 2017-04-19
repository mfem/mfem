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
     virtual void Step(Vector &x, double &t, double &dt) {
        mfem_error("ODESolver::Step(Vector) is not overloaded!");
     }
  #ifdef MFEM_USE_OCCA
     virtual void Step(OccaVector &x, double &t, double &dt) {
        mfem_error("ODESolver::Step(OccaVector) is not overloaded!");
     }
  #endif

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
#ifdef MFEM_USE_OCCA
   virtual void Run(OccaVector &x, double &t, double &dt, double tf)
   {
      while (t < tf) { Step(x, t, dt); }
   }
#endif

   virtual ~ODESolver() { }
};


/// The classical forward Euler method
template <class TVector>
class TForwardEulerSolver : public ODESolver
{
private:
   TVector dxdt;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/** A family of explicit second-order RK2 methods. Some choices for the
    parameter 'a' are:
    a = 1/2 - the midpoint method
    a =  1  - Heun's method
    a = 2/3 - default, has minimal truncation error. */
template <class TVector>
class TRK2Solver : public ODESolver
{
private:
   double a;
   TVector dxdt, x1;

public:
   TRK2Solver(const double _a = 2./3.) : a(_a) { }

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/// Third-order, strong stability preserving (SSP) Runge-Kutta method
template <class TVector>
class TRK3SSPSolver : public ODESolver
{
private:
   TVector y, k;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/// The classical explicit forth-order Runge-Kutta method, RK4
template <class TVector>
class TRK4Solver : public ODESolver
{
private:
   TVector y, k, z;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
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
template <class TVector>
class TExplicitRKSolver : public ODESolver
{
private:
   int s;
   const double *a, *b, *c;
   TVector y, *k;

public:
   TExplicitRKSolver(int _s, const double *_a, const double *_b,
                    const double *_c);

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);

   virtual ~TExplicitRKSolver();
};


/** An 8-stage, 6th order RK method. From Verner's "efficient" 9-stage 6(5)
    pair. */
template <class TVector>
class TRK6Solver : public TExplicitRKSolver<TVector>
{
private:
   static const double a[28], b[8], c[7];

public:
   TRK6Solver() : TExplicitRKSolver<TVector>(8, a, b, c) { }
};


/** A 12-stage, 8th order RK method. From Verner's "efficient" 13-stage 8(7)
    pair. */
template <class TVector>
class TRK8Solver : public TExplicitRKSolver<TVector>
{
private:
   static const double a[66], b[12], c[11];

public:
   TRK8Solver() : TExplicitRKSolver<TVector>(12, a, b, c) { }
};


/// Backward Euler ODE solver. L-stable.
template <class TVector>
class TBackwardEulerSolver : public ODESolver
{
protected:
   TVector k;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/// Implicit midpoint method. A-stable, not L-stable.
template <class TVector>
class TImplicitMidpointSolver : public ODESolver
{
protected:
   TVector k;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/** Two stage, singly diagonal implicit Runge-Kutta (SDIRK) methods;
    the choices for gamma_opt are:
    0 - 3rd order method, not A-stable
    1 - 3rd order method, A-stable, not L-stable (default)
    2 - 2nd order method, L-stable
    3 - 2nd order method, L-stable (has solves outside [t,t+dt]). */
template <class TVector>
class TSDIRK23Solver : public ODESolver
{
protected:
   double gamma;
   TVector k, y;

public:
   TSDIRK23Solver(int gamma_opt = 1);

   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/** Three stage, singly diagonal implicit Runge-Kutta (SDIRK) method of
    order 4. A-stable, not L-stable. */
template <class TVector>
class TSDIRK34Solver : public ODESolver
{
protected:
   TVector k, y, z;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};


/** Three stage, singly diagonal implicit Runge-Kutta (SDIRK) method of
    order 3. L-stable. */
template <class TVector>
class TSDIRK33Solver : public ODESolver
{
protected:
   TVector k, y;

public:
   virtual void Init(TimeDependentOperator &_f);

   virtual void Step(TVector &x, double &t, double &dt);
};

typedef TForwardEulerSolver<Vector>     ForwardEulerSolver;
typedef TRK2Solver<Vector>              RK2Solver;
typedef TRK3SSPSolver<Vector>           RK3SSPSolver;
typedef TRK4Solver<Vector>              RK4Solver;
typedef TExplicitRKSolver<Vector>       ExplicitRKSolver;
typedef TRK6Solver<Vector>              RK6Solver;
typedef TRK8Solver<Vector>              RK8Solver;
typedef TBackwardEulerSolver<Vector>    BackwardEulerSolver;
typedef TImplicitMidpointSolver<Vector> ImplicitMidpointSolver;
typedef TSDIRK23Solver<Vector>          SDIRK23Solver;
typedef TSDIRK34Solver<Vector>          SDIRK34Solver;
typedef TSDIRK33Solver<Vector>          SDIRK33Solver;

#ifdef MFEM_USE_OCCA
typedef TForwardEulerSolver<OccaVector>     OccaForwardEulerSolver;
typedef TRK2Solver<OccaVector>              OccaRK2Solver;
typedef TRK3SSPSolver<OccaVector>           OccaRK3SSPSolver;
typedef TRK4Solver<OccaVector>              OccaRK4Solver;
typedef TExplicitRKSolver<OccaVector>       OccaExplicitRKSolver;
typedef TRK6Solver<OccaVector>              OccaRK6Solver;
typedef TRK8Solver<OccaVector>              OccaRK8Solver;
typedef TBackwardEulerSolver<OccaVector>    OccaBackwardEulerSolver;
typedef TImplicitMidpointSolver<OccaVector> OccaImplicitMidpointSolver;
typedef TSDIRK23Solver<OccaVector>          OccaSDIRK23Solver;
typedef TSDIRK34Solver<OccaVector>          OccaSDIRK34Solver;
typedef TSDIRK33Solver<OccaVector>          OccaSDIRK33Solver;
#endif

}

#include "ode.tpp"

#endif
