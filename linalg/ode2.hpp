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

#ifndef MFEM_ODE2
#define MFEM_ODE2

#include "../config/config.hpp"
#include "operator.hpp"

namespace mfem
{

/// Abstract class for solving systems of ODEs: d2x/dt2 = f(x,dxdt,t)
class ODE2Solver
{
protected:
   /// Pointer to the associated TimeDependent2Operator.
   TimeDependent2Operator *f;  // f(.,t) : R^n --> R^n

public:
   ODE2Solver() : f(NULL) { }

   virtual void PrintProperties(std::ostream &out = mfem::out) {}

   /// Associate a TimeDependentOperator with the ODE solver.
   /** This method has to be called:
       - Before the first call to Step().
       - When the dimensions of the associated TimeDependentOperator change.
       - When a time stepping sequence has to be restarted.
       - To change the associated TimeDependent2Operator. */
   virtual void Init(TimeDependent2Operator &f)
   {
      this->f = &f;
   }

   /** @brief Perform a time step from time @a t [in] to time @a t [out] based
       on the requested step size @a dt [in]. */
   /** @param[in,out] x    Approximate solution.
       @param[in,out] dxdt Approximate rate.
       @param[in,out] t    Time associated with the approximate solution @a x.
       @param[in,out] dt   Time step size.

       The following rules describe the common behavior of the method:
       - The input @a x [in] is the approximate solution for the input time
         @a t [in].
       - The input @a dxdt [in] is the approximate rate for the input time
         @a t [in].
       - The input @a dt [in] is the desired time step size, defining the desired
         target time: t [target] = @a t [in] + @a dt [in].
       - The output @a x [out] is the approximate solution for the output time
         @a t [out].
       - The output @a dxdt [out] is the approximate rate for the output time
         @a t [out].
       - The output @a dt [out] is the last time step taken by the method which
         may be smaller or larger than the input @a dt [in] value, e.g. because
         of time step control.
       - The method may perform more than one time step internally; in this case
         @a dt [out] is the last internal time step size.
       - The output value of @a t [out] may be smaller or larger than
         t [target], however, it is not smaller than @a t [in] + @a dt [out], if
         at least one internal time step was performed.
       - The values @a x [out] and @a dxdt [out] may be obtained by interpolation
         using internally stored data.
       - In some cases, the contents of @a x [in] or @a dxdt [in] may not be used,
         e.g. when @a x [out] or @a dxdt [out] from a previous Step() call was
         obtained by interpolation.
       - In consecutive calls to this method, the output @a t [out] of one
         Step() call has to be the same as the input @a t [in] to the next
         Step() call.
       - If the previous rule has to be broken, e.g. to restart a time stepping
         sequence, then the ODE solver must be re-initialized by calling Init()
         between the two Step() calls. */
   virtual void Step(Vector &x, Vector &dxdt, double &t, double &dt) = 0;

   /// Perform time integration from time @a t [in] to time @a tf [in].
   /** @param[in,out] x    Approximate solution.
       @param[in,out] dxdt Approximate solution.
       @param[in,out] t    Time associated with the approximate solution @a x.
       @param[in,out] dt   Time step size.
       @param[in]     tf   Requested final time.

       The default implementation makes consecutive calls to Step() until
       reaching @a tf.
       The following rules describe the common behavior of the method:
       - The input @a x [in] is the approximate solution for the input time
         @a t [in].
       - The input @a dxdt [in] is the approximate rate for the input time
         @a t [in].
       - The input @a dt [in] is the initial time step size.
       - The output @a dt [out] is the last time step taken by the method which
         may be smaller or larger than the input @a dt [in] value, e.g. because
         of time step control.
       - The output value of @a t [out] is not smaller than @a tf [in]. */
   virtual void Run(Vector &x, Vector &dxdt, double &t, double &dt, double tf)
   {
      while (t < tf) { Step(x, dxdt, t, dt); }
   }

   virtual ~ODE2Solver() { }
};

/// The classical newmark method.
/// Newmark, N. M. (1959) A method of computation for structural dynamics.
/// Journal of Engineering Mechanics, ASCE, 85 (EM3) 67-94.
class NewmarkSolver : public ODE2Solver
{
private:
   Vector d2xdt2;

   double beta, gamma;
   bool first;

public:
   NewmarkSolver(double beta_ = 0.25, double gamma_ = 0.5) { beta = beta_; gamma = gamma_; };

   virtual void PrintProperties(std::ostream &out = mfem::out);

   virtual void Init(TimeDependent2Operator &_f);

   virtual void Step(Vector &x, Vector &dxdt, double &t, double &dt);
};

class AverageAccelerationSolver : public NewmarkSolver
{
public:
   AverageAccelerationSolver() : NewmarkSolver(0.25, 0.5) { };
};

class LinearAccelerationSolver : public NewmarkSolver
{
public:
   LinearAccelerationSolver() : NewmarkSolver(1.0/6.0, 0.5) { };
};

class CentralDifferenceSolver : public NewmarkSolver
{
public:
   CentralDifferenceSolver() : NewmarkSolver(0.0, 0.5) { };
};

class FoxGoodwinSolver : public NewmarkSolver
{
public:
   FoxGoodwinSolver() : NewmarkSolver(1.0/12.0, 0.5) { };
};


/// Generalized-alpha ODE solver
/// A Time Integration Algorithm for Structural Dynamics With Improved Numerical Dissipation: The Generalized-α Method
/// J.Chung and G.M. Hulbert,  J. Appl. Mech 60(2), 371-375, 1993
class GeneralizedAlpha2Solver : public ODE2Solver
{
protected:
   Vector k,d2xdt2;
   double alpha_f, alpha_m, beta, gamma;
   bool first;

public:

   GeneralizedAlpha2Solver(double rho_inf = 1.0) 
   {
      rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
      rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

      alpha_m = (2.0 - rho_inf)/(1.0 + rho_inf);
      alpha_f = 1.0/(1.0 + rho_inf);
      beta    = 0.25*pow(1.0 + alpha_m - alpha_f,2);
      gamma   = 0.5 + alpha_m - alpha_f;
   };

   virtual void PrintProperties(std::ostream &out = mfem::out);

   virtual void Init(TimeDependent2Operator &_f);

   virtual void Step(Vector &x, Vector &dxdt, double &t, double &dt);
};

/// HHT-alpha ODE solver
class HHTAlphaSolver : public GeneralizedAlpha2Solver
{
public:

   HHTAlphaSolver(double rho_inf = 1.0) 
   {
      rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
      rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

      alpha_m = 1.0;
      alpha_f = 2.0*rho_inf/(1.0+rho_inf);
      beta    = 0.25*pow(1.0 + alpha_m - alpha_f,2);
      gamma   = 0.5 + alpha_m - alpha_f;
   };

};


/// WBZ-alpha ODE solver
class WBZAlphaSolver : public GeneralizedAlpha2Solver
{
public:

   WBZAlphaSolver(double rho_inf = 1.0) 
   {
      rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
      rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

      alpha_f = 1.0;
      alpha_m = 2.0/(1.0+rho_inf);
      beta    = 0.25*pow(1.0 + alpha_m - alpha_f,2);
      gamma   = 0.5 + alpha_m - alpha_f;
   };

};

/// The classical newmark method.
/// Newmark, N. M. (1959) A method of computation for structural dynamics.
/// Journal of Engineering Mechanics, ASCE, 85 (EM3) 67-94.
class Newmark2Solver : public GeneralizedAlpha2Solver
{
public:
   Newmark2Solver(double beta_ = 0.25, double gamma_ = 0.5)
   {
      alpha_f = 1.0;
      alpha_m = 1.0;
      beta    = beta_;
      gamma   = gamma_;
   };
};


}

#endif
