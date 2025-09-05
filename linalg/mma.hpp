// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MMA
#define MFEM_MMA

#include "../config/config.hpp"

#include <memory>

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{
// forward declaration
class Vector;

/** \brief MMA (Method of Moving Asymptotes) solves a nonlinear optimization
 *         problem involving an objective function, inequality constraints,
 *         and variable bounds.
 *
 *  \details
 *  This class finds ${\bf x} \in R^n$ that solves the following nonlinear
 *  program:
 *   $$
 *    \begin{array}{ll}
 *    \min_{{\bf x} \in R^n} & F({\bf x})\\
 *     \textrm{subject to}   & C({\bf x})_i \leq 0,\quad
 *                             \textrm{for all}\quad i = 1,\ldots m\\
 *                           & {\bf x}_{\textrm{lo}} \leq {\bf x} \leq
 *                             {\bf x}_{\textrm{hi}}.
 *    \end{array}
 *   $$
 *   Here $F : R^n \to R$ is the objective function, and
 *   $C : R^n \to R^m$ is a set of $m$ inequality constraints. The
 *   variable bounds are sometimes called box constraints. By
 *   convention, the routine seeks ${\bf x}$ that minimizes the
 *   objective function, $F$. Maximization problems should be
 *   reformulated as a minimization of $-F$.
 *
 *    The objective functions are replaced by convex functions
 *    chosen based on gradient information, and solved using a dual method.
 *    The unique optimal solution of this subproblem is returned as the next
 *    iteration point. Optimality is determined by the KKT conditions.
 *
 *  The "Update" function in MMA advances the optimization and must be
 *  called in every optimization iteration. Current and previous iteration
 *  points construct the "moving asymptotes". The design variables,
 *  objective function, constraints are passed to an approximating
 *  subproblem. The design variables are updated and returned. Its
 *  implementation closely follows the original formulation of <a
 *  href="https://people.kth.se/~krille/mmagcmma.pdf">'Svanberg, K. (2007).
 *  MMA and GCMMA-two methods for nonlinear optimization. vol, 1, 1-15.'</a>
 *
 *  When used in parallel, all Vectors are assumed to be true dof vectors,
 *  and the operators are expected to be defined for tdof vectors.
 * */

class MMA
{
public:
   /**
    * \brief Serial constructor
    * \param nVar   total number of design parameters
    * \param nCon   number of inequality constraints (i.e., $C$)
    * \param xval   initial values for design parameters (a pointer
    *               to \p nVar doubles). Caller retains ownership of
    *               this pointer/data.
    * \param iterationNumber the starting iteration number
    */
   MMA(int nVar, int nCon, real_t *xval, int iterationNumber = 0);

   /**
    * \brief Serial constructor
    * \param nVar   total number of design parameters
    * \param nCon   number of inequality constraints (i.e., $C$)
    * \param xval   initial values for design parameters (size should
    *               be \p nVar). Caller retains ownership of
    *               this Vector.
    * \param iterationNumber the starting iteration number
    */
   MMA(const int nVar, int nCon, Vector & xval, int iterationNumber = 0);

#ifdef MFEM_USE_MPI
   /**
    * \brief Parallel constructor
    * \param comm_  the MPI communicator participating in the NLP solve
    * \param nVar   number of design parameters on this MPI rank
    * \param nCon   total number of inequality constraints (i.e., $C$).
    *               Every MPI rank provides the same value here.
    * \param xval   initial values for design parameters on this MPI rank
    *               (a pointer to \p nVar doubles). Caller retains ownership
    *               of this pointer/data.
    * \param iterationNumber the starting iteration number. All MPI ranks
    *               should pass in the same value here.
    *
    * \details
    * Each MPI rank has a subset of the total design variable vector, and
    * calls for that MPI rank always address its subset of the design
    * variable vector and gradients with respect to its subset of the design
    * variable vector.
    *
    * If you wanted to determine the global number of design variables, it
    * would be determined as follows:
    * \code{.cpp}
    * int globalDesignVars;
    * MPI_Allreduce(&nVar, &globalDesignVars, 1, MPI_INT, MPI_SUM, comm_);
    * \endcode
    */
   MMA(MPI_Comm comm_, int nVar, int nCon, real_t *xval,
       int iterationNumber = 0);
   /**
    * \brief Parallel constructor
    * \param comm_  the MPI communicator participating in the NLP solve
    * \param nVar   number of design parameters on this MPI rank
    * \param nCon   total number of inequality constraints (i.e., $C$).
    *               Every MPI rank provides the same value here.
    * \param xval   initial values for design parameters (size should
    *               be \p nVar). Caller retains ownership of
    *               this Vector.
    * \param iterationNumber the starting iteration number. All MPI ranks
    *               should pass in the same value here.
    *
    * \details
    * Each MPI rank has a subset of the total design variable vector, and
    * calls for that MPI rank always address its subset of the design
    * variable vector and gradients with respect to its subset of the design
    * variable vector.
    *
    * If you wanted to determine the global number of design variables, it
    * would be determined as follows:
    * \code{.cpp}
    * int globalDesignVars;
    * MPI_Allreduce(&nVar, &globalDesignVars, 1, MPI_INT, MPI_SUM, comm_);
    * \endcode
    */
   MMA(MPI_Comm comm_, const int nVar, const int nCon, const Vector & xval,
       int iterationNumber = 0);
#endif

   /// Destructor
   ~MMA();

   /**
    * \brief Update the optimization parameters for a constrained
    *        nonlinear program
    * \param dfdx    vector of size nVar holding the gradients of the
    *                objective function with respect to
    *                the design variables,
    *                $\frac{\partial F}{\partial {\bf x}_i}$
    *                for each variable on this rank.
    * \param gx      vector of size nCon holding the values of the
    *                inequality constraints. Every MPI rank should
    *                pass in the same values here.
    * \param dgdx    vector of size $\textrm{nCon}\cdot\textrm{nVar}$
    *                holding the gradients of the constraints in
    *                row-major order. For example, {dg0dx0, dg0dx1, ...,}
    *                {dg1dx0, dg1dx1, ..., }, ...
    * \param xmin    vector of size nVar holding the lower bounds on
    *                the design values. \p xmin and \p xmax are
    *                the box constraints.
    * \param xmax    vector of size nVar holding the upper bounds on
    *                the design values.  \p xmin and \p xmax are
    *                the box constraints.
    * \param xval    vector of size nVar. On entry, this holds the
    *                value of the design variables where the objective,
    *                constraints, and their gradients were evaluated.
    *                On exit, this holds the result of the MMA iteration,
    *                the next design variable value to use.
    *
    * \details
    * The caller retains ownership of all Vectors passed into this method.
    */
   void Update(const Vector& dfdx,
               const Vector& gx, const Vector& dgdx,
               const Vector& xmin, const Vector& xmax,
               Vector& xval);

   /**
    * \brief Update the optimization parameters for an unconstrained
    *        nonlinear program
    * \param dfdx    vector of size nVar holding the gradients of the
    *                objective function with respect to
    *                the design variables,
    *                $\frac{\partial F}{\partial {\bf x}_i}$
    *                for each variable on this rank.
    * \param xmin    vector of size nVar holding the lower bounds on
    *                the design values. \p xmin and \p xmax are
    *                the box constraints.
    * \param xmax    vector of size nVar holding the upper bounds on
    *                the design values.  \p xmin and \p xmax are
    *                the box constraints.
    * \param xval    vector of size nVar. On entry, this holds the
    *                value of the design variables where the objective,
    *                constraints, and their gradients were evaluated.
    *                On exit, this holds the result of the MMA iteration,
    *                the next design variable value to use.
    *
    * \details
    * The caller retains ownership of all Vectors passed into this method.
    * This should be used when the number of inequality constraints is zero.
    */
   void Update( const Vector& dfdx,
                const Vector& xmin, const Vector& xmax,
                Vector& xval);

   /**
    * \brief Change the iteration number
    * \param iterationNumber    the new iteration number
    */
   void SetIteration( int iterationNumber ) { iter = iterationNumber; };

   /// Return the current iteration number
   int GetIteration() const { return iter; };

   /**
    * \brief change the print level
    * \param print_lvl    the new print level
    */
   void SetPrintLevel(int print_lvl) { print_level = print_lvl; }

protected:
   // Local vectors
   ::std::unique_ptr<real_t[]> a, b, c, d;
   real_t a0, machineEpsilon, epsimin;
   real_t z, zet;
   int nCon, nVar;

   // counter for Update() calls
   int iter = 0;

   // print level: 1 = none, 2 = warnings
   int print_level = 1;

   // Global: Asymptotes, bounds, objective approx., constraint approx.
   ::std::unique_ptr<real_t[]> low, upp;
   ::std::unique_ptr<real_t[]> x, y, xsi, eta, lam, mu, s;

private:
   // MMA-specific
   real_t asyinit, asyincr, asydecr;
   real_t xmamieps, lowmin, lowmax, uppmin, uppmax, zz;
   ::std::unique_ptr<real_t[]> factor;

   /// values from the previous two iterations
   ::std::unique_ptr<real_t[]> xo1, xo2;

   /// KKT norm
   real_t kktnorm;

   /// initialization state
   bool isInitialized = false;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

   /// Allocate the memory for MMA
   void AllocData(int nVar, int nCon);

   /// Initialize data
   void  InitData(real_t *xval);

   /// Update the optimization parameters
   /// dfdx[nVar] - gradients of the objective
   /// gx[nCon] - values of the constraints
   /// dgdx[nCon*nVar] - gradients of the constraints
   /// xxmin[nVar] - lower bounds
   /// xxmax[nVar] - upper bounds
   /// xval[nVar] - (input: current optimization parameters)
   /// xval[nVar] - (output: updated optimization parameters)
   void Update(const real_t* dfdx, const real_t* gx,
               const real_t* dgdx,
               const real_t* xxmin,const real_t* xxmax,
               real_t* xval);

   /// Subproblem base class
   class MMASubBase
   {
   public:
      /// Constructor
      MMASubBase(MMA& mma) :mma_ref(mma) {}

      /// Destructor
      virtual ~MMASubBase() {}

      /// Update the optimization parameters
      virtual
      void Update(const real_t* dfdx,
                  const real_t* gx,
                  const real_t* dgdx,
                  const real_t* xmin,
                  const real_t* xmax,
                  const real_t* xval)=0;

   protected:
      MMA& mma_ref;
   };

   ::std::unique_ptr<MMASubBase> mSubProblem;

   friend class MMASubSvanberg;

   class MMASubSvanberg:public MMASubBase
   {
   public:
      /// Constructor
      MMASubSvanberg(MMA& mma, int nVar, int nCon):MMASubBase(mma)
      {
         AllocSubData(nVar,nCon);

         nVar_global = nVar;
#ifdef MFEM_USE_MPI
         MPI_Allreduce(&nVar, &nVar_global, 1, MPI_INT, MPI_SUM, mma.comm);
#endif
      }

      /// Destructor
      virtual
      ~MMASubSvanberg() = default;

      /// Update the optimization parameters
      virtual
      void Update(const real_t* dfdx,
                  const real_t* gx,
                  const real_t* dgdx,
                  const real_t* xmin,
                  const real_t* xmax,
                  const real_t* xval);

   private:
      int ittt, itto, itera, nVar_global;

      real_t epsi, delz, dz, dzet, stmxx, stmalfa, stmbeta,
             sum, stminv, steg, zold, zetold,
             residunorm, residumax, resinew, raa0, albefa, move, xmamieps;

      ::std::unique_ptr<real_t[]> sum1, ux1, xl1, plam, qlam, gvec, residu, GG, delx,
      dely,
      dellam,
      dellamyi, diagx, diagy, diaglamyi, bb, bb1, Alam, AA, AA1,
      dlam, dx, dy, dxsi, deta, dmu, Axx, axz, ds, xx, dxx,
      stepxx, stepalfa, stepbeta, xold, yold,
      lamold, xsiold, etaold, muold, sold, p0, q0, P, Q, alfa,
      beta, xmami, b;

      // parallel helper variables
      real_t global_max = 0.0;
      real_t global_norm = 0.0;
      real_t stmxx_global = 0.0;
      real_t stmalfa_global = 0.0;
      real_t stmbeta_global = 0.0;

      ::std::unique_ptr<real_t[]> b_local, gvec_local, Alam_local, sum_local,
      sum_global;

      /// Allocate the memory for the subproblem
      void AllocSubData(int nVar, int nCon);

   };
};

} // mfem namespace

#endif // MFEM_MMA
