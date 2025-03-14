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

#ifndef MFEM_MMA
#define MFEM_MMA

#include <iostream>
#include "../config/config.hpp"
#include "vector.hpp"
#include "../general/communication.hpp"
#include "solvers.hpp"

#ifdef MFEM_USE_MPI
#include <mpi.h>
#endif

namespace mfem
{

class MMA
{
public:

   /// Serial constructor:
   /// nVar - number of design parameters;
   /// nCon - number of constraints;
   /// xval[nVar] - initial parameter values
   MMA(int nVar, int nCon, real_t *xval);

   /// Serial constructor:
   /// nVar - number of design parameters;
   /// nCon - number of constraints;
   /// xval[nVar] - initial parameter values
   MMA(const int nVar, int nCon, Vector & xval);

   /// Serial constructor for unconstraint problem:
   /// nVar - number of design parameters;
   /// xval[nVar] - initial parameter values
   MMA(int nVar, Vector & xval);

#ifdef MFEM_USE_MPI
   /// Parallel constructor:
   /// comm_ - communicator
   /// nVar - number of design parameters;
   /// nCon - number of constraints;
   /// xval[nVar] - initial parameter values
   MMA(MPI_Comm comm_, int nVar, int nCon, real_t *xval);

   /// Parallel constructor:
   /// comm_ - communicator
   /// nVar - number of design parameters;
   /// nCon - number of constraints;
   /// xval[nVar] - initial parameter values
   MMA(MPI_Comm comm_, const int & nVar, const int & nCon, const Vector & xval);
#endif

   /// Destructor
   ~MMA();


   /// Update the optimization parameters
   /// iter - current iteration number
   /// dfdx[nVar] - gradients of the objective
   /// gx[nCon] - values of the constraints
   /// dgdx[nCon*nVar] - gradients of the constraints
   /// xxmin[nVar] - lower bounds
   /// xxmax[nVar] - upper bounds
   /// xval[nVar] - (input: current optimization parameters)
   /// xval[nVar] - (output: updated optimization parameters)
   void Update(int iter, const Vector& dfdx,
               const Vector& gx, const Vector& dgdx,
               const Vector& xmin, const Vector& xmax,
               Vector& xval);

   /// Update the optimization parameters
   /// iter - current iteration number
   /// dfdx[nVar] - gradients of the objective
   /// xxmin[nVar] - lower bounds
   /// xxmax[nVar] - upper bounds
   /// xval[nVar] - (input: current optimization parameters)
   /// xval[nVar] - (output: updated optimization parameters)
   void Update(int iter, const Vector& dfdx,
               const Vector& xmin, const Vector& xmax,
               Vector& xval);

   /// Dump the internal state into a file
   /// xval[nVar] - current optimization parameters
   /// iter - current interation
   /// fname, fpath - file name and file path
   void WriteState(real_t* xval, int iter,
                   std::string fname,
                   std::string fpath = "./");

   /// Load the internal state
   void LoadState(real_t* xval, int iter,
                  std::string fname,
                  std::string fpath = "./");

protected:
   // Local vectors
   real_t *a, *b, *c, *d;
   real_t a0, machineEpsilon, epsimin;
   real_t z, zet;
   int nCon, nVar;

   // Global: Asymptotes, bounds, objective approx., constraint approx.
   real_t *low, *upp;
   real_t *x, *y, *xsi, *eta, *lam, *mu, *s;


private:

   // MMA-specific
   real_t asyinit, asyincr, asydecr;
   real_t xmamieps, lowmin, lowmax, uppmin, uppmax, zz;
   real_t *factor;

   /// values from the previous two iterations
   real_t *xo1, *xo2;

   /// KKT norm
   real_t kktnorm;

   /// intialization state
   bool isInitialized = false;

#ifdef MFEM_USE_MPI
   MPI_Comm comm;
#endif

   /// Allocate the memory for MMA
   void AllocData(int nVar, int nCon);

   /// Free the memory for MMA
   void FreeData();

   /// Initialize data
   void  InitData(real_t *xval);

   /// Update the optimization parameters
   /// iter - current iteration number
   /// dfdx[nVar] - gradients of the objective
   /// gx[nCon] - values of the constraints
   /// dgdx[nCon*nVar] - gradients of the constraints
   /// xxmin[nVar] - lower bounds
   /// xxmax[nVar] - upper bounds
   /// xval[nVar] - (input: current optimization parameters)
   /// xval[nVar] - (output: updated optimization parameters)
   void Update(int iter, const real_t* dfdx, const real_t* gx,
               const real_t* dgdx,
               const real_t* xxmin,const real_t* xxmax,
               real_t* xval);

   /// Subproblem base class
   class MMASubBase
   {
   public:
      /// Constructor
      MMASubBase(MMA* mma) {mma_ptr=mma;}

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
      MMA* mma_ptr;
   };

   MMASubBase* mSubProblem;

   friend class MMASubParallel;

   class MMASubParallel:public MMASubBase
   {
   public:

      /// Constructor
      MMASubParallel(MMA* mma, int nVar, int nCon):MMASubBase(mma)
      {
         AllocSubData(nVar,nCon);

         nVar_global = nVar;
#ifdef MFEM_USE_MPI
         MPI_Allreduce(&nVar, &nVar_global, 1, MPI_INT, MPI_SUM, mma->comm);
#endif
      }

      /// Destructor
      virtual
      ~MMASubParallel()
      {
         FreeSubData();
      }

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

      real_t *sum1, *ux1, *xl1, *plam, *qlam, *gvec, *residu, *GG, *delx, *dely,
             *dellam,
             *dellamyi, *diagx, *diagy, *diaglamyi, *bb, *bb1, *Alam, *AA, *AA1,
             *dlam, *dx, *dy, *dxsi, *deta, *dmu, *Axx, *axz, *ds, *xx, *dxx, *stepxx,
             *stepalfa, *stepbeta, *xold, *yold,
             *lamold, *xsiold, *etaold, *muold, *sold, *p0, *q0, *P, *Q, *alfa, *beta,
             *xmami, *b;

      // parallel helper variables
      real_t global_max = 0.0;
      real_t global_norm = 0.0;
      real_t stmxx_global = 0.0;
      real_t stmalfa_global = 0.0;
      real_t stmbeta_global = 0.0;

      real_t *b_local, *gvec_local, *Alam_local, *sum_local, *sum_global;

      /// Allocate the memory for the subproblem
      void AllocSubData(int nVar, int nCon);

      /// Free the memeory for the subproblem
      void FreeSubData();
   };
};

class MMAOptimizer:public OptimizationSolver
{
public:
    MMAOptimizer()
    {
#ifdef MFEM_USE_MPI
        comm=MPI_COMM_SELF;
#endif
    }

#ifdef MFEM_USE_MPI
    MMAOptimizer(MPI_Comm comm_)
    {
        comm=comm_;
    }
#endif

    virtual ~MMAOptimizer()
    {
    }

    virtual
    void SetOptimizationProblem (const OptimizationProblem &prob)
    {
      //   problem=&prob;
      //   height = width = problem->input_size;
      //   dfdx.SetSize(height);
      //   xmin=problem->GetInequalityVec_Lo();
      //   xmax=problem->GetInequalityVec_Hi();

      //   gx.SetSize(problem->GetNumConstraints());
      //   dgdx.SetSize((problem->GetNumConstraints())*height);
    }

    virtual
    void Mult (const Vector &xt, Vector &x) const
    {

        // we do not use the values of the assumptotes from the previous iterations
        MMA* mma;
        x=xt;
        //allocate the solver
#ifdef MFEM_USE_MPI
        mma=new MMA(comm,height,problem->GetNumConstraints(),x);
#else
        mma=new MMA(height,problem->GetNumConstraints(),x);
#endif

        for(int i=0;i<mfem::IterativeSolver::max_iter;i++)
        {


            mma->Update(i,dfdx,gx,dgdx,xmin,xmax,x);
        }

        delete mma;
    }


private:
#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif
    Vector gx;
    Vector dfdx;
    Vector dgdx;
    Vector xmin;
    Vector xmax;

    const OptimizationProblem* problem;

};



} // mfem namespace

#endif // MFEM_MMA
