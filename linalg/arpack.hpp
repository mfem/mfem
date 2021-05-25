// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ARPACK
#define MFEM_ARPACK

#include "../config/config.hpp"

#ifdef MFEM_USE_ARPACK

#include <string>

using namespace std;

#ifdef MFEM_USE_MPI
#include <mpi.h>
#include "hypre.hpp"
#endif

#include "operator.hpp"

#define DSAUPD  dsaupd_
#define DSEUPD  dseupd_

#ifdef MFEM_USE_MPI
#define PDSAUPD pdsaupd_
#define PDSEUPD pdseupd_
#endif

extern "C" void DSAUPD(int *ido,char *bmat, int *n,
                       char *which, int *nev,double *tol,double *resid,
                       int *ncv,double *v, int *ldv,
                       int *iparam, int *ipntr,
                       double *workd, double *workl, int *lworkl, int *info);

extern "C" void DSEUPD(int *, char *,int *, double *,
                       double *,int *, double *,char *, int *, char *,
                       int *,double *,double *,int *, double *,
                       int *, int  *,int *, double *,
                       double *,int *, int *);

#ifdef MFEM_USE_MPI

extern "C" void PDSAUPD(int *comm, int *ido,char *bmat, int *n,
                        char *which, int *nev,double *tol,double *resid,
                        int *ncv,double *v, int *ldv,
                        int *iparam, int *ipntr,
                        double *workd, double *workl, int *lworkl, int *info);

extern "C" void PDSEUPD(int *comm, int *, char *,int *, double *,
                        double *,int *, double *,char *, int *, char *,
                        int *,double *,double *,int *, double *,
                        int *, int  *,int *, double *,
                        double *,int *, int *);

#endif

extern "C" {
   void arpackgetcommdbg_(int *,int *,int *);
   void arpacksetcommdbg_(int *,int *,int *);
   void arpacksymdbg_(int *,int *,int *,int *,int *,int *,int *);
   void arpacknonsymdbg_(int *,int *,int *,int *,int *,int *,int *);
   void arpackcmplxdbg_(int *,int *,int *,int *,int *,int *,int *);
}

namespace mfem
{

class ArPackSym : public Eigensolver
{
public:

   ArPackSym();
   virtual ~ArPackSym();

   /** ARPACK modes are described in section 3.5 of the ARPACK manual.
       Mode 1: regular mode to solve A x = lambda x
               No solver and no mass matrix are needed.
       Mode 2: regular inverse mode to solve A x = lambda M x
               Both A and M are needed and the solver should compute M^{-1}.
       Mode 3: shift-invert mode to solve either A x = lambda x
               or A x = lambda M x
               Mass matrix is optional.  The solver should compute
               (A-sigma I)^{-1} or (A-sigma M)^{-1}.  The shift parameter,
               sigma, also needs to be set with SetShift().
       Mode 4: Buckling mode to solve K x = lambda K_G x
               K is set using SetMassMatrix(), K_G is set using SetOperator(),
          and the solver should compute (K-sigma K_G)^{-1}.  The shift
               parameter, sigma, also needs to be set with SetShift().
       Mode 5: Cayley mode to solve A x = lambda M x
               Both A and M are needed and the solver should compute
               (A - sigma M)^{-1}.  The shift parameter, sigma, also needs
               to be set with SetShift().
    */
   void SetMode(int mode);

   inline void SetTol(double tol)         {      tol_ = tol;      }
   inline void SetMaxIter(int max_iter)   { max_iter_ = max_iter; }
   inline void SetPrintLevel(int logging) {  logging_ = logging;  }
   inline void SetShift(double sigma)     {    sigma_ = sigma;    }
   inline void SetNumModes(int num_eigs)  {      nev_ = num_eigs; }

   virtual void SetSolver(Solver & solver);
   virtual void SetOperator(Operator & A);
   virtual void SetMassMatrix(Operator & M);

   void Solve();

   /// Collect the converged eigenvalues
   virtual void GetEigenvalues(Array<double> & eigenvalues);

   /// Extract a single eigenvector
   virtual Vector & GetEigenvector(unsigned int i);

   /// Transfer ownership of the converged eigenvectors
   Vector ** StealEigenvectors();

protected:

   int myid_;       // Index of this processor
   int max_iter_;
   int logging_;

   // The following variables are for ARPACK
   int nloc_;       // number of items stored locally
   int nev_;        // number of requested eigenvalues
   int ncv_;        // number of ritz vectors
   int rvec_;       // boolean to return eigenvectors as well
   int mode_;       // 1 = standard, 2 = generalized, 3 = shift invert,
   // 4 = buckling, 5 = Cayley
   int lworkl_;     // length of lworkl_ work array
   int iparam_[12]; // arpack parameters
   int ipntr_[12];  // arpack pointers

   char bmat_;      // I for standard problem, G for generalized
   char which_[3];  // spectrum portion: LA, SA, LM, SM, BE
   char hwmny_;     // DSEUPD: A for all eigenvalues, S for some

   double tol_;     // relative accuracy bound for Ritz values
   double sigma_;   // eigenvalue shift parameter

   int    * select_;// workspace used during eigenvalue computation
   double * dv_;    // Ritz values
   double * v_;     // ncv Lanczos basis vectors
   double * resid_; // residual vector
   double * workd_; // work array for 3 vectors used in Arnoldi iteration
   double * workl_; // work array

   // Operators and Vectors needed outside of ARPACK
   Solver   * solver_;
   Operator * A_;
   Operator * B_;

   Vector * w_;
   Vector * x_;
   Vector * y_;
   Vector * z_;

   Vector ** eigenvectors_;

   string solverName_;

   void reverseComm();

   int reverseCommMode1();
   int reverseCommMode2();
   int reverseCommMode3();
   int reverseCommMode4();
   int reverseCommMode5();

   virtual void prepareEigenvectors();

   void printErrors(const int & info, const int iparam[],
                    const char & bmat, const int & n,
                    const char which[],
                    const int & nev, const int & ncv,
                    const int & lworkl );

private:

   virtual int computeNlocf() { return nloc_; }
   virtual int computeIter(int & ido);
   virtual int computeEigs();

};

#ifdef MFEM_USE_MPI

class ParArPackSym : public ArPackSym
{
public:
   ParArPackSym(MPI_Comm comm);
   virtual ~ParArPackSym() {}

   void SetOperator(Operator & A);
   void SetMassMatrix(Operator & M);

   /// Collect the converged eigenvalues
   void GetEigenvalues(Array<double> & eigenvalues);

   /// Extract a single eigenvector
   Vector & GetEigenvector(unsigned int i);

   /// Transfer ownership of the converged eigenvectors
   // HypreParVector ** StealEigenvectors();
   Vector ** StealEigenvectors();

protected:

   void prepareEigenvectors();

private:

   MPI_Comm comm_;
   MPI_Fint commf_; // Fortran style MPI communicator
   int numProcs_;   // Number of processors

   HYPRE_Int * part_; // parallel partitioning for eigenvectors

   int computeNlocf();
   int computeIter(int & ido);
   int computeEigs();

};

#endif // MFEM_USE_MPI

};

#endif // MFEM_USE_ARPACK

#endif // MFEM_ARPACK