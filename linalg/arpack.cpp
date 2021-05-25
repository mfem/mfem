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

#include "../config/config.hpp"

#ifdef MFEM_USE_ARPACK

#include <typeinfo>

#include "linalg.hpp"
#include "arpack.hpp"

using namespace std;

namespace mfem
{

ArPackSym::ArPackSym()
   : myid_(0),
     nloc_(-1),
     ncv_(20),
     rvec_(0),
     mode_(1),
     lworkl_(-1),
     bmat_('I'),
     hwmny_('A'),
     sigma_(0.0),
     select_(NULL),
     dv_(NULL),
     v_(NULL),
     resid_(NULL),
     workd_(NULL),
     workl_(NULL),
     solver_(NULL),
     A_(NULL),
     B_(NULL),
     w_(NULL),
     x_(NULL),
     y_(NULL),
     z_(NULL),
     eigenvectors_(NULL),
     solverName_("DSAUPD")
{
}

ArPackSym::~ArPackSym()
{
   delete w_;
   delete x_;
   delete y_;
   delete z_;

   delete [] dv_;
   delete [] v_;

   delete [] resid_;
   delete [] workd_;
   delete [] workl_;
   delete [] select_;
}

void
ArPackSym::SetMode(int mode)
{
   mode_ = mode;

   // Set the default spectrum range for this mode
   switch (mode_)
   {
      case 1:
      case 2:
         strncpy(which_,"SM",2);
         break;
      case 3:
      case 4:
      case 5:
         strncpy(which_,"LM",2);
         break;
   }
}

void
ArPackSym::SetSolver(Solver & solver)
{
   solver_ = &solver;
}

void
ArPackSym::SetOperator(Operator & A)
{
   A_ = &A;

   if ( x_ == NULL )
   {
      w_ = new Vector(NULL, A_->Width());
      x_ = new Vector(NULL, A_->Width());
      y_ = new Vector(NULL, A_->Width());
      z_ = new Vector(A_->Width());
   }
}

void
ArPackSym::SetMassMatrix(Operator & B)
{
   B_ = &B;

   bmat_ = 'G'; // If the user sets a B, we can assume B != I

   if ( x_ == NULL )
   {
      w_ = new Vector(NULL, B_->Width());
      x_ = new Vector(NULL, B_->Width());
      y_ = new Vector(NULL, B_->Width());
      z_ = new Vector(B_->Width());
   }
}

int
ArPackSym::reverseCommMode1()
{
   ////////////////////////////////////////////////////////////////////////////
   //
   //   Standard Eigenvalue Mode
   //
   ////////////////////////////////////////////////////////////////////////////

   // The following variables are for ARPACK
   int ido = 0;             // reverse communication what to do flag
   int info = 0;            // arpack info

   if ( myid_ == 0 && logging_ >= 1 )
   {
      mfem::out << "  Starting mode 1 main loop." << endl << flush;
   }

   // the main loop
   while (ido != 99 && info >= 0)
   {
      // call the main arpack routine
      info = this->computeIter(ido);
      if (info < 0)
      {
         printErrors(info,iparam_,bmat_,nloc_,which_,nev_,ncv_,lworkl_);
      }

      // now we have to do something. examine the ido flag
      if (ido == -1 || ido == 1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing y = A x"
                      << endl << flush;
         }

         // we have to compute y = A x;

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // y = A * x;
         A_->Mult(*x_,*y_);
      }
      else if (ido == 2)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido << ", computing y = x"
                      << endl << flush;
         }

         // we have to compute y = x.

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global vector copy
         // y = x;
         *y_ = *x_;
      }
   }
   return info;
}

int
ArPackSym::reverseCommMode2()
{
   ////////////////////////////////////////////////////////////////////////////
   //
   //   General Eigenvalue Mode
   //
   ////////////////////////////////////////////////////////////////////////////

   // The following variables are for ARPACK
   int ido = 0;             // reverse communication what to do flag
   int info = 0;            // arpack info

   if ( myid_ == 0 && logging_ >= 1 )
   {
      mfem::out << "  Starting mode 2 main loop." << endl << flush;
   }

   // the main loop
   while (ido != 99 && info >= 0)
   {
      // call the main arpack routine
      info = this->computeIter(ido);
      if (info < 0)
      {
         printErrors(info,iparam_,bmat_,nloc_,which_,nev_,ncv_,lworkl_);
      }

      // now we have to do something. examine the ido flag
      if (ido == -1 || ido == 1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = M^-1 y, where y = A v"
                      << endl << flush;
         }

         // we have to compute y = A v, then solve M w = y;

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply (b is y, x is v)
         // b = A * x;
         A_->Mult(*x_,*z_);

         // copy part of b to workd , workd[ipntr[0] ... n] = b **/
         *x_ = *z_;

         // now do the big solve M w = y; (w is x, y is b)
         solver_->Mult(*z_,*y_);
      }
      else if (ido == 2)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido << ", computing w = M v"
                      << endl << flush;
         }

         // we have to compute w = M v. (w is b, v is x)

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // b = M * x;
         B_->Mult(*x_,*y_);
      }
   }
   return info;
}

int
ArPackSym::reverseCommMode3()
{
   ////////////////////////////////////////////////////////////////////////////
   //
   //   Shift-Invert Eigenvalue Mode
   //
   ////////////////////////////////////////////////////////////////////////////

   // The following variables are for ARPACK
   int ido = 0;             // reverse communication what to do flag
   int info = 0;            // arpack info

   if ( myid_ == 0 && logging_ >= 1 )
   {
      mfem::out << "  Starting mode 3 main loop." << endl << flush;
   }

   // the main loop
   while (ido != 99 && info >= 0)
   {
      // call the main arpack routine
      info = this->computeIter(ido);
      if (info < 0)
      {
         printErrors(info,iparam_,bmat_,nloc_,which_,nev_,ncv_,lworkl_);
      }

      // now we have to do something. examine the ido flag
      if (ido == -1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = (A - sigma M)^-1 y, where y = M v"
                      << endl << flush;
         }

         // we have to compute y = M v, then solve (A - sigma M) w = y;

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply (b is y, x is v)
         // b = M * x;
         B_->Mult(*x_,*z_);

         // now do the big solve (A - sigma M) w = y; (w is x, y is b)

         solver_->Mult(*z_,*y_);
      }
      else if (ido == 1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = (A - sigma M)^-1 v"
                      << endl << flush;
         }

         // we have to solve (A - sigma M)w = v

         // first, extract from workd (b is v)
         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[2]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do the big solve (A - sigma M) w = v; (w is x, v is b)
         solver_->Mult(*x_,*y_);
      }
      else if (ido == 2)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido << ", computing w = M v"
                      << endl << flush;
         }

         // we have to compute w = M v. (w is b, v is x)

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // b = M * x;
         B_->Mult(*x_,*y_);
      }
   }
   return info;
}


int
ArPackSym::reverseCommMode4()
{
   ////////////////////////////////////////////////////////////////////////////
   //
   //   Buckling Eigenvalue Mode
   //
   ////////////////////////////////////////////////////////////////////////////

   // The following variables are for ARPACK
   int ido = 0;             // reverse communication what to do flag
   int info = 0;            // arpack info

   if ( myid_ == 0 && logging_ >= 1 )
   {
      mfem::out << "  Starting mode 4 main loop." << endl << flush;
   }

   // the main loop
   while (ido != 99 && info >= 0)
   {
      // call the main arpack routine
      info = this->computeIter(ido);
      if (info < 0)
      {
         printErrors(info,iparam_,bmat_,nloc_,which_,nev_,ncv_,lworkl_);
      }

      // now we have to do something. examine the ido flag
      if (ido == -1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = (K - sigma KG)^-1 y, where y = K v"
                      << endl << flush;
         }

         // we have to compute y = K v, then solve (K - sigma KG) w = y;

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // z = K * x;
         B_->Mult(*x_,*z_);

         // now do the big solve (K - sigma KG) z = y;

         solver_->Mult(*z_,*y_);
      }
      else if (ido == 1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = (K - sigma KG)^-1 v"
                      << endl << flush;
         }

         // we have to solve (K - sigma KG)w = v

         // first, extract from workd
         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[2]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do the big solve (K - sigma KG) w = v;
         solver_->Mult(*x_,*y_);
      }
      else if (ido == 2)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido << ", computing w = K v"
                      << endl << flush;
         }

         // we have to compute y = K x.

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // y = K * x;
         B_->Mult(*x_,*y_);
      }
   }
   return info;
}

int
ArPackSym::reverseCommMode5()
{
   ////////////////////////////////////////////////////////////////////////////
   //
   //   Cayley transformed Eigenvalue Mode
   //
   ////////////////////////////////////////////////////////////////////////////

   // The following variables are for ARPACK
   int ido = 0;             // reverse communication what to do flag
   int info = 0;            // arpack info

   if ( myid_ == 0 && logging_ >= 1 )
   {
      mfem::out << "  Starting mode 5 main loop." << endl << flush;
   }

   // the main loop
   while (ido != 99 && info >= 0)
   {
      // call the main arpack routine
      info = this->computeIter(ido);
      if (info < 0)
      {
         printErrors(info,iparam_,bmat_,nloc_,which_,nev_,ncv_,lworkl_);
      }

      // now we have to do something. examine the ido flag
      if (ido == -1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = (A - sigma M)^-1 y, "
                      << "where y = (A + sigma M) v"
                      << endl << flush;
         }

         // we have to compute y = (A + sigma M) v,
         // then solve (A - sigma M) w = y;

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // y = M * x;
         B_->Mult(*x_,*y_); // use y as a temporary vector

         // z = A * x
         A_->Mult(*x_,*z_);

         // z += sigma M v
         z_->Add(sigma_,*y_); // y is now available

         // now do the big solve (A - sigma M) y = z

         solver_->Mult(*z_,*y_);
      }
      else if (ido == 1)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido
                      << ", computing w = (A - sigma M)^-1 (A + sigma M) v"
                      << endl << flush;
         }

         // we have to solve (A - sigma M)w = v with v partially formed

         // first, extract from workd
         // wrap the relevant portions of workd with our vector skeletons
         w_->SetData(&workd_[ipntr_[0]-1]);
         x_->SetData(&workd_[ipntr_[2]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // z = A v
         A_->Mult(*w_,*z_);

         // z += sigma M v
         z_->Add(sigma_,*x_);

         // now do the big solve (A - sigma M) y = z;
         solver_->Mult(*z_,*y_);
      }
      else if (ido == 2)
      {
         if ( myid_ == 0 && logging_ >= 2 )
         {
            mfem::out << "  ido = " << ido << ", computing w = M v"
                      << endl << flush;
         }

         // we have to compute w = M v.

         // wrap the relevant portions of workd with our vector skeletons
         x_->SetData(&workd_[ipntr_[0]-1]);
         y_->SetData(&workd_[ipntr_[1]-1]);

         // now do global matrix vector multiply
         // y = M * x;
         B_->Mult(*x_,*y_);
      }
   }
   return info;
}

void
ArPackSym::reverseComm()
{
   // The following variables are for ARPACK
   // int ido = 0;             // reverse communication what to do flag
   int info = 0;            // arpack info
   int ishifts = 1;         // method for selecting implicit shifts

   // fill in the iparam array
   iparam_[ 0] = ishifts;   // implicit shift method
   iparam_[ 1] = -1;        // Not used
   iparam_[ 2] = max_iter_; // maximum number of iterations
   iparam_[ 3] = 1;         // Block size only 1 works
   iparam_[ 4] = -1;        // output = number of coverged ritz values
   iparam_[ 5] = -1;        // Not used
   iparam_[ 6] = mode_;     // Must be 1,2,3,4,5;
   iparam_[ 7] = 0;         // output number of shifts for user to provide
   iparam_[ 8] = 0;         // output numops (OP*x)
   iparam_[ 9] = 0;         // output numopB (B*x)
   iparam_[10] = 0;         // output number of steps of re-orthogonalization

   if ( mode_ == 2 )
   {
      strncpy(which_,"SM",2);
   }
   else if ( mode_ == 3 )
   {
      strncpy(which_,"LM",2);
   }

   nloc_ = A_->Height();

   int nlocf; // size of largest local vector

   nlocf = this->computeNlocf();

   ncv_ = max(2*nev_,20);

   int nevf = nev_;
   int ncvf = ncv_;

   // Initializes the vectors dv and v.
   // these are used to store the eigenvalues
   // and the eigenvectors.
   dv_     = new double[nevf];       // eigenvalues
   v_      = new double[nlocf*ncvf]; // eigenvectors

   // the following vectors are work space for PDSAUPD
   resid_  = new double[nlocf];      // initial vector, and residual vector
   workd_  = new double[3*nlocf];    // workspace for reverse communication
   lworkl_ = ncvf * (ncvf + 8);      // length of workl_ array
   workl_  = new double[lworkl_];    // workspace for tridiagnonal system

   ////////////////////////////////////////////////////////////////////////////
   //   Set ARPACK Debug Parameters
   ////////////////////////////////////////////////////////////////////////////
   /*
   int logfil = 27, ndigit = -1, mgetv0 = 0;
   arpacksetcommdbg_(&logfil, &ndigit, &mgetv0);

   // setenv("FORT27","ARPACK.debug",1);

   int msaupd = 0, msaup2 = 0, msaitr = 0, mseigt = 0,
       msapps = 0, msgets = 0, mseupd = 0;

   arpacksymdbg_(&msaupd, &msaup2, &msaitr, &mseigt,
                 &msapps, &msgets, &mseupd);
   */
   ////////////////////////////////////////////////////////////////////////////
   //   Print PDSAUPD Parameters
   ////////////////////////////////////////////////////////////////////////////

   if ( myid_ == 0 && logging_ >= 0 )
   {
      mfem::out << "ArPack::reverseComm() - " << solverName_
                << " Parameters" << endl
                << "  BMAT:    " << bmat_      << endl
                << "  N:       " << nloc_      << endl
                << "  WHICH:   " << which_[0] << which_[1] << endl
                << "  NEV:     " << nev_      << endl
                << "  TOL:     " << tol_      << endl
                << "  NCV:     " << ncv_       << endl
                << "  LDV:     " << nloc_      << endl
                << "  ISHIFT:  " << iparam_[0] << endl
                << "  MXITER:  " << iparam_[2] << endl
                << "  NB:      " << iparam_[3] << endl
                << "  MODE:    " << iparam_[6] << endl
                << "  LWORKL:  " << lworkl_    << endl
                << "  INFO:    " << info      << endl
                << endl << flush;
   }

   switch (mode_)
   {
      case 1:
         info = this->reverseCommMode1();
         break;
      case 2:
         info = this->reverseCommMode2();
         break;
      case 3:
         info = this->reverseCommMode3();
         break;
      case 4:
         info = this->reverseCommMode4();
         break;
      case 5:
         info = this->reverseCommMode5();
         break;
      default:
         info = -1;
   }

   // if we got to here, we are done with PDSAUPD. check info to see if
   // there were any problems
   if (info < 0)
   {
      if ( myid_ == 0 )
      {
         mfem::out << endl << "error with PDSAUPD, info = " << info
                   << endl << flush;
         mfem::out << "check the documentation of PDSAUPD" << endl
                   << endl << flush;
      }
      exit(1);
   }
}

void
ArPackSym::Solve()
{
   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Running reverse communication loop ..." << endl;
   }

   this->reverseComm();

   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Computing eigenvalues and eigenvectors ..." << endl;
   }

   rvec_ = 1; // do compute eigenvectors
   select_ = new int[ncv_]; // not really used since we want all eigenvectors

   int info = this->computeEigs();
   if (info < 0)
   {
      printErrors(info,iparam_,bmat_,nloc_,which_,nev_,ncv_,lworkl_);
   }

   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Done computing eigenvalues and eigenvectors ..." << endl;
   }
}

void ArPackSym::GetEigenvalues(Array<double> & eigenvalues)
{
   eigenvalues.SetSize(nev_);
   eigenvalues = NAN;

   for (int i=0; i<iparam_[4]; i++)
   {
      eigenvalues[i] = dv_[i];
   }
}

void
ArPackSym::prepareEigenvectors()
{
   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Entering ArPAckSym::prepareEigenvectors ..." << endl;
   }

   if ( eigenvectors_ )
   {
      for (int i=0; i<nev_; i++)
      {
         delete eigenvectors_[i];
      }
      delete eigenvectors_;
   }

   eigenvectors_ = new Vector*[nev_];

   for (int i=0; i<nev_; i++)
   {
      eigenvectors_[i] = new Vector(nloc_);

      // This data copy is necessary to support stealing the vectors.
      // The reason is that the stolen vectors will be deleted one by
      // one which means their data arrays will be deleted one by one.
      // The array v_, on the other hand, must be allocated as a
      // contiguous array which cannot be deallocated piecemeal.
      //
      for (int j=0; j<nloc_; j++)
      {
         (*eigenvectors_[i])(j) = v_[nloc_*i+j];
      }
   }

   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Leaving ArPAckSym::prepareEigenvectors ..." << endl;
   }
}

Vector & ArPackSym::GetEigenvector(unsigned int i)
{
   if ( !eigenvectors_ )
   {
      this->prepareEigenvectors();
   }

   return *eigenvectors_[i];
}

Vector ** ArPackSym::StealEigenvectors()
{
   if ( !eigenvectors_ )
   {
      this->prepareEigenvectors();
   }

   Vector ** vecs = eigenvectors_;
   eigenvectors_ = NULL;

   return vecs;
}

int
ArPackSym::computeIter(int & ido)
{
   int info = 0;

   DSAUPD(&ido, &bmat_, &nloc_, which_, &nev_, &tol_,
          resid_,&ncv_, v_, &nloc_, iparam_, ipntr_,
          workd_, workl_, &lworkl_, &info );

   return info;
}

int
ArPackSym::computeEigs()
{
   int ierr = 0;

   DSEUPD(&rvec_, &hwmny_, select_, dv_, v_, &nloc_, &sigma_,
          &bmat_, &nloc_,
          which_, &nev_, &tol_, resid_, &ncv_, v_, &nloc_,
          iparam_, ipntr_, workd_, workl_, &lworkl_, &ierr );

   return ierr;
}

void
ArPackSym::printErrors(const int & info, const int iparam[],
                       const char & bmat, const int & n,
                       const char which[],
                       const int & nev, const int & ncv,
                       const int & lworkl )
{
   switch (info)
   {
      case -1:
         mfem::out << solverName_ << " Reports:  "
                   << "N must be positive. "
                   << "N = " << n << endl << flush;
         break;
      case -2:
         mfem::out << solverName_ << " Reports:  "
                   << "NEV must be positive. "
                   << "NEV = " << nev << endl << flush;
         break;
      case -3:
         mfem::out << solverName_ << " Reports:  "
                   << "NCV must be greater than NEV and "
                   << "less than or equal to N. "
                   << "NEV/NCV/N = " << nev << "/" << ncv << "/" << n
                   << endl << flush;
         break;
      case -4:
         mfem::out << solverName_ << " Reports:  "
                   << "The maximum number of Arnoldi update iterations allowed "
                   << "must be greater than zero. "
                   << "IPARAM(3) = " << iparam[2] << endl << flush;
         break;
      case -5:
         mfem::out << solverName_ << " Reports:  "
                   << "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'. "
                   << "WHICH = " << which[0] << which[1] << endl << flush;
         break;
      case -6:
         mfem::out << solverName_ << " Reports:  "
                   << "BMAT must be one of 'I' or 'G'. "
                   << "BMAT = " << bmat << endl << flush;
         break;
      case -7:
         mfem::out << solverName_ << " Reports:  "
                   << "Length of private work array WORKL is not sufficient. "
                   << "LWORKL = " << lworkl << endl << flush;
         break;
      case -8:
         mfem::out << solverName_ << " Reports:  "
                   << "Error return from trid. eigenvalue calculation; "
                   << "Informatinal error from LAPACK routine dsteqr."
                   << endl << flush;
         break;
      case -9:
         mfem::out << solverName_ << " Reports:  "
                   << "Starting vector is zero." << endl << flush;
         break;
      case -10:
         mfem::out << solverName_ << " Reports:  "
                   << "IPARAM(7) must be 1,2,3,4,5. "
                   << "IPARAM(7) = " << iparam[6] << endl << flush;
         break;
      case -11:
         mfem::out << solverName_ << " Reports:  "
                   << "IPARAM(7) = 1 and BMAT = 'G' "
                   << "are incompatable." << endl << flush;
         break;
      case -12:
         mfem::out << solverName_ << " Reports:  "
                   << "IPARAM(1) must be equal to 0 or 1. "
                   << "IPARAM(7) = " << iparam[6] << endl << flush;
         break;
      case -13:
         mfem::out << solverName_ << " Reports:  "
                   << "NEV and WHICH = 'BE' are incompatable. "
                   << "NEV/WHICH = " << nev << "/" << which[0] << which[1]
                   << endl << flush;
         break;
      case -9999:
         mfem::out << solverName_ << " Reports:  "
                   << "Could not build an Arnoldi factorization. "
                   << "IPARAM(5) returns the size of the current Arnoldi "
                   << "factorization. The user is advised to check that "
                   << "enough workspace and array storage has been allocated. "
                   << endl << flush
                   << "IPARAM(5) = " << iparam[4] << endl << flush;
         break;
      default:
         mfem::out << solverName_ << " Reports "
                   << "an unrecognized Info Value:  info = " << info << endl
                   << flush;
   }

}

#ifdef MFEM_USE_MPI

ParArPackSym::ParArPackSym(MPI_Comm comm)
   : ArPackSym(),
     comm_(comm),
     commf_(MPI_Comm_c2f(comm))
{
   solverName_ = "PDSAUPD";

   MPI_Comm_rank(comm_,&myid_);
   MPI_Comm_size(comm_,&numProcs_);
}


void
ParArPackSym::SetOperator(Operator & A)
{
   A_ = &A;

   if ( typeid(A) == typeid(HypreParMatrix) && x_ == NULL )
   {
      HypreParMatrix * A_hypre = (HypreParMatrix*)A_;
      w_ = new HypreParVector(comm_, A_hypre->N(), NULL, A_hypre->ColPart());
      x_ = new HypreParVector(comm_, A_hypre->N(), NULL, A_hypre->ColPart());
      y_ = new HypreParVector(comm_, A_hypre->N(), NULL, A_hypre->ColPart());
      z_ = new HypreParVector(*A_hypre);
   }
}

void
ParArPackSym::SetMassMatrix(Operator & B)
{
   B_ = &B;

   bmat_ = 'G'; // If the user sets a B, we can assume B != I

   if ( typeid(B) == typeid(HypreParMatrix) && x_ == NULL )
   {
      HypreParMatrix * B_hypre = (HypreParMatrix*)B_;
      x_ = new HypreParVector(comm_, B_hypre->N(), NULL, B_hypre->ColPart());
      y_ = new HypreParVector(comm_, B_hypre->N(), NULL, B_hypre->ColPart());
      z_ = new HypreParVector(*B_hypre);
   }
}

void ParArPackSym::GetEigenvalues(Array<double> & eigenvalues)
{
   eigenvalues.SetSize(nev_);
   eigenvalues = NAN;

   for (int i=0; i<iparam_[4]; i++)
   {
      eigenvalues[i] = dv_[i];
   }
}

void
ParArPackSym::prepareEigenvectors()
{
   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Entering ParArPAckSym::prepareEigenvectors ..." << endl;
   }

   if ( eigenvectors_ )
   {
      for (int i=0; i<nev_; i++)
      {
         delete eigenvectors_[i];
      }
      delete eigenvectors_;
   }

   int locSize = nloc_;
   int glbSize = 0;

   if (HYPRE_AssumedPartitionCheck())
   {
      part_ = new HYPRE_Int[2];

      MPI_Scan(&locSize, &part_[1], 1, HYPRE_MPI_INT, MPI_SUM, comm_);

      part_[0] = part_[1] - locSize;

      MPI_Allreduce(&locSize, &glbSize, 1, HYPRE_MPI_INT, MPI_SUM, comm_);
   }
   else
   {
      part_ = new HYPRE_Int[numProcs_+1];

      MPI_Allgather(&locSize, 1, MPI_INT,
                    &part_[1], 1, HYPRE_MPI_INT, comm_);

      part_[0] = 0;
      for (int i=0; i<numProcs_; i++)
      {
         part_[i+1] += part_[i];
      }

      glbSize = part_[numProcs_];
   }

   eigenvectors_ = (Vector**)new HypreParVector*[nev_];

   for (int i=0; i<nev_; i++)
   {
      HypreParVector *vec = new HypreParVector(comm_,glbSize,part_);

      // This data copy is necessary to support stealing the vectors.
      // The reason is that the stolen vectors will be deleted one by
      // one which means their data arrays will be deleted one by one.
      // The array v_, on the other hand, must be allocated as a
      // contiguous array which cannot be deallocated piecemeal.
      //
      for (int j=0; j<locSize; j++)
      {
         ((double*)*vec)[j] = v_[nloc_*i+j];
      }
      eigenvectors_[i] = (Vector*)vec;
   }

   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Leaving ParArPAckSym::prepareEigenvectors ..." << endl;
   }
}

//HypreParVector & ParArPackSym::GetEigenvector(unsigned int i)
Vector & ParArPackSym::GetEigenvector(unsigned int i)
{
   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Entering ParArPackSym::GetEigenvector" << endl;
   }

   if ( !eigenvectors_ )
   {
      this->prepareEigenvectors();
   }

   HypreParVector * vec = dynamic_cast<HypreParVector*>(eigenvectors_[i]);

   if ( myid_ == 0 && logging_ >= 3 )
   {
      mfem::out << "Leaving ParArPackSym::GetEigenvector" << endl;
   }
   return *vec;
}

//HypreParVector ** ParArPackSym::StealEigenvectors()
Vector ** ParArPackSym::StealEigenvectors()
{
   if ( !eigenvectors_ )
   {
      this->prepareEigenvectors();
   }

   HypreParVector ** vecs = (HypreParVector**)eigenvectors_;
   eigenvectors_ = NULL;

   return (Vector**)vecs;
}

int
ParArPackSym::computeIter(int & ido)
{
   int info = 0;

   PDSAUPD(&commf_, &ido, &bmat_, &nloc_, which_, &nev_, &tol_,
           resid_, &ncv_, v_, &nloc_, iparam_, ipntr_,
           workd_, workl_, &lworkl_, &info );

   return info;
}

int
ParArPackSym::computeNlocf()
{
   int nlocf = 0;

   MPI_Allreduce((void*) &nloc_,(void*) &nlocf, 1, MPI_INT, MPI_MAX, comm_);

   return nlocf;
}

int
ParArPackSym::computeEigs()
{
   int ierr = 0;

   PDSEUPD (&commf_,&rvec_, &hwmny_, select_, dv_, v_, &nloc_, &sigma_,
            &bmat_, &nloc_,
            which_, &nev_, &tol_, resid_, &ncv_, v_, &nloc_,
            iparam_, ipntr_, workd_, workl_, &lworkl_, &ierr );

   return ierr;
}

#endif // MFEM_USE_MPI

};

#endif // MFEM_USE_ARPACK