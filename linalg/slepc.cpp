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

#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC
#ifdef MFEM_USE_SLEPC

#include "linalg.hpp"

#include "slepc.h"

#include "petscinternals.hpp"

static PetscErrorCode ierr;

namespace mfem
{

void MFEMInitializeSlepc()
{
   MFEMInitializeSlepc(NULL,NULL,NULL,NULL);
}

void MFEMInitializeSlepc(int *argc,char*** argv)
{
   MFEMInitializeSlepc(argc,argv,NULL,NULL);
}

void MFEMInitializeSlepc(int *argc,char ***argv,const char rc_file[],
                         const char help[])
{
   ierr = SlepcInitialize(argc,argv,rc_file,help);
   MFEM_VERIFY(!ierr,"Unable to initialize SLEPc");
}

void MFEMFinalizeSlepc()
{
   ierr = SlepcFinalize();
   MFEM_VERIFY(!ierr,"Unable to finalize SLEPc");
}


SlepcEigenSolver::SlepcEigenSolver(MPI_Comm comm, const std::string &prefix)
{
   clcustom = false;
   VR = NULL;
   VC = NULL;

   ierr = EPSCreate(comm,&eps); CCHKERRQ(comm,ierr);
   ierr = EPSSetOptionsPrefix(eps, prefix.c_str()); PCHKERRQ(eps, ierr);
}

SlepcEigenSolver::~SlepcEigenSolver()
{
   delete VR;
   delete VC;

   MPI_Comm comm;
   ierr = PetscObjectGetComm((PetscObject)eps,&comm); PCHKERRQ(eps,ierr);
   ierr = EPSDestroy(&eps); CCHKERRQ(comm,ierr);
}


void SlepcEigenSolver::SetOperator(const PetscParMatrix &op)
{
   delete VR;
   delete VC;
   VR = VC = NULL;

   ierr = EPSSetOperators(eps,op,NULL); PCHKERRQ(eps, ierr);

   VR = new PetscParVector(op, true, false);
   VC = new PetscParVector(op, true, false);

}

void SlepcEigenSolver::SetOperators(const PetscParMatrix &op,
                                    const PetscParMatrix&opB)
{
   delete VR;
   delete VC;
   VR = VC = NULL;

   ierr = EPSSetOperators(eps,op,opB); PCHKERRQ(eps,ierr);

   VR = new PetscParVector(op, true, false);
   VC = new PetscParVector(op, true, false);
}

void SlepcEigenSolver::SetTol(double tol)
{
   int max_its;

   ierr = EPSGetTolerances(eps,NULL,&max_its); PCHKERRQ(eps,ierr);
   // Work around uninitialized maximum iterations
   if (max_its==0) { max_its = PETSC_DECIDE; }
   ierr = EPSSetTolerances(eps,tol,max_its); PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::SetMaxIter(int max_its)
{
   double tol;

   ierr = EPSGetTolerances(eps,&tol,NULL); PCHKERRQ(eps,ierr);
   ierr = EPSSetTolerances(eps,tol,max_its); PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::SetNumModes(int num_eigs)
{
   ierr = EPSSetDimensions(eps,num_eigs,PETSC_DECIDE,PETSC_DECIDE);
   PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::Solve()
{
   Customize();

   ierr = EPSSolve(eps); PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::Customize(bool customize) const
{
   if (!customize) {clcustom = true; }
   if (!clcustom)
   {
      ierr = EPSSetFromOptions(eps); PCHKERRQ(eps,ierr);
   }
   clcustom = true;
}

void SlepcEigenSolver::GetEigenvalue(unsigned int i, double & lr) const
{
   ierr = EPSGetEigenvalue(eps,i,&lr,NULL); PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::GetEigenvalue(unsigned int i, double & lr,
                                     double & lc) const
{
   ierr = EPSGetEigenvalue(eps,i,&lr,&lc); PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::GetEigenvector(unsigned int i, Vector & vr) const
{
   MFEM_VERIFY(VR,"Missing real vector");

   MFEM_ASSERT(vr.Size() == VR->Size(), "invalid vr.Size() = " << vr.Size()
               << ", expected size = " << VR->Size());

   VR->PlaceArray(vr.GetData());
   ierr = EPSGetEigenvector(eps,i,*VR,NULL); PCHKERRQ(eps,ierr);
   VR->ResetArray();

}

void SlepcEigenSolver::GetEigenvector(unsigned int i, Vector & vr,
                                      Vector & vc) const
{
   MFEM_VERIFY(VR,"Missing real vector");
   MFEM_VERIFY(VC,"Missing imaginary vector");
   MFEM_ASSERT(vr.Size() == VR->Size(), "invalid vr.Size() = " << vr.Size()
               << ", expected size = " << VR->Size());
   MFEM_ASSERT(vc.Size() == VC->Size(), "invalid vc.Size() = " << vc.Size()
               << ", expected size = " << VC->Size());

   VR->PlaceArray(vr.GetData());
   VC->PlaceArray(vc.GetData());
   ierr = EPSGetEigenvector(eps,i,*VR,*VC); PCHKERRQ(eps,ierr);
   VR->ResetArray();
   VC->ResetArray();
}

int SlepcEigenSolver::GetNumConverged()
{
   int num_conv;
   ierr = EPSGetConverged(eps,&num_conv); PCHKERRQ(eps,ierr);
   return num_conv;
}

void SlepcEigenSolver::SetWhichEigenpairs(SlepcEigenSolver::Which which)
{
   switch (which)
   {
      case SlepcEigenSolver::LARGEST_MAGNITUDE:
         ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::SMALLEST_MAGNITUDE:
         ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_MAGNITUDE); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::LARGEST_REAL:
         ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::SMALLEST_REAL:
         ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::LARGEST_IMAGINARY:
         ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_IMAGINARY); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::SMALLEST_IMAGINARY:
         ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_IMAGINARY); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::TARGET_MAGNITUDE:
         ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::TARGET_REAL:
         ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_REAL); PCHKERRQ(eps,ierr);
         break;
      default:
         MFEM_ABORT("Which eigenpair not implemented!");
         break;
   }
}

void SlepcEigenSolver::SetTarget(double target)
{
   ierr = EPSSetTarget(eps,target); PCHKERRQ(eps,ierr);
}

void SlepcEigenSolver::SetSpectralTransformation(
   SlepcEigenSolver::SpectralTransformation transformation)
{
   ST st;
   ierr = EPSGetST(eps,&st); PCHKERRQ(eps,ierr);
   switch (transformation)
   {
      case SlepcEigenSolver::SHIFT:
         ierr = STSetType(st,STSHIFT); PCHKERRQ(eps,ierr);
         break;
      case SlepcEigenSolver::SHIFT_INVERT:
         ierr = STSetType(st,STSINVERT); PCHKERRQ(eps,ierr);
         break;
      default:
         MFEM_ABORT("Spectral transformation not implemented!");
         break;
   }
}

}

#endif  // MFEM_USE_SLEPC
#endif  // MFEM_USE_PETSC
#endif  // MFEM_USE_MPI
