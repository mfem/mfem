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
#include "../kernels.hpp"

// *****************************************************************************
void rMassAssemble2D(const int, const int, const double,
                     const double*, const double*, double*);
void rMassAssemble3D(const int, const int, const double,
                     const double*, const double*, double*);

// *****************************************************************************
void rMassAssemble(const int dim,
                   const int NUM_QUAD,
                   const int numElements,
                   const double* quadWeights,
                   const double* J,
                   const double COEFF,
                   double* oper)
{
   push(Lime);
   //assert(false);
   if (dim==1) { assert(false); }
   if (dim==2) { rMassAssemble2D(numElements,NUM_QUAD,COEFF,quadWeights,J,oper); }
   if (dim==3) { rMassAssemble3D(numElements,NUM_QUAD,COEFF,quadWeights,J,oper); }
   pop();
}
