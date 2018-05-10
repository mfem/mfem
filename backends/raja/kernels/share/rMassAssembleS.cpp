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
#include "../raja.hpp"

// *****************************************************************************
extern "C" kernel
void rMassAssemble2S0(const int numElements,
                      const int NUM_QUAD_2D,
                      const double COEFF,
                      const double* quadWeights,
                      const double* J,
                      double* __restrict oper) {
#ifdef __LAMBDA__
  forallS(eOff,numElements,A2_ELEMENT_BATCH,
#else
  const int idx = blockIdx.x;
  const int eOff = idx * A2_ELEMENT_BATCH;
  if (eOff < numElements)
#endif
  {
#ifdef __LAMBDA__
    for (int e = eOff; e < (eOff + A2_ELEMENT_BATCH); ++e) {
#else
    { const int e = threadIdx.x;
#endif
      if (e < numElements) {
#ifdef __LAMBDA__
        for (int qOff = 0; qOff < A2_QUAD_BATCH; ++qOff) {
#else
       { const int qOff = threadIdx.y;
#endif
          for (int q = qOff; q < NUM_QUAD_2D; q += A2_QUAD_BATCH) {
            const double J11 = J[ijklNM(0, 0, q, e,2,NUM_QUAD_2D)];
            const double J12 = J[ijklNM(1, 0, q, e,2,NUM_QUAD_2D)];
            const double J21 = J[ijklNM(0, 1, q, e,2,NUM_QUAD_2D)];
            const double J22 = J[ijklNM(1, 1, q, e,2,NUM_QUAD_2D)];

            oper[ijN(q,e,NUM_QUAD_2D)] = quadWeights[q] * COEFF * ((J11 * J22) - (J21 * J12));
          }
        }
      }
    }
  }
#ifdef __LAMBDA__
  );
#endif
}
// *****************************************************************************
static void rMassAssemble2S(const int NUM_QUAD_2D,
                            const int numElements,
                            const double COEFF,
                            const double* quadWeights,
                            const double* J,
                            double* __restrict oper) {
#ifndef __LAMBDA__
  const int gX = ((numElements+A2_ELEMENT_BATCH-1)/A2_ELEMENT_BATCH);
  const int tX = A2_ELEMENT_BATCH;
  const int tY = A2_QUAD_BATCH;
  dim3 threads(tX, tY, 1);
  dim3 blocks(gX, 1, 1);
#endif
  cuKerGBS(rMassAssemble2S,blocks,threads,numElements,NUM_QUAD_2D,COEFF,quadWeights,J,oper);
}

// *****************************************************************************
void rMassAssembleS(const int dim,
                    const int NUM_QUAD,
                    const int numElements,
                    const double* quadWeights,
                    const double* J,
                    const double COEFF,
                    double* __restrict oper) {
  assert(false);
  push(Green);
  if (dim==1) {assert(false);}
  if (dim==2) rMassAssemble2S(NUM_QUAD,numElements,COEFF,quadWeights,J,oper);
  if (dim==3) {assert(false);}
  pop();
}
