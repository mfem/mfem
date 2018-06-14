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
#ifndef MFEM_KERNELS_QUADRATURE_POINTS
#define MFEM_KERNELS_QUADRATURE_POINTS

// *****************************************************************************
void rGridFuncToQuad(const int dim,
                     const int NUM_VDIM,
                     const int NUM_DOFS_1D,
                     const int NUM_QUAD_1D,
                     const int numElements,
                     const double* restrict dofToQuad,
                     const int* l2gMap,
                     const double* restrict gf,
                     double* restrict out);

void rGridFuncToQuadS(const int dim,
                      const int NUM_VDIM,
                      const int NUM_DOFS_1D,
                      const int NUM_QUAD_1D,
                      const int numElements,
                      const double* restrict dofToQuad,
                      const int* l2gMap,
                      const double* restrict gf,
                      double* restrict out);

// kQuadratureData *************************************************************
void rInitQuadratureData(const int NUM_QUAD,
                         const int numElements,
                         const double* restrict rho0,
                         const double* restrict detJ,
                         const double* restrict quadWeights,
                         double* restrict rho0DetJ0w);

void rUpdateQuadratureData(const double GAMMA,
                           const double H0,
                           const double CFL,
                           const bool USE_VISCOSITY,
                           const int NUM_DIM,
                           const int NUM_QUAD,
                           const int NUM_QUAD_1D,
                           const int NUM_DOFS_1D,
                           const int numElements,
                           const double* restrict dofToQuad,
                           const double* restrict dofToQuadD,
                           const double* restrict quadWeights,
                           const double* restrict v,
                           const double* restrict e,
                           const double* restrict rho0DetJ0w,
                           const double* restrict invJ0,
                           const double* restrict J,
                           const double* restrict invJ,
                           const double* restrict detJ,
                           double* restrict stressJinvT,
                           double* restrict dtEst);
void rUpdateQuadratureDataS(const double GAMMA,
                            const double H0,
                            const double CFL,
                            const bool USE_VISCOSITY,
                            const int NUM_DIM,
                            const int NUM_QUAD,
                            const int NUM_QUAD_1D,
                            const int NUM_DOFS_1D,
                            const int numElements,
                            const double* restrict dofToQuad,
                            const double* restrict dofToQuadD,
                            const double* restrict quadWeights,
                            const double* restrict v,
                            const double* restrict e,
                            const double* restrict rho0DetJ0w,
                            const double* restrict invJ0,
                            const double* restrict J,
                            const double* restrict invJ,
                            const double* restrict detJ,
                            double* restrict stressJinvT,
                            double* restrict dtEst);

#endif // MFEM_KERNELS_QUADRATURE_POINTS
