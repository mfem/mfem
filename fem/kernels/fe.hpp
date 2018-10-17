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
#ifndef MFEM_FEM_KERNELS_FE
#define MFEM_FEM_KERNELS_FE

MFEM_NAMESPACE

// *****************************************************************************
void kH1_TriangleElement(const size_t p,
                         const size_t k,
                         const size_t height,
                         const double *shape_x,
                         const double *shape_y,
                         const double *shape_l,
                         double *T);

// *****************************************************************************
void kH1_TriangleElement_CalcShape(const size_t p,
                                   const double *shape_x,
                                   const double *shape_y,
                                   const double *shape_l,
                                   double *T);

// *****************************************************************************
void kH1_TriangleElement_CalcDShape(const size_t p,
                                    const size_t height,
                                    const double *shape_x,
                                    const double *shape_y,
                                    const double *shape_l,
                                    const double *dshape_x,
                                    const double *dshape_y,
                                    const double *dshape_l,
                                    double *du);

// *****************************************************************************
void kLinear3DFiniteElementHeightEq4(double*);
      
// *****************************************************************************
void kBasis(const size_t p, const double *x, double *w);

// *****************************************************************************
void kNodesAreIncreasing(const size_t p, const double *x);

MFEM_NAMESPACE_END

#endif // MFEM_FEM_KERNELS_FE
