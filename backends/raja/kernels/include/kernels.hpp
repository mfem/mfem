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
#ifndef LAGHOS_RAJA_KERNELS
#define LAGHOS_RAJA_KERNELS

#define restrict __restrict__

// **** BLAS1 ******************************************************************
void vector_neg(const int, double* restrict);
//extern "C" kernel void d_vector_op_eq(const int, const double, double* restrict);
void vector_op_eq(const int, const double, double* restrict);
void vector_xpay(const int, const double, double* restrict, const double* restrict,
                 const double* restrict);
void vector_xsy(const int, double* restrict, const double* restrict, const double* restrict);
void vector_axpy(const int, const double, double* restrict, const double* restrict);
void vector_map_dofs(const int, double* restrict, const double* restrict, const int* restrict);
template <class T>
void vector_map_add_dofs(const int, T* restrict, const T* restrict, const int* restrict);
void vector_clear_dofs(const int, double* restrict, const int* restrict);
void vector_vec_sub(const int, double* restrict, const double* restrict);
void vector_vec_add(const int, double* restrict, const double* restrict);
void vector_vec_mul(const int, double* restrict, const double);
void vector_set_subvector(const int, double* restrict, const double* restrict,
                          const int* restrict);
void vector_get_subvector(const int, double* restrict, const double* restrict,
                          const int* restrict);
void vector_set_subvector_const(const int, const double, double* restrict,
                                const int* restrict);
double vector_dot(const int, const double* restrict, const double* restrict);
double vector_min(const int, const double* restrict);

// *****************************************************************************
void reduceMin(int, const double*, double*);
void reduceSum(int, const double*, const double*, double*);

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

// mapping *********************************************************************
void rSetSubVector(const int entries,
                   const int* restrict indices,
                   const double* restrict in,
                   double* restrict out);

void rMapSubVector(const int entries,
                   const int* restrict indices,
                   const double* restrict in,
                   double* restrict out);

void rExtractSubVector(const int entries,
                       const int* restrict indices,
                       const double* restrict in,
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

// kForce **********************************************************************
void rForceMult(const int NUM_DIM,
                const int NUM_DOFS_1D,
                const int NUM_QUAD_1D,
                const int L2_DOFS_1D,
                const int H1_DOFS_1D,
                const int nzones,
                const double* restrict L2DofToQuad,
                const double* restrict H1QuadToDof,
                const double* restrict H1QuadToDofD,
                const double* restrict stressJinvT,
                const double* restrict e,
                double* restrict v);
void rForceMultS(const int NUM_DIM,
                 const int NUM_DOFS_1D,
                 const int NUM_QUAD_1D,
                 const int L2_DOFS_1D,
                 const int H1_DOFS_1D,
                 const int nzones,
                 const double* restrict L2DofToQuad,
                 const double* restrict H1QuadToDof,
                 const double* restrict H1QuadToDofD,
                 const double* restrict stressJinvT,
                 const double* restrict e,
                 double* restrict v);

void rForceMultTranspose(const int NUM_DIM,
                         const int NUM_DOFS_1D,
                         const int NUM_QUAD_1D,
                         const int L2_DOFS_1D,
                         const int H1_DOFS_1D,
                         const int nzones,
                         const double* restrict L2QuadToDof,
                         const double* restrict H1DofToQuad,
                         const double* restrict H1DofToQuadD,
                         const double* restrict stressJinvT,
                         const double* restrict v,
                         double* restrict e);
void rForceMultTransposeS(const int NUM_DIM,
                          const int NUM_DOFS_1D,
                          const int NUM_QUAD_1D,
                          const int L2_DOFS_1D,
                          const int H1_DOFS_1D,
                          const int nzones,
                          const double* restrict L2QuadToDof,
                          const double* restrict H1DofToQuad,
                          const double* restrict H1DofToQuadD,
                          const double* restrict stressJinvT,
                          const double* restrict v,
                          double* restrict e);

// *****************************************************************************
void rNodeCopyByVDim(const int elements,
                     const int numDofs,
                     const int ndofs,
                     const int dims,
                     const int* eMap,
                     const double* Sx,
                     double* nodes);

// *****************************************************************************
void rIniGeom(const int dim,
              const int nDofs,
              const int nQuads,
              const int nzones,
              const double* restrict dofToQuadD,
              const double* restrict nodes,
              double* restrict J,
              double* restrict invJ,
              double* restrict detJ);

// *****************************************************************************
void rGlobalToLocal(const int NUM_VDIM,
                    const bool VDIM_ORDERING,
                    const int globalEntries,
                    const int localEntries,
                    const int* restrict offsets,
                    const int* restrict indices,
                    const double* restrict globalX,
                    double* restrict localX);

void rLocalToGlobal(const int NUM_VDIM,
                    const bool VDIM_ORDERING,
                    const int globalEntries,
                    const int localEntries,
                    const int* restrict offsets,
                    const int* restrict indices,
                    const double* restrict localX,
                    double* restrict globalX);

// *****************************************************************************
void rMassMultAdd(const int dim,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int numElements,
                  const double* restrict dofToQuad,
                  const double* restrict dofToQuadD,
                  const double* restrict quadToDof,
                  const double* restrict quadToDofD,
                  const double* restrict op,
                  const double* restrict x,
                  double* restrict y);
void rMassMultAddS(const int dim,
                  const int NUM_DOFS_1D,
                  const int NUM_QUAD_1D,
                  const int numElements,
                  const double* restrict dofToQuad,
                  const double* restrict dofToQuadD,
                  const double* restrict quadToDof,
                  const double* restrict quadToDofD,
                  const double* restrict op,
                  const double* restrict x,
                  double* restrict y);

#endif // LAGHOS_RAJA_KERNELS
