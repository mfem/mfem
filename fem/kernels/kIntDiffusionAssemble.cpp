// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../general/okina.hpp"
#include "occa.hpp"

// *****************************************************************************
namespace mfem
{

// **************************************************************************
static void oAssemble2D(const int NUM_QUAD_1D,
                        const int numElements,
                        const double* __restrict quadWeights,
                        const double* __restrict J,
                        const double COEFF,
                        double* __restrict oper)
{
  GET_OCCA_CONST_MEMORY(quadWeights);
  GET_OCCA_CONST_MEMORY(J);
  GET_OCCA_MEMORY(oper);
  static bool setup = false;
  static occa::kernel assemble2D = NULL;
  //static occa::memory o_quadWeights, o_J, o_oper;
  if (not setup){
    static occa::device device;
    device.setup("mode: 'Serial'");
    const size_t W_SZ = 4;
    const size_t J_SZ = 327680;
    const size_t O_SZ = 245760;
    o_quadWeights = device.malloc(W_SZ*sizeof(double));
    o_J = device.malloc(J_SZ*sizeof(double));
    o_oper = device.malloc(O_SZ*sizeof(double));
    assemble2D =
      device.buildKernel("/Users/camier1/home/mfem/okina-occa/fem/kernels/oIntDiffusionAssemble.okl", "Assemble2D");
  }
  o_quadWeights.copyFrom(quadWeights);
  o_J.copyFrom(J);  
  assemble2D(NUM_QUAD_1D, numElements, o_quadWeights, o_J, COEFF, o_oper);
  o_oper.copyTo(oper);
}

// **************************************************************************
static void kAssemble2D(const int NUM_QUAD_1D,
                        const int numElements,
                        const double* __restrict quadWeights,
                        const double* __restrict J,
                        const double COEFF,
                        double* __restrict oper)
{ 
   GET_CONST_ADRS(quadWeights);
   GET_CONST_ADRS(J);
   GET_ADRS(oper);
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e,numElements,
   {
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         const double J11 = J[ijklNM(0,0,q,e,2,NUM_QUAD)];
         const double J12 = J[ijklNM(1,0,q,e,2,NUM_QUAD)];
         const double J21 = J[ijklNM(0,1,q,e,2,NUM_QUAD)];
         const double J22 = J[ijklNM(1,1,q,e,2,NUM_QUAD)];
         const double c_detJ = d_quadWeights[q] * COEFF / ((J11*J22)-(J21*J12));
         oper[ijkNM(0,q,e,3,NUM_QUAD)] =  c_detJ * (J21*J21 + J22*J22);
         oper[ijkNM(1,q,e,3,NUM_QUAD)] = -c_detJ * (J21*J11 + J22*J12);
         oper[ijkNM(2,q,e,3,NUM_QUAD)] =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}

// **************************************************************************
static void kAssemble3D(const int NUM_QUAD_1D,
                        const int numElements,
                        const double* __restrict quadWeights,
                        const double* __restrict J,
                        const double COEFF,
                        double* __restrict oper)
{
   GET_CONST_ADRS(quadWeights);
   GET_CONST_ADRS(J);
   GET_ADRS(oper);
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e,numElements,
   {
      for (int q = 0; q < NUM_QUAD; ++q)
      {
         const double J11 = J[ijklNM(0,0,q,e,3,NUM_QUAD)];
         const double J12 = J[ijklNM(1,0,q,e,3,NUM_QUAD)];
         const double J13 = J[ijklNM(2,0,q,e,3,NUM_QUAD)];
         const double J21 = J[ijklNM(0,1,q,e,3,NUM_QUAD)];
         const double J22 = J[ijklNM(1,1,q,e,3,NUM_QUAD)];
         const double J23 = J[ijklNM(2,1,q,e,3,NUM_QUAD)];
         const double J31 = J[ijklNM(0,2,q,e,3,NUM_QUAD)];
         const double J32 = J[ijklNM(1,2,q,e,3,NUM_QUAD)];
         const double J33 = J[ijklNM(2,2,q,e,3,NUM_QUAD)];
         const double detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
                              (J13 * J21 * J32) - (J13 * J22 * J31) -
                              (J12 * J21 * J33) - (J11 * J23 * J32));
         const double c_detJ = d_quadWeights[q] * COEFF / detJ;
         // adj(J)
         const double A11 = (J22 * J33) - (J23 * J32);
         const double A12 = (J23 * J31) - (J21 * J33);
         const double A13 = (J21 * J32) - (J22 * J31);
         const double A21 = (J13 * J32) - (J12 * J33);
         const double A22 = (J11 * J33) - (J13 * J31);
         const double A23 = (J12 * J31) - (J11 * J32);
         const double A31 = (J12 * J23) - (J13 * J22);
         const double A32 = (J13 * J21) - (J11 * J23);
         const double A33 = (J11 * J22) - (J12 * J21);
         // adj(J)^Tadj(J)
         oper[ijkNM(0,q,e,6,NUM_QUAD)] = c_detJ *
            (A11*A11 + A21*A21 + A31*A31); // (1,1)
         oper[ijkNM(1,q,e,6,NUM_QUAD)] = c_detJ *
            (A11*A12 + A21*A22 + A31*A32); // (1,2), (2,1)
         oper[ijkNM(2,q,e,6,NUM_QUAD)] = c_detJ *
            (A11*A13 + A21*A23 + A31*A33); // (1,3), (3,1)
         oper[ijkNM(3,q,e,6,NUM_QUAD)] = c_detJ *
            (A12*A12 + A22*A22 + A32*A32); // (2,2)
         oper[ijkNM(4,q,e,6,NUM_QUAD)] = c_detJ *
            (A12*A13 + A22*A23 + A32*A33); // (2,3), (3,2)
         oper[ijkNM(5,q,e,6,NUM_QUAD)] = c_detJ *
            (A13*A13 + A23*A23 + A33*A33); // (3,3)
      }
   });
}

// *****************************************************************************
void kIntDiffusionAssemble(const int dim,
                           const int NUM_QUAD_1D,
                           const int numElements,
                           const double* __restrict quadWeights,
                           const double* __restrict J,
                           const double COEFF,
                           double* __restrict oper)
{
  if (dim==1) { assert(false); }
  if (dim==2){
    if (config::Get().occa())
      oAssemble2D(NUM_QUAD_1D, numElements, quadWeights, J, COEFF, oper);
    else
      kAssemble2D(NUM_QUAD_1D, numElements, quadWeights, J, COEFF, oper);
  }
  if (dim==3) { kAssemble3D(NUM_QUAD_1D, numElements, quadWeights, J, COEFF, oper); }
}

}
