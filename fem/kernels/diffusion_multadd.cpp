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

// *****************************************************************************
namespace mfem
{
namespace kernels
{
namespace fem
{

#ifdef __OCCA__
// *****************************************************************************
static void oBiPADiffusionMultAdd2D(const int NUM_DOFS_1D,
                                    const int NUM_QUAD_1D,
                                    const int numElements,
                                    const double* __restrict dofToQuad,
                                    const double* __restrict dofToQuadD,
                                    const double* __restrict quadToDof,
                                    const double* __restrict quadToDofD,
                                    const double* __restrict oper,
                                    const double* __restrict solIn,
                                    double* __restrict solOut)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;

   GET_OCCA_CONST_MEMORY(dofToQuad);
   GET_OCCA_CONST_MEMORY(dofToQuadD);
   GET_OCCA_CONST_MEMORY(quadToDof);
   GET_OCCA_CONST_MEMORY(quadToDofD);
   GET_OCCA_CONST_MEMORY(oper);
   GET_OCCA_CONST_MEMORY(solIn);
   GET_OCCA_MEMORY(solOut);

   NEW_OCCA_PROPERTY(props);
   SET_OCCA_PROPERTY(props, NUM_DOFS_1D);
   SET_OCCA_PROPERTY(props, NUM_QUAD_1D);
   SET_OCCA_PROPERTY(props, NUM_QUAD_2D);

   if (!config::usingGpu()){
      NEW_OCCA_KERNEL(MultAdd2D_CPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_CPU(numElements,
                    o_dofToQuad, o_dofToQuadD,
                    o_quadToDof, o_quadToDofD,
                    o_oper, o_solIn,
                    o_solOut);
   }else{
      NEW_OCCA_KERNEL(MultAdd2D_GPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_GPU(numElements,
                    o_dofToQuad, o_dofToQuadD,
                    o_quadToDof, o_quadToDofD,
                    o_oper, o_solIn,
                    o_solOut);
   }
}
#endif // __OCCA__

// *****************************************************************************
#define QUAD_2D_ID(X, Y) (X + ((Y) * NUM_QUAD_1D))
#define QUAD_3D_ID(X, Y, Z) (X + ((Y) * NUM_QUAD_1D) + ((Z) * NUM_QUAD_1D*NUM_QUAD_1D))

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void biPADiffusionMultAdd2D(const int numElements,
                            const double* __restrict dofToQuad,
                            const double* __restrict dofToQuadD,
                            const double* __restrict quadToDof,
                            const double* __restrict quadToDofD,
                            const double* __restrict oper,
                            const double* __restrict solIn,
                            double* __restrict solOut)
{
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e, numElements,
   {
      double grad[NUM_QUAD_1D][NUM_QUAD_1D][2];
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
      {
         double gradX[NUM_QUAD_1D][2];
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }

         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            const double s = solIn[ijkN(dx,dy,e,NUM_DOFS_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               gradX[qx][0] += s * dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
               gradX[qx][1] += s * dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
            }
         }

         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            const double wy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
            const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }

      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const int q = QUAD_2D_ID(qx, qy);

            const double O11 = oper[ijkNM(0,q,e,3,NUM_QUAD)];
            const double O12 = oper[ijkNM(1,q,e,3,NUM_QUAD)];
            const double O22 = oper[ijkNM(2,q,e,3,NUM_QUAD)];

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }

      for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
      {
         double gradX[NUM_DOFS_1D][2];
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }

         for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const double wx  = quadToDof[ijN(dx,qx,NUM_DOFS_1D)];
               const double wDx = quadToDofD[ijN(dx,qx,NUM_DOFS_1D)];
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }

         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            const double wy  = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
            const double wDy = quadToDofD[ijN(dy,qy,NUM_DOFS_1D)];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               solOut[ijkN(dx,dy,e,NUM_DOFS_1D)] += ((gradX[dx][0] * wy) +
                                                     (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void biPADiffusionMultAdd3D(const int numElements,
                            const double* __restrict dofToQuad,
                            const double* __restrict dofToQuadD,
                            const double* __restrict quadToDof,
                            const double* __restrict quadToDofD,
                            const double* __restrict oper,
                            const double* __restrict solIn,
                            double* __restrict solOut)
{
   const int NUM_QUAD = NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D;
   MFEM_FORALL(e, numElements,
   {
      double grad[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D][4];
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
      {
         double gradXY[NUM_QUAD_1D][NUM_QUAD_1D][4];
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            double gradX[NUM_QUAD_1D][2];
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const double s = solIn[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  gradX[qx][0] += s * dofToQuad[ijN(qx,dx,NUM_QUAD_1D)];
                  gradX[qx][1] += s * dofToQuadD[ijN(qx,dx,NUM_QUAD_1D)];
               }
            }
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               const double wy  = dofToQuad[ijN(qy,dy,NUM_QUAD_1D)];
               const double wDy = dofToQuadD[ijN(qy,dy,NUM_QUAD_1D)];
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
         {
            const double wz  = dofToQuad[ijN(qz,dz,NUM_QUAD_1D)];
            const double wDz = dofToQuadD[ijN(qz,dz,NUM_QUAD_1D)];
            for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
            {
               for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }

      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               const int q = QUAD_3D_ID(qx, qy, qz);
               const double O11 = oper[ijkNM(0,q,e,6,NUM_QUAD)];
               const double O12 = oper[ijkNM(1,q,e,6,NUM_QUAD)];
               const double O13 = oper[ijkNM(2,q,e,6,NUM_QUAD)];
               const double O22 = oper[ijkNM(3,q,e,6,NUM_QUAD)];
               const double O23 = oper[ijkNM(4,q,e,6,NUM_QUAD)];
               const double O33 = oper[ijkNM(5,q,e,6,NUM_QUAD)];

               const double gradX = grad[qz][qy][qx][0];
               const double gradY = grad[qz][qy][qx][1];
               const double gradZ = grad[qz][qy][qx][2];

               grad[qz][qy][qx][0] = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ);
               grad[qz][qy][qx][1] = (O12 * gradX) + (O22 * gradY) + (O23 * gradZ);
               grad[qz][qy][qx][2] = (O13 * gradX) + (O23 * gradY) + (O33 * gradZ);
            }
         }
      }

      for (int qz = 0; qz < NUM_QUAD_1D; ++qz)
      {
         double gradXY[NUM_DOFS_1D][NUM_DOFS_1D][4];
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }

         for (int qy = 0; qy < NUM_QUAD_1D; ++qy)
         {
            double gradX[NUM_DOFS_1D][4];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }

            for (int qx = 0; qx < NUM_QUAD_1D; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  const double wx  = quadToDof[ijN(dx,qx,NUM_DOFS_1D)];
                  const double wDx = quadToDofD[ijN(dx,qx,NUM_DOFS_1D)];
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }

            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               const double wy  = quadToDof[ijN(dy,qy,NUM_DOFS_1D)];
               const double wDy = quadToDofD[ijN(dy,qy,NUM_DOFS_1D)];
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }

         for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
         {
            const double wz  = quadToDof[ijN(dz,qz,NUM_DOFS_1D)];
            const double wDz = quadToDofD[ijN(dz,qz,NUM_DOFS_1D)];
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  solOut[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] +=
                     ((gradXY[dy][dx][0] * wz) +
                      (gradXY[dy][dx][1] * wz) +
                      (gradXY[dy][dx][2] * wDz));
               }
            }
         }
      }
   });
}

// *****************************************************************************
typedef void (*fDiffusionMultAdd)(const int numElements,
                                  const double* __restrict dofToQuad,
                                  const double* __restrict dofToQuadD,
                                  const double* __restrict quadToDof,
                                  const double* __restrict quadToDofD,
                                  const double* __restrict oper,
                                  const double* __restrict solIn,
                                  double* __restrict solOut);

// *****************************************************************************
template<const int DIM,
         const int NUM_DOFS_1D,
         const int NUM_QUAD_1D> static
void biPADiffusionMultAdd(const int numElements,
                          const double* __restrict dofToQuad,
                          const double* __restrict dofToQuadD,
                          const double* __restrict quadToDof,
                          const double* __restrict quadToDofD,
                          const double* __restrict oper,
                          const double* __restrict solIn,
                          double* __restrict solOut){
   fDiffusionMultAdd f = NULL;
   if (DIM==2) f = &biPADiffusionMultAdd2D<NUM_DOFS_1D,NUM_QUAD_1D>;
   if (DIM==3) f = &biPADiffusionMultAdd3D<NUM_DOFS_1D,NUM_QUAD_1D>;
   f(numElements,dofToQuad,dofToQuadD,quadToDof,quadToDofD,oper,solIn,solOut);
}

// *****************************************************************************
constexpr std::size_t NID = 5;
using Key_t = unsigned int;
using Kernel_t = fDiffusionMultAdd;
using Params_t = std::array<Key_t, 5>;
constexpr Params_t ids {{0x222, 0x233, 0x244, 0x323, 0x334}};
template<size_t I> constexpr Key_t GetKey() {
   return I==0?0x222:I==1?0x233:I==3?0x244:I==4?0x323:0x334;
}
template<typename Kernel_t, Key_t N> constexpr Kernel_t GetValue()
{
   return &biPADiffusionMultAdd<(N>>8)&0xF, (N>>4)&0xF, (N)&0xF>;
}
#include "../../general/instantiator.hpp"

// *****************************************************************************
void biPADiffusionMultAdd(const int DIM,
                          const int NUM_DOFS_1D,
                          const int NUM_QUAD_1D,
                          const int numElements,
                          const double* __restrict dofToQuad,
                          const double* __restrict dofToQuadD,
                          const double* __restrict quadToDof,
                          const double* __restrict quadToDofD,
                          const double* __restrict op,
                          const double* __restrict x,
                          double* __restrict y)
{

#ifdef __OCCA__
   if (config::usingOcca()){
      assert(DIM==2);
      oBiPADiffusionMultAdd2D(NUM_DOFS_1D, NUM_QUAD_1D,
                              numElements,
                              dofToQuad, dofToQuadD,
                              quadToDof, quadToDofD,
                              op, x, y);
      return;
   }
#endif // __OCCA__

   Instantiator<Params_t, NID, Key_t, Kernel_t> kernels(ids);
   const Key_t id = (DIM<<8)+(NUM_DOFS_1D<<4)+(NUM_QUAD_1D);
   const bool found = kernels.Find(id);
   if (!found){
      mfem_error("Kernel is not available!");
   }   
   GET_CONST_PTR(dofToQuad);
   GET_CONST_PTR(dofToQuadD);
   GET_CONST_PTR(quadToDof);
   GET_CONST_PTR(quadToDofD);
   GET_CONST_PTR(op);
   GET_CONST_PTR(x);
   GET_PTR(y);
   kernels.At(id)(numElements,
                  d_dofToQuad, d_dofToQuadD, d_quadToDof, d_quadToDofD,
                  d_op, d_x, d_y);
}

// *****************************************************************************
} // namespace fem
} // namespace kernels
} // namespace mfem
