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

// ****************************************************************************
// * OCCA 2D Assemble kernel
// *****************************************************************************
#ifdef __OCCA__
static void occaDiffusionAssemble2D(const int NUM_QUAD_1D,
                                    const int numElements,
                                    const double* __restrict quadWeights,
                                    const double* __restrict J,
                                    const double COEFF,
                                    double* __restrict oper)
{
   const int NUM_QUAD_2D = NUM_QUAD_1D*NUM_QUAD_1D;

   GET_OCCA_CONST_MEMORY(quadWeights);
   GET_OCCA_CONST_MEMORY(J);
   GET_OCCA_MEMORY(oper);

   NEW_OCCA_PROPERTY(props);
   SET_OCCA_PROPERTY(props, NUM_QUAD_1D);
   SET_OCCA_PROPERTY(props, NUM_QUAD_2D);

   NEW_OCCA_KERNEL(Assemble2D, fem, bidiffusionAssemble.okl, props);
   Assemble2D(numElements, o_quadWeights, o_J, COEFF, o_oper);
}
#endif // __OCCA__

// *****************************************************************************
// * Diffusion Assemble 2D kernel
// *****************************************************************************
static void DiffusionAssemble2D(const int NQ,
                                const int NE,
                                const double* __restrict w,
                                const double* __restrict j,
                                const double COEFF,
                                double* __restrict o)
{
   const dArray W(NQ, w);
   const dArray J(2,2,2,NQ,j);
   dArray oper(3,NQ,NE,o);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double c_detJ = W(q) * COEFF / ((J11*J22)-(J21*J12));
         oper(0,q,e) =  c_detJ * (J21*J21 + J22*J22);
         oper(1,q,e) = -c_detJ * (J21*J11 + J22*J12);
         oper(2,q,e) =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}

// *****************************************************************************
// * Diffusion Assemble 3D kernel
// *****************************************************************************
static void DiffusionAssemble3D(const int NQ,
                                const int NE,
                                const double* __restrict w,
                                const double* __restrict j,
                                const double COEFF,
                                double* __restrict y)
{
   const dArray W(NQ, w);
   const dArray J(3,3,3,NQ,j);
   dArray oper(6,NQ,NE,y);   
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J13 = J(2,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double J23 = J(2,1,q,e);
         const double J31 = J(0,2,q,e);
         const double J32 = J(1,2,q,e);
         const double J33 = J(2,2,q,e);
         const double detJ = ((J11 * J22 * J33) + (J12 * J23 * J31) +
         (J13 * J21 * J32) - (J13 * J22 * J31) -
         (J12 * J21 * J33) - (J11 * J23 * J32));
         const double c_detJ = W(q) * COEFF / detJ;
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
         oper(0,q,e) = c_detJ * (A11*A11 + A21*A21 + A31*A31);
         oper(1,q,e) = c_detJ * (A11*A12 + A21*A22 + A31*A32);
         oper(2,q,e) = c_detJ * (A11*A13 + A21*A23 + A31*A33);
         oper(3,q,e) = c_detJ * (A12*A12 + A22*A22 + A32*A32);
         oper(4,q,e) = c_detJ * (A12*A13 + A22*A23 + A32*A33);
         oper(5,q,e) = c_detJ * (A13*A13 + A23*A23 + A33*A33);
      }
   });
}

// *****************************************************************************
void DiffusionAssemble(const int dim,
                       const int NQ1d,
                       const int NE,
                       const double* __restrict quadWeights,
                       const double* __restrict J,
                       const double COEFF,
                       double* __restrict oper)
{
   if (dim==1) { assert(false); }
   if (dim==2)
   {
#ifdef __OCCA__
      if (config::usingOcca())
      {
         occaDiffusionAssemble2D(NQ1d*NQ1d, NE,
                                 quadWeights, J, COEFF, oper);
         return;
      }
#endif // __OCCA__
      DiffusionAssemble2D(NQ1d*NQ1d, NE,
                          quadWeights, J, COEFF, oper);
   }
   if (dim==3)
   {
      DiffusionAssemble3D(NQ1d*NQ1d*NQ1d, NE,
                          quadWeights, J, COEFF, oper);
   }
}

#ifdef __OCCA__
// *****************************************************************************
static void occaDiffusionMultAdd2D(const int NUM_DOFS_1D,
                                   const int NQ1d,
                                   const int NE,
                                   const double* __restrict B,
                                   const double* __restrict dofToQuadD,
                                   const double* __restrict Bt,
                                   const double* __restrict quadToDofD,
                                   const double* __restrict oper,
                                   const double* __restrict solIn,
                                   double* __restrict y)
{
   const int NUM_QUAD_2D = NQ1d*NQ1d;

   GET_OCCA_CONST_MEMORY(B);
   GET_OCCA_CONST_MEMORY(dofToQuadD);
   GET_OCCA_CONST_MEMORY(Bt);
   GET_OCCA_CONST_MEMORY(quadToDofD);
   GET_OCCA_CONST_MEMORY(oper);
   GET_OCCA_CONST_MEMORY(solIn);
   GET_OCCA_MEMORY(y);

   NEW_OCCA_PROPERTY(props);
   SET_OCCA_PROPERTY(props, NUM_DOFS_1D);
   SET_OCCA_PROPERTY(props, NQ1d);
   SET_OCCA_PROPERTY(props, NUM_QUAD_2D);

   if (!config::usingGpu())
   {
      NEW_OCCA_KERNEL(MultAdd2D_CPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_CPU(NE,
                    o_B, o_dofToQuadD,
                    o_Bt, o_quadToDofD,
                    o_oper, o_solIn,
                    o_y);
   }
   else
   {
      NEW_OCCA_KERNEL(MultAdd2D_GPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_GPU(NE,
                    o_B, o_dofToQuadD,
                    o_Bt, o_quadToDofD,
                    o_oper, o_solIn,
                    o_y);
   }
}
#endif // __OCCA__

// *****************************************************************************
#define QUAD_2D_ID(X, Y) (X + ((Y) * NQ1d))
#define QUAD_3D_ID(X, Y, Z) (X + ((Y) * NQ1d) + ((Z) * NQ1d*NQ1d))

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NQ1d> static
void DiffusionMultAssembled2D(const int NE,
                              const double* __restrict B,
                              const double* __restrict G,
                              const double* __restrict Bt,
                              const double* __restrict Gt,
                              const double* __restrict oper,
                              const double* __restrict x,
                              double* __restrict y)
{
   const int NUM_QUAD = NQ1d*NQ1d;
   MFEM_FORALL(e, NE,
   {
      double grad[NQ1d][NQ1d][2];
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            grad[qy][qx][0] = 0.0;
            grad[qy][qx][1] = 0.0;
         }
      }

      for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
      {
         double gradX[NQ1d][2];
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }

         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            const double s = x[ijkN(dx,dy,e,NUM_DOFS_1D)];
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               gradX[qx][0] += s * B[ijN(qx,dx,NQ1d)];
               gradX[qx][1] += s * G[ijN(qx,dx,NQ1d)];
            }
         }

         for (int qy = 0; qy < NQ1d; ++qy)
         {
            const double wy  = B[ijN(qy,dy,NQ1d)];
            const double wDy = G[ijN(qy,dy,NQ1d)];
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               grad[qy][qx][0] += gradX[qx][1] * wy;
               grad[qy][qx][1] += gradX[qx][0] * wDy;
            }
         }
      }

      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         for (int qx = 0; qx < NQ1d; ++qx)
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

      for (int qy = 0; qy < NQ1d; ++qy)
      {
         double gradX[NUM_DOFS_1D][2];
         for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }

         for (int qx = 0; qx < NQ1d; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const double wx  = Bt[ijN(dx,qx,NUM_DOFS_1D)];
               const double wDx = Gt[ijN(dx,qx,NUM_DOFS_1D)];
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }

         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            const double wy  = Bt[ijN(dy,qy,NUM_DOFS_1D)];
            const double wDy = Gt[ijN(dy,qy,NUM_DOFS_1D)];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               y[ijkN(dx,dy,e,NUM_DOFS_1D)] += ((gradX[dx][0] * wy) +
                                                     (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// *****************************************************************************
template<const int NUM_DOFS_1D,
         const int NQ1d> static
void DiffusionMultAssembled3D(const int NE,
                              const double* __restrict B,
                              const double* __restrict G,
                              const double* __restrict Bt,
                              const double* __restrict Gt,
                              const double* __restrict oper,
                              const double* __restrict x,
                              double* __restrict y)
{
   const int NUM_QUAD = NQ1d*NQ1d*NQ1d;
   MFEM_FORALL(e, NE,
   {
      double grad[NQ1d][NQ1d][NQ1d][4];
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               grad[qz][qy][qx][0] = 0.0;
               grad[qz][qy][qx][1] = 0.0;
               grad[qz][qy][qx][2] = 0.0;
            }
         }
      }
      for (int dz = 0; dz < NUM_DOFS_1D; ++dz)
      {
         double gradXY[NQ1d][NQ1d][4];
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               gradXY[qy][qx][0] = 0.0;
               gradXY[qy][qx][1] = 0.0;
               gradXY[qy][qx][2] = 0.0;
            }
         }
         for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
         {
            double gradX[NQ1d][2];
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               const double s = x[ijklN(dx,dy,dz,e,NUM_DOFS_1D)];
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  gradX[qx][0] += s * B[ijN(qx,dx,NQ1d)];
                  gradX[qx][1] += s * G[ijN(qx,dx,NQ1d)];
               }
            }
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               const double wy  = B[ijN(qy,dy,NQ1d)];
               const double wDy = G[ijN(qy,dy,NQ1d)];
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  const double wx  = gradX[qx][0];
                  const double wDx = gradX[qx][1];
                  gradXY[qy][qx][0] += wDx * wy;
                  gradXY[qy][qx][1] += wx  * wDy;
                  gradXY[qy][qx][2] += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < NQ1d; ++qz)
         {
            const double wz  = B[ijN(qz,dz,NQ1d)];
            const double wDz = G[ijN(qz,dz,NQ1d)];
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  grad[qz][qy][qx][0] += gradXY[qy][qx][0] * wz;
                  grad[qz][qy][qx][1] += gradXY[qy][qx][1] * wz;
                  grad[qz][qy][qx][2] += gradXY[qy][qx][2] * wDz;
               }
            }
         }
      }

      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < NQ1d; ++qz)
      {
         for (int qy = 0; qy < NQ1d; ++qy)
         {
            for (int qx = 0; qx < NQ1d; ++qx)
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

      for (int qz = 0; qz < NQ1d; ++qz)
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

         for (int qy = 0; qy < NQ1d; ++qy)
         {
            double gradX[NUM_DOFS_1D][4];
            for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
            {
               gradX[dx][0] = 0;
               gradX[dx][1] = 0;
               gradX[dx][2] = 0;
            }

            for (int qx = 0; qx < NQ1d; ++qx)
            {
               const double gX = grad[qz][qy][qx][0];
               const double gY = grad[qz][qy][qx][1];
               const double gZ = grad[qz][qy][qx][2];
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  const double wx  = Bt[ijN(dx,qx,NUM_DOFS_1D)];
                  const double wDx = Gt[ijN(dx,qx,NUM_DOFS_1D)];
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }

            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               const double wy  = Bt[ijN(dy,qy,NUM_DOFS_1D)];
               const double wDy = Gt[ijN(dy,qy,NUM_DOFS_1D)];
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
            const double wz  = Bt[ijN(dz,qz,NUM_DOFS_1D)];
            const double wDz = Gt[ijN(dz,qz,NUM_DOFS_1D)];
            for (int dy = 0; dy < NUM_DOFS_1D; ++dy)
            {
               for (int dx = 0; dx < NUM_DOFS_1D; ++dx)
               {
                  y[ijklN(dx,dy,dz,e,NUM_DOFS_1D)] +=
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
typedef void (*fDiffusionMultAdd)(const int NE,
                                  const double* __restrict B,
                                  const double* __restrict G,
                                  const double* __restrict Bt,
                                  const double* __restrict Gt,
                                  const double* __restrict oper,
                                  const double* __restrict x,
                                  double* __restrict y);

// *****************************************************************************
void DiffusionMultAssembled(const int DIM,
                            const int NUM_DOFS_1D,
                            const int NQ1d,
                            const int NE,
                            const double* __restrict B,
                            const double* __restrict G,
                            const double* __restrict Bt,
                            const double* __restrict Gt,
                            const double* __restrict op,
                            const double* __restrict x,
                            double* __restrict y)
{

#ifdef __OCCA__
   if (config::usingOcca())
   {
      assert(DIM==2);
      occaDiffusionMultAssembled2D(NUM_DOFS_1D, NQ1d,
                                   NE,
                                   B, G,
                                   Bt, Gt,
                                   op, x, y);
      return;
   }
#endif // __OCCA__

   const unsigned int id = (DIM<<16)|(NUM_DOFS_1D<<8)|(NQ1d);
   assert(LOG2(NUM_DOFS_1D)<=8);
   assert(LOG2(NQ1d)<=8);
   static std::unordered_map<unsigned int, fDiffusionMultAdd> call =
   {
      //{0x20101,&DiffusionMultAssembled2D<1,1>},
      //{0x20201,&DiffusionMultAssembled2D<2,1>},
      {0x20202,&DiffusionMultAssembled2D<2,2>},/*
      {0x20303,&DiffusionMultAssembled2D<3,3>},
      {0x20404,&DiffusionMultAssembled2D<4,4>},
      {0x20505,&DiffusionMultAssembled2D<5,5>},
      {0x20606,&DiffusionMultAssembled2D<6,6>},
      {0x20707,&DiffusionMultAssembled2D<7,7>},
      {0x20808,&DiffusionMultAssembled2D<8,8>},
                                               *//*
      {0x20909,&DiffusionMultAssembled2D<9,9>},
      {0x20A0A,&DiffusionMultAssembled2D<10,10>},
      {0x20B0B,&DiffusionMultAssembled2D<11,11>},
      {0x20C0C,&DiffusionMultAssembled2D<12,12>},
      {0x20D0D,&DiffusionMultAssembled2D<13,13>},
      {0x20E0E,&DiffusionMultAssembled2D<14,14>},
      {0x20F0F,&DiffusionMultAssembled2D<15,15>},
      {0x21010,&DiffusionMultAssembled2D<16,16>},
      {0x21111,&DiffusionMultAssembled2D<17,17>},*/

      /*
      {0x30101,&DiffusionMultAssembled3D<1,1>},
      {0x30201,&DiffusionMultAssembled3D<2,1>},
      {0x30202,&DiffusionMultAssembled3D<2,2>},*/
      {0x30203,&DiffusionMultAssembled3D<2,3>},/*
      {0x30303,&DiffusionMultAssembled3D<3,3>},
      {0x30404,&DiffusionMultAssembled3D<4,4>},
      {0x30505,&DiffusionMultAssembled3D<5,5>},
      {0x30606,&DiffusionMultAssembled3D<6,6>},
      {0x30707,&DiffusionMultAssembled3D<7,7>},
      {0x30808,&DiffusionMultAssembled3D<8,8>},
      */
/*
      {0x30909,&DiffusionMultAssembled3D<9,9>},
      {0x30A0A,&DiffusionMultAssembled3D<10,10>},
      {0x30B0B,&DiffusionMultAssembled3D<11,11>},
      {0x30C0C,&DiffusionMultAssembled3D<12,12>},
      {0x30D0D,&DiffusionMultAssembled3D<13,13>},
      {0x30E0E,&DiffusionMultAssembled3D<14,14>},
      {0x30F0F,&DiffusionMultAssembled3D<15,15>},
      {0x31010,&DiffusionMultAssembled3D<16,16>},*/
   };
   if (!call[id])
   {
      printf("\n[kIntDiffusionMultAdd] id \033[33m0x%X\033[m ",id);
      fflush(stdout);
   }
   assert(call[id]);

   GET_CONST_PTR(B);
   GET_CONST_PTR(G);
   GET_CONST_PTR(Bt);
   GET_CONST_PTR(Gt);
   GET_CONST_PTR(op);
   GET_CONST_PTR(x);
   GET_PTR(y);

   call[id](NE,
            d_B, d_G, d_Bt, d_Gt,
            d_op, d_x, d_y);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
