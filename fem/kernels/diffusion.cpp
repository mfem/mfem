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
#include "../../linalg/device.hpp"

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
static void occaDiffusionAssemble2D(const int NQ1d,
                                    const int NE,
                                    const double* __restrict quadWeights,
                                    const double* __restrict J,
                                    const double COEFF,
                                    double* __restrict oper)
{
   const int NUM_QUAD_2D = NQ1d*NQ1d;

   GET_OCCA_CONST_MEMORY(quadWeights);
   GET_OCCA_CONST_MEMORY(J);
   GET_OCCA_MEMORY(oper);

   NEW_OCCA_PROPERTY(props);
   SET_OCCA_PROPERTY(props, NQ1d);
   SET_OCCA_PROPERTY(props, NUM_QUAD_2D);

   NEW_OCCA_KERNEL(Assemble2D, fem, bidiffusionAssemble.okl, props);
   Assemble2D(NE, o_quadWeights, o_J, COEFF, o_oper);
}
#endif // __OCCA__

// *****************************************************************************
// * Diffusion Assemble 2D kernel
// *****************************************************************************
static void DiffusionAssemble2D(const int NQ1d,
                                const int NE,
                                const double* __restrict w,
                                const double* __restrict j,
                                const double COEFF,
                                double* __restrict op)
{
   const int NQ = NQ1d*NQ1d;
   const DeviceVector W(w, NQ);
   const DeviceTensor<4> J(j, 2, 2, NQ, NE);
   DeviceTensor<3> y(op, 3, NQ, NE);
   MFEM_FORALL(e, NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = J(0,0,q,e);
         const double J12 = J(1,0,q,e);
         const double J21 = J(0,1,q,e);
         const double J22 = J(1,1,q,e);
         const double c_detJ = W(q) * COEFF / ((J11*J22)-(J21*J12));
         y(0,q,e) =  c_detJ * (J21*J21 + J22*J22);
         y(1,q,e) = -c_detJ * (J21*J11 + J22*J12);
         y(2,q,e) =  c_detJ * (J11*J11 + J12*J12);
      }
   });
}

// *****************************************************************************
// * Diffusion Assemble 3D kernel
// *****************************************************************************
static void DiffusionAssemble3D(const int NQ1d,
                                const int NE,
                                const double* __restrict quadWeights,
                                const double* __restrict J,
                                const double COEFF,
                                double* __restrict oper)
{
   GET_CONST_PTR(quadWeights);
   GET_CONST_PTR(J);
   GET_PTR(oper);
   const int NQ = NQ1d*NQ1d*NQ1d;
   MFEM_FORALL(e,NE,
   {
      for (int q = 0; q < NQ; ++q)
      {
         const double J11 = d_J[ijklNM(0,0,q,e,3,NQ)];
         const double J12 = d_J[ijklNM(1,0,q,e,3,NQ)];
         const double J13 = d_J[ijklNM(2,0,q,e,3,NQ)];
         const double J21 = d_J[ijklNM(0,1,q,e,3,NQ)];
         const double J22 = d_J[ijklNM(1,1,q,e,3,NQ)];
         const double J23 = d_J[ijklNM(2,1,q,e,3,NQ)];
         const double J31 = d_J[ijklNM(0,2,q,e,3,NQ)];
         const double J32 = d_J[ijklNM(1,2,q,e,3,NQ)];
         const double J33 = d_J[ijklNM(2,2,q,e,3,NQ)];
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
         d_oper[ijkNM(0,q,e,6,NQ)] = c_detJ *
         (A11*A11 + A21*A21 + A31*A31); // (1,1)
         d_oper[ijkNM(1,q,e,6,NQ)] = c_detJ *
         (A11*A12 + A21*A22 + A31*A32); // (1,2), (2,1)
         d_oper[ijkNM(2,q,e,6,NQ)] = c_detJ *
         (A11*A13 + A21*A23 + A31*A33); // (1,3), (3,1)
         d_oper[ijkNM(3,q,e,6,NQ)] = c_detJ *
         (A12*A12 + A22*A22 + A32*A32); // (2,2)
         d_oper[ijkNM(4,q,e,6,NQ)] = c_detJ *
         (A12*A13 + A22*A23 + A32*A33); // (2,3), (3,2)
         d_oper[ijkNM(5,q,e,6,NQ)] = c_detJ *
         (A13*A13 + A23*A23 + A33*A33); // (3,3)
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
         occaDiffusionAssemble2D(NQ1d, NE,
                                 quadWeights, J, COEFF, oper);
         return;
      }
#endif // __OCCA__
      DiffusionAssemble2D(NQ1d, NE,
                          quadWeights, J, COEFF, oper);
   }
   if (dim==3)
   {
      DiffusionAssemble3D(NQ1d, NE,
                          quadWeights, J, COEFF, oper);
   }
}

#ifdef __OCCA__
// *****************************************************************************
static void occaDiffusionMultAdd2D(const int ND1d,
                                   const int NQ1d,
                                   const int NE,
                                   const double* __restrict dofToQuad,
                                   const double* __restrict dofToQuadD,
                                   const double* __restrict quadToDof,
                                   const double* __restrict quadToDofD,
                                   const double* __restrict oper,
                                   const double* __restrict solIn,
                                   double* __restrict solOut)
{
   const int NUM_QUAD_2D = NQ1d*NQ1d;

   GET_OCCA_CONST_MEMORY(dofToQuad);
   GET_OCCA_CONST_MEMORY(dofToQuadD);
   GET_OCCA_CONST_MEMORY(quadToDof);
   GET_OCCA_CONST_MEMORY(quadToDofD);
   GET_OCCA_CONST_MEMORY(oper);
   GET_OCCA_CONST_MEMORY(solIn);
   GET_OCCA_MEMORY(solOut);

   NEW_OCCA_PROPERTY(props);
   SET_OCCA_PROPERTY(props, ND1d);
   SET_OCCA_PROPERTY(props, NQ1d);
   SET_OCCA_PROPERTY(props, NUM_QUAD_2D);

   if (!config::usingGpu())
   {
      NEW_OCCA_KERNEL(MultAdd2D_CPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_CPU(NE,
                    o_dofToQuad, o_dofToQuadD,
                    o_quadToDof, o_quadToDofD,
                    o_oper, o_solIn,
                    o_solOut);
   }
   else
   {
      NEW_OCCA_KERNEL(MultAdd2D_GPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_GPU(NE,
                    o_dofToQuad, o_dofToQuadD,
                    o_quadToDof, o_quadToDofD,
                    o_oper, o_solIn,
                    o_solOut);
   }
}
#endif // __OCCA__

// *****************************************************************************
#define QUAD_2D_ID(X, Y) (X + ((Y) * NQ1d))
#define QUAD_3D_ID(X, Y, Z) (X + ((Y) * NQ1d) + ((Z) * NQ1d*NQ1d))

// *****************************************************************************
template<const int ND1d,
         const int NQ1d> static
void DiffusionMultAssembled2D(const int NE,
                              const double* __restrict dofToQuad,
                              const double* __restrict dofToQuadD,
                              const double* __restrict quadToDof,
                              const double* __restrict quadToDofD,
                              const double* __restrict oper,
                              const double* __restrict solIn,
                              double* __restrict solOut)
{
   const int NQ = NQ1d*NQ1d;
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

      for (int dy = 0; dy < ND1d; ++dy)
      {
         double gradX[NQ1d][2];
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            gradX[qx][0] = 0.0;
            gradX[qx][1] = 0.0;
         }

         for (int dx = 0; dx < ND1d; ++dx)
         {
            const double s = solIn[ijkN(dx,dy,e,ND1d)];
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               gradX[qx][0] += s * dofToQuad[ijN(qx,dx,NQ1d)];
               gradX[qx][1] += s * dofToQuadD[ijN(qx,dx,NQ1d)];
            }
         }

         for (int qy = 0; qy < NQ1d; ++qy)
         {
            const double wy  = dofToQuad[ijN(qy,dy,NQ1d)];
            const double wDy = dofToQuadD[ijN(qy,dy,NQ1d)];
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

            const double O11 = oper[ijkNM(0,q,e,3,NQ)];
            const double O12 = oper[ijkNM(1,q,e,3,NQ)];
            const double O22 = oper[ijkNM(2,q,e,3,NQ)];

            const double gradX = grad[qy][qx][0];
            const double gradY = grad[qy][qx][1];

            grad[qy][qx][0] = (O11 * gradX) + (O12 * gradY);
            grad[qy][qx][1] = (O12 * gradX) + (O22 * gradY);
         }
      }

      for (int qy = 0; qy < NQ1d; ++qy)
      {
         double gradX[ND1d][2];
         for (int dx = 0; dx < ND1d; ++dx)
         {
            gradX[dx][0] = 0;
            gradX[dx][1] = 0;
         }

         for (int qx = 0; qx < NQ1d; ++qx)
         {
            const double gX = grad[qy][qx][0];
            const double gY = grad[qy][qx][1];
            for (int dx = 0; dx < ND1d; ++dx)
            {
               const double wx  = quadToDof[ijN(dx,qx,ND1d)];
               const double wDx = quadToDofD[ijN(dx,qx,ND1d)];
               gradX[dx][0] += gX * wDx;
               gradX[dx][1] += gY * wx;
            }
         }

         for (int dy = 0; dy < ND1d; ++dy)
         {
            const double wy  = quadToDof[ijN(dy,qy,ND1d)];
            const double wDy = quadToDofD[ijN(dy,qy,ND1d)];
            for (int dx = 0; dx < ND1d; ++dx)
            {
               solOut[ijkN(dx,dy,e,ND1d)] += ((gradX[dx][0] * wy) +
                                                     (gradX[dx][1] * wDy));
            }
         }
      }
   });
}

// *****************************************************************************
template<const int ND1d,
         const int NQ1d> static
void DiffusionMultAssembled3D(const int NE,
                              const double* __restrict dofToQuad,
                              const double* __restrict dofToQuadD,
                              const double* __restrict quadToDof,
                              const double* __restrict quadToDofD,
                              const double* __restrict oper,
                              const double* __restrict solIn,
                              double* __restrict solOut)
{
   const int NQ = NQ1d*NQ1d*NQ1d;
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
      for (int dz = 0; dz < ND1d; ++dz)
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
         for (int dy = 0; dy < ND1d; ++dy)
         {
            double gradX[NQ1d][2];
            for (int qx = 0; qx < NQ1d; ++qx)
            {
               gradX[qx][0] = 0.0;
               gradX[qx][1] = 0.0;
            }
            for (int dx = 0; dx < ND1d; ++dx)
            {
               const double s = solIn[ijklN(dx,dy,dz,e,ND1d)];
               for (int qx = 0; qx < NQ1d; ++qx)
               {
                  gradX[qx][0] += s * dofToQuad[ijN(qx,dx,NQ1d)];
                  gradX[qx][1] += s * dofToQuadD[ijN(qx,dx,NQ1d)];
               }
            }
            for (int qy = 0; qy < NQ1d; ++qy)
            {
               const double wy  = dofToQuad[ijN(qy,dy,NQ1d)];
               const double wDy = dofToQuadD[ijN(qy,dy,NQ1d)];
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
            const double wz  = dofToQuad[ijN(qz,dz,NQ1d)];
            const double wDz = dofToQuadD[ijN(qz,dz,NQ1d)];
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
               const double O11 = oper[ijkNM(0,q,e,6,NQ)];
               const double O12 = oper[ijkNM(1,q,e,6,NQ)];
               const double O13 = oper[ijkNM(2,q,e,6,NQ)];
               const double O22 = oper[ijkNM(3,q,e,6,NQ)];
               const double O23 = oper[ijkNM(4,q,e,6,NQ)];
               const double O33 = oper[ijkNM(5,q,e,6,NQ)];

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
         double gradXY[ND1d][ND1d][4];
         for (int dy = 0; dy < ND1d; ++dy)
         {
            for (int dx = 0; dx < ND1d; ++dx)
            {
               gradXY[dy][dx][0] = 0;
               gradXY[dy][dx][1] = 0;
               gradXY[dy][dx][2] = 0;
            }
         }

         for (int qy = 0; qy < NQ1d; ++qy)
         {
            double gradX[ND1d][4];
            for (int dx = 0; dx < ND1d; ++dx)
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
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  const double wx  = quadToDof[ijN(dx,qx,ND1d)];
                  const double wDx = quadToDofD[ijN(dx,qx,ND1d)];
                  gradX[dx][0] += gX * wDx;
                  gradX[dx][1] += gY * wx;
                  gradX[dx][2] += gZ * wx;
               }
            }

            for (int dy = 0; dy < ND1d; ++dy)
            {
               const double wy  = quadToDof[ijN(dy,qy,ND1d)];
               const double wDy = quadToDofD[ijN(dy,qy,ND1d)];
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  gradXY[dy][dx][0] += gradX[dx][0] * wy;
                  gradXY[dy][dx][1] += gradX[dx][1] * wDy;
                  gradXY[dy][dx][2] += gradX[dx][2] * wy;
               }
            }
         }

         for (int dz = 0; dz < ND1d; ++dz)
         {
            const double wz  = quadToDof[ijN(dz,qz,ND1d)];
            const double wDz = quadToDofD[ijN(dz,qz,ND1d)];
            for (int dy = 0; dy < ND1d; ++dy)
            {
               for (int dx = 0; dx < ND1d; ++dx)
               {
                  solOut[ijklN(dx,dy,dz,e,ND1d)] +=
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
                                  const double* __restrict dofToQuad,
                                  const double* __restrict dofToQuadD,
                                  const double* __restrict quadToDof,
                                  const double* __restrict quadToDofD,
                                  const double* __restrict oper,
                                  const double* __restrict solIn,
                                  double* __restrict solOut);

// *****************************************************************************
void DiffusionMultAssembled(const int DIM,
                            const int ND1d,
                            const int NQ1d,
                            const int NE,
                            const double* __restrict dofToQuad,
                            const double* __restrict dofToQuadD,
                            const double* __restrict quadToDof,
                            const double* __restrict quadToDofD,
                            const double* __restrict op,
                            const double* __restrict x,
                            double* __restrict y)
{

#ifdef __OCCA__
   if (config::usingOcca())
   {
      assert(DIM==2);
      occaDiffusionMultAssembled2D(ND1d, NQ1d,
                                   NE,
                                   dofToQuad, dofToQuadD,
                                   quadToDof, quadToDofD,
                                   op, x, y);
      return;
   }
#endif // __OCCA__

   const unsigned int id = (DIM<<16)|(ND1d<<8)|(NQ1d);
   assert(LOG2(ND1d)<=8);
   assert(LOG2(NQ1d)<=8);
   static std::unordered_map<unsigned int, fDiffusionMultAdd> call =
   {
      {0x20101,&DiffusionMultAssembled2D<1,1>},
      {0x20201,&DiffusionMultAssembled2D<2,1>},
      {0x20202,&DiffusionMultAssembled2D<2,2>},
      {0x20303,&DiffusionMultAssembled2D<3,3>},
      {0x20404,&DiffusionMultAssembled2D<4,4>},
      {0x20505,&DiffusionMultAssembled2D<5,5>},
      {0x20606,&DiffusionMultAssembled2D<6,6>},
      {0x20707,&DiffusionMultAssembled2D<7,7>},
      {0x20808,&DiffusionMultAssembled2D<8,8>},/*
      {0x20909,&DiffusionMultAssembled2D<9,9>},
      {0x20A0A,&DiffusionMultAssembled2D<10,10>},
      {0x20B0B,&DiffusionMultAssembled2D<11,11>},
      {0x20C0C,&DiffusionMultAssembled2D<12,12>},
      {0x20D0D,&DiffusionMultAssembled2D<13,13>},
      {0x20E0E,&DiffusionMultAssembled2D<14,14>},
      {0x20F0F,&DiffusionMultAssembled2D<15,15>},
      {0x21010,&DiffusionMultAssembled2D<16,16>},
      {0x21111,&DiffusionMultAssembled2D<17,17>},*/

      {0x30101,&DiffusionMultAssembled3D<1,1>},
      {0x30201,&DiffusionMultAssembled3D<2,1>},
      {0x30202,&DiffusionMultAssembled3D<2,2>},
      {0x30203,&DiffusionMultAssembled3D<2,3>},
      {0x30303,&DiffusionMultAssembled3D<3,3>},
      {0x30404,&DiffusionMultAssembled3D<4,4>},
      {0x30505,&DiffusionMultAssembled3D<5,5>},
      {0x30606,&DiffusionMultAssembled3D<6,6>},
      {0x30707,&DiffusionMultAssembled3D<7,7>},
      {0x30808,&DiffusionMultAssembled3D<8,8>},/*
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

   GET_CONST_PTR(dofToQuad);
   GET_CONST_PTR(dofToQuadD);
   GET_CONST_PTR(quadToDof);
   GET_CONST_PTR(quadToDofD);
   GET_CONST_PTR(op);
   GET_CONST_PTR(x);
   GET_PTR(y);

   call[id](NE,
            d_dofToQuad, d_dofToQuadD, d_quadToDof, d_quadToDofD,
            d_op, d_x, d_y);
}

} // namespace fem
} // namespace kernels
} // namespace mfem
