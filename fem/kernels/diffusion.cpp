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
   const Vector W(NQ, w);
   const Vector J(2,2,2,NQ,j);
   Vector oper(3,NQ,NE,o);
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
   const Vector W(NQ, w);
   const Vector J(3,3,3,NQ,j);
   Vector oper(6,NQ,NE,y);
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
         occaDiffusionAssemble2D(NQ1d*NQ1d, NE, quadWeights, J, COEFF, oper);
         return;
      }
#endif // __OCCA__
      DiffusionAssemble2D(NQ1d*NQ1d, NE, quadWeights, J, COEFF, oper);
   }
   if (dim==3)
   {
      DiffusionAssemble3D(NQ1d*NQ1d*NQ1d, NE, quadWeights, J, COEFF, oper);
   }
}

#ifdef __OCCA__
// *****************************************************************************
static void occaDiffusionMultAdd2D(const int ND1d,
                                   const int NQ1d,
                                   const int NE,
                                   const double* __restrict B,
                                   const double* __restrict G,
                                   const double* __restrict Bt,
                                   const double* __restrict Gt,
                                   const double* __restrict oper,
                                   const double* __restrict solIn,
                                   double* __restrict y)
{
   const int NUM_QUAD_2D = NQ1d*NQ1d;

   GET_OCCA_CONST_MEMORY(B);
   GET_OCCA_CONST_MEMORY(G);
   GET_OCCA_CONST_MEMORY(Bt);
   GET_OCCA_CONST_MEMORY(Gt);
   GET_OCCA_CONST_MEMORY(oper);
   GET_OCCA_CONST_MEMORY(solIn);
   GET_OCCA_MEMORY(y);

   NEW_OCCA_PROPERTY(props);
   SET_OCCA_PROPERTY(props, ND1d);
   SET_OCCA_PROPERTY(props, NQ1d);
   SET_OCCA_PROPERTY(props, NUM_QUAD_2D);

   if (!config::usingGpu())
   {
      NEW_OCCA_KERNEL(MultAdd2D_CPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_CPU(NE,
                    o_B, o_G,
                    o_Bt, o_Gt,
                    o_oper, o_solIn,
                    o_y);
   }
   else
   {
      NEW_OCCA_KERNEL(MultAdd2D_GPU, fem, bidiffusionMultAdd.okl, props);
      MultAdd2D_GPU(NE,
                    o_B, o_G,
                    o_Bt, o_Gt,
                    o_oper, o_solIn,
                    o_y);
   }
}
#endif // __OCCA__

// *****************************************************************************
#define QUAD_2D_ID(X, Y) (X + ((Y) * NQ1d))
#define QUAD_3D_ID(X, Y, Z) (X + ((Y) * NQ1d) + ((Z) * NQ1d*NQ1d))

// *****************************************************************************
static void DiffusionMultAssembled2D(const int ND1d,
                                     const int NQ1d,
                                     const int NE,
                                     const double* __restrict _B,
                                     const double* __restrict _G,
                                     const double* __restrict _Bt,
                                     const double* __restrict _Gt,
                                     const double* __restrict _op,
                                     const double* __restrict _x,
                                     double* __restrict _y)
{
   const int NQ = NQ1d * NQ1d;
   const int maxDQ = max(NQ1d, ND1d);
   const int Nspt = 2*(NQ + maxDQ);

   const Vector B(NQ1d,ND1d,_B);
   const Vector G(NQ1d,ND1d,_G);
   const Vector Bt(ND1d,NQ1d,_Bt);
   const Vector Gt(ND1d,NQ1d,_Gt);
   const Vector op(3,NQ,_op);
   const Vector x(ND1d,ND1d,_x);
   Vector y(ND1d,ND1d,_y);

   MFEM_FORALL_SHARED(e, NE, Nspt,
   {
      Vector3 grad(NQ1d, NQ1d, 2, __shared);
      for (int qy = 0; qy < NQ1d; ++qy) {
         for (int qx = 0; qx < NQ1d; ++qx) {
            grad(qy,qx,0) = 0.0;
            grad(qy,qx,1) = 0.0;
         }
      }
      for (int dy = 0; dy < ND1d; ++dy) {
         Vector2 gradX(NQ1d, 2, __shared + 2*NQ);
         for (int qx = 0; qx < NQ1d; ++qx)
         {
            gradX(qx,0) = 0.0;
            gradX(qx,1) = 0.0;
         }
         for (int dx = 0; dx < ND1d; ++dx) {
            const double s = x(dx,dy,e);
            for (int qx = 0; qx < NQ1d; ++qx) {
               gradX(qx,0) += s * B(qx,dx);
               gradX(qx,1) += s * G(qx,dx);
            }
         }
         for (int qy = 0; qy < NQ1d; ++qy) {
            const double wy  = B(qy,dy);
            const double wDy = G(qy,dy);
            for (int qx = 0; qx < NQ1d; ++qx) {
               grad(qy,qx,0) += gradX(qx,1) * wy;
               grad(qy,qx,1) += gradX(qx,0) * wDy;
            }
         }
      }
      // Calculate Dxy, xDy in plane
      for (int qy = 0; qy < NQ1d; ++qy) {
         for (int qx = 0; qx < NQ1d; ++qx) {
            const int q = QUAD_2D_ID(qx, qy);
            const double O11 = op(0,q,e);
            const double O12 = op(1,q,e);
            const double O22 = op(2,q,e);
            const double gradX = grad(qy,qx,0);
            const double gradY = grad(qy,qx,1);
            grad(qy,qx,0) = (O11 * gradX) + (O12 * gradY);
            grad(qy,qx,1) = (O12 * gradX) + (O22 * gradY);
         }
      }
      for (int qy = 0; qy < NQ1d; ++qy)
      {
         Vector2 gradX(ND1d, 2, __shared + 2*NQ);
         for (int dx = 0; dx < ND1d; ++dx) {
            gradX(dx,0) = 0.0;
            gradX(dx,1) = 0.0;
         }
         for (int qx = 0; qx < NQ1d; ++qx) {
            const double gX = grad(qy,qx,0);
            const double gY = grad(qy,qx,1);
            for (int dx = 0; dx < ND1d; ++dx) {
               const double wx  = Bt(dx,qx);
               const double wDx = Gt(dx,qx);
               gradX(dx,0) += gX * wDx;
               gradX(dx,1) += gY * wx;
            }
         }
         for (int dy = 0; dy < ND1d; ++dy) {
            const double wy  = Bt(dy,qy);
            const double wDy = Gt(dy,qy);
            for (int dx = 0; dx < ND1d; ++dx) {
               y(dx,dy,e) += ((gradX(dx,0) * wy) + (gradX(dx,1) * wDy));
            }
         }
      }
   });
}

// *****************************************************************************
static void DiffusionMultAssembled3D(const int ND1d,
                                     const int NQ1d,
                                     const int NE,
                                     const double* __restrict _B,
                                     const double* __restrict _G,
                                     const double* __restrict _Bt,
                                     const double* __restrict _Gt,
                                     const double* __restrict _op,
                                     const double* __restrict _x,
                                     double* __restrict _y)
{
   const int NQ = NQ1d * NQ1d * NQ1d;
   const int maxDQ = max(NQ1d, ND1d);
   const int maxDQ2 = maxDQ * maxDQ;
   const int Nspt = 3*(NQ + maxDQ2 + maxDQ);

   const Vector B(NQ1d,ND1d,_B);
   const Vector G(NQ1d,ND1d,_G);
   const Vector Bt(ND1d,NQ1d,_Bt);
   const Vector Gt(ND1d,NQ1d,_Gt);
   const Vector op(6,NQ,_op);
   const Vector x(ND1d,ND1d,ND1d,_x);
   Vector y(ND1d,ND1d,ND1d,_y);

   MFEM_FORALL_SHARED(e, NE, Nspt,
   {
      Vector4 grad(NQ1d,NQ1d,NQ1d,3,__shared);
      for (int qz = 0; qz < NQ1d; ++qz) {
         for (int qy = 0; qy < NQ1d; ++qy) {
            for (int qx = 0; qx < NQ1d; ++qx) {
               grad(qz,qy,qx,0) = 0.0;
               grad(qz,qy,qx,1) = 0.0;
               grad(qz,qy,qx,2) = 0.0;
            }
         }
      }
      for (int dz = 0; dz < ND1d; ++dz) {
         Vector3 gradXY(NQ1d,NQ1d,3,__shared+3*NQ);
         for (int qy = 0; qy < NQ1d; ++qy) {
            for (int qx = 0; qx < NQ1d; ++qx) {
               gradXY(qy,qx,0) = 0.0;
               gradXY(qy,qx,1) = 0.0;
               gradXY(qy,qx,2) = 0.0;
            }
         }
         for (int dy = 0; dy < ND1d; ++dy) {
            Vector2 gradX(NQ1d, 3, __shared + 3*NQ + 3*maxDQ2);
            for (int qx = 0; qx < NQ1d; ++qx) {
               gradX(qx,0) = 0.0;
               gradX(qx,1) = 0.0;
            }
            for (int dx = 0; dx < ND1d; ++dx) {
               const double s = x(dx,dy,dz,e);
               for (int qx = 0; qx < NQ1d; ++qx) {
                  gradX(qx,0) += s * B(qx,dx);
                  gradX(qx,1) += s * G(qx,dx);
               }
            }
            for (int qy = 0; qy < NQ1d; ++qy) {
               const double wy  = B(qy,dy);
               const double wDy = G(qy,dy);
               for (int qx = 0; qx < NQ1d; ++qx) {
                  const double wx  = gradX(qx,0);
                  const double wDx = gradX(qx,1);
                  gradXY(qy,qx,0) += wDx * wy;
                  gradXY(qy,qx,1) += wx  * wDy;
                  gradXY(qy,qx,2) += wx  * wy;
               }
            }
         }
         for (int qz = 0; qz < NQ1d; ++qz) {
            const double wz  = B(qz,dz);
            const double wDz = G(qz,dz);
            for (int qy = 0; qy < NQ1d; ++qy) {
               for (int qx = 0; qx < NQ1d; ++qx) {
                  grad(qz,qy,qx,0) += gradXY(qy,qx,0) * wz;
                  grad(qz,qy,qx,1) += gradXY(qy,qx,1) * wz;
                  grad(qz,qy,qx,2) += gradXY(qy,qx,2) * wDz;
               }
            }
         }
      }

      // Calculate Dxyz, xDyz, xyDz in plane
      for (int qz = 0; qz < NQ1d; ++qz) {
         for (int qy = 0; qy < NQ1d; ++qy) {
            for (int qx = 0; qx < NQ1d; ++qx) {
               const int q = QUAD_3D_ID(qx, qy, qz);
               const double O11 = op(0,q,e);
               const double O12 = op(1,q,e);
               const double O13 = op(2,q,e);
               const double O22 = op(3,q,e);
               const double O23 = op(4,q,e);
               const double O33 = op(5,q,e);
               const double gradX = grad(qz,qy,qx,0);
               const double gradY = grad(qz,qy,qx,1);
               const double gradZ = grad(qz,qy,qx,2);
               grad(qz,qy,qx,0) = (O11 * gradX) + (O12 * gradY) + (O13 * gradZ);
               grad(qz,qy,qx,1) = (O12 * gradX) + (O22 * gradY) + (O23 * gradZ);
               grad(qz,qy,qx,2) = (O13 * gradX) + (O23 * gradY) + (O33 * gradZ);
            }
         }
      }
      for (int qz = 0; qz < NQ1d; ++qz) {
         Vector3 gradXY(ND1d,ND1d, 3, __shared + 3*NQ);
         for (int dy = 0; dy < ND1d; ++dy) {
            for (int dx = 0; dx < ND1d; ++dx) {
               gradXY(dy,dx,0) = 0.0;
               gradXY(dy,dx,1) = 0.0;
               gradXY(dy,dx,2) = 0.0;
            }
         }
         for (int qy = 0; qy < NQ1d; ++qy) {
            Vector2 gradX(ND1d, 3, __shared + 3*NQ + 3*maxDQ2);
            for (int dx = 0; dx < ND1d; ++dx) {
               gradX(dx,0) = 0.0;
               gradX(dx,1) = 0.0;
               gradX(dx,2) = 0.0;
            }
            for (int qx = 0; qx < NQ1d; ++qx) {
               const double gX = grad(qz,qy,qx,0);
               const double gY = grad(qz,qy,qx,1);
               const double gZ = grad(qz,qy,qx,2);
               for (int dx = 0; dx < ND1d; ++dx) {
                  const double wx  = Bt(dx,qx);
                  const double wDx = Gt(dx,qx);
                  gradX(dx,0) += gX * wDx;
                  gradX(dx,1) += gY * wx;
                  gradX(dx,2) += gZ * wx;
               }
            }
            for (int dy = 0; dy < ND1d; ++dy) {
               const double wy  = Bt(dy,qy);
               const double wDy = Gt(dy,qy);
               for (int dx = 0; dx < ND1d; ++dx) {
                  gradXY(dy,dx,0) += gradX(dx,0) * wy;
                  gradXY(dy,dx,1) += gradX(dx,1) * wDy;
                  gradXY(dy,dx,2) += gradX(dx,2) * wy;
               }
            }
         }
         for (int dz = 0; dz < ND1d; ++dz) {
            const double wz  = Bt(dz,qz);
            const double wDz = Gt(dz,qz);
            for (int dy = 0; dy < ND1d; ++dy) {
               for (int dx = 0; dx < ND1d; ++dx) {
                  y(dx,dy,dz,e) +=
                     ((gradXY(dy,dx,0) * wz) +
                      (gradXY(dy,dx,1) * wz) +
                      (gradXY(dy,dx,2) * wDz));
               }
            }
         }
      }
   });
}

// *****************************************************************************
void DiffusionMultAssembled(const int DIM,
                            const int ND1d,
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
      occaDiffusionMultAssembled2D(ND1d, NQ1d, NE, B, G, Bt, Gt, op, x, y);
      return;
   }
#endif // __OCCA__
   if (DIM==2)
   {
      DiffusionMultAssembled2D(ND1d, NQ1d, NE, B, G, Bt, Gt, op, x, y);
      return;
   }
   if (DIM==3)
   {
      DiffusionMultAssembled3D(ND1d, NQ1d, NE, B, G, Bt, Gt, op, x, y);
      return;
   }
}

} // namespace fem
} // namespace kernels
} // namespace mfem
