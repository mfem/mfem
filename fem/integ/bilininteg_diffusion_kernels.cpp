// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "bilininteg_diffusion_kernels.hpp"

namespace mfem
{

namespace internal
{

template<>
void PADiffusionSetup2D<2>(const int Q1D,
                           const int coeffDim,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           const Vector &c,
                           Vector &d);

template<>
void PADiffusionSetup2D<3>(const int Q1D,
                           const int coeffDim,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           const Vector &c,
                           Vector &d);

void PADiffusionSetup(const int dim,
                      const int sdim,
                      const int D1D,
                      const int Q1D,
                      const int coeffDim,
                      const int NE,
                      const Array<double> &W,
                      const Vector &J,
                      const Vector &C,
                      Vector &D)
{
   if (dim == 1) { MFEM_ABORT("dim==1 not supported in PADiffusionSetup"); }
   if (dim == 2)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPADiffusionSetup2D(D1D, Q1D, NE, W, J, C, D);
         return;
      }
#else
      MFEM_CONTRACT_VAR(D1D);
#endif // MFEM_USE_OCCA
      if (sdim == 2) { PADiffusionSetup2D<2>(Q1D, coeffDim, NE, W, J, C, D); }
      if (sdim == 3) { PADiffusionSetup2D<3>(Q1D, coeffDim, NE, W, J, C, D); }
   }
   if (dim == 3)
   {
#ifdef MFEM_USE_OCCA
      if (DeviceCanUseOcca())
      {
         OccaPADiffusionSetup3D(D1D, Q1D, NE, W, J, C, D);
         return;
      }
#endif // MFEM_USE_OCCA
      PADiffusionSetup3D(Q1D, coeffDim, NE, W, J, C, D);
   }
}

template<>
void PADiffusionSetup2D<2>(const int Q1D,
                           const int coeffDim,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           const Vector &c,
                           Vector &d)
{
   const bool symmetric = (coeffDim != 4);
   const bool const_c = c.Size() == 1;
   MFEM_VERIFY(coeffDim < 3 ||
               !const_c, "Constant matrix coefficient not supported");
   const auto W = Reshape(w.Read(), Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,2,2,NE);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1,1) :
                  Reshape(c.Read(), coeffDim,Q1D,Q1D,NE);
   auto D = Reshape(d.Write(), Q1D,Q1D, symmetric ? 3 : 4, NE);
   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            const double J11 = J(qx,qy,0,0,e);
            const double J21 = J(qx,qy,1,0,e);
            const double J12 = J(qx,qy,0,1,e);
            const double J22 = J(qx,qy,1,1,e);
            const double w_detJ = W(qx,qy) / ((J11*J22)-(J21*J12));
            if (coeffDim == 3 || coeffDim == 4) // Matrix coefficient
            {
               // First compute entries of R = MJ^{-T}, without det J factor.
               const double M11 = C(0,qx,qy,e);
               const double M12 = C(1,qx,qy,e);
               const double M21 = symmetric ? M12 : C(2,qx,qy,e);
               const double M22 = symmetric ? C(2,qx,qy,e) : C(3,qx,qy,e);
               const double R11 = M11*J22 - M12*J12;
               const double R21 = M21*J22 - M22*J12;
               const double R12 = -M11*J21 + M12*J11;
               const double R22 = -M21*J21 + M22*J11;

               // Now set y to J^{-1}R.
               D(qx,qy,0,e) = w_detJ * ( J22*R11 - J12*R21); // 1,1
               D(qx,qy,1,e) = w_detJ * (-J21*R11 + J11*R21); // 2,1
               D(qx,qy,2,e) = w_detJ * (symmetric ? (-J21*R12 + J11*R22) :
                                        (J22*R12 - J12*R22)); // 2,2 or 1,2
               if (!symmetric)
               {
                  D(qx,qy,3,e) = w_detJ * (-J21*R12 + J11*R22); // 2,2
               }
            }
            else // Vector or scalar coefficient
            {
               const double C1 = const_c ? C(0,0,0,0) : C(0,qx,qy,e);
               const double C2 = const_c ? C(0,0,0,0) :
                                 (coeffDim == 2 ? C(1,qx,qy,e) : C(0,qx,qy,e));

               D(qx,qy,0,e) =  w_detJ * (C2*J12*J12 + C1*J22*J22); // 1,1
               D(qx,qy,1,e) = -w_detJ * (C2*J12*J11 + C1*J22*J21); // 1,2
               D(qx,qy,2,e) =  w_detJ * (C2*J11*J11 + C1*J21*J21); // 2,2
            }
         }
      }
   });
}

template<>
void PADiffusionSetup2D<3>(const int Q1D,
                           const int coeffDim,
                           const int NE,
                           const Array<double> &w,
                           const Vector &j,
                           const Vector &c,
                           Vector &d)
{
   MFEM_VERIFY(coeffDim == 1, "Matrix and vector coefficients not supported");
   constexpr int DIM = 2;
   constexpr int SDIM = 3;
   const bool const_c = c.Size() == 1;
   const auto W = Reshape(w.Read(), Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,SDIM,DIM,NE);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1) :
                  Reshape(c.Read(), Q1D,Q1D,NE);
   auto D = Reshape(d.Write(), Q1D,Q1D, 3, NE);
   mfem::forall_2D(NE, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            const double wq = W(qx,qy);
            const double J11 = J(qx,qy,0,0,e);
            const double J21 = J(qx,qy,1,0,e);
            const double J31 = J(qx,qy,2,0,e);
            const double J12 = J(qx,qy,0,1,e);
            const double J22 = J(qx,qy,1,1,e);
            const double J32 = J(qx,qy,2,1,e);
            const double E = J11*J11 + J21*J21 + J31*J31;
            const double G = J12*J12 + J22*J22 + J32*J32;
            const double F = J11*J12 + J21*J22 + J31*J32;
            const double iw = 1.0 / sqrt(E*G - F*F);
            const double coeff = const_c ? C(0,0,0) : C(qx,qy,e);
            const double alpha = wq * coeff * iw;
            D(qx,qy,0,e) =  alpha * G; // 1,1
            D(qx,qy,1,e) = -alpha * F; // 1,2
            D(qx,qy,2,e) =  alpha * E; // 2,2
         }
      }
   });
}

void PADiffusionSetup3D(const int Q1D,
                        const int coeffDim,
                        const int NE,
                        const Array<double> &w,
                        const Vector &j,
                        const Vector &c,
                        Vector &d)
{
   const bool symmetric = (coeffDim != 9);
   const bool const_c = c.Size() == 1;
   MFEM_VERIFY(coeffDim < 6 ||
               !const_c, "Constant matrix coefficient not supported");
   const auto W = Reshape(w.Read(), Q1D,Q1D,Q1D);
   const auto J = Reshape(j.Read(), Q1D,Q1D,Q1D,3,3,NE);
   const auto C = const_c ? Reshape(c.Read(), 1,1,1,1,1) :
                  Reshape(c.Read(), coeffDim,Q1D,Q1D,Q1D,NE);
   auto D = Reshape(d.Write(), Q1D,Q1D,Q1D, symmetric ? 6 : 9, NE);
   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_FOREACH_THREAD(qx,x,Q1D)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1D)
            {
               const double J11 = J(qx,qy,qz,0,0,e);
               const double J21 = J(qx,qy,qz,1,0,e);
               const double J31 = J(qx,qy,qz,2,0,e);
               const double J12 = J(qx,qy,qz,0,1,e);
               const double J22 = J(qx,qy,qz,1,1,e);
               const double J32 = J(qx,qy,qz,2,1,e);
               const double J13 = J(qx,qy,qz,0,2,e);
               const double J23 = J(qx,qy,qz,1,2,e);
               const double J33 = J(qx,qy,qz,2,2,e);
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                   J21 * (J12 * J33 - J32 * J13) +
                                   J31 * (J12 * J23 - J22 * J13);
               const double w_detJ = W(qx,qy,qz) / detJ;
               // adj(J)
               const double A11 = (J22 * J33) - (J23 * J32);
               const double A12 = (J32 * J13) - (J12 * J33);
               const double A13 = (J12 * J23) - (J22 * J13);
               const double A21 = (J31 * J23) - (J21 * J33);
               const double A22 = (J11 * J33) - (J13 * J31);
               const double A23 = (J21 * J13) - (J11 * J23);
               const double A31 = (J21 * J32) - (J31 * J22);
               const double A32 = (J31 * J12) - (J11 * J32);
               const double A33 = (J11 * J22) - (J12 * J21);

               if (coeffDim == 6 || coeffDim == 9) // Matrix coefficient version
               {
                  // Compute entries of R = MJ^{-T} = M adj(J)^T, without det J.
                  const double M11 = C(0, qx,qy,qz, e);
                  const double M12 = C(1, qx,qy,qz, e);
                  const double M13 = C(2, qx,qy,qz, e);
                  const double M21 = (!symmetric) ? C(3, qx,qy,qz, e) : M12;
                  const double M22 = (!symmetric) ? C(4, qx,qy,qz, e) : C(3, qx,qy,qz, e);
                  const double M23 = (!symmetric) ? C(5, qx,qy,qz, e) : C(4, qx,qy,qz, e);
                  const double M31 = (!symmetric) ? C(6, qx,qy,qz, e) : M13;
                  const double M32 = (!symmetric) ? C(7, qx,qy,qz, e) : M23;
                  const double M33 = (!symmetric) ? C(8, qx,qy,qz, e) : C(5, qx,qy,qz, e);

                  const double R11 = M11*A11 + M12*A12 + M13*A13;
                  const double R12 = M11*A21 + M12*A22 + M13*A23;
                  const double R13 = M11*A31 + M12*A32 + M13*A33;
                  const double R21 = M21*A11 + M22*A12 + M23*A13;
                  const double R22 = M21*A21 + M22*A22 + M23*A23;
                  const double R23 = M21*A31 + M22*A32 + M23*A33;
                  const double R31 = M31*A11 + M32*A12 + M33*A13;
                  const double R32 = M31*A21 + M32*A22 + M33*A23;
                  const double R33 = M31*A31 + M32*A32 + M33*A33;

                  // Now set D to J^{-1} R = adj(J) R
                  D(qx,qy,qz,0,e) = w_detJ * (A11*R11 + A12*R21 + A13*R31); // 1,1
                  const double D12 = w_detJ * (A11*R12 + A12*R22 + A13*R32);
                  D(qx,qy,qz,1,e) = D12; // 1,2
                  D(qx,qy,qz,2,e) = w_detJ * (A11*R13 + A12*R23 + A13*R33); // 1,3

                  const double D22 = w_detJ * (A21*R12 + A22*R22 + A23*R32);
                  const double D23 = w_detJ * (A21*R13 + A22*R23 + A23*R33);

                  const double D33 = w_detJ * (A31*R13 + A32*R23 + A33*R33);

                  D(qx,qy,qz,4,e) = symmetric ? D23 : D22; // 2,3 or 2,2
                  D(qx,qy,qz,5,e) = symmetric ? D33 : D23; // 3,3 or 2,3

                  if (symmetric)
                  {
                     D(qx,qy,qz,3,e) = D22; // 2,2
                  }
                  else
                  {
                     D(qx,qy,qz,3,e) = w_detJ * (A21*R11 + A22*R21 + A23*R31); // 2,1
                     D(qx,qy,qz,6,e) = w_detJ * (A31*R11 + A32*R21 + A33*R31); // 3,1
                     D(qx,qy,qz,7,e) = w_detJ * (A31*R12 + A32*R22 + A33*R32); // 3,2
                     D(qx,qy,qz,8,e) = D33; // 3,3
                  }
               }
               else  // Vector or scalar coefficient version
               {
                  const double C1 = const_c ? C(0,0,0,0,0) : C(0,qx,qy,qz,e);
                  const double C2 = const_c ? C(0,0,0,0,0) :
                                    (coeffDim == 3 ? C(1,qx,qy,qz,e) : C(0,qx,qy,qz,e));
                  const double C3 = const_c ? C(0,0,0,0,0) :
                                    (coeffDim == 3 ? C(2,qx,qy,qz,e) : C(0,qx,qy,qz,e));

                  // detJ J^{-1} J^{-T} = (1/detJ) adj(J) adj(J)^T
                  D(qx,qy,qz,0,e) = w_detJ * (C1*A11*A11 + C2*A12*A12 + C3*A13*A13); // 1,1
                  D(qx,qy,qz,1,e) = w_detJ * (C1*A11*A21 + C2*A12*A22 + C3*A13*A23); // 2,1
                  D(qx,qy,qz,2,e) = w_detJ * (C1*A11*A31 + C2*A12*A32 + C3*A13*A33); // 3,1
                  D(qx,qy,qz,3,e) = w_detJ * (C1*A21*A21 + C2*A22*A22 + C3*A23*A23); // 2,2
                  D(qx,qy,qz,4,e) = w_detJ * (C1*A21*A31 + C2*A22*A32 + C3*A23*A33); // 3,2
                  D(qx,qy,qz,5,e) = w_detJ * (C1*A31*A31 + C2*A32*A32 + C3*A33*A33); // 3,3
               }
            }
         }
      }
   });
}

#ifdef MFEM_USE_OCCA
void OccaPADiffusionSetup2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &W,
                            const Vector &J,
                            const Vector &C,
                            Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   const occa::memory o_C = OccaMemoryRead(C.GetMemory(), C.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const bool const_c = C.Size() == 1;
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup2D_ker;
   if (OccaDiffSetup2D_ker.find(id) == OccaDiffSetup2D_ker.end())
   {
      const occa::kernel DiffusionSetup2D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup2D", props);
      OccaDiffSetup2D_ker.emplace(id, DiffusionSetup2D);
   }
   OccaDiffSetup2D_ker.at(id)(NE, o_W, o_J, o_C, o_op, const_c);
}

void OccaPADiffusionSetup3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &W,
                            const Vector &J,
                            const Vector &C,
                            Vector &op)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_W = OccaMemoryRead(W.GetMemory(), W.Size());
   const occa::memory o_J = OccaMemoryRead(J.GetMemory(), J.Size());
   const occa::memory o_C = OccaMemoryRead(C.GetMemory(), C.Size());
   occa::memory o_op = OccaMemoryWrite(op.GetMemory(), op.Size());
   const bool const_c = C.Size() == 1;
   const occa_id_t id = std::make_pair(D1D,Q1D);
   static occa_kernel_t OccaDiffSetup3D_ker;
   if (OccaDiffSetup3D_ker.find(id) == OccaDiffSetup3D_ker.end())
   {
      const occa::kernel DiffusionSetup3D =
         mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                     "DiffusionSetup3D", props);
      OccaDiffSetup3D_ker.emplace(id, DiffusionSetup3D);
   }
   OccaDiffSetup3D_ker.at(id)(NE, o_W, o_J, o_C, o_op, const_c);
}
#endif // MFEM_USE_OCCA

void PADiffusionAssembleDiagonal(const int dim,
                                 const int D1D,
                                 const int Q1D,
                                 const int NE,
                                 const bool symm,
                                 const Array<double> &B,
                                 const Array<double> &G,
                                 const Vector &D,
                                 Vector &Y)
{
   if (dim == 2)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPADiffusionDiagonal2D<2,2,8>(NE,symm,B,G,D,Y);
         case 0x33: return SmemPADiffusionDiagonal2D<3,3,8>(NE,symm,B,G,D,Y);
         case 0x44: return SmemPADiffusionDiagonal2D<4,4,4>(NE,symm,B,G,D,Y);
         case 0x55: return SmemPADiffusionDiagonal2D<5,5,4>(NE,symm,B,G,D,Y);
         case 0x66: return SmemPADiffusionDiagonal2D<6,6,2>(NE,symm,B,G,D,Y);
         case 0x77: return SmemPADiffusionDiagonal2D<7,7,2>(NE,symm,B,G,D,Y);
         case 0x88: return SmemPADiffusionDiagonal2D<8,8,1>(NE,symm,B,G,D,Y);
         case 0x99: return SmemPADiffusionDiagonal2D<9,9,1>(NE,symm,B,G,D,Y);
         default: return PADiffusionDiagonal2D(NE,symm,B,G,D,Y,D1D,Q1D);
      }
   }
   else if (dim == 3)
   {
      switch ((D1D << 4 ) | Q1D)
      {
         case 0x22: return SmemPADiffusionDiagonal3D<2,2>(NE,symm,B,G,D,Y);
         case 0x23: return SmemPADiffusionDiagonal3D<2,3>(NE,symm,B,G,D,Y);
         case 0x34: return SmemPADiffusionDiagonal3D<3,4>(NE,symm,B,G,D,Y);
         case 0x45: return SmemPADiffusionDiagonal3D<4,5>(NE,symm,B,G,D,Y);
         case 0x46: return SmemPADiffusionDiagonal3D<4,6>(NE,symm,B,G,D,Y);
         case 0x56: return SmemPADiffusionDiagonal3D<5,6>(NE,symm,B,G,D,Y);
         case 0x67: return SmemPADiffusionDiagonal3D<6,7>(NE,symm,B,G,D,Y);
         case 0x78: return SmemPADiffusionDiagonal3D<7,8>(NE,symm,B,G,D,Y);
         case 0x89: return SmemPADiffusionDiagonal3D<8,9>(NE,symm,B,G,D,Y);
         case 0x9A: return SmemPADiffusionDiagonal3D<9,10>(NE,symm,B,G,D,Y);
         default: return PADiffusionDiagonal3D(NE,symm,B,G,D,Y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

void PADiffusionApply(const int dim,
                      const int D1D,
                      const int Q1D,
                      const int NE,
                      const bool symm,
                      const Array<double> &B,
                      const Array<double> &G,
                      const Array<double> &Bt,
                      const Array<double> &Gt,
                      const Vector &D,
                      const Vector &X,
                      Vector &Y)
{
#ifdef MFEM_USE_OCCA
   if (DeviceCanUseOcca())
   {
      if (dim == 2)
      {
         OccaPADiffusionApply2D(D1D,Q1D,NE,B,G,Bt,Gt,D,X,Y);
         return;
      }
      if (dim == 3)
      {
         OccaPADiffusionApply3D(D1D,Q1D,NE,B,G,Bt,Gt,D,X,Y);
         return;
      }
      MFEM_ABORT("OCCA PADiffusionApply unknown kernel!");
   }
#endif // MFEM_USE_OCCA
   const int id = (D1D << 4) | Q1D;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x22: return SmemPADiffusionApply2D<2,2,16>(NE,symm,B,G,D,X,Y);
         case 0x33: return SmemPADiffusionApply2D<3,3,16>(NE,symm,B,G,D,X,Y);
         case 0x44: return SmemPADiffusionApply2D<4,4,8>(NE,symm,B,G,D,X,Y);
         case 0x55: return SmemPADiffusionApply2D<5,5,8>(NE,symm,B,G,D,X,Y);
         case 0x66: return SmemPADiffusionApply2D<6,6,4>(NE,symm,B,G,D,X,Y);
         case 0x77: return SmemPADiffusionApply2D<7,7,4>(NE,symm,B,G,D,X,Y);
         case 0x88: return SmemPADiffusionApply2D<8,8,2>(NE,symm,B,G,D,X,Y);
         case 0x99: return SmemPADiffusionApply2D<9,9,2>(NE,symm,B,G,D,X,Y);
         default:   return PADiffusionApply2D(NE,symm,B,G,Bt,Gt,D,X,Y,D1D,Q1D);
      }
   }

   if (dim == 3)
   {
      switch (id)
      {
         case 0x22: return SmemPADiffusionApply3D<2,2>(NE,symm,B,G,D,X,Y);
         case 0x23: return SmemPADiffusionApply3D<2,3>(NE,symm,B,G,D,X,Y);
         case 0x34: return SmemPADiffusionApply3D<3,4>(NE,symm,B,G,D,X,Y);
         case 0x45: return SmemPADiffusionApply3D<4,5>(NE,symm,B,G,D,X,Y);
         case 0x46: return SmemPADiffusionApply3D<4,6>(NE,symm,B,G,D,X,Y);
         case 0x56: return SmemPADiffusionApply3D<5,6>(NE,symm,B,G,D,X,Y);
         case 0x58: return SmemPADiffusionApply3D<5,8>(NE,symm,B,G,D,X,Y);
         case 0x67: return SmemPADiffusionApply3D<6,7>(NE,symm,B,G,D,X,Y);
         case 0x78: return SmemPADiffusionApply3D<7,8>(NE,symm,B,G,D,X,Y);
         case 0x89: return SmemPADiffusionApply3D<8,9>(NE,symm,B,G,D,X,Y);
         default:   return PADiffusionApply3D(NE,symm,B,G,Bt,Gt,D,X,Y,D1D,Q1D);
      }
   }
   MFEM_ABORT("Unknown kernel: 0x"<<std::hex << id << std::dec);
}

#ifdef MFEM_USE_OCCA
void OccaPADiffusionApply2D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &B,
                            const Array<double> &G,
                            const Array<double> &Bt,
                            const Array<double> &Gt,
                            const Vector &D,
                            const Vector &X,
                            Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_G = OccaMemoryRead(G.GetMemory(), G.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_Gt = OccaMemoryRead(Gt.GetMemory(), Gt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply2D_cpu;
      if (OccaDiffApply2D_cpu.find(id) == OccaDiffApply2D_cpu.end())
      {
         const occa::kernel DiffusionApply2D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_CPU", props);
         OccaDiffApply2D_cpu.emplace(id, DiffusionApply2D_CPU);
      }
      OccaDiffApply2D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply2D_gpu;
      if (OccaDiffApply2D_gpu.find(id) == OccaDiffApply2D_gpu.end())
      {
         const occa::kernel DiffusionApply2D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply2D_GPU", props);
         OccaDiffApply2D_gpu.emplace(id, DiffusionApply2D_GPU);
      }
      OccaDiffApply2D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
}

void OccaPADiffusionApply3D(const int D1D,
                            const int Q1D,
                            const int NE,
                            const Array<double> &B,
                            const Array<double> &G,
                            const Array<double> &Bt,
                            const Array<double> &Gt,
                            const Vector &D,
                            const Vector &X,
                            Vector &Y)
{
   occa::properties props;
   props["defines/D1D"] = D1D;
   props["defines/Q1D"] = Q1D;
   const occa::memory o_B = OccaMemoryRead(B.GetMemory(), B.Size());
   const occa::memory o_G = OccaMemoryRead(G.GetMemory(), G.Size());
   const occa::memory o_Bt = OccaMemoryRead(Bt.GetMemory(), Bt.Size());
   const occa::memory o_Gt = OccaMemoryRead(Gt.GetMemory(), Gt.Size());
   const occa::memory o_D = OccaMemoryRead(D.GetMemory(), D.Size());
   const occa::memory o_X = OccaMemoryRead(X.GetMemory(), X.Size());
   occa::memory o_Y = OccaMemoryReadWrite(Y.GetMemory(), Y.Size());
   const occa_id_t id = std::make_pair(D1D,Q1D);
   if (!Device::Allows(Backend::OCCA_CUDA))
   {
      static occa_kernel_t OccaDiffApply3D_cpu;
      if (OccaDiffApply3D_cpu.find(id) == OccaDiffApply3D_cpu.end())
      {
         const occa::kernel DiffusionApply3D_CPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_CPU", props);
         OccaDiffApply3D_cpu.emplace(id, DiffusionApply3D_CPU);
      }
      OccaDiffApply3D_cpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
   else
   {
      static occa_kernel_t OccaDiffApply3D_gpu;
      if (OccaDiffApply3D_gpu.find(id) == OccaDiffApply3D_gpu.end())
      {
         const occa::kernel DiffusionApply3D_GPU =
            mfem::OccaDev().buildKernel("occa://mfem/fem/occa.okl",
                                        "DiffusionApply3D_GPU", props);
         OccaDiffApply3D_gpu.emplace(id, DiffusionApply3D_GPU);
      }
      OccaDiffApply3D_gpu.at(id)(NE, o_B, o_G, o_Bt, o_Gt, o_D, o_X, o_Y);
   }
}
#endif // MFEM_USE_OCCA

} // namespace internal

} // namespace mfem
