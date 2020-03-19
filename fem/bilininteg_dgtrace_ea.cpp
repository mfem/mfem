// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "../linalg/elementmatrix.hpp"

namespace mfem
{

static void EADGTraceAssemble1D(const int NF,
                                const Array<double> &basis,
                                const Vector &padata,
                                Vector &eadata_int,
                                Vector &eadata_ext,
                                FaceType type)
{
   const bool bdr = type==FaceType::Boundary;
   auto D = Reshape(padata.Read(), 2, 2, NF);
   auto A_int = Reshape(eadata_int.Write(), 2, NF);
   auto A_ext = Reshape(eadata_ext.Write(), 2, NF);
   MFEM_FORALL(f, NF,
   {
      double val_int0, val_int1, val_ext01, val_ext10;
      val_int0  = D(0, 0, f);
      if (bdr)
      {
         val_ext01 = D(1, 0, f);
         val_ext10 = D(0, 1, f);
      }
      val_int1  = D(1, 1, f);
         A_int(0, f) = val_int0;
         A_int(1, f) = val_int1;
      if (bdr)
      {
         A_ext(0, f) = val_ext01;
         A_ext(1, f) = val_ext10;
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADGTraceAssemble2D(const int NF,
                                const Array<double> &basis,
                                const Vector &padata,
                                Vector &eadata_int,
                                Vector &eadata_ext,
                                FaceType type,
                                const int d1d = 0,
                                const int q1d = 0)
{
   const bool bdr = type==FaceType::Boundary;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, 2, 2, NF);
   auto A_int = Reshape(eadata_int.Write(), D1D, D1D, 2, NF);
   auto A_ext = Reshape(eadata_ext.Write(), D1D, D1D, 2, NF);
   MFEM_FORALL_3D(f, NF, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double r_Bi[MQ1];
      double r_Bj[MQ1];
      for (int q = 0; q < Q1D; q++)
      {
         r_Bi[q] = B(q,MFEM_THREAD_ID(x));
         r_Bj[q] = B(q,MFEM_THREAD_ID(y));
      }
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            double val_int0 = 0.0;
            double val_int1 = 0.0;
            double val_ext01 = 0.0;
            double val_ext10 = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val_int0  += r_Bi[k1] * r_Bj[k1] * D(k1, 0, 0, f);
               if (bdr)
               {
                  val_ext01 += r_Bi[k1] * r_Bj[k1] * D(k1, 1, 0, f);
                  val_ext10 += r_Bi[k1] * r_Bj[k1] * D(k1, 0, 1, f);
               }
               val_int1  += r_Bi[k1] * r_Bj[k1] * D(k1, 1, 1, f);
            }
            A_int(i1, j1, 0, f) = val_int0;
            A_int(i1, j1, 1, f) = val_int1;
            if (bdr)
            {
               A_ext(i1, j1, 0, f) = val_ext01;
               A_ext(i1, j1, 1, f) = val_ext10;
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADGTraceAssemble3D(const int NF,
                                const Array<double> &basis,
                                const Vector &padata,
                                Vector &eadata_int,
                                Vector &eadata_ext,
                                FaceType type,
                                const int d1d = 0,
                                const int q1d = 0)
{
   const bool bdr = type==FaceType::Boundary;
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, 2, 2, NF);
   auto A_int = Reshape(eadata_int.Write(), D1D, D1D, D1D, D1D, 2, NF);
   auto A_ext = Reshape(eadata_ext.Write(), D1D, D1D, D1D, D1D, 2, NF);
   MFEM_FORALL_3D(f, NF, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      double r_B[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
         }
      }
      MFEM_SHARED double s_D[MQ1][MQ1][2][2];
      for (int i; i < 2; i++)
      {
         for (int j; j < 2; j++)
         {
            MFEM_FOREACH_THREAD(k1,x,Q1D)
            {
               MFEM_FOREACH_THREAD(k2,y,Q1D)
               {
                  s_D[k1][k2][i][j] = D(k1,k2,i,j,f);
               }
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            for (int j1 = 0; j1 < D1D; ++j1)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  double val_int0 = 0.0;
                  double val_int1 = 0.0;
                  double val_ext01 = 0.0;
                  double val_ext10 = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        val_int0 += r_B[k1][i1] * r_B[k1][j1]
                                  * r_B[k2][i2] * r_B[k2][j2]
                                  * s_D[k1][k2][0][0];
                        val_int1 += r_B[k1][i1] * r_B[k1][j1]
                                  * r_B[k2][i2] * r_B[k2][j2]
                                  * s_D[k1][k2][1][1];
                        if (bdr)
                        {
                           val_ext01+= r_B[k1][i1] * r_B[k1][j1]
                                    * r_B[k2][i2] * r_B[k2][j2]
                                    * s_D[k1][k2][1][0];
                           val_ext10+= r_B[k1][i1] * r_B[k1][j1]
                                    * r_B[k2][i2] * r_B[k2][j2]
                                    * s_D[k1][k2][0][1];
                        }
                     }
                  }
                  A_int(i1, i2, j1, j2, 0, f) = val_int0;
                  A_int(i1, i2, j1, j2, 1, f) = val_int1;
                  if (bdr)
                  {
                     A_ext(i1, i2, j1, j2, 0, f) = val_ext01;
                     A_ext(i1, i2, j1, j2, 1, f) = val_ext10;
                  }
               }
            }
         }
      }
   });
}

void DGTraceIntegrator::SetupEA(const FiniteElementSpace &fes,
                                Vector &ea_data_int,
                                Vector &ea_data_ext,
                                FaceType type)
{
   SetupPA(fes, type);
   nf = fes.GetNFbyType(type);
   const Array<double> &B = maps->B;
   if (dim == 1)
   {
      return EADGTraceAssemble1D(nf,B,pa_data,ea_data_int,ea_data_ext,type);
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EADGTraceAssemble2D<2,2>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x33: return EADGTraceAssemble2D<3,3>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x44: return EADGTraceAssemble2D<4,4>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x55: return EADGTraceAssemble2D<5,5>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x66: return EADGTraceAssemble2D<6,6>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x77: return EADGTraceAssemble2D<7,7>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x88: return EADGTraceAssemble2D<8,8>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x99: return EADGTraceAssemble2D<9,9>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         default:   return EADGTraceAssemble2D(nf,B,pa_data,ea_data_int,ea_data_ext,type,dofs1D,quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: return EADGTraceAssemble3D<2,3>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x34: return EADGTraceAssemble3D<3,4>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x45: return EADGTraceAssemble3D<4,5>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x56: return EADGTraceAssemble3D<5,6>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x67: return EADGTraceAssemble3D<6,7>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x78: return EADGTraceAssemble3D<7,8>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         case 0x89: return EADGTraceAssemble3D<8,9>(nf,B,pa_data,ea_data_int,ea_data_ext,type);
         default:   return EADGTraceAssemble3D(nf,B,pa_data,ea_data_int,ea_data_ext,type,dofs1D,quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

void DGTraceIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace& fes,
                                                Vector &ea_data_int,
                                                Vector &ea_data_ext)
{
   SetupEA(fes, ea_data_int, ea_data_ext, FaceType::Interior);
}

void DGTraceIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace& fes,
                                                Vector &ea_data_bdr)
{
   SetupEA(fes, ea_data_bdr, ea_data_bdr, FaceType::Boundary);
}

}
