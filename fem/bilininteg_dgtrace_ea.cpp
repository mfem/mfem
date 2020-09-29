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

namespace mfem
{

static void EADGTraceAssemble1DInt(const int NF,
                                   const Array<double> &basis,
                                   const Vector &padata,
                                   Vector &eadata_int,
                                   Vector &eadata_ext,
                                   const bool add)
{
   auto D = Reshape(padata.Read(), 2, 2, NF);
   auto A_int = Reshape(eadata_int.ReadWrite(), 2, NF);
   auto A_ext = Reshape(eadata_ext.ReadWrite(), 2, NF);
   MFEM_FORALL(f, NF,
   {
      double val_int0, val_int1, val_ext01, val_ext10;
      val_int0  = D(0, 0, f);
      val_ext10 = D(1, 0, f);
      val_ext01 = D(0, 1, f);
      val_int1  = D(1, 1, f);
      if (add)
      {
         A_int(0, f) += val_int0;
         A_int(1, f) += val_int1;
         A_ext(0, f) += val_ext01;
         A_ext(1, f) += val_ext10;
      }
      else
      {
         A_int(0, f) = val_int0;
         A_int(1, f) = val_int1;
         A_ext(0, f) = val_ext01;
         A_ext(1, f) = val_ext10;
      }
   });
}

static void EADGTraceAssemble1DBdr(const int NF,
                                   const Array<double> &basis,
                                   const Vector &padata,
                                   Vector &eadata_bdr,
                                   const bool add)
{
   auto D = Reshape(padata.Read(), 2, 2, NF);
   auto A_bdr = Reshape(eadata_bdr.ReadWrite(), NF);
   MFEM_FORALL(f, NF,
   {
      if (add)
      {
         A_bdr(f) += D(0, 0, f);
      }
      else
      {
         A_bdr(f) = D(0, 0, f);
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADGTraceAssemble2DInt(const int NF,
                                   const Array<double> &basis,
                                   const Vector &padata,
                                   Vector &eadata_int,
                                   Vector &eadata_ext,
                                   const bool add,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, 2, 2, NF);
   auto A_int = Reshape(eadata_int.ReadWrite(), D1D, D1D, 2, NF);
   auto A_ext = Reshape(eadata_ext.ReadWrite(), D1D, D1D, 2, NF);
   MFEM_FORALL_3D(f, NF, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
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
               val_int0  += B(k1,i1) * B(k1,j1) * D(k1, 0, 0, f);
               val_ext01 += B(k1,i1) * B(k1,j1) * D(k1, 0, 1, f);
               val_ext10 += B(k1,i1) * B(k1,j1) * D(k1, 1, 0, f);
               val_int1  += B(k1,i1) * B(k1,j1) * D(k1, 1, 1, f);
            }
            if (add)
            {
               A_int(i1, j1, 0, f) += val_int0;
               A_int(i1, j1, 1, f) += val_int1;
               A_ext(i1, j1, 0, f) += val_ext01;
               A_ext(i1, j1, 1, f) += val_ext10;
            }
            else
            {
               A_int(i1, j1, 0, f) = val_int0;
               A_int(i1, j1, 1, f) = val_int1;
               A_ext(i1, j1, 0, f) = val_ext01;
               A_ext(i1, j1, 1, f) = val_ext10;
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADGTraceAssemble2DBdr(const int NF,
                                   const Array<double> &basis,
                                   const Vector &padata,
                                   Vector &eadata_bdr,
                                   const bool add,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, 2, 2, NF);
   auto A_bdr = Reshape(eadata_bdr.ReadWrite(), D1D, D1D, NF);
   MFEM_FORALL_3D(f, NF, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            double val_bdr = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val_bdr  += B(k1,i1) * B(k1,j1) * D(k1, 0, 0, f);
            }
            if (add)
            {
               A_bdr(i1, j1, f) += val_bdr;
            }
            else
            {
               A_bdr(i1, j1, f) = val_bdr;
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADGTraceAssemble3DInt(const int NF,
                                   const Array<double> &basis,
                                   const Vector &padata,
                                   Vector &eadata_int,
                                   Vector &eadata_ext,
                                   const bool add,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, 2, 2, NF);
   auto A_int = Reshape(eadata_int.ReadWrite(), D1D, D1D, D1D, D1D, 2, NF);
   auto A_ext = Reshape(eadata_ext.ReadWrite(), D1D, D1D, D1D, D1D, 2, NF);
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
      for (int i=0; i < 2; i++)
      {
         for (int j=0; j < 2; j++)
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
                        val_ext01+= r_B[k1][i1] * r_B[k1][j1]
                                    * r_B[k2][i2] * r_B[k2][j2]
                                    * s_D[k1][k2][0][1];
                        val_ext10+= r_B[k1][i1] * r_B[k1][j1]
                                    * r_B[k2][i2] * r_B[k2][j2]
                                    * s_D[k1][k2][1][0];
                     }
                  }
                  if (add)
                  {
                     A_int(i1, i2, j1, j2, 0, f) += val_int0;
                     A_int(i1, i2, j1, j2, 1, f) += val_int1;
                     A_ext(i1, i2, j1, j2, 0, f) += val_ext01;
                     A_ext(i1, i2, j1, j2, 1, f) += val_ext10;
                  }
                  else
                  {
                     A_int(i1, i2, j1, j2, 0, f) = val_int0;
                     A_int(i1, i2, j1, j2, 1, f) = val_int1;
                     A_ext(i1, i2, j1, j2, 0, f) = val_ext01;
                     A_ext(i1, i2, j1, j2, 1, f) = val_ext10;
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EADGTraceAssemble3DBdr(const int NF,
                                   const Array<double> &basis,
                                   const Vector &padata,
                                   Vector &eadata_bdr,
                                   const bool add,
                                   const int d1d = 0,
                                   const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, 2, 2, NF);
   auto A_bdr = Reshape(eadata_bdr.ReadWrite(), D1D, D1D, D1D, D1D, NF);
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
      for (int i=0; i < 2; i++)
      {
         for (int j=0; j < 2; j++)
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
                  double val_bdr = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        val_bdr += r_B[k1][i1] * r_B[k1][j1]
                                   * r_B[k2][i2] * r_B[k2][j2]
                                   * s_D[k1][k2][0][0];
                     }
                  }
                  if (add)
                  {
                     A_bdr(i1, i2, j1, j2, f) += val_bdr;
                  }
                  else
                  {
                     A_bdr(i1, i2, j1, j2, f) = val_bdr;
                  }
               }
            }
         }
      }
   });
}

void DGTraceIntegrator::AssembleEAInteriorFaces(const FiniteElementSpace& fes,
                                                Vector &ea_data_int,
                                                Vector &ea_data_ext,
                                                const bool add)
{
   SetupPA(fes, FaceType::Interior);
   nf = fes.GetNFbyType(FaceType::Interior);
   if (nf==0) { return; }
   const Array<double> &B = maps->B;
   if (dim == 1)
   {
      return EADGTraceAssemble1DInt(nf,B,pa_data,ea_data_int,ea_data_ext,add);
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22:
            return EADGTraceAssemble2DInt<2,2>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x33:
            return EADGTraceAssemble2DInt<3,3>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x44:
            return EADGTraceAssemble2DInt<4,4>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x55:
            return EADGTraceAssemble2DInt<5,5>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x66:
            return EADGTraceAssemble2DInt<6,6>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x77:
            return EADGTraceAssemble2DInt<7,7>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x88:
            return EADGTraceAssemble2DInt<8,8>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x99:
            return EADGTraceAssemble2DInt<9,9>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         default:
            return EADGTraceAssemble2DInt(nf,B,pa_data,ea_data_int,
                                          ea_data_ext,add,dofs1D,quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23:
            return EADGTraceAssemble3DInt<2,3>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x34:
            return EADGTraceAssemble3DInt<3,4>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x45:
            return EADGTraceAssemble3DInt<4,5>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x56:
            return EADGTraceAssemble3DInt<5,6>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x67:
            return EADGTraceAssemble3DInt<6,7>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x78:
            return EADGTraceAssemble3DInt<7,8>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         case 0x89:
            return EADGTraceAssemble3DInt<8,9>(nf,B,pa_data,ea_data_int,
                                               ea_data_ext,add);
         default:
            return EADGTraceAssemble3DInt(nf,B,pa_data,ea_data_int,
                                          ea_data_ext,add,dofs1D,quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

void DGTraceIntegrator::AssembleEABoundaryFaces(const FiniteElementSpace& fes,
                                                Vector &ea_data_bdr,
                                                const bool add)
{
   SetupPA(fes, FaceType::Boundary);
   nf = fes.GetNFbyType(FaceType::Boundary);
   if (nf==0) { return; }
   const Array<double> &B = maps->B;
   if (dim == 1)
   {
      return EADGTraceAssemble1DBdr(nf,B,pa_data,ea_data_bdr,add);
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EADGTraceAssemble2DBdr<2,2>(nf,B,pa_data,ea_data_bdr,add);
         case 0x33: return EADGTraceAssemble2DBdr<3,3>(nf,B,pa_data,ea_data_bdr,add);
         case 0x44: return EADGTraceAssemble2DBdr<4,4>(nf,B,pa_data,ea_data_bdr,add);
         case 0x55: return EADGTraceAssemble2DBdr<5,5>(nf,B,pa_data,ea_data_bdr,add);
         case 0x66: return EADGTraceAssemble2DBdr<6,6>(nf,B,pa_data,ea_data_bdr,add);
         case 0x77: return EADGTraceAssemble2DBdr<7,7>(nf,B,pa_data,ea_data_bdr,add);
         case 0x88: return EADGTraceAssemble2DBdr<8,8>(nf,B,pa_data,ea_data_bdr,add);
         case 0x99: return EADGTraceAssemble2DBdr<9,9>(nf,B,pa_data,ea_data_bdr,add);
         default:
            return EADGTraceAssemble2DBdr(nf,B,pa_data,ea_data_bdr,add,dofs1D,quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: return EADGTraceAssemble3DBdr<2,3>(nf,B,pa_data,ea_data_bdr,add);
         case 0x34: return EADGTraceAssemble3DBdr<3,4>(nf,B,pa_data,ea_data_bdr,add);
         case 0x45: return EADGTraceAssemble3DBdr<4,5>(nf,B,pa_data,ea_data_bdr,add);
         case 0x56: return EADGTraceAssemble3DBdr<5,6>(nf,B,pa_data,ea_data_bdr,add);
         case 0x67: return EADGTraceAssemble3DBdr<6,7>(nf,B,pa_data,ea_data_bdr,add);
         case 0x78: return EADGTraceAssemble3DBdr<7,8>(nf,B,pa_data,ea_data_bdr,add);
         case 0x89: return EADGTraceAssemble3DBdr<8,9>(nf,B,pa_data,ea_data_bdr,add);
         default:
            return EADGTraceAssemble3DBdr(nf,B,pa_data,ea_data_bdr,add,dofs1D,quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

}
