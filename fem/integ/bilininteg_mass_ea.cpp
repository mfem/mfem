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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble1D(const int NE,
                             const Array<real_t> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(), D1D, D1D, NE);
   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t r_Bi[MQ1];
      real_t r_Bj[MQ1];
      for (int q = 0; q < Q1D; q++)
      {
         r_Bi[q] = B(q,MFEM_THREAD_ID(x));
         r_Bj[q] = B(q,MFEM_THREAD_ID(y));
      }
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            real_t val = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val += r_Bi[k1] * r_Bj[k1] * D(k1, e);
            }
            if (add)
            {
               M(i1, j1, e) += val;
            }
            else
            {
               M(i1, j1, e) = val;
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble2D(const int NE,
                             const Array<real_t> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(), D1D, D1D, D1D, D1D,
                    NE);
   mfem::forall_2D(NE, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      real_t r_B[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            r_B[q][d] = B(q,d);
         }
      }
      MFEM_SHARED real_t s_D[MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            s_D[k1][k2] = D(k1,k2,e);
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
                  real_t val = 0.0;
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        val += r_B[k1][i1] * r_B[k1][j1]
                               * r_B[k2][i2] * r_B[k2][j2]
                               * s_D[k1][k2];
                     }
                  }
                  if (add)
                  {
                     M(i1, i2, j1, j2, e) += val;
                  }
                  else
                  {
                     M(i1, i2, j1, j2, e) = val;
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble3D(const int NE,
                             const Array<real_t> &basis,
                             const Vector &padata,
                             Vector &eadata,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().MAX_Q1D, "");
   auto B = Reshape(basis.Read(), Q1D, D1D);
   auto D = Reshape(padata.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(add ? eadata.ReadWrite() : eadata.Write(), D1D, D1D, D1D, D1D,
                    D1D, D1D, NE);
   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int DQ = T_D1D * T_Q1D;

      // For quadratic and lower it's better to use registers but for higher-order you start to
      // spill and it's better to use shared memory
      constexpr bool USE_REG = DQ != 0 && DQ <= 12;
      constexpr int MD1r = USE_REG ? MD1 : 1;
      constexpr int MQ1r = USE_REG ? MQ1 : 1;
      constexpr int MD1s = USE_REG ? 1 : MD1;
      constexpr int MQ1s = USE_REG ? 1 : MQ1;

      MFEM_SHARED real_t s_B[MQ1s][MD1s];
      real_t r_B[MQ1r][MD1r];
      real_t (*l_B)[MD1] = nullptr;
      if (USE_REG)
      {
         for (int d = 0; d < D1D; d++)
         {
            for (int q = 0; q < Q1D; q++)
            {
               r_B[q][d] = B(q,d);
            }
         }
         l_B = (real_t (*)[MD1])r_B;
      }
      else
      {
         if (MFEM_THREAD_ID(z) == 0)
         {
            MFEM_FOREACH_THREAD(d,x,D1D)
            {
               MFEM_FOREACH_THREAD(q,y,Q1D)
               {
                  s_B[q][d] = B(q,d);
               }
            }
         }
         l_B = (real_t (*)[MD1])s_B;
      }

      MFEM_SHARED real_t s_D[MQ1][MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            MFEM_FOREACH_THREAD(k3,z,Q1D)
            {
               s_D[k1][k2][k3] = D(k1,k2,k3,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        real_t val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 val += l_B[k1][i1] * l_B[k1][j1]
                                        * l_B[k2][i2] * l_B[k2][j2]
                                        * l_B[k3][i3] * l_B[k3][j3]
                                        * s_D[k1][k2][k3];
                              }
                           }
                        }
                        if (add)
                        {
                           M(i1, i2, i3, j1, j2, j3, e) += val;
                        }
                        else
                        {
                           M(i1, i2, i3, j1, j2, j3, e) = val;
                        }
                     }
                  }
               }
            }
         }
      }
   });
}

void MassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                Vector &ea_data,
                                const bool add)
{
   AssemblePA(fes);
   ne = fes.GetMesh()->GetNE();
   const Array<real_t> &B = maps->B;
   if (dim == 1)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EAMassAssemble1D<2,2>(ne,B,pa_data,ea_data,add);
         case 0x33: return EAMassAssemble1D<3,3>(ne,B,pa_data,ea_data,add);
         case 0x44: return EAMassAssemble1D<4,4>(ne,B,pa_data,ea_data,add);
         case 0x55: return EAMassAssemble1D<5,5>(ne,B,pa_data,ea_data,add);
         case 0x66: return EAMassAssemble1D<6,6>(ne,B,pa_data,ea_data,add);
         case 0x77: return EAMassAssemble1D<7,7>(ne,B,pa_data,ea_data,add);
         case 0x88: return EAMassAssemble1D<8,8>(ne,B,pa_data,ea_data,add);
         case 0x99: return EAMassAssemble1D<9,9>(ne,B,pa_data,ea_data,add);
         default:   return EAMassAssemble1D(ne,B,pa_data,ea_data,add,
                                               dofs1D,quad1D);
      }
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EAMassAssemble2D<2,2>(ne,B,pa_data,ea_data,add);
         case 0x33: return EAMassAssemble2D<3,3>(ne,B,pa_data,ea_data,add);
         case 0x44: return EAMassAssemble2D<4,4>(ne,B,pa_data,ea_data,add);
         case 0x55: return EAMassAssemble2D<5,5>(ne,B,pa_data,ea_data,add);
         case 0x66: return EAMassAssemble2D<6,6>(ne,B,pa_data,ea_data,add);
         case 0x77: return EAMassAssemble2D<7,7>(ne,B,pa_data,ea_data,add);
         case 0x88: return EAMassAssemble2D<8,8>(ne,B,pa_data,ea_data,add);
         case 0x99: return EAMassAssemble2D<9,9>(ne,B,pa_data,ea_data,add);
         default:   return EAMassAssemble2D(ne,B,pa_data,ea_data,add,
                                               dofs1D,quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: return EAMassAssemble3D<2,3>(ne,B,pa_data,ea_data,add);
         case 0x34: return EAMassAssemble3D<3,4>(ne,B,pa_data,ea_data,add);
         case 0x45: return EAMassAssemble3D<4,5>(ne,B,pa_data,ea_data,add);
         case 0x56: return EAMassAssemble3D<5,6>(ne,B,pa_data,ea_data,add);
         case 0x67: return EAMassAssemble3D<6,7>(ne,B,pa_data,ea_data,add);
         case 0x78: return EAMassAssemble3D<7,8>(ne,B,pa_data,ea_data,add);
         case 0x89: return EAMassAssemble3D<8,9>(ne,B,pa_data,ea_data,add);
         default:   return EAMassAssemble3D(ne,B,pa_data,ea_data,add,
                                               dofs1D,quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

}
