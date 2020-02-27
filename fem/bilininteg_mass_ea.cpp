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

#include "../general/forall.hpp"
#include "bilininteg.hpp"
#include "gridfunc.hpp"
#include "libceed/mass.hpp"
#include "../linalg/elementmatrix.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble1D(const int NE,
                             const Array<double> &b,
                             const Vector &d,
                             Vector &y,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, NE);
   auto M = Reshape(y.Write(), D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, 1,
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
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
            double val = 0.0;
            for (int k1 = 0; k1 < Q1D; ++k1)
            {
               val += r_Bi[k1] * r_Bj[k1] * D(k1, e);
            }
            M(i1, j1, e) = val;
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble2D(const int NE,
                             const Array<double> &b,
                             const Vector &d,
                             Vector &y,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, Q1D, NE);
   auto M = Reshape(y.Write(), D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, 1,
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
      MFEM_SHARED double s_D[MQ1][MQ1];
      MFEM_FOREACH_THREAD(k1,x,Q1D)
      {
         MFEM_FOREACH_THREAD(k2,y,Q1D)
         {
            s_D[k1][k2] = D(k1,k2,e);
         }
      }
      MFEM_SYNC_THREAD;
      double C[MQ1];
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            for (int k2 = 0; k2 < Q1D; ++k2)
            {
               C[k2] = 0.0;
               for (int k1 = 0; k1 < Q1D; ++k1)
               {
                  C[k2] += r_B[k1][i1] * r_B[k1][j1] * s_D[k1][k2];
               }
            }
         }
      }
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(j1,y,D1D)
         {
            for (int i2 = 0; i2 < D1D; ++i2)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  double val = 0.0;
                  for (int k2 = 0; k2 < Q1D; ++k2)
                  {
                     val += r_B[k2][i2] * r_B[k2][j2] * C[k2];
                  }
                  M(i1, i2, j1, j2, e) = val;
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble3D0D(const int NE,
                               const Array<double> &b,
                               const Vector &d,
                               Vector &y,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(y.Write(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, D1D,
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
      MFEM_SHARED double s_D[MQ1][MQ1][MQ1];
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
                        double val = 0.0;
                        for (int k1 = 0; k1 < Q1D; ++k1)
                        {
                           for (int k2 = 0; k2 < Q1D; ++k2)
                           {
                              for (int k3 = 0; k3 < Q1D; ++k3)
                              {
                                 val += r_B[k1][i1] * r_B[k1][j1]
                                      * r_B[k2][i2] * r_B[k2][j2]
                                      * r_B[k3][i3] * r_B[k3][j3]
                                      * s_D[k1][k2][k3];
                              }
                           }
                        }
                        M(i1, i2, i3, j1, j2, j3, e) = val;
                     }
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble3D1D(const int NE,
                               const Array<double> &b,
                               const Vector &d,
                               Vector &y,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(y.Write(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, D1D,
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
      MFEM_SHARED double s_D[MQ1][MQ1][MQ1];
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
      double val[MD1];
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j2 = 0; j2 < D1D; ++j2)
               {
                  for (int j3 = 0; j3 < D1D; ++j3)
                  {
                     for (int j1 = 0; j1 < D1D; ++j1)
                     {
                        val[j1] = 0.0;
                     }
                     for (int k1 = 0; k1 < Q1D; ++k1)
                     {
                        for (int k2 = 0; k2 < Q1D; ++k2)
                        {
                           for (int k3 = 0; k3 < Q1D; ++k3)
                           {
                              double tmp = r_B[k1][i1]
                                         * r_B[k2][i2] * r_B[k2][j2]
                                         * r_B[k3][i3] * r_B[k3][j3]
                                         * s_D[k1][k2][k3];
                              for (int j1 = 0; j1 < D1D; ++j1)
                              {
                                 val[j1] += r_B[k1][j1] * tmp;
                              }
                           }
                        }
                     }
                     for (int j1 = 0; j1 < D1D; ++j1)
                     {
                        M(i1, i2, i3, j1, j2, j3, e) = val[j1];
                     }
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble3D2D(const int NE,
                               const Array<double> &b,
                               const Vector &d,
                               Vector &y,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(y.Write(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, D1D,
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
      MFEM_SHARED double s_D[MQ1][MQ1][MQ1];
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
      double val[MD1][MD1];
      MFEM_FOREACH_THREAD(i1,x,D1D)
      {
         MFEM_FOREACH_THREAD(i2,y,D1D)
         {
            MFEM_FOREACH_THREAD(i3,z,D1D)
            {
               for (int j3 = 0; j3 < D1D; ++j3)
               {
                  for (int j1 = 0; j1 < D1D; ++j1)
                  {
                     for (int j2 = 0; j2 < D1D; ++j2)
                     {
                        val[j1][j2] = 0.0;
                     }
                  }
                  for (int k1 = 0; k1 < Q1D; ++k1)
                  {
                     for (int k2 = 0; k2 < Q1D; ++k2)
                     {
                        for (int k3 = 0; k3 < Q1D; ++k3)
                        {
                           double tmp = r_B[k1][i1]
                                      * r_B[k2][i2]
                                      * r_B[k3][i3] * r_B[k3][j3]
                                      * s_D[k1][k2][k3];
                           for (int j1 = 0; j1 < D1D; ++j1)
                           {
                              for (int j2 = 0; j2 < D1D; ++j2)
                              {
                                 val[j1][j2] += r_B[k1][j1] * r_B[k2][j2] * tmp;
                              }
                           }
                        }
                     }
                  }
                  for (int j1 = 0; j1 < D1D; ++j1)
                  {
                     for (int j2 = 0; j2 < D1D; ++j2)
                     {
                        M(i1, i2, i3, j1, j2, j3, e) = val[j1][j2];
                     }
                  }
               }
            }
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0>
static void EAMassAssemble3D3D(const int NE,
                               const Array<double> &b,
                               const Vector &d,
                               Vector &y,
                               const int d1d = 0,
                               const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= MAX_D1D, "");
   MFEM_VERIFY(Q1D <= MAX_Q1D, "");
   auto B = Reshape(b.Read(), Q1D, D1D);
   auto D = Reshape(d.Read(), Q1D, Q1D, Q1D, NE);
   auto M = Reshape(y.Write(), D1D, D1D, D1D, D1D, D1D, D1D, NE);
   MFEM_FORALL_3D(e, NE, D1D, D1D, D1D,
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
      MFEM_SHARED double s_D[MQ1][MQ1][MQ1];
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
      double val[MD1][MD1][MD1];
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
                        val[j1][j2][j3] = 0.0;
                     }
                  }
               }
               for (int k1 = 0; k1 < Q1D; ++k1)
               {
                  for (int k2 = 0; k2 < Q1D; ++k2)
                  {
                     for (int k3 = 0; k3 < Q1D; ++k3)
                     {
                        double tmp = r_B[k1][i1]
                                   * r_B[k2][i2]
                                   * r_B[k3][i3]
                                   * s_D[k1][k2][k3];
                        for (int j1 = 0; j1 < D1D; ++j1)
                        {
                           for (int j2 = 0; j2 < D1D; ++j2)
                           {
                              for (int j3 = 0; j3 < D1D; ++j3)
                              {
                                 val[j1][j2][j3] += r_B[k1][j1] * r_B[k2][j2] * r_B[k3][j3] * tmp;
                              }
                           }
                        }
                     }
                  }
               }
               for (int j1 = 0; j1 < D1D; ++j1)
               {
                  for (int j2 = 0; j2 < D1D; ++j2)
                  {
                     for (int j3 = 0; j3 < D1D; ++j3)
                     {
                        M(i1, i2, i3, j1, j2, j3, e) = val[j1][j2][j3];
                     }
                  }
               }
            }
         }
      }
   });
}

ElementMatrix MassIntegrator::AssembleEA(const FiniteElementSpace &fes)
{
   const int ne = fes.GetMesh()->GetNE();
   const int ndofs = fes.GetFE(0)->GetDof();
   ea_data.SetSize(ne*ndofs*ndofs, Device::GetMemoryType());
   AssemblePA(fes);
   ElementMatrix emat(ea_data.ReadWrite(), ne, ndofs);

   return emat;
}

}
