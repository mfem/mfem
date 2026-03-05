// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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

// For H(div) mass, Bo and Bc are the basis evaluation operators, and the
// pa_data corresponds to a (potentially symmetric) matrix coefficient.
// coeff_dim must be 3 or 4 depending on symmetry.
//
// For div-div, Bc is the derivative evaluation operator, and pa_data
// corresponds to a scalar coefficient. coeff_dim must be 1.
//
// These two integrators are distinguished using coeff_dim.
template<int T_D1D = 0, int T_Q1D = 0>
static void EAHdivAssemble2D(const int NE,
                             const Array<real_t> &Bo_,
                             const Array<real_t> &Bc_,
                             const int coeff_dim,
                             const Vector &pa_data,
                             Vector &ea_data,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D, "");
   const int NDOF = 2*(D1D-1)*D1D;
   const auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, coeff_dim, NE);
   const bool symmetric = (coeff_dim == 3);
   auto M = Reshape(add ? ea_data.ReadWrite() : ea_data.Write(), NDOF, NDOF, NE);
   mfem::forall_2D(NE, NDOF, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      // Load Bo and Bc matrices into registers
      real_t r_Bo[MQ1][MD1];
      real_t r_Bc[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            if (d < D1D - 1) { r_Bo[q][d] = Bo(q,d); }
            r_Bc[q][d] = Bc(q,d);
         }
      }
      // Store PA data in shared memory
      MFEM_SHARED real_t s_D[4][MQ1][MQ1];
      MFEM_FOREACH_THREAD(idx_q, x, Q1D*Q1D)
      {
         const int qx = idx_q % Q1D;
         const int qy = idx_q / Q1D;
         if (coeff_dim == 1)
         {
            const real_t val = D(qx, qy, 0, e);
            for (int i = 0; i < 4; ++i) { s_D[i][qx][qy] = val; }
         }
         else
         {
            s_D[0][qx][qy] = D(qx, qy, 0, e);
            s_D[1][qx][qy] = D(qx, qy, 1, e);
            s_D[2][qx][qy] = (symmetric) ? s_D[1][qx][qy] : D(qx, qy, 2, e);
            s_D[3][qx][qy] = (symmetric) ? D(qx, qy, 2, e) : D(qx, qy, 3, e);
         }
      }
      MFEM_SYNC_THREAD;
      // Assemble (one row per thread)
      MFEM_FOREACH_THREAD(idx_i, x, NDOF)
      {
         const int ic = idx_i / D1D / (D1D-1);
         const int idx_ii = idx_i % (D1D * (D1D-1));
         const int ix = (ic == 0) ? idx_ii%D1D : idx_ii%(D1D-1);
         const int iy = (ic == 0) ? idx_ii/D1D : idx_ii/(D1D-1);

         const real_t (&Bi1)[MQ1][MD1] = (ic == 0) ? r_Bc : r_Bo;
         const real_t (&Bi2)[MQ1][MD1] = (ic == 0) ? r_Bo : r_Bc;

         for (int idx_j = 0; idx_j < NDOF; ++idx_j)
         {
            const int jc = idx_j / (D1D*(D1D-1));
            const int idx_jj = idx_j % (D1D * (D1D-1));
            const int jx = (jc == 0) ? idx_jj%D1D : idx_jj%(D1D-1);
            const int jy = (jc == 0) ? idx_jj/D1D : idx_jj/(D1D-1);

            const real_t (&Bj1)[MQ1][MD1] = (jc == 0) ? r_Bc : r_Bo;
            const real_t (&Bj2)[MQ1][MD1] = (jc == 0) ? r_Bo : r_Bc;

            real_t val = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  const double coeff = s_D[ic + jc*2][qx][qy];
                  val += coeff*Bi1[qx][ix]*Bi2[qy][iy]*Bj1[qx][jx]*Bj2[qy][jy];
               }
            }
            if (add)
            {
               M(idx_i, idx_j, e) += val;
            }
            else
            {
               M(idx_i, idx_j, e) = val;
            }
         }
      }
   });
}

// For H(div) mass, Bo and Bc are the basis evaluation operators, and the
// pa_data corresponds to a (potentially symmetric) matrix coefficient.
// coeff_dim must be 6 or 9 depending on symmetry.
//
// For div-div, Bc is the derivative evaluation operator, and pa_data
// corresponds to a scalar coefficient. coeff_dim must be 1.
//
// These two integrators are distinguished using coeff_dim.
template<int T_D1D = 0, int T_Q1D = 0>
static void EAHdivAssemble3D(const int NE,
                             const Array<real_t> &Bo_,
                             const Array<real_t> &Bc_,
                             const int coeff_dim,
                             const Vector &pa_data,
                             Vector &ea_data,
                             const bool add,
                             const int d1d = 0,
                             const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   MFEM_VERIFY(D1D <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D, "");
   MFEM_VERIFY(Q1D <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D, "");
   const int NDOF_C = (D1D-1)*(D1D-1)*D1D;
   const int NDOF = 3*NDOF_C;
   const auto Bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto Bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto D = Reshape(pa_data.Read(), Q1D, Q1D, Q1D, coeff_dim, NE);
   const bool symmetric = (coeff_dim == 6);
   auto M = Reshape(add ? ea_data.ReadWrite() : ea_data.Write(), NDOF, NDOF, NE);
   mfem::forall_2D(NE, NDOF, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      // Load Bo and Bc matrices into registers
      real_t r_Bo[MQ1][MD1];
      real_t r_Bc[MQ1][MD1];
      for (int d = 0; d < D1D; d++)
      {
         for (int q = 0; q < Q1D; q++)
         {
            if (d < D1D - 1) { r_Bo[q][d] = Bo(q,d); }
            r_Bc[q][d] = Bc(q,d);
         }
      }
      // Store PA data in shared memory
      MFEM_SHARED real_t s_D[9][MQ1][MQ1][MQ1];
      MFEM_FOREACH_THREAD(idx_q, x, Q1D*Q1D*Q1D)
      {
         const int qx = idx_q % Q1D;
         const int qy = (idx_q / Q1D) % Q1D;
         const int qz = (idx_q / Q1D) / Q1D;
         if (coeff_dim == 1)
         {
            const real_t val = D(qx,qy,qz,0,e);
            for (int i = 0; i < 9; ++i) { s_D[i][qx][qy][qz] = val; }
         }
         else
         {
            s_D[0][qx][qy][qz] = D(qx,qy,qz,0,e);
            s_D[1][qx][qy][qz] = D(qx,qy,qz,1,e);
            s_D[2][qx][qy][qz] = D(qx,qy,qz,2,e);
            s_D[3][qx][qy][qz] = symmetric ? s_D[1][qx][qy][qz] : D(qx,qy,qz,3,e);
            s_D[4][qx][qy][qz] = symmetric ? D(qx,qy,qz,3,e) : D(qx,qy,qz,4,e);
            s_D[5][qx][qy][qz] = symmetric ? D(qx,qy,qz,4,e) : D(qx,qy,qz,5,e);
            s_D[6][qx][qy][qz] = symmetric ? s_D[2][qx][qy][qz] : D(qx,qy,qz,6,e);
            s_D[7][qx][qy][qz] = symmetric ? s_D[5][qx][qy][qz] : D(qx,qy,qz,7,e);
            s_D[8][qx][qy][qz] = symmetric ? D(qx,qy,qz,5,e) : D(qx,qy,qz,8,e);
         }
      }
      MFEM_SYNC_THREAD;
      // Assemble (one row per thread)
      MFEM_FOREACH_THREAD(idx_i, x, NDOF)
      {
         const int ic = idx_i / NDOF_C;
         const int idx_ii = idx_i % NDOF_C;

         const int nx_i = (ic == 0) ? D1D : D1D-1;
         const int ny_i = (ic == 1) ? D1D : D1D-1;

         const int ix = idx_ii % nx_i;
         const int iy = (idx_ii / nx_i) % ny_i;
         const int iz = (idx_ii / nx_i) / ny_i;

         const real_t (&Bi1)[MQ1][MD1] = (ic == 0) ? r_Bc : r_Bo;
         const real_t (&Bi2)[MQ1][MD1] = (ic == 1) ? r_Bc : r_Bo;
         const real_t (&Bi3)[MQ1][MD1] = (ic == 2) ? r_Bc : r_Bo;

         for (int idx_j = 0; idx_j < NDOF; ++idx_j)
         {
            const int jc = idx_j / NDOF_C;
            const int idx_jj = idx_j % NDOF_C;

            const int nx_j = (jc == 0) ? D1D : D1D-1;
            const int ny_j = (jc == 1) ? D1D : D1D-1;

            const int jx = idx_jj % nx_j;
            const int jy = (idx_jj / nx_j) % ny_j;
            const int jz = (idx_jj / nx_j) / ny_j;

            const real_t (&Bj1)[MQ1][MD1] = (jc == 0) ? r_Bc : r_Bo;
            const real_t (&Bj2)[MQ1][MD1] = (jc == 1) ? r_Bc : r_Bo;
            const real_t (&Bj3)[MQ1][MD1] = (jc == 2) ? r_Bc : r_Bo;

            real_t val = 0.0;
            for (int qx = 0; qx < Q1D; ++qx)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qz = 0; qz < Q1D; ++qz)
                  {
                     const double coeff = s_D[ic + jc*3][qx][qy][qz];
                     val += coeff*Bi1[qx][ix]*Bi2[qy][iy]*Bi3[qz][iz]*
                            Bj1[qx][jx]*Bj2[qy][jy]*Bj3[qz][jz];
                  }
               }
            }
            if (add)
            {
               M(idx_i, idx_j, e) += val;
            }
            else
            {
               M(idx_i, idx_j, e) = val;
            }
         }
      }
   });
}

void VectorFEMassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                        Vector &ea_data,
                                        const bool add)
{
   AssemblePA(fes);

   if (trial_fetype != mfem::FiniteElement::DIV ||
       test_fetype != mfem::FiniteElement::DIV)
   {
      MFEM_ABORT("Unsupported kernel.");
   }

   const Array<real_t> &Bo = mapsO->B;
   const Array<real_t> &Bc = mapsC->B;

   if (dim == 2)
   {
      const int coeff_dim = symmetric ? 3 : 4;
      auto kernel = EAHdivAssemble2D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: kernel = EAHdivAssemble2D<2,2>; break;
         case 0x33: kernel = EAHdivAssemble2D<3,3>; break;
         case 0x44: kernel = EAHdivAssemble2D<4,4>; break;
         case 0x55: kernel = EAHdivAssemble2D<5,5>; break;
      }
      return kernel(ne,Bo,Bc,coeff_dim,pa_data,ea_data,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      const int coeff_dim = symmetric ? 6 : 9;
      auto kernel = EAHdivAssemble3D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: kernel = EAHdivAssemble3D<2,3>; break;
         case 0x34: kernel = EAHdivAssemble3D<3,4>; break;
         case 0x45: kernel = EAHdivAssemble3D<4,5>; break;
         case 0x56: kernel = EAHdivAssemble3D<5,6>; break;
      }
      return kernel(ne,Bo,Bc,coeff_dim,pa_data,ea_data,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

void DivDivIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                  Vector &ea_data,
                                  const bool add)
{
   AssemblePA(fes);

   const Array<real_t> &Bo = mapsO->B;
   const Array<real_t> &Gc = mapsC->G;

   if (dim == 2)
   {
      auto kernel = EAHdivAssemble2D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: kernel = EAHdivAssemble2D<2,2>; break;
         case 0x33: kernel = EAHdivAssemble2D<3,3>; break;
         case 0x44: kernel = EAHdivAssemble2D<4,4>; break;
         case 0x55: kernel = EAHdivAssemble2D<5,5>; break;
      }
      return kernel(ne,Bo,Gc,1,pa_data,ea_data,add,dofs1D,quad1D);
   }
   else if (dim == 3)
   {
      auto kernel = EAHdivAssemble3D<0,0>;
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: kernel = EAHdivAssemble3D<2,3>; break;
         case 0x34: kernel = EAHdivAssemble3D<3,4>; break;
         case 0x45: kernel = EAHdivAssemble3D<4,5>; break;
         case 0x56: kernel = EAHdivAssemble3D<5,6>; break;
      }
      return kernel(ne,Bo,Gc,1,pa_data,ea_data,add,dofs1D,quad1D);
   }
   MFEM_ABORT("Unknown kernel.");
}

}
