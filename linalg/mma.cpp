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

#include "mma.hpp"

#include "vector.hpp"
#include "../general/communication.hpp"
#include "../general/error.hpp"

#include <fstream>
#include <math.h>

namespace
{
// check if C++ 14 or beyond
#if __cplusplus >= 201402L
inline ::std::unique_ptr<::mfem::real_t[]> allocArray(int size) { return ::std::make_unique<::mfem::real_t[]>(size); }
#else
inline ::std::unique_ptr<::mfem::real_t[]> allocArray(int size) { return ::std::unique_ptr<::mfem::real_t[]>(new ::mfem::real_t[size]); }
#endif
}

#ifdef MFEM_USE_LAPACK
extern "C" void dgesv_(int* nLAP, int* nrhs, double* AA, int* lda,
                       int* ipiv,
                       double* bb, int* ldb, int* info);

extern "C" void sgesv_(int* nLAP, int* nrhs, float* AA, int* lda,
                       int* ipiv,
                       float* bb, int* ldb, int* info);
#endif

namespace mfem
{

void solveLU(int nCon, real_t* AA1, real_t* bb1)
{
   // Solve linear system with LU decomposition ifndef LAPACK
   int nLAP = nCon + 1;

   // Convert AA1 to matrix A and bb1 to vector B
   ::std::unique_ptr<::std::unique_ptr<real_t[]>[]> A(
      new ::std::unique_ptr<real_t[]>[nLAP]);
   for (int i = 0; i < nLAP; ++i)
   {
      A[i] = allocArray(nLAP);
   }
   ::std::unique_ptr<real_t[]> B = allocArray(nLAP);
   for (int i = 0; i < nLAP; ++i)
   {
      for (int j = 0; j < nLAP; ++j)
      {
         A[i][j] = AA1[j * nLAP + i];
      }
      B[i] = bb1[i];
   }

   // Perform LU decomposition
   ::std::unique_ptr<::std::unique_ptr<real_t[]>[]> L(
      new ::std::unique_ptr<real_t[]>[nLAP]),
          U(new ::std::unique_ptr<real_t[]>[nLAP]);
   for (int i = 0; i < nLAP; ++i)
   {
      L[i] = allocArray(nLAP);
      U[i] = allocArray(nLAP);
      for (int j = 0; j < nLAP; ++j)
      {
         L[i][j] = 0.0;
         U[i][j] = 0.0;
      }
   }

   for (int i = 0; i < nLAP; ++i)
   {
      for (int k = i; k < nLAP; ++k)
      {
         real_t sum = 0.0;
         for (int j = 0; j < i; ++j)
         {
            sum += (L[i][j] * U[j][k]);
         }
         U[i][k] = A[i][k] - sum;
      }
      for (int k = i; k < nLAP; ++k)
      {
         if (i == k)
         {
            L[i][i] = 1.0;
         }
         else
         {
            real_t sum = 0.0;
            for (int j = 0; j < i; ++j)
            {
               sum += (L[k][j] * U[j][i]);
            }
            L[k][i] = (A[k][i] - sum) / U[i][i];
         }
      }
   }

   // Check for singular matrix
   for (int i = 0; i < nLAP; ++i)
   {
      if (U[i][i] == 0.0)
      {
         MFEM_ABORT("Error: matrix in MMA LU Solve is singular.");
      }
   }

   // Forward substitution to solve L * Y = B
   ::std::unique_ptr<real_t[]> Y=allocArray(nLAP);
   for (int i = 0; i < nLAP; ++i)
   {
      real_t sum = 0.0;
      for (int j = 0; j < i; ++j)
      {
         sum += L[i][j] * Y[j];
      }
      Y[i] = (B[i] - sum) / L[i][i];
   }

   // Backward substitution to solve U * X = Y
   ::std::unique_ptr<real_t[]> X=allocArray(nLAP);
   for (int i = nLAP - 1; i >= 0; --i)
   {
      real_t sum = 0.0;
      for (int j = i + 1; j < nLAP; ++j)
      {
         sum += U[i][j] * X[j];
      }
      X[i] = (Y[i] - sum) / U[i][i];
   }


   // Copy results back to bb1
   for (int i = 0; i < (nCon + 1); i++)
   {
      bb1[i] = X[i];
   }
}

void MMA::MMASubSvanberg::AllocSubData(int nvar, int ncon)
{
   epsi = 1.0;
   ittt = itto = itera = 0;
   raa0 = 0.00001;
   move = 0.5;
   albefa = 0.1;
   xmamieps = 1e-5;
   ux1 = allocArray(nvar); // ini
   xl1 = allocArray(nvar); // ini
   plam = allocArray(nvar); // ini
   qlam = allocArray(nvar); // ini
   gvec = allocArray(ncon); // ini
   residu = allocArray(3 * nvar + 4 * ncon + 2); // ini
   GG = allocArray(nvar * ncon); // ini
   delx = allocArray(nvar); // ini
   dely = allocArray(ncon); // ini
   dellam = allocArray(ncon); // ini
   dellamyi = allocArray(ncon);
   diagx = allocArray(nvar); // ini
   diagy = allocArray(ncon); // ini
   diaglamyi = allocArray(ncon); // ini
   bb = allocArray(nvar + 1);
   bb1 = allocArray(ncon + 1); // ini
   Alam = allocArray(ncon * ncon); // ini
   AA = allocArray((nvar + 1) * (nvar + 1));
   AA1 = allocArray((ncon + 1) * (ncon + 1)); // ini
   dlam = allocArray(ncon); // ini
   dx = allocArray(nvar); // ini
   dy = allocArray(ncon); // ini
   dxsi = allocArray(nvar); // ini
   deta = allocArray(nvar); // ini
   dmu = allocArray(ncon); // ini
   Axx = allocArray(nvar * ncon); // ini
   axz = allocArray(nvar); // ini
   ds = allocArray(ncon); // ini
   xx = allocArray(4 * ncon + 2 * nvar + 2); // ini
   dxx = allocArray(4 * ncon + 2 * nvar + 2); // ini
   stepxx = allocArray(4 * ncon + 2 * nvar + 2); // ini
   sum = 0;
   sum1 = allocArray(nvar);
   stepalfa = allocArray(nvar); // ini
   stepbeta = allocArray(nvar); // ini
   xold = allocArray(nvar); // ini
   yold = allocArray(ncon); // ini
   lamold = allocArray(ncon); // ini
   xsiold = allocArray(nvar); // ini
   etaold = allocArray(nvar); // ini
   muold = allocArray(ncon); // ini
   sold = allocArray(ncon); // ini
   q0 = allocArray(nvar); // ini
   p0 = allocArray(nvar); // ini
   P = allocArray(ncon * nvar); // ini
   Q = allocArray(ncon * nvar); // ini
   alfa = allocArray(nvar); // ini
   beta = allocArray(nvar); // ini
   xmami = allocArray(nvar);
   b = allocArray(ncon); // ini

   b_local = allocArray(ncon);
   gvec_local = allocArray(ncon);
   Alam_local = allocArray(ncon * ncon);
   sum_local = allocArray(ncon);
   sum_global = allocArray(ncon);


   for (int i=0; i<(3 * nvar + 4 * ncon + 2); i++)
   {
      residu[i]=0.0;
   }
}

void MMA::MMASubSvanberg::Update(const real_t* dfdx,
                                 const real_t* gx,
                                 const real_t* dgdx,
                                 const real_t* xmin,
                                 const real_t* xmax,
                                 const real_t* xval)
{
   MMA& mma = this->mma_ref;

   int rank = 0;
#ifdef MFEM_USE_MPI
   MPI_Comm_rank(mma.comm, &rank);
#endif

   int ncon = mma.nCon;
   int nvar = mma.nVar;

   real_t zero = 0.0;

   ittt = 0;
   itto = 0;
   epsi = 1.0;
   itera = 0;
   mma.z = 1.0;
   mma.zet = 1.0;

   for (int i = 0; i < ncon; i++)
   {
      b[i] = 0.0;
      b_local[i] = 0.0;
   }

   for (int i = 0; i < nvar; i++)
   {
      // Calculation of bounds alfa and beta according to:
      // alfa = max{xmin, low + 0.1(xval-low), xval-0.5(xmax-xmin)}
      // beta = min{xmax, upp - 0.1(upp-xval), xval+0.5(xmax-xmin)}

      alfa[i] = std::max(std::max(mma.low[i] + albefa * (xval[i] - mma.low[i]),
                                  xval[i] - move * (xmax[i] - xmin[i])), xmin[i]);
      beta[i] = std::min(std::min(mma.upp[i] - albefa * (mma.upp[i] - xval[i]),
                                  xval[i] + move * (xmax[i] - xmin[i])), xmax[i]);
      xmami[i] = std::max(xmax[i] - xmin[i], xmamieps);

      // Calculations of p0, q0, P, Q, and b
      ux1[i] = mma.upp[i] - xval[i];
      if (std::fabs(ux1[i]) <= mma.machineEpsilon)
      {
         ux1[i] = mma.machineEpsilon;
      }
      xl1[i] = xval[i] - mma.low[i];
      if (std::fabs(xl1[i]) <= mma.machineEpsilon)
      {
         xl1[i] = mma.machineEpsilon;
      }
      p0[i] = ( std::max(dfdx[i], zero) + 0.001 * (std::max(dfdx[i],
                                                            zero) + std::max(-dfdx[i], zero)) + raa0 / xmami[i]) * ux1[i] * ux1[i];
      q0[i] = ( std::max(-dfdx[i], zero) + 0.001 * (std::max(dfdx[i],
                                                             zero) + std::max(-dfdx[i], zero)) + raa0 / xmami[i]) * xl1[i] * xl1[i];
   }

   // P = max(dgdx,0)
   // Q = max(-dgdx,0)
   // P = P + 0.001(P+Q) + raa0/xmami
   // Q = Q + 0.001(P+Q) + raa0/xmami
   for (int i = 0; i < ncon; i++)
   {
      for (int j = 0; j < nvar; j++)
      {
         // P = P * spdiags(ux2,0,n,n)
         // Q = Q * spdiags(xl2,0,n,n)
         P[i * nvar + j] = (std::max(dgdx[i * nvar + j],
                                     zero) + 0.001 * (std::max(dgdx[i * nvar + j],
                                                               zero) + std::max(-1*dgdx[i * nvar + j],
                                                                                zero)) + raa0 / xmami[j]) * ux1[j] * ux1[j];
         Q[i * nvar + j] = (std::max(-1*dgdx[i * nvar + j],
                                     zero) + 0.001 * (std::max(dgdx[i * nvar + j],
                                                               zero) + std::max(-1*dgdx[i * nvar + j],
                                                                                zero)) + raa0 / xmami[j]) * xl1[j] * xl1[j];
         // b = P/ux1 + Q/xl1 - gx
         b_local[i] = b_local[i] + P[i * nvar + j] / ux1[j] + Q[i * nvar + j] / xl1[j];
      }
   }

   std::copy(b_local.get(), b_local.get() + ncon, b.get());

#ifdef MFEM_USE_MPI
   MPI_Allreduce(b_local.get(), b.get(), ncon, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM,
                 mma.comm);
#endif

   for (int i = 0; i < ncon; i++)
   {
      b[i] = b[i] - gx[i];
   }


   for (int i = 0; i < nvar; i++)
   {
      mma.x[i] = 0.5 * (alfa[i] + beta[i]);
      mma.xsi[i] = 1.0/(mma.x[i] - alfa[i]);
      mma.xsi[i] = std::max(mma.xsi[i], static_cast<real_t>(1.0));
      mma.eta[i] = 1.0/(beta[i] - mma.x[i]);
      mma.eta[i] = std::max(mma.eta[i], static_cast<real_t>(1.0));
      ux1[i] = 0.0;
      xl1[i] = 0.0;
   }

   for (int i = 0; i < ncon; i++)
   {
      mma.y[i] = 1.0;
      mma.lam[i] = 1.0;
      mma.mu[i] = std::max(1.0, 0.5 * mma.c[i]);
      mma.s[i] = 1.0;
   }

   while (epsi > mma.epsimin)
   {
      residu[nvar + ncon] = mma.a0 - mma.zet; // rez
      for (int i = 0; i < nvar; i++)
      {
         ux1[i] = mma.upp[i] - mma.x[i];
         if (std::fabs(ux1[i]) < mma.machineEpsilon)
         {
            ux1[i] = mma.machineEpsilon;
         }

         xl1[i] = mma.x[i] - mma.low[i];
         if (std::fabs(xl1[i]) < mma.machineEpsilon)
         {
            xl1[i] = mma.machineEpsilon;
         }

         // plam = P' * lam, qlam = Q' * lam
         plam[i] = p0[i];
         qlam[i] = q0[i];
         for (int j = 0; j < ncon; j++)
         {
            plam[i] += P[j * nvar + i] * mma.lam[j];
            qlam[i] += Q[j * nvar + i] * mma.lam[j];
            residu[nvar + ncon] -= mma.a[j] * mma.lam[j]; // rez
         }
         residu[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) -
                     mma.xsi[i] + mma.eta[i]; // rex
         // residu[nvar + ncon] -= mma.a[i] * mma.lam[i]; // rez
         residu[nvar + ncon + 1 + ncon + i] = mma.xsi[i] * (mma.x[i] - alfa[i]) -
                                              epsi; // rexsi
         if (std::fabs(mma.x[i]-alfa[i]) < mma.machineEpsilon)
         {
            residu[nvar + ncon + 1 + ncon + i] = mma.xsi[i] * mma.machineEpsilon - epsi;
         }
         residu[nvar + ncon + 1 + ncon + nvar + i] = mma.eta[i] *
                                                     (beta[i] - mma.x[i]) - epsi; // reeta
         if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
         {
            residu[nvar + ncon + 1 + ncon + nvar + i] = mma.eta[i] * mma.machineEpsilon -
                                                        epsi;
         }
      }
      for (int i = 0; i < ncon; i++)
      {
         gvec_local[i] = 0.0;
         // gvec = P/ux + Q/xl
         for (int j = 0; j < nvar; j++)
         {
            gvec_local[i] = gvec_local[i] + P[i * nvar + j] / ux1[j] + Q[i * nvar + j] /
                            xl1[j];
         }
      }

      std::copy(gvec_local.get(), gvec_local.get() + ncon, gvec.get());

#ifdef MFEM_USE_MPI
      MPI_Allreduce(gvec_local.get(), gvec.get(), ncon, MPITypeMap<real_t>::mpi_type,
                    MPI_SUM,
                    mma.comm);
#endif

      if ( rank == 0)
      {
         for (int i = 0; i < ncon; i++)
         {
            residu[nvar + i] = mma.c[i] + mma.d[i] * mma.y[i] - mma.mu[i] -
                               mma.lam[i]; // rey
            residu[nvar + ncon + 1 + i] = gvec[i] - mma.a[i] * mma.z - mma.y[i] +
                                          mma.s[i] - b[i]; // relam
            residu[nvar + ncon + 1 + ncon + 2 * nvar + i] = mma.mu[i] * mma.y[i] -
                                                            epsi; // remu
            residu[nvar + ncon + 1 + 2 * nvar + 2 * ncon + 1 + i] = mma.lam[i] * mma.s[i]
                                                                    - epsi; // res
         }
         residu[nvar + ncon + 1 + 2 * nvar + 2 * ncon] = mma.zet * mma.z - epsi;
      }

      // Get vector product and maximum absolute value
      residunorm = 0.0;
      residumax = 0.0;
      for (int i = 0; i < (3 * nvar + 4 * ncon + 2); i++)
      {
         residunorm += residu[i] * residu[i];
         residumax = std::max(residumax, std::abs(residu[i]));
      }

      global_norm = residunorm;
      global_max = residumax;

#ifdef MFEM_USE_MPI
      MPI_Allreduce(&residunorm, &global_norm, 1
                    , MPITypeMap<real_t>::mpi_type, MPI_SUM, mma.comm);
      MPI_Allreduce(&residumax, &global_max, 1
                    , MPITypeMap<real_t>::mpi_type, MPI_MAX, mma.comm);
#endif
      // Norm of the residual
      residunorm = std::sqrt(global_norm);
      residumax = global_max;

      ittt = 0;

      while (residumax > 0.9 * epsi && ittt < 200)
      {
         ittt++;
         for (int i = 0; i < nvar; i++)
         {
            ux1[i] = mma.upp[i] - mma.x[i];
            if (std::fabs(ux1[i]) < mma.machineEpsilon)
            {
               ux1[i] = mma.machineEpsilon;
            }

            xl1[i] = mma.x[i] - mma.low[i];
            if (std::fabs(xl1[i]) <= mma.machineEpsilon)
            {
               xl1[i] = mma.machineEpsilon;
            }
            // plam = P' * lam, qlam = Q' * lam
            plam[i] = p0[i];
            qlam[i] = q0[i];
            for (int j = 0; j < ncon; j++)
            {
               plam[i] += P[j * nvar + i] * mma.lam[j];
               qlam[i] += Q[j * nvar + i] * mma.lam[j];
            }
            // NaN-Avoidance
            if (std::fabs(mma.x[i] - alfa[i]) < mma.machineEpsilon)
            {
               if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
               {
                  delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]);
                  diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                  (xl1[i] * xl1[i] * xl1[i])) + mma.xsi[i] / mma.machineEpsilon + mma.eta[i] /
                             mma.machineEpsilon;
               }
               else
               {
                  delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                            mma.machineEpsilon + epsi / (beta[i] - mma.x[i]);
                  diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                                  (xl1[i] * xl1[i] * xl1[i])) + mma.xsi[i] / (mma.x[i] - alfa[i]) +
                             mma.eta[i] / (beta[i] - mma.x[i]);
               }
            }
            else if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
            {
               delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                         (mma.x[i] - alfa[i]) + epsi / mma.machineEpsilon;
               diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                               (xl1[i] * xl1[i] * xl1[i])) + mma.xsi[i] / (mma.x[i] - alfa[i]) +
                          mma.eta[i] / mma.machineEpsilon;
            }
            else
            {
               delx[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) - epsi /
                         (mma.x[i] - alfa[i]) + epsi / (beta[i] - mma.x[i]);
               diagx[i] = 2 * (plam[i] / (ux1[i] * ux1[i] * ux1[i]) + qlam[i] /
                               (xl1[i] * xl1[i] * xl1[i])) + mma.xsi[i] / (mma.x[i] - alfa[i]) +
                          mma.eta[i] / (beta[i] - mma.x[i]);
            }
         }

         for (int i = 0; i < ncon; i++)
         {
            gvec_local[i] = 0.0;
            // gvec = P/ux + Q/xl
            for (int j = 0; j < nvar; j++)
            {
               gvec_local[i] = gvec_local[i] + P[i * nvar + j] / ux1[j] + Q[i * nvar + j] /
                               xl1[j];
               GG[i * nvar + j] = P[i * nvar + j] / (ux1[j] * ux1[j]) - Q[i * nvar + j] /
                                  (xl1[j] * xl1[j]);
            }
         }

         std::copy(gvec_local.get(), gvec_local.get() + ncon, gvec.get());
#ifdef MFEM_USE_MPI
         MPI_Allreduce(gvec_local.get(), gvec.get(), ncon,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, mma.comm);
#endif

         delz = mma.a0 - epsi / mma.z;
         for (int i = 0; i < ncon; i++)
         {
            dely[i] = mma.c[i] + mma.d[i] * mma.y[i] - mma.lam[i] - epsi / mma.y[i];
            delz -= mma.a[i] * mma.lam[i];
            dellam[i] = gvec[i] - mma.a[i] * mma.z - mma.y[i] - b[i] + epsi /
                        mma.lam[i];
            diagy[i] = mma.d[i] + mma.mu[i] / mma.y[i];
            diaglamyi[i] = mma.s[i] / mma.lam[i] + 1.0 / diagy[i];
         }

         if (ncon < nVar_global)
         {
            // bb1 = dellam + dely./diagy - GG*(delx./diagx);
            // bb1 = [bb1; delz];
            for (int j = 0; j < ncon; j++)
            {
               sum_local[j] = 0.0;
               for (int i = 0; i < nvar; i++)
               {
                  sum_local[j] = sum_local[j] + GG[j * nvar + i] * (delx[i] / diagx[i]);
               }
            }

            std::copy(sum_local.get(), sum_local.get() + ncon, sum_global.get());

#ifdef MFEM_USE_MPI
            MPI_Allreduce(sum_local.get(), sum_global.get(), ncon,
                          MPITypeMap<real_t>::mpi_type, MPI_SUM, mma.comm);
#endif

            for (int j = 0; j < ncon; j++)
            {
               bb1[j] = - sum_global[j] + dellam[j] + dely[j] / diagy[j];
            }
            bb1[ncon] = delz;

            // Alam = spdiags(diaglamyi,0,m,m) + GG*spdiags(diagxinv,0,n,n)*GG';
            for (int i = 0; i < ncon; i++)
            {
               // Axx = GG*spdiags(diagxinv,0,n,n);
               for (int k = 0; k < nvar; k++)
               {
                  Axx[i * nvar + k] = GG[k * ncon + i] / diagx[k];
               }
            }
            // Alam = spdiags(diaglamyi,0,m,m) + Axx*GG';
            for (int i = 0; i < ncon; i++)
            {
               for (int j = 0; j < ncon; j++)
               {
                  Alam_local[i * ncon + j] = 0.0;
                  for (int k = 0; k < nvar; k++)
                  {
                     Alam_local[i * ncon + j] += Axx[i * nvar + k] * GG[j * nvar + k];
                  }
               }
            }

            std::copy(Alam_local.get(), Alam_local.get() + ncon * ncon, Alam.get());
#ifdef MFEM_USE_MPI
            MPI_Reduce(Alam_local.get(), Alam.get(), ncon * ncon,
                       MPITypeMap<real_t>::mpi_type, MPI_SUM, 0, mma.comm);
#endif

            if (0 == rank)
            {
               for (int i = 0; i < ncon; i++)
               {
                  for (int j = 0; j < ncon; j++)
                  {
                     if (i == j)
                     {
                        Alam[i * ncon + j] += diaglamyi[i];
                     }
                  }
               }
               // AA1 = [Alam     a   ]
               //       [ a'    -zet/z]
               for (int i = 0; i < ncon; i++)
               {
                  for (int j = 0; j < ncon; j++)
                  {
                     AA1[i * (ncon + 1) + j] = Alam[i * ncon + j];
                  }
                  AA1[i * (ncon + 1) + ncon] = mma.a[i];
               }
               for (int i = 0; i < ncon; i++)
               {
                  AA1[ncon * (ncon + 1) + i] = mma.a[i];
               }
               AA1[(ncon + 1) * (ncon + 1) - 1] = -mma.zet / mma.z;

#ifdef MFEM_USE_LAPACK
               // bb1 = AA1\bb1 --> solve linear system of equations using LAPACK
               int info;
               int nLAP = ncon + 1;
               int nrhs = 1;
               int lda = nLAP;
               int ldb = nLAP;
               int* ipiv = new int[nLAP];
#if defined(MFEM_USE_DOUBLE)
               dgesv_(&nLAP, &nrhs, AA1.get(), &lda, ipiv, bb1.get(), &ldb, &info);
#elif defined(MFEM_USE_SINGLE)
               sgesv_(&nLAP, &nrhs, AA1.get(), &lda, ipiv, bb1.get(), &ldb, &info);
#else
#error "Only single and double precision are supported!"
#endif
               if (info == 0)
               {
                  delete[] ipiv;
               }
               else if (info > 0)
               {
                  MFEM_ABORT("MMA: matrix is singular.");
               }
               else
               {
                  MFEM_ABORT("MMA: Argument " << info <<
                             " in linear system solve has illegal value.");
               }
#else
               solveLU(ncon, AA1.get(), bb1.get());
#endif
            }
#ifdef MFEM_USE_MPI
            MPI_Bcast(bb1.get(), ncon + 1,
                      MPITypeMap<real_t>::mpi_type, 0, mma.comm);
#endif
            // Reassign results
            for (int i = 0; i < ncon; i++)
            {
               dlam[i] = bb1[i];
            }
            dz = bb1[ncon];

            // dx = -(GG'*dlam)./diagx - delx./diagx;
            for (int i = 0; i < nvar; i++)
            {
               sum = 0.0;
               for (int j = 0; j < ncon; j++)
               {
                  sum = sum + GG[j * nvar + i] * dlam[j];
               }
               dx[i] = -sum / diagx[i] - delx[i] / diagx[i];
            }
         }
         else
         {
            MFEM_ABORT("MMA: Optimization problem case which has more constraints than design variables is not implemented!");
         }

         for (int i = 0; i < ncon; i++)
         {
            dy[i] = -dely[i] / diagy[i] + dlam[i] / diagy[i];
            dmu[i] = -mma.mu[i] + epsi / mma.y[i] - (mma.mu[i] * dy[i]) / mma.y[i];
            ds[i] = -mma.s[i] + epsi / mma.lam[i] - (mma.s[i] * dlam[i]) / mma.lam[i];
            // xx = [y z lam xsi eta mu zet s]
            // dxx = [dy dz dlam dxsi deta dmu dzet ds]
            xx[i] = mma.y[i];
            xx[ncon + 1 + i] = mma.lam[i];
            xx[2 * ncon + 1 + 2 * nvar + i] = mma.mu[i];
            xx[3 * ncon + 2 * nvar + 2 + i] = mma.s[i];

            dxx[i] = dy[i];
            dxx[ncon + 1 + i] = dlam[i];
            dxx[2 * ncon + 1 + 2 * nvar + i] = dmu[i];
            dxx[3 * ncon + 2 * nvar + 2 + i] = ds[i];
         }
         xx[ncon] = mma.z;
         xx[3 * ncon + 2 * nvar + 1] = mma.zet;
         dxx[ncon] = dz;
         for (int i = 0; i < nvar; i++)
         {
            // NaN-Avoidance
            if (std::fabs(mma.x[i] - alfa[i]) < mma.machineEpsilon)
            {
               if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
               {
                  dxsi[i] = -mma.xsi[i] + epsi / mma.machineEpsilon - (mma.xsi[i] * dx[i]) /
                            mma.machineEpsilon;
                  deta[i] = -mma.eta[i] + epsi / mma.machineEpsilon + (mma.eta[i] * dx[i]) /
                            mma.machineEpsilon;
               }
               else
               {
                  dxsi[i] = -mma.xsi[i] + epsi / mma.machineEpsilon - (mma.xsi[i] * dx[i]) /
                            mma.machineEpsilon;
                  deta[i] = -mma.eta[i] + epsi / (beta[i] - mma.x[i]) +
                            (mma.eta[i] * dx[i]) / (beta[i] - mma.x[i]);
               }
            }
            else if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
            {
               dxsi[i] = -mma.xsi[i] + epsi / (mma.x[i] - alfa[i]) -
                         (mma.xsi[i] * dx[i]) / (mma.x[i] - alfa[i]);
               deta[i] = -mma.eta[i] + epsi / mma.machineEpsilon + (mma.eta[i] * dx[i]) /
                         mma.machineEpsilon;
            }
            else
            {
               dxsi[i] = -mma.xsi[i] + epsi / (mma.x[i] - alfa[i]) -
                         (mma.xsi[i] * dx[i]) / (mma.x[i] - alfa[i]);
               deta[i] = -mma.eta[i] + epsi / (beta[i] - mma.x[i]) +
                         (mma.eta[i] * dx[i]) / (beta[i] - mma.x[i]);
            }
            xx[ncon + 1 + ncon + i] = mma.xsi[i];
            xx[ncon + 1 + ncon + nvar + i] = mma.eta[i];
            dxx[ncon + 1 + ncon + i] = dxsi[i];
            dxx[ncon + 1 + ncon + nvar + i] = deta[i];
         }
         dzet = -mma.zet + epsi / mma.z - mma.zet * dz / mma.z;
         dxx[3 * ncon + 2 * nvar + 1] = dzet;

         stmxx = 0.0;
         for (int i = 0; i < (4 * ncon + 2 * nvar + 2); i++)
         {
            stepxx[i] = -1.01*dxx[i] /  xx[i];
            stmxx = std::max(stepxx[i], stmxx);
         }
         stmxx_global = stmxx;
#ifdef MFEM_USE_MPI
         MPI_Allreduce(&stmxx, &stmxx_global, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, mma.comm);
#endif

         stmalfa = 0.0;
         stmbeta = 0.0;
         for (int i = 0; i < nvar; i++)
         {
            // NaN-Avoidance
            if (std::fabs(mma.x[i] - alfa[i]) < mma.machineEpsilon)
            {
               stepalfa[i] = -1.01*dx[i] / mma.machineEpsilon;
            }
            else
            {
               stepalfa[i] = -1.01*dx[i] / (mma.x[i] - alfa[i]);
            }
            if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
            {
               stepbeta[i] = 1.01*dx[i] / mma.machineEpsilon;
            }
            else
            {
               stepbeta[i] = 1.01*dx[i] / (beta[i] - mma.x[i]);
            }
            stmalfa = std::max(stepalfa[i], stmalfa);
            stmbeta = std::max(stepbeta[i], stmbeta);
         }
         stmalfa_global = stmalfa;
         stmbeta_global = stmbeta;
#ifdef MFEM_USE_MPI
         MPI_Allreduce(&stmalfa, &stmalfa_global, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, mma.comm);
         MPI_Allreduce(&stmbeta, &stmbeta_global, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, mma.comm);
#endif
         stminv = std::max(std::max(std::max(stmalfa_global, stmbeta_global),
                                    stmxx_global), static_cast<real_t>(1.0));
         steg = 1.0 / stminv;

         for (int i = 0; i < nvar; i++)
         {
            xold[i] = mma.x[i];
            xsiold[i] = mma.xsi[i];
            etaold[i] = mma.eta[i];
         }
         for (int i = 0; i < ncon; i++)
         {
            yold[i] = mma.y[i];
            lamold[i] = mma.lam[i];
            muold[i] = mma.mu[i];
            sold[i] = mma.s[i];
         }
         zold = mma.z;
         zetold = mma.zet;

         itto = 0;
         resinew = 2.0 * residunorm;
         while (resinew > residunorm && itto < 50)
         {
            itto++;

            for (int i = 0; i < ncon; ++i)
            {
               mma.y[i] = yold[i] + steg * dy[i];
               if (std::fabs(mma.y[i])< mma.machineEpsilon)
               {
                  mma.y[i] = mma.machineEpsilon;
               }

               mma.lam[i] = lamold[i] + steg * dlam[i];
               if (std::fabs(mma.lam[i])< mma.machineEpsilon )
               {
                  mma.lam[i] = mma.machineEpsilon;
               }
               mma.mu[i] = muold[i] + steg * dmu[i];
               mma.s[i] = sold[i] + steg * ds[i];
            }

            residu[nvar + ncon] = mma.a0 - mma.zet; // rez
            for (int i = 0; i < nvar; ++i)
            {
               mma.x[i] = xold[i] + steg * dx[i];
               mma.xsi[i] = xsiold[i] + steg * dxsi[i];
               mma.eta[i] = etaold[i] + steg * deta[i];

               ux1[i] = mma.upp[i] - mma.x[i];
               if (std::fabs(ux1[i]) < mma.machineEpsilon)
               {
                  ux1[i] = mma.machineEpsilon;
               }
               xl1[i] = mma.x[i] - mma.low[i];
               if (std::fabs(xl1[i]) < mma.machineEpsilon )
               {
                  xl1[i] = mma.machineEpsilon;
               }
               // plam & qlam
               plam[i] = p0[i];
               qlam[i] = q0[i];
               for (int j = 0; j < ncon; j++)
               {
                  plam[i] += P[j * nvar + i] * mma.lam[j];
                  qlam[i] += Q[j * nvar + i] * mma.lam[j];
                  residu[nvar + ncon] -= mma.a[j] * mma.lam[j]; // rez
               }

               // Assembly starts here

               residu[i] = plam[i] / (ux1[i] * ux1[i]) - qlam[i] / (xl1[i] * xl1[i]) -
                           mma.xsi[i] + mma.eta[i]; // rex
               // residu[nvar + ncon] -= mma.a[i] * mma.lam[i]; // rez
               residu[nvar + ncon + 1 + ncon + i] = mma.xsi[i] * (mma.x[i] - alfa[i]) -
                                                    epsi; // rexsi
               if (std::fabs(mma.x[i] - alfa[i]) < mma.machineEpsilon)
               {
                  residu[nvar + ncon + 1 + ncon + i] = mma.xsi[i] * mma.machineEpsilon - epsi;
               }
               residu[nvar + ncon + 1 + ncon + nvar + i] = mma.eta[i] *
                                                           (beta[i] - mma.x[i]) - epsi; // reeta
               if (std::fabs(beta[i] - mma.x[i]) < mma.machineEpsilon)
               {
                  residu[nvar + ncon + 1 + ncon + nvar + i] = mma.eta[i] * mma.machineEpsilon -
                                                              epsi;
               }
            }
            mma.z = zold + steg * dz;
            if (std::fabs(mma.z) < mma.machineEpsilon)
            {
               mma.z = mma.machineEpsilon;
            }
            mma.zet = zetold + steg * dzet;

            // gvec = P/ux + Q/xl
            for (int i = 0; i < ncon; i++)
            {
               gvec_local[i] = 0.0;
               for (int j = 0; j < nvar; j++)
               {
                  gvec_local[i] = gvec_local[i] + P[i * nvar + j] / ux1[j] + Q[i * nvar + j] /
                                  xl1[j];
               }
            }
            std::copy(gvec_local.get(), gvec_local.get() + ncon, gvec.get());

#ifdef MFEM_USE_MPI
            MPI_Allreduce(gvec_local.get(), gvec.get(), ncon,
                          MPITypeMap<real_t>::mpi_type, MPI_SUM, mma.comm);
#endif
            if (rank == 0)
            {
               for (int i = 0; i < ncon; i++)
               {
                  residu[nvar + i] = mma.c[i] + mma.d[i] * mma.y[i]
                                     - mma.mu[i] - mma.lam[i]; // rey
                  residu[nvar + ncon + 1 + i] = gvec[i] - mma.a[i] * mma.z
                                                - mma.y[i] + mma.s[i] - b[i];
                  // relam
                  residu[nvar + ncon + 1 + ncon + 2 * nvar + i] = mma.mu[i]
                                                                  * mma.y[i] -
                                                                  epsi; // remu
                  residu[nvar + ncon + 1 + 2 * nvar + 2 * ncon + 1 + i] =
                     mma.lam[i] * mma.s[i] - epsi; // res
               }
               residu[nvar + ncon + 1 + 2 * nvar + 2 * ncon] =
                  mma.zet * mma.z - epsi; // rezet
            }

            // Get vector product and maximum absolute value
            resinew = 0.0;
            for (int i = 0; i < (3 * nvar + 4 * ncon + 2); i++)
            {
               resinew = resinew + residu[i] * residu[i];
            }

            global_norm = resinew;
#ifdef MFEM_USE_MPI
            MPI_Allreduce(&resinew, &global_norm, 1,
                          MPITypeMap<real_t>::mpi_type, MPI_SUM, mma.comm);
#endif

            // Norm of the residual
            resinew = std::sqrt(global_norm);

            steg = steg / 2.0;
         }

         residunorm = resinew;
         residumax = 0.0;
         for (int i = 0; i < (3 * nvar + 4 * ncon + 2); i++)
         {
            residumax = std::max(residumax, std::abs(residu[i]));
         }
         global_max = residumax;
#ifdef MFEM_USE_MPI
         MPI_Allreduce(&residumax, &global_max, 1,
                       MPITypeMap<real_t>::mpi_type, MPI_MAX, mma.comm);
#endif
         residumax = global_max;
         steg = steg * 2.0;

      }
      if (ittt > 198 && mma.print_level>=2)
      {
         out << "Warning: Max number of iterations reached in MMA subsolve.\n";
      }
      epsi = 0.1 * epsi;
   }

   // returns x, y, z, lam, xsi, eta, mu, zet, s
}

void MMA::InitData(real_t *xval)
{
   for (int i = 0; i < nVar; i++)
   {
      x[i]=xval[i];
      xo1[i] = 0.0;
      xo2[i] = 0.0;
   }


   for (int i = 0; i < nCon; i++)
   {
      a[i] = 0.0;
      c[i] = 1000.0;
      d[i] = 1.0;
   }
   a0 = 1.0;
}

/// Serial MMA
MMA::MMA(int nVar, int nCon, real_t *xval, int iter)
{
#ifdef MFEM_USE_MPI
   comm=MPI_COMM_SELF;
#endif

   AllocData(nVar,nCon);
   InitData(xval);
   // allocate the serial subproblem
#if  __cplusplus >= 201402L
   mSubProblem = ::std::make_unique<MMA::MMASubSvanberg>(*this, nVar, nCon);
#else
   mSubProblem.reset(new MMA::MMASubSvanberg(*this, nVar, nCon));
#endif
}

MMA::MMA(const int nVar, int nCon, Vector &xval, int iter) : MMA(nVar, nCon,
                                                                    xval.GetData(), iter)
{}

#ifdef MFEM_USE_MPI
MMA::MMA(MPI_Comm comm_, int nVar, int nCon, real_t *xval, int iter)
{
   int rank = 0;
   MPI_Comm_rank(comm_, &rank);

   // create new communicator
   int colour;

   if ( 0 != nVar)
   {
      colour = 0;
   }
   else
   {
      colour = MPI_UNDEFINED;
   }

   // Split the global communicator
   MPI_Comm_split(comm_, colour, rank, &comm);

   AllocData(nVar,nCon);
   InitData(xval);
   // allocate the serial subproblem
   mSubProblem.reset(new MMA::MMASubSvanberg(*this, nVar, nCon));
}

MMA::MMA(MPI_Comm comm_, const int nVar, const int nCon,
         const Vector & xval, int iter) : MMA(comm_, nVar, nCon, xval.GetData(), iter)
{}
#endif


MMA::~MMA()
{
}

void MMA::AllocData(int nVariables,int nConstr)
{
   // accessed by the subproblems
   nVar = nVariables;
   nCon = nConstr;

   x= allocArray(nVar); // ini
   xo1 = allocArray(nVar); // ini
   xo2 = allocArray(nVar); // ini

   y = allocArray(nCon); // ini
   c = allocArray(nCon); // ini
   d = allocArray(nCon); // ini
   a = allocArray(nCon); // ini

   lam = allocArray(nCon); // ini

   xsi = allocArray(nVar); // ini
   eta = allocArray(nVar); // ini

   mu = allocArray(nCon); // ini
   s = allocArray(nCon); // ini

   z = zet = 1.0;
   kktnorm = 10;
   machineEpsilon = 1e-10;


   // accessed by MMA
   epsimin = 1e-7;
   asyinit = 0.5;
   asyincr = 1.1;
   asydecr = 0.7;
   low = allocArray(nVar); // ini
   upp = allocArray(nVar); // ini
   factor = allocArray(nVar); // ini
   lowmin = lowmax = uppmin = uppmax = zz = 0.0;

}

void MMA::Update( const Vector& dfdx,
                  const Vector& gx, const Vector& dgdx,
                  const Vector& xmin, const Vector& xmax,
                  Vector& xval)
{
   this->Update(dfdx.GetData(),
                gx.GetData(),dgdx.GetData(),
                xmin.GetData(), xmax.GetData(),
                xval.GetData());
}

void MMA::Update( const Vector& dfdx,
                  const Vector& xmin, const Vector& xmax,
                  Vector& xval)
{
   MFEM_ASSERT(0 == nCon,
               "MMA nCon != 0. Provide constraint values and gradients");

   this->Update(dfdx.GetData(),
                nullptr,nullptr,
                xmin.GetData(), xmax.GetData(),
                xval.GetData());
}

void MMA::Update(const real_t* dfdx,
                 const real_t* gx,const real_t* dgdx,
                 const real_t* xmin, const real_t* xmax,
                 real_t* xval)
{
   // Calculation of the asymptotes low and upp
   if (iter < 3)
   {
      for (int i = 0; i < nVar; i++)
      {
         low[i] = xval[i] - asyinit * (xmax[i] - xmin[i]);
         upp[i] = xval[i] + asyinit * (xmax[i] - xmin[i]);
      }
   }
   else
   {
      for (int i = 0; i < nVar; i++)
      {
         // Determine sign
         zz = (xval[i] - xo1[i]) * (xo1[i] - xo2[i]);
         if ( zz > 0.0)
         {
            factor[i] =  asyincr;
         }
         else if ( zz < 0.0)
         {
            factor[i] =  asydecr;
         }
         else
         {
            factor[i] =  1.0;
         }


         // Find new asymptote
         low[i] = xval[i] - factor[i] * (xo1[i] - low[i]);
         upp[i] = xval[i] + factor[i] * (upp[i] - xo1[i]);

         lowmin = xval[i] - 10.0 * (xmax[i] - xmin[i]);
         lowmax = xval[i] - 0.01 * (xmax[i] - xmin[i]);
         uppmin = xval[i] + 0.01 * (xmax[i] - xmin[i]);
         uppmax = xval[i] + 10.0 * (xmax[i] - xmin[i]);

         low[i] = std::max(low[i], lowmin);
         low[i] = std::min(low[i], lowmax);
         upp[i] = std::max(upp[i], uppmin);
         upp[i] = std::min(upp[i], uppmax);
      }
   }

   mSubProblem->Update(dfdx,gx,dgdx,xmin,xmax,xval);
   // Update design variables
   for (int i = 0; i < nVar; i++)
   {
      xo2[i] = xo1[i];
      xo1[i] = xval[i];
      xval[i] = x[i];
   }

   iter++;
}

}
