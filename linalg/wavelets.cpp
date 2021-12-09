// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "wavelets.hpp"
#include "../general/forall.hpp"

#define MFEM_NVTX_COLOR Orange
#include "../general/nvtx.hpp"

using namespace mfem;

namespace mfem
{

////////////////////////////////////////////////////////////////////////////////
MFEM_HOST_DEVICE inline int RotateLeft(int i, int M) { return (i+1) % M; }
MFEM_HOST_DEVICE inline int RotateLeft2(int i, int M) { return (i+2) % M; }
MFEM_HOST_DEVICE inline int RotateRight(int i, int M) { return (i+M-1) % M; }
MFEM_HOST_DEVICE inline int RotateRight2(int i, int M) { return (i+M-2) % M; }

////////////////////////////////////////////////////////////////////////////////
Operator *Wavelet::New(const Wavelet::Type &wavelet, int n, bool lowpass)
{
   switch (wavelet)
   {
      case Wavelet::HAAR: return new HaarWavelet(n,lowpass);
      case Wavelet::DAUBECHIES: return new DaubechiesWavelet(n,lowpass);
      // Lowpass to add to the folowing wavelets
      case Wavelet::CDF53: return new CDF53Wavelet(n);
      case Wavelet::CDF97: return new CDF97Wavelet(n);
      case Wavelet::LEGENDRE2: return new Legendre2Wavelet(n);
      case Wavelet::LEGENDRE3: return new Legendre3Wavelet(n);
      default: MFEM_ABORT("No Wavelet Operator found!");
   }
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
std::string Wavelet::GetType(const Wavelet::Type &wavelet)
{
   switch (wavelet)
   {
      case Wavelet::HAAR: return std::string("HAAR");
      case Wavelet::DAUBECHIES: return std::string("DAUBECHIES");
      case Wavelet::CDF53: return std::string("CDF53");
      case Wavelet::CDF97: return std::string("CDF97");
      case Wavelet::LEGENDRE2: return std::string("LEGENDRE2");
      case Wavelet::LEGENDRE3: return std::string("LEGENDRE3");
      default: MFEM_ABORT("Not a valid wavelet type!");
   }
   return std::string("???");
}

////////////////////////////////////////////////////////////////////////////////
SparseMatrix *Wavelet::GetMatrix()
{
   MFEM_ABORT("Not implemented");
   return nullptr;
}

SparseMatrix *Wavelet::GetTransposedMatrix()
{
   MFEM_ABORT("Not implemented");
   return nullptr;
}

double Wavelet::GetOddRowValue(const int, const int, const bool)
{
   MFEM_ABORT("Not implemented");
   return 0.0;
}

////////////////////////////////////////////////////////////////////////////////
SparseMatrix *Wavelet::GetEvenMatrix(const double *coeffs, const int rowsize)
{
   MFEM_NVTX;
   const int ncols = width;
   const int nrows = height;
   const int nnz = nrows * rowsize;

   SparseMatrix *A = new SparseMatrix(nrows, ncols, rowsize);

   int *J = A->GetJ();
   int *I = A->GetI();
   int row_offsets = 0;
   double *data = A->GetData();

   assert(m*2 == width);
   assert(((width-rowsize)%2&1) == 0);
   const int shifts = 1 + (width-rowsize)/2; // 0 + possible shifts
   const int warped = m - shifts;

   // Lowpass
   for (int i=0, j=0; i<shifts; i++, j+=2)
   {
      I[i] = row_offsets;
      for (int nz=0; nz<rowsize; nz++)
      {
         J[I[i]+nz] = j + nz;
         *data++ = coeffs[nz];
      }
      row_offsets += rowsize;
   }

   // Warped lines
   for (int i=shifts, nz=0; i<shifts+warped; i++)
   {
      // Lowpass-warped
      for (int j=0; nz<rowsize-2; nz++,j++)
      {
         J[I[i]+nz] = j;
         *data++ = coeffs[nz+2];
      }
      for (int j=width-2; nz<rowsize; nz++, j++)
      {
         J[I[i]+nz] = j;
         *data++ = coeffs[nz-2];
      }
      row_offsets += rowsize;
   }

   if (!lowpass)
   {
      for (int i=shifts+warped, nz=0; i<shifts+2*warped; i++)
      {
         // Highpass-warped
         I[i] = row_offsets;
         double sign = -1.0;
         for (int j=0; nz<rowsize-2; nz++,j++)
         {
            J[I[i]+nz] = j;
            *data++ = sign*coeffs[rowsize-1-nz-2];
            sign *= -1.0;
         }
         for (int j=width-2; nz<rowsize; nz++,j++)
         {
            J[I[i]+nz] = j;
            *data++ = sign*coeffs[rowsize-1-nz+2];
            sign *= -1.0;
         }
         row_offsets += rowsize;
      }

      // Highpass
      for (int i=shifts+2*warped, j=0; i<height; i++, j+=2)
      {
         I[i] = row_offsets;
         double sign = -1.0;
         for (int nz=0; nz<rowsize; nz++)
         {
            J[I[i]+nz] = j+nz;
            *data++ = sign*coeffs[rowsize-1-nz];
            sign *= -1.0;
         }
         row_offsets += rowsize;
      }
   }

   I[height] = row_offsets;
   MFEM_VERIFY(I[0] == 0, "Error");
   MFEM_VERIFY(I[height] == nnz, "Error");
   A->Finalize();
   return A;
}

////////////////////////////////////////////////////////////////////////////////
SparseMatrix *Wavelet::GetOddMatrix(const double *coeffs, const int rowsize,
                                    const bool for_transpose)
{
   MFEM_NVTX;
   const int nrows = height;
   const int ncols = width;
   // sparse matrix with flexible sparsity structure
   SparseMatrix *A = new SparseMatrix(nrows, ncols);

   Vector x(width), y(height);

   const int shifts = 1 + (width-rowsize)/2;
   const int warped = m - shifts - 1;

   // Lowpass
   for (int i=0, j=0; i<shifts; i++, j+=2)
   {
      for (int nz=0; nz<rowsize; nz++)
      {
         A->Set(i, j+nz, coeffs[nz]);
      }
   }

   // Warped lines
   if (!for_transpose)
   {
      // lowpass, warped
      for (int i=shifts; i<shifts+warped; i++)
      {
         // DAUBECHIES only
         assert(type != HAAR);
         assert(rowsize == 4);
         for (int nz=0; nz<rowsize-1; nz++)
         {
            const int j[3] = {width-3, width-2, width-1};
            const double val = GetOddRowValue(i, j[nz], false);
            A->Set(i, j[nz], val);
         }
      }

      // middle line
      {
         const int i = m - 1;
         for (int nz=0; nz<rowsize-1; nz++)
         {
            const int j = type == HAAR ? width-1 :
                          // DAUBECHIES
                          nz==0 ? 0 :
                          nz==1 ? 1 :
                          width-1;
            const double val = GetOddRowValue(i, j, false);
            A->Set(i, j, val);
         }
      }

      if (!lowpass)
      {
         // highpass, warped
         for (int i=shifts+warped+1; i<shifts+2*warped+1; i++)
         {
            assert(type != HAAR);
            assert(rowsize == 4);
            for (int nz = 0; nz < rowsize-1; nz++)
            {
               const int j[3] = {0, 1, width-1};
               const double val = GetOddRowValue(i, j[nz], false);
               A->Set(i, j[nz], val);
            }
         }
      }
   }
   else // transpose
   {
      if (lowpass)
      {
         const bool root = type == HAAR ? true : false;

         for (int i=shifts; i<shifts+warped; i++)
         {
            // DAUBECHIES only
            assert(type != HAAR);
            assert(rowsize == 4);
            for (int nz=0; nz<rowsize-1; nz++)
            {
               const int j[3] = {width-3, width-2, width-1};
               const double val = GetOddRowValue(i, j[nz], root);
               A->Set(i, j[nz], val);
            }
         }

         // middle line
         {
            const int i = m-1;
            for (int nz=0; nz<rowsize-1; nz++)
            {
               const int j = type == HAAR ? width-1 :
                             // DAUBECHIES
                             nz==0 ? 0 :
                             nz==1 ? 1 :
                             nz==2 ? width-1 :
                             -1;
               const double val = GetOddRowValue(i, j, root);
               A->Set(i, j, val);
            }
         }
      }
      else // !lowpass
      {
         for (int i=shifts; i<shifts+warped; i++)
         {
            // DAUBECHIES only
            assert(type != HAAR);
            assert(rowsize == 4);
            for (int nz = 0; nz < rowsize-2; nz++)
            {
               const int j[2] = {width-3, width-2};
               const double val = GetOddRowValue(i, j[nz], true);
               A->Set(i,j[nz], val);
            }
         }

         // middle line
         {
            const int i = m-1;
            for (int nz=0; nz<rowsize+1; nz++)
            {
               const int j = type == HAAR ? width-1 :
                             // DAUBECHIES
                             nz==0 ? 0 :
                             nz==1 ? 1 :
                             nz==2 ? width-3 :
                             nz==3 ? width-2 :
                             width-1;
               const double val = GetOddRowValue(i, j, true);
               A->Set(i,j, val);
            }
         }

         for (int i=shifts+warped+1; i<shifts+2*warped+1; i++)
         {
            // DAUBECHIES only
            assert(type != HAAR);
            assert(rowsize == 4);
            for (int nz=0; nz<rowsize+1; nz++)
            {
               const int j[5] = {0, 1, width-3, width-2, width-1};
               const double val = GetOddRowValue(i, j[nz], true);
               A->Set(i, j[nz], val);
            }
         }
      }
   }

   // Highpass
   if (!lowpass)
   {
      for (int i=shifts+2*warped+1, j=0; i<height; i++, j+=2)
      {
         double sign = -1.0;
         for (int nz = 0; nz < rowsize; nz++)
         {
            A->Set(i, j+nz, sign*coeffs[rowsize-1-nz]);
            sign *= -1.0;
         }
      }
   }

   A->Finalize();
   return A;
}

////////////////////////////////////////////////////////////////////////////////
SparseMatrix *Wavelet::GetMatrix(const double *coeffs, const int rowsize)
{
   MFEM_NVTX;
   return odd ?
          GetOddMatrix(coeffs, rowsize, false) :
          GetEvenMatrix(coeffs, rowsize);
}

SparseMatrix *Wavelet::GetTransposedMatrix(const double *coeffs,
                                           const int rowsize)
{
   MFEM_NVTX;
   return odd ?
          Transpose(*GetOddMatrix(coeffs, rowsize, true)) :
          Transpose(*GetEvenMatrix(coeffs, rowsize));
}

////////////////////////////////////////////////////////////////////////////////
HaarWavelet::HaarWavelet(int n, bool lowpass): Wavelet(HAAR, n,lowpass) { }

void HaarWavelet::Mult(const Vector &x, Vector &y) const
{
   MFEM_NVTX;
   const int M = m;
   const int O = odd;
   const bool LOWPASS = lowpass;

   const double isq2 = 1.0 / sqrt(2.0);

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const double Si = X[2*i];
      const double Di = (O && i==M-1) ? 0.0 : X[2*i+1];
      const double ds = (Di - Si) * isq2;
      Y[i] = 2.0 * Si * isq2 + ds;
      if (LOWPASS) { return; }
      if (O && i==M-1) { return; }
      Y[M+i] = ds;
   });
}

double HaarWavelet::GetOddRowValue(const int I, const int J,
                                   const bool root)
{
   MFEM_NVTX;
   assert(odd);
   assert(I == m-1);
   assert(I == J/2);
   assert(J == width-1);
   const double sq2 = sqrt(2.0);
   const double isq2 = 1.0 / sq2;
   const double Di = root ? -1.0 : 0.0;
   const double Si = (1.0 - Di) * isq2;
   return Si; // root ? sq2 : isq2;
}

void HaarWavelet::MultTranspose(const Vector &x, Vector &y) const
{
   MFEM_NVTX;
   const int M = m;
   const int O = odd;
   const bool LOWPASS = lowpass;

   const double sq2 = sqrt(2.0);
   const double isq2 = 1.0 / sq2;

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const double Di = (O && i==M-1) ? -X[M-1] : LOWPASS ? 0.0 : X[M+i];
      const double Si = (X[i] - Di) * isq2;
      Y[2*i] = Si;
      if (O && i==M-1) { return; }
      Y[2*i+1] = Di * sq2 + Si;
   });
}

SparseMatrix *HaarWavelet::GetMatrix()
{
   MFEM_NVTX;
   constexpr int rowsize = 2;
   const double isq2 = 1.0 / sqrt(2.0);
   const double coeffs[rowsize] = { isq2, isq2 };
   return Wavelet::GetMatrix(coeffs, rowsize);
}

SparseMatrix *HaarWavelet::GetTransposedMatrix()
{
   MFEM_NVTX;
   constexpr int rowsize = 2;
   const double isq2 = 1.0 / sqrt(2.0);
   const double coeffs[rowsize] = { isq2, isq2 };
   return Wavelet::GetTransposedMatrix(coeffs, rowsize);
}

////////////////////////////////////////////////////////////////////////////////
DaubechiesWavelet::DaubechiesWavelet(int n, bool lowpass):
   Wavelet(DAUBECHIES, n, lowpass) {}

void DaubechiesWavelet::Mult(const Vector &x, Vector &y) const
{
   const int M = m;
   const int O = odd;
   const bool LOWPASS = lowpass;

   const double sq2 = sqrt(2.0);
   const double sq3 = sqrt(3.0);
   const double sq34mh = sq3 / 4.0 - 0.5;
   const double sq2i3m1 = sq2 / (sq3 - 1.0);

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int r = RotateRight(i,M);
      double Si = X[2*i];
      double Sl = X[2*l];
      double Sr = X[2*r];
      double Di = (O && i==M-1) ? 0.0 : X[2*i+1];
      double Dl = (O && l==M-1) ? 0.0 : X[2*l+1];
      double Dr = (O && r==M-1) ? 0.0 : X[2*r+1];
      Di -= sq3 * Si;
      Dl -= sq3 * Sl;
      Dr -= sq3 * Sr;
      Si += sq3 * Di / 4.0 + sq34mh * Dl;
      Sr += sq3 * Dr / 4.0 + sq34mh * Di;
      Di += Sr;
      Si *= sq2i3m1;
      Di /= sq2i3m1;
      Y[i] = Si;
      if (LOWPASS) { return; }
      if (O && i==M-1) { return; }
      Y[M+i] = Di;
   });
}

double DaubechiesWavelet::GetOddRowValue(const int I,
                                         const int J,
                                         const bool root)
{
   assert(odd);

   const auto X = [&](int i) { return I==i ? 1.0 : 0.0; };

   const double sq3 = sqrt(3.0);
   const double msq3q = -sq3 / 4.0;
   const double hmsq34 = 0.5 + msq3q;
   const double sq2i3m1 = sqrt(2.0) / (sq3 - 1.0);

   const int i = J/2;
   const int l = RotateLeft(i,m);
   const int r = RotateRight(i,m);
   double Si = X(i);
   double Sr = X(r);
   const bool oi = i==m-1, ol = l==m-1;
   const bool ox = oi || ol;
   const double Xi =
      ox && root ? (3-2*sq3)*X(m)-sq3*X(m-1)+(2-sq3)*X(2*(m-1)%m) : 0.0;
   double Di = oi ? Xi : X(m+i);
   double Dl = ol ? Xi : X(m+l);
   Si /= sq2i3m1;
   Sr /= sq2i3m1;
   Di *= sq2i3m1;
   Dl *= sq2i3m1;
   Di -= Sr;
   Dl -= Si;
   Si += msq3q * Di + hmsq34 * Dl;
   Di += sq3 * Si;
   return J%2 ? Di : Si;
}

void DaubechiesWavelet::MultTranspose(const Vector &x, Vector &y) const
{
   const int M = m;
   const int O = odd;
   const bool LOWPASS = lowpass;

   const double sq3 = sqrt(3.0);
   const double msq3q = -sq3 / 4.0;
   const double hmsq34 = 0.5 + msq3q;
   const double sq2i3m1 = sqrt(2.0) / (sq3 - 1.0);

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int r = RotateRight(i,M);
      double Si = X[i];
      double Sr = X[r];
      const bool oi = O && i==M-1;
      const bool ol = O && l==M-1;
      const bool ox = oi || ol;
      const double Xi =
      ox ? LOWPASS ? 0.0 : (3-2*sq3)*X[M]-sq3*X[M-1]+(2-sq3)*X[2*(M-1)%M] : 0.0;
      double Di = oi ? Xi : LOWPASS ? 0.0 : X[M+i];
      double Dl = ol ? Xi : LOWPASS ? 0.0 : X[M+l];
      Si /= sq2i3m1;
      Sr /= sq2i3m1;
      Di *= sq2i3m1;
      Dl *= sq2i3m1;
      Di -= Sr;
      Dl -= Si;
      Si += msq3q * Di + hmsq34 * Dl;
      Di += sq3 * Si;
      Y[2*i] = Si;
      if (O && i==M-1) { return; }
      Y[2*i+1] = Di;
   });
}

SparseMatrix *DaubechiesWavelet::GetMatrix()
{
   constexpr int rowsize = 4;
   const double sqrt2 = sqrt(2);
   const double coeffs[rowsize] = { sqrt2*(1+sqrt(3))/8.,
                                    sqrt2*(3+sqrt(3))/8.,
                                    sqrt2*(3-sqrt(3))/8.,
                                    sqrt2*(1-sqrt(3))/8.
                                  };
   return Wavelet::GetMatrix(coeffs, rowsize);
}

SparseMatrix *DaubechiesWavelet::GetTransposedMatrix()
{
   constexpr int rowsize = 4;
   const double sqrt2 = sqrt(2);
   const double coeffs[rowsize] = { sqrt2*(1+sqrt(3))/8.,
                                    sqrt2*(3+sqrt(3))/8.,
                                    sqrt2*(3-sqrt(3))/8.,
                                    sqrt2*(1-sqrt(3))/8.
                                  };
   return Wavelet::GetTransposedMatrix(coeffs, rowsize);
}

////////////////////////////////////////////////////////////////////////////////
CDF53Wavelet::CDF53Wavelet(int n): Wavelet(CDF53, n) { }

void CDF53Wavelet::Mult(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const double isq2 = 1.0 / sqrt(2.0);

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int r = RotateRight(i,M);
      double Si = X[2*i];
      double Sl = X[2*l];
      double Di = (odd && i==M-1) ? 0.0 : X[2*i+1];
      double Dl = (odd && l==M-1) ? 0.0 : X[2*l+1];
      double Dr = (odd && r==M-1) ? 0.0 : X[2*r+1];
      Si += Di/2.0 + Dr/2.0;
      Sl += Dl/2.0 + Di/2.0;
      Di -= Si/4.0 + Sl/4.0;
      Si *= isq2;
      Di /= isq2;
      Y[i] = Si;
      if (odd && i==M-1) { return; }
      Y[M+i] = Di;
   });
}

void CDF53Wavelet::MultTranspose(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const double isq2 = 1.0 / sqrt(2.0);

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int r = RotateRight(i,M);
      double Si = X[i];
      double Sl = X[l];
      double Sr = X[r];
      double Di = (odd && i==M-1) ? -(X[0] + X[M-1]) / 2.0: X[M+i];
      double Dr = (odd && r==M-1) ? -(X[0] + X[M-1]) / 2.0: X[M+r];
      Si /= isq2;
      Sl /= isq2;
      Sr /= isq2;
      Di *= isq2;
      Dr *= isq2;
      Di += Si / 4.0 + Sl / 4.0;
      Dr += Sr / 4.0 + Si / 4.0;
      Si -= Di / 2.0 + Dr / 2.0;
      Y[2*i] = Si;
      if (odd && i==M-1) { return; }
      Y[2*i+1] = Di;
   });
}

////////////////////////////////////////////////////////////////////////////////
CDF97Wavelet::CDF97Wavelet(int n): Wavelet(CDF97, n) { }

void CDF97Wavelet::Mult(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int r = RotateRight(i,M);
      const int k = RotateLeft2(i,M);
      const int s = RotateRight2(i,M);
      double Si = X[2*i];
      double Sk = X[2*k];
      double Sl = X[2*l];
      double Sr = X[2*r];
      double Ss = X[2*s];
      double Di = (odd && i==M-1) ? 0.0 : X[2*i+1];
      double Dl = (odd && l==M-1) ? 0.0 : X[2*l+1];
      double Dr = (odd && r==M-1) ? 0.0 : X[2*r+1];
      double Ds = (odd && s==M-1) ? 0.0 : X[2*s+1];
      Di -= alpha * (Si + Sl);
      Dl -= alpha * (Sl + Sk);
      Dr -= alpha * (Sr + Si);
      Ds -= alpha * (Ss + Sr);
      Si -=  beta * (Di + Dr);
      Sl -=  beta * (Dl + Di);
      Sr -=  beta * (Dr + Ds);
      Di += gamma * (Si + Sl);
      Dr += gamma * (Sr + Si);
      Si += delta * (Di + Dr);
      Si *= kappa;
      Di /= kappa;
      Y[i] = Si;
      if (odd && i==M-1) { return; }
      Y[M+i] = Di;
   });
}

void CDF97Wavelet::MultTranspose(const Vector &x, Vector &y) const
{
   const int n = height;
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   constexpr double a = 0.490316548523234605288;
   constexpr double b = 0.075687794783398174545;
   constexpr double c = 0.129734462057497172811;
   constexpr double d = 0.044363215797333603554;

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int r = RotateRight(i,M);
      const int k = RotateLeft2(i,M);
      const int s = RotateRight2(i,M);
      const bool io = odd && i==M-1;
      const bool lo = odd && l==M-1;
      const bool ro = odd && r==M-1;
      const bool so = odd && s==M-1;
      const bool ko = odd && k==M-1;
      const bool xo = io || lo || ro || so || ko;
      const double Xi = xo ?
      - a * (X[0] + X[M-1]) + b * (X[1] + X[2*(M-1)%M])
      + c * (X[M] + X[n-1]) - d * (X[M+1] + X[n-2]) : 0.0;
      double Si = X[i];
      double Sl = X[l];
      double Sk = X[k];
      double Sr = X[r];
      double Di = io ? Xi : X[M+i];
      double Dl = lo ? Xi : X[M+l];
      double Dk = ko ? Xi : X[M+k];
      double Dr = ro ? Xi : X[M+r];
      double Ds = so ? Xi : X[M+s];
      Si /= kappa;
      Sl /= kappa;
      Sk /= kappa;
      Sr /= kappa;
      Di *= kappa;
      Dk *= kappa;
      Dl *= kappa;
      Dr *= kappa;
      Ds *= kappa;
      Si -= delta * (Di + Dr);
      Sr -= delta * (Dr + Ds);
      Sl -= delta * (Dl + Di);
      Sk -= delta * (Dk + Dl);
      Di -= gamma * (Si + Sl);
      Dr -= gamma * (Sr + Si);
      Dl -= gamma * (Sl + Sk);
      Si +=  beta * (Di + Dr);
      Sl +=  beta * (Dl + Di);
      Di += alpha * (Si + Sl);
      Y[2*i] = Si;
      if (odd && i==M-1) { return; }
      Y[2*i+1] = Di;
   });
}

////////////////////////////////////////////////////////////////////////////////
Legendre2Wavelet::Legendre2Wavelet(int n): Wavelet(LEGENDRE2, n) { }

void Legendre2Wavelet::Mult(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const double k2 = sqrt(2.0) / 5.0;

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      double Si = X[2*i];
      double Sl = X[2*l];
      double Di = (odd && i==M-1) ? 0.0 : X[2*i+1];
      double Dl = (odd && l==M-1) ? 0.0 : X[2*l+1];
      Di += Si * 3.0 / 5.0;
      Dl += Sl * 3.0 / 5.0;
      Si += (15.0 * Di + 25.0 * Dl) / 16.0;
      Si *= k2;
      Di /= k2;
      Y[i] = Si;
      if (odd && i==M-1) { return; }
      Y[M+i] = Di;
   });
}

void Legendre2Wavelet::MultTranspose(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const double k2 = sqrt(2.0) / 5.0;

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      double Si = X[i];
      const bool oi = odd && i==M-1;
      const bool ol = odd && l==M-1;
      const bool ox = oi || ol;
      const double Xi = ox ? (3.0 * (8.0 * X[M-1] - X[M])) / 5.0 : 0.0;
      double Di = oi ? Xi : +X[M+i];
      double Dl = ol ? Xi : +X[M+l];
      Si /= k2;
      Di *= k2;
      Dl *= k2;
      Si -= (15.0 * Di + 25.0 * Dl) / 16.0;
      Di -= 3.0 * Si / 5.0;
      Y[2*i] = Si;
      if (odd && i==M-1) { return; }
      Y[2*i+1] = Di;
   });
}

////////////////////////////////////////////////////////////////////////////////
Legendre3Wavelet::Legendre3Wavelet(int n): Wavelet(LEGENDRE3, n) { }

void Legendre3Wavelet::Mult(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const double k3 = 153.0 / (343.0 * sqrt(2.0));
   const double i12 = 1215.0 / 2744.0;
   const double i59 = 5.0 / 9.0;

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int r = RotateRight(i,M);
      const int l = RotateLeft(i,M);
      const int k = RotateLeft2(i,M);
      double Si = X[2*i];
      double Sr = X[2*r];
      double Sl = X[2*l];
      double Sk = X[2*k];
      double Di = (odd && i==M-1) ? 0.0 : X[2*i+1];
      double Dr = (odd && r==M-1) ? 0.0 : X[2*r+1];
      double Dl = (odd && l==M-1) ? 0.0 : X[2*l+1];
      double Dk = (odd && k==M-1) ? 0.0 : X[2*k+1];
      Si += Di * i59;
      Sr += Dr * i59;
      Sl += Dl * i59;
      Sk += Dk * i59;
      Di += 45.0 * Si / 56.0 + i12 * Sr;
      Dl += 45.0 * Sl / 56.0 + i12 * Si;
      Dk += 45.0 * Sk / 56.0 + i12 * Sl;
      Si += 1715.0 * Dl / 7344.0 + 16807.0 * Dk / 22032.0;
      Si *= k3;
      Di /= k3;
      Y[i] = Si;
      if (odd && i==M-1) { return; }
      Y[M+i] = Di;
   });
}

void Legendre3Wavelet::MultTranspose(const Vector &x, Vector &y) const
{
   const int M = (height+1)>>1;
   const int odd = height%2 &1;

   const double k3 = 153.0 / (343.0 * sqrt(2.0));

   const auto X = x.Read();
   auto Y = y.Write();

   MFEM_FORALL(i, M,
   {
      const int l = RotateLeft(i,M);
      const int k = RotateLeft2(i,M);
      const int r = RotateRight(i,M);
      double Si = X[i];
      double Sr = X[r];
      const bool io = odd && i==M-1;
      const bool il = odd && l==M-1;
      const bool ik = odd && k==M-1;
      const bool xo = io || il|| ik;
      const double Xi = xo ?
      (5.*(864.*X[2*(M-1)%M]+1568.*X[M-1]-17.*(6.*X[M]+7.*X[M+1])))/1071. : 0.0;
      double Di = io ? Xi : X[M+i];
      double Dl = il ? Xi : X[M+l];
      double Dk = ik ? Xi : X[M+k];
      Si /= k3;
      Sr /= k3;
      Di *= k3;
      Dl *= k3;
      Dk *= k3;
      Si -= 1715.0 * Dl / 7344.0 + 16807.0 * Dk / 22032.0;
      Sr -= 1715.0 * Di / 7344.0 + 16807.0 * Dl / 22032.0;
      Di -= 45.0 * Si / 56.0 + 1215.0 * Sr / 2744.0;
      Si -= Di * 5.0 / 9.0;
      Y[2*i] = Si;
      if (odd && i==M-1) { return; }
      Y[2*i+1] = Di;
   });
}

} // namespace mfem
