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

namespace mfem
{

// *****************************************************************************
class RowNode
{
public:
   double Value;
   RowNode *Prev;
   int Column;
};

// *****************************************************************************
void kSparseMatrix(const int nrows, RowNode** Rows)
{
   GET_ADRS_T(Rows,RowNode*);
   MFEM_FORALL(i, nrows, d_Rows[i] = NULL;);
}

// *****************************************************************************
void kGauss_Seidel_forw_A_NULL(const size_t s,
                               RowNode **R,
                               const double *xp,
                               double *yp)
{
   GET_ADRS_T(R,RowNode*);
   GET_CONST_ADRS(xp);
   GET_ADRS(yp);
   MFEM_FORALL(i,s,
   {
      int c;
      double sum = 0.0;
      RowNode *diag_p = NULL;
      RowNode *n_p;
      for (n_p = d_R[i]; n_p != NULL; n_p = n_p->Prev)
      {
         if ((c = n_p->Column) == (int)i)
         {
            diag_p = n_p;
         }
         else
         {
            sum += n_p->Value * d_yp[c];
         }
      }
      if (diag_p != NULL && diag_p->Value != 0.0)
      {
         d_yp[i] = (d_xp[i] - sum) / diag_p->Value;
      }
      else if (d_xp[i] == sum)
      {
         d_yp[i] = sum;
      }
      else{
         assert(false);
      }
   });
}

// *****************************************************************************
void kGauss_Seidel_forw(const size_t height,
                        const int *Ip, const int *Jp, const double *Ap,
                        const double *xp,
                        double *yp)
{
   GET_CONST_ADRS_T(Ip,int);
   GET_CONST_ADRS_T(Jp,int);
   GET_CONST_ADRS(Ap);
   GET_CONST_ADRS(xp);
   GET_ADRS(yp);
   MFEM_FORALL(k,1,
   {
      for (size_t i=0; i<height; i+=1)
      {
         int d = -1;
         const int end = d_Ip[i+1];
         double sum = 0.0;
         for (int j = d_Ip[i]; j < end; j+=1)
         {
            const size_t c = d_Jp[j];
            const bool c_eq_i = c == i;
            d = c_eq_i ? j : d;
            const double Ay = d_Ap[j] * d_yp[c];
            sum += c_eq_i ? 0.0 : Ay;
         }
         const double A = d_Ap[d];
         const double x = d_xp[i];
         const double xmsda = (x - sum) / A;
         const bool dpaann = d >= 0 && A != 0.0;
         const bool xeqs = x == sum;
         assert(dpaann || xeqs);
         d_yp[i] = dpaann ? xmsda:sum;
      }
   });
}

// *****************************************************************************
void kGauss_Seidel_back(const size_t height,
                        const int *Ip, const int *Jp, const double *Ap,
                        const double *xp,
                        double *yp)
{
   GET_CONST_ADRS_T(Ip,int);
   GET_CONST_ADRS_T(Jp,int);
   GET_CONST_ADRS(Ap);
   GET_CONST_ADRS(xp);
   GET_ADRS(yp);
   MFEM_FORALL(k, 1,
   {
      for (int i = height-1; i >= 0; i--)
      {
         int d = -1;
         const int beg = d_Ip[i];
         double sum = 0.0;
         for (int j = d_Ip[i+1]-1; j >= beg; j-=1)
         {
            const int c = d_Jp[j];
            const bool c_eq_i = c == i;
            d = c_eq_i ? j : d;
            const double Ay = d_Ap[j] * d_yp[c];
            sum += c_eq_i ? 0.0 : Ay;
         }
         const double A = d_Ap[d];
         const double x = d_xp[i];
         const double xmsda = (x - sum) / A;
         const bool dpaann = d >= 0 && A != 0.0;
         const bool xeqs = x == sum;
         assert(dpaann || xeqs);
         d_yp[i] = dpaann ? xmsda:sum;
      }
   });
}


// *****************************************************************************
void kAddMult(const size_t height,
              const int *I, const int *J, const double *A,
              const double *x, double *y)
{
   GET_CONST_ADRS_T(I,int);
   GET_CONST_ADRS_T(J,int);
   GET_CONST_ADRS(A);
   GET_CONST_ADRS(x);
   GET_ADRS(y);
   MFEM_FORALL(i, height,
   {
      double d = 0.0;
      const size_t end = d_I[i+1];
      for (size_t j=d_I[i]; j < end; j+=1)
      {
         d += d_A[j] * d_x[d_J[j]];
      }
      d_y[i] += d;
   });
}

}
