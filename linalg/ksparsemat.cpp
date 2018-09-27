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

#include "../general/okina.hpp"
using namespace std;

// *****************************************************************************
MFEM_NAMESPACE

class RowNode{
public:
   double Value;
   RowNode *Prev;
   int Column;
};

// *****************************************************************************
void kGauss_Seidel_forw_A_NULL(const size_t s,
                               RowNode **R,
                               const double *xp,
                               double *yp){
   GET_CUDA;
   GET_ADRS_T(R,RowNode*);
   GET_CONST_ADRS(xp);
   GET_ADRS(yp);
   forall(i,s,{
         int c;
         double sum = 0.0;
         RowNode *diag_p = NULL;
         RowNode *n_p;
         for (n_p = d_R[i]; n_p != NULL; n_p = n_p->Prev){
            if ((c = n_p->Column) == i){
               diag_p = n_p;
            }else{
               sum += n_p->Value * d_yp[c];
            }
         }
         if (diag_p != NULL && diag_p->Value != 0.0){
            d_yp[i] = (d_xp[i] - sum) / diag_p->Value;
         }
         else if (d_xp[i] == sum){
            d_yp[i] = sum;
         }else{
            assert(false);
         }
      });
}

// *****************************************************************************
void kGauss_Seidel_forw(const size_t s,
                        const int *Ip, const int *Jp, const double *Ap,
                        const double *xp,
                        double *yp){
   GET_CUDA;
   GET_CONST_ADRS_T(Ip,int);
   GET_CONST_ADRS_T(Jp,int);
   GET_CONST_ADRS(Ap);
   GET_CONST_ADRS(xp);
   GET_ADRS(yp);
   
   forall(k,1,{
         int j = d_Ip[0];
         for (int i = 0; i < s; i++){
            int c;
            //int j_nxt = 0;
            //int j = (i==0) ? d_Ip[0]: j_nxt;
            int end = d_Ip[i+1];
            double sum = 0.0;
            int d = -1;
            for ( ; j < end; j++){
               if ((c = d_Jp[j]) == i){
                  d = j;
               } else {
                  sum += d_Ap[j] * d_yp[c];
               }
            }
            //j_nxt = j;
            if (d >= 0 && d_Ap[d] != 0.0) {
               d_yp[i] = (d_xp[i] - sum) / d_Ap[d];
            }
            else if (d_xp[i] == sum) {
               d_yp[i] = sum;
            } else {
               assert(false);
               //mfem_error("SparseMatrix::Gauss_Seidel_forw(...) #2");
            }
         }
      });
}

// *****************************************************************************
void kGauss_Seidel_back(const size_t height,
                        const int *Ip, const int *Jp, const double *Ap,
                        const double *xp,
                        double *yp){

   GET_CUDA;
   GET_CONST_ADRS_T(Ip,int);
   GET_CONST_ADRS_T(Jp,int);
   GET_CONST_ADRS(Ap);
   GET_CONST_ADRS(xp);
   GET_ADRS(yp);
   
   //j = Ip[height]-1;
   forall(k,1,{
         int j = d_Ip[height]-1;
         for (int i = height-1; i >= 0; i--){
            //const int i = (height-1)-k;
            int c;
            //int j_nxt = 0;
            //int j = (k==0) ? d_Ip[height]-1: j_nxt;
            int beg = d_Ip[i];
            double sum = 0.0;
            int d = -1;
            for ( ; j >= beg; j--){
               if ((c = d_Jp[j]) == i)
               {
                  d = j;
               }
               else
               {
                  sum += d_Ap[j] * d_yp[c];
               }
            }
            //j_nxt = j;
            
            if (d >= 0 && d_Ap[d] != 0.0)
            {
               d_yp[i] = (d_xp[i] - sum) / d_Ap[d];
            }
            else if (d_xp[i] == sum)
            {
               d_yp[i] = sum;
            }
            else
            {
               assert(false);
               //mfem_error("SparseMatrix::Gauss_Seidel_back(...) #2");
            }
         }
      });
}
   

// *****************************************************************************
void kAddMult(const size_t height,
              const int *I, const int *J, const double *A,
              const double *x, double *y){
   GET_CUDA;
   GET_CONST_ADRS_T(I,int);
   GET_CONST_ADRS_T(J,int);
   GET_CONST_ADRS(A);
   GET_CONST_ADRS(x);
   GET_ADRS(y);
//#warning dummy external forloop
   forall(k, 1, {
         int j;
         for (int i = j = 0; i < height; i++){
            double d = 0.0;
            for (int end = d_I[i+1]; j < end; j++){
               d += d_A[j] * d_x[d_J[j]];
            }
            d_y[i] += d;
         }
      });
   
      /*for (i = j = 0; i < height; i++)
      {
         double d = 0.0;
         for (end = Ip[i+1]; j < end; j++)
         {
            d += Ap[j] * xp[Jp[j]];
         }
         yp[i] += d;
         }*/

   
}

// *****************************************************************************
MFEM_NAMESPACE_END
