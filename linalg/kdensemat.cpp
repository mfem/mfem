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

// **************************************************************************
void DenseMatrixSet(const double d,
                    const size_t size,
                    double *data){
   GET_CUDA;
   GET_ADRS(data);
   forall(i, size, d_data[i] = d;);
}

// **************************************************************************
void DenseMatrixTranspose(const size_t height,
                          const size_t width,
                          double *data,
                          const double *mdata){
   GET_CUDA;
   GET_CONST_ADRS(mdata);
   GET_ADRS(data);
   forall(i,height,{
         for (size_t j=0; j<width; j+=1){
            d_data[i+j*height] = d_mdata[j+i*height];
         }
      });
}

// *****************************************************************************
void kMultAAt(const size_t height, const size_t width,
              const double *a, double *aat){
   dbg();
   GET_CUDA;
   GET_CONST_ADRS(a);
   GET_ADRS(aat);
   forall(i, height,{
         for(size_t j=0; j<=i; j++){
            double temp = 0.0;
            for(size_t k=0; k<width; k++){
               temp += d_a[i+k*height] * d_a[j+k*height];
            }
            d_aat[j+i*height] = d_aat[i+j*height] = temp;
         }
      });
}

// *****************************************************************************
void kGradToDiv(const size_t n, const double *data, double *ddata){
   dbg();
   GET_CUDA;
   GET_CONST_ADRS(data);
   GET_ADRS(ddata);
   forall(i, n, d_ddata[i] = d_data[i];);
}

// *****************************************************************************
void kAddMult_a_VVt(const size_t n, const double a, const double *v,
                    const size_t height, double *VVt){
   dbg();
   GET_CUDA;
   GET_CONST_ADRS(v);
   GET_ADRS(VVt);
   forall(i, n, {
         double avi = a * d_v[i];
         for (int j = 0; j < i; j++){
            double avivj = avi * d_v[j];
            d_VVt[i+j*height] += avivj;
            d_VVt[j+i*height] += avivj;
         }
         d_VVt[i+i*height] += avi * d_v[i];
      });
      
}

// *****************************************************************************
void kMult0(const size_t height, double *y){
   dbg();
   GET_CUDA;
   GET_ADRS(y);
   forall(row, height, d_y[row] = 0.0;);
}

// *****************************************************************************
void kMult(const size_t height, const size_t width,
           const double *data, const double *x, double *y){
   dbg();
   GET_CUDA;
   GET_CONST_ADRS(data);
   GET_CONST_ADRS(x);
   GET_ADRS(y);
   forall(i, height,{
         double sum = 0.0;
         for(int j=0; j<width; j+=1){
            sum += d_x[j]*d_data[i+j*height];
         }
         d_y[i] = sum;
      });
}
// *****************************************************************************
MFEM_NAMESPACE_END
