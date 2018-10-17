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

#include "../okina.hpp"
#include "array.hpp"

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
// for (int i = size; i < nsize; i++){
//    ((T*)data)[i] = initval;
// }
// *****************************************************************************
void kArrayInitVal(const size_t size, const size_t nsize,
                   void *data, size_t Tsizeof, void *initval){
   GET_ADRS_T(data,void);
   
   forall(k, nsize-size, {
         const size_t i = k+size;
         size_t *xs = ((size_t*)d_data)+i;
         *xs = *(size_t*)initval;
      });
}

// *****************************************************************************
double *kArrayInitGet(const int p, double **pts){
   GET_CUDA;
   GET_ADRS_T(pts,double*);
   if (cuda){ assert(false); }
   return d_pts[p];
}

// *****************************************************************************
void* kArrayVoidGet(const int p, void **bases){
   GET_CUDA;
   GET_ADRS_T(bases,void*);
   if (cuda){ assert(false); }
   return d_bases[p];
}

// *****************************************************************************
void kArrayInitSet(double **d_pts, double *adrs){
   GET_CUDA;
   if (cuda){ assert(false); }
   *d_pts = adrs;
}

// *****************************************************************************
MFEM_NAMESPACE_END
