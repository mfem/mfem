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
MFEM_NAMESPACE_END
