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
#include "../../fem/fem.hpp"
#include "fe.hpp"

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
void kH1_TriangleElement(const size_t p,
                         const size_t k,
                         const size_t height,
                         const double *shape_x,
                         const double *shape_y,
                         const double *shape_l,
                         double *T){
   GET_CONST_ADRS(shape_x);
   GET_CONST_ADRS(shape_y);
   GET_CONST_ADRS(shape_l);
   GET_ADRS(T);
   forall(_dummy_,1,{
         int o = 0;
         for (size_t j = 0; j <= p; j++){
            for (size_t i = 0; i + j <= p; i++)
            {
               d_T[o + k*height] = d_shape_x[i]*d_shape_y[j]*d_shape_l[p-i-j];
               o+=1;
            }
         }
      });
}

// *****************************************************************************
MFEM_NAMESPACE_END
