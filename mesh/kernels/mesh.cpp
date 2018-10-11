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
using namespace std;

// *****************************************************************************
MFEM_NAMESPACE

// *****************************************************************************
__constant__ double d_quad_children_init[2*4*4] = {
   0.0,0.0, 0.5,0.0, 0.5,0.5, 0.0,0.5,
   0.5,0.0, 1.0,0.0, 1.0,0.5, 0.5,0.5, 
   0.5,0.5, 1.0,0.5, 1.0,1.0, 0.5,1.0, 
   0.0,0.5, 0.5,0.5, 0.5,1.0, 0.0,1.0  
};

const double h_quad_children_init[2*4*4] = {
   0.0,0.0, 0.5,0.0, 0.5,0.5, 0.0,0.5,
   0.5,0.0, 1.0,0.0, 1.0,0.5, 0.5,0.5, 
   0.5,0.5, 1.0,0.5, 1.0,1.0, 0.5,1.0, 
   0.0,0.5, 0.5,0.5, 0.5,1.0, 0.0,1.0  
};

// *****************************************************************************
void kQuadChildren(double *data){
   GET_CUDA;
   GET_ADRS(data);
   const double *p_h_quad_children_init = h_quad_children_init;
   const int N = 2*4*4;
   forall(i, N,{
          d_data[i] = cuda?
          d_quad_children_init[i]:
          p_h_quad_children_init[i];
          //printf(" %f",d_data[i]);
      });
}

// *****************************************************************************
MFEM_NAMESPACE_END
