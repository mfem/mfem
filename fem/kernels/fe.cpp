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
void kH1_TriangleElement_CalcShape(const size_t p,
                                   const double *shape_x,
                                   const double *shape_y,
                                   const double *shape_l,
                                   double *u){
   GET_CONST_ADRS(shape_x);
   GET_CONST_ADRS(shape_y);
   GET_CONST_ADRS(shape_l);
   GET_ADRS(u);
   forall(_dummy_,1,{
         int o = 0;
         for (size_t j = 0; j <= p; j++){
            for (size_t i = 0; i + j <= p; i++)
            {
               d_u[o++] = d_shape_x[i]*d_shape_y[j]*d_shape_l[p-i-j];
            }
         }
      });
}

// *****************************************************************************
void kH1_TriangleElement_CalcDShape(const size_t p,
                                    const size_t height,
                                    const double *shape_x,
                                    const double *shape_y,
                                    const double *shape_l,
                                    const double *dshape_x,
                                    const double *dshape_y,
                                    const double *dshape_l,
                                    double *du){
   GET_CONST_ADRS(shape_x);
   GET_CONST_ADRS(shape_y);
   GET_CONST_ADRS(shape_l);
   GET_CONST_ADRS(dshape_x);
   GET_CONST_ADRS(dshape_y);
   GET_CONST_ADRS(dshape_l);
   GET_ADRS(du);
   forall(_dummy_,1,{
         for (size_t o = 0, j = 0; j <= p; j++)
            for (size_t i = 0; i + j <= p; i++)
            {
               const size_t k = p - i - j;
               d_du[o+height*0] = ((d_dshape_x[i]* d_shape_l[k]) - ( d_shape_x[i]*d_dshape_l[k]))*d_shape_y[j];
               d_du[o+height*1] = ((d_dshape_y[j]* d_shape_l[k]) - ( d_shape_y[j]*d_dshape_l[k]))*d_shape_x[i];
               o++;
            }
      });
}

// *****************************************************************************
void kBasis(const size_t p, const double *x, double *w){
   GET_CONST_ADRS(x);
   GET_ADRS(w);
   forall(_dummy_,1,{
         for (size_t i = 0; i <= p; i++)
         {
            for (size_t j = 0; j < i; j++)
            {
               const double xij = d_x[i] - d_x[j];
               d_w[i] *=  xij;
               d_w[j] *= -xij;
            }
         }
         for (size_t i = 0; i <= p; i++)
         {
            d_w[i] = 1.0/d_w[i];
         }
      });
}

// *****************************************************************************
void kNodesAreIncreasing(const size_t p, const double *x){
   GET_CONST_ADRS(x);
   forall(i, p, {
         printf("\n\t%f >? %f",d_x[i],d_x[i+1]);fflush(0);
         assert(d_x[i] < d_x[i+1]);
      });
}

// *****************************************************************************
void kLinear3DFiniteElementHeightEq4(double *A){
   GET_ADRS(A);
   d_A[0] = -1.; d_A[4] = -1.; d_A[8]  = -1.;
   d_A[1] =  1.; d_A[5] =  0.; d_A[9]  =  0.;
   d_A[2] =  0.; d_A[6] =  1.; d_A[10] =  0.;
   d_A[3] =  0.; d_A[7] =  0.; d_A[11] =  1.;
}

// *****************************************************************************
MFEM_NAMESPACE_END
