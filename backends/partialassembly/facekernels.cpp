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

#include "facekernels.hpp"

namespace mfem{

namespace pa{

void Permutation::Permutation2d(int face_id, int nbe, int dofs1d, const Tensor3d& T0, Tensor3d& T0p) const
{
   for (int e = 0; e < nbe; ++e)
   {
      const int trial = kernel_data(e,face_id).indirection;
      const int permutation = kernel_data(e,face_id).permutation;
      if(trial!=-1)
      {
         switch(permutation)
         {
         case 0:
            for (int i2 = 0; i2 < dofs1d; ++i2)
            {
               for (int i1 = 0; i1 < dofs1d; ++i1)
               {
                  T0p(i1,i2,e) = T0(i1,i2,trial);
               }
            }
            break;
         case 1:
            for (int i2 = 0, j1 = dofs1d-1; i2 < dofs1d; ++i2, --j1)
            {
               for (int i1 = 0, j2 = 0; i1 < dofs1d; ++i1, ++j2)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
            break;
         case 2:
            for (int i2 = 0, j2 = dofs1d-1; i2 < dofs1d; ++i2, --j2)
            {
               for (int i1 = 0, j1 = dofs1d-1; i1 < dofs1d; ++i1, --j1)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
            break;
         case 3:
            for (int i2 = 0, j1 = 0; i2 < dofs1d; ++i2, ++j1)
            {
               for (int i1 = 0, j2 = dofs1d-1; i1 < dofs1d; ++i1, --j2)
               {
                  T0p(i1,i2,e) = T0(j1,j2,trial);
               }
            }
            break;
         default:
            mfem_error("This permutation id does not exist in 2D");
         }
      }else{
         for (int i2 = 0; i2 < dofs1d; ++i2)
         {
            for (int i1 = 0; i1 < dofs1d; ++i1)
            {
               T0p(i1,i2,e) = 0.0;
            }
         }
      }
   }
}

/**
*  The Permutation3d works differently than the 2d version, here we receive a change of basis matrix encrypted in one integer.
*/
void Permutation::Permutation3d(int face_id, int nbe, int dofs1d, const Tensor4d& T0, Tensor4d& T0p) const
{
   const double* U = T0.getData();
   int elt, ii, jj, kk;
   const int step_elt = dofs1d*dofs1d*dofs1d;
   for (int e = 0; e < nbe; ++e)
   {
      const int trial = kernel_data(e,face_id).indirection;
      const int permutation = kernel_data(e,face_id).permutation;
      if (trial!=-1)
      {
         elt = trial*step_elt;
         IntMatrix P(3,3);
         GetChangeOfBasis(permutation, P);
         int begin_ii = (P(0,0)==-1)*(dofs1d-1) + (P(1,0)==-1)*(dofs1d*dofs1d-1) + (P(2,0)==-1)*(dofs1d*dofs1d*dofs1d-1);
         int begin_jj = (P(0,1)==-1)*(dofs1d-1) + (P(1,1)==-1)*(dofs1d*dofs1d-1) + (P(2,1)==-1)*(dofs1d*dofs1d*dofs1d-1);
         int begin_kk = (P(0,2)==-1)*(dofs1d-1) + (P(1,2)==-1)*(dofs1d*dofs1d-1) + (P(2,2)==-1)*(dofs1d*dofs1d*dofs1d-1);
         int step_ii  = P(0,0) + P(1,0)*dofs1d + P(2,0)*dofs1d*dofs1d;
         int step_jj  = P(0,1) + P(1,1)*dofs1d + P(2,1)*dofs1d*dofs1d;
         int step_kk  = P(0,2) + P(1,2)*dofs1d + P(2,2)*dofs1d*dofs1d;
         kk = begin_kk;
         for (int k = 0; k < dofs1d; ++k)
         {
            jj = begin_jj;
            for (int j = 0; j < dofs1d; ++j)
            {
               ii = begin_ii;
               for (int i = 0; i < dofs1d; ++i)
               {
                  T0p(i,j,k,e) = U[ elt + ii + jj + kk ];
                  ii += step_ii;
               }
               jj += step_jj;
            }
            kk += step_kk;
         }
      }
      else
      {
         for (int k = 0; k < dofs1d; ++k)
         {
            for (int j = 0; j < dofs1d; ++j)
            {
               for (int i = 0; i < dofs1d; ++i)
               {
                  T0p(i,j,k,e) = 0.0;
               }
            }
         }
      }
   }
}

}

}