// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../tmop.hpp"
#include "tmop_pa.hpp"
#include "../linearform.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/kernels.hpp"
#include "../../linalg/dinvariants.hpp"

namespace mfem
{

MFEM_REGISTER_TMOP_KERNELS(void, AssembleDiagonalPA_Kernel_Fit_3D,
                           const int NE,
                           const Vector &h0,
                           Vector &diagonal,
                           const int d1d,
                           const int q1d)
{
   constexpr int DIM = 3;
   const int D1D = T_D1D ? T_D1D : d1d;

   const auto H0 = Reshape(h0.Read(), DIM, DIM, D1D, D1D, D1D, NE);
   auto D = Reshape(diagonal.ReadWrite(), D1D, D1D, D1D, DIM, NE);

   mfem::forall_3D(NE, D1D, D1D, D1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;

      MFEM_FOREACH_THREAD(qz,z,D1D)
      {
        MFEM_FOREACH_THREAD(qy,y,D1D)
        {
            MFEM_FOREACH_THREAD(qx,x,D1D)
            {
                for (int v = 0; v < DIM; v++)
                {
                    D(qx,qy,qz,v,e) += H0(v,v,qx,qy,qz,e);;                 
                }
            }
        }
      }
      MFEM_SYNC_THREAD;
   });

}
void TMOP_Integrator::AssembleDiagonalPA_Fit_3D(Vector &D) const
{
   const int N = PA.ne;
   const int meshOrder = surf_fit_gf->FESpace()->GetMaxElementOrder();
   const int D1D = meshOrder + 1;
   const int Q1D = D1D;
   const int id = (D1D << 4 ) | Q1D;

   Vector &H0 = PA.H0Fit;

   MFEM_LAUNCH_TMOP_KERNEL(AssembleDiagonalPA_Kernel_Fit_3D,id,N,H0,D);
}

} // namespace mfem