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

#include "../../general/forall.hpp"
#include "../bilininteg.hpp"
#include "../gridfunc.hpp"
#include "bilininteg_mass_kernels.hpp"

namespace mfem
{

void MassIntegrator::AssembleEA(const FiniteElementSpace &fes,
                                Vector &ea_data,
                                const bool add)
{
   using internal::EAMassAssemble1D;
   using internal::EAMassAssemble2D;
   using internal::EAMassAssemble3D;

   AssemblePA(fes);
   ne = fes.GetMesh()->GetNE();
   const Array<real_t> &B = maps->B;
   if (dim == 1)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EAMassAssemble1D<2,2>(ne,B,pa_data,ea_data,add);
         case 0x33: return EAMassAssemble1D<3,3>(ne,B,pa_data,ea_data,add);
         case 0x44: return EAMassAssemble1D<4,4>(ne,B,pa_data,ea_data,add);
         case 0x55: return EAMassAssemble1D<5,5>(ne,B,pa_data,ea_data,add);
         case 0x66: return EAMassAssemble1D<6,6>(ne,B,pa_data,ea_data,add);
         case 0x77: return EAMassAssemble1D<7,7>(ne,B,pa_data,ea_data,add);
         case 0x88: return EAMassAssemble1D<8,8>(ne,B,pa_data,ea_data,add);
         case 0x99: return EAMassAssemble1D<9,9>(ne,B,pa_data,ea_data,add);
         default:   return EAMassAssemble1D(ne,B,pa_data,ea_data,add,
                                               dofs1D,quad1D);
      }
   }
   else if (dim == 2)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x22: return EAMassAssemble2D<2,2>(ne,B,pa_data,ea_data,add);
         case 0x33: return EAMassAssemble2D<3,3>(ne,B,pa_data,ea_data,add);
         case 0x44: return EAMassAssemble2D<4,4>(ne,B,pa_data,ea_data,add);
         case 0x55: return EAMassAssemble2D<5,5>(ne,B,pa_data,ea_data,add);
         case 0x66: return EAMassAssemble2D<6,6>(ne,B,pa_data,ea_data,add);
         case 0x77: return EAMassAssemble2D<7,7>(ne,B,pa_data,ea_data,add);
         case 0x88: return EAMassAssemble2D<8,8>(ne,B,pa_data,ea_data,add);
         case 0x99: return EAMassAssemble2D<9,9>(ne,B,pa_data,ea_data,add);
         default:   return EAMassAssemble2D(ne,B,pa_data,ea_data,add,
                                               dofs1D,quad1D);
      }
   }
   else if (dim == 3)
   {
      switch ((dofs1D << 4 ) | quad1D)
      {
         case 0x23: return EAMassAssemble3D<2,3>(ne,B,pa_data,ea_data,add);
         case 0x34: return EAMassAssemble3D<3,4>(ne,B,pa_data,ea_data,add);
         case 0x45: return EAMassAssemble3D<4,5>(ne,B,pa_data,ea_data,add);
         case 0x56: return EAMassAssemble3D<5,6>(ne,B,pa_data,ea_data,add);
         case 0x67: return EAMassAssemble3D<6,7>(ne,B,pa_data,ea_data,add);
         case 0x78: return EAMassAssemble3D<7,8>(ne,B,pa_data,ea_data,add);
         case 0x89: return EAMassAssemble3D<8,9>(ne,B,pa_data,ea_data,add);
         default:   return EAMassAssemble3D(ne,B,pa_data,ea_data,add,
                                               dofs1D,quad1D);
      }
   }
   MFEM_ABORT("Unknown kernel.");
}

}
