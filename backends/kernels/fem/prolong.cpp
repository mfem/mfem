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

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#include "../kernels.hpp"

namespace mfem {
  
namespace kernels {
/*
   // ***************************************************************************
   // * RajaProlongationOperator
   // ***************************************************************************
   RajaProlongationOperator::RajaProlongationOperator
   (const RajaConformingProlongationOperator* Op):
      Operator(Op->Height(), Op->Width()),pmat(Op){}
  
   // ***************************************************************************
   void RajaProlongationOperator::Mult(const kernels::kvector& x,
                                       kernels::kvector& y) const {
      push(LightSteelBlue);
      if (config::Get().IAmAlone()){
         y=x;
         pop();
         return;
      }
    
      if (!config::Get().DoHostConformingProlongationOperator()){
         dbg("\n\033[35m[DEVICE::Mult]\033[m");
         pmat->d_Mult(x, y);
         pop();
         return;
      }else{
         dbg("\n\033[35m[HOST::Mult]\033[m");
      }

      push(hostX:D2H,Red);
      const Vector hostX=x;//D2H
      pop(); 
    
      push(hostY,LightSteelBlue);
      Vector hostY(y.Size());
      pop();
    
      push(pmat->Mult,LightSteelBlue);
      pmat->h_Mult(hostX, hostY);
      pop();
    
      push(hostY:H2D,Red);
      y=hostY;//H2D
      pop();
    
      pop();
   }

   // ***************************************************************************
   void RajaProlongationOperator::MultTranspose(const kernels::kvector& x,
                                                kernels::kvector& y) const {
      push(LightSteelBlue);
      if (config::Get().IAmAlone()){
         y=x;
         pop();
         return;
      }
    
      if (!config::Get().DoHostConformingProlongationOperator()){
         dbg("\n\033[35m[DEVICE::MultTranspose]\033[m");
         pmat->d_MultTranspose(x, y);
         pop();
         return;
      }else{
         dbg("\n\033[35m[HOST::MultTranspose]\033[m");
      }

      push(hostX:D2H,Red);
      const Vector hostX=x;
      pop();
   
      push(hostY,LightSteelBlue);
      mfem::Vector hostY(y.Size());
      pop();

      push(pmat->MultT,LightSteelBlue);
      pmat->h_MultTranspose(hostX, hostY);
      pop();
    
      push(hostY:H2D,Red);
      y=hostY;//H2D
      pop();
    
      pop();
   }
*/
} // namespace mfem::kernels
   
} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)
