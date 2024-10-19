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

#include "tmop_pa_h3s.hpp"

namespace mfem {

extern void TMOPAssembleGradPA_302(TMOPSetupGradPA3D &k);
extern void TMOPAssembleGradPA_303(TMOPSetupGradPA3D &k);
extern void TMOPAssembleGradPA_315(TMOPSetupGradPA3D &k);
extern void TMOPAssembleGradPA_318(TMOPSetupGradPA3D &k);
extern void TMOPAssembleGradPA_321(TMOPSetupGradPA3D &k);
extern void TMOPAssembleGradPA_332(TMOPSetupGradPA3D &k);
extern void TMOPAssembleGradPA_338(TMOPSetupGradPA3D &k);

void TMOP_Integrator::AssembleGradPA_3D(const Vector &x) const {
  const int mid = metric->Id();

  TMOPSetupGradPA3D ker(this, x);

  if (mid == 302) {
    return TMOPAssembleGradPA_302(ker);
  }
  if (mid == 303) {
    return TMOPAssembleGradPA_303(ker);
  }
  if (mid == 315) {
    return TMOPAssembleGradPA_315(ker);
  }
  if (mid == 318) {
    return TMOPAssembleGradPA_318(ker);
  }
  if (mid == 321) {
    return TMOPAssembleGradPA_321(ker);
  }
  if (mid == 332) {
    return TMOPAssembleGradPA_332(ker);
  }
  if (mid == 338) {
    return TMOPAssembleGradPA_338(ker);
  }

  MFEM_ABORT("Unsupported TMOP metric " << mid);
}

}  // namespace mfem
