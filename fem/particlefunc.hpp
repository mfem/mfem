// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.


#ifndef MFEM_PARTICLEFUNC
#define MFEM_PARTICLEFUNC

#include "particlespace.hpp"
#include "../linalg/linalg.hpp"

#ifdef MFEM_USE_GSLIB

namespace mfem
{

class ParticleFunction : public Vector
{
protected:
   const ParticleSpace &pspace;

public:
   ParticleFunction(const ParticleSpace &pspace_, int vdim=1)
   : Vector(pspace_.GetNP()*vdim), pspace(pspace_) {}

};

} // namespace mfem

#endif // MFEM_USE_GSLIB

#endif // MFEM_PARTICLEFUNC