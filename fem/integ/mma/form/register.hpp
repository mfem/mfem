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
#pragma once

/** Shared MMA specialization tables.
    Mass↔VectorMass and Diffusion↔VectorDiffusion share identical simplex
    lists; all four bilinear integrators share the tensor (p=3..7) list.
    DomainLF keeps its own (smaller) simplex table. */

namespace mfem
{
namespace internal
{
namespace mma
{

/// Mass / VectorMass simplex (DIM, D1D, QND). Order: DIM, then D1D, then QND.
template <class Integrator>
void RegisterMassSimplexMmaSpecializations()
{
   Integrator::template AddSimplexMmaSpecialization<2,2,3>();
   Integrator::template AddSimplexMmaSpecialization<2,2,4>();
   Integrator::template AddSimplexMmaSpecialization<2,2,9>();
   Integrator::template AddSimplexMmaSpecialization<2,2,12>();
   Integrator::template AddSimplexMmaSpecialization<2,2,16>();
   Integrator::template AddSimplexMmaSpecialization<2,2,25>();
   Integrator::template AddSimplexMmaSpecialization<2,2,33>();
   Integrator::template AddSimplexMmaSpecialization<2,3,6>();
   Integrator::template AddSimplexMmaSpecialization<2,3,9>();
   Integrator::template AddSimplexMmaSpecialization<2,3,16>();
   Integrator::template AddSimplexMmaSpecialization<2,3,25>();
   Integrator::template AddSimplexMmaSpecialization<2,3,33>();
   Integrator::template AddSimplexMmaSpecialization<2,3,36>();
   Integrator::template AddSimplexMmaSpecialization<2,3,42>();
   Integrator::template AddSimplexMmaSpecialization<2,4,12>();
   Integrator::template AddSimplexMmaSpecialization<2,4,16>();
   Integrator::template AddSimplexMmaSpecialization<2,4,25>();
   Integrator::template AddSimplexMmaSpecialization<2,5,16>();
   Integrator::template AddSimplexMmaSpecialization<2,5,33>();
   Integrator::template AddSimplexMmaSpecialization<2,6,25>();
   Integrator::template AddSimplexMmaSpecialization<2,6,36>();
   Integrator::template AddSimplexMmaSpecialization<2,6,42>();
   Integrator::template AddSimplexMmaSpecialization<2,6,49>();
   Integrator::template AddSimplexMmaSpecialization<2,6,55>();
   Integrator::template AddSimplexMmaSpecialization<2,6,64>();
   Integrator::template AddSimplexMmaSpecialization<2,6,67>();
   Integrator::template AddSimplexMmaSpecialization<2,6,79>();
   Integrator::template AddSimplexMmaSpecialization<2,6,81>();
   Integrator::template AddSimplexMmaSpecialization<2,7,33>();
   Integrator::template AddSimplexMmaSpecialization<2,7,49>();
   Integrator::template AddSimplexMmaSpecialization<2,7,55>();
   Integrator::template AddSimplexMmaSpecialization<2,7,64>();
   Integrator::template AddSimplexMmaSpecialization<2,7,67>();
   Integrator::template AddSimplexMmaSpecialization<2,7,79>();
   Integrator::template AddSimplexMmaSpecialization<2,7,81>();
   Integrator::template AddSimplexMmaSpecialization<2,7,100>();
   Integrator::template AddSimplexMmaSpecialization<2,7,126>();
   Integrator::template AddSimplexMmaSpecialization<2,8,42>();
   Integrator::template AddSimplexMmaSpecialization<3,2,4>();
   Integrator::template AddSimplexMmaSpecialization<3,2,8>();
   Integrator::template AddSimplexMmaSpecialization<3,2,14>();
   Integrator::template AddSimplexMmaSpecialization<3,2,24>();
   Integrator::template AddSimplexMmaSpecialization<3,3,14>();
   Integrator::template AddSimplexMmaSpecialization<3,3,27>();
   Integrator::template AddSimplexMmaSpecialization<3,3,35>();
   Integrator::template AddSimplexMmaSpecialization<3,3,46>();
   Integrator::template AddSimplexMmaSpecialization<3,4,24>();
   Integrator::template AddSimplexMmaSpecialization<3,4,59>();
   Integrator::template AddSimplexMmaSpecialization<3,4,81>();
   Integrator::template AddSimplexMmaSpecialization<3,5,46>();
   Integrator::template AddSimplexMmaSpecialization<3,5,96>();
   Integrator::template AddSimplexMmaSpecialization<3,5,123>();
   Integrator::template AddSimplexMmaSpecialization<3,6,81>();
   Integrator::template AddSimplexMmaSpecialization<3,6,145>();
   Integrator::template AddSimplexMmaSpecialization<3,6,175>();
   Integrator::template AddSimplexMmaSpecialization<3,6,216>();
   Integrator::template AddSimplexMmaSpecialization<3,7,123>();
   Integrator::template AddSimplexMmaSpecialization<3,7,209>();
   Integrator::template AddSimplexMmaSpecialization<3,7,248>();
   Integrator::template AddSimplexMmaSpecialization<3,8,175>();
   Integrator::template AddSimplexMmaSpecialization<3,8,284>();
}

/// Diffusion / VectorDiffusion simplex (DIM, D1D, QND).
template <class Integrator>
void RegisterDiffusionSimplexMmaSpecializations()
{
   Integrator::template AddSimplexMmaSpecialization<2,2,1>();
   Integrator::template AddSimplexMmaSpecialization<2,2,4>();
   Integrator::template AddSimplexMmaSpecialization<2,2,9>();
   Integrator::template AddSimplexMmaSpecialization<2,2,12>();
   Integrator::template AddSimplexMmaSpecialization<2,2,16>();
   Integrator::template AddSimplexMmaSpecialization<2,2,25>();
   Integrator::template AddSimplexMmaSpecialization<2,2,33>();
   Integrator::template AddSimplexMmaSpecialization<2,3,3>();
   Integrator::template AddSimplexMmaSpecialization<2,3,9>();
   Integrator::template AddSimplexMmaSpecialization<2,3,16>();
   Integrator::template AddSimplexMmaSpecialization<2,3,25>();
   Integrator::template AddSimplexMmaSpecialization<2,3,33>();
   Integrator::template AddSimplexMmaSpecialization<2,3,36>();
   Integrator::template AddSimplexMmaSpecialization<2,3,42>();
   Integrator::template AddSimplexMmaSpecialization<2,4,6>();
   Integrator::template AddSimplexMmaSpecialization<2,4,16>();
   Integrator::template AddSimplexMmaSpecialization<2,4,25>();
   Integrator::template AddSimplexMmaSpecialization<2,5,12>();
   Integrator::template AddSimplexMmaSpecialization<2,5,33>();
   Integrator::template AddSimplexMmaSpecialization<2,6,16>();
   Integrator::template AddSimplexMmaSpecialization<2,6,36>();
   Integrator::template AddSimplexMmaSpecialization<2,6,42>();
   Integrator::template AddSimplexMmaSpecialization<2,6,49>();
   Integrator::template AddSimplexMmaSpecialization<2,6,55>();
   Integrator::template AddSimplexMmaSpecialization<2,6,64>();
   Integrator::template AddSimplexMmaSpecialization<2,6,67>();
   Integrator::template AddSimplexMmaSpecialization<2,6,79>();
   Integrator::template AddSimplexMmaSpecialization<2,6,81>();
   Integrator::template AddSimplexMmaSpecialization<2,7,25>();
   Integrator::template AddSimplexMmaSpecialization<2,7,49>();
   Integrator::template AddSimplexMmaSpecialization<2,7,55>();
   Integrator::template AddSimplexMmaSpecialization<2,7,64>();
   Integrator::template AddSimplexMmaSpecialization<2,7,67>();
   Integrator::template AddSimplexMmaSpecialization<2,7,79>();
   Integrator::template AddSimplexMmaSpecialization<2,7,81>();
   Integrator::template AddSimplexMmaSpecialization<2,7,100>();
   Integrator::template AddSimplexMmaSpecialization<2,7,126>();
   Integrator::template AddSimplexMmaSpecialization<2,8,33>();
   Integrator::template AddSimplexMmaSpecialization<3,2,1>();
   Integrator::template AddSimplexMmaSpecialization<3,2,8>();
   Integrator::template AddSimplexMmaSpecialization<3,2,24>();
   Integrator::template AddSimplexMmaSpecialization<3,3,4>();
   Integrator::template AddSimplexMmaSpecialization<3,3,27>();
   Integrator::template AddSimplexMmaSpecialization<3,3,46>();
   Integrator::template AddSimplexMmaSpecialization<3,4,14>();
   Integrator::template AddSimplexMmaSpecialization<3,4,81>();
   Integrator::template AddSimplexMmaSpecialization<3,5,24>();
   Integrator::template AddSimplexMmaSpecialization<3,5,123>();
   Integrator::template AddSimplexMmaSpecialization<3,6,46>();
   Integrator::template AddSimplexMmaSpecialization<3,6,175>();
   Integrator::template AddSimplexMmaSpecialization<3,6,216>();
   Integrator::template AddSimplexMmaSpecialization<3,7,81>();
   Integrator::template AddSimplexMmaSpecialization<3,7,248>();
   Integrator::template AddSimplexMmaSpecialization<3,8,123>();
}

/// Shared tensor MMA list: p = 3..7 (D1D = 4..8, Q1D = D1D+1).
template <class Integrator>
void RegisterTensorsMmaSpecializations()
{
   Integrator::template AddTensorsMmaSpecialization<2,4,5>();
   Integrator::template AddTensorsMmaSpecialization<2,5,6>();
   Integrator::template AddTensorsMmaSpecialization<2,6,7>();
   Integrator::template AddTensorsMmaSpecialization<2,7,8>();
   Integrator::template AddTensorsMmaSpecialization<2,8,9>();
   Integrator::template AddTensorsMmaSpecialization<3,4,5>();
   Integrator::template AddTensorsMmaSpecialization<3,5,6>();
   Integrator::template AddTensorsMmaSpecialization<3,6,7>();
   Integrator::template AddTensorsMmaSpecialization<3,7,8>();
   Integrator::template AddTensorsMmaSpecialization<3,8,9>();
}

} // namespace mma
} // namespace internal
} // namespace mfem
