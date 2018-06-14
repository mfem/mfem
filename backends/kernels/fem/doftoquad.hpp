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

#ifndef MFEM_BACKENDS_KERNELS_BILIN_DQM_HPP
#define MFEM_BACKENDS_KERNELS_BILIN_DQM_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

// ***************************************************************************
// * KernelsDofQuadMaps
// ***************************************************************************
class KernelsDofQuadMaps
{
private:
   std::string hash;
public:
   kernels::array<double, false> dofToQuad, dofToQuadD; // B
   kernels::array<double, false> quadToDof, quadToDofD; // B^T
   kernels::array<double> quadWeights;
public:
   ~KernelsDofQuadMaps();
   static void delKernelsDofQuadMaps();
   static KernelsDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static KernelsDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                               const mfem::FiniteElementSpace&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static KernelsDofQuadMaps* Get(const mfem::FiniteElement&,
                               const mfem::FiniteElement&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static KernelsDofQuadMaps* GetTensorMaps(const mfem::FiniteElement&,
                                         const mfem::FiniteElement&,
                                         const mfem::IntegrationRule&,
                                         const bool = false);
   static KernelsDofQuadMaps* GetD2QTensorMaps(const mfem::FiniteElement&,
                                            const mfem::IntegrationRule&,
                                            const bool = false);
   static KernelsDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                          const mfem::IntegrationRule&,
                                          const bool = false);
   static KernelsDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                          const mfem::FiniteElement&,
                                          const mfem::IntegrationRule&,
                                          const bool = false);
   static KernelsDofQuadMaps* GetD2QSimplexMaps(const mfem::FiniteElement&,
                                             const mfem::IntegrationRule&,
                                             const bool = false);
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_BILIN_DQM_HPP
