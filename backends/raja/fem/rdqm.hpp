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

#ifndef MFEM_BACKENDS_RAJA_BILIN_DQM_HPP
#define MFEM_BACKENDS_RAJA_BILIN_DQM_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

// ***************************************************************************
// * RajaDofQuadMaps
// ***************************************************************************
class RajaDofQuadMaps
{
private:
   std::string hash;
public:
   raja::array<double, false> dofToQuad, dofToQuadD; // B
   raja::array<double, false> quadToDof, quadToDofD; // B^T
   raja::array<double> quadWeights;
public:
   ~RajaDofQuadMaps();
   static void delRajaDofQuadMaps();
   static RajaDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                               const mfem::FiniteElementSpace&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* Get(const mfem::FiniteElement&,
                               const mfem::FiniteElement&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* GetTensorMaps(const mfem::FiniteElement&,
                                         const mfem::FiniteElement&,
                                         const mfem::IntegrationRule&,
                                         const bool = false);
   static RajaDofQuadMaps* GetD2QTensorMaps(const mfem::FiniteElement&,
                                            const mfem::IntegrationRule&,
                                            const bool = false);
   static RajaDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                          const mfem::IntegrationRule&,
                                          const bool = false);
   static RajaDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                          const mfem::FiniteElement&,
                                          const mfem::IntegrationRule&,
                                          const bool = false);
   static RajaDofQuadMaps* GetD2QSimplexMaps(const mfem::FiniteElement&,
                                             const mfem::IntegrationRule&,
                                             const bool = false);
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_BILIN_DQM_HPP
