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

#ifndef MFEM_BILIN_DQM_HPP
#define MFEM_BILIN_DQM_HPP

#include "../config/config.hpp"
#include "../general/karray.hpp"

namespace mfem
{

// ***************************************************************************
// * kDofQuadMaps
// ***************************************************************************
class kDofQuadMaps
{
private:
   std::string hash;
public:
   karray<double, false> dofToQuad, dofToQuadD; // B
   karray<double, false> quadToDof, quadToDofD; // B^T
   karray<double> quadWeights;
public:
   ~kDofQuadMaps();
   static void delkDofQuadMaps();
   static kDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                            const mfem::IntegrationRule&,
                            const bool = false);
   static kDofQuadMaps* Get(const mfem::FiniteElementSpace&,
                            const mfem::FiniteElementSpace&,
                            const mfem::IntegrationRule&,
                            const bool = false);
   static kDofQuadMaps* Get(const mfem::FiniteElement&,
                            const mfem::FiniteElement&,
                            const mfem::IntegrationRule&,
                            const bool = false);
   static kDofQuadMaps* GetTensorMaps(const mfem::FiniteElement&,
                                      const mfem::FiniteElement&,
                                      const mfem::IntegrationRule&,
                                      const bool = false);
   static kDofQuadMaps* GetD2QTensorMaps(const mfem::FiniteElement&,
                                         const mfem::IntegrationRule&,
                                         const bool = false);
   static kDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                       const mfem::IntegrationRule&,
                                       const bool = false);
   static kDofQuadMaps* GetSimplexMaps(const mfem::FiniteElement&,
                                       const mfem::FiniteElement&,
                                       const mfem::IntegrationRule&,
                                       const bool = false);
   static kDofQuadMaps* GetD2QSimplexMaps(const mfem::FiniteElement&,
                                          const mfem::IntegrationRule&,
                                          const bool = false);
};

} // namespace mfem

#endif // MFEM_BACKENDS_KERNELS_BILIN_DQM_HPP
