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

#ifndef MFEM_BILIN_INTEG_DIFFUSION_HPP
#define MFEM_BILIN_INTEG_DIFFUSION_HPP

#include "../config/config.hpp"
#include "doftoquad.hpp"

namespace mfem
{

class kGeometry;

// *****************************************************************************
class KDiffusionIntegrator
{
private:
   Vector vec;
   kGeometry *geo;
   kDofQuadMaps *maps;
   const FiniteElementSpace *fes;
   const IntegrationRule *ir;
public:
   KDiffusionIntegrator(const FiniteElementSpace*,const IntegrationRule*);
   ~KDiffusionIntegrator();
   void Assemble();
   void MultAdd(Vector &x, Vector &y);
};

}

#endif // MFEM_BILIN_INTEG_DIFFUSION_HPP
