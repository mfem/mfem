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

#ifndef MFEM_BACKENDS_RAJA_BILIN_INTEG_MASS_HPP
#define MFEM_BACKENDS_RAJA_BILIN_INTEG_MASS_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

class RajaMassIntegrator : public RajaIntegrator
{
private:
   RajaCoefficient coeff;
   raja::Vector assembledOperator;
   mfem::Vector op;
public:
   RajaMassIntegrator(const mfem::Engine&);
   RajaMassIntegrator(const RajaCoefficient&);
   virtual ~RajaMassIntegrator();
   virtual std::string GetName();
   virtual void SetupIntegrationRule();
   virtual void Setup();
   virtual void Assemble();
   void SetOperator(mfem::Vector &v);
   virtual void MultAdd(Vector &x, Vector &y);
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_BILIN_INTEG_MASS_HPP
