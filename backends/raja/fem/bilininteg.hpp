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

#ifndef MFEM_BACKENDS_RAJA_BILIN_INTEG_HPP
#define MFEM_BACKENDS_RAJA_BILIN_INTEG_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

namespace mfem
{

namespace raja
{

class RajaIntegrator
{
protected:
   SharedPtr<const Engine> engine;

   raja::RajaBilinearForm *bform;
   mfem::Mesh *mesh;

   RajaFiniteElementSpace *otrialFESpace;
   RajaFiniteElementSpace *otestFESpace;

   mfem::FiniteElementSpace *trialFESpace;
   mfem::FiniteElementSpace *testFESpace;

   RajaIntegratorType itype;

   const IntegrationRule *ir;
   bool hasTensorBasis;
   RajaDofQuadMaps *maps;
   RajaDofQuadMaps *mapsTranspose;

public:
   RajaIntegrator(const Engine &e);
   virtual ~RajaIntegrator();
   const Engine &RajaEngine() const { return *engine; }
   raja::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }
   virtual std::string GetName() = 0;
   RajaFiniteElementSpace& GetTrialRajaFESpace() const;
   RajaFiniteElementSpace& GetTestRajaFESpace() const;
   mfem::FiniteElementSpace& GetTrialFESpace() const;
   mfem::FiniteElementSpace& GetTestFESpace() const;
   void SetIntegrationRule(const mfem::IntegrationRule &ir_);
   const mfem::IntegrationRule& GetIntegrationRule() const;
   RajaDofQuadMaps* GetDofQuadMaps();
   void SetupMaps();
   virtual void SetupIntegrationRule() = 0;
   virtual void SetupIntegrator(RajaBilinearForm &bform_,
                                const RajaIntegratorType itype_);
   virtual void Setup() = 0;
   virtual void Assemble() = 0;
   /// This method works on E-vectors!
   virtual void MultAdd(Vector &x, Vector &y) = 0;
   virtual void MultTransposeAdd(Vector &x, Vector &y)
   {
      mfem_error("RajaIntegrator::MultTransposeAdd() is not overloaded!");
   }
   RajaGeometry *GetGeometry(const int flags = (RajaGeometry::Jacobian    |
                                                RajaGeometry::JacobianInv |
                                                RajaGeometry::JacobianDet));
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_BILIN_INTEG_HPP
