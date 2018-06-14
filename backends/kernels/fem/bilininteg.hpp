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

#ifndef MFEM_BACKENDS_KERNELS_BILIN_INTEG_HPP
#define MFEM_BACKENDS_KERNELS_BILIN_INTEG_HPP

#include "../../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

namespace mfem
{

namespace kernels
{

class KernelsIntegrator
{
protected:
   SharedPtr<const Engine> engine;

   kernels::KernelsBilinearForm *bform;
   mfem::Mesh *mesh;

   KernelsFiniteElementSpace *rtrialFESpace;
   KernelsFiniteElementSpace *rtestFESpace;

   mfem::FiniteElementSpace *trialFESpace;
   mfem::FiniteElementSpace *testFESpace;

   KernelsIntegratorType itype;

   const IntegrationRule *ir;
   bool hasTensorBasis;
   kernels::KernelsDofQuadMaps *maps;
   KernelsDofQuadMaps *mapsTranspose;

public:
   KernelsIntegrator(const kernels::Engine &e);
   virtual ~KernelsIntegrator();
   const Engine &KernelsEngine() const { assert(engine); return *engine; }
   kernels::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }
   virtual std::string GetName() = 0;
   KernelsFiniteElementSpace& GetTrialKernelsFESpace() const;
   KernelsFiniteElementSpace& GetTestKernelsFESpace() const;
   mfem::FiniteElementSpace& GetTrialFESpace() const;
   mfem::FiniteElementSpace& GetTestFESpace() const;
   void SetIntegrationRule(const mfem::IntegrationRule &ir_);
   const mfem::IntegrationRule& GetIntegrationRule() const;
   KernelsDofQuadMaps* GetDofQuadMaps();
   void SetupMaps();
   virtual void SetupIntegrationRule() = 0;
   virtual void SetupIntegrator(KernelsBilinearForm &bform_,
                                const KernelsIntegratorType itype_);
   virtual void Setup() = 0;
   virtual void Assemble() = 0;
   /// This method works on E-vectors!
   virtual void MultAdd(Vector &x, Vector &y) = 0;
   virtual void MultTransposeAdd(Vector &x, Vector &y)
   {
      mfem_error("KernelsIntegrator::MultTransposeAdd() is not overloaded!");
   }
   KernelsGeometry *GetGeometry(const int flags = (KernelsGeometry::Jacobian    |
                                                   KernelsGeometry::JacobianInv |
                                                   KernelsGeometry::JacobianDet));
};

} // namespace mfem::kernels

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_KERNELS)

#endif // MFEM_BACKENDS_KERNELS_BILIN_INTEG_HPP
