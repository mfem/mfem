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

// ***************************************************************************
// * RajaGeometry
// ***************************************************************************
class RajaGeometry
{
public:
   ~RajaGeometry();
   raja::array<int> eMap;
   raja::array<double> meshNodes;
   raja::array<double> J, invJ, detJ;
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);
   static RajaGeometry* Get(RajaParFiniteElementSpace&,
                            const IntegrationRule&);
   static RajaGeometry* GetV(RajaParFiniteElementSpace&,
                             const IntegrationRule&,
                             const RajaVector&);
   static void ReorderByVDim(GridFunction& nodes);
   static void ReorderByNodes(GridFunction& nodes);
};
   
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
   static RajaDofQuadMaps* Get(const RajaParFiniteElementSpace&,
                               const mfem::IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* Get(const RajaParFiniteElementSpace&,
                               const RajaParFiniteElementSpace&,
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
   
//---[ Base Integrator ]--------------
class RajaIntegrator
{
protected:
   SharedPtr<const Engine> engine;

   raja::RajaBilinearForm *bform;
   mfem::Mesh *mesh;

   RajaParFiniteElementSpace *otrialFESpace;
   RajaParFiniteElementSpace *otestFESpace;

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
   RajaParFiniteElementSpace& GetTrialRajaFESpace() const;
   RajaParFiniteElementSpace& GetTestRajaFESpace() const;
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
//====================================


//---[ Diffusion Integrator ]---------
class RajaDiffusionIntegrator : public RajaIntegrator
{
private:
   RajaCoefficient coeff;
   Vector assembledOperator;
public:
   RajaDiffusionIntegrator(const RajaCoefficient &coeff_);
   virtual ~RajaDiffusionIntegrator();
   virtual std::string GetName();
   virtual void SetupIntegrationRule();
   virtual void Setup();
   virtual void Assemble();
   virtual void MultAdd(Vector &x, Vector &y);
};
//====================================


//---[ Mass Integrator ]--------------
class RajaMassIntegrator : public RajaIntegrator
{
private:
   RajaCoefficient coeff;
   Vector assembledOperator;
public:
   RajaMassIntegrator(const RajaCoefficient &coeff_);
   virtual ~RajaMassIntegrator();
   virtual std::string GetName();
   virtual void SetupIntegrationRule();
   virtual void Setup();
   virtual void Assemble();
   void SetOperator(Vector &v);
   virtual void MultAdd(Vector &x, Vector &y);
};
//====================================

//---[ Vector Mass Integrator ]--------------
class RajaVectorMassIntegrator : public RajaIntegrator
{
private:
   RajaCoefficient coeff;
   Vector assembledOperator;
public:
   RajaVectorMassIntegrator(const RajaCoefficient &coeff_);
   virtual ~RajaVectorMassIntegrator();
   virtual std::string GetName();
   virtual void SetupIntegrationRule();
   virtual void Setup();
   virtual void Assemble();
   virtual void MultAdd(Vector &x, Vector &y);
};

} // namespace mfem::raja

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_RAJA)

#endif // MFEM_BACKENDS_RAJA_BILIN_INTEG_HPP
