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

#include "../fem/fespace.hpp"
#include "../fem/bilinearform.hpp"
#include "../fem/coefficient.hpp"

namespace mfem
{

namespace raja
{
   
// ***************************************************************************
// * RajaGeometry
// ***************************************************************************
class RajaGeometry {
 public:
  ~RajaGeometry();
   raja::array<int> eMap;
   raja::array<double> meshNodes;
   raja::array<double> J, invJ, detJ;
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);
  static RajaGeometry* Get(FiniteElementSpace&,
                           const IntegrationRule&);
  static RajaGeometry* GetV(FiniteElementSpace&,
                           const IntegrationRule&,
                           const RajaVector&);
  static void ReorderByVDim(GridFunction& nodes);
  static void ReorderByNodes(GridFunction& nodes);
};
/*
class RajaGeometry
{
public:
   raja::array<double> meshNodes;
   raja::array<double> J, invJ, detJ;

   // byVDIM  -> [x y z x y z x y z]
   // byNodes -> [x x x y y y z z z]
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);

   static RajaGeometry Get(raja::device device,
                           FiniteElementSpace &ofespace,
                           const IntegrationRule &ir,
                           const int flags = (Jacobian    |
                                              JacobianInv |
                                              JacobianDet));
};
*/
// ***************************************************************************
// * RajaDofQuadMaps
// ***************************************************************************
class RajaDofQuadMaps {
 private:
  std::string hash;
 public:
   raja::array<double, false> dofToQuad, dofToQuadD; // B
   raja::array<double, false> quadToDof, quadToDofD; // B^T
   raja::array<double> quadWeights;
public:
  ~RajaDofQuadMaps();
  static void delRajaDofQuadMaps();
  static RajaDofQuadMaps* Get(const FiniteElementSpace&,
                              const mfem::IntegrationRule&,
                              const bool = false);
  static RajaDofQuadMaps* Get(const FiniteElementSpace&,
                              const FiniteElementSpace&,
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
/*class RajaDofQuadMaps
{
private:
   // Reuse dof-quad maps
   static std::map<std::string, RajaDofQuadMaps> AllDofQuadMaps;
   std::string hash;

public:
   // Local stiffness matrices (B and B^T operators)
   raja::array<double, raja::dynamic> dofToQuad, dofToQuadD; // B
   raja::array<double, raja::dynamic> quadToDof, quadToDofD; // B^T
   raja::array<double> quadWeights;

   RajaDofQuadMaps();
   RajaDofQuadMaps(const RajaDofQuadMaps &maps);
   RajaDofQuadMaps& operator = (const RajaDofQuadMaps &maps);

   // [[x y] [x y] [x y]]
   // [[x y z] [x y z] [x y z]]
   // mfem::GridFunction* mfem::Mesh::GetNodes() { return Nodes; }

   // mfem::FiniteElementSpace *Nodes->FESpace()
   // 25
   // 1D [x x x x x x]
   // 2D [x y x y x y]
   // GetVdim()
   // 3D ordering == byVDIM  -> [x y z x y z x y z x y z x y z x y z]
   //    ordering == byNODES -> [x x x x x x y y y y y y z z z z z z]
   static RajaDofQuadMaps& Get(raja::device device,
                               const FiniteElementSpace &fespace,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static RajaDofQuadMaps& Get(raja::device device,
                               const mfem::FiniteElement &fe,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static RajaDofQuadMaps& Get(raja::device device,
                               const FiniteElementSpace &trialFESpace,
                               const FiniteElementSpace &testFESpace,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static RajaDofQuadMaps& Get(raja::device device,
                               const mfem::FiniteElement &trialFE,
                               const mfem::FiniteElement &testFE,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static RajaDofQuadMaps& GetTensorMaps(raja::device device,
                                         const mfem::FiniteElement &fe,
                                         const mfem::IntegrationRule &ir,
                                         const bool transpose = false);

   static RajaDofQuadMaps& GetTensorMaps(raja::device device,
                                         const mfem::FiniteElement &trialFE,
                                         const mfem::FiniteElement &testFE,
                                         const mfem::IntegrationRule &ir,
                                         const bool transpose = false);

   static RajaDofQuadMaps GetD2QTensorMaps(raja::device device,
                                           const mfem::FiniteElement &fe,
                                           const mfem::IntegrationRule &ir,
                                           const bool transpose = false);

   static RajaDofQuadMaps& GetSimplexMaps(raja::device device,
                                          const mfem::FiniteElement &fe,
                                          const mfem::IntegrationRule &ir,
                                          const bool transpose = false);

   static RajaDofQuadMaps& GetSimplexMaps(raja::device device,
                                          const mfem::FiniteElement &trialFE,
                                          const mfem::FiniteElement &testFE,
                                          const mfem::IntegrationRule &ir,
                                          const bool transpose = false);

   static RajaDofQuadMaps GetD2QSimplexMaps(raja::device device,
                                            const mfem::FiniteElement &fe,
                                            const mfem::IntegrationRule &ir,
                                            const bool transpose = false);
};

//---[ Define Methods ]---------------
std::string stringWithDim(const std::string &s, const int dim);
int closestWarpBatch(const int multiple, const int maxSize);

void SetProperties(FiniteElementSpace &fespace,
                   const mfem::IntegrationRule &ir);

void SetProperties(FiniteElementSpace &trialFESpace,
                   FiniteElementSpace &testFESpace,
                   const mfem::IntegrationRule &ir);

void SetTensorProperties(FiniteElementSpace &fespace,
                         const mfem::IntegrationRule &ir);

void SetTensorProperties(FiniteElementSpace &trialFESpace,
                         FiniteElementSpace &testFESpace,
                         const IntegrationRule &ir);

void SetSimplexProperties(FiniteElementSpace &fespace,
                          const IntegrationRule &ir);

void SetSimplexProperties(FiniteElementSpace &trialFESpace,
                          FiniteElementSpace &testFESpace,
                          const IntegrationRule &ir);
*/
//---[ Base Integrator ]--------------
class RajaIntegrator
{
protected:
   SharedPtr<const Engine> engine;

   RajaBilinearForm *bform;
   mfem::Mesh *mesh;

   FiniteElementSpace *otrialFESpace;
   FiniteElementSpace *otestFESpace;

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
   FiniteElementSpace& GetTrialRajaFESpace() const;
   FiniteElementSpace& GetTestRajaFESpace() const;
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
