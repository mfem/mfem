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

#ifndef MFEM_BACKENDS_OCCA_BILIN_INTEG_HPP
#define MFEM_BACKENDS_OCCA_BILIN_INTEG_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#include "fespace.hpp"
#include "bilinearform.hpp"
#include "coefficient.hpp"

namespace mfem
{

namespace occa
{

class OccaGeometry
{
public:
   ::occa::array<double> meshNodes;
   ::occa::array<double> J, invJ, detJ;

   // byVDIM  -> [x y z x y z x y z]
   // byNodes -> [x x x y y y z z z]
   static const int Jacobian    = (1 << 0);
   static const int JacobianInv = (1 << 1);
   static const int JacobianDet = (1 << 2);

   static OccaGeometry Get(::occa::device device,
                           FiniteElementSpace &ofespace,
                           const IntegrationRule &ir,
                           const int flags = (Jacobian    |
                                              JacobianInv |
                                              JacobianDet));
};

class OccaDofQuadMaps
{
private:
   // Reuse dof-quad maps
   static std::map<std::string, OccaDofQuadMaps> AllDofQuadMaps;
   std::string hash;

public:
   // Local stiffness matrices (B and B^T operators)
   ::occa::array<double, ::occa::dynamic> dofToQuad, dofToQuadD; // B
   ::occa::array<double, ::occa::dynamic> quadToDof, quadToDofD; // B^T
   ::occa::array<double> quadWeights;

   OccaDofQuadMaps();
   OccaDofQuadMaps(const OccaDofQuadMaps &maps);
   OccaDofQuadMaps& operator = (const OccaDofQuadMaps &maps);

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
   static OccaDofQuadMaps& Get(::occa::device device,
                               const FiniteElementSpace &fespace,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static OccaDofQuadMaps& Get(::occa::device device,
                               const mfem::FiniteElement &fe,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static OccaDofQuadMaps& Get(::occa::device device,
                               const FiniteElementSpace &trialFESpace,
                               const FiniteElementSpace &testFESpace,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static OccaDofQuadMaps& Get(::occa::device device,
                               const mfem::FiniteElement &trialFE,
                               const mfem::FiniteElement &testFE,
                               const mfem::IntegrationRule &ir,
                               const bool transpose = false);

   static OccaDofQuadMaps& GetTensorMaps(::occa::device device,
                                         const mfem::FiniteElement &fe,
                                         const mfem::IntegrationRule &ir,
                                         const bool transpose = false);

   static OccaDofQuadMaps& GetTensorMaps(::occa::device device,
                                         const mfem::FiniteElement &trialFE,
                                         const mfem::FiniteElement &testFE,
                                         const mfem::IntegrationRule &ir,
                                         const bool transpose = false);

   static OccaDofQuadMaps GetD2QTensorMaps(::occa::device device,
                                           const mfem::FiniteElement &fe,
                                           const mfem::IntegrationRule &ir,
                                           const bool transpose = false);

   static OccaDofQuadMaps& GetSimplexMaps(::occa::device device,
                                          const mfem::FiniteElement &fe,
                                          const mfem::IntegrationRule &ir,
                                          const bool transpose = false);

   static OccaDofQuadMaps& GetSimplexMaps(::occa::device device,
                                          const mfem::FiniteElement &trialFE,
                                          const mfem::FiniteElement &testFE,
                                          const mfem::IntegrationRule &ir,
                                          const bool transpose = false);

   static OccaDofQuadMaps GetD2QSimplexMaps(::occa::device device,
                                            const mfem::FiniteElement &fe,
                                            const mfem::IntegrationRule &ir,
                                            const bool transpose = false);
};

//---[ Define Methods ]---------------
std::string stringWithDim(const std::string &s, const int dim);
int closestWarpBatch(const int multiple, const int maxSize);

void SetProperties(FiniteElementSpace &fespace,
                   const mfem::IntegrationRule &ir,
                   ::occa::properties &props);

void SetProperties(FiniteElementSpace &trialFESpace,
                   FiniteElementSpace &testFESpace,
                   const mfem::IntegrationRule &ir,
                   ::occa::properties &props);

void SetTensorProperties(FiniteElementSpace &fespace,
                         const mfem::IntegrationRule &ir,
                         ::occa::properties &props);

void SetTensorProperties(FiniteElementSpace &trialFESpace,
                         FiniteElementSpace &testFESpace,
                         const IntegrationRule &ir,
                         ::occa::properties &props);

void SetSimplexProperties(FiniteElementSpace &fespace,
                          const IntegrationRule &ir,
                          ::occa::properties &props);

void SetSimplexProperties(FiniteElementSpace &trialFESpace,
                          FiniteElementSpace &testFESpace,
                          const IntegrationRule &ir,
                          ::occa::properties &props);

//---[ Base Integrator ]--------------
class OccaIntegrator
{
protected:
   SharedPtr<const Engine> engine;

   OccaBilinearForm *bform;
   mfem::Mesh *mesh;

   FiniteElementSpace *otrialFESpace;
   FiniteElementSpace *otestFESpace;

   mfem::FiniteElementSpace *trialFESpace;
   mfem::FiniteElementSpace *testFESpace;

   ::occa::properties props;
   OccaIntegratorType itype;

   const IntegrationRule *ir;
   bool hasTensorBasis;
   OccaDofQuadMaps maps;
   OccaDofQuadMaps mapsTranspose;

public:
   OccaIntegrator(const Engine &e);
   virtual ~OccaIntegrator();

   const Engine &OccaEngine() const { return *engine; }

   ::occa::device GetDevice(int idx = 0) const
   { return engine->GetDevice(idx); }

   virtual std::string GetName() = 0;

   FiniteElementSpace& GetTrialOccaFESpace() const;
   FiniteElementSpace& GetTestOccaFESpace() const;

   mfem::FiniteElementSpace& GetTrialFESpace() const;
   mfem::FiniteElementSpace& GetTestFESpace() const;

   void SetIntegrationRule(const mfem::IntegrationRule &ir_);
   const mfem::IntegrationRule& GetIntegrationRule() const;

   OccaDofQuadMaps& GetDofQuadMaps();

   void SetupMaps();

   virtual void SetupIntegrationRule() = 0;

   virtual void SetupIntegrator(OccaBilinearForm &bform_,
                                const ::occa::properties &props_,
                                const OccaIntegratorType itype_);

   virtual void Setup() = 0;

   virtual void Assemble() = 0;
   /// This method works on E-vectors!
   virtual void MultAdd(Vector &x, Vector &y) = 0;

   virtual void MultTransposeAdd(Vector &x, Vector &y)
   {
      mfem_error("OccaIntegrator::MultTransposeAdd() is not overloaded!");
   }

   OccaGeometry GetGeometry(const int flags = (OccaGeometry::Jacobian    |
                                               OccaGeometry::JacobianInv |
                                               OccaGeometry::JacobianDet));

   ::occa::kernel GetAssembleKernel(const ::occa::properties &props);
   ::occa::kernel GetMultAddKernel(const ::occa::properties &props);

   ::occa::kernel GetKernel(const std::string &kernelName,
                            const ::occa::properties &props);
};
//====================================


//---[ Diffusion Integrator ]---------
class OccaDiffusionIntegrator : public OccaIntegrator
{
private:
   OccaCoefficient coeff;

   ::occa::kernel assembleKernel, multKernel;

   Vector assembledOperator;

public:
   OccaDiffusionIntegrator(const OccaCoefficient &coeff_);
   virtual ~OccaDiffusionIntegrator();

   virtual std::string GetName();

   virtual void SetupIntegrationRule();

   virtual void Setup();

   virtual void Assemble();
   virtual void MultAdd(Vector &x, Vector &y);
};
//====================================


//---[ Mass Integrator ]--------------
class OccaMassIntegrator : public OccaIntegrator
{
private:
   OccaCoefficient coeff;

   ::occa::kernel assembleKernel, multKernel;

   Vector assembledOperator;

public:
   OccaMassIntegrator(const OccaCoefficient &coeff_);
   virtual ~OccaMassIntegrator();

   virtual std::string GetName();

   virtual void SetupIntegrationRule();

   virtual void Setup();

   virtual void Assemble();
   void SetOperator(Vector &v);

   virtual void MultAdd(Vector &x, Vector &y);
};
//====================================

//---[ Vector Mass Integrator ]--------------
class OccaVectorMassIntegrator : public OccaIntegrator
{
private:
   OccaCoefficient coeff;

   ::occa::kernel assembleKernel, multKernel;

   Vector assembledOperator;

public:
   OccaVectorMassIntegrator(const OccaCoefficient &coeff_);
   virtual ~OccaVectorMassIntegrator();

   virtual std::string GetName();

   virtual void SetupIntegrationRule();

   virtual void Setup();

   virtual void Assemble();

   virtual void MultAdd(Vector &x, Vector &y);
};

} // namespace mfem::occa

} // namespace mfem

#endif // defined(MFEM_USE_BACKENDS) && defined(MFEM_USE_OCCA)

#endif // MFEM_BACKENDS_OCCA_BILIN_INTEG_HPP
