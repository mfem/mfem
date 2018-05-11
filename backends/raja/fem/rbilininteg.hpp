// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#ifndef LAGHOS_RAJA_BILININTEG
#define LAGHOS_RAJA_BILININTEG

namespace mfem
{

// ***************************************************************************
// * RajaGeometry
// ***************************************************************************
class RajaGeometry
{
public:
   ~RajaGeometry();
   RajaArray<int> eMap;
   RajaArray<double> meshNodes;
   RajaArray<double> J, invJ, detJ;
   static RajaGeometry* Get(RajaFiniteElementSpace&,
                            const IntegrationRule&);
   static RajaGeometry* Get(RajaFiniteElementSpace&,
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
   RajaArray<double, false> dofToQuad, dofToQuadD; // B
   RajaArray<double, false> quadToDof, quadToDofD; // B^T
   RajaArray<double> quadWeights;
public:
   ~RajaDofQuadMaps();
   static void delRajaDofQuadMaps();
   static RajaDofQuadMaps* Get(const RajaFiniteElementSpace&,
                               const IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* Get(const RajaFiniteElementSpace&,
                               const RajaFiniteElementSpace&,
                               const IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* Get(const FiniteElement&,
                               const FiniteElement&,
                               const IntegrationRule&,
                               const bool = false);
   static RajaDofQuadMaps* GetTensorMaps(const FiniteElement&,
                                         const FiniteElement&,
                                         const IntegrationRule&,
                                         const bool = false);
   static RajaDofQuadMaps* GetD2QTensorMaps(const FiniteElement&,
                                            const IntegrationRule&,
                                            const bool = false);
   static RajaDofQuadMaps* GetSimplexMaps(const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
   static RajaDofQuadMaps* GetSimplexMaps(const FiniteElement&,
                                          const FiniteElement&,
                                          const IntegrationRule&,
                                          const bool = false);
   static RajaDofQuadMaps* GetD2QSimplexMaps(const FiniteElement&,
                                             const IntegrationRule&,
                                             const bool = false);
};

// ***************************************************************************
// * Base Integrator
// ***************************************************************************
class RajaIntegrator
{
protected:
   Mesh* mesh = NULL;
   RajaFiniteElementSpace* trialFESpace = NULL;
   RajaFiniteElementSpace* testFESpace = NULL;
   RajaIntegratorType itype;
   const IntegrationRule* ir = NULL;
   RajaDofQuadMaps* maps;
   RajaDofQuadMaps* mapsTranspose;
private:
public:
   virtual std::string GetName() = 0;
   void SetIntegrationRule(const IntegrationRule& ir_);
   const IntegrationRule& GetIntegrationRule() const;
   virtual void SetupIntegrationRule() = 0;
   virtual void SetupIntegrator(RajaBilinearForm& bform_,
                                const RajaIntegratorType itype_);
   virtual void Setup() = 0;
   virtual void Assemble() = 0;
   virtual void MultAdd(RajaVector& x, RajaVector& y) = 0;
   virtual void MultTransposeAdd(RajaVector&, RajaVector&) {assert(false);}
   RajaGeometry* GetGeometry();
};

// ***************************************************************************
// * Mass Integrator
// ***************************************************************************
class RajaMassIntegrator : public RajaIntegrator
{
private:
   RajaVector op;
public:
   RajaMassIntegrator() {}
   virtual ~RajaMassIntegrator() {}
   virtual std::string GetName() {return "MassIntegrator";}
   virtual void SetupIntegrationRule();
   virtual void Setup() {}
   virtual void Assemble();
   void SetOperator(RajaVector& v);
   virtual void MultAdd(RajaVector& x, RajaVector& y);
};

} // mfem

#endif // LAGHOS_RAJA_BILININTEG
