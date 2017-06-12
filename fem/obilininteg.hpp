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

#include "../config/config.hpp"

#ifdef MFEM_USE_OCCA
#  ifndef MFEM_OCCA_BILININTEG
#  define MFEM_OCCA_BILININTEG

#include "obilinearform.hpp"
#include "ocoefficient.hpp"

namespace mfem {
  class OccaGeometry {
  public:
    occa::array<double> meshNodes;
    occa::array<double> J, invJ, detJ;

    // byVDIM  -> [x y z x y z x y z]
    // byNodes -> [x x x y y y z z z]
    static const int Jacobian    = (1 << 0);
    static const int JacobianInv = (1 << 1);
    static const int JacobianDet = (1 << 2);

    static OccaGeometry Get(occa::device device,
                            OccaFiniteElementSpace &ofespace,
                            const IntegrationRule &ir,
                            const int flags = (Jacobian    |
                                               JacobianInv |
                                               JacobianDet));
  };

  class OccaDofQuadMaps {
  private:
    // Reuse dof-quad maps
    static std::map<std::string, OccaDofQuadMaps> AllDofQuadMaps;
    std::string hash;

  public:
    // Local stiffness matrices (B and B^T operators)
    occa::array<double, occa::dynamic> dofToQuad, dofToQuadD; // B
    occa::array<double, occa::dynamic> quadToDof, quadToDofD; // B^T
    occa::array<double> quadWeights;

    OccaDofQuadMaps();
    OccaDofQuadMaps(const OccaDofQuadMaps &maps);
    OccaDofQuadMaps& operator = (const OccaDofQuadMaps &maps);

    // [[x y] [x y] [x y]]
    // [[x y z] [x y z] [x y z]]
    // GridFunction* Mesh::GetNodes() { return Nodes; }

    // FiniteElementSpace *Nodes->FESpace()
    // 25
    // 1D [x x x x x x]
    // 2D [x y x y x y]
    // GetVdim()
    // 3D ordering == byVDIM  -> [x y z x y z x y z x y z x y z x y z]
    //    ordering == byNODES -> [x x x x x x y y y y y y z z z z z z]
    static OccaDofQuadMaps& GetTensorMaps(occa::device device,
                                          const FiniteElement &fe,
                                          const TensorBasisElement &tfe,
                                          const IntegrationRule &ir,
                                          const bool transpose = false);

    static OccaDofQuadMaps& GetTensorMaps(occa::device device,
                                          const FiniteElement &trialFE,
                                          const FiniteElement &testFE,
                                          const TensorBasisElement &trialTFE,
                                          const TensorBasisElement &testTFE,
                                          const IntegrationRule &ir,
                                          const bool transpose = false);

    static OccaDofQuadMaps GetD2QTensorMaps(occa::device device,
                                            const FiniteElement &fe,
                                            const TensorBasisElement &tfe,
                                            const IntegrationRule &ir,
                                            const bool transpose = false);

    static OccaDofQuadMaps& GetSimplexMaps(occa::device device,
                                           const FiniteElement &fe,
                                           const IntegrationRule &ir,
                                           const bool transpose = false);

    static OccaDofQuadMaps& GetSimplexMaps(occa::device device,
                                           const FiniteElement &trialFE,
                                           const FiniteElement &testFE,
                                           const IntegrationRule &ir,
                                           const bool transpose = false);

    static OccaDofQuadMaps GetD2QSimplexMaps(occa::device device,
                                             const FiniteElement &fe,
                                             const IntegrationRule &ir,
                                             const bool transpose = false);
  };

  //---[ Define Methods ]---------------
  std::string stringWithDim(const std::string &s, const int dim);
  int closestWarpBatch(const int multiple, const int maxSize);

  void setTensorProperties(const FiniteElement &fe,
                           const IntegrationRule &ir,
                           occa::properties &props);

  void setTensorProperties(const FiniteElement &fe,
                           const FiniteElement &fe2,
                           const IntegrationRule &ir,
                           occa::properties &props);

  void setSimplexProperties(const FiniteElement &fe,
                            const IntegrationRule &ir,
                            occa::properties &props);

  void setSimplexProperties(const FiniteElement &fe,
                           const FiniteElement &fe2,
                            const IntegrationRule &ir,
                            occa::properties &props);

  //---[ Base Integrator ]--------------
  class OccaIntegrator {
  protected:
    occa::device device;

    OccaBilinearForm *bform;
    Mesh *mesh;

    OccaFiniteElementSpace *otrialFespace;
    OccaFiniteElementSpace *otestFespace;

    FiniteElementSpace *trialFespace;
    FiniteElementSpace *testFespace;

    occa::properties props;
    OccaIntegratorType itype;

    const IntegrationRule *ir;
    bool hasTensorBasis;
    OccaDofQuadMaps maps;
    OccaDofQuadMaps mapsTranspose;

  public:
    OccaIntegrator();
    virtual ~OccaIntegrator();

    occa::device GetDevice();

    virtual std::string GetName() = 0;

    FiniteElementSpace& GetTrialFESpace();
    FiniteElementSpace& GetTestFESpace();

    const IntegrationRule& GetIntegrationRule();
    OccaDofQuadMaps& GetDofQuadMaps();

    void SetupMaps();
    void SetupProperties(occa::properties &props);

    virtual void SetupIntegrationRule() = 0;

    virtual void SetupIntegrator(OccaBilinearForm &bform_,
                                 const occa::properties &props_,
                                 const OccaIntegratorType itype_);

    virtual void Setup() = 0;

    virtual void Assemble() = 0;
    virtual void MultAdd(OccaVector &x, OccaVector &y) = 0;

    virtual void MultTransposeAdd(OccaVector &x, OccaVector &y)
    {
       mfem_error("OccaIntegrator::MultTransposeAdd() is not overloaded!");
    }

    OccaGeometry GetGeometry(const int flags = (OccaGeometry::Jacobian    |
                                                OccaGeometry::JacobianInv |
                                                OccaGeometry::JacobianDet));

    occa::kernel GetAssembleKernel(const occa::properties &props);
    occa::kernel GetMultAddKernel(const occa::properties &props);

    occa::kernel GetKernel(const std::string &kernelName,
                           const occa::properties &props);
  };
  //====================================

  //---[ Diffusion Integrator ]---------
  class OccaDiffusionIntegrator : public OccaIntegrator {
  private:
    OccaCoefficient coeff;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> jacobian, assembledOperator;

  public:
    OccaDiffusionIntegrator(const OccaCoefficient &coeff_);
    virtual ~OccaDiffusionIntegrator();

    virtual std::string GetName();

    virtual void SetupIntegrationRule();

    virtual void Setup();

    virtual void Assemble();
    virtual void MultAdd(OccaVector &x, OccaVector &y);
  };
  //====================================

  //---[ Mass Integrator ]--------------
  class OccaMassIntegrator : public OccaIntegrator {
  private:
    OccaCoefficient coeff;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> jacobian, assembledOperator;

  public:
    OccaMassIntegrator(const OccaCoefficient &coeff_);
    virtual ~OccaMassIntegrator();

    virtual std::string GetName();

    virtual void SetupIntegrationRule();

    virtual void Setup();

    virtual void Assemble();

    virtual void MultAdd(OccaVector &x, OccaVector &y);
  };
  //====================================

  //---[ Vector Mass Integrator ]--------------
  class OccaVectorMassIntegrator : public OccaIntegrator {
  private:
    OccaCoefficient coeff;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> jacobian, assembledOperator;

  public:
    OccaVectorMassIntegrator(const OccaCoefficient &coeff_);
    virtual ~OccaVectorMassIntegrator();

    virtual std::string GetName();

    virtual void SetupIntegrationRule();

    virtual void Setup();

    virtual void Assemble();

    virtual void MultAdd(OccaVector &x, OccaVector &y);
  };
  //====================================
}

#  endif
#endif
