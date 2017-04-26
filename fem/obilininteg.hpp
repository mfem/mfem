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

namespace mfem {
  class OccaDofQuadMaps {
  private:
    // Reuse dof-quad maps
    static std::map<occa::hash_t, OccaDofQuadMaps> AllDofQuadMaps;
    occa::hash_t hash;

  public:
    // Local stiffness matrices (B and B^T operators)
    occa::array<double> dofToQuad, dofToQuadD; // B
    occa::array<double> quadToDof, quadToDofD; // B^T
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
                                          const H1_TensorBasisElement &fe,
                                          const IntegrationRule &ir);

    static OccaDofQuadMaps& GetSimplexMaps(occa::device device,
                                           const FiniteElement &fe,
                                           const IntegrationRule &ir);
  };

  //---[ Define Methods ]---------------
  std::string stringWithDim(const std::string &s, const int dim);
  int closestWarpBatch(const int multiple, const int maxSize);

  void setTensorProperties(const FiniteElement &fe,
                           const IntegrationRule &ir,
                           occa::properties &props);

  void setSimplexProperties(const FiniteElement &fe,
                            const IntegrationRule &ir,
                            occa::properties &props);

  occa::array<double> getJacobian(occa::device device,
                                  FiniteElementSpace *fespace,
                                  const IntegrationRule &ir);

  void getJacobianData(occa::device device,
                       FiniteElementSpace *fespace,
                       const IntegrationRule &ir,
                       occa::array<double> &J,
                       occa::array<double> &Jinv,
                       occa::array<double> &Jdet);  

  //---[ Base Integrator ]--------------
  class OccaIntegrator {
  protected:
    occa::device device;

    BilinearFormIntegrator *integrator;
    FiniteElementSpace *fespace;
    occa::properties props;
    OccaIntegratorType itype;

  public:
    OccaIntegrator();
    virtual ~OccaIntegrator();

    OccaIntegrator* CreateInstance(occa::device device_,
                                   BilinearFormIntegrator *integrator_,
                                   FiniteElementSpace *fespace_,
                                   const occa::properties &props_,
                                   const OccaIntegratorType itype_);

    virtual OccaIntegrator* CreateInstance() = 0;

    virtual std::string GetName();

    virtual void Setup();
    virtual void Assemble() = 0;
    virtual void Mult(OccaVector &x) = 0;

    void SetupCoefficient(const Coefficient *coeff,
                          occa::properties &kernelProps);

    occa::kernel GetAssembleKernel(const occa::properties &props);
    occa::kernel GetMultKernel(const occa::properties &props);

    occa::kernel GetKernel(const std::string &kernelName,
                           const occa::properties &props);
  };
  //====================================

  //---[ Diffusion Integrator ]---------
  class OccaDiffusionIntegrator : public OccaIntegrator {
  private:
    OccaDofQuadMaps maps;
    const Coefficient *coeff;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> jacobian, assembledOperator;

  public:
    OccaDiffusionIntegrator();
    virtual ~OccaDiffusionIntegrator();

    virtual OccaIntegrator* CreateInstance();

    virtual void Setup();

    virtual void Assemble();
    virtual void Mult(OccaVector &x);
  };
  //====================================

  //---[ Mass Integrator ]--------------
  class OccaMassIntegrator : public OccaIntegrator {
  private:
    OccaDofQuadMaps maps;
    const Coefficient *coeff;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> jacobian, assembledOperator;

  public:
    OccaMassIntegrator();
    virtual ~OccaMassIntegrator();

    virtual OccaIntegrator* CreateInstance();

    virtual void Setup();

    virtual void Assemble();
    virtual void Mult(OccaVector &x);
  };
  //====================================
}

#  endif
#endif
