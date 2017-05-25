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
                            Mesh &mesh,
                            const IntegrationRule &ir,
                            const int flags = (Jacobian    |
                                               JacobianInv |
                                               JacobianDet));
  };

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
                                          const TensorBasisElement &fe,
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

  //---[ Base Integrator ]--------------
  class OccaIntegrator {
  protected:
    occa::device device;

    OccaBilinearForm *bform;
    FiniteElementSpace *fespace;
    Mesh *mesh;
    occa::properties props;
    OccaIntegratorType itype;

    const IntegrationRule *ir;
    bool hasTensorBasis;
    OccaDofQuadMaps maps;

  public:
    OccaIntegrator();
    virtual ~OccaIntegrator();

    void SetMaps(const IntegrationRule &ir_);
    void SetProperties(occa::properties &props);

    occa::device GetDevice();

    virtual std::string GetName() = 0;

    FiniteElementSpace& GetFESpace();
    const IntegrationRule& GetIntegrationRule();
    OccaDofQuadMaps& GetDofQuadMaps();

    virtual void SetupIntegrator(OccaBilinearForm &bform_,
                                 const occa::properties &props_,
                                 const OccaIntegratorType itype_);

    virtual void Setup() = 0;

    virtual void Assemble() = 0;
    virtual void Mult(OccaVector &x) = 0;

    occa::kernel GetAssembleKernel(const occa::properties &props);
    occa::kernel GetMultKernel(const occa::properties &props);

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

    virtual void Setup();

    virtual void Assemble();
    virtual void Mult(OccaVector &x);
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

    virtual void Setup();

    virtual void Assemble();
    virtual void Mult(OccaVector &x);
  };
  //====================================
}

#  endif
#endif
