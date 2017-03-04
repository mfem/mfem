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

#include <cmath>

#include "obilininteg.hpp"

namespace mfem {
  std::map<occa::hash_t, OccaDofQuadMaps> OccaDofQuadMaps::AllDofQuadMaps;

  OccaDofQuadMaps::OccaDofQuadMaps() :
    hash() {}

  OccaDofQuadMaps::OccaDofQuadMaps(const OccaDofQuadMaps &maps) {
    *this = maps;
  }

  OccaDofQuadMaps& OccaDofQuadMaps::operator = (const OccaDofQuadMaps &maps) {
    hash = maps.hash;
    dofToQuad  = maps.dofToQuad;
    dofToQuadD = maps.dofToQuadD;
    quadToDof  = maps.quadToDof;
    quadToDofD = maps.quadToDofD;
    return *this;
  }

  OccaDofQuadMaps& OccaDofQuadMaps::Get(occa::device device,
                                        const H1_TensorBasisElement &el,
                                        const IntegrationRule &ir) {

    occa::hash_t hash = (occa::hash(device)
                         ^ ("BasisType" + occa::toString(el.GetBasisType()))
                         ^ ("Order" + occa::toString(el.GetOrder()))
                         ^ ("Quad" + occa::toString(ir.GetNPoints())));

    // If we've already made the dof-quad maps, reuse them
    OccaDofQuadMaps &maps = AllDofQuadMaps[hash];
    if (maps.hash.isInitialized()) {
      return maps;
    }

    // Create the dof-quad maps
    maps.hash = hash;

    const Poly_1D::Basis &basis = el.GetBasis();
    const int order = el.GetOrder();
    const int dofs = order + 1;

    // Create the dof -> quadrature point map
    // [MISSING] Use the ir rule
    H1_SegmentElement se(dofs - 1, el.GetBasisType());
    const IntegrationRule &ir2 = IntRules.Get(Geometry::SEGMENT, 2*dofs);
    const int quadPoints = ir2.GetNPoints();

    // Initialize the dof -> quad mapping
    const int d2qEntries = quadPoints * dofs;
    double *dofToQuadData = new double[d2qEntries];
    double *dofToQuadDData = new double[d2qEntries];
    maps.dofToQuad.allocate(device, d2qEntries);
    maps.dofToQuadD.allocate(device, d2qEntries);

    for (int q = 0; q < quadPoints; ++q) {
      mfem::Vector d2q(dofToQuadData + q*dofs, dofs);
      mfem::Vector d2qD(dofToQuadDData + q*dofs, dofs);
      basis.Eval(ir2.IntPoint(q).x, d2q, d2qD);
    }

    occa::memcpy(maps.dofToQuad.memory(), dofToQuadData);
    occa::memcpy(maps.dofToQuadD.memory(), dofToQuadDData);

    // Create the quadrature -> dof point map
    double *quadToDofData = new double[d2qEntries];
    double *quadToDofDData = new double[d2qEntries];
    maps.quadToDof.allocate(device, d2qEntries);
    maps.quadToDofD.allocate(device, d2qEntries);

    for (int q = 0; q < quadPoints; ++q) {
      for (int p = 0; p < dofs; ++p) {
        quadToDofData[q + p*quadPoints] = dofToQuadData[p + q*dofs];
        quadToDofDData[q + p*quadPoints] = dofToQuadDData[p + q*dofs];
      }
    }

    occa::memcpy(maps.quadToDof.memory(), quadToDofData);
    occa::memcpy(maps.quadToDofD.memory(), quadToDofDData);

    delete [] dofToQuadData;
    delete [] dofToQuadDData;
    delete [] quadToDofData;
    delete [] quadToDofDData;

    return maps;
  }

  //---[ Base Integrator ]--------------
  OccaIntegrator::OccaIntegrator(OccaBilinearForm &bilinearForm_) :
    bilinearForm(bilinearForm_) {}

  OccaIntegrator::~OccaIntegrator() {}

  OccaIntegrator* OccaIntegrator::CreateInstance(BilinearFormIntegrator &integrator_,
                                                 const occa::properties &props_,
                                                 const OccaIntegratorType itype_) {
    OccaIntegrator *newIntegrator = CreateInstance();

    newIntegrator->device = bilinearForm.getDevice();

    newIntegrator->integrator = &integrator_;
    newIntegrator->props = props_;
    newIntegrator->itype = itype_;

    newIntegrator->Setup();

    return newIntegrator;
  }

  void OccaIntegrator::Setup() {}

  occa::kernel OccaIntegrator::GetAssembleKernel(const occa::properties &props) {
    return GetKernel("Assemble", props);
  }

  occa::kernel OccaIntegrator::GetMultKernel(const occa::properties &props) {
    return GetKernel("Mult", props);
  }

  occa::kernel OccaIntegrator::GetKernel(const std::string &kernelName,
                                         const occa::properties &props) {
    // Get kernel name
    const std::string filename = integrator->Name() + ".okl";
    // Get kernel suffix
    std::string dimSuffix;
    dimSuffix += '0' + (char) bilinearForm.GetDim();
    dimSuffix += 'D';

    return device.buildKernel("occa://mfem/fem/" + filename,
                              kernelName + dimSuffix,
                              props);
  }
  //====================================

  //---[ Diffusion Integrator ]---------
  OccaDiffusionIntegrator::OccaDiffusionIntegrator(OccaBilinearForm &bilinearForm_) :
    OccaIntegrator(bilinearForm_) {}

  OccaDiffusionIntegrator::~OccaDiffusionIntegrator() {
    assembleKernel.free();
    multKernel.free();

    coefficients.free();
    jacobian.free();
    assembledOperator.free();
  }

  OccaIntegrator* OccaDiffusionIntegrator::CreateInstance() {
    return new OccaDiffusionIntegrator(bilinearForm);
  }

  void OccaDiffusionIntegrator::Setup() {
    // Assumption that all FiniteElements are H1_TensorBasisElement is checked
    //   inside OccaBilinearForm
    const FiniteElement &fe = bilinearForm.GetFE(0);
    const H1_TensorBasisElement &el = dynamic_cast<const H1_TensorBasisElement&>(fe);

    DiffusionIntegrator &integ = (DiffusionIntegrator&) *integrator;
    const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);

    maps = OccaDofQuadMaps::Get(device, el, ir);

    // Get coefficient from integrator
    // [MISSING] Hard-coded to ConstantCoefficient for now
    const ConstantCoefficient* coeff =
      dynamic_cast<const ConstantCoefficient*>(integ.GetCoefficient());

    if (!coeff) {
      mfem_error("OccaDiffusionIntegrator can only handle ConstantCoefficients");
    }

    hasConstantCoefficient = true;

    // Setup kernel compiled defines
    const int dims = bilinearForm.GetDim();
    const int elements = bilinearForm.GetNE();
    const int quadraturePoints = ir.GetNPoints();

    const int dofs1D = el.GetOrder() + 1;
    int dofsND = dofs1D;

    occa::properties kernelProps = props;
    for (int d = 0; d < dims; ++d) {
      std::string define = "defines/NUM_DOFS_";
      define += ('1' + d);
      define += 'D';
      kernelProps[define] = dofsND;
      dofsND *= dofs1D;
    }

    // [MISSING] Use the ir rule
    const IntegrationRule &ir2 = IntRules.Get(Geometry::SEGMENT, 2*dofs1D);
    const int quad1D = ir2.GetNPoints();
    int quadND = quad1D;

    for (int d = 0; d < dims; ++d) {
      std::string define = "defines/NUM_QUAD_";
      define += ('1' + d);
      define += 'D';
      kernelProps[define] = quadND;
      quadND *= quad1D;
    }

    // [MISSING] Logic based on quadrature points
    // If quadrature is small, we can process assembly in batches
    kernelProps["defines/QUAD_BATCH"] = 1;

    // Redundant, but for future codes
    if (coeff) {
      kernelProps["defines/CONST_COEFF"] = coeff->constant;
    }

    const int dims2 = dims * dims;
    const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

    assembledOperator.allocate(device,
                               elements * quadraturePoints * symmDims);

    const int jacobianEntries = elements * quadraturePoints * dims2;
    jacobian.allocate(device, jacobianEntries);
    double *jacobianData = new double[jacobianEntries];
    double *eJacobian = jacobianData;

    Mesh &mesh = bilinearForm.GetMesh();
    for (int e = 0; e < elements; ++e) {
      ElementTransformation &trans = *(mesh.GetElementTransformation(e));
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const IntegrationPoint &ip = ir.IntPoint(q);
        trans.SetIntPoint(&ip);
        const DenseMatrix &qJ = trans.Jacobian();
        for (int j = 0; j < dims; ++j) {
          for (int i = 0; i < dims; ++i) {
            // Column-major -> Row-major
            eJacobian[j + i*dims] = qJ(i,j);
          }
        }
        eJacobian += dims2;
      }
    }

    occa::memcpy(jacobian.memory(), jacobianData);
    delete [] jacobianData;

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel = GetMultKernel(kernelProps);
  }

  void OccaDiffusionIntegrator::Assemble() {
    if (hasConstantCoefficient) {
      // Dummy coefficient since we're defining it at compile time
      assembleKernel((int) bilinearForm.GetNE(),
                     jacobian.memory(),
                     (double) 0,
                     assembledOperator.memory());
    } else {
      assembleKernel((int) bilinearForm.GetNE(),
                     jacobian.memory(),
                     coefficients.memory(),
                     assembledOperator.memory());
    }
  }

  void OccaDiffusionIntegrator::Mult(OccaVector &x) {
    multKernel((int) bilinearForm.GetNE(),
               maps.dofToQuad.memory(),
               maps.dofToQuadD.memory(),
               maps.quadToDof.memory(),
               maps.quadToDofD.memory(),
               assembledOperator.memory(),
               x.GetData());
  }
  //====================================
}

#endif