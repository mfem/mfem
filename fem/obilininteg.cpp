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
    dofToQuad   = maps.dofToQuad;
    dofToQuadD  = maps.dofToQuadD;
    quadToDof   = maps.quadToDof;
    quadToDofD  = maps.quadToDofD;
    quadWeights = maps.quadWeights;
    return *this;
  }

  OccaDofQuadMaps& OccaDofQuadMaps::GetTensorMaps(occa::device device,
                                                  const OccaBilinearForm &bilinearForm,
                                                  const H1_TensorBasisElement &el,
                                                  const IntegrationRule &ir) {

    occa::hash_t hash = (occa::hash(device)
                         ^ "Tensor Element"
                         ^ ("BasisType: " + occa::toString(el.GetBasisType()))
                         ^ ("Order: " + occa::toString(el.GetOrder()))
                         ^ ("Quad: " + occa::toString(ir.GetNPoints())));

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
    const int dims = bilinearForm.GetDim();

    // Create the dof -> quadrature point map
    const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());
    const int quadPoints = ir1D.GetNPoints();
    const int quadPoints2D = quadPoints*quadPoints;
    const int quadPoints3D = quadPoints2D*quadPoints;
    const int quadPointsND = ((dims == 1) ? quadPoints :
                              ((dims == 2) ? quadPoints2D : quadPoints3D));

    // Initialize the dof -> quad mapping
    const int d2qEntries = quadPoints * dofs;
    double *dofToQuadData = new double[d2qEntries];
    double *dofToQuadDData = new double[d2qEntries];
    // Initialize quad weights
    double *quadWeights1DData = new double[quadPoints];
    double *quadWeightsData = new double[quadPointsND];

    for (int q = 0; q < quadPoints; ++q) {
      mfem::Vector d2q(dofs);
      mfem::Vector d2qD(dofs);
      const IntegrationPoint &ip = ir1D.IntPoint(q);
      basis.Eval(ip.x, d2q, d2qD);
      quadWeights1DData[q] = ip.weight;
      for (int d = 0; d < dofs; ++d) {
        dofToQuadData[q + d*quadPoints]  = d2q[d];
        dofToQuadDData[q + d*quadPoints] = d2qD[d];
      }
    }

    for (int q = 0; q < quadPointsND; ++q) {
      const int qx = q % quadPoints;
      const int qz = q / quadPoints2D;
      const int qy = (q - qz*quadPoints2D) / quadPoints;
      quadWeightsData[q] = quadWeights1DData[qx];
      if (dims > 1) {
        quadWeightsData[q] *= quadWeights1DData[qy];
      }
      if(dims > 2) {
        quadWeightsData[q] *= quadWeights1DData[qz];
      }
    }

    maps.dofToQuad = device.malloc(d2qEntries * sizeof(double),
                                   dofToQuadData);
    maps.dofToQuadD = device.malloc(d2qEntries * sizeof(double),
                                    dofToQuadDData);
    maps.quadWeights = device.malloc(quadPointsND * sizeof(double),
                                     quadWeightsData);

    // Create the quadrature -> dof point map
    double *quadToDofData = new double[d2qEntries];
    double *quadToDofDData = new double[d2qEntries];

    for (int q = 0; q < quadPoints; ++q) {
      for (int d = 0; d < dofs; ++d) {
        quadToDofData[d + q*dofs] = dofToQuadData[q + d*quadPoints];
        quadToDofDData[d + q*dofs] = dofToQuadDData[q + d*quadPoints];
      }
    }

    maps.quadToDof = device.malloc(d2qEntries * sizeof(double),
                                   quadToDofData);
    maps.quadToDofD = device.malloc(d2qEntries * sizeof(double),
                                    quadToDofDData);

    delete [] dofToQuadData;
    delete [] dofToQuadDData;
    delete [] quadToDofData;
    delete [] quadToDofDData;

    delete [] quadWeights1DData;
    delete [] quadWeightsData;

    return maps;
  }

  //---[ Integrator Defines ]-----------
  std::string stringWithDim(const std::string &s, const int dim) {
    std::string ret = s;
    ret += ('0' + (char) dim);
    ret += 'D';
    return ret;
  }

  int closestWarpBatch(const int multiple, const int maxSize) {
    int batch = (32 / multiple);
    int minDiff = 32 - (multiple * batch);
    for (int i = 64; i <= maxSize; i += 32) {
      const int newDiff = i - (multiple * (i / multiple));
      if (newDiff < minDiff) {
        batch = (i / multiple);
        minDiff = newDiff;
      }
    }
    return batch;
  }

  void setTensorProperties(const FiniteElement &fe,
                           const IntegrationRule &ir,
                           occa::properties &props) {

    const IntegrationRule &ir1D = IntRules.Get(Geometry::SEGMENT, ir.GetOrder());

    const int dofs1D = fe.GetOrder() + 1;
    const int quad1D = ir1D.GetNPoints();
    int dofsND = dofs1D;
    int quadND = quad1D;

    for (int d = 1; d <= 3; ++d) {
      if (d > 1) {
        dofsND *= dofs1D;
        quadND *= quad1D;
      }
      props["defines"][stringWithDim("NUM_DOFS_", d)] = dofsND;
      props["defines"][stringWithDim("NUM_QUAD_", d)] = quadND;
    }

    // 1D Defines
    const int m1InnerBatch = 32 * ((quad1D + 31) / 32);
    props["defines/A1_ELEMENT_BATCH"]       = closestWarpBatch(quad1D, 2048);
    props["defines/M1_OUTER_ELEMENT_BATCH"] = closestWarpBatch(m1InnerBatch, 2048);
    props["defines/M1_INNER_ELEMENT_BATCH"] = m1InnerBatch;

    // 2D Defines
    props["defines/A2_ELEMENT_BATCH"] = 1;
    props["defines/A2_QUAD_BATCH"]    = 1;

    // 3D Defines
    const int a3QuadBatch = closestWarpBatch(quadND, 2048);
    props["defines/A3_ELEMENT_BATCH"] = closestWarpBatch(a3QuadBatch, 2048);
    props["defines/A3_QUAD_BATCH"]    = a3QuadBatch;
  }

  occa::memory getJacobian(occa::device device,
                           const OccaBilinearForm &bilinearForm,
                           const IntegrationRule &ir) {
    const int dims = bilinearForm.GetDim();
    const int elements = bilinearForm.GetNE();
    const int quadraturePoints = ir.GetNPoints();

    const int dims2 = dims * dims;
    const int jacobianEntries = elements * quadraturePoints * dims2;

    double *jacobianData = new double[jacobianEntries];
    double *eJacobian = jacobianData;

    Mesh &mesh = bilinearForm.GetMesh();
    for (int e = 0; e < elements; ++e) {
      ElementTransformation &trans = *(mesh.GetElementTransformation(e));
      for (int q = 0; q < quadraturePoints; ++q) {
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

    occa::memory jacobian = device.malloc(jacobianEntries * sizeof(double),
                                          jacobianData);

    delete [] jacobianData;

    return jacobian;
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

    return device.buildKernel("occa://mfem/fem/" + filename,
                              stringWithDim(kernelName, bilinearForm.GetDim()),
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
    occa::properties kernelProps = props;

    DiffusionIntegrator &integ = (DiffusionIntegrator&) *integrator;

    const FiniteElement &fe   = bilinearForm.GetFE(0);
    const IntegrationRule &ir = integ.GetIntegrationRule(fe, fe);

    const H1_TensorBasisElement *el = dynamic_cast<const H1_TensorBasisElement*>(&fe);
    if (el) {
      maps = OccaDofQuadMaps::GetTensorMaps(device, bilinearForm, *el, ir);
      setTensorProperties(fe, ir, kernelProps);
    } else {
    }

    const int dims = bilinearForm.GetDim();
    const int symmDims = (dims * (dims + 1)) / 2; // 1x1: 1, 2x2: 3, 3x3: 6

    const int elements = bilinearForm.GetNE();
    const int quadraturePoints = ir.GetNPoints();

    // Get coefficient from integrator
    // [MISSING] Hard-coded to ConstantCoefficient for now
    const ConstantCoefficient* coeff =
      dynamic_cast<const ConstantCoefficient*>(integ.GetCoefficient());

    if (coeff) {
        hasConstantCoefficient = true;
        kernelProps["defines/CONST_COEFF"] = coeff->constant;
    } else {
      mfem_error("OccaDiffusionIntegrator can only handle ConstantCoefficients");
    }

    assembledOperator = device.malloc(elements
                                      * quadraturePoints
                                      * symmDims
                                      * sizeof(double));

    jacobian = getJacobian(device, bilinearForm, ir);

    // Setup assemble and mult kernels
    assembleKernel = GetAssembleKernel(kernelProps);
    multKernel     = GetMultKernel(kernelProps);
  }

  void OccaDiffusionIntegrator::Assemble() {
    if (hasConstantCoefficient) {
      // Dummy coefficient since we're defining it at compile time
      assembleKernel((int) bilinearForm.GetNE(),
                     maps.quadWeights,
                     jacobian,
                     (double) 0,
                     assembledOperator);
    } else {
      assembleKernel((int) bilinearForm.GetNE(),
                     maps.quadWeights,
                     jacobian,
                     coefficients,
                     assembledOperator);
    }
  }

  void OccaDiffusionIntegrator::Mult(OccaVector &x) {
    multKernel((int) bilinearForm.GetNE(),
               maps.dofToQuad,
               maps.dofToQuadD,
               maps.quadToDof,
               maps.quadToDofD,
               assembledOperator,
               x.GetData());
  }
  //====================================
}

#endif