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

#ifndef MFEM_BACKENDS_OMP_ADIFFUSIONINTEG_HPP
#define MFEM_BACKENDS_OMP_ADIFFUSIONINTEG_HPP

#include "../../config/config.hpp"
#if defined(MFEM_USE_BACKENDS) &&               \
   defined(MFEM_USE_OMP) &&                     \
   defined(MFEM_USE_ACROTENSOR)

#include "../../fem/bilininteg.hpp"
#include "../../fem/fem.hpp"
#include "vector.hpp"
#include "fespace.hpp"
#include "bilinearform.hpp"
#include "AcroTensor.hpp"

namespace mfem
{

namespace omp
{

class PAIntegrator : public TensorBilinearFormIntegrator
{
protected:
   Coefficient *Q;
   FiniteElementSpace *ofes;
   mfem::FiniteElementSpace *fes;
   const FiniteElement *fe;
   const TensorBasisElement *tfe;
   const IntegrationRule *ir;
   mfem::Array<int> tDofMap;
   int GeomType;
   int FEOrder;
   bool onGPU;
   bool hasTensorBasis;
   int nDim;
   int nElem;
   int nDof;
   int nQuad;

public:
   PAIntegrator(Coefficient &q, FiniteElementSpace &f);
   virtual ~PAIntegrator();
};

class AcroDiffusionIntegrator : public PAIntegrator
{
private:
   acro::TensorEngine TE;
   int nDof1D;
   int nQuad1D;

   acro::Tensor B, G;         //Basis and dbasis evaluated on the quad points
   acro::Tensor W;            //Integration weights
   mfem::Array<acro::Tensor*> Btil; //Btilde used to compute stiffness matrix
   acro::Tensor D;            //Product of integration weight, physical consts, and element shape info
   acro::Tensor S;            //The assembled local stiffness matrices
   acro::Tensor U, Z, T1, T2; //Intermediate computations for tensor product partial assembly
   acro::Tensor X, Y;

   void ComputeBTilde();

public:
   AcroDiffusionIntegrator(BilinearFormIntegrator *integ);
   AcroDiffusionIntegrator(Coefficient &q, FiniteElementSpace &f);
   virtual ~AcroDiffusionIntegrator();

   void BatchedPartialAssemble();
   void BatchedAssembleElementMatrices(DenseTensor &elmats);
   void ComputeElementMatrices(Vector &elmats);
   void PAMult(const Vector &x, Vector &y);
   virtual void MultTransposeAdd(const Vector &x, Vector &y) const;
   virtual void MultAdd(const Vector &x, Vector &y) const;
   virtual void ReassembleOperator();
};


} // namespace mfem::omp

} // namespace mfem

#endif

#endif
