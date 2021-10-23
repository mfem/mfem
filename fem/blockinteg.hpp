// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BLOCKLININTEG
#define MFEM_BLOCKLININTEG

#include "../config/config.hpp"
#include "fe.hpp"
#include "coefficient.hpp"
#include "fespace.hpp"

namespace mfem
{
/** The abstract base class BlockBilinearFormIntegrator is
 a generalization of the BilinearFormIntegrator class suitable
 for block formulations. */
class BlockBilinearFormIntegrator
{
protected:
   BlockBilinearFormIntegrator(const IntegrationRule *ir = NULL) {}

public:
   /// Given a particular Finite Element computes the element matrix elmat.
   virtual void AssembleElementMatrix(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~BlockBilinearFormIntegrator() { }
};

/** The abstract base class BlockBilinearFormIntegrator is
 a generalization of the BilinearFormIntegrator class suitable
 for block formulations. */
class BlockLinearFormIntegrator
{
protected:
   BlockLinearFormIntegrator(const IntegrationRule *ir = NULL) {}

public:
   /// Given a particular Finite Element computes the element matrix elmat.
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Trans,
                                      Vector &elvect);

   virtual ~BlockLinearFormIntegrator() { }
};




/** Class for local matrix assembling a(u,v) := (u, v) */
class TestBlockBilinearFormIntegrator: public BlockBilinearFormIntegrator
{
protected:
   Coefficient *Q;
   Array<const FiniteElementSpace * > fespaces;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:

   /// Construct a mass integrator with coefficient q
   TestBlockBilinearFormIntegrator(Coefficient &q,
                                   const IntegrationRule *ir = NULL)
      : BlockBilinearFormIntegrator(ir), Q(&q), maps(NULL), geom(NULL) { }

   /** Given a particular Finite Element computes the element matrix
       elmat. */
   virtual void AssembleElementMatrix(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

};

/** Class for local vector assembly */
class TestBlockLinearFormIntegrator: public BlockLinearFormIntegrator
{
protected:
   Coefficient *Q;
   Array<const FiniteElementSpace * > fespaces;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:

   /// Construct a mass integrator with coefficient q
   TestBlockLinearFormIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BlockLinearFormIntegrator(ir), Q(&q), maps(NULL), geom(NULL) { }

   /** Given a particular Finite Element computes the element vector */
   virtual void AssembleElementVector(const Array<const FiniteElement *> &el,
                                      ElementTransformation &Trans,
                                      Vector &elvector);

};


} // namespace mfem


#endif
