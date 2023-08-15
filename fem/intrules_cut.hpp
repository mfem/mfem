// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CUTINTRULES
#define MFEM_CUTINTRULES

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "intrules.hpp"
#include "eltrans.hpp"
#include "coefficient.hpp"

#ifdef MFEM_USE_LAPACK

namespace mfem
{

/**
 @brief Class for surface integration rule by moment fitting

 Highly accurate surface integration on implicit domains by means of moment-
 fitting (see https://onlinelibrary.wiley.com/doi/full/10.1002/nme.4569)
 */
class SIntegrationRule : public Array<IntegrationPoint>
{
private:
   /**
    @brief The transformation for the current element the integration rule is
    for
    */
   IsoparametricTransformation Trafo;
   /**
    @brief The level-set function defining the implicit surface
    */
   Coefficient &LvlSet;
   /**
    @brief The order of the quadrature rule
    */
   int Order;
   /**
    @brief The number of basis functions
    */
   int nBasis;
   /**
    @brief The current element number of the integration rule
    */
   int ElementNo;

   /**
    @brief Compute the quadrature weights

    Compute the quadrature weights for the surface quadrature rule by mmeans of
    moment-fitting. To construct the quadrature rule, special integrals are
    reduced to integrals over the edges of the subcell where the level-set is
    positiv.
    */
   void ComputeWeights();

   /**
    @brief A divergence free basis on the element [-1,1]x[-1,1]
    */
   void Basis(const IntegrationPoint& ip, DenseMatrix& shape);
   /**
    @brief A orthogonalized divergence free basis on the element [-1,1]x[-1,1]
    */
   void OrthoBasis(const IntegrationPoint& ip, DenseMatrix& shape);
   /**
    @brief A step of the modified Gram-Schmidt algorithm
    */
   void mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN, int step);

public:
   /**
    @brief Constructor for the surface integration rule

    Constructor for the surface integration rule by means of moment-fitting.

    @param [in] q order of the integration rule
    @param [in] Tr Transformation of the element the integration rule is for
    @param [in] levelset level-set defining the implicit interface
    */
   SIntegrationRule(int q, ElementTransformation& Tr,
                    Coefficient &levelset);

   /**
    @brief Update the surface integration rule

    Update the surface integration for a new element given by the
    transformation.

    @param [in] transformation for the new element
    */
   void Update(IsoparametricTransformation& Tr);

   /**
    @brief Get the order of the integration rule
    */
   int GetOrder() const { return Order; }

   /**
    @brief Get the number of quadrature points of the integration rule
    */
   int GetNPoints() const { return Size(); }

   /**
    @brief Get a reference to the i-th quadrature point
    */
   IntegrationPoint &IntPoint(int i) { return (*this)[i]; }

   /**
    @brief Get a const reference to the i-th quadrature point
    */
   const IntegrationPoint &IntPoint(int i) const { return (*this)[i]; }

   /**
    @brief Gett the elemnt number the integration rule is for
    */
   int GetElement() { return ElementNo; }

   /**
    @brief Destructor of the surface integration rule
    */
   ~SIntegrationRule() { }
};

////////////////////////////////////////////////////////////////////////////////

/**
 @brief Class for surface integration rule by moment fitting

 Highly accurate surface integration on implicit domains by means of moment-
 fitting (see https://onlinelibrary.wiley.com/doi/full/10.1002/nme.4569)
 */
class CutIntegrationRule : public Array<IntegrationPoint>
{
private:
   /**
    @brief Surface integration rule for the boundary of the subdomain
    */
   SIntegrationRule* SIR;
   /**
    @brief The transformation for the current element the integration rule is
    for
    */
   IsoparametricTransformation Trafo;
   /**
    @brief The level-set function defining the implicit surface
    */
   Coefficient &LvlSet;
   /**
    @brief The order of the quadrature rule
    */
   int Order;
   /**
    @brief The number of basis functions
    */
   int nBasis;
   /**
    @brief The current element number of the integration rule
    */
   int ElementNo;
   /**
    @brief The quadrature weights for elements fully inside the subdomain
    */
   Vector InteriorWeights;

   /**
    @brief Singular value decomposition of the moment-fitting system
    */
   DenseMatrixSVD* SVD;

   /**
    @brief Compute the quadrature weights

    Compute the quadrature weights for the volumetric subdomain quadrature rule
    by mmeans of moment-fitting. To construct the quadrature rule, special
    integrals are reduced to integrals over the boundary of the subcell where
    the level-set is positiv.
    */
   void ComputeWeights();

   /**
    @brief Monomial basis on the element [-1,1]x[-1,1]
    */
   void Basis(const IntegrationPoint& ip, Vector& shape);
   /**
    @brief Antiderivatives of the ,onomial basis on the element [-1,1]x[-1,1]
    */
   void BasisAntiDerivative(const IntegrationPoint& ip, DenseMatrix& shape);

public:
   /**
      @brief Constructor for the volumetric subdomain integration rule

      Constructor for the volumetric subdomain integration rule by means of
      moment-fitting.

      @param [in] q order of the integration rule
      @param [in] Tr Transformation of the element the integration rule is for
      @param [in] levelset level-set defining the implicit interface
      */
   CutIntegrationRule(int q, ElementTransformation& Tr,
                      Coefficient &levelset);

   /**
    @brief Get a pointer to the surface integration rule

    Get a pointer to the surface integration rule. The surface integration rule
    can be updated, but this will not change the volumetric integration rule.
    The update has to be called.
    */
   SIntegrationRule* GetSurfaceIntegrationRule() { return SIR; }

   /**
    @brief Update the surface integration rule

    Update the surface integration for a new element given by the
    transformation. The surface integration rule will be updated if needed.

    @param [in] transformation for the new element
    */
   void Update(IsoparametricTransformation& Tr);

   /**
    @brief Get the order of the integration rule
    */
   int GetOrder() const { return Order; }

   /**
    @brief Get the number of quadrature points of the integration rule
    */
   int GetNPoints() const { return Size(); }

   /**
    @brief Get a reference to the i-th quadrature point
    */
   IntegrationPoint &IntPoint(int i) { return (*this)[i]; }

   /**
    @brief Get a const reference to the i-th quadrature point
    */
   const IntegrationPoint &IntPoint(int i) const { return (*this)[i]; }

   /**
    @brief Gett the elemnt number the integration rule is for
    */
   int GetElement() { return ElementNo; }

   /**
    @brief Destructor of the subdomain integration rule
    */
   ~CutIntegrationRule();
};

}

#endif //MFEM_USE_LAPACK

#endif