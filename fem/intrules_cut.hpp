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
 fitting (see https://onlinelibrary.wiley.com/doi/full/10.1002/nme.4569).
 */
class SIntegrationRule : public IntegrationRule
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
    @brief The number of basis functions
    */
   int nBasis;
   /**
    @brief Column-wise matrix of the quadtrature weights
    */
   DenseMatrix Weights;
   /**
    @brief Array of face integration points
    */
   Array<IntegrationPoint> FaceIP;
   /**
    @brief Column-wise Matrix f the face quadrature weights
    */
   DenseMatrix FaceWeights;

   /**
    @brief Compute 2D quadrature weights

    Compute the quadrature weights for the 2D surface quadrature rule by means
    of moment-fitting. To construct the quadrature rule, special integrals are
    reduced to integrals over the edges of the subcell where the level-set is
    positiv.
    */
   void ComputeWeights2D();

   /**
    @brief Compute 3D quadrature weights

    Compute the quadrature weights for the 3D surface quadrature rule by means
    of moment-fitting. To construct the quadrature rule, special integrals are
    reduced to integrals over the edges of the subcell where the level-set is
    positiv.
    */
   void ComputeWeights3D();

   /**
    @brief A divergence free basis on the element [-1,1]x[-1,1]
    */
   void Basis2D(const IntegrationPoint& ip, DenseMatrix& shape);
   /**
    @brief A orthogonalized divergence free basis on the element [-1,1]x[-1,1]
    */
   void OrthoBasis2D(const IntegrationPoint& ip, DenseMatrix& shape);
   /**
    @brief A orthogonalized divergence free basis on the element [-1,1]^3
    */
   void OrthoBasis3D(const IntegrationPoint& ip, DenseMatrix& shape);
   /**
    @brief A step of the modified Gram-Schmidt algorithm
    */
   void mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN, int step);

public:
   /**
    @brief Constructor for the surface integration rule

    Constructor for the surface integration rule by means of moment-fitting.

    @param [in] q order of the integration rule
    @param [in] Tr volumetric transformation of the element
    @param [in] levelset level-set defining the implicit interface
    */
   SIntegrationRule(int q, ElementTransformation& Tr,
                    Coefficient &levelset);

   /**
    @brief Update the surface integration rule

    Update the surface integration for a new element given by the
    transformation.

    @param [in] Tr volumetric transformation for the new element
    */
   void Update(IsoparametricTransformation& Tr);

   /**
    @brief Update the interface

    Update the surface integration rule for a new implicit interface.

    @param [in] levelset level-set defining the implicit interface
    */
   void UpdateInterface(Coefficient& levelset);

   /**
    @brief Set the elemnt number the integration rule is for
    */
   void SetElement(int ElementNo);
   
   /**
    @brief Get the elemnt number the integration rule is for
    */
   int GetElement() { return Trafo.ElementNo; }

   /**
    @brief Destructor of the surface integration rule
    */
   ~SIntegrationRule() { }
};

////////////////////////////////////////////////////////////////////////////////

/**
 @brief Class for surface integration rule by moment fitting

 Highly accurate surface integration on implicit domains by means of moment-
 fitting (see https://onlinelibrary.wiley.com/doi/full/10.1002/nme.4569).
 */
class CutIntegrationRule : public IntegrationRule
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
    @brief The number of basis functions
    */
   int nBasis;
   /**
    @brief Column-wise matrix of the quadtrature weights
    */
   DenseMatrix Weights;
   /**
    @brief Array of face integration points
    */
   Array<IntegrationPoint> FaceIP;
   /**
    @brief Column-wise Matrix f the face quadrature weights
    */
   DenseMatrix FaceWeights;
   
   /**
    @brief The quadrature weights for elements fully inside the subdomain
    */
   Vector InteriorWeights;

   /**
    @brief Singular value decomposition of the moment-fitting system
    */
   DenseMatrixSVD* SVD;

   /**
    @brief Compute the 2D quadrature weights

    Compute the 2D quadrature weights for the volumetric subdomain quadrature
    rule by mmeans of moment-fitting. To construct the quadrature rule, special
    integrals are reduced to integrals over the boundary of the subcell where
    the level-set is positiv.
    */
   void ComputeWeights2D();

   /**
    @brief Compute the 3D quadrature weights

    Compute the 3D quadrature weights for the volumetric subdomain quadrature
    rule by mmeans of moment-fitting. To construct the quadrature rule, special
    integrals are reduced to integrals over the boundary of the subcell where
    the level-set is positiv.
    */
   void ComputeWeights3D();

   /**
    @brief Monomial basis on the element [-1,1]x[-1,1]
    */
   void Basis2D(const IntegrationPoint& ip, Vector& shape);
   /**
    @brief Antiderivatives of the ,onomial basis on the element [-1,1]x[-1,1]
    */
   void BasisAntiDerivative2D(const IntegrationPoint& ip, DenseMatrix& shape);
   /**
    @brief Monomial basis on the element [-1,1]x[-1,1]
    */
   void Basis3D(const IntegrationPoint& ip, Vector& shape);
   /**
    @brief Antiderivatives of the ,onomial basis on the element [-1,1]x[-1,1]
    */
   void BasisAntiDerivative3D(const IntegrationPoint& ip, DenseMatrix& shape);

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

    @param [in] Tr transformation for the new element
    */
   void Update(IsoparametricTransformation& Tr);

   /**
    @brief Update the interface

    Update the surface integration rule for a new implicit interface.

    @param [in] levelset level-set defining the implicit interface
    */
   void UpdateInterface(Coefficient& levelset);

   /**
    @brief Set the elemnt number the integration rule is for
    */
   void SetElement(int ElementNo);
   
   /**
    @brief Get the elemnt number the integration rule is for
    */
   int GetElement() { return Trafo.ElementNo; }


   /**
    @brief Destructor of the subdomain integration rule
    */
   ~CutIntegrationRule();
};

namespace DivFreeBasis
{

inline void Get3DBasis(const Vector& X, DenseMatrix& shape, int Order)
{
   int nBasis;
   if(Order == 0) { nBasis = 3; }
   else if(Order == 1) { nBasis = 11; }
   else if(Order == 2) { nBasis = 26; }
   else if(Order == 3) { nBasis = 50; }
   else if(Order == 4) { nBasis = 83; }
   else if(Order == 5) { nBasis = 133; }
   else if(Order == 6) { nBasis = 196; }
   else if(Order == 7) { nBasis = 276; }
   else if(Order == 8) { nBasis = 375; }
   else if(Order == 9) { nBasis = 495; }
   else if(Order == 10) { nBasis = 638; }
   else if(Order == 11) { nBasis = 806; }
   else if(Order == 12) { nBasis = 1001; }

   shape.SetSize(nBasis, 3);
   shape = 0.;

   double xi = X(0);
   double eta = X(1);
   double nu = X(2);

   switch (Order)
   {
      case 12:
      case 11:
      case 10:
      case 9:
      case 8:
      case 7:
      case 6:
      case 5:
      case 4:
      case 3:
         shape(  26,0) = -1.403121520040228*nu + 2.3385358667337135*pow(nu, 3.);
         shape(  27,1) = -1.403121520040228*nu + 2.3385358667337135*pow(nu, 3.);
         shape(  28,0) = -0.6846531968814576*eta + 2.053959590644373*eta*pow(nu, 2.);
         shape(  29,1) = -0.6846531968814576*eta + 2.053959590644373*eta*pow(nu, 2.);
         shape(  29,2) =  0.6846531968814576*nu - 0.6846531968814576*pow(nu, 3.);
         shape(  30,0) = -0.6846531968814576*nu + 2.053959590644373*pow(eta, 2.)*nu;
         shape(  31,1) = -0.6846531968814576*nu + 2.053959590644373*pow(eta, 2.)*nu;
         shape(  31,2) =  0.6846531968814576*eta - 2.053959590644373*eta*pow(nu, 2.);
         shape(  32,0) = -1.403121520040228*eta + 2.3385358667337135*pow(eta, 3.);
         shape(  33,1) = -1.403121520040228*eta + 2.3385358667337135*pow(eta, 3.);
         shape(  33,2) =  1.403121520040228*nu - 7.0156076002011405*pow(eta, 2.)*nu;
         shape(  34,2) = -1.403121520040228*eta + 2.3385358667337135*pow(eta, 3.);
         shape(  35,0) = -0.6846531968814576*xi + 2.053959590644373*xi*pow(nu, 2.);
         shape(  35,2) =  0.6846531968814576*nu - 0.6846531968814576*pow(nu, 3.);
         shape(  36,1) = -0.6846531968814576*xi + 2.053959590644373*xi*pow(nu, 2.);
         shape(  37,0) =  1.8371173070873836*xi*eta*nu;
         shape(  37,2) =  0.30618621784789724*eta - 0.9185586535436918*eta*pow(nu, 2.);
         shape(  38,1) =  1.8371173070873836*xi*eta*nu;
         shape(  38,2) =  0.30618621784789724*xi - 0.9185586535436918*xi*pow(nu, 2.);
         shape(  39,0) = -0.6846531968814576*xi + 2.053959590644373*xi*pow(eta, 2.);
         shape(  39,2) =  0.6846531968814576*nu - 2.053959590644373*pow(eta, 2.)*nu;
         shape(  40,1) = -0.6846531968814576*xi + 2.053959590644373*xi*pow(eta, 2.);
         shape(  40,2) = -4.107919181288746*xi*eta*nu;
         shape(  41,2) = -0.6846531968814576*xi + 2.053959590644373*xi*pow(eta, 2.);
         shape(  42,0) = -0.6846531968814576*nu + 2.053959590644373*pow(xi, 2.)*nu;
         shape(  42,2) =  0.6846531968814576*xi - 2.053959590644373*xi*pow(nu, 2.);
         shape(  43,1) = -0.6846531968814576*nu + 2.053959590644373*pow(xi, 2.)*nu;
         shape(  44,0) = -0.6846531968814576*eta + 2.053959590644373*pow(xi, 2.)*eta;
         shape(  44,2) = -4.107919181288746*xi*eta*nu;
         shape(  45,1) = -0.6846531968814576*eta + 2.053959590644373*pow(xi, 2.)*eta;
         shape(  45,2) =  0.6846531968814576*nu - 2.053959590644373*pow(xi, 2.)*nu;
         shape(  46,2) = -0.6846531968814576*eta + 2.053959590644373*pow(xi, 2.)*eta;
         shape(  47,0) = -1.403121520040228*xi + 2.3385358667337135*pow(xi, 3.);
         shape(  47,2) =  1.403121520040228*nu - 7.0156076002011405*pow(xi, 2.)*nu;
         shape(  48,1) = -1.403121520040228*xi + 2.3385358667337135*pow(xi, 3.);
         shape(  49,2) = -1.403121520040228*xi + 2.3385358667337135*pow(xi, 3.);

      case 2:
         shape(  11,0) = -0.39528470752104744 + 1.1858541225631423*pow(nu, 2.);
         shape(  12,1) = -0.39528470752104744 + 1.1858541225631423*pow(nu, 2.);
         shape(  13,0) =  1.0606601717798212*eta*nu;
         shape(  14,1) =  1.0606601717798212*eta*nu;
         shape(  14,2) =  0.1767766952966369 - 0.5303300858899106*pow(nu, 2.);
         shape(  15,0) = -0.39528470752104744 + 1.1858541225631423*pow(eta, 2.);
         shape(  16,1) = -0.39528470752104744 + 1.1858541225631423*pow(eta, 2.);
         shape(  16,2) = -2.3717082451262845*eta*nu;
         shape(  17,2) = -0.39528470752104744 + 1.1858541225631423*pow(eta, 2.);
         shape(  18,0) =  1.0606601717798212*xi*nu;
         shape(  18,2) =  0.1767766952966369 - 0.5303300858899106*pow(nu, 2.);
         shape(  19,1) =  1.0606601717798212*xi*nu;
         shape(  20,0) =  1.0606601717798212*xi*eta;
         shape(  20,2) = -1.0606601717798212*eta*nu;
         shape(  21,1) =  1.0606601717798212*xi*eta;
         shape(  21,2) = -1.0606601717798212*xi*nu;
         shape(  22,2) =  1.0606601717798212*xi*eta;
         shape(  23,0) = -0.39528470752104744 + 1.1858541225631423*pow(xi, 2.);
         shape(  23,2) = -2.3717082451262845*xi*nu;
         shape(  24,1) = -0.39528470752104744 + 1.1858541225631423*pow(xi, 2.);
         shape(  25,2) = -0.39528470752104744 + 1.1858541225631423*pow(xi, 2.);
      case 1:
         shape(   3,0) =  0.6123724356957945*nu;
         shape(   4,1) =  0.6123724356957945*nu;
         shape(   5,0) =  0.6123724356957945*eta;
         shape(   6,1) =  0.6123724356957945*eta;
         shape(   6,2) = -0.6123724356957945*nu;
         shape(   7,2) =  0.6123724356957945*eta;
         shape(   8,0) =  0.6123724356957945*xi;
         shape(   8,2) = -0.6123724356957945*nu;
         shape(   9,1) =  0.6123724356957945*xi;
         shape(  10,2) =  0.6123724356957945*xi;
      case 0:
         shape(   0,0) =  0.3535533905932738;
         shape(   1,1) =  0.3535533905932738;
         shape(   2,2) =  0.3535533905932738;
   }
}

}

}

#endif //MFEM_USE_LAPACK

#endif