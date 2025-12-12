// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
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


#ifdef MFEM_USE_ALGOIM
#ifdef MFEM_HAVE_GCC_PRAGMA_DIAGNOSTIC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <algoim_quad.hpp>
#pragma GCC diagnostic pop
#endif

namespace mfem
{
/**
 @brief Abstract class for construction of IntegrationRules in cut elements.

 Interface for construction of cut-surface and cut-volume IntegrationRules. The
 cut is specified by the zero level set of a given Coefficient.
*/
class CutIntegrationRules
{
protected:
   /// Order of the IntegrationRule.
   int Order;
   /// The zero level set of this Coefficient defines the cut surface.
   Coefficient* LvlSet;
   /// Space order for the LS projection.
   int lsOrder;

   /// @name Tolerances used for point comparisons
   ///@{
#ifdef MFEM_USE_DOUBLE
   static constexpr real_t tol_1 = 1e-12;
   static constexpr real_t tol_2 = 1e-15;
#elif defined(MFEM_USE_SINGLE)
   static constexpr real_t tol_1 = 1e-5;
   static constexpr real_t tol_2 = 1e-7;
#endif
   ///@}

   /** @brief Constructor to set up the generated cut IntegrationRules.

       @param [in] order  Order of the constructed IntegrationRule.
       @param [in] lvlset Coefficient whose zero level set specifies the cut.
       @param [in] lsO    Polynomial degree for projecting the level-set
                          Coefficient to a GridFunction, which is used to
                          compute gradients and normals. */
   CutIntegrationRules(int order, Coefficient& lvlset, int lsO = 2)
      : Order(order), LvlSet(&lvlset), lsOrder(lsO)
   { MFEM_VERIFY(order > 0 && lsO > 0, "Invalid input") }

public:

   /// Change the order of the constructed IntegrationRule.
   virtual void SetOrder(int order);

   /// Change the Coefficient whose zero level set specifies the cut.
   virtual void SetLevelSetCoefficient(Coefficient &ls) { LvlSet = &ls; }

   /// Change the polynomial degree for projecting the level set Coefficient
   /// to a GridFunction, which is used to compute local gradients and normals.
   virtual void SetLevelSetProjectionOrder(int order);

   /**
    @brief Construct a cut-surface IntegrationRule.

    Construct an IntegrationRule to integrate on the surface given by the
    already specified level set function, for the element given by @a Tr.

    @param [in]  Tr     Specifies the IntegrationRule's associated mesh element.
    @param [out] result IntegrationRule on the cut-surface
   */
   virtual void GetSurfaceIntegrationRule(ElementTransformation& Tr,
                                          IntegrationRule& result) = 0;

   /**
    @brief Construct a cut-volume IntegrationRule.

    Construct an IntegrationRule to integrate in the subdomain given by the
    positive values of the already specified level set function, for the element
    given by @a Tr.

    @param [in]  Tr     Specifies the IntegrationRule's associated mesh element.
    @param [out] result IntegrationRule for the cut-volume
    @param [in]  sir    Corresponding IntegrationRule for the surface, which can
                        be used to avoid computations.
   */
   virtual void GetVolumeIntegrationRule(ElementTransformation& Tr,
                                         IntegrationRule& result,
                                         const IntegrationRule* sir = NULL) = 0;

   /**
    @brief Compute transformation quadrature weights for surface integration.

    Compute the transformation weights for integration over the cut-surface in
    reference space.

    @param [in]  Tr      Specifies the IntegrationRule's associated element.
    @param [in]  sir     IntegrationRule defining the IntegrationPoints
    @param [out] weights Vector containing the transformation weights.
   */
   virtual void GetSurfaceWeights(ElementTransformation &Tr,
                                  const IntegrationRule &sir,
                                  Vector &weights) = 0;

   /// @brief Destructor of CutIntegrationRules
   virtual ~CutIntegrationRules() {}
};

#ifdef MFEM_USE_ALGOIM
// define templated element bases
namespace TmplPoly_1D
{

/// Templated version of CalcBinomTerms
template<typename float_type>
void CalcBinomTerms(const int p, const float_type x, const float_type y,
                    float_type* u)
{
   if (p == 0)
   {
      u[0] = float_type(1.);
   }
   else
   {
      int i;
      const int *b = Poly_1D::Binom(p);
      float_type z = x;
      for (i = 1; i < p; i++)
      {
         u[i] = b[i]*z;
         z *= x;
      }
      u[p] = z;
      z = y;
      for (i--; i > 0; i--)
      {
         u[i] *= z;
         z *= y;
      }
      u[0] = z;
   }
}

/// Templated version of CalcBinomTerms
template<typename float_type>
void CalcBinomTerms(const int p, const float_type x, const float_type y,
                    float_type* u, float_type* d)
{
   if (p == 0)
   {
      u[0] = float_type(1.);
      d[0] = float_type(0.);
   }
   else
   {
      int i;
      const int *b = Poly_1D::Binom(p);
      const float_type xpy = x + y, ptx = p*x;
      float_type z = float_type(1.);

      for (i = 1; i < p; i++)
      {
         d[i] = b[i]*z*(i*xpy - ptx);
         z *= x;
         u[i] = b[i]*z;
      }
      d[p] = p*z;
      u[p] = z*x;
      z = float_type(1.);
      for (i--; i > 0; i--)
      {
         d[i] *= z;
         z *= y;
         u[i] *= z;
      }
      d[0] = -p*z;
      u[0] = z*y;
   }

}

/// Templated evaluation of Bernstein basis
template <typename float_type>
void CalcBernstein(const int p, const float_type x, float_type *u)
{
   CalcBinomTerms(p, x, 1. - x, u);
}


/// Templated evaluation of Bernstein basis
template <typename float_type>
void CalcBernstein(const int p, const float_type x,
                   float_type *u, float_type *d)
{
   CalcBinomTerms(p, x, 1. - x, u, d);
}


}

class AlgoimIntegrationRules : public CutIntegrationRules
{
public:

   /** @brief Constructor to set up the generated cut IntegrationRules.

       @param [in] order  Order of the constructed IntegrationRule.
       @param [in] lvlset Coefficient whose zero level set specifies the cut.
       @param [in] lsO    Polynomial degree for projecting the level-set
                          Coefficient to a GridFunction, which is used to
                          compute gradients and normals. */
   AlgoimIntegrationRules(int order, Coefficient &lvlset, int lsO = 2)
      : CutIntegrationRules(order, lvlset, lsO)
   {
      pe=nullptr;
      le=nullptr;
      currentLvlSet=nullptr;
      currentGeometry=Geometry::Type::INVALID;
      currentElementNo = -1;
   }

   virtual ~AlgoimIntegrationRules()
   {
      delete pe;
      delete le;
   }

   virtual void SetOrder(int order) override
   {
      MFEM_VERIFY(order > 0, "Invalid input");
      Order = order;
      delete pe;
      delete le;
      pe=nullptr;
      le=nullptr;
      currentLvlSet=nullptr;
      currentGeometry=Geometry::Type::INVALID;
      currentElementNo=-1;
   }

   virtual void SetLevelSetProjectionOrder(int order) override
   {
      MFEM_VERIFY(order > 0, "Invalid input");
      lsOrder = order;
      delete pe;
      delete le;
      pe=nullptr;
      le=nullptr;
      currentLvlSet=nullptr;
      currentGeometry=Geometry::Type::INVALID;
      currentElementNo=-1;
   }


   /**
   @brief Construct a cut-surface IntegrationRule.

   Construct an IntegrationRule to integrate on the surface given by the
   already specified level set function, for the element given by @a Tr.

   @param [in]  Tr     Specifies the IntegrationRule's associated mesh element.
   @param [out] result IntegrationRule on the cut-surface
   */
   virtual
   void GetSurfaceIntegrationRule(ElementTransformation &Tr,
                                  IntegrationRule &result) override;

   /**
   @brief Construct a cut-volume IntegrationRule.

   Construct an IntegrationRule to integrate in the subdomain given by the
   positive values of the already specified level set function, for the element
   given by @a Tr.

   @param [in]  Tr     Specifies the IntegrationRule's associated mesh element.
   @param [out] result IntegrationRule for the cut-volume
   @param [in]  sir    Corresponding IntegrationRule for the surface, which can
                       be used to avoid computations.
   */
   virtual
   void GetVolumeIntegrationRule(ElementTransformation &Tr,
                                 IntegrationRule &result,
                                 const IntegrationRule *sir = nullptr) override;


   /**
   @brief Compute transformation quadrature weights for surface integration.

   Compute the transformation weights for integration over the cut-surface in
   reference space.

   @param [in]  Tr      Specifies the IntegrationRule's associated element.
   @param [in]  sir     IntegrationRule defining the IntegrationPoints
   @param [out] weights Vector containing the transformation weights.
   */
   virtual
   void GetSurfaceWeights(ElementTransformation &Tr,
                          const IntegrationRule &sir,
                          Vector &weights) override;

private:

   /// projects the lvlset coefficient onto the lsvec,
   /// i.e., represent the level-set using Bernstein bases
   void GenerateLSVector(ElementTransformation &Tr, Coefficient* lvlset);


   /// Lagrange finite element used for converting coefficients to positive basis
   FiniteElement* le;
   PositiveTensorFiniteElement *pe;
   DenseMatrix T; //Projection matrix from nodal basis to positive basis
   Vector lsvec; // level-set in Bernstein basis
   Vector lsfun; // level-set in nodal basis
   Geometry::Type currentGeometry; // the current element geometry
   Coefficient* currentLvlSet; //the current level-set coefficient
   int currentElementNo; //the current element No

   /// 3D level-set function object required by Algoim.
   struct LevelSet3D
   {
      /// Constructor for 3D level-set function object required by Algoim.
      LevelSet3D(PositiveTensorFiniteElement* el_, Vector& lsfun_)
         : el(el_), lsfun(lsfun_) { }

      /// Returns the value of the LSF for point x.
      template<typename T>
      T operator() (const blitz::TinyVector<T,3>& x) const
      {
         const int el_order = el->GetOrder();
         std::vector<T> u1(el_order+1);
         std::vector<T> u2(el_order+1);
         std::vector<T> u3(el_order+1);
         TmplPoly_1D::CalcBernstein(el_order, x[0], u1.data());
         TmplPoly_1D::CalcBernstein(el_order, x[1], u2.data());
         TmplPoly_1D::CalcBernstein(el_order, x[2], u3.data());

         const Array<int>& dof_map=el->GetDofMap();

         T res=T(0.0);
         for (int oo = 0, kk = 0; kk <= el_order; kk++)
            for (int jj = 0; jj <= el_order; jj++)
               for (int ii = 0; ii <= el_order; ii++)
               {
                  res=res-u1[ii]*u2[jj]*u3[kk]*lsfun(dof_map[oo++]);
               }
         return res;
      }

      /// Returns the gradients of the LSF for point x.
      template<typename T>
      blitz::TinyVector<T,3> grad(const blitz::TinyVector<T,3>& x) const
      {
         const int el_order = el->GetOrder();
         std::vector<T> u1(el_order+1);
         std::vector<T> u2(el_order+1);
         std::vector<T> u3(el_order+1);
         std::vector<T> d1(el_order+1);
         std::vector<T> d2(el_order+1);
         std::vector<T> d3(el_order+1);

         TmplPoly_1D::CalcBernstein(el_order,x[0], u1.data(), d1.data());
         TmplPoly_1D::CalcBernstein(el_order,x[1], u2.data(), d2.data());
         TmplPoly_1D::CalcBernstein(el_order,x[2], u3.data(), d3.data());

         blitz::TinyVector<T,3> res(T(0.0),T(0.0),T(0.0));

         const Array<int>& dof_map=el->GetDofMap();

         for (int oo = 0, kk = 0; kk <= el_order; kk++)
            for (int jj = 0; jj <= el_order; jj++)
               for (int ii = 0; ii <= el_order; ii++)
               {
                  res[0]=res[0]-d1[ii]*u2[jj]*u3[kk]*lsfun(dof_map[oo]);
                  res[1]=res[1]-u1[ii]*d2[jj]*u3[kk]*lsfun(dof_map[oo]);
                  res[2]=res[2]-u1[ii]*u2[jj]*d3[kk]*lsfun(dof_map[oo]);
                  oo++;
               }

         return res;
      }

   private:
      PositiveTensorFiniteElement* el;
      Vector& lsfun;
   };

   /// 2D level-set function object required by Algoim.
   struct LevelSet2D
   {
      /// Constructor for 2D level-set function object required by Algoim.
      LevelSet2D(PositiveTensorFiniteElement* el_, Vector& lsfun_)
         :el(el_), lsfun(lsfun_) { }

      /// Returns the value of the LSF for point x.
      template<typename T>
      T operator() (const blitz::TinyVector<T,2>& x) const
      {
         const int el_order = el->GetOrder();
         std::vector<T> u1(el_order+1);
         std::vector<T> u2(el_order+1);
         TmplPoly_1D::CalcBernstein(el_order, x[0], u1.data());
         TmplPoly_1D::CalcBernstein(el_order, x[1], u2.data());

         const Array<int>& dof_map=el->GetDofMap();

         T res=T(0.0);

         for (int oo = 0, jj = 0; jj <= el_order; jj++)
            for (int ii = 0; ii <= el_order; ii++)
            {
               res=res-u1[ii]*u2[jj]*lsfun(dof_map[oo++]);
            }
         return res;
      }

      /// Returns the gradients of the LSF for point x.
      template<typename T>
      blitz::TinyVector<T,2> grad(const blitz::TinyVector<T,2>& x) const
      {
         const int el_order = el->GetOrder();
         std::vector<T> u1(el_order+1);
         std::vector<T> u2(el_order+1);
         std::vector<T> d1(el_order+1);
         std::vector<T> d2(el_order+1);

         TmplPoly_1D::CalcBernstein(el_order,x[0], u1.data(), d1.data());
         TmplPoly_1D::CalcBernstein(el_order,x[1], u2.data(), d2.data());

         blitz::TinyVector<T,2> res(T(0.0),T(0.0));

         const Array<int>& dof_map=el->GetDofMap();

         for (int oo = 0, jj = 0; jj <= el_order; jj++)
            for (int ii = 0; ii <= el_order; ii++)
            {
               res[0]=res[0]-(d1[ii]*u2[jj])*lsfun(dof_map[oo]);
               res[1]=res[1]-(u1[ii]*d2[jj])*lsfun(dof_map[oo]);
               oo++;
            }

         return res;
      }


   private:
      PositiveTensorFiniteElement* el;
      Vector& lsfun;
   };
};
#endif //MFEM_USE_ALGOIM

#ifdef MFEM_USE_LAPACK

/**
 @brief Class for subdomain IntegrationRules by means of moment-fitting

 Class for subdomain (surface and subdomain) IntegrationRules by means of
 moment-fitting. The class provides different functions to construct the
 IntegrationRules. The construction is done as described in Mueller et al. in
 "Highly accurate surface and volume integration on implicit domains by means of
 moment-fitting" (2013, see
 https://onlinelibrary.wiley.com/doi/full/10.1002/nme.4569).
*/
class MomentFittingIntRules : public CutIntegrationRules
{
protected:
   /// @brief Space Dimension of the element
   int dim;
   /// @brief Number of divergence-free basis functions for surface integration
   int nBasis;
   /// @brief Number of basis functions for volume integration
   int nBasisVolume;
   /// @brief IntegrationRule representing the reused IntegrationPoints
   IntegrationRule ir;
   /// @brief SVD of the matrix for volumetric IntegrationRules
   DenseMatrixSVD* VolumeSVD;
   /// @brief Array of face integration points
   Array<IntegrationPoint> FaceIP;
   /// @brief Column-wise Matrix for the face quadrature weights
   DenseMatrix FaceWeights;
   /// @brief Indicates the already computed face IntegrationRules
   Vector FaceWeightsComp;

   /**
    @brief Initialization for surface IntegrationRule

    Initialize the members for computation of surface IntegrationRule. This is
    called when the first IntegrationRule is computed or when Order or level-set
    change.

    @param [in] order Order of the IntegrationRule
    @param [in] levelset level-set function defining the implicit interface
    @param [in] lsO polynomial degree for approximation of level-set function
    @param [in] Tr ElemenTransformation to initalize the members with
   */
   void InitSurface(int order, Coefficient& levelset, int lsO,
                    ElementTransformation& Tr);

   /**
    @brief Initialization for volume IntegrationRule

    Initialize the members for computation of surface IntegrationRule. This is
    called when the first IntegrationRule is computed or when Order or level-set
    change.

    @param [in] order Order of the IntegrationRule
    @param [in] levelset level-set function defining the implicit interface
    @param [in] lsO polynomial degree for approximation of level-set function
    @param [in] Tr ElemenTransformation to initalize the members with
   */
   void InitVolume(int order, Coefficient& levelset, int lsO,
                   ElementTransformation& Tr);

   /// @brief Initialize the MomentFittingIntRules
   void Init(int order, Coefficient& levelset, int lsO)
   { Order = order; LvlSet = &levelset; lsOrder = lsO; FaceWeightsComp = 0.;}

   /// @brief Clear stored data of the MomentFittingIntRules
   void Clear();

   /**
    @brief Compute the IntegrationRules on the faces

    Compute the IntegrationRules on the (cut) faces of an element. These will
    be saved and reused if possible.

    @param [in] Tr ElementTransformation of the element the IntegrationRules on
   the faces are computed
   */
   void ComputeFaceWeights(ElementTransformation& Tr);

   /**
    @brief Compute 1D quadrature weights

    Compute the quadrature weights for the 1D surface quadrature rule.

    @param [in] Tr ElementTransformation of the current element
    */
   void ComputeSurfaceWeights1D(ElementTransformation& Tr);

   /**
    @brief Compute the 1D quadrature weights

    Compute the 1D quadrature weights for the volumetric subdomain quadrature
    rule.

    @param [in] Tr ElementTransformation of the current element
    */
   void ComputeVolumeWeights1D(ElementTransformation& Tr);

   /**
    @brief Compute 2D quadrature weights

    Compute the quadrature weights for the 2D surface quadrature rule by means
    of moment-fitting. To construct the quadrature rule, special integrals are
    reduced to integrals over the edges of the subcell where the level-set is
    positive.

    @param [in] Tr ElementTransformation of the current element
    */
   void ComputeSurfaceWeights2D(ElementTransformation& Tr);

   /**
    @brief Compute the 2D quadrature weights

    Compute the 2D quadrature weights for the volumetric subdomain quadrature
    rule by means of moment-fitting. To construct the quadrature rule, special
    integrals are reduced to integrals over the boundary of the subcell where
    the level-set is positive.

    @param [in] Tr ElementTransformation of the current element
    @param [in] sir corresponding IntegrationRule on surface
    */
   void ComputeVolumeWeights2D(ElementTransformation& Tr,
                               const IntegrationRule* sir);

   /**
    @brief Compute 2D quadrature weights

    Compute the quadrature weights for the 3D surface quadrature rule by means
    of moment-fitting. To construct the quadrature rule, special integrals are
    reduced to integrals over the edges of the subcell where the level-set is
    positive.

    @param [in] Tr ElementTransformation of the current element
    */
   void ComputeSurfaceWeights3D(ElementTransformation& Tr);


   /**
    @brief Compute the 3D quadrature weights

    Compute the 3D quadrature weights for the volumetric subdomain quadrature
    rule by means of moment-fitting. To construct the quadrature rule, special
    integrals are reduced to integrals over the boundary of the subcell where
    the level-set is positive.

    @param [in] Tr ElementTransformation of the current element
    @param [in] sir corresponding IntegrationRule on surface
    */
   void ComputeVolumeWeights3D(ElementTransformation& Tr,
                               const IntegrationRule* sir);

   /// @brief A divergence free basis on the element [-1,1]^2
   void DivFreeBasis2D(const IntegrationPoint& ip, DenseMatrix& shape);
   /// @brief A orthogonalized divergence free basis on the element [-1,1]^2
   void OrthoBasis2D(const IntegrationPoint& ip, DenseMatrix& shape);
   /// @brief A orthogonalized divergence free basis on the element [-1,1]^3
   void OrthoBasis3D(const IntegrationPoint& ip, DenseMatrix& shape);
   /// @brief A step of the modified Gram-Schmidt algorithm
   void mGSStep(DenseMatrix& shape, DenseTensor& shapeMFN, int step);

   /// @brief Monomial basis on the element [-1,1]^2
   void Basis2D(const IntegrationPoint& ip, Vector& shape);
   /// @brief Antiderivatives of the monomial basis on the element [-1,1]^2
   void BasisAD2D(const IntegrationPoint& ip, DenseMatrix& shape);
   /// @brief Monomial basis on the element [-1,1]^3
   void Basis3D(const IntegrationPoint& ip, Vector& shape);
   /// @brief Antiderivatives of the monomial basis on the element [-1,1]^3
   void BasisAD3D(const IntegrationPoint& ip, DenseMatrix& shape);

public:

   /** @brief Constructor to set up the generated cut IntegrationRules.

       @param [in] order  Order of the constructed IntegrationRule.
       @param [in] lvlset Coefficient whose zero level set specifies the cut.
       @param [in] lsO    Polynomial degree for projecting the level-set
                          Coefficient to a GridFunction, which is used to
                          compute gradients and normals. */
   MomentFittingIntRules(int order, Coefficient& lvlset, int lsO)
      : CutIntegrationRules(order, lvlset, lsO),
        dim(-1), nBasis(-1), nBasisVolume(-1), VolumeSVD(nullptr)
   { FaceWeights.SetSize(1); FaceWeightsComp.SetSize(1); }

   /// Change the order of the constructed IntegrationRule.
   void SetOrder(int order) override;

   using CutIntegrationRules::SetLevelSetCoefficient;
   using CutIntegrationRules::SetLevelSetProjectionOrder;

   /**
    @brief Construct a cut-surface IntegrationRule.

    Construct an IntegrationRule to integrate on the surface given by the
    already specified level set function, for the element given by @a Tr.

    @param [in]  Tr     Specifies the IntegrationRule's associated mesh element.
    @param [out] result IntegrationRule on the cut-surface
   */
   void GetSurfaceIntegrationRule(ElementTransformation& Tr,
                                  IntegrationRule& result) override;

   /**
    @brief Construct a cut-volume IntegrationRule.

    Construct an IntegrationRule to integrate in the subdomain given by the
    positive values of the already specified level set function, for the element
    given by @a Tr.

    @param [in]  Tr     Specifies the IntegrationRule's associated mesh element.
    @param [out] result IntegrationRule for the cut-volume
    @param [in]  sir    Corresponding IntegrationRule for the surface, which can
                        be used to avoid computations.
   */
   void GetVolumeIntegrationRule(ElementTransformation& Tr,
                                 IntegrationRule& result,
                                 const IntegrationRule* sir = nullptr) override;

   /**
    @brief Compute transformation quadrature weights for surface integration.

    Compute the transformation weights for integration over the cut-surface in
    reference space.

    @param [in]  Tr      Specifies the IntegrationRule's associated element.
    @param [in]  sir     IntegrationRule defining the IntegrationPoints
    @param [out] weights Vector containing the transformation weights.
   */
   void GetSurfaceWeights(ElementTransformation &Tr,
                          const IntegrationRule &sir,
                          Vector &weights) override;

   /// @brief Destructor of MomentFittingIntRules
   ~MomentFittingIntRules() override { delete VolumeSVD; }
};

#endif //MFEM_USE_LAPACK

namespace DivFreeBasis
{

/// @brief 3 dimensional divergence free basis functions on [-1,1]^3
inline void GetDivFree3DBasis(const Vector& X, DenseMatrix& shape, int Order)
{
   int nBasis;
   if (Order == 0) { nBasis = 3; }
   else if (Order == 1) { nBasis = 11; }
   else if (Order == 2) { nBasis = 26; }
   else if (Order == 3) { nBasis = 50; }
   else if (Order == 4) { nBasis = 85; }
   else if (Order == 5) { nBasis = 133; }
   else if (Order == 6) { nBasis = 196; }
   else { nBasis = 276; Order = 7; }

   shape.SetSize(nBasis, 3);
   shape = 0.;

   real_t xi = X(0);
   real_t eta = X(1);
   real_t nu = X(2);

   switch (Order)
   {
      case 7:
         shape(196,0) = -2.995357736356377*nu + 26.958219627207395*pow(nu,3.)
                        - 59.30808317985627*pow(nu,5.)
                        + 36.714527682768164*pow(nu,7.);
         shape(197,1) = -2.995357736356377*nu + 26.958219627207395*pow(nu,3.)
                        - 59.30808317985627*pow(nu,5.)
                        + 36.714527682768164*pow(nu,7.);
         shape(198,0) = -0.689981317681863*eta
                        + 14.489607671319124*eta*pow(nu,2.)
                        - 43.46882301395737*eta*pow(nu,4.)
                        + 31.87713687690207*eta*pow(nu,6.);
         shape(199,1) = -0.689981317681863*eta
                        + 14.489607671319124*eta*pow(nu,2.)
                        - 43.46882301395737*eta*pow(nu,4.)
                        + 31.87713687690207*eta*pow(nu,6.);
         shape(199,2) = 0.689981317681863*nu - 4.829869223773041*pow(nu,3.)
                        + 8.693764602791473*pow(nu,5.)
                        - 4.553876696700296*pow(nu,7.);
         shape(200,0) = -2.4581457378987928*nu + 11.471346776861033*pow(nu,3.)
                        - 10.324212099174929*pow(nu,5.)
                        + 7.374437213696378*pow(eta,2.)*nu
                        - 34.4140403305831*pow(eta,2.)*pow(nu,3.)
                        + 30.972636297524787*pow(eta,2.)*pow(nu,5.);
         shape(201,1) = -2.4581457378987928*nu + 11.471346776861033*pow(nu,3.)
                        - 10.324212099174929*pow(nu,5.)
                        + 7.374437213696378*pow(eta,2.)*nu
                        - 34.4140403305831*pow(eta,2.)*pow(nu,3.)
                        + 30.972636297524787*pow(eta,2.)*pow(nu,5.);
         shape(201,2) = 0.49162914757975856*eta
                        - 7.374437213696378*eta*pow(nu,2.)
                        + 17.20702016529155*eta*pow(nu,4.)
                        - 10.324212099174929*eta*pow(nu,6.);
         shape(202,0) = -1.5785117100452566*eta
                        + 15.785117100452565*eta*pow(nu,2.)
                        - 18.415969950527995*eta*pow(nu,4.)
                        + 2.6308528500754274*pow(eta,3.)
                        - 26.308528500754274*pow(eta,3.)*pow(nu,2.)
                        + 30.69328325087999*pow(eta,3.)*pow(nu,4.);
         shape(203,1) = -1.5785117100452566*eta
                        + 15.785117100452565*eta*pow(nu,2.)
                        - 18.415969950527995*eta*pow(nu,4.)
                        + 2.6308528500754274*pow(eta,3.)
                        - 26.308528500754274*pow(eta,3.)*pow(nu,2.)
                        + 30.69328325087999*pow(eta,3.)*pow(nu,4.);
         shape(203,2) = 1.5785117100452566*nu - 5.261705700150855*pow(nu,3.)
                        + 3.6831939901055986*pow(nu,5.)
                        - 7.8925585502262825*pow(eta,2.)*nu
                        + 26.308528500754274*pow(eta,2.)*pow(nu,3.)
                        - 18.415969950527995*pow(eta,2.)*pow(nu,5.);
         shape(204,0) = -1.5785117100452566*nu + 2.6308528500754274*pow(nu,3.)
                        + 15.785117100452565*pow(eta,2.)*nu
                        - 26.308528500754274*pow(eta,2.)*pow(nu,3.)
                        - 18.415969950527995*pow(eta,4.)*nu
                        + 30.69328325087999*pow(eta,4.)*pow(nu,3.);
         shape(205,1) = -1.5785117100452566*nu + 2.6308528500754274*pow(nu,3.)
                        + 15.785117100452565*pow(eta,2.)*nu
                        - 26.308528500754274*pow(eta,2.)*pow(nu,3.)
                        - 18.415969950527995*pow(eta,4.)*nu
                        + 30.69328325087999*pow(eta,4.)*pow(nu,3.);
         shape(205,2) = 2.6308528500754274*eta
                        - 15.785117100452565*eta*pow(nu,2.)
                        + 13.154264250377137*eta*pow(nu,4.)
                        - 6.138656650175998*pow(eta,3.)
                        + 36.83193990105599*pow(eta,3.)*pow(nu,2.)
                        - 30.69328325087999*pow(eta,3.)*pow(nu,4.);
         shape(206,0) = -2.4581457378987928*eta
                        + 7.374437213696378*eta*pow(nu,2.)
                        + 11.471346776861033*pow(eta,3.)
                        - 34.4140403305831*pow(eta,3.)*pow(nu,2.)
                        - 10.324212099174929*pow(eta,5.)
                        + 30.972636297524787*pow(eta,5.)*pow(nu,2.);
         shape(207,1) = -2.4581457378987928*eta
                        + 7.374437213696378*eta*pow(nu,2.)
                        + 11.471346776861033*pow(eta,3.)
                        - 34.4140403305831*pow(eta,3.)*pow(nu,2.)
                        - 10.324212099174929*pow(eta,5.)
                        + 30.972636297524787*pow(eta,5.)*pow(nu,2.);
         shape(207,2) = 2.4581457378987928*nu - 2.4581457378987928*pow(nu,3.)
                        - 34.4140403305831*pow(eta,2.)*nu
                        + 34.4140403305831*pow(eta,2.)*pow(nu,3.)
                        + 51.621060495874644*pow(eta,4.)*nu
                        - 51.621060495874644*pow(eta,4.)*pow(nu,3.);
         shape(208,0) = -0.689981317681863*nu
                        + 14.489607671319124*pow(eta,2.)*nu
                        - 43.46882301395737*pow(eta,4.)*nu
                        + 31.87713687690207*pow(eta,6.)*nu;
         shape(209,1) = -0.689981317681863*nu
                        + 14.489607671319124*pow(eta,2.)*nu
                        - 43.46882301395737*pow(eta,4.)*nu
                        + 31.87713687690207*pow(eta,6.)*nu;
         shape(209,2) = 4.829869223773041*eta
                        - 14.489607671319124*eta*pow(nu,2.)
                        - 28.97921534263825*pow(eta,3.)
                        + 86.93764602791474*pow(eta,3.)*pow(nu,2.)
                        + 31.87713687690207*pow(eta,5.)
                        - 95.63141063070621*pow(eta,5.)*pow(nu,2.);
         shape(210,0) = -2.995357736356377*eta + 26.958219627207395*pow(eta,3.)
                        - 59.30808317985627*pow(eta,5.)
                        + 36.714527682768164*pow(eta,7.);
         shape(211,1) = -2.995357736356377*eta + 26.958219627207395*pow(eta,3.)
                        - 59.30808317985627*pow(eta,5.)
                        + 36.714527682768164*pow(eta,7.);
         shape(211,2) = 2.995357736356377*nu
                        - 80.87465888162218*pow(eta,2.)*nu
                        + 296.54041589928136*pow(eta,4.)*nu
                        - 257.00169377937715*pow(eta,6.)*nu;
         shape(212,2) = -2.995357736356377*eta + 26.958219627207395*pow(eta,3.)
                        - 59.30808317985627*pow(eta,5.)
                        + 36.714527682768164*pow(eta,7.);
         shape(213,0) = -0.689981317681863*xi
                        + 14.489607671319124*xi*pow(nu,2.)
                        - 43.46882301395737*xi*pow(nu,4.)
                        + 31.87713687690207*xi*pow(nu,6.);
         shape(213,2) = 0.689981317681863*nu - 4.829869223773041*pow(nu,3.)
                        + 8.693764602791473*pow(nu,5.)
                        - 4.553876696700296*pow(nu,7.);
         shape(214,1) = -0.689981317681863*xi
                        + 14.489607671319124*xi*pow(nu,2.)
                        - 43.46882301395737*xi*pow(nu,4.)
                        + 31.87713687690207*xi*pow(nu,6.);
         shape(215,0) = 6.595897162251698*xi*eta*nu
                        - 30.780853423841258*xi*eta*pow(nu,3.)
                        + 27.70276808145713*xi*eta*pow(nu,5.);
         shape(215,2) = 0.21986323874172325*eta
                        - 3.297948581125849*eta*pow(nu,2.)
                        + 7.695213355960314*eta*pow(nu,4.)
                        - 4.617128013576188*eta*pow(nu,6.);
         shape(216,1) = 6.595897162251698*xi*eta*nu
                        - 30.780853423841258*xi*eta*pow(nu,3.)
                        + 27.70276808145713*xi*eta*pow(nu,5.);
         shape(216,2) = 0.21986323874172325*xi
                        - 3.297948581125849*xi*pow(nu,2.)
                        + 7.695213355960314*xi*pow(nu,4.)
                        - 4.617128013576188*xi*pow(nu,6.);
         shape(217,0) = -0.7702348464916399*xi
                        + 7.7023484649163985*xi*pow(nu,2.)
                        - 8.986073209069131*xi*pow(nu,4.)
                        + 2.3107045394749197*xi*pow(eta,2.)
                        - 23.107045394749196*xi*pow(eta,2.)*pow(nu,2.)
                        + 26.958219627207395*xi*pow(eta,2.)*pow(nu,4.);
         shape(217,2) = 0.7702348464916399*nu - 2.567449488305466*pow(nu,3.)
                        + 1.7972146418138264*pow(nu,5.)
                        - 2.3107045394749197*pow(eta,2.)*nu
                        + 7.7023484649163985*pow(eta,2.)*pow(nu,3.)
                        - 5.391643925441479*pow(eta,2.)*pow(nu,5.);
         shape(218,1) = -0.7702348464916399*xi
                        + 7.7023484649163985*xi*pow(nu,2.)
                        - 8.986073209069131*xi*pow(nu,4.)
                        + 2.3107045394749197*xi*pow(eta,2.)
                        - 23.107045394749196*xi*pow(eta,2.)*pow(nu,2.)
                        + 26.958219627207395*xi*pow(eta,2.)*pow(nu,4.);
         shape(218,2) = -4.621409078949839*xi*eta*nu
                        + 15.404696929832797*xi*eta*pow(nu,3.)
                        - 10.783287850882958*xi*eta*pow(nu,5.);
         shape(219,0) = 9.644865862208764*xi*eta*nu
                        - 16.074776437014606*xi*eta*pow(nu,3.)
                        - 16.074776437014606*xi*pow(eta,3.)*nu
                        + 26.79129406169101*xi*pow(eta,3.)*pow(nu,3.);
         shape(219,2) = 0.8037388218507303*eta
                        - 4.822432931104382*eta*pow(nu,2.)
                        + 4.0186941092536514*eta*pow(nu,4.)
                        - 1.3395647030845506*pow(eta,3.)
                        + 8.037388218507303*pow(eta,3.)*pow(nu,2.)
                        - 6.697823515422753*pow(eta,3.)*pow(nu,4.);
         shape(220,1) = 9.644865862208764*xi*eta*nu
                        - 16.074776437014606*xi*eta*pow(nu,3.)
                        - 16.074776437014606*xi*pow(eta,3.)*nu
                        + 26.79129406169101*xi*pow(eta,3.)*pow(nu,3.);
         shape(220,2) = 0.8037388218507303*xi
                        - 4.822432931104382*xi*pow(nu,2.)
                        + 4.0186941092536514*xi*pow(nu,4.)
                        - 4.0186941092536514*xi*pow(eta,2.)
                        + 24.11216465552191*xi*pow(eta,2.)*pow(nu,2.)
                        - 20.093470546268257*xi*pow(eta,2.)*pow(nu,4.);
         shape(221,0) = -0.7702348464916399*xi
                        + 2.3107045394749197*xi*pow(nu,2.)
                        + 7.7023484649163985*xi*pow(eta,2.)
                        - 23.107045394749196*xi*pow(eta,2.)*pow(nu,2.)
                        - 8.986073209069131*xi*pow(eta,4.)
                        + 26.958219627207395*xi*pow(eta,4.)*pow(nu,2.);
         shape(221,2) = 0.7702348464916399*nu - 0.7702348464916399*pow(nu,3.)
                        - 7.7023484649163985*pow(eta,2.)*nu
                        + 7.7023484649163985*pow(eta,2.)*pow(nu,3.)
                        + 8.986073209069131*pow(eta,4.)*nu
                        - 8.986073209069131*pow(eta,4.)*pow(nu,3.);
         shape(222,1) = -0.7702348464916399*xi
                        + 2.3107045394749197*xi*pow(nu,2.)
                        + 7.7023484649163985*xi*pow(eta,2.)
                        - 23.107045394749196*xi*pow(eta,2.)*pow(nu,2.)
                        - 8.986073209069131*xi*pow(eta,4.)
                        + 26.958219627207395*xi*pow(eta,4.)*pow(nu,2.);
         shape(222,2) = -15.404696929832797*xi*eta*nu
                        + 15.404696929832797*xi*eta*pow(nu,3.)
                        + 35.944292836276524*xi*pow(eta,3.)*nu
                        - 35.944292836276524*xi*pow(eta,3.)*pow(nu,3.);
         shape(223,0) = 6.595897162251698*xi*eta*nu
                        - 30.780853423841258*xi*pow(eta,3.)*nu
                        + 27.70276808145713*xi*pow(eta,5.)*nu;
         shape(223,2) = 1.0993161937086162*eta
                        - 3.297948581125849*eta*pow(nu,2.)
                        - 5.130142237306876*pow(eta,3.)
                        + 15.390426711920629*pow(eta,3.)*pow(nu,2.)
                        + 4.617128013576188*pow(eta,5.)
                        - 13.851384040728565*pow(eta,5.)*pow(nu,2.);
         shape(224,1) = 6.595897162251698*xi*eta*nu
                        - 30.780853423841258*xi*pow(eta,3.)*nu
                        + 27.70276808145713*xi*pow(eta,5.)*nu;
         shape(224,2) = 1.0993161937086162*xi
                        - 3.297948581125849*xi*pow(nu,2.)
                        - 15.390426711920629*xi*pow(eta,2.)
                        + 46.17128013576188*xi*pow(eta,2.)*pow(nu,2.)
                        + 23.08564006788094*xi*pow(eta,4.)
                        - 69.25692020364282*xi*pow(eta,4.)*pow(nu,2.);
         shape(225,0) = -0.689981317681863*xi
                        + 14.489607671319124*xi*pow(eta,2.)
                        - 43.46882301395737*xi*pow(eta,4.)
                        + 31.87713687690207*xi*pow(eta,6.);
         shape(225,2) = 0.689981317681863*nu
                        - 14.489607671319124*pow(eta,2.)*nu
                        + 43.46882301395737*pow(eta,4.)*nu
                        - 31.87713687690207*pow(eta,6.)*nu;
         shape(226,1) = -0.689981317681863*xi
                        + 14.489607671319124*xi*pow(eta,2.)
                        - 43.46882301395737*xi*pow(eta,4.)
                        + 31.87713687690207*xi*pow(eta,6.);
         shape(226,2) = -28.97921534263825*xi*eta*nu
                        + 173.87529205582948*xi*pow(eta,3.)*nu
                        - 191.26282126141243*xi*pow(eta,5.)*nu;
         shape(227,2) = -0.689981317681863*xi
                        + 14.489607671319124*xi*pow(eta,2.)
                        - 43.46882301395737*xi*pow(eta,4.)
                        + 31.87713687690207*xi*pow(eta,6.);
         shape(228,0) = -2.4581457378987928*nu + 11.471346776861033*pow(nu,3.)
                        - 10.324212099174929*pow(nu,5.)
                        + 7.374437213696378*pow(xi,2.)*nu
                        - 34.4140403305831*pow(xi,2.)*pow(nu,3.)
                        + 30.972636297524787*pow(xi,2.)*pow(nu,5.);
         shape(228,2) = 0.49162914757975856*xi
                        - 7.374437213696378*xi*pow(nu,2.)
                        + 17.20702016529155*xi*pow(nu,4.)
                        - 10.324212099174929*xi*pow(nu,6.);
         shape(229,1) = 2.4581457378987928*nu + 11.471346776861033*pow(nu,3.)
                        - 10.324212099174929*pow(nu,5.)
                        + 7.374437213696378*pow(xi,2.)*nu
                        - 34.4140403305831*pow(xi,2.)*pow(nu,3.)
                        + 30.972636297524787*pow(xi,2.)*pow(nu,5.);
         shape(230,0) = -0.7702348464916399*eta
                        + 7.7023484649163985*eta*pow(nu,2.)
                        - 8.986073209069131*eta*pow(nu,4.)
                        + 2.3107045394749197*pow(xi,2.)*eta
                        - 23.107045394749196*pow(xi,2.)*eta*pow(nu,2.)
                        + 26.958219627207395*pow(xi,2.)*eta*pow(nu,4.);
         shape(230,2) = -4.621409078949839*xi*eta*nu
                        + 15.404696929832797*xi*eta*pow(nu,3.)
                        - 10.783287850882958*xi*eta*pow(nu,5.);
         shape(231,1) = -0.7702348464916399*eta
                        + 7.7023484649163985*eta*pow(nu,2.)
                        - 8.986073209069131*eta*pow(nu,4.)
                        + 2.3107045394749197*pow(xi,2.)*eta
                        - 23.107045394749196*pow(xi,2.)*eta*pow(nu,2.)
                        + 26.958219627207395*pow(xi,2.)*eta*pow(nu,4.);
         shape(231,2) = 0.7702348464916399*nu - 2.567449488305466*pow(nu,3.)
                        + 1.7972146418138264*pow(nu,5.)
                        - 2.3107045394749197*pow(xi,2.)*nu
                        + 7.7023484649163985*pow(xi,2.)*pow(nu,3.)
                        - 5.391643925441479*pow(xi,2.)*pow(nu,5.);
         shape(232,0) = -1.753901900050285*nu + 2.9231698334171416*pow(nu,3.)
                        + 5.261705700150855*pow(eta,2.)*nu
                        - 8.769509500251425*pow(eta,2.)*pow(nu,3.)
                        + 5.261705700150855*pow(xi,2.)*nu
                        - 8.769509500251425*pow(xi,2.)*pow(nu,3.)
                        - 15.785117100452565*pow(xi,2.)*pow(eta,2.)*nu
                        + 26.308528500754274*pow(xi,2.)*pow(eta,2.)*pow(nu,3.);
         shape(232,2) = 0.8769509500251426*xi
                        - 5.261705700150855*xi*pow(nu,2.)
                        + 4.384754750125713*xi*pow(nu,4.)
                        - 2.6308528500754274*xi*pow(eta,2.)
                        + 15.785117100452565*xi*pow(eta,2.)*pow(nu,2.)
                        - 13.154264250377137*xi*pow(eta,2.)*pow(nu,4.);
         shape(233,1) = 1.753901900050285*nu + 2.9231698334171416*pow(nu,3.)
                        + 5.261705700150855*pow(eta,2.)*nu
                        - 8.769509500251425*pow(eta,2.)*pow(nu,3.)
                        + 5.261705700150855*pow(xi,2.)*nu
                        - 8.769509500251425*pow(xi,2.)*pow(nu,3.)
                        - 15.785117100452565*pow(xi,2.)*pow(eta,2.)*nu
                        + 26.308528500754274*pow(xi,2.)*pow(eta,2.)*pow(nu,3.);
         shape(233,2) = 0.8769509500251426*eta
                        - 5.261705700150855*eta*pow(nu,2.)
                        + 4.384754750125713*eta*pow(nu,4.)
                        - 2.6308528500754274*pow(xi,2.)*eta
                        + 15.785117100452565*pow(xi,2.)*eta*pow(nu,2.)
                        - 13.154264250377137*pow(xi,2.)*eta*pow(nu,4.);
         shape(234,0) = -1.753901900050285*eta
                        + 5.261705700150855*eta*pow(nu,2.)
                        + 2.9231698334171416*pow(eta,3.)
                        - 8.769509500251425*pow(eta,3.)*pow(nu,2.)
                        + 5.261705700150855*pow(xi,2.)*eta
                        - 15.785117100452565*pow(xi,2.)*eta*pow(nu,2.)
                        - 8.769509500251425*pow(xi,2.)*pow(eta,3.)
                        + 26.308528500754274*pow(xi,2.)*pow(eta,3.)*pow(nu,2.);
         shape(234,2) = -10.52341140030171*xi*eta*nu
                        + 10.52341140030171*xi*eta*pow(nu,3.)
                        + 17.53901900050285*xi*pow(eta,3.)*nu
                        - 17.53901900050285*xi*pow(eta,3.)*pow(nu,3.);
         shape(235,1) = -1.753901900050285*eta
                        + 5.261705700150855*eta*pow(nu,2.)
                        + 2.9231698334171416*pow(eta,3.)
                        - 8.769509500251425*pow(eta,3.)*pow(nu,2.)
                        + 5.261705700150855*pow(xi,2.)*eta
                        - 15.785117100452565*pow(xi,2.)*eta*pow(nu,2.)
                        - 8.769509500251425*pow(xi,2.)*pow(eta,3.)
                        + 26.308528500754274*pow(xi,2.)*pow(eta,3.)*pow(nu,2.);
         shape(235,2) = 1.753901900050285*nu - 1.753901900050285*pow(nu,3.)
                        - 8.769509500251425*pow(eta,2.)*nu
                        + 8.769509500251425*pow(eta,2.)*pow(nu,3.)
                        - 5.261705700150855*pow(xi,2.)*nu
                        + 5.261705700150855*pow(xi,2.)*pow(nu,3.)
                        + 26.308528500754274*pow(xi,2.)*pow(eta,2.)*nu
                        - 26.308528500754274*pow(xi,2.)*pow(eta,2.)*pow(nu,3.);
         shape(236,0) = -0.7702348464916399*nu
                        + 7.7023484649163985*pow(eta,2.)*nu
                        - 8.986073209069131*pow(eta,4.)*nu
                        + 2.3107045394749197*pow(xi,2.)*nu
                        - 23.107045394749196*pow(xi,2.)*pow(eta,2.)*nu
                        + 26.958219627207395*pow(xi,2.)*pow(eta,4.)*nu;
         shape(236,2) = 0.7702348464916399*xi
                        - 2.3107045394749197*xi*pow(nu,2.)
                        - 7.7023484649163985*xi*pow(eta,2.)
                        + 23.107045394749196*xi*pow(eta,2.)*pow(nu,2.)
                        + 8.986073209069131*xi*pow(eta,4.)
                        - 26.958219627207395*xi*pow(eta,4.)*pow(nu,2.);
         shape(237,1) = -0.7702348464916399*nu
                        + 7.7023484649163985*pow(eta,2.)*nu
                        - 8.986073209069131*pow(eta,4.)*nu
                        + 2.3107045394749197*pow(xi,2.)*nu
                        - 23.107045394749196*pow(xi,2.)*pow(eta,2.)*nu
                        + 26.958219627207395*pow(xi,2.)*pow(eta,4.)*nu;
         shape(237,2) = 2.567449488305466*eta
                        - 7.7023484649163985*eta*pow(nu,2.)
                        - 5.990715472712754*pow(eta,3.)
                        + 17.972146418138262*pow(eta,3.)*pow(nu,2.)
                        - 7.7023484649163985*pow(xi,2.)*eta
                        + 23.107045394749196*pow(xi,2.)*eta*pow(nu,2.)
                        + 17.972146418138262*pow(xi,2.)*pow(eta,3.)
                        - 53.91643925441479*pow(xi,2.)*pow(eta,3.)*pow(nu,2.);
         shape(238,0) = -2.4581457378987928*eta
                        + 11.471346776861033*pow(eta,3.)
                        - 10.324212099174929*pow(eta,5.)
                        + 7.374437213696378*pow(xi,2.)*eta
                        - 34.4140403305831*pow(xi,2.)*pow(eta,3.)
                        + 30.972636297524787*pow(xi,2.)*pow(eta,5.);
         shape(238,2) = -14.748874427392757*xi*eta*nu
                        + 68.8280806611662*xi*pow(eta,3.)*nu
                        - 61.94527259504957*xi*pow(eta,5.)*nu;
         shape(239,1) = -2.4581457378987928*eta
                        + 11.471346776861033*pow(eta,3.)
                        - 10.324212099174929*pow(eta,5.)
                        + 7.374437213696378*pow(xi,2.)*eta
                        - 34.4140403305831*pow(xi,2.)*pow(eta,3.)
                        + 30.972636297524787*pow(xi,2.)*pow(eta,5.);
         shape(239,2) = 2.4581457378987928*nu
                        - 34.4140403305831*pow(eta,2.)*nu
                        + 51.621060495874644*pow(eta,4.)*nu
                        - 7.374437213696378*pow(xi,2.)*nu
                        + 103.24212099174929*pow(xi,2.)*pow(eta,2.)*nu
                        - 154.86318148762393*pow(xi,2.)*pow(eta,4.)*nu;
         shape(240,2) = -2.4581457378987928*eta
                        + 11.471346776861033*pow(eta,3.)
                        - 10.324212099174929*pow(eta,5.)
                        + 7.374437213696378*pow(xi,2.)*eta
                        - 34.4140403305831*pow(xi,2.)*pow(eta,3.)
                        + 30.972636297524787*pow(xi,2.)*pow(eta,5.);
         shape(241,0) = -1.5785117100452566*xi
                        + 15.785117100452565*xi*pow(nu,2.)
                        - 18.415969950527995*xi*pow(nu,4.)
                        + 2.6308528500754274*pow(xi,3.)
                        - 26.308528500754274*pow(xi,3.)*pow(nu,2.)
                        + 30.69328325087999*pow(xi,3.)*pow(nu,4.);
         shape(241,2) = 1.5785117100452566*nu - 5.261705700150855*pow(nu,3.)
                        + 3.6831939901055986*pow(nu,5.)
                        - 7.8925585502262825*pow(xi,2.)*nu
                        + 26.308528500754274*pow(xi,2.)*pow(nu,3.)
                        - 18.415969950527995*pow(xi,2.)*pow(nu,5.);
         shape(242,1) = -1.5785117100452566*xi
                        + 15.785117100452565*xi*pow(nu,2.)
                        - 18.415969950527995*xi*pow(nu,4.)
                        + 2.6308528500754274*pow(xi,3.)
                        - 26.308528500754274*pow(xi,3.)*pow(nu,2.)
                        + 30.69328325087999*pow(xi,3.)*pow(nu,4.);
         shape(243,0) = 9.644865862208764*xi*eta*nu
                        - 16.074776437014606*xi*eta*pow(nu,3.)
                        - 16.074776437014606*pow(xi,3.)*eta*nu
                        + 26.79129406169101*pow(xi,3.)*eta*pow(nu,3.);
         shape(243,2) = 0.8037388218507303*eta
                        - 4.822432931104382*eta*pow(nu,2.)
                        + 4.0186941092536514*eta*pow(nu,4.)
                        - 4.0186941092536514*pow(xi,2.)*eta
                        + 24.11216465552191*pow(xi,2.)*eta*pow(nu,2.)
                        - 20.093470546268257*pow(xi,2.)*eta*pow(nu,4.);
         shape(244,1) = 9.644865862208764*xi*eta*nu
                        - 16.074776437014606*xi*eta*pow(nu,3.)
                        - 16.074776437014606*pow(xi,3.)*eta*nu
                        + 26.79129406169101*pow(xi,3.)*eta*pow(nu,3.);
         shape(244,2) = 0.8037388218507303*xi
                        - 4.822432931104382*xi*pow(nu,2.)
                        + 4.0186941092536514*xi*pow(nu,4.)
                        - 1.3395647030845506*pow(xi,3.)
                        + 8.037388218507303*pow(xi,3.)*pow(nu,2.)
                        - 6.697823515422753*pow(xi,3.)*pow(nu,4.);
         shape(245,0) = -1.753901900050285*xi + 5.261705700150855*xi*pow(nu,2.)
                        + 5.261705700150855*xi*pow(eta,2.)
                        - 15.785117100452565*xi*pow(eta,2.)*pow(nu,2.)
                        + 2.9231698334171416*pow(xi,3.)
                        - 8.769509500251425*pow(xi,3.)*pow(nu,2.)
                        - 8.769509500251425*pow(xi,3.)*pow(eta,2.)
                        + 26.308528500754274*pow(xi,3.)*pow(eta,2.)*pow(nu,2.);
         shape(245,2) = 1.753901900050285*nu - 1.753901900050285*pow(nu,3.)
                        - 5.261705700150855*pow(eta,2.)*nu
                        + 5.261705700150855*pow(eta,2.)*pow(nu,3.)
                        - 8.769509500251425*pow(xi,2.)*nu
                        + 8.769509500251425*pow(xi,2.)*pow(nu,3.)
                        + 26.308528500754274*pow(xi,2.)*pow(eta,2.)*nu
                        - 26.308528500754274*pow(xi,2.)*pow(eta,2.)*pow(nu,3.);
         shape(246,1) = -1.753901900050285*xi + 5.261705700150855*xi*pow(nu,2.)
                        + 5.261705700150855*xi*pow(eta,2.)
                        - 15.785117100452565*xi*pow(eta,2.)*pow(nu,2.)
                        + 2.9231698334171416*pow(xi,3.)
                        - 8.769509500251425*pow(xi,3.)*pow(nu,2.)
                        - 8.769509500251425*pow(xi,3.)*pow(eta,2.)
                        + 26.308528500754274*pow(xi,3.)*pow(eta,2.)*pow(nu,2.);
         shape(246,2) = -10.52341140030171*xi*eta*nu
                        + 10.52341140030171*xi*eta*pow(nu,3.)
                        + 17.53901900050285*pow(xi,3.)*eta*nu
                        - 17.53901900050285*pow(xi,3.)*eta*pow(nu,3.);
         shape(247,0) = 9.644865862208764*xi*eta*nu
                        - 16.074776437014606*xi*pow(eta,3.)*nu
                        - 16.074776437014606*pow(xi,3.)*eta*nu
                        + 26.79129406169101*pow(xi,3.)*pow(eta,3.)*nu;
         shape(247,2) = 1.6074776437014606*eta
                        - 4.822432931104382*eta*pow(nu,2.)
                        - 2.679129406169101*pow(eta,3.)
                        + 8.037388218507303*pow(eta,3.)*pow(nu,2.)
                        - 8.037388218507303*pow(xi,2.)*eta
                        + 24.11216465552191*pow(xi,2.)*eta*pow(nu,2.)
                        + 13.395647030845504*pow(xi,2.)*pow(eta,3.)
                        - 40.186941092536514*pow(xi,2.)*pow(eta,3.)*pow(nu,2.);
         shape(248,1) = 9.644865862208764*xi*eta*nu
                        - 16.074776437014606*xi*pow(eta,3.)*nu
                        - 16.074776437014606*pow(xi,3.)*eta*nu
                        + 26.79129406169101*pow(xi,3.)*pow(eta,3.)*nu;
         shape(248,2) = 1.6074776437014606*xi
                        - 4.822432931104382*xi*pow(nu,2.)
                        - 8.037388218507303*xi*pow(eta,2.)
                        + 24.11216465552191*xi*pow(eta,2.)*pow(nu,2.)
                        - 2.679129406169101*pow(xi,3.)
                        + 8.037388218507303*pow(xi,3.)*pow(nu,2.)
                        + 13.395647030845504*pow(xi,3.)*pow(eta,2.)
                        - 40.186941092536514*pow(xi,3.)*pow(eta,2.)*pow(nu,2.);
         shape(249,0) = -1.5785117100452566*xi
                        + 15.785117100452565*xi*pow(eta,2.)
                        - 18.415969950527995*xi*pow(eta,4.)
                        + 2.6308528500754274*pow(xi,3.)
                        - 26.308528500754274*pow(xi,3.)*pow(eta,2.)
                        + 30.69328325087999*pow(xi,3.)*pow(eta,4.);
         shape(249,2) = 1.5785117100452566*nu
                        - 15.785117100452565*pow(eta,2.)*nu
                        + 18.415969950527995*pow(eta,4.)*nu
                        - 7.8925585502262825*pow(xi,2.)*nu
                        + 78.92558550226282*pow(xi,2.)*pow(eta,2.)*nu
                        - 92.07984975263996*pow(xi,2.)*pow(eta,4.)*nu;
         shape(250,1) = -1.5785117100452566*xi
                        + 15.785117100452565*xi*pow(eta,2.)
                        - 18.415969950527995*xi*pow(eta,4.)
                        + 2.6308528500754274*pow(xi,3.)
                        - 26.308528500754274*pow(xi,3.)*pow(eta,2.)
                        + 30.69328325087999*pow(xi,3.)*pow(eta,4.);
         shape(250,2) = -31.57023420090513*xi*eta*nu
                        + 73.66387980211196*xi*pow(eta,3.)*nu
                        + 52.61705700150855*pow(xi,3.)*eta*nu
                        - 122.77313300351994*pow(xi,3.)*pow(eta,3.)*nu;
         shape(251,2) = -1.5785117100452566*xi
                        + 15.785117100452565*xi*pow(eta,2.)
                        - 18.415969950527995*xi*pow(eta,4.)
                        + 2.6308528500754274*pow(xi,3.)
                        - 26.308528500754274*pow(xi,3.)*pow(eta,2.)
                        + 30.69328325087999*pow(xi,3.)*pow(eta,4.);
         shape(252,0) = -1.5785117100452566*nu + 2.6308528500754274*pow(nu,3.)
                        + 15.785117100452565*pow(xi,2.)*nu
                        - 26.308528500754274*pow(xi,2.)*pow(nu,3.)
                        - 18.415969950527995*pow(xi,4.)*nu
                        + 30.69328325087999*pow(xi,4.)*pow(nu,3.);
         shape(252,2) = 2.6308528500754274*xi
                        - 15.785117100452565*xi*pow(nu,2.)
                        + 13.154264250377137*xi*pow(nu,4.)
                        - 6.138656650175998*pow(xi,3.)
                        + 36.83193990105599*pow(xi,3.)*pow(nu,2.)
                        - 30.69328325087999*pow(xi,3.)*pow(nu,4.);
         shape(253,1) = -1.5785117100452566*nu + 2.6308528500754274*pow(nu,3.)
                        + 15.785117100452565*pow(xi,2.)*nu
                        - 26.308528500754274*pow(xi,2.)*pow(nu,3.)
                        - 18.415969950527995*pow(xi,4.)*nu
                        + 30.69328325087999*pow(xi,4.)*pow(nu,3.);
         shape(254,0) = -0.7702348464916399*eta
                        + 2.3107045394749197*eta*pow(nu,2.)
                        + 7.7023484649163985*pow(xi,2.)*eta
                        - 23.107045394749196*pow(xi,2.)*eta*pow(nu,2.)
                        - 8.986073209069131*pow(xi,4.)*eta
                        + 26.958219627207395*pow(xi,4.)*eta*pow(nu,2.);
         shape(254,2) = -15.404696929832797*xi*eta*nu
                        + 15.404696929832797*xi*eta*pow(nu,3.)
                        + 35.944292836276524*pow(xi,3.)*eta*nu
                        - 35.944292836276524*pow(xi,3.)*eta*pow(nu,3.);
         shape(255,1) = -0.7702348464916399*eta
                        + 2.3107045394749197*eta*pow(nu,2.)
                        + 7.7023484649163985*pow(xi,2.)*eta
                        - 23.107045394749196*pow(xi,2.)*eta*pow(nu,2.)
                        - 8.986073209069131*pow(xi,4.)*eta
                        + 26.958219627207395*pow(xi,4.)*eta*pow(nu,2.);
         shape(255,2) = 0.7702348464916399*nu - 0.7702348464916399*pow(nu,3.)
                        - 7.7023484649163985*pow(xi,2.)*nu
                        + 7.7023484649163985*pow(xi,2.)*pow(nu,3.)
                        + 8.986073209069131*pow(xi,4.)*nu
                        - 8.986073209069131*pow(xi,4.)*pow(nu,3.);
         shape(256,0) = -0.7702348464916399*nu
                        + 2.3107045394749197*pow(eta,2.)*nu
                        + 7.7023484649163985*pow(xi,2.)*nu
                        - 23.107045394749196*pow(xi,2.)*pow(eta,2.)*nu
                        - 8.986073209069131*pow(xi,4.)*nu
                        + 26.958219627207395*pow(xi,4.)*pow(eta,2.)*nu;
         shape(256,2) = 2.567449488305466*xi
                        - 7.7023484649163985*xi*pow(nu,2.)
                        - 7.7023484649163985*xi*pow(eta,2.)
                        + 23.107045394749196*xi*pow(eta,2.)*pow(nu,2.)
                        - 5.990715472712754*pow(xi,3.)
                        + 17.972146418138262*pow(xi,3.)*pow(nu,2.)
                        + 17.972146418138262*pow(xi,3.)*pow(eta,2.)
                        - 53.91643925441479*pow(xi,3.)*pow(eta,2.)*pow(nu,2.);
         shape(257,1) = -0.7702348464916399*nu
                        + 2.3107045394749197*pow(eta,2.)*nu
                        + 7.7023484649163985*pow(xi,2.)*nu
                        - 23.107045394749196*pow(xi,2.)*pow(eta,2.)*nu
                        - 8.986073209069131*pow(xi,4.)*nu
                        + 26.958219627207395*pow(xi,4.)*pow(eta,2.)*nu;
         shape(257,2) = 0.7702348464916399*eta
                        - 2.3107045394749197*eta*pow(nu,2.)
                        - 7.7023484649163985*pow(xi,2.)*eta
                        + 23.107045394749196*pow(xi,2.)*eta*pow(nu,2.)
                        + 8.986073209069131*pow(xi,4.)*eta
                        - 26.958219627207395*pow(xi,4.)*eta*pow(nu,2.);
         shape(258,0) = -1.5785117100452566*eta
                        + 2.6308528500754274*pow(eta,3.)
                        + 15.785117100452565*pow(xi,2.)*eta
                        - 26.308528500754274*pow(xi,2.)*pow(eta,3.)
                        - 18.415969950527995*pow(xi,4.)*eta
                        + 30.69328325087999*pow(xi,4.)*pow(eta,3.);
         shape(258,2) = -31.57023420090513*xi*eta*nu
                        + 52.61705700150855*xi*pow(eta,3.)*nu
                        + 73.66387980211196*pow(xi,3.)*eta*nu
                        - 122.77313300351994*pow(xi,3.)*pow(eta,3.)*nu;
         shape(259,1) = -1.5785117100452566*eta
                        + 2.6308528500754274*pow(eta,3.)
                        + 15.785117100452565*pow(xi,2.)*eta
                        - 26.308528500754274*pow(xi,2.)*pow(eta,3.)
                        - 18.415969950527995*pow(xi,4.)*eta
                        + 30.69328325087999*pow(xi,4.)*pow(eta,3.);
         shape(259,2) = 1.5785117100452566*nu
                        - 7.8925585502262825*pow(eta,2.)*nu
                        - 15.785117100452565*pow(xi,2.)*nu
                        + 78.92558550226282*pow(xi,2.)*pow(eta,2.)*nu
                        + 18.415969950527995*pow(xi,4.)*nu
                        - 92.07984975263996*pow(xi,4.)*pow(eta,2.)*nu;
         shape(260,2) = -1.5785117100452566*eta
                        + 2.6308528500754274*pow(eta,3.)
                        + 15.785117100452565*pow(xi,2.)*eta
                        - 26.308528500754274*pow(xi,2.)*pow(eta,3.)
                        - 18.415969950527995*pow(xi,4.)*eta
                        + 30.69328325087999*pow(xi,4.)*pow(eta,3.);
         shape(261,0) = -2.4581457378987928*xi
                        + 7.374437213696378*xi*pow(nu,2.)
                        + 11.471346776861033*pow(xi,3.)
                        - 34.4140403305831*pow(xi,3.)*pow(nu,2.)
                        - 10.324212099174929*pow(xi,5.)
                        + 30.972636297524787*pow(xi,5.)*pow(nu,2.);
         shape(261,2) = 2.4581457378987928*nu - 2.4581457378987928*pow(nu,3.)
                        - 34.4140403305831*pow(xi,2.)*nu
                        + 34.4140403305831*pow(xi,2.)*pow(nu,3.)
                        + 51.621060495874644*pow(xi,4.)*nu
                        - 51.621060495874644*pow(xi,4.)*pow(nu,3.);
         shape(262,1) = -2.4581457378987928*xi
                        + 7.374437213696378*xi*pow(nu,2.)
                        + 11.471346776861033*pow(xi,3.)
                        - 34.4140403305831*pow(xi,3.)*pow(nu,2.)
                        - 10.324212099174929*pow(xi,5.)
                        + 30.972636297524787*pow(xi,5.)*pow(nu,2.);
         shape(263,0) = 6.595897162251698*xi*eta*nu
                        - 30.780853423841258*pow(xi,3.)*eta*nu
                        + 27.70276808145713*pow(xi,5.)*eta*nu;
         shape(263,2) = 1.0993161937086162*eta
                        - 3.297948581125849*eta*pow(nu,2.)
                        - 15.390426711920629*pow(xi,2.)*eta
                        + 46.17128013576188*pow(xi,2.)*eta*pow(nu,2.)
                        + 23.08564006788094*pow(xi,4.)*eta
                        - 69.25692020364282*pow(xi,4.)*eta*pow(nu,2.);
         shape(264,1) = 6.595897162251698*xi*eta*nu
                        - 30.780853423841258*pow(xi,3.)*eta*nu
                        + 27.70276808145713*pow(xi,5.)*eta*nu;
         shape(264,2) = 1.0993161937086162*xi
                        - 3.297948581125849*xi*pow(nu,2.)
                        - 5.130142237306876*pow(xi,3.)
                        + 15.390426711920629*pow(xi,3.)*pow(nu,2.)
                        + 4.617128013576188*pow(xi,5.)
                        - 13.851384040728565*pow(xi,5.)*pow(nu,2.);
         shape(265,0) = -2.4581457378987928*xi
                        + 7.374437213696378*xi*pow(eta,2.)
                        + 11.471346776861033*pow(xi,3.)
                        - 34.4140403305831*pow(xi,3.)*pow(eta,2.)
                        - 10.324212099174929*pow(xi,5.)
                        + 30.972636297524787*pow(xi,5.)*pow(eta,2.);
         shape(265,2) = 2.4581457378987928*nu
                        - 7.374437213696378*pow(eta,2.)*nu
                        - 34.4140403305831*pow(xi,2.)*nu
                        + 103.24212099174929*pow(xi,2.)*pow(eta,2.)*nu
                        + 51.621060495874644*pow(xi,4.)*nu
                        - 154.86318148762393*pow(xi,4.)*pow(eta,2.)*nu;
         shape(266,1) = -2.4581457378987928*xi
                        + 7.374437213696378*xi*pow(eta,2.)
                        + 11.471346776861033*pow(xi,3.)
                        - 34.4140403305831*pow(xi,3.)*pow(eta,2.)
                        - 10.324212099174929*pow(xi,5.)
                        + 30.972636297524787*pow(xi,5.)*pow(eta,2.);
         shape(266,2) = -14.748874427392757*xi*eta*nu
                        + 68.8280806611662*pow(xi,3.)*eta*nu
                        - 61.94527259504957*pow(xi,5.)*eta*nu;
         shape(267,2) = -2.4581457378987928*xi
                        + 7.374437213696378*xi*pow(eta,2.)
                        + 11.471346776861033*pow(xi,3.)
                        - 34.4140403305831*pow(xi,3.)*pow(eta,2.)
                        - 10.324212099174929*pow(xi,5.)
                        + 30.972636297524787*pow(xi,5.)*pow(eta,2.);
         shape(268,0) = -0.689981317681863*nu
                        + 14.489607671319124*pow(xi,2.)*nu
                        - 43.46882301395737*pow(xi,4.)*nu
                        + 31.87713687690207*pow(xi,6.)*nu;
         shape(268,2) = 4.829869223773041*xi
                        - 14.489607671319124*xi*pow(nu,2.)
                        - 28.97921534263825*pow(xi,3.)
                        + 86.93764602791474*pow(xi,3.)*pow(nu,2.)
                        + 31.87713687690207*pow(xi,5.)
                        - 95.63141063070621*pow(xi,5.)*pow(nu,2.);
         shape(269,1) = -0.689981317681863*nu
                        + 14.489607671319124*pow(xi,2.)*nu
                        - 43.46882301395737*pow(xi,4.)*nu
                        + 31.87713687690207*pow(xi,6.)*nu;
         shape(270,0) = -0.689981317681863*eta
                        + 14.489607671319124*pow(xi,2.)*eta
                        - 43.46882301395737*pow(xi,4.)*eta
                        + 31.87713687690207*pow(xi,6.)*eta;
         shape(270,2) = -28.97921534263825*xi*eta*nu
                        + 173.87529205582948*pow(xi,3.)*eta*nu
                        - 191.26282126141243*pow(xi,5.)*eta*nu;;
         shape(271,1) = -0.689981317681863*eta
                        + 14.489607671319124*pow(xi,2.)*eta
                        - 43.46882301395737*pow(xi,4.)*eta
                        + 31.87713687690207*pow(xi,6.)*eta;
         shape(271,2) = 0.689981317681863*nu
                        - 14.489607671319124*pow(xi,2.)*nu
                        + 43.46882301395737*pow(xi,4.)*nu
                        - 31.87713687690207*pow(xi,6.)*nu;
         shape(272,2) = -0.689981317681863*eta
                        + 14.489607671319124*pow(xi,2.)*eta
                        - 43.46882301395737*pow(xi,4.)*eta
                        + 31.87713687690207*pow(xi,6.)*eta;
         shape(273,0) = -2.995357736356377*xi + 26.958219627207395*pow(xi,3.)
                        - 59.30808317985627*pow(xi,5.)
                        + 36.714527682768164*pow(xi,7.);
         shape(273,2) = 2.995357736356377*nu - 80.87465888162218*pow(xi,2.)*nu
                        + 296.54041589928136*pow(xi,4.)*nu
                        - 257.00169377937715*pow(xi,6.)*nu;
         shape(274,1) = -2.995357736356377*xi + 26.958219627207395*pow(xi,3.)
                        - 59.30808317985627*pow(xi,5.)
                        + 36.714527682768164*pow(xi,7.);
         shape(275,2) = -2.995357736356377*xi + 26.958219627207395*pow(xi,3.)
                        - 59.30808317985627*pow(xi,5.)
                        + 36.714527682768164*pow(xi,7.);
      case 6:
         shape(133,0) = -0.3983608994994363 + 8.365578889488162*pow(nu,2.)
                        - 25.096736668464487*pow(nu,4.)
                        + 18.404273556873957*pow(nu,6.);
         shape(134,1) = -0.3983608994994363 + 8.365578889488162*pow(nu,2.)
                        - 25.096736668464487*pow(nu,4.)
                        + 18.404273556873957*pow(nu,6.);
         shape(135,0) = 3.8081430021731064*eta*nu
                        - 17.771334010141164*eta*pow(nu,3.)
                        + 15.994200609127047*eta*pow(nu,5.);
         shape(136,1) = 3.8081430021731064*eta*nu
                        - 17.771334010141164*eta*pow(nu,3.)
                        + 15.994200609127047*eta*pow(nu,5.);
         shape(136,2) = 0.1269381000724369 - 1.9040715010865532*pow(nu,2.)
                        + 4.442833502535291*pow(nu,4.)
                        - 2.6657001015211743*pow(nu,6.);
         shape(137,0) = -0.44469529596117835 + 4.446952959611783*pow(nu,2.)
                        - 5.188111786213748*pow(nu,4.)
                        + 1.334085887883535*pow(eta,2.)
                        - 13.34085887883535*pow(eta,2.)*pow(nu,2.)
                        + 15.564335358641243*pow(eta,2.)*pow(nu,4.);
         shape(138,1) = -0.44469529596117835 + 4.446952959611783*pow(nu,2.)
                        - 5.188111786213748*pow(nu,4.)
                        + 1.334085887883535*pow(eta,2.)
                        - 13.34085887883535*pow(eta,2.)*pow(nu,2.)
                        + 15.564335358641243*pow(eta,2.)*pow(nu,4.);
         shape(138,2) = -2.66817177576707*eta*nu
                        + 8.893905919223567*eta*pow(nu,3.)
                        - 6.225734143456497*eta*pow(nu,5.);
         shape(139,0) = 5.568465901844061*eta*nu
                        - 9.280776503073437*eta*pow(nu,3.)
                        - 9.280776503073437*pow(eta,3.)*nu
                        + 15.467960838455728*pow(eta,3.)*pow(nu,3.);
         shape(140,1) = 5.568465901844061*eta*nu
                        - 9.280776503073437*eta*pow(nu,3.)
                        - 9.280776503073437*pow(eta,3.)*nu
                        + 15.467960838455728*pow(eta,3.)*pow(nu,3.);
         shape(140,2) = 0.4640388251536718 - 2.7842329509220307*pow(nu,2.)
                        + 2.320194125768359*pow(nu,4.)
                        - 2.320194125768359*pow(eta,2.)
                        + 13.921164754610153*pow(eta,2.)*pow(nu,2.)
                        - 11.600970628841795*pow(eta,2.)*pow(nu,4.);
         shape(141,0) = -0.44469529596117835 + 1.334085887883535*pow(nu,2.)
                        + 4.446952959611783*pow(eta,2.)
                        - 13.34085887883535*pow(eta,2.)*pow(nu,2.)
                        - 5.188111786213748*pow(eta,4.)
                        + 15.564335358641243*pow(eta,4.)*pow(nu,2.);
         shape(142,1) = -0.44469529596117835 + 1.334085887883535*pow(nu,2.)
                        + 4.446952959611783*pow(eta,2.)
                        - 13.34085887883535*pow(eta,2.)*pow(nu,2.)
                        - 5.188111786213748*pow(eta,4.)
                        + 15.564335358641243*pow(eta,4.)*pow(nu,2.);
         shape(142,2) = -8.893905919223567*eta*nu
                        + 8.893905919223567*eta*pow(nu,3.)
                        + 20.75244714485499*pow(eta,3.)*nu
                        - 20.75244714485499*pow(eta,3.)*pow(nu,3.);
         shape(143,0) = 3.8081430021731064*eta*nu
                        - 17.771334010141164*pow(eta,3.)*nu
                        + 15.994200609127047*pow(eta,5.)*nu;
         shape(144,1) = 3.8081430021731064*eta*nu
                        - 17.771334010141164*pow(eta,3.)*nu
                        + 15.994200609127047*pow(eta,5.)*nu;
         shape(144,2) = 0.6346905003621844 - 1.9040715010865532*pow(nu,2.)
                        - 8.885667005070582*pow(eta,2.)
                        + 26.657001015211744*pow(eta,2.)*pow(nu,2.)
                        + 13.328500507605872*pow(eta,4.)
                        - 39.985501522817614*pow(eta,4.)*pow(nu,2.);
         shape(145,0) = -0.3983608994994363 + 8.365578889488162*pow(eta,2.)
                        - 25.096736668464487*pow(eta,4.)
                        + 18.404273556873957*pow(eta,6.);
         shape(146,1) = -0.3983608994994363 + 8.365578889488162*pow(eta,2.)
                        - 25.096736668464487*pow(eta,4.)
                        + 18.404273556873957*pow(eta,6.);
         shape(146,2) = -16.731157778976325*eta*nu
                        + 100.38694667385795*pow(eta,3.)*nu
                        - 110.42564134124375*pow(eta,5.)*nu;
         shape(147,2) = -0.3983608994994363 + 8.365578889488162*pow(eta,2.)
                        - 25.096736668464487*pow(eta,4.)
                        + 18.404273556873957*pow(eta,6.);
         shape(148,0) = 3.8081430021731064*xi*nu
                        - 17.771334010141164*xi*pow(nu,3.)
                        + 15.994200609127047*xi*pow(nu,5.);
         shape(148,2) = 0.1269381000724369 - 1.9040715010865532*pow(nu,2.)
                        + 4.442833502535291*pow(nu,4.)
                        - 2.6657001015211743*pow(nu,6.);
         shape(149,1) = 3.8081430021731064*xi*nu
                        - 17.771334010141164*xi*pow(nu,3.)
                        + 15.994200609127047*xi*pow(nu,5.);
         shape(150,0) = 1.1932426932522988*xi*eta
                        - 11.932426932522988*xi*eta*pow(nu,2.)
                        + 13.921164754610153*xi*eta*pow(nu,4.);
         shape(150,2) = -1.1932426932522988*eta*nu
                        + 3.97747564417433*eta*pow(nu,3.)
                        - 2.7842329509220307*eta*pow(nu,5.);
         shape(151,1) = 1.1932426932522988*xi*eta
                        - 11.932426932522988*xi*eta*pow(nu,2.)
                        + 13.921164754610153*xi*eta*pow(nu,4.);
         shape(151,2) = -1.1932426932522988*xi*nu
                        + 3.97747564417433*xi*pow(nu,3.)
                        - 2.7842329509220307*xi*pow(nu,5.);
         shape(152,0) = 2.7171331399105196*xi*nu
                        - 4.5285552331842*xi*pow(nu,3.)
                        - 8.15139941973156*xi*pow(eta,2.)*nu
                        + 13.5856656995526*xi*pow(eta,2.)*pow(nu,3.);
         shape(152,2) = 0.22642776165920997 - 1.3585665699552598*pow(nu,2.)
                        + 1.13213880829605*pow(nu,4.)
                        - 0.6792832849776299*pow(eta,2.)
                        + 4.07569970986578*pow(eta,2.)*pow(nu,2.)
                        - 3.39641642488815*pow(eta,2.)*pow(nu,4.);
         shape(153,1) = 2.7171331399105196*xi*nu
                        - 4.5285552331842*xi*pow(nu,3.)
                        - 8.15139941973156*xi*pow(eta,2.)*nu
                        + 13.5856656995526*xi*pow(eta,2.)*pow(nu,3.);
         shape(153,2) = -1.3585665699552598*xi*eta
                        + 8.15139941973156*xi*eta*pow(nu,2.)
                        - 6.792832849776299*xi*eta*pow(nu,4.);
         shape(154,0) = 2.7171331399105196*xi*eta
                        - 8.15139941973156*xi*eta*pow(nu,2.)
                        - 4.5285552331842*xi*pow(eta,3.)
                        + 13.5856656995526*xi*pow(eta,3.)*pow(nu,2.);
         shape(154,2) = -2.7171331399105196*eta*nu
                        + 2.7171331399105196*eta*pow(nu,3.)
                        + 4.5285552331842*pow(eta,3.)*nu
                        - 4.5285552331842*pow(eta,3.)*pow(nu,3.);
         shape(155,1) = 2.7171331399105196*xi*eta
                        - 8.15139941973156*xi*eta*pow(nu,2.)
                        - 4.5285552331842*xi*pow(eta,3.)
                        + 13.5856656995526*xi*pow(eta,3.)*pow(nu,2.);
         shape(155,2) = -2.7171331399105196*xi*nu
                        + 2.7171331399105196*xi*pow(nu,3.)
                        + 13.5856656995526*xi*pow(eta,2.)*nu
                        - 13.5856656995526*xi*pow(eta,2.)*pow(nu,3.);
         shape(156,0) = 1.1932426932522988*xi*nu
                        - 11.932426932522988*xi*pow(eta,2.)*nu
                        + 13.921164754610153*xi*pow(eta,4.)*nu;
         shape(156,2) = 0.1988737822087165 - 0.5966213466261495*pow(nu,2.)
                        - 1.988737822087165*pow(eta,2.)
                        + 5.966213466261495*pow(eta,2.)*pow(nu,2.)
                        + 2.320194125768359*pow(eta,4.)
                        - 6.9605823773050775*pow(eta,4.)*pow(nu,2.);
         shape(157,1) = 1.1932426932522988*xi*nu
                        - 11.932426932522988*xi*pow(eta,2.)*nu
                        + 13.921164754610153*xi*pow(eta,4.)*nu;
         shape(157,2) = -3.97747564417433*xi*eta
                        + 11.932426932522988*xi*eta*pow(nu,2.)
                        + 9.280776503073437*xi*pow(eta,3.)
                        - 27.84232950922031*xi*pow(eta,3.)*pow(nu,2.);
         shape(158,0) = 3.8081430021731064*xi*eta
                        - 17.771334010141164*xi*pow(eta,3.)
                        + 15.994200609127047*xi*pow(eta,5.);
         shape(158,2) = -3.8081430021731064*eta*nu
                        + 17.771334010141164*pow(eta,3.)*nu
                        - 15.994200609127047*pow(eta,5.)*nu;
         shape(159,1) = 3.8081430021731064*xi*eta
                        - 17.771334010141164*xi*pow(eta,3.)
                        + 15.994200609127047*xi*pow(eta,5.);
         shape(159,2) = -3.8081430021731064*xi*nu
                        + 53.31400203042349*xi*pow(eta,2.)*nu
                        - 79.97100304563523*xi*pow(eta,4.)*nu;
         shape(160,2) = 3.8081430021731064*xi*eta
                        - 17.771334010141164*xi*pow(eta,3.)
                        + 15.994200609127047*xi*pow(eta,5.);
         shape(161,0) = -0.44469529596117835 + 4.446952959611783*pow(nu,2.)
                        - 5.188111786213748*pow(nu,4.)
                        + 1.334085887883535*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(nu,2.)
                        + 15.564335358641243*pow(xi,2.)*pow(nu,4.);
         shape(161,2) = -2.66817177576707*xi*nu
                        + 8.893905919223567*xi*pow(nu,3.)
                        - 6.225734143456497*xi*pow(nu,5.);
         shape(162,1) = -0.44469529596117835 + 4.446952959611783*pow(nu,2.)
                        - 5.188111786213748*pow(nu,4.)
                        + 1.334085887883535*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(nu,2.)
                        + 15.564335358641243*pow(xi,2.)*pow(nu,4.);
         shape(163,0) = 2.7171331399105196*eta*nu
                        - 4.5285552331842*eta*pow(nu,3.)
                        - 8.15139941973156*pow(xi,2.)*eta*nu
                        + 13.5856656995526*pow(xi,2.)*eta*pow(nu,3.);
         shape(163,2) = -1.3585665699552598*xi*eta
                        + 8.15139941973156*xi*eta*pow(nu,2.)
                        - 6.792832849776299*xi*eta*pow(nu,4.);
         shape(164,1) = 2.7171331399105196*eta*nu
                        - 4.5285552331842*eta*pow(nu,3.)
                        - 8.15139941973156*pow(xi,2.)*eta*nu
                        + 13.5856656995526*pow(xi,2.)*eta*pow(nu,3.);
         shape(164,2) = 0.22642776165920997 - 1.3585665699552598*pow(nu,2.)
                        + 1.13213880829605*pow(nu,4.)
                        - 0.6792832849776299*pow(xi,2.)
                        + 4.07569970986578*pow(xi,2.)*pow(nu,2.)
                        - 3.39641642488815*pow(xi,2.)*pow(nu,4.);
         shape(165,0) = -0.49410588440130926 + 1.4823176532039277*pow(nu,2.)
                        + 1.4823176532039277*pow(eta,2.)
                        - 4.446952959611783*pow(eta,2.)*pow(nu,2.)
                        + 1.4823176532039277*pow(xi,2.)
                        - 4.446952959611783*pow(xi,2.)*pow(nu,2.)
                        - 4.446952959611783*pow(xi,2.)*pow(eta,2.)
                        + 13.34085887883535*pow(xi,2.)*pow(eta,2.)*pow(nu,2.);
         shape(165,2) = -2.9646353064078554*xi*nu
                        + 2.9646353064078554*xi*pow(nu,3.)
                        + 8.893905919223567*xi*pow(eta,2.)*nu
                        - 8.893905919223567*xi*pow(eta,2.)*pow(nu,3.);
         shape(166,1) = -0.49410588440130926 + 1.4823176532039277*pow(nu,2.)
                        + 1.4823176532039277*pow(eta,2.)
                        - 4.446952959611783*pow(eta,2.)*pow(nu,2.)
                        + 1.4823176532039277*pow(xi,2.)
                        - 4.446952959611783*pow(xi,2.)*pow(nu,2.)
                        - 4.446952959611783*pow(xi,2.)*pow(eta,2.)
                        + 13.34085887883535*pow(xi,2.)*pow(eta,2.)*pow(nu,2.);
         shape(166,2) = -2.9646353064078554*eta*nu
                        + 2.9646353064078554*eta*pow(nu,3.)
                        + 8.893905919223567*pow(xi,2.)*eta*nu
                        - 8.893905919223567*pow(xi,2.)*eta*pow(nu,3.);
         shape(167,0) = 2.7171331399105196*eta*nu
                        - 4.5285552331842*pow(eta,3.)*nu
                        - 8.15139941973156*pow(xi,2.)*eta*nu
                        + 13.5856656995526*pow(xi,2.)*pow(eta,3.)*nu;
         shape(167,2) = -2.7171331399105196*xi*eta
                        + 8.15139941973156*xi*eta*pow(nu,2.)
                        + 4.5285552331842*xi*pow(eta,3.)
                        - 13.5856656995526*xi*pow(eta,3.)*pow(nu,2.);
         shape(168,1) = 2.7171331399105196*eta*nu
                        - 4.5285552331842*pow(eta,3.)*nu
                        - 8.15139941973156*pow(xi,2.)*eta*nu
                        + 13.5856656995526*pow(xi,2.)*pow(eta,3.)*nu;
         shape(168,2) = 0.45285552331841994 - 1.3585665699552598*pow(nu,2.)
                        - 2.2642776165921*pow(eta,2.)
                        + 6.792832849776299*pow(eta,2.)*pow(nu,2.)
                        - 1.3585665699552598*pow(xi,2.)
                        + 4.07569970986578*pow(xi,2.)*pow(nu,2.)
                        + 6.792832849776299*pow(xi,2.)*pow(eta,2.)
                        - 20.3784985493289*pow(xi,2.)*pow(eta,2.)*pow(nu,2.);
         shape(169,0) = -0.44469529596117835 + 4.446952959611783*pow(eta,2.)
                        - 5.188111786213748*pow(eta,4.)
                        + 1.334085887883535*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(eta,2.)
                        + 15.564335358641243*pow(xi,2.)*pow(eta,4.);
         shape(169,2) = -2.66817177576707*xi*nu
                        + 26.6817177576707*xi*pow(eta,2.)*nu
                        - 31.128670717282485*xi*pow(eta,4.)*nu;
         shape(170,1) = -0.44469529596117835 + 4.446952959611783*pow(eta,2.)
                        - 5.188111786213748*pow(eta,4.)
                        + 1.334085887883535*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(eta,2.)
                        + 15.564335358641243*pow(xi,2.)*pow(eta,4.);
         shape(170,2) = -8.893905919223567*eta*nu
                        + 20.75244714485499*pow(eta,3.)*nu
                        + 26.6817177576707*pow(xi,2.)*eta*nu
                        - 62.25734143456497*pow(xi,2.)*pow(eta,3.)*nu;
         shape(171,2) = -0.44469529596117835 + 4.446952959611783*pow(eta,2.)
                        - 5.188111786213748*pow(eta,4.)
                        + 1.334085887883535*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(eta,2.)
                        + 15.564335358641243*pow(xi,2.)*pow(eta,4.);
         shape(172,0) = 5.568465901844061*xi*nu
                        - 9.280776503073437*xi*pow(nu,3.)
                        - 9.280776503073437*pow(xi,3.)*nu
                        + 15.467960838455728*pow(xi,3.)*pow(nu,3.);
         shape(172,2) = 0.4640388251536718 - 2.7842329509220307*pow(nu,2.)
                        + 2.320194125768359*pow(nu,4.)
                        - 2.320194125768359*pow(xi,2.)
                        + 13.921164754610153*pow(xi,2.)*pow(nu,2.)
                        - 11.600970628841795*pow(xi,2.)*pow(nu,4.);
         shape(173,1) = 5.568465901844061*xi*nu
                        - 9.280776503073437*xi*pow(nu,3.)
                        - 9.280776503073437*pow(xi,3.)*nu
                        + 15.467960838455728*pow(xi,3.)*pow(nu,3.);
         shape(174,0) = 2.7171331399105196*xi*eta
                        - 8.15139941973156*xi*eta*pow(nu,2.)
                        - 4.5285552331842*pow(xi,3.)*eta
                        + 13.5856656995526*pow(xi,3.)*eta*pow(nu,2.);
         shape(174,2) = -2.7171331399105196*eta*nu
                        + 2.7171331399105196*eta*pow(nu,3.)
                        + 13.5856656995526*pow(xi,2.)*eta*nu
                        - 13.5856656995526*pow(xi,2.)*eta*pow(nu,3.);
         shape(175,1) = 2.7171331399105196*xi*eta
                        - 8.15139941973156*xi*eta*pow(nu,2.)
                        - 4.5285552331842*pow(xi,3.)*eta
                        + 13.5856656995526*pow(xi,3.)*eta*pow(nu,2.);
         shape(175,2) = -2.7171331399105196*xi*nu
                        + 2.7171331399105196*xi*pow(nu,3.)
                        + 4.5285552331842*pow(xi,3.)*nu
                        - 4.5285552331842*pow(xi,3.)*pow(nu,3.);
         shape(176,0) = 2.7171331399105196*xi*nu
                        - 8.15139941973156*xi*pow(eta,2.)*nu
                        - 4.5285552331842*pow(xi,3.)*nu
                        + 13.5856656995526*pow(xi,3.)*pow(eta,2.)*nu;
         shape(176,2) = 0.45285552331841994 - 1.3585665699552598*pow(nu,2.)
                        - 1.3585665699552598*pow(eta,2.)
                        + 4.07569970986578*pow(eta,2.)*pow(nu,2.)
                        - 2.2642776165921*pow(xi,2.)
                        + 6.792832849776299*pow(xi,2.)*pow(nu,2.)
                        + 6.792832849776299*pow(xi,2.)*pow(eta,2.)
                        - 20.3784985493289*pow(xi,2.)*pow(eta,2.)*pow(nu,2.);
         shape(177,1) = 2.7171331399105196*xi*nu
                        - 8.15139941973156*xi*pow(eta,2.)*nu
                        - 4.5285552331842*pow(xi,3.)*nu
                        + 13.5856656995526*pow(xi,3.)*pow(eta,2.)*nu;
         shape(177,2) = -2.7171331399105196*xi*eta
                        + 8.15139941973156*xi*eta*pow(nu,2.)
                        + 4.5285552331842*pow(xi,3.)*eta
                        - 13.5856656995526*pow(xi,3.)*eta*pow(nu,2.);
         shape(178,0) = 5.568465901844061*xi*eta
                        - 9.280776503073437*xi*pow(eta,3.)
                        - 9.280776503073437*pow(xi,3.)*eta
                        + 15.467960838455728*pow(xi,3.)*pow(eta,3.);
         shape(178,2) = -5.568465901844061*eta*nu
                        + 9.280776503073437*pow(eta,3.)*nu
                        + 27.84232950922031*pow(xi,2.)*eta*nu
                        - 46.40388251536718*pow(xi,2.)*pow(eta,3.)*nu;
         shape(179,1) = 5.568465901844061*xi*eta
                        - 9.280776503073437*xi*pow(eta,3.)
                        - 9.280776503073437*pow(xi,3.)*eta
                        + 15.467960838455728*pow(xi,3.)*pow(eta,3.);
         shape(179,2) = -5.568465901844061*xi*nu
                        + 27.84232950922031*xi*pow(eta,2.)*nu
                        + 9.280776503073437*pow(xi,3.)*nu
                        - 46.40388251536718*pow(xi,3.)*pow(eta,2.)*nu;
         shape(180,2) = 5.568465901844061*xi*eta
                        - 9.280776503073437*xi*pow(eta,3.)
                        - 9.280776503073437*pow(xi,3.)*eta
                        + 15.467960838455728*pow(xi,3.)*pow(eta,3.);
         shape(181,0) = -0.44469529596117835 + 1.334085887883535*pow(nu,2.)
                        + 4.446952959611783*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(nu,2.)
                        - 5.188111786213748*pow(xi,4.)
                        + 15.564335358641243*pow(xi,4.)*pow(nu,2.);
         shape(181,2) = -8.893905919223567*xi*nu
                        + 8.893905919223567*xi*pow(nu,3.)
                        + 20.75244714485499*pow(xi,3.)*nu
                        - 20.75244714485499*pow(xi,3.)*pow(nu,3.);
         shape(182,1) = -0.44469529596117835 + 1.334085887883535*pow(nu,2.)
                        + 4.446952959611783*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(nu,2.)
                        - 5.188111786213748*pow(xi,4.)
                        + 15.564335358641243*pow(xi,4.)*pow(nu,2.);
         shape(183,0) = 1.1932426932522988*eta*nu
                        - 11.932426932522988*pow(xi,2.)*eta*nu
                        + 13.921164754610153*pow(xi,4.)*eta*nu;
         shape(183,2) = -3.97747564417433*xi*eta
                        + 11.932426932522988*xi*eta*pow(nu,2.)
                        + 9.280776503073437*pow(xi,3.)*eta
                        - 27.84232950922031*pow(xi,3.)*eta*pow(nu,2.);
         shape(184,1) = 1.1932426932522988*eta*nu
                        - 11.932426932522988*pow(xi,2.)*eta*nu
                        + 13.921164754610153*pow(xi,4.)*eta*nu;
         shape(184,2) = 0.1988737822087165 - 0.5966213466261495*pow(nu,2.)
                        - 1.988737822087165*pow(xi,2.)
                        + 5.966213466261495*pow(xi,2.)*pow(nu,2.)
                        + 2.320194125768359*pow(xi,4.)
                        - 6.9605823773050775*pow(xi,4.)*pow(nu,2.);
         shape(185,0) = -0.44469529596117835 + 1.334085887883535*pow(eta,2.)
                        + 4.446952959611783*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(eta,2.)
                        - 5.188111786213748*pow(xi,4.)
                        + 15.564335358641243*pow(xi,4.)*pow(eta,2.);
         shape(185,2) = -8.893905919223567*xi*nu
                        + 26.6817177576707*xi*pow(eta,2.)*nu
                        + 20.75244714485499*pow(xi,3.)*nu
                        - 62.25734143456497*pow(xi,3.)*pow(eta,2.)*nu;
         shape(186,1) = -0.44469529596117835 + 1.334085887883535*pow(eta,2.)
                        + 4.446952959611783*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(eta,2.)
                        - 5.188111786213748*pow(xi,4.)
                        + 15.564335358641243*pow(xi,4.)*pow(eta,2.);
         shape(186,2) = -2.66817177576707*eta*nu
                        + 26.6817177576707*pow(xi,2.)*eta*nu
                        - 31.128670717282485*pow(xi,4.)*eta*nu;
         shape(187,2) = -0.44469529596117835 + 1.334085887883535*pow(eta,2.)
                        + 4.446952959611783*pow(xi,2.)
                        - 13.34085887883535*pow(xi,2.)*pow(eta,2.)
                        - 5.188111786213748*pow(xi,4.)
                        + 15.564335358641243*pow(xi,4.)*pow(eta,2.);
         shape(188,0) = 3.8081430021731064*xi*nu
                        - 17.771334010141164*pow(xi,3.)*nu
                        + 15.994200609127047*pow(xi,5.)*nu;
         shape(188,2) = 0.6346905003621844 - 1.9040715010865532*pow(nu,2.)
                        - 8.885667005070582*pow(xi,2.)
                        + 26.657001015211744*pow(xi,2.)*pow(nu,2.)
                        + 13.328500507605872*pow(xi,4.)
                        - 39.985501522817614*pow(xi,4.)*pow(nu,2.);
         shape(189,1) = 3.8081430021731064*xi*nu
                        - 17.771334010141164*pow(xi,3.)*nu
                        + 15.994200609127047*pow(xi,5.)*nu;
         shape(190,0) = 3.8081430021731064*xi*eta
                        - 17.771334010141164*pow(xi,3.)*eta
                        + 15.994200609127047*pow(xi,5.)*eta;
         shape(190,2) = -3.8081430021731064*eta*nu
                        + 53.31400203042349*pow(xi,2.)*eta*nu
                        - 79.97100304563523*pow(xi,4.)*eta*nu;
         shape(191,1) = 3.8081430021731064*xi*eta
                        - 17.771334010141164*pow(xi,3.)*eta
                        + 15.994200609127047*pow(xi,5.)*eta;
         shape(191,2) = -3.8081430021731064*xi*nu
                        + 17.771334010141164*pow(xi,3.)*nu
                        - 15.994200609127047*pow(xi,5.)*nu;
         shape(192,2) = 3.8081430021731064*xi*eta
                        - 17.771334010141164*pow(xi,3.)*eta
                        + 15.994200609127047*pow(xi,5.)*eta;
         shape(193,0) = -0.3983608994994363 + 8.365578889488162*pow(xi,2.)
                        - 25.096736668464487*pow(xi,4.)
                        + 18.404273556873957*pow(xi,6.);
         shape(193,2) = -16.731157778976325*xi*nu
                        + 100.38694667385795*pow(xi,3.)*nu
                        - 110.42564134124375*pow(xi,5.)*nu;
         shape(194,1) = -0.3983608994994363 + 8.365578889488162*pow(xi,2.)
                        - 25.096736668464487*pow(xi,4.)
                        + 18.404273556873957*pow(xi,6.);
         shape(195,2) = -0.3983608994994363 + 8.365578889488162*pow(xi,2.)
                        - 25.096736668464487*pow(xi,4.)
                        + 18.404273556873957*pow(xi,6.);
      case 5:
         shape( 85,0) = 2.1986323874172324*nu - 10.260284474613751*pow(nu,3.)
                        + 9.234256027152377*pow(nu,5.);
         shape( 86,1) = 2.1986323874172324*nu - 10.260284474613751*pow(nu,3.)
                        + 9.234256027152377*pow(nu,5.);
         shape( 87,0) = 0.6889189901577688*eta
                        - 6.889189901577688*eta*pow(nu,2.)
                        + 8.037388218507303*eta*pow(nu,4.);
         shape( 88,1) = 0.6889189901577688*eta
                        - 6.889189901577688*eta*pow(nu,2.)
                        + 8.037388218507303*eta*pow(nu,4.);
         shape( 88,2) = -0.6889189901577688*nu + 2.2963966338592297*pow(nu,3.)
                        - 1.6074776437014606*pow(nu,5.);
         shape( 89,0) = 1.5687375497513918*nu - 2.6145625829189862*pow(nu,3.)
                        - 4.706212649254175*pow(eta,2.)*nu
                        + 7.843687748756959*pow(eta,2.)*pow(nu,3.);
         shape( 90,1) = 1.5687375497513918*nu - 2.6145625829189862*pow(nu,3.)
                        - 4.706212649254175*pow(eta,2.)*nu
                        + 7.843687748756959*pow(eta,2.)*pow(nu,3.);
         shape( 90,2) = -0.7843687748756958*eta
                        + 4.706212649254175*eta*pow(nu,2.)
                        - 3.921843874378479*eta*pow(nu,4.);
         shape( 91,0) = 1.5687375497513918*eta
                        - 4.706212649254175*eta*pow(nu,2.)
                        - 2.6145625829189862*pow(eta,3.)
                        + 7.843687748756959*pow(eta,3.)*pow(nu,2.);
         shape( 92,1) = 1.5687375497513918*eta
                        - 4.706212649254175*eta*pow(nu,2.)
                        - 2.6145625829189862*pow(eta,3.)
                        + 7.843687748756959*pow(eta,3.)*pow(nu,2.);
         shape( 92,2) = -1.5687375497513918*nu + 1.5687375497513918*pow(nu,3.)
                        + 7.843687748756959*pow(eta,2.)*nu
                        - 7.843687748756959*pow(eta,2.)*pow(nu,3.);
         shape( 93,0) = 0.6889189901577688*nu
                        - 6.889189901577688*pow(eta,2.)*nu
                        + 8.037388218507303*pow(eta,4.)*nu;
         shape( 94,1) = 0.6889189901577688*nu
                        - 6.889189901577688*pow(eta,2.)*nu
                        + 8.037388218507303*pow(eta,4.)*nu;
         shape( 94,2) = -2.2963966338592297*eta
                        + 6.889189901577688*eta*pow(nu,2.)
                        + 5.358258812338202*pow(eta,3.)
                        - 16.074776437014606*pow(eta,3.)*pow(nu,2.);
         shape( 95,0) = 2.1986323874172324*eta
                        - 10.260284474613751*pow(eta,3.)
                        + 9.234256027152377*pow(eta,5.);
         shape( 96,1) = 2.1986323874172324*eta
                        - 10.260284474613751*pow(eta,3.)
                        + 9.234256027152377*pow(eta,5.);
         shape( 96,2) = -2.1986323874172324*nu
                        + 30.780853423841258*pow(eta,2.)*nu
                        - 46.17128013576188*pow(eta,4.)*nu;
         shape( 97,2) = 2.1986323874172324*eta
                        - 10.260284474613751*pow(eta,3.)
                        + 9.234256027152377*pow(eta,5.);
         shape( 98,0) = 0.6889189901577688*xi
                        - 6.889189901577688*xi*pow(nu,2.)
                        + 8.037388218507303*xi*pow(nu,4.);
         shape( 98,2) = -0.6889189901577688*nu + 2.2963966338592297*pow(nu,3.)
                        - 1.6074776437014606*pow(nu,5.);
         shape( 99,1) = 0.6889189901577688*xi
                        - 6.889189901577688*xi*pow(nu,2.)
                        + 8.037388218507303*xi*pow(nu,4.);
         shape(100,0) = -4.209364560120684*xi*eta*nu
                        + 7.0156076002011405*xi*eta*pow(nu,3.);
         shape(100,2) = -0.350780380010057*eta
                        + 2.104682280060342*eta*pow(nu,2.)
                        - 1.753901900050285*eta*pow(nu,4.);
         shape(101,1) = -4.209364560120684*xi*eta*nu
                        + 7.0156076002011405*xi*eta*pow(nu,3.);
         shape(101,2) = -0.350780380010057*xi + 2.104682280060342*xi*pow(nu,2.)
                        - 1.753901900050285*xi*pow(nu,4.);
         shape(102,0) = 0.7654655446197431*xi
                        - 2.2963966338592297*xi*pow(nu,2.)
                        - 2.2963966338592297*xi*pow(eta,2.)
                        + 6.889189901577688*xi*pow(eta,2.)*pow(nu,2.);
         shape(102,2) = -0.7654655446197431*nu + 0.7654655446197431*pow(nu,3.)
                        + 2.2963966338592297*pow(eta,2.)*nu
                        - 2.2963966338592297*pow(eta,2.)*pow(nu,3.);
         shape(103,1) = 0.7654655446197431*xi
                        - 2.2963966338592297*xi*pow(nu,2.)
                        - 2.2963966338592297*xi*pow(eta,2.)
                        + 6.889189901577688*xi*pow(eta,2.)*pow(nu,2.);
         shape(103,2) = 4.592793267718459*xi*eta*nu
                        - 4.592793267718459*xi*eta*pow(nu,3.);
         shape(104,0) = -4.209364560120684*xi*eta*nu
                        + 7.0156076002011405*xi*pow(eta,3.)*nu;
         shape(104,2) = -0.701560760020114*eta
                        + 2.104682280060342*eta*pow(nu,2.)
                        + 1.1692679333668567*pow(eta,3.)
                        - 3.50780380010057*pow(eta,3.)*pow(nu,2.);
         shape(105,1) = -4.209364560120684*xi*eta*nu
                        + 7.0156076002011405*xi*pow(eta,3.)*nu;
         shape(105,2) = -0.701560760020114*xi + 2.104682280060342*xi*pow(nu,2.)
                        + 3.50780380010057*xi*pow(eta,2.)
                        - 10.52341140030171*xi*pow(eta,2.)*pow(nu,2.);
         shape(106,0) = 0.6889189901577688*xi
                        - 6.889189901577688*xi*pow(eta,2.)
                        + 8.037388218507303*xi*pow(eta,4.);
         shape(106,2) = -0.6889189901577688*nu
                        + 6.889189901577688*pow(eta,2.)*nu
                        - 8.037388218507303*pow(eta,4.)*nu;
         shape(107,1) = 0.6889189901577688*xi
                        - 6.889189901577688*xi*pow(eta,2.)
                        + 8.037388218507303*xi*pow(eta,4.);
         shape(107,2) = 13.778379803155376*xi*eta*nu
                        - 32.14955287402921*xi*pow(eta,3.)*nu;
         shape(108,2) = 0.6889189901577688*xi
                        - 6.889189901577688*xi*pow(eta,2.)
                        + 8.037388218507303*xi*pow(eta,4.);
         shape(109,0) = 1.5687375497513918*nu - 2.6145625829189862*pow(nu,3.)
                        - 4.706212649254175*pow(xi,2.)*nu
                        + 7.843687748756959*pow(xi,2.)*pow(nu,3.);
         shape(109,2) = -0.7843687748756958*xi
                        + 4.706212649254175*xi*pow(nu,2.)
                        - 3.921843874378479*xi*pow(nu,4.);
         shape(110,1) = 1.5687375497513918*nu - 2.6145625829189862*pow(nu,3.)
                        - 4.706212649254175*pow(xi,2.)*nu
                        + 7.843687748756959*pow(xi,2.)*pow(nu,3.);
         shape(111,0) = 0.7654655446197431*eta
                        - 2.2963966338592297*eta*pow(nu,2.)
                        - 2.2963966338592297*pow(xi,2.)*eta
                        + 6.889189901577688*pow(xi,2.)*eta*pow(nu,2.);
         shape(111,2) = 4.592793267718459*xi*eta*nu
                        - 4.592793267718459*xi*eta*pow(nu,3.);
         shape(112,1) = 0.7654655446197431*eta
                        - 2.2963966338592297*eta*pow(nu,2.)
                        - 2.2963966338592297*pow(xi,2.)*eta
                        + 6.889189901577688*pow(xi,2.)*eta*pow(nu,2.);
         shape(112,2) = -0.7654655446197431*nu + 0.7654655446197431*pow(nu,3.)
                        + 2.2963966338592297*pow(xi,2.)*nu
                        - 2.2963966338592297*pow(xi,2.)*pow(nu,3.);
         shape(113,0) = 0.7654655446197431*nu
                        - 2.2963966338592297*pow(eta,2.)*nu
                        - 2.2963966338592297*pow(xi,2.)*nu
                        + 6.889189901577688*pow(xi,2.)*pow(eta,2.)*nu;
         shape(113,2) = -0.7654655446197431*xi
                        + 2.2963966338592297*xi*pow(nu,2.)
                        + 2.2963966338592297*xi*pow(eta,2.)
                        - 6.889189901577688*xi*pow(eta,2.)*pow(nu,2.);
         shape(114,1) = 0.7654655446197431*nu
                        - 2.2963966338592297*pow(eta,2.)*nu
                        - 2.2963966338592297*pow(xi,2.)*nu
                        + 6.889189901577688*pow(xi,2.)*pow(eta,2.)*nu;
         shape(114,2) = -0.7654655446197431*eta
                        + 2.2963966338592297*eta*pow(nu,2.)
                        + 2.2963966338592297*pow(xi,2.)*eta
                        - 6.889189901577688*pow(xi,2.)*eta*pow(nu,2.);
         shape(115,0) = 1.5687375497513918*eta
                        - 2.6145625829189862*pow(eta,3.)
                        - 4.706212649254175*pow(xi,2.)*eta
                        + 7.843687748756959*pow(xi,2.)*pow(eta,3.);
         shape(115,2) = 9.41242529850835*xi*eta*nu
                        - 15.687375497513917*xi*pow(eta,3.)*nu;
         shape(116,1) = 1.5687375497513918*eta
                        - 2.6145625829189862*pow(eta,3.)
                        - 4.706212649254175*pow(xi,2.)*eta
                        + 7.843687748756959*pow(xi,2.)*pow(eta,3.);
         shape(116,2) = -1.5687375497513918*nu
                        + 7.843687748756959*pow(eta,2.)*nu
                        + 4.706212649254175*pow(xi,2.)*nu
                        - 23.531063246270875*pow(xi,2.)*pow(eta,2.)*nu;
         shape(117,2) = 1.5687375497513918*eta
                        - 2.6145625829189862*pow(eta,3.)
                        - 4.706212649254175*pow(xi,2.)*eta
                        + 7.843687748756959*pow(xi,2.)*pow(eta,3.);
         shape(118,0) = 1.5687375497513918*xi
                        - 4.706212649254175*xi*pow(nu,2.)
                        - 2.6145625829189862*pow(xi,3.)
                        + 7.843687748756959*pow(xi,3.)*pow(nu,2.);
         shape(118,2) = -1.5687375497513918*nu + 1.5687375497513918*pow(nu,3.)
                        + 7.843687748756959*pow(xi,2.)*nu
                        - 7.843687748756959*pow(xi,2.)*pow(nu,3.);
         shape(119,1) = 1.5687375497513918*xi
                        - 4.706212649254175*xi*pow(nu,2.)
                        - 2.6145625829189862*pow(xi,3.)
                        + 7.843687748756959*pow(xi,3.)*pow(nu,2.);
         shape(120,0) = -4.209364560120684*xi*eta*nu
                        + 7.0156076002011405*pow(xi,3.)*eta*nu;
         shape(120,2) = -0.701560760020114*eta
                        + 2.104682280060342*eta*pow(nu,2.)
                        + 3.50780380010057*pow(xi,2.)*eta
                        - 10.52341140030171*pow(xi,2.)*eta*pow(nu,2.);
         shape(121,1) = -4.209364560120684*xi*eta*nu
                        + 7.0156076002011405*pow(xi,3.)*eta*nu;
         shape(121,2) = -0.701560760020114*xi + 2.104682280060342*xi*pow(nu,2.)
                        + 1.1692679333668567*pow(xi,3.)
                        - 3.50780380010057*pow(xi,3.)*pow(nu,2.);
         shape(122,0) = 1.5687375497513918*xi
                        - 4.706212649254175*xi*pow(eta,2.)
                        - 2.6145625829189862*pow(xi,3.)
                        + 7.843687748756959*pow(xi,3.)*pow(eta,2.);
         shape(122,2) = -1.5687375497513918*nu
                        + 4.706212649254175*pow(eta,2.)*nu
                        + 7.843687748756959*pow(xi,2.)*nu
                        - 23.531063246270875*pow(xi,2.)*pow(eta,2.)*nu;
         shape(123,1) = 1.5687375497513918*xi
                        - 4.706212649254175*xi*pow(eta,2.)
                        - 2.6145625829189862*pow(xi,3.)
                        + 7.843687748756959*pow(xi,3.)*pow(eta,2.);
         shape(123,2) = 9.41242529850835*xi*eta*nu
                        - 15.687375497513917*pow(xi,3.)*eta*nu;
         shape(124,2) = 1.5687375497513918*xi
                        - 4.706212649254175*xi*pow(eta,2.)
                        - 2.6145625829189862*pow(xi,3.)
                        + 7.843687748756959*pow(xi,3.)*pow(eta,2.);
         shape(125,0) = 0.6889189901577688*nu
                        - 6.889189901577688*pow(xi,2.)*nu
                        + 8.037388218507303*pow(xi,4.)*nu;
         shape(125,2) = -2.2963966338592297*xi
                        + 6.889189901577688*xi*pow(nu,2.)
                        + 5.358258812338202*pow(xi,3.)
                        - 16.074776437014606*pow(xi,3.)*pow(nu,2.);
         shape(126,1) = 0.6889189901577688*nu
                        - 6.889189901577688*pow(xi,2.)*nu
                        + 8.037388218507303*pow(xi,4.)*nu;
         shape(127,0) = 0.6889189901577688*eta
                        - 6.889189901577688*pow(xi,2.)*eta
                        + 8.037388218507303*pow(xi,4.)*eta;
         shape(127,2) = 13.778379803155376*xi*eta*nu
                        - 32.14955287402921*pow(xi,3.)*eta*nu;
         shape(128,1) = 0.6889189901577688*eta
                        - 6.889189901577688*pow(xi,2.)*eta
                        + 8.037388218507303*pow(xi,4.)*eta;
         shape(128,2) = -0.6889189901577688*nu
                        + 6.889189901577688*pow(xi,2.)*nu
                        - 8.037388218507303*pow(xi,4.)*nu;
         shape(129,2) = 0.6889189901577688*eta
                        - 6.889189901577688*pow(xi,2.)*eta
                        + 8.037388218507303*pow(xi,4.)*eta;
         shape(130,0) = 2.1986323874172324*xi - 10.260284474613751*pow(xi,3.)
                        + 9.234256027152377*pow(xi,5.);
         shape(130,2) = -2.1986323874172324*nu
                        + 30.780853423841258*pow(xi,2.)*nu
                        - 46.17128013576188*pow(xi,4.)*nu;
         shape(131,1) = 2.1986323874172324*xi - 10.260284474613751*pow(xi,3.)
                        + 9.234256027152377*pow(xi,5.);
         shape(132,2) = 2.1986323874172324*xi - 10.260284474613751*pow(xi,3.)
                        + 9.234256027152377*pow(xi,5.);
      case 4:
         shape( 50,0) = 0.397747564417433 - 3.97747564417433*pow(nu,2.)
                        + 4.640388251536718*pow(nu,4.);
         shape( 51,1) = 0.397747564417433 - 3.97747564417433*pow(nu,2.)
                        + 4.640388251536718*pow(nu,4.);
         shape( 52,0) = -2.4302777619029476*eta*nu
                        + 4.050462936504912*eta*pow(nu,3.);
         shape( 53,1) = -2.4302777619029476*eta*nu
                        + 4.050462936504912*eta*pow(nu,3.);
         shape( 53,2) = -0.20252314682524564 + 1.2151388809514738*pow(nu,2.)
                        - 1.0126157341262283*pow(nu,4.);
         shape( 54,0) = 0.4419417382415922 - 1.3258252147247767*pow(nu,2.)
                        - 1.3258252147247767*pow(eta,2.)
                        + 3.97747564417433*pow(eta,2.)*pow(nu,2.);
         shape( 55,1) = 0.4419417382415922 - 1.3258252147247767*pow(nu,2.)
                        - 1.3258252147247767*pow(eta,2.)
                        + 3.97747564417433*pow(eta,2.)*pow(nu,2.);
         shape( 55,2) = 2.6516504294495533*eta*nu
                        - 2.6516504294495533*eta*pow(nu,3.);
         shape( 56,0) = -2.4302777619029476*eta*nu
                        + 4.050462936504912*pow(eta,3.)*nu;
         shape( 57,1) = -2.4302777619029476*eta*nu
                        + 4.050462936504912*pow(eta,3.)*nu;
         shape( 57,2) = -0.4050462936504913 + 1.2151388809514738*pow(nu,2.)
                        + 2.025231468252456*pow(eta,2.)
                        - 6.075694404757369*pow(eta,2.)*pow(nu,2.);
         shape( 58,0) = 0.397747564417433 - 3.97747564417433*pow(eta,2.)
                        + 4.640388251536718*pow(eta,4.);
         shape( 59,1) = 0.397747564417433 - 3.97747564417433*pow(eta,2.)
                        + 4.640388251536718*pow(eta,4.);
         shape( 59,2) = 7.95495128834866*eta*nu
                        - 18.561553006146873*pow(eta,3.)*nu;
         shape( 60,2) = 0.397747564417433 - 3.97747564417433*pow(eta,2.)
                        + 4.640388251536718*pow(eta,4.);
         shape( 61,0) = -2.4302777619029476*xi*nu
                        + 4.050462936504912*xi*pow(nu,3.);
         shape( 61,2) = -0.20252314682524564 + 1.2151388809514738*pow(nu,2.)
                        - 1.0126157341262283*pow(nu,4.);
         shape( 62,1) = -2.4302777619029476*xi*nu
                        + 4.050462936504912*xi*pow(nu,3.);
         shape( 63,0) = -1.1858541225631423*xi*eta
                        + 3.557562367689427*xi*eta*pow(nu,2.);
         shape( 63,2) = 1.1858541225631423*eta*nu
                        - 1.1858541225631423*eta*pow(nu,3.);
         shape( 64,1) = -1.1858541225631423*xi*eta
                        + 3.557562367689427*xi*eta*pow(nu,2.);
         shape( 64,2) = 1.1858541225631423*xi*nu
                        - 1.1858541225631423*xi*pow(nu,3.);
         shape( 65,0) = -1.1858541225631423*xi*nu
                        + 3.557562367689427*xi*pow(eta,2.)*nu;
         shape( 65,2) = -0.19764235376052372
                        + 0.5929270612815711*pow(nu,2.)
                        + 0.5929270612815711*pow(eta,2.)
                        - 1.7787811838447134*pow(eta,2.)*pow(nu,2.);
         shape( 66,1) = -1.1858541225631423*xi*nu
                        + 3.557562367689427*xi*pow(eta,2.)*nu;
         shape( 66,2) = 1.1858541225631423*xi*eta
                        - 3.557562367689427*xi*eta*pow(nu,2.);
         shape( 67,0) = -2.4302777619029476*xi*eta
                        + 4.050462936504912*xi*pow(eta,3.);
         shape( 67,2) = 2.4302777619029476*eta*nu
                        - 4.050462936504912*pow(eta,3.)*nu;
         shape( 68,1) = -2.4302777619029476*xi*eta
                        + 4.050462936504912*xi*pow(eta,3.);
         shape( 68,2) = 2.4302777619029476*xi*nu
                        - 12.151388809514739*xi*pow(eta,2.)*nu;
         shape( 69,2) = -2.4302777619029476*xi*eta
                        + 4.050462936504912*xi*pow(eta,3.);
         shape( 70,0) = 0.4419417382415922 - 1.3258252147247767*pow(nu,2.)
                        - 1.3258252147247767*pow(xi,2.)
                        + 3.97747564417433*pow(xi,2.)*pow(nu,2.);
         shape( 70,2) = 2.6516504294495533*xi*nu
                        - 2.6516504294495533*xi*pow(nu,3.);
         shape( 71,1) = 0.4419417382415922 - 1.3258252147247767*pow(nu,2.)
                        - 1.3258252147247767*pow(xi,2.)
                        + 3.97747564417433*pow(xi,2.)*pow(nu,2.);
         shape( 72,0) = -1.1858541225631423*eta*nu
                        + 3.557562367689427*pow(xi,2.)*eta*nu;
         shape( 72,2) = 1.1858541225631423*xi*eta
                        - 3.557562367689427*xi*eta*pow(nu,2.);
         shape( 73,1) = -1.1858541225631423*eta*nu
                        + 3.557562367689427*pow(xi,2.)*eta*nu;
         shape( 73,2) = -0.19764235376052372 + 0.5929270612815711*pow(nu,2.)
                        + 0.5929270612815711*pow(xi,2.)
                        - 1.7787811838447134*pow(xi,2.)*pow(nu,2.);
         shape( 74,0) = 0.4419417382415922 - 1.3258252147247767*pow(eta,2.)
                        - 1.3258252147247767*pow(xi,2.)
                        + 3.97747564417433*pow(xi,2.)*pow(eta,2.);
         shape( 74,2) = 2.6516504294495533*xi*nu
                        - 7.95495128834866*xi*pow(eta,2.)*nu;
         shape( 75,1) = 0.4419417382415922 - 1.3258252147247767*pow(eta,2.)
                        - 1.3258252147247767*pow(xi,2.)
                        + 3.97747564417433*pow(xi,2.)*pow(eta,2.);
         shape( 75,2) = 2.6516504294495533*eta*nu
                        - 7.95495128834866*pow(xi,2.)*eta*nu;
         shape( 76,2) = 0.4419417382415922 - 1.3258252147247767*pow(eta,2.)
                        - 1.3258252147247767*pow(xi,2.)
                        + 3.97747564417433*pow(xi,2.)*pow(eta,2.);
         shape( 77,0) = -2.4302777619029476*xi*nu
                        + 4.050462936504912*pow(xi,3.)*nu;
         shape( 77,2) = -0.4050462936504913 + 1.2151388809514738*pow(nu,2.)
                        + 2.025231468252456*pow(xi,2.)
                        - 6.075694404757369*pow(xi,2.)*pow(nu,2.);
         shape( 78,1) = -2.4302777619029476*xi*nu
                        + 4.050462936504912*pow(xi,3.)*nu;
         shape( 79,0) = -2.4302777619029476*xi*eta
                        + 4.050462936504912*pow(xi,3.)*eta;
         shape( 79,2) = 2.4302777619029476*eta*nu
                        - 12.151388809514739*pow(xi,2.)*eta*nu;
         shape( 80,1) = 2.4302777619029476*xi*eta
                        + 4.050462936504912*pow(xi,3.)*eta;
         shape( 80,2) = 2.4302777619029476*xi*nu
                        - 4.050462936504912*pow(xi,3.)*nu;
         shape( 81,2) = -2.4302777619029476*xi*eta
                        + 4.050462936504912*pow(xi,3.)*eta;
         shape( 82,0) = 0.397747564417433 - 3.97747564417433*pow(xi,2.)
                        + 4.640388251536718*pow(xi,4.);
         shape( 82,2) = 7.95495128834866*xi*nu
                        - 18.561553006146873*pow(xi,3.)*nu;
         shape( 83,1) = 0.397747564417433 - 3.97747564417433*pow(xi,2.)
                        + 4.640388251536718*pow(xi,4.);
         shape( 84,2) = 0.397747564417433 - 3.97747564417433*pow(xi,2.)
                        + 4.640388251536718*pow(xi,4.);
      case 3:
         shape( 26,0) = -1.403121520040228*nu + 2.3385358667337135*pow(nu,3.);
         shape( 27,1) = -1.403121520040228*nu + 2.3385358667337135*pow(nu,3.);
         shape( 28,0) = -0.6846531968814576*eta
                        + 2.053959590644373*eta*pow(nu,2.);
         shape( 29,1) = -0.6846531968814576*eta
                        + 2.053959590644373*eta*pow(nu,2.);
         shape( 29,2) = 0.6846531968814576*nu - 0.6846531968814576*pow(nu,3.);
         shape( 30,0) = -0.6846531968814576*nu
                        + 2.053959590644373*pow(eta,2.)*nu;
         shape( 31,1) = -0.6846531968814576*nu
                        + 2.053959590644373*pow(eta,2.)*nu;
         shape( 31,2) = 0.6846531968814576*eta
                        - 2.053959590644373*eta*pow(nu,2.);
         shape( 32,0) = -1.403121520040228*eta
                        + 2.3385358667337135*pow(eta,3.);
         shape( 33,1) = -1.403121520040228*eta
                        + 2.3385358667337135*pow(eta,3.);
         shape( 33,2) = 1.403121520040228*nu
                        - 7.0156076002011405*pow(eta,2.)*nu;
         shape( 34,2) = -1.403121520040228*eta
                        + 2.3385358667337135*pow(eta,3.);
         shape( 35,0) = -0.6846531968814576*xi
                        + 2.053959590644373*xi*pow(nu,2.);
         shape( 35,2) = 0.6846531968814576*nu - 0.6846531968814576*pow(nu,3.);
         shape( 36,1) = -0.6846531968814576*xi
                        + 2.053959590644373*xi*pow(nu,2.);
         shape( 37,0) = 1.8371173070873836*xi*eta*nu;
         shape( 37,2) = 0.30618621784789724*eta
                        - 0.9185586535436918*eta*pow(nu,2.);
         shape( 38,1) = 1.8371173070873836*xi*eta*nu;
         shape( 38,2) = 0.30618621784789724*xi
                        - 0.9185586535436918*xi*pow(nu,2.);
         shape( 39,0) = -0.6846531968814576*xi
                        + 2.053959590644373*xi*pow(eta,2.);
         shape( 39,2) = 0.6846531968814576*nu
                        - 2.053959590644373*pow(eta,2.)*nu;
         shape( 40,1) = -0.6846531968814576*xi
                        + 2.053959590644373*xi*pow(eta,2.);
         shape( 40,2) = -4.107919181288746*xi*eta*nu;
         shape( 41,2) = -0.6846531968814576*xi
                        + 2.053959590644373*xi*pow(eta,2.);
         shape( 42,0) = -0.6846531968814576*nu
                        + 2.053959590644373*pow(xi,2.)*nu;
         shape( 42,2) = 0.6846531968814576*xi
                        - 2.053959590644373*xi*pow(nu,2.);
         shape( 43,1) = -0.6846531968814576*nu
                        + 2.053959590644373*pow(xi,2.)*nu;
         shape( 44,0) = -0.6846531968814576*eta
                        + 2.053959590644373*pow(xi,2.)*eta;
         shape( 44,2) = -4.107919181288746*xi*eta*nu;
         shape( 45,1) = -0.6846531968814576*eta
                        + 2.053959590644373*pow(xi,2.)*eta;
         shape( 45,2) = 0.6846531968814576*nu
                        - 2.053959590644373*pow(xi,2.)*nu;
         shape( 46,2) = -0.6846531968814576*eta
                        + 2.053959590644373*pow(xi,2.)*eta;
         shape( 47,0) = -1.403121520040228*xi + 2.3385358667337135*pow(xi,3.);
         shape( 47,2) = 1.403121520040228*nu
                        - 7.0156076002011405*pow(xi,2.)*nu;
         shape( 48,1) = -1.403121520040228*xi + 2.3385358667337135*pow(xi,3.);
         shape( 49,2) = -1.403121520040228*xi + 2.3385358667337135*pow(xi,3.);
      case 2:
         shape( 11,0) = -0.39528470752104744 + 1.1858541225631423*pow(nu,2.);
         shape( 12,1) = -0.39528470752104744 + 1.1858541225631423*pow(nu,2.);
         shape( 13,0) = 1.0606601717798212*eta*nu;
         shape( 14,1) = 1.0606601717798212*eta*nu;
         shape( 14,2) = 0.1767766952966369 - 0.5303300858899106*pow(nu,2.);
         shape( 15,0) = -0.39528470752104744 + 1.1858541225631423*pow(eta,2.);
         shape( 16,1) = -0.39528470752104744 + 1.1858541225631423*pow(eta,2.);
         shape( 16,2) = -2.3717082451262845*eta*nu;
         shape( 17,2) = -0.39528470752104744 + 1.1858541225631423*pow(eta,2.);
         shape( 18,0) = 1.0606601717798212*xi*nu;
         shape( 18,2) = 0.1767766952966369 - 0.5303300858899106*pow(nu,2.);
         shape( 19,1) = 1.0606601717798212*xi*nu;
         shape( 20,0) = 1.0606601717798212*xi*eta;
         shape( 20,2) = -1.0606601717798212*eta*nu;
         shape( 21,1) = 1.0606601717798212*xi*eta;
         shape( 21,2) = -1.0606601717798212*xi*nu;
         shape( 22,2) = 1.0606601717798212*xi*eta;
         shape( 23,0) = -0.39528470752104744 + 1.1858541225631423*pow(xi,2.);
         shape( 23,2) = -2.3717082451262845*xi*nu;
         shape( 24,1) = -0.39528470752104744 + 1.1858541225631423*pow(xi,2.);
         shape( 25,2) = -0.39528470752104744 + 1.1858541225631423*pow(xi,2.);
      case 1:
         shape(  3,0) = 0.6123724356957945*nu;
         shape(  4,1) = 0.6123724356957945*nu;
         shape(  5,0) = 0.6123724356957945*eta;
         shape(  6,1) = 0.6123724356957945*eta;
         shape(  6,2) = -0.6123724356957945*nu;
         shape(  7,2) = 0.6123724356957945*eta;
         shape(  8,0) = 0.6123724356957945*xi;
         shape(  8,2) = -0.6123724356957945*nu;
         shape(  9,1) = 0.6123724356957945*xi;
         shape( 10,2) = 0.6123724356957945*xi;
      case 0:
         shape(  0,0) = 0.3535533905932738;
         shape(  1,1) = 0.3535533905932738;
         shape(  2,2) = 0.3535533905932738;
   }
}

}

}

#endif
