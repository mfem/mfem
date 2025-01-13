// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_BILININTEG
#define MFEM_BILININTEG

#include "../config/config.hpp"
#include "nonlininteg.hpp"
#include "fespace.hpp"
#include "ceed/interface/util.hpp"
#include "qfunction.hpp"
#include <memory>

#include "kernel_dispatch.hpp"

namespace mfem
{

/// Abstract base class BilinearFormIntegrator
class BilinearFormIntegrator : public NonlinearFormIntegrator
{
protected:
   BilinearFormIntegrator(const IntegrationRule *ir = NULL)
      : NonlinearFormIntegrator(ir) { }

public:
   // TODO: add support for other assembly levels (in addition to PA) and their
   // actions.

   // TODO: for mixed meshes the quadrature rules to be used by methods like
   // AssemblePA() can be given as a QuadratureSpace, e.g. using a new method:
   // SetQuadratureSpace().

   // TODO: the methods for the various assembly levels make sense even in the
   // base class NonlinearFormIntegrator, except that not all assembly levels
   // make sense for the action of the nonlinear operator (but they all make
   // sense for its Jacobian).

   /// Method defining partial assembly.
   /** The result of the partial assembly is stored internally so that it can be
       used later in the methods AddMultPA() and AddMultTransposePA(). */
   void AssemblePA(const FiniteElementSpace &fes) override;
   /** Used with BilinearFormIntegrators that have different spaces. */
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   /// Method defining partial assembly on NURBS patches.
   /** The result of the partial assembly is stored internally so that it can be
       used later in the method AddMultNURBSPA(). */
   virtual void AssembleNURBSPA(const FiniteElementSpace &fes);

   virtual void AssemblePABoundary(const FiniteElementSpace &fes);

   virtual void AssemblePAInteriorFaces(const FiniteElementSpace &fes);

   virtual void AssemblePABoundaryFaces(const FiniteElementSpace &fes);

   /// Assemble diagonal and add it to Vector @a diag.
   virtual void AssembleDiagonalPA(Vector &diag);

   /// Assemble diagonal of $A D A^T$ ($A$ is this integrator) and add it to @a diag.
   virtual void AssembleDiagonalPA_ADAt(const Vector &D, Vector &diag);

   /// Method for partially assembled action.
   /** Perform the action of integrator on the input @a x and add the result to
       the output @a y. Both @a x and @a y are E-vectors, i.e. they represent
       the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssemblePA() has been
       called. */
   void AddMultPA(const Vector &x, Vector &y) const override;

   /// Method for partially assembled action on NURBS patches.
   virtual void AddMultNURBSPA(const Vector&x, Vector&y) const;

   /// Method for partially assembled transposed action.
   /** Perform the transpose action of integrator on the input @a x and add the
       result to the output @a y. Both @a x and @a y are E-vectors, i.e. they
       represent the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssemblePA() has been
       called. */
   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;

   /// Method defining element assembly.
   /** The result of the element assembly is added to the @a emat Vector if
       @a add is true. Otherwise, if @a add is false, we set @a emat. */
   virtual void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                           const bool add = true);
   /** Used with BilinearFormIntegrators that have different spaces. */
   // virtual void AssembleEA(const FiniteElementSpace &trial_fes,
   //                         const FiniteElementSpace &test_fes,
   //                         Vector &emat);

   /// Method defining matrix-free assembly.
   /** The result of fully matrix-free assembly is stored internally so that it
       can be used later in the methods AddMultMF() and AddMultTransposeMF(). */
   void AssembleMF(const FiniteElementSpace &fes) override;

   /** Perform the action of integrator on the input @a x and add the result to
       the output @a y. Both @a x and @a y are E-vectors, i.e. they represent
       the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssembleMF() has been
       called. */
   void AddMultMF(const Vector &x, Vector &y) const override;

   /** Perform the transpose action of integrator on the input @a x and add the
       result to the output @a y. Both @a x and @a y are E-vectors, i.e. they
       represent the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssemblePA() has been
       called. */
   virtual void AddMultTransposeMF(const Vector &x, Vector &y) const;

   /// Assemble diagonal and add it to Vector @a diag.
   virtual void AssembleDiagonalMF(Vector &diag);

   virtual void AssembleEAInteriorFaces(const FiniteElementSpace &fes,
                                        Vector &ea_data_int,
                                        Vector &ea_data_ext,
                                        const bool add = true);

   virtual void AssembleEABoundaryFaces(const FiniteElementSpace &fes,
                                        Vector &ea_data_bdr,
                                        const bool add = true);

   /// Given a particular Finite Element computes the element matrix elmat.
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   /** Compute the local matrix representation of a bilinear form
       $a(u,v)$ defined on different trial (given by $u$) and test
       (given by $v$) spaces. The rows in the local matrix correspond
       to the test dofs and the columns -- to the trial dofs. */
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   /** Given a particular NURBS patch, computes the patch matrix as a
       SparseMatrix @a smat.
    */
   virtual void AssemblePatchMatrix(const int patch,
                                    const FiniteElementSpace &fes,
                                    SparseMatrix*& smat);

   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   virtual void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &trial_fe2,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   /** Abstract method used for assembling TraceFaceIntegrators in a
       MixedBilinearForm. */
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   /** Abstract method used for assembling TraceFaceIntegrators for
       DPG weak formulations. */
   virtual void AssembleTraceFaceMatrix(int elem,
                                        const FiniteElement &trial_face_fe,
                                        const FiniteElement &test_fe,
                                        FaceElementTransformations &Trans,
                                        DenseMatrix &elmat);


   /// @brief Perform the local action of the BilinearFormIntegrator.
   /// Note that the default implementation in the base class is general but not
   /// efficient.
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   /// @brief Perform the local action of the BilinearFormIntegrator resulting
   /// from a face integral term.
   /// Note that the default implementation in the base class is general but not
   /// efficient.
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override
   { AssembleElementMatrix(el, Tr, elmat); }

   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override
   { AssembleFaceMatrix(el1, el2, Tr, elmat); }

   /** @brief Virtual method required for Zienkiewicz-Zhu type error estimators.

       The purpose of the method is to compute a local "flux" finite element
       function given a local finite element solution. The "flux" function has
       to be computed in terms of its coefficients (represented by the Vector
       @a flux) which multiply the basis functions defined by the FiniteElement
       @a fluxelem. Typically, the "flux" function will have more than one
       component and consequently @a flux should be store the coefficients of
       all components: first all coefficient for component 0, then all
       coefficients for component 1, etc. What the "flux" function represents
       depends on the specific integrator. For example, in the case of
       DiffusionIntegrator, the flux is the gradient of the solution multiplied
       by the diffusion coefficient.

       @param[in] el     FiniteElement of the solution.
       @param[in] Trans  The ElementTransformation describing the physical
                         position of the mesh element.
       @param[in] u      Solution coefficients representing the expansion of the
                         solution function in the basis of @a el.
       @param[in] fluxelem  FiniteElement of the "flux".
       @param[out] flux  "Flux" coefficients representing the expansion of the
                         "flux" function in the basis of @a fluxelem. The size
                         of @a flux as a Vector has to be set by this method,
                         e.g. using Vector::SetSize().
       @param[in] with_coef  If zero (the default value is 1) the implementation
                             of the method may choose not to scale the "flux"
                             function by any coefficients describing the
                             integrator.
       @param[in] ir  If passed (the default value is NULL), the implementation
                      of the method will ignore the integration rule provided
                      by the @a fluxelem parameter and, instead, compute the
                      discrete flux at the points specified by the integration
                      rule @a ir.
    */
   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u,
                                   const FiniteElement &fluxelem,
                                   Vector &flux, bool with_coef = true,
                                   const IntegrationRule *ir = NULL) { }

   /** @brief Virtual method required for Zienkiewicz-Zhu type error estimators.

       The purpose of this method is to compute a local number that measures the
       energy of a given "flux" function (see ComputeElementFlux() for a
       description of the "flux" function). Typically, the energy of a "flux"
       function should be equal to a_local(u,u), if the "flux" is defined from
       a solution u; here a_local(.,.) denotes the element-local bilinear
       form represented by the integrator.

       @param[in] fluxelem  FiniteElement of the "flux".
       @param[in] Trans  The ElementTransformation describing the physical
                         position of the mesh element.
       @param[in] flux   "Flux" coefficients representing the expansion of the
                         "flux" function in the basis of @a fluxelem.
       @param[out] d_energy  If not NULL, the given Vector should be set to
                             represent directional energy split that can be used
                             for anisotropic error estimation.
       @returns The computed energy.
    */
   virtual real_t ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL)
   { return 0.0; }

   /** @brief For bilinear forms on element faces, specifies if the normal
              derivatives are needed on the faces or just the face restriction.

       @details if RequiresFaceNormalDerivatives() == true, then
                AddMultPAFaceNormalDerivatives(...) should be invoked in place
                of AddMultPA(...) and L2NormalDerivativeFaceRestriction should
                be used to compute the normal derivatives. This is used for some
                DG integrators, for example DGDiffusionIntegrator.

       @returns whether normal derivatives appear in the bilinear form.
   */
   virtual bool RequiresFaceNormalDerivatives() const { return false; }

   /// Method for partially assembled action.
   /** @brief For bilinear forms on element faces that depend on the normal
              derivative on the faces, computes the action of integrator to the
              face values @a x and reference-normal derivatives @a dxdn and adds
              the result to @a y and @a dydn.

      @details This method can be called only after the method AssemblePA() has
               been called.

      @param[in]     x E-vector of face values (provided by
                       FaceRestriction::Mult)
      @param[in]     dxdn E-vector of face reference-normal derivatives
                          (provided by FaceRestriction::NormalDerivativeMult)
      @param[in,out] y E-vector of face values to add action to.
      @param[in,out] dydn E-vector of face reference-normal derivative values to
                          add action to.
   */
   virtual void AddMultPAFaceNormalDerivatives(const Vector &x, const Vector &dxdn,
                                               Vector &y, Vector &dydn) const;

   virtual ~BilinearFormIntegrator() { }
};

/** Wraps a given @a BilinearFormIntegrator and transposes the resulting element
    matrices. See for example ex9, ex9p. */
class TransposeIntegrator : public BilinearFormIntegrator
{
private:
   int own_bfi;
   BilinearFormIntegrator *bfi;

   DenseMatrix bfi_elmat;

public:
   TransposeIntegrator(BilinearFormIntegrator *bfi_, int own_bfi_ = 1)
   { bfi = bfi_; own_bfi = own_bfi_; }

   void SetIntRule(const IntegrationRule *ir) override;

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                           const FiniteElement &test_fe1,
                           const FiniteElement &trial_fe2,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssemblePA(const FiniteElementSpace& fes) override
   {
      bfi->AssemblePA(fes);
   }

   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override
   {
      bfi->AssemblePA(test_fes, trial_fes); // Reverse test and trial
   }

   void AssemblePAInteriorFaces(const FiniteElementSpace &fes) override
   {
      bfi->AssemblePAInteriorFaces(fes);
   }

   void AssemblePABoundaryFaces(const FiniteElementSpace &fes) override
   {
      bfi->AssemblePABoundaryFaces(fes);
   }

   void AddMultTransposePA(const Vector &x, Vector &y) const override
   {
      bfi->AddMultPA(x, y);
   }

   void AddMultPA(const Vector& x, Vector& y) const override
   {
      bfi->AddMultTransposePA(x, y);
   }

   void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                   const bool add) override;

   void AssembleEAInteriorFaces(const FiniteElementSpace &fes,
                                Vector &ea_data_int,
                                Vector &ea_data_ext,
                                const bool add) override;

   void AssembleEABoundaryFaces(const FiniteElementSpace &fes,
                                Vector &ea_data_bdr,
                                const bool add) override;

   virtual ~TransposeIntegrator() { if (own_bfi) { delete bfi; } }
};

class LumpedIntegrator : public BilinearFormIntegrator
{
private:
   int own_bfi;
   BilinearFormIntegrator *bfi;

public:
   LumpedIntegrator (BilinearFormIntegrator *bfi_, int own_bfi_ = 1)
   { bfi = bfi_; own_bfi = own_bfi_; }

   void SetIntRule(const IntegrationRule *ir) override;

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   virtual ~LumpedIntegrator() { if (own_bfi) { delete bfi; } }
};

/// Integrator that inverts the matrix assembled by another integrator.
class InverseIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrator;
   BilinearFormIntegrator *integrator;

public:
   InverseIntegrator(BilinearFormIntegrator *integ, int own_integ = 1)
   { integrator = integ; own_integrator = own_integ; }

   void SetIntRule(const IntegrationRule *ir) override;

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   virtual ~InverseIntegrator() { if (own_integrator) { delete integrator; } }
};

/// Integrator defining a sum of multiple Integrators.
class SumIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrators;
   mutable DenseMatrix elem_mat;
   Array<BilinearFormIntegrator*> integrators;

public:
   SumIntegrator(int own_integs = 1) { own_integrators = own_integs; }

   void SetIntRule(const IntegrationRule *ir) override;

   void AddIntegrator(BilinearFormIntegrator *integ)
   { integrators.Append(integ); }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace& fes) override;

   void AssembleDiagonalPA(Vector &diag) override;

   void AssemblePAInteriorFaces(const FiniteElementSpace &fes) override;

   void AssemblePABoundaryFaces(const FiniteElementSpace &fes) override;

   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   void AddMultPA(const Vector& x, Vector& y) const override;

   void AssembleMF(const FiniteElementSpace &fes) override;

   void AddMultMF(const Vector &x, Vector &y) const override;

   void AddMultTransposeMF(const Vector &x, Vector &y) const override;

   void AssembleDiagonalMF(Vector &diag) override;

   void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                   const bool add) override;

   void AssembleEAInteriorFaces(const FiniteElementSpace &fes,
                                Vector &ea_data_int,
                                Vector &ea_data_ext,
                                const bool add) override;

   void AssembleEABoundaryFaces(const FiniteElementSpace &fes,
                                Vector &ea_data_bdr,
                                const bool add) override;

   virtual ~SumIntegrator();
};

/** An abstract class for integrating the product of two scalar basis functions
    with an optional scalar coefficient. */
class MixedScalarIntegrator: public BilinearFormIntegrator
{
public:
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   /// Support for use in BilinearForm. Can be used only when appropriate.
   void AssembleElementMatrix(const FiniteElement &fe,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override
   { AssembleElementMatrix2(fe, fe, Trans, elmat); }

protected:
   /// This parameter can be set by derived methods to enable single shape
   /// evaluation in case CalcTestShape() and CalcTrialShape() return the same
   /// result if given the same FiniteElement. The default is false.
   bool same_calc_shape;

   MixedScalarIntegrator() : same_calc_shape(false), Q(NULL) {}
   MixedScalarIntegrator(Coefficient &q) : same_calc_shape(false), Q(&q) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarIntegrator:  "
             "Trial and test spaces must both be scalar fields.";
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }


   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     Vector & shape)
   { test_fe.CalcPhysShape(Trans, shape); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      Vector & shape)
   { trial_fe.CalcPhysShape(Trans, shape); }

   Coefficient *Q;

private:

#ifndef MFEM_THREAD_SAFE
   Vector test_shape;
   Vector trial_shape;
#endif

};

/** An abstract class for integrating the inner product of two vector basis
    functions with an optional scalar, vector, or matrix coefficient. */
class MixedVectorIntegrator: public BilinearFormIntegrator
{
public:

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   /// Support for use in BilinearForm. Can be used only when appropriate.
   void AssembleElementMatrix(const FiniteElement &fe,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override
   { AssembleElementMatrix2(fe, fe, Trans, elmat); }

protected:
   /// This parameter can be set by derived methods to enable single shape
   /// evaluation in case CalcTestShape() and CalcTrialShape() return the same
   /// result if given the same FiniteElement. The default is false.
   bool same_calc_shape;

   MixedVectorIntegrator()
      : same_calc_shape(false), Q(NULL), VQ(NULL), DQ(NULL), MQ(NULL) {}
   MixedVectorIntegrator(Coefficient &q)
      : same_calc_shape(false), Q(&q), VQ(NULL), DQ(NULL), MQ(NULL) {}
   MixedVectorIntegrator(VectorCoefficient &vq, bool diag = true)
      : same_calc_shape(false), Q(NULL), VQ(diag?NULL:&vq), DQ(diag?&vq:NULL),
        MQ(NULL) {}
   MixedVectorIntegrator(MatrixCoefficient &mq)
      : same_calc_shape(false), Q(NULL), VQ(NULL), DQ(NULL), MQ(&mq) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedVectorIntegrator:  "
             "Trial and test spaces must both be vector fields";
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }


   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return std::max(space_dim, test_fe.GetRangeDim()); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcVShape(Trans, shape); }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return std::max(space_dim, trial_fe.GetRangeDim()); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcVShape(Trans, shape); }

   int space_dim;
   Coefficient *Q;
   VectorCoefficient *VQ;
   DiagonalMatrixCoefficient *DQ;
   MatrixCoefficient *MQ;

private:

#ifndef MFEM_THREAD_SAFE
   Vector V;
   Vector D;
   DenseMatrix M;
   DenseMatrix test_shape;
   DenseMatrix trial_shape;
   DenseMatrix shape_tmp;
#endif

};

/** An abstract class for integrating the product of a scalar basis function and
    the inner product of a vector basis function with a vector coefficient. In
    2D the inner product can be replaced with a cross product. */
class MixedScalarVectorIntegrator: public BilinearFormIntegrator
{
public:

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   /// Support for use in BilinearForm. Can be used only when appropriate.
   /** Appropriate use cases are classes derived from
       MixedScalarVectorIntegrator where the trial and test spaces can be the
       same. Examples of such classes are: MixedVectorDivergenceIntegrator,
       MixedScalarWeakDivergenceIntegrator, etc. */
   void AssembleElementMatrix(const FiniteElement &fe,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override
   { AssembleElementMatrix2(fe, fe, Trans, elmat); }

protected:

   MixedScalarVectorIntegrator(VectorCoefficient &vq, bool transpose_ = false,
                               bool cross_2d_ = false)
      : VQ(&vq), transpose(transpose_), cross_2d(cross_2d_) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return ((transpose &&
               trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
               test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR ) ||
              (!transpose &&
               trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
               test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR )
             );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      if ( transpose )
      {
         return "MixedScalarVectorIntegrator:  "
                "Trial space must be a vector field "
                "and the test space must be a scalar field";
      }
      else
      {
         return "MixedScalarVectorIntegrator:  "
                "Trial space must be a scalar field "
                "and the test space must be a vector field";
      }
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW(); }


   inline virtual int GetVDim(const FiniteElement & vector_fe)
   { return std::max(space_dim, vector_fe.GetRangeDim()); }

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape_)
   { vector_fe.CalcVShape(Trans, shape_); }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape_)
   { scalar_fe.CalcPhysShape(Trans, shape_); }

   VectorCoefficient *VQ;
   int space_dim;
   bool transpose;
   bool cross_2d;  // In 2D use a cross product rather than a dot product

private:

#ifndef MFEM_THREAD_SAFE
   Vector V;
   DenseMatrix vshape;
   Vector      shape;
   Vector      vshape_tmp;
#endif

};

/** Class for integrating the bilinear form $a(u,v) := (Q u, v)$ in either 1D, 2D,
    or 3D and where $Q$ is an optional scalar coefficient, $u$ and $v$ are each in $H^1$
    or $L_2$. */
class MixedScalarMassIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarMassIntegrator() { same_calc_shape = true; }
   MixedScalarMassIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) { same_calc_shape = true; }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} u, v)$ in either 2D, or
    3D and where $\vec{V}$ is a vector coefficient, $u$ is in $H^1$ or $L_2$ and $v$ is in $H(curl$
    or $H(div)$. */
class MixedVectorProductIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedVectorProductIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq) {}
};

/** Class for integrating the bilinear form $a(u,v) := (Q \nabla u, v)$ in 1D where Q
    is an optional scalar coefficient, $u$ is in $H^1$, and $v$ is in $L_2$. */
class MixedScalarDerivativeIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarDerivativeIntegrator() {}
   MixedScalarDerivativeIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 1 && test_fe.GetDim() == 1 &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD  &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarDerivativeIntegrator:  "
             "Trial and test spaces must both be scalar fields in 1D "
             "and the trial space must implement CalcDShape.";
   }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      trial_fe.CalcPhysDShape(Trans, dshape);
   }
};

/** Class for integrating the bilinear form $a(u,v) := -(Q u, \nabla v)$ in 1D where $Q$
    is an optional scalar coefficient, $u$ is in $L_2$, and $v$ is in $H^1$. */
class MixedScalarWeakDerivativeIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarWeakDerivativeIntegrator() {}
   MixedScalarWeakDerivativeIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 1 && test_fe.GetDim() == 1 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarWeakDerivativeIntegrator:  "
             "Trial and test spaces must both be scalar fields in 1D "
             "and the test space must implement CalcDShape with "
             "map type \"VALUE\".";
   }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      test_fe.CalcPhysDShape(Trans, dshape);
      shape *= -1.0;
   }
};

/** Class for integrating the bilinear form $a(u,v) := (Q \nabla \cdot u, v)$ in either 2D
    or 3D where $Q$ is an optional scalar coefficient, $u$ is in $H(div)$, and $v$ is a
    scalar field. */
class MixedScalarDivergenceIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarDivergenceIntegrator() {}
   MixedScalarDivergenceIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDerivType() == mfem::FiniteElement::DIV  &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarDivergenceIntegrator:  "
             "Trial must be $H(div)$ and the test space must be a "
             "scalar field";
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1; }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      Vector & shape)
   { trial_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \nabla \cdot u, v)$ in either 2D
    or 3D where $\vec{V}$ is a vector coefficient, $u$ is in $H(div)$, and $v$ is in $H(div)$. */
class MixedVectorDivergenceIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedVectorDivergenceIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDerivType() == mfem::FiniteElement::DIV  &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedVectorDivergenceIntegrator:  "
             "Trial must be H(Div) and the test space must be a "
             "vector field";
   }

   // Subtract one due to the divergence and add one for the coefficient
   // which is assumed to be at least linear.
   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1 + 1; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := -(Q u, \nabla \cdot v)$ in either 2D
    or 3D where $Q$ is an optional scalar coefficient, $u$ is in $L_2$ or $H^1$, and $v$ is
    in $H(div)$. */
class MixedScalarWeakGradientIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarWeakGradientIntegrator() {}
   MixedScalarWeakGradientIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::DIV );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarWeakGradientIntegrator:  "
             "Trial space must be a scalar field "
             "and the test space must be H(Div)";
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1; }

   virtual void CalcTestShape(const FiniteElement & test_fe,
                              ElementTransformation &Trans,
                              Vector & shape)
   {
      test_fe.CalcPhysDivShape(Trans, shape);
      shape *= -1.0;
   }
};

/** Class for integrating the bilinear form $a(u,v) := (Q \mathrm{curl}(u), v)$ in 2D where
    $Q$ is an optional scalar coefficient, $u$ is in $H(curl$, and $v$ is in $L_2$ or
    $H^1$. */
class MixedScalarCurlIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarCurlIntegrator() {}
   MixedScalarCurlIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const override
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR);
   }

   inline const char * FiniteElementTypeFailureMessage() const override
   {
      return "MixedScalarCurlIntegrator:  "
             "Trial must be H(Curl) and the test space must be a "
             "scalar field";
   }

   inline int GetIntegrationOrder(const FiniteElement & trial_fe,
                                  const FiniteElement & test_fe,
                                  ElementTransformation &Trans) override
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1; }

   inline void CalcTrialShape(const FiniteElement & trial_fe,
                              ElementTransformation &Trans,
                              Vector & shape) override
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      trial_fe.CalcPhysCurlShape(Trans, dshape);
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector&, Vector&) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   int dim, ne, dofs1D, quad1D, dofs1Dtest;
};

/** Class for integrating the bilinear form $a(u,v) := (Q u, \mathrm{curl}(v))$ in 2D where
    $Q$ is an optional scalar coefficient, $u$ is in $L_2$ or $H^1$, and $v$ is in
    $H(curl$. Partial assembly (PA) is supported but could be further optimized
    by using more efficient threading and shared memory.
*/
class MixedScalarWeakCurlIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarWeakCurlIntegrator() {}
   MixedScalarWeakCurlIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarWeakCurlIntegrator:  "
             "Trial space must be a scalar field "
             "and the test space must be H(Curl)";
   }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      test_fe.CalcPhysCurlShape(Trans, dshape);
   }
};

/** Class for integrating the bilinear form $a(u,v) := (Q u, v)$ in either 2D or
    3D and where $Q$ is an optional coefficient (of type scalar, matrix, or
    diagonal matrix) $u$ and $v$ are each in $H(curl$ or $H(div)$. */
class MixedVectorMassIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorMassIntegrator() { same_calc_shape = true; }
   MixedVectorMassIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) { same_calc_shape = true; }
   MixedVectorMassIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) { same_calc_shape = true; }
   MixedVectorMassIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) { same_calc_shape = true; }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times u, v)$ in 3D and where
    $\vec{V}$ is a vector coefficient $u$ and $v$ are each in $H(curl$ or $H(div)$. */
class MixedCrossProductIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossProductIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) { same_calc_shape = true; }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \cdot u, v)$ in 2D or 3D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ or $H(div)$ and $v$ is in $H^1$ or
    $L_2$. */
class MixedDotProductIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedDotProductIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedDotProductIntegrator:  "
             "Trial space must be a vector field "
             "and the test space must be a scalar field";
   }
};

/** Class for integrating the bilinear form $a(u,v) := (-\vec{V} \cdot u, \nabla \cdot v)$ in 2D or
    3D and where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ or $H(div)$ and $v$ is in
    $H(div)$. */
class MixedWeakGradDotIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedWeakGradDotIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::DIV );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedWeakGradDotIntegrator:  "
             "Trial space must be a vector field "
             "and the test space must be a vector field with a divergence";
   }

   // Subtract one due to the gradient and add one for the coefficient
   // which is assumed to be at least linear.
   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1 + 1; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form $a(u,v) := (v \vec{V} \times u, \nabla v)$ in 3D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ or $H(div)$ and $v$ is in $H^1$. */
class MixedWeakDivCrossIntegrator : public MixedVectorIntegrator
{
public:
   MixedWeakDivCrossIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedWeakDivCrossIntegrator:  "
             "Trial space must be a vector field in 3D "
             "and the test space must be a scalar field with a gradient";
   }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return space_dim; }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form $a(u,v) := (Q \nabla u, \nabla v)$ in 3D
    or in 2D and where $Q$ is a scalar or matrix coefficient $u$ and $v$ are both in
    $H^1$. */
class MixedGradGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedGradGradIntegrator() { same_calc_shape = true; }
   MixedGradGradIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) { same_calc_shape = true; }
   MixedGradGradIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) { same_calc_shape = true; }
   MixedGradGradIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) { same_calc_shape = true; }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedGradGradIntegrator:  "
             "Trial and test spaces must both be scalar fields "
             "with a gradient operator.";
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   {
      // Same as DiffusionIntegrator
      return test_fe.Space() == FunctionSpace::Pk ?
             trial_fe.GetOrder() + test_fe.GetOrder() - 2 :
             trial_fe.GetOrder() + test_fe.GetOrder() + test_fe.GetDim() - 1;
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return space_dim; }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return space_dim; }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \nabla u, \nabla v)$ in 3D
    or in 2D and where $\vec{V}$ is a vector coefficient $u$ and $v$ are both in $H^1$. */
class MixedCrossGradGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossGradGradIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) { same_calc_shape = true; }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossGradGradIntegrator:  "
             "Trial and test spaces must both be scalar fields "
             "with a gradient operator.";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return space_dim; }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return space_dim; }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (Q \mathrm{curl}(u), \mathrm{curl}(v))$ in 3D
    and where $Q$ is a scalar or matrix coefficient $u$ and $v$ are both in
    $H(curl$. */
class MixedCurlCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCurlCurlIntegrator() { same_calc_shape = true; }
   MixedCurlCurlIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) { same_calc_shape = true; }
   MixedCurlCurlIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) { same_calc_shape = true; }
   MixedCurlCurlIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) { same_calc_shape = true; }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetCurlDim() == 3 && test_fe.GetCurlDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCurlCurlIntegrator"
             "Trial and test spaces must both be vector fields in 3D "
             "with a curl.";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return trial_fe.GetCurlDim(); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return test_fe.GetCurlDim(); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \mathrm{curl}(u), \mathrm{curl}(v))$ in 3D
    and where $\vec{V}$ is a vector coefficient $u$ and $v$ are both in $H(curl$. */
class MixedCrossCurlCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossCurlCurlIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) { same_calc_shape = true; }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetCurlDim() == 3 && trial_fe.GetRangeDim() == 3 &&
              test_fe.GetCurlDim() == 3 && test_fe.GetRangeDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossCurlCurlIntegrator:  "
             "Trial and test spaces must both be vector fields in 3D "
             "with a curl.";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return trial_fe.GetCurlDim(); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return test_fe.GetCurlDim(); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \mathrm{curl}(u), \nabla \cdot v)$ in 3D
    and where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ and $v$ is in $H^1$. */
class MixedCrossCurlGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossCurlGradIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetCurlDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossCurlGradIntegrator"
             "Trial space must be a vector field in 3D with a curl"
             "and the test space must be a scalar field with a gradient";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return trial_fe.GetCurlDim(); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return space_dim; }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (v \times \nabla \cdot u, \mathrm{curl}(v))$ in 3D
    and where $v$ is a scalar coefficient $u$ is in $H^1$ and $v$ is in $H(curl$. */
class MixedCrossGradCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossGradCurlIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (test_fe.GetCurlDim() == 3 &&
              trial_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType()  == mfem::FiniteElement::GRAD &&
              test_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType() == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossGradCurlIntegrator"
             "Trial space must be a scalar field in 3D with a gradient"
             "and the test space must be a vector field with a curl";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return space_dim; }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return test_fe.GetCurlDim(); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times u, \mathrm{curl}(v))$ in 3D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ or $H(div)$ and $v$ is in
    $H(curl$. */
class MixedWeakCurlCrossIntegrator : public MixedVectorIntegrator
{
public:
   MixedWeakCurlCrossIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeDim() == 3 && test_fe.GetCurlDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedWeakCurlCrossIntegrator:  "
             "Trial space must be a vector field in 3D "
             "and the test space must be a vector field with a curl";
   }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return test_fe.GetCurlDim(); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times u, \mathrm{curl}(v))$ in 2D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ or $H(div)$ and $v$ is in
    $H(curl$. */
class MixedScalarWeakCurlCrossIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedScalarWeakCurlCrossIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarWeakCurlCrossIntegrator:  "
             "Trial space must be a vector field in 2D "
             "and the test space must be a vector field with a curl";
   }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      scalar_fe.CalcPhysCurlShape(Trans, dshape);
   }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \nabla \cdot u, v)$ in 3D or
    in 2D and where $\vec{V}$ is a vector coefficient $u$ is in $H^1$ and $v$ is in $H(curl$ or
    $H(div)$. */
class MixedCrossGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossGradIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (test_fe.GetRangeDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossGradIntegrator:  "
             "Trial space must be a scalar field with a gradient operator"
             " and the test space must be a vector field both in 3D.";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return space_dim; }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcVShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \mathrm{curl}(u), v)$ in 3D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ and $v$ is in $H(curl$ or
    $H(div)$. */
class MixedCrossCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossCurlIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetCurlDim() == 3 && test_fe.GetRangeDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL   &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossCurlIntegrator:  "
             "Trial space must be a vector field in 3D with a curl "
             "and the test space must be a vector field";
   }

   inline virtual int GetTrialVDim(const FiniteElement & trial_fe)
   { return trial_fe.GetCurlDim(); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \mathrm{curl}(u), v)$ in 2D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ and $v$ is in $H(curl$ or
    $H(div)$. */
class MixedScalarCrossCurlIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedScalarCrossCurlIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, false, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL   &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedCrossCurlIntegrator:  "
             "Trial space must be a vector field in 2D with a curl "
             "and the test space must be a vector field";
   }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      scalar_fe.CalcPhysCurlShape(Trans, dshape); shape *= -1.0;
   }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times \nabla \cdot u, v)$ in 2D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H^1$ and $v$ is in $H^1$ or $L_2$. */
class MixedScalarCrossGradIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedScalarCrossGradIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD   &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarCrossGradIntegrator:  "
             "Trial space must be a scalar field in 2D with a gradient "
             "and the test space must be a scalar field";
   }

   inline int GetVDim(const FiniteElement & vector_fe)
   { return space_dim; }

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times u, v)$ in 2D and where
    $\vec{V}$ is a vector coefficient $u$ is in $H(curl$ or $H(div)$ and $v$ is in $H^1$ or $L_2$. */
class MixedScalarCrossProductIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedScalarCrossProductIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarCrossProductIntegrator:  "
             "Trial space must be a vector field in 2D "
             "and the test space must be a scalar field";
   }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \times u \hat{z}, v)$ in 2D and
    where $\vec{V}$ is a vector coefficient $u$ is in $H^1$ or $L_2$ and $v$ is in $H(curl$ or $H(div)$.

    \todo Documentation what $\hat{z}$ is (also missing in https://mfem.org/bilininteg/).
   */
class MixedScalarWeakCrossProductIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedScalarWeakCrossProductIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, false, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarWeakCrossProductIntegrator:  "
             "Trial space must be a scalar field in 2D "
             "and the test space must be a vector field";
   }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form $a(u,v) := (\vec{V} \cdot \nabla u, v)$ in 2D or
    3D and where $\vec{V}$ is a vector coefficient, $u$ is in $H^1$ and $v$ is in $H^1$ or $L_2$. */
class MixedDirectionalDerivativeIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedDirectionalDerivativeIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD   &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedDirectionalDerivativeIntegrator:  "
             "Trial space must be a scalar field with a gradient "
             "and the test space must be a scalar field";
   }

   inline virtual int GetVDim(const FiniteElement & vector_fe)
   { return space_dim; }

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (-\hat{V} \cdot \nabla u, \nabla \cdot v)$ in 2D
    or 3D and where $\hat{V}$ is a vector coefficient, $u$ is in $H^1$ and $v$ is in $H(div)$. */
class MixedGradDivIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedGradDivIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, true) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::GRAD   &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::DIV   );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedGradDivIntegrator:  "
             "Trial space must be a scalar field with a gradient"
             "and the test space must be a vector field with a divergence";
   }

   inline virtual int GetVDim(const FiniteElement & vector_fe)
   { return space_dim; }

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (-\hat{V} \nabla \cdot u, \nabla v)$ in 2D
    or 3D and where $\hat{V}$ is a vector coefficient, $u$ is in $H(div)$ and $v$ is in $H^1$. */
class MixedDivGradIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedDivGradIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              trial_fe.GetDerivType() == mfem::FiniteElement::DIV    &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD
             );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedDivGradIntegrator:  "
             "Trial space must be a vector field with a divergence"
             "and the test space must be a scalar field with a gradient";
   }

   inline virtual int GetVDim(const FiniteElement & vector_fe)
   { return space_dim; }

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form $a(u,v) := (-\hat{V} u, \nabla v)$ in 2D or 3D
    and where $\hat{V}$ is a vector coefficient, $u$ is in $H^1$ or $L_2$ and $v$ is in $H^1$. */
class MixedScalarWeakDivergenceIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedScalarWeakDivergenceIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::SCALAR &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD   );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarWeakDivergenceIntegrator:  "
             "Trial space must be a scalar field "
             "and the test space must be a scalar field with a gradient";
   }

   inline int GetVDim(const FiniteElement & vector_fe)
   { return space_dim; }

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form $a(u,v) := (Q \nabla u, v)$ in either 2D
    or 3D and where $Q$ is an optional coefficient (of type scalar, matrix, or
    diagonal matrix) $u$ is in $H^1$ and $v$ is in $H(curl$ or $H(div)$. Partial assembly
    (PA) is supported but could be further optimized by using more efficient
    threading and shared memory.
*/
class MixedVectorGradientIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorGradientIntegrator() {}
   MixedVectorGradientIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorGradientIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorGradientIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const override
   {
      return (trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline const char * FiniteElementTypeFailureMessage() const override
   {
      return "MixedVectorGradientIntegrator:  "
             "Trial spaces must be $H^1$ and the test space must be a "
             "vector field in 2D or 3D";
   }

   inline int GetTrialVDim(const FiniteElement & trial_fe) override
   { return space_dim; }

   inline void CalcTrialShape(const FiniteElement & trial_fe,
                              ElementTransformation &Trans,
                              DenseMatrix & shape) override
   {
      trial_fe.CalcPhysDShape(Trans, shape);
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector&, Vector&) const override;
   void AddMultTransposePA(const Vector&, Vector&) const override;

private:
   DenseMatrix Jinv;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, dofs1D, quad1D;
};

/** Class for integrating the bilinear form $a(u,v) := (Q \mathrm{curl}(u), v)$ in 3D and
    where $Q$ is an optional coefficient (of type scalar, matrix, or diagonal
    matrix) $u$ is in $H(curl$ and $v$ is in $H(div)$ or $H(curl$. */
class MixedVectorCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorCurlIntegrator() {}
   MixedVectorCurlIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorCurlIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorCurlIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const override
   {
      return (trial_fe.GetCurlDim() == 3 && test_fe.GetRangeDim() == 3 &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL  &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline const char * FiniteElementTypeFailureMessage() const override
   {
      return "MixedVectorCurlIntegrator:  "
             "Trial space must be H(Curl) and the test space must be a "
             "vector field in 3D";
   }

   inline int GetTrialVDim(const FiniteElement & trial_fe) override
   { return trial_fe.GetCurlDim(); }

   inline void CalcTrialShape(const FiniteElement & trial_fe,
                              ElementTransformation &Trans,
                              DenseMatrix & shape) override
   {
      trial_fe.CalcPhysCurlShape(Trans, shape);
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector&, Vector&) const override;
   void AddMultTransposePA(const Vector&, Vector&) const override;

private:
   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const DofToQuad *mapsOtest;     ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsCtest;     ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, dofs1D, dofs1Dtest,quad1D, testType, trialType, coeffDim;
};

/** Class for integrating the bilinear form $a(u,v) := (Q u, \mathrm{curl}(v))$ in 3D and
    where $Q$ is an optional coefficient (of type scalar, matrix, or diagonal
    matrix) $u$ is in $H(div)$ or $H(curl$ and $v$ is in $H(curl$. */
class MixedVectorWeakCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorWeakCurlIntegrator() {}
   MixedVectorWeakCurlIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorWeakCurlIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorWeakCurlIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const override
   {
      return (trial_fe.GetRangeDim() == 3 && test_fe.GetCurlDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline const char * FiniteElementTypeFailureMessage() const override
   {
      return "MixedVectorWeakCurlIntegrator:  "
             "Trial space must be vector field in 3D and the "
             "test space must be H(Curl)";
   }

   inline int GetTestVDim(const FiniteElement & test_fe) override
   { return test_fe.GetCurlDim(); }

   inline void CalcTestShape (const FiniteElement & test_fe,
                              ElementTransformation &Trans,
                              DenseMatrix & shape) override
   {
      test_fe.CalcPhysCurlShape(Trans, shape);
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector&, Vector&) const override;
   void AddMultTransposePA(const Vector&, Vector&) const override;

private:
   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, dofs1D, quad1D, testType, trialType, coeffDim;
};

/** Class for integrating the bilinear form $a(u,v) := - (Q u, \nabla v)$ in either
    2D or 3D and where $Q$ is an optional coefficient (of type scalar, matrix, or
    diagonal matrix) $u$ is in $H(div)$ or $H(curl$ and $v$ is in $H^1$. */
class MixedVectorWeakDivergenceIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorWeakDivergenceIntegrator() {}
   MixedVectorWeakDivergenceIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorWeakDivergenceIntegrator(DiagonalMatrixCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorWeakDivergenceIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::GRAD );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedVectorWeakDivergenceIntegrator:  "
             "Trial space must be vector field and the "
             "test space must be H1";
   }

   inline virtual int GetTestVDim(const FiniteElement & test_fe)
   { return space_dim; }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   {
      test_fe.CalcPhysDShape(Trans, shape);
      shape *= -1.0;
   }
};

/** Class for integrating the bilinear form $a(u,v) := (Q \nabla u, v)$ where $Q$ is a
    scalar coefficient, $u$ is in ($H^1$), and $v$ is a vector with components
    $v_i$ in ($H^1$) or ($L^2$).

    See also MixedVectorGradientIntegrator when $v$ is in $H(curl$. */
class GradientIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;
   DenseMatrix elmat_comp;
   // PA extension
   Vector pa_data;
   const DofToQuad *trial_maps, *test_maps; ///< Not owned
   const GeometricFactors *geom;            ///< Not owned
   int dim, ne, nq;
   int trial_dofs1D, test_dofs1D, quad1D;

public:
   GradientIntegrator() :
      Q{NULL}, trial_maps{NULL}, test_maps{NULL}, geom{NULL}
   { }
   GradientIntegrator(Coefficient *q_) :
      Q{q_}, trial_maps{NULL}, test_maps{NULL}, geom{NULL}
   { }
   GradientIntegrator(Coefficient &q) :
      Q{&q}, trial_maps{NULL}, test_maps{NULL}, geom{NULL}
   { }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};

/** Class for integrating the bilinear form $a(u,v) := (Q \nabla u, \nabla v)$ where $Q$
    can be a scalar or a matrix coefficient. */
class DiffusionIntegrator: public BilinearFormIntegrator
{
public:

   using ApplyKernelType = void(*)(const int, const bool, const Array<real_t>&,
                                   const Array<real_t>&, const Array<real_t>&,
                                   const Array<real_t>&,
                                   const Vector&, const Vector&,
                                   Vector&, const int, const int);

   using DiagonalKernelType = void(*)(const int, const bool, const Array<real_t>&,
                                      const Array<real_t>&, const Vector&, Vector&,
                                      const int, const int);

   MFEM_REGISTER_KERNELS(ApplyPAKernels, ApplyKernelType, (int, int, int));
   MFEM_REGISTER_KERNELS(DiagonalPAKernels, DiagonalKernelType, (int, int, int));
   struct Kernels { Kernels(); };

protected:
   Coefficient *Q;
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;

private:
   Vector vec, vecdxt, pointflux, shape;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, dshapedxt, invdfdx, M, dshapedxt_m;
   DenseMatrix te_dshape, te_dshapedxt;
   Vector D;
#endif

   // PA extension
   const FiniteElementSpace *fespace;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, dofs1D, quad1D;
   Vector pa_data;
   bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient

   // Data for NURBS patch PA

   // Type for a variable-row-length 2D array, used for data related to 1D
   // quadrature rules in each dimension.
   typedef std::vector<std::vector<int>> IntArrayVar2D;

   int numPatches = 0;
   static constexpr int numTypes = 2;  // Number of rule types

   // In the case integrationMode == Mode::PATCHWISE_REDUCED, an approximate
   // integration rule with sparse nonzero weights is computed by NNLSSolver,
   // for each 1D basis function on each patch, in each spatial dimension. For a
   // fixed 1D basis function b_i with DOF index i, in the tensor product basis
   // of patch p, the prescribed exact 1D rule is of the form
   // \sum_k a_{i,j,k} w_k for some integration points indexed by k, with
   // weights w_k and coefficients a_{i,j,k} depending on Q(x), an element
   // transformation, b_i, and b_j, for all 1D basis functions b_j whose support
   // overlaps that of b_i. Define the constraint matrix G = [g_{j,k}] with
   // g_{j,k} = a_{i,j,k} and the vector of exact weights w = [w_k]. A reduced
   // rule should have different weights w_r, many of them zero, and should
   // approximately satisfy Gw_r = Gw. A sparse approximate solution to this
   // underdetermined system is computed by NNLSSolver, and its data is stored
   // in the following members.

   // For each patch p, spatial dimension d (total dim), and rule type t (total
   // numTypes), an std::vector<Vector> of reduced quadrature weights for all
   // basis functions is stored in reducedWeights[t + numTypes * (d + dim * p)],
   // reshaped as rw(t,d,p). Note that nd may vary with respect to the patch and
   // spatial dimension. Array reducedIDs is treated similarly.
   std::vector<std::vector<Vector>> reducedWeights;
   std::vector<IntArrayVar2D> reducedIDs;
   std::vector<Array<int>> pQ1D, pD1D;
   std::vector<std::vector<Array2D<real_t>>> pB, pG;
   std::vector<IntArrayVar2D> pminD, pmaxD, pminQ, pmaxQ, pminDD, pmaxDD;

   std::vector<Array<const IntegrationRule*>> pir1d;

   void SetupPatchPA(const int patch, Mesh *mesh, bool unitWeights=false);

   void SetupPatchBasisData(Mesh *mesh, unsigned int patch);

   /** Called by AssemblePatchMatrix for sparse matrix assembly on a NURBS patch
    with full 1D quadrature rules. */
   void AssemblePatchMatrix_fullQuadrature(const int patch,
                                           const FiniteElementSpace &fes,
                                           SparseMatrix*& smat);

   /** Called by AssemblePatchMatrix for sparse matrix assembly on a NURBS patch
    with reduced 1D quadrature rules. */
   void AssemblePatchMatrix_reducedQuadrature(const int patch,
                                              const FiniteElementSpace &fes,
                                              SparseMatrix*& smat);

public:
   /// Construct a diffusion integrator with coefficient Q = 1
   DiffusionIntegrator(const IntegrationRule *ir = nullptr);

   /// Construct a diffusion integrator with a scalar coefficient q
   DiffusionIntegrator(Coefficient &q, const IntegrationRule *ir = nullptr);

   /// Construct a diffusion integrator with a vector coefficient q
   DiffusionIntegrator(VectorCoefficient &q, const IntegrationRule *ir = nullptr);

   /// Construct a diffusion integrator with a matrix coefficient q
   DiffusionIntegrator(MatrixCoefficient &q, const IntegrationRule *ir = nullptr);

   /** Given a particular Finite Element computes the element stiffness matrix
       elmat. */
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   /** Given a trial and test Finite Element computes the element stiffness
       matrix elmat. */
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   void AssemblePatchMatrix(const int patch,
                            const FiniteElementSpace &fes,
                            SparseMatrix*& smat) override;

   void AssembleNURBSPA(const FiniteElementSpace &fes) override;

   void AssemblePatchPA(const int patch, const FiniteElementSpace &fes);

   /// Perform the local action of the BilinearFormIntegrator
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   void ComputeElementFlux(const FiniteElement &el,
                           ElementTransformation &Trans,
                           Vector &u, const FiniteElement &fluxelem,
                           Vector &flux, bool with_coef = true,
                           const IntegrationRule *ir = NULL) override;

   real_t ComputeFluxEnergy(const FiniteElement &fluxelem,
                            ElementTransformation &Trans,
                            Vector &flux, Vector *d_energy = NULL) override;

   void AssembleMF(const FiniteElementSpace &fes) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;

   void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                   const bool add) override;

   void AssembleDiagonalPA(Vector &diag) override;

   void AssembleDiagonalMF(Vector &diag) override;

   void AddMultMF(const Vector&, Vector&) const override;

   void AddMultPA(const Vector&, Vector&) const override;

   void AddMultTransposePA(const Vector&, Vector&) const override;

   void AddMultNURBSPA(const Vector&, Vector&) const override;

   void AddMultPatchPA(const int patch, const Vector &x, Vector &y) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe);

   bool SupportsCeed() const override { return DeviceCanUseCeed(); }

   Coefficient *GetCoefficient() const { return Q; }

   template <int DIM, int D1D, int Q1D>
   static void AddSpecialization()
   {
      ApplyPAKernels::Specialization<DIM,D1D,Q1D>::Add();
      DiagonalPAKernels::Specialization<DIM,D1D,Q1D>::Add();
   }
};

/** Class for local mass matrix assembling $a(u,v) := (Q u, v)$ */
class MassIntegrator: public BilinearFormIntegrator
{
   friend class DGMassInverse;
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   Coefficient *Q;
   // PA extension
   const FiniteElementSpace *fespace;
   Vector pa_data;
   const DofToQuad *maps;                 ///< Not owned
   const GeometricFactors *geom;          ///< Not owned
   const FaceGeometricFactors *face_geom; ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:

   using ApplyKernelType = void(*)(const int, const Array<real_t>&,
                                   const Array<real_t>&, const Vector&,
                                   const Vector&, Vector&, const int, const int);

   using DiagonalKernelType =  void(*)(const int, const Array<real_t>&,
                                       const Vector&, Vector&, const int,
                                       const int);

   MFEM_REGISTER_KERNELS(ApplyPAKernels, ApplyKernelType, (int, int, int));
   MFEM_REGISTER_KERNELS(DiagonalPAKernels, DiagonalKernelType, (int, int, int));
   struct Kernels { Kernels(); };

public:
   MassIntegrator(const IntegrationRule *ir = nullptr);

   /// Construct a mass integrator with coefficient q
   MassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL);

   /** Given a particular Finite Element computes the element mass matrix
       elmat. */
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   void AssembleMF(const FiniteElementSpace &fes) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;

   void AssemblePABoundary(const FiniteElementSpace &fes) override;

   void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                   const bool add) override;

   void AssembleDiagonalPA(Vector &diag) override;

   void AssembleDiagonalMF(Vector &diag) override;

   void AddMultMF(const Vector&, Vector&) const override;

   void AddMultPA(const Vector&, Vector&) const override;

   void AddMultTransposePA(const Vector&, Vector&) const override;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);

   bool SupportsCeed() const override { return DeviceCanUseCeed(); }

   const Coefficient *GetCoefficient() const { return Q; }

   template <int DIM, int D1D, int Q1D>
   static void AddSpecialization()
   {
      ApplyPAKernels::Specialization<DIM,D1D,Q1D>::Add();
      DiagonalPAKernels::Specialization<DIM,D1D,Q1D>::Add();
   }
};

/** Mass integrator $(u, v)$ restricted to the boundary of a domain */
class BoundaryMassIntegrator : public MassIntegrator
{
public:
   BoundaryMassIntegrator(Coefficient &q) : MassIntegrator(q) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

/// $\alpha (Q \cdot \nabla u, v)$
class ConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *Q;
   real_t alpha;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif

public:
   ConvectionIntegrator(VectorCoefficient &q, real_t a = 1.0)
      : Q(&q) { alpha = a; }

   void AssembleElementMatrix(const FiniteElement &,
                              ElementTransformation &,
                              DenseMatrix &) override;

   void AssembleMF(const FiniteElementSpace &fes) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace&) override;

   void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                   const bool add) override;

   void AssembleDiagonalPA(Vector &diag) override;

   void AssembleDiagonalMF(Vector &diag) override;

   void AddMultMF(const Vector&, Vector&) const override;

   void AddMultPA(const Vector&, Vector&) const override;

   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   static const IntegrationRule &GetRule(const FiniteElement &el,
                                         ElementTransformation &Trans);

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);

   bool SupportsCeed() const override { return DeviceCanUseCeed(); }
};

// Alias for @ConvectionIntegrator.
using NonconservativeConvectionIntegrator = ConvectionIntegrator;

/// $-\alpha (u, q \cdot \nabla v)$, negative transpose of ConvectionIntegrator
class ConservativeConvectionIntegrator : public TransposeIntegrator
{
public:
   ConservativeConvectionIntegrator(VectorCoefficient &q, real_t a = 1.0)
      : TransposeIntegrator(new ConvectionIntegrator(q, -a)) { }
};

/// $\alpha (Q \cdot \nabla u, v)$ using the "group" FE discretization
class GroupConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *Q;
   real_t alpha;

private:
   DenseMatrix dshape, adjJ, Q_nodal, grad;
   Vector shape;

public:
   GroupConvectionIntegrator(VectorCoefficient &q, real_t a = 1.0)
      : Q(&q) { alpha = a; }
   void AssembleElementMatrix(const FiniteElement &,
                              ElementTransformation &,
                              DenseMatrix &) override;
};

/** Class for integrating the bilinear form $a(u,v) := (Q u, v)$,
    where $u=(u_1,\dots,u_n)$ and $v=(v_1,\dots,v_n)$, $u_i$ and $v_i$ are defined
    by scalar FE through standard transformation. */
class VectorMassIntegrator: public BilinearFormIntegrator
{
private:
   int vdim;
   Vector shape, te_shape, vec;
   DenseMatrix partelmat;
   DenseMatrix mcoeff;
   int Q_order;

protected:
   Coefficient *Q;
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:
   /// Construct an integrator with coefficient 1.0
   VectorMassIntegrator()
      : vdim(-1), Q_order(0), Q(NULL), VQ(NULL), MQ(NULL) { }
   /** Construct an integrator with scalar coefficient q.  If possible, save
       memory by using a scalar integrator since the resulting matrix is block
       diagonal with the same diagonal block repeated. */
   VectorMassIntegrator(Coefficient &q, int qo = 0)
      : vdim(-1), Q_order(qo), Q(&q), VQ(NULL), MQ(NULL) { }
   VectorMassIntegrator(Coefficient &q, const IntegrationRule *ir)
      : BilinearFormIntegrator(ir), vdim(-1), Q_order(0), Q(&q), VQ(NULL),
        MQ(NULL) { }
   /// Construct an integrator with diagonal coefficient q
   VectorMassIntegrator(VectorCoefficient &q, int qo = 0)
      : vdim(q.GetVDim()), Q_order(qo), Q(NULL), VQ(&q), MQ(NULL) { }
   /// Construct an integrator with matrix coefficient q
   VectorMassIntegrator(MatrixCoefficient &q, int qo = 0)
      : vdim(q.GetVDim()), Q_order(qo), Q(NULL), VQ(NULL), MQ(&q) { }

   int GetVDim() const { return vdim; }
   void SetVDim(int vdim_) { vdim = vdim_; }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AssembleMF(const FiniteElementSpace &fes) override;
   void AssembleDiagonalPA(Vector &diag) override;
   void AssembleDiagonalMF(Vector &diag) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultMF(const Vector &x, Vector &y) const override;
   bool SupportsCeed() const override { return DeviceCanUseCeed(); }
};


/** Class for integrating $(\nabla \cdot u, p)$ where $u$ is a vector field given by
    VectorFiniteElement through Piola transformation (for Raviart-Thomas elements); $p$ is
    scalar function given by FiniteElement through standard transformation.
    Here, $u$ is the trial function and $p$ is the test function.

    Note: if the test space does not have map type INTEGRAL, then the element
    matrix returned by AssembleElementMatrix2 will not depend on the
    ElementTransformation Trans. */
class VectorFEDivergenceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector&, Vector&) const override;
   void AddMultTransposePA(const Vector&, Vector&) const override;

private:
#ifndef MFEM_THREAD_SAFE
   Vector divshape, shape;
#endif

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *L2mapsO;       ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   int dim, ne, dofs1D, L2dofs1D, quad1D;

public:
   VectorFEDivergenceIntegrator() { Q = NULL; }
   VectorFEDivergenceIntegrator(Coefficient &q) { Q = &q; }
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override { }
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   void AssembleDiagonalPA_ADAt(const Vector &D, Vector &diag) override;
};


/** Integrator for $(-Q u, \nabla v)$ for Nedelec ($u$) and $H^1$ ($v$) elements.
    This is equivalent to a weak divergence of the $H(curl$ basis functions. */
class VectorFEWeakDivergenceIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix vshape;
   DenseMatrix invdfdx;
#endif

public:
   VectorFEWeakDivergenceIntegrator() { Q = NULL; }
   VectorFEWeakDivergenceIntegrator(Coefficient &q) { Q = &q; }
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override { }
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/** Integrator for $(\mathrm{curl}(u), v)$ for Nedelec and Raviart-Thomas elements. If the trial and
    test spaces are switched, assembles the form $(u, \mathrm{curl}(v))$. */
class VectorFECurlIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix curlshapeTrial;
   DenseMatrix vshapeTest;
   DenseMatrix curlshapeTrial_dFT;
#endif

public:
   VectorFECurlIntegrator() { Q = NULL; }
   VectorFECurlIntegrator(Coefficient &q) { Q = &q; }
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override { }
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/// Integrator for (Q u.n, v.n) for RT elements
class VectorFEBoundaryFluxIntegrator : public BilinearFormIntegrator
{
   Coefficient *Q;
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
public:
   VectorFEBoundaryFluxIntegrator() { Q = NULL; }
   VectorFEBoundaryFluxIntegrator(Coefficient &q) { Q = &q; }
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/// Class for integrating $ (Q \partial_i(u), v) $ where $u$ and $v$ are scalars
class DerivativeIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient* Q;

private:
   int xi;
   DenseMatrix dshape, dshapedxt, invdfdx;
   Vector shape, dshapedxi;

public:
   DerivativeIntegrator(Coefficient &q, int i) : Q(&q), xi(i) { }
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override
   { AssembleElementMatrix2(el,el,Trans,elmat); }
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/// Integrator for $(\mathrm{curl}(u), \mathrm{curl}(v))$ for Nedelec elements
class CurlCurlIntegrator: public BilinearFormIntegrator
{
private:
   Vector vec, pointflux;
#ifndef MFEM_THREAD_SAFE
   Vector D;
   DenseMatrix curlshape, curlshape_dFt, M;
   DenseMatrix te_curlshape, te_curlshape_dFt;
   DenseMatrix vshape, projcurl;
#endif

protected:
   Coefficient *Q;
   DiagonalMatrixCoefficient *DQ;
   MatrixCoefficient *MQ;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;
   bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient

public:
   CurlCurlIntegrator() { Q = NULL; DQ = NULL; MQ = NULL; }
   /// Construct a bilinear form integrator for Nedelec elements
   CurlCurlIntegrator(Coefficient &q, const IntegrationRule *ir = NULL) :
      BilinearFormIntegrator(ir), Q(&q), DQ(NULL), MQ(NULL) { }
   CurlCurlIntegrator(DiagonalMatrixCoefficient &dq,
                      const IntegrationRule *ir = NULL) :
      BilinearFormIntegrator(ir), Q(NULL), DQ(&dq), MQ(NULL) { }
   CurlCurlIntegrator(MatrixCoefficient &mq, const IntegrationRule *ir = NULL) :
      BilinearFormIntegrator(ir), Q(NULL), DQ(NULL), MQ(&mq) { }

   /* Given a particular Finite Element, compute the
      element curl-curl matrix elmat */
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   void ComputeElementFlux(const FiniteElement &el,
                           ElementTransformation &Trans,
                           Vector &u, const FiniteElement &fluxelem,
                           Vector &flux, bool with_coef,
                           const IntegrationRule *ir = NULL) override;

   real_t ComputeFluxEnergy(const FiniteElement &fluxelem,
                            ElementTransformation &Trans,
                            Vector &flux, Vector *d_energy = NULL) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AssembleDiagonalPA(Vector& diag) override;

   const Coefficient *GetCoefficient() const { return Q; }
};

/** Integrator for $(\mathrm{curl}(u), \mathrm{curl}(v))$ for FE spaces defined by 'dim' copies of a
    scalar FE space. */
class VectorCurlCurlIntegrator: public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape_hat, dshape, curlshape, Jadj, grad_hat, grad;
#endif

protected:
   Coefficient *Q;

public:
   VectorCurlCurlIntegrator() { Q = NULL; }

   VectorCurlCurlIntegrator(Coefficient &q) : Q(&q) { }

   /// Assemble an element matrix
   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   /// Compute element energy: $ \frac{1}{2} (\mathrm{curl}(u), \mathrm{curl}(u))_E$
   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &Tr,
                           const Vector &elfun) override;
};

/** Class for integrating the bilinear form $a(u,v) := (Q \mathrm{curl}(u), v)$ where $Q$ is
    an optional scalar coefficient, and $v$ is a vector with components $v_i$ in
    the $L_2$ or $H^1$ space. This integrator handles 3 cases:
    1. u ∈ $H(curl$ in 3D, $v$ is a 3D vector with components $v_i$ in $L^2$ or $H^1$
    2. u ∈ $H(curl$ in 2D, $v$ is a scalar field in $L^2$ or $H^1$
    3. u is a scalar field in $H^1$, i.e, $\mathrm{curl}(u) := \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, $\nabla u$ and $v$ is a
        2D vector field with components $v_i$ in $L^2$ or $H^1$ space.

    Note: Case 2 can also be handled by MixedScalarCurlIntegrator  */
class MixedCurlIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix curlshape;
   DenseMatrix elmat_comp;
public:
   MixedCurlIntegrator() : Q{NULL} { }
   MixedCurlIntegrator(Coefficient *q_) :  Q{q_} { }
   MixedCurlIntegrator(Coefficient &q) :  Q{&q} { }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/** Integrator for $(Q u, v)$, where $Q$ is an optional coefficient (of type scalar,
    vector (diagonal matrix), or matrix), trial function $u$ is in $H(curl$ or
    $H(div)$, and test function $v$ is in $H(curl$, $H(div)$, or $v=(v_1,\dots,v_n)$, where
    $v_i$ are in $H^1$. */
class VectorFEMassIntegrator: public BilinearFormIntegrator
{
private:
   void Init(Coefficient *q, DiagonalMatrixCoefficient *dq, MatrixCoefficient *mq)
   { Q = q; DQ = dq; MQ = mq; }

#ifndef MFEM_THREAD_SAFE
   Vector shape;
   Vector D;
   DenseMatrix K;
   DenseMatrix partelmat;
   DenseMatrix test_vshape;
   DenseMatrix trial_vshape;
#endif

protected:
   Coefficient *Q;
   DiagonalMatrixCoefficient *DQ;
   MatrixCoefficient *MQ;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const DofToQuad *mapsOtest;     ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsCtest;     ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, nq, dofs1D, dofs1Dtest, quad1D, trial_fetype, test_fetype;
   bool symmetric = true; ///< False if using a nonsymmetric matrix coefficient

public:
   VectorFEMassIntegrator() { Init(NULL, NULL, NULL); }
   VectorFEMassIntegrator(Coefficient *q_) { Init(q_, NULL, NULL); }
   VectorFEMassIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
   VectorFEMassIntegrator(DiagonalMatrixCoefficient *dq_) { Init(NULL, dq_, NULL); }
   VectorFEMassIntegrator(DiagonalMatrixCoefficient &dq) { Init(NULL, &dq, NULL); }
   VectorFEMassIntegrator(MatrixCoefficient *mq_) { Init(NULL, NULL, mq_); }
   VectorFEMassIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   void AssemblePA(const FiniteElementSpace &fes) override;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;
   void AssembleDiagonalPA(Vector& diag) override;

   const Coefficient *GetCoefficient() const { return Q; }
};

/** Integrator for $(Q \nabla \cdot u, v)$ where $u=(u_1,\cdots,u_n)$ and all $u_i$ are in the same
    scalar FE space; $v$ is also in a (different) scalar FE space.  */
class VectorDivergenceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
   Vector shape;
   Vector divshape;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix Jadj;
   // PA extension
   Vector pa_data;
   const DofToQuad *trial_maps, *test_maps; ///< Not owned
   const GeometricFactors *geom;            ///< Not owned
   int dim, ne, nq;
   int trial_dofs1D, test_dofs1D, quad1D;

public:
   VectorDivergenceIntegrator() :
      Q(NULL), trial_maps(NULL), test_maps(NULL), geom(NULL)
   {  }
   VectorDivergenceIntegrator(Coefficient *q_) :
      Q(q_), trial_maps(NULL), test_maps(NULL), geom(NULL)
   { }
   VectorDivergenceIntegrator(Coefficient &q) :
      Q(&q), trial_maps(NULL), test_maps(NULL), geom(NULL)
   { }

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};

/// $(Q \nabla \cdot u, \nabla \cdot v)$ for Raviart-Thomas elements
class DivDivIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AssembleDiagonalPA(Vector& diag) override;

private:
#ifndef MFEM_THREAD_SAFE
   Vector divshape, te_divshape;
#endif

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, dofs1D, quad1D;

public:
   DivDivIntegrator() { Q = NULL; }
   DivDivIntegrator(Coefficient &q, const IntegrationRule *ir = NULL) :
      BilinearFormIntegrator(ir), Q(&q) { }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;

   void AssembleElementMatrix2(const FiniteElement &trial_fe,
                               const FiniteElement &test_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

   const Coefficient *GetCoefficient() const { return Q; }
};

/** Integrator for
    $$
      (Q \nabla u, \nabla v) = \sum_i (Q \nabla u_i, \nabla v_i) e_i e_i^{\mathrm{T}}
    $$
    for vector FE spaces, where $e_i$ is the unit vector in the $i$-th direction.
    The resulting local element matrix is square, of size <tt> vdim*dof </tt>,
    where \c vdim is the vector dimension space and \c dof is the local degrees
    of freedom. The integrator is not aware of the true vector dimension and
    must use \c VectorCoefficient, \c MatrixCoefficient, or a caller-specified
    value to determine the vector space. For a scalar coefficient, the caller
    may manually specify the vector dimension or the vector dimension is assumed
    to be the spatial dimension (i.e. 2-dimension or 3-dimension).
*/
class VectorDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q = NULL;
   VectorCoefficient *VQ = NULL;
   MatrixCoefficient *MQ = NULL;

   // PA extension
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, sdim, ne, dofs1D, quad1D;
   Vector pa_data;

private:
   DenseMatrix dshape, dshapedxt, pelmat;
   int vdim = -1;
   DenseMatrix mcoeff;
   Vector vcoeff;

public:
   VectorDiffusionIntegrator() { }

   /** \brief Integrator with unit coefficient for caller-specified vector
       dimension.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   VectorDiffusionIntegrator(int vector_dimension)
      : vdim(vector_dimension) { }

   VectorDiffusionIntegrator(Coefficient &q)
      : Q(&q) { }

   VectorDiffusionIntegrator(Coefficient &q, const IntegrationRule *ir)
      : BilinearFormIntegrator(ir), Q(&q) { }

   /** \brief Integrator with scalar coefficient for caller-specified vector
       dimension.

       The element matrix is block-diagonal with \c vdim copies of the element
       matrix integrated with the \c Coefficient.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   VectorDiffusionIntegrator(Coefficient &q, int vector_dimension)
      : Q(&q), vdim(vector_dimension) { }

   /** \brief Integrator with \c VectorCoefficient. The vector dimension of the
       \c FiniteElementSpace is assumed to be the same as the dimension of the
       \c Vector.

       The element matrix is block-diagonal and each block is integrated with
       coefficient $q_{i}$.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   VectorDiffusionIntegrator(VectorCoefficient &vq)
      : VQ(&vq), vdim(vq.GetVDim()) { }

   /** \brief Integrator with \c MatrixCoefficient. The vector dimension of the
       \c FiniteElementSpace is assumed to be the same as the dimension of the
       \c Matrix.

       The element matrix is populated in each block. Each block is integrated
       with coefficient $q_{ij}$.

       If the vector dimension does not match the true dimension of the space,
       the resulting element matrix will be mathematically invalid. */
   VectorDiffusionIntegrator(MatrixCoefficient& mq)
      : MQ(&mq), vdim(mq.GetVDim()) { }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans,
                              DenseMatrix &elmat) override;
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;
   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;
   void AssembleMF(const FiniteElementSpace &fes) override;
   void AssembleDiagonalPA(Vector &diag) override;
   void AssembleDiagonalMF(Vector &diag) override;
   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultMF(const Vector &x, Vector &y) const override;
   bool SupportsCeed() const override { return DeviceCanUseCeed(); }
};

/** Integrator for the linear elasticity form:
    $$
      a(u,v) = (\lambda \mathrm{div}(u), \mathrm{div}(v)) + (2 \mu \varepsilon(u), \varepsilon(v)),
    $$
    where $\varepsilon(v) = \frac{1}{2} (\mathrm{grad}(v) + \mathrm{grad}(v)^{\mathrm{T}})$.
    This is a 'Vector' integrator, i.e. defined for FE spaces
    using multiple copies of a scalar FE space. */
class ElasticityIntegrator : public BilinearFormIntegrator
{
   friend class ElasticityComponentIntegrator;

protected:
   real_t q_lambda, q_mu;
   Coefficient *lambda, *mu;

private:
#ifndef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape, gshape, pelmat;
   Vector divshape;
#endif

   // PA extension

   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int vdim, ndofs;
   const FiniteElementSpace *fespace;   ///< Not owned.

   std::unique_ptr<QuadratureSpace> q_space;
   /// Coefficients projected onto q_space
   std::unique_ptr<CoefficientVector> lambda_quad, mu_quad;
   /// Workspace vector
   std::unique_ptr<QuadratureFunction> q_vec;

   /// Set up the quadrature space and project lambda and mu coefficients
   void SetUpQuadratureSpaceAndCoefficients(const FiniteElementSpace &fes);

public:
   ElasticityIntegrator(Coefficient &l, Coefficient &m)
   { lambda = &l; mu = &m; }
   /** With this constructor $\lambda = q_l m$ and $\mu = q_m m$
       if $dim q_l + 2 q_m = 0$ then $tr(\sigma) = 0$. */
   ElasticityIntegrator(Coefficient &m, real_t q_l, real_t q_m)
   { lambda = NULL; mu = &m; q_lambda = q_l; q_mu = q_m; }

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Tr,
                              DenseMatrix &elmat) override;

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;

   void AssembleDiagonalPA(Vector &diag) override;

   void AddMultPA(const Vector &x, Vector &y) const override;

   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   /** Compute the stress corresponding to the local displacement @a $u$ and
       interpolate it at the nodes of the given @a fluxelem. Only the symmetric
       part of the stress is stored, so that the size of @a flux is equal to
       the number of DOFs in @a fluxelem times dim*(dim+1)/2. In 2D, the order
       of the stress components is: $s_xx, s_yy, s_xy$. In 3D, it is: $s_xx, s_yy,
       s_zz, s_xy, s_xz, s_yz$. In other words, @a flux is the local vector for
       a FE space with dim*(dim+1)/2 vector components, based on the finite
       element @a fluxelem. The integration rule is taken from @a fluxelem.
       @a ir exists to specific an alternative integration rule. */
   void ComputeElementFlux(const FiniteElement &el,
                           ElementTransformation &Trans,
                           Vector &u,
                           const FiniteElement &fluxelem,
                           Vector &flux, bool with_coef = true,
                           const IntegrationRule *ir = NULL) override;

   /** Compute the element energy (integral of the strain energy density)
       corresponding to the stress represented by @a flux which is a vector of
       coefficients multiplying the basis functions defined by @a fluxelem. In
       other words, @a flux is the local vector for a FE space with
       dim*(dim+1)/2 vector components, based on the finite element @a fluxelem.
       The number of components, dim*(dim+1)/2 is such that it represents the
       symmetric part of the (symmetric) stress tensor. The order of the
       components is: $s_xx, s_yy, s_xy$ in 2D, and $s_xx, s_yy, s_zz, s_xy, s_xz,
       s_yz$ in 3D. */
   real_t ComputeFluxEnergy(const FiniteElement &fluxelem,
                            ElementTransformation &Trans,
                            Vector &flux, Vector *d_energy = NULL) override;
};

/// @brief Integrator that computes the PA action of one of the blocks in an
/// ElasticityIntegrator, considering the elasticity operator as a dim x dim
/// block operator.
class ElasticityComponentIntegrator : public BilinearFormIntegrator
{
   ElasticityIntegrator &parent;
   const int i_block;
   const int j_block;

   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   const FiniteElementSpace *fespace;   ///< Not owned.

public:
   /// @brief Given an ElasticityIntegrator, create an integrator that
   /// represents the $(i,j)$th component block.
   ///
   /// @note The parent ElasticityIntegrator must remain valid throughout the
   /// lifetime of this integrator.
   ElasticityComponentIntegrator(ElasticityIntegrator &parent_, int i_, int j_);

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;

   void AssembleEA(const FiniteElementSpace &fes, Vector &emat,
                   const bool add = true) override;

   void AddMultPA(const Vector &x, Vector &y) const override;

   void AddMultTransposePA(const Vector &x, Vector &y) const override;
};

/** Integrator for the DG form:
    $$
      \alpha \langle \rho_u (u \cdot n) \{v\},[w] \rangle + \beta \langle \rho_u |u \cdot n| [v],[w] \rangle,
    $$
    where $v$ and $w$ are the trial and test variables, respectively, and $\rho$/$u$ are
    given scalar/vector coefficients. $\{v\}$ represents the average value of $v$ on
    the face and $[v]$ is the jump such that $\{v\}=(v_1+v_2)/2$ and $[v]=(v_1-v_2)$ for the
    face between elements $1$ and $2$. For boundary elements, $v2=0$. The vector
    coefficient, $u$, is assumed to be continuous across the faces and when given
    the scalar coefficient, $\rho$, is assumed to be discontinuous. The integrator
    uses the upwind value of $\rho$, denoted by $\rho_u$, which is value from the side into which
    the vector coefficient, $u$, points.

    One use case for this integrator is to discretize the operator $-u \cdot \nabla v$
    with a DG formulation. The resulting formulation uses the
    ConvectionIntegrator (with coefficient $u$, and parameter $\alpha = -1$) and the
    transpose of the DGTraceIntegrator (with coefficient $u$, and parameters $\alpha = 1$,
    $\beta = -1/2$ to use the upwind face flux, see also
    NonconservativeDGTraceIntegrator). This discretization and the handling of
    the inflow and outflow boundaries is illustrated in Example 9/9p.

    Another use case for this integrator is to discretize the operator $-\mathrm{div}(u v)$
    with a DG formulation. The resulting formulation is conservative and
    consists of the ConservativeConvectionIntegrator (with coefficient $u$, and
    parameter $\alpha = -1$) plus the DGTraceIntegrator (with coefficient $u$, and
    parameters $\alpha = -1$, $\beta = -1/2$ to use the upwind face flux).
    */
class DGTraceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *rho;
   VectorCoefficient *u;
   real_t alpha, beta;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;             ///< Not owned
   const FaceGeometricFactors *geom;  ///< Not owned
   int dim, nf, nq, dofs1D, quad1D;

private:
   Vector shape1, shape2;
   Vector tr_shape1, te_shape1, tr_shape2, te_shape2;

public:
   /// Construct integrator with $\rho = 1$, $\beta = \alpha/2$.
   DGTraceIntegrator(VectorCoefficient &u_, real_t a)
   { rho = NULL; u = &u_; alpha = a; beta = 0.5*a; }

   /// Construct integrator with $\rho = 1$.
   DGTraceIntegrator(VectorCoefficient &u_, real_t a, real_t b)
   { rho = NULL; u = &u_; alpha = a; beta = b; }

   DGTraceIntegrator(Coefficient &rho_, VectorCoefficient &u_,
                     real_t a, real_t b)
   { rho = &rho_; u = &u_; alpha = a; beta = b; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssembleFaceMatrix(const FiniteElement &trial_fe1,
                           const FiniteElement &test_fe1,
                           const FiniteElement &trial_fe2,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   void AssemblePAInteriorFaces(const FiniteElementSpace &fes) override;

   void AssemblePABoundaryFaces(const FiniteElementSpace &fes) override;

   void AddMultTransposePA(const Vector &x, Vector &y) const override;

   void AddMultPA(const Vector&, Vector&) const override;

   void AssembleEAInteriorFaces(const FiniteElementSpace& fes,
                                Vector &ea_data_int,
                                Vector &ea_data_ext,
                                const bool add) override;

   void AssembleEABoundaryFaces(const FiniteElementSpace& fes,
                                Vector &ea_data_bdr,
                                const bool add) override;

   static const IntegrationRule &GetRule(Geometry::Type geom, int order,
                                         const FaceElementTransformations &T);

   static const IntegrationRule &GetRule(Geometry::Type geom, int order,
                                         const ElementTransformation &T);

private:
   void SetupPA(const FiniteElementSpace &fes, FaceType type);
};

// Alias for @a DGTraceIntegrator.
using ConservativeDGTraceIntegrator = DGTraceIntegrator;

/** Integrator that represents the face terms used for the non-conservative
    DG discretization of the convection equation:
    $$
      -\alpha \langle \rho_u (u \cdot n) \{v\},[w] \rangle + \beta \langle \rho_u |u \cdot n| [v],[w] \rangle.
    $$
    This integrator can be used with together with ConvectionIntegrator to
    implement an upwind DG discretization in non-conservative form, see ex9 and
    ex9p. */
class NonconservativeDGTraceIntegrator : public TransposeIntegrator
{
public:
   NonconservativeDGTraceIntegrator(VectorCoefficient &u, real_t a)
      : TransposeIntegrator(new DGTraceIntegrator(u, -a, 0.5*a)) { }

   NonconservativeDGTraceIntegrator(VectorCoefficient &u, real_t a, real_t b)
      : TransposeIntegrator(new DGTraceIntegrator(u, -a, b)) { }

   NonconservativeDGTraceIntegrator(Coefficient &rho, VectorCoefficient &u,
                                    real_t a, real_t b)
      : TransposeIntegrator(new DGTraceIntegrator(rho, u, -a, b)) { }
};

/** Integrator for the DG form:
    $$
        - \langle \{(Q \nabla u) \cdot n\}, [v] \rangle + \sigma \langle [u], \{(Q \nabla v) \cdot n \} \rangle
        + \kappa \langle \{h^{-1} Q\} [u], [v] \rangle
    $$
    where $Q$ is a scalar or matrix diffusion coefficient and $u$, $v$ are the trial
    and test spaces, respectively. The parameters $\sigma$ and $\kappa$ determine the
    DG method to be used (when this integrator is added to the "broken"
    DiffusionIntegrator):
    - $\sigma = -1$, $\kappa \geq \kappa_0$: symm. interior penalty (IP or SIPG) method,
    - $\sigma = +1$, $\kappa > 0$: non-symmetric interior penalty (NIPG) method,
    - $\sigma = +1$, $\kappa = 0$: the method of Baumann and Oden.

    \todo Clarify used notation. */
class DGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   real_t sigma, kappa;

   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;


   // PA extension
   Vector pa_data; // (Q, h, dot(n,J)|el0, dot(n,J)|el1)
   const DofToQuad *maps; ///< Not owned
   int dim, nf, nq, dofs1D, quad1D;
   IntegrationRules irs{0, Quadrature1D::GaussLobatto};

public:
   DGDiffusionIntegrator(const real_t s, const real_t k)
      : Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   DGDiffusionIntegrator(Coefficient &q, const real_t s, const real_t k)
      : Q(&q), MQ(NULL), sigma(s), kappa(k) { }
   DGDiffusionIntegrator(MatrixCoefficient &q, const real_t s, const real_t k)
      : Q(NULL), MQ(&q), sigma(s), kappa(k) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

   bool RequiresFaceNormalDerivatives() const override { return true; }

   using BilinearFormIntegrator::AssemblePA;

   void AssemblePAInteriorFaces(const FiniteElementSpace &fes) override;

   void AssemblePABoundaryFaces(const FiniteElementSpace &fes) override;

   void AddMultPAFaceNormalDerivatives(const Vector &x, const Vector &dxdn,
                                       Vector &y, Vector &dydn) const override;

   const IntegrationRule &GetRule(int order, FaceElementTransformations &T);

   const IntegrationRule &GetRule(int order, Geometry::Type geom);

private:
   void SetupPA(const FiniteElementSpace &fes, FaceType type);
};

/** Integrator for the "BR2" diffusion stabilization term
    $$
      \sum_e \eta (r_e([u]), r_e([v]))
    $$
    where $r_e$ is the lifting operator defined on each edge $e$ (potentially
    weighted by a coefficient $Q$). The parameter eta can be chosen to be one to
    obtain a stable discretization. The constructor for this integrator requires
    the finite element space because the lifting operator depends on the
    element-wise inverse mass matrix.

    BR2 stands for the second method of Bassi and Rebay:

    - F. Bassi and S. Rebay. A high order discontinuous Galerkin method for
      compressible turbulent flows. In B. Cockburn, G. E. Karniadakis, and
      C.-W. Shu, editors, Discontinuous Galerkin Methods, pages 77-88. Springer
      Berlin Heidelberg, 2000.
    - D. N. Arnold, F. Brezzi, B. Cockburn, and L. D. Marini. Unified analysis
      of discontinuous Galerkin methods for elliptic problems. SIAM Journal on
      Numerical Analysis, 39(5):1749-1779, 2002.
*/
class DGDiffusionBR2Integrator : public BilinearFormIntegrator
{
protected:
   real_t eta;

   // Block factorizations of local mass matrices, with offsets for the case of
   // not equally sized blocks (mixed meshes, p-refinement)
   Array<real_t> Minv;
   Array<int> ipiv;
   Array<int> ipiv_offsets, Minv_offsets;

   Coefficient *Q;

   Vector shape1, shape2;

   DenseMatrix R11, R12, R21, R22;
   DenseMatrix MinvR11, MinvR12, MinvR21, MinvR22;
   DenseMatrix Re, MinvRe;

   /// Precomputes the inverses (LU factorizations) of the local mass matrices.
   /** @a fes must be a DG space, so the mass matrix is block diagonal, and its
       inverse can be computed locally. This is required for the computation of
       the lifting operators @a r_e.
   */
   void PrecomputeMassInverse(class FiniteElementSpace &fes);

public:
   DGDiffusionBR2Integrator(class FiniteElementSpace &fes, real_t e = 1.0);
   DGDiffusionBR2Integrator(class FiniteElementSpace &fes, Coefficient &Q_,
                            real_t e = 1.0);
   MFEM_DEPRECATED DGDiffusionBR2Integrator(class FiniteElementSpace *fes,
                                            real_t e = 1.0);

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

/** Integrator for the DG elasticity form, for the formulations see:
    - PhD Thesis of Jonas De Basabe, High-Order Finite %Element Methods for
      Seismic Wave Propagation, UT Austin, 2009, p. 23, and references therein
    - Peter Hansbo and Mats G. Larson, Discontinuous Galerkin and the
      Crouzeix-Raviart %Element: Application to Elasticity, PREPRINT 2000-09,
      p.3

    $$
    - \left< \{ \tau(u) \}, [v] \right> + \alpha \left< \{ \tau(v) \}, [u]
        \right> + \kappa \left< h^{-1} \{ \lambda + 2 \mu \} [u], [v] \right>
    $$

    where $ \left<u, v\right> = \int_{F} u \cdot v $, and $ F $ is a
    face which is either a boundary face $ F_b $ of an element $ K $ or
    an interior face $ F_i $ separating elements $ K_1 $ and $ K_2 $.

    In the bilinear form above $ \tau(u) $ is traction, and it's also
    $ \tau(u) = \sigma(u) \cdot \vec{n} $, where $ \sigma(u) $ is
    stress, and $ \vec{n} $ is the unit normal vector w.r.t. to $ F $.

    In other words, we have
    $$
    - \left< \{ \sigma(u) \cdot \vec{n} \}, [v] \right> + \alpha \left< \{
        \sigma(v) \cdot \vec{n} \}, [u] \right> + \kappa \left< h^{-1} \{
        \lambda + 2 \mu \} [u], [v] \right>
    $$

    For isotropic media
    $$
    \begin{split}
    \sigma(u) &= \lambda \nabla \cdot u I + 2 \mu \varepsilon(u) \\
              &= \lambda \nabla \cdot u I + 2 \mu \frac{1}{2} (\nabla u + \nabla
                 u^{\mathrm{T}}) \\
              &= \lambda \nabla \cdot u I + \mu (\nabla u + \nabla u^{\mathrm{T}})
    \end{split}
    $$

    where $ I $ is identity matrix, $ \lambda $ and $ \mu $ are Lame
    coefficients (see ElasticityIntegrator), $ u, v $ are the trial and test
    functions, respectively.

    The parameters $ \alpha $ and $ \kappa $ determine the DG method to
    use (when this integrator is added to the "broken" ElasticityIntegrator):

    - IIPG, $\alpha = 0$,
      C. Dawson, S. Sun, M. Wheeler, Compatible algorithms for coupled flow and
      transport, Comp. Meth. Appl. Mech. Eng., 193(23-26), 2565-2580, 2004.

    - SIPG, $\alpha = -1$,
      M. Grote, A. Schneebeli, D. Schotzau, Discontinuous Galerkin Finite
      %Element Method for the Wave Equation, SINUM, 44(6), 2408-2431, 2006.

    - NIPG, $\alpha = 1$,
      B. Riviere, M. Wheeler, V. Girault, A Priori Error Estimates for Finite
      %Element Methods Based on Discontinuous Approximation Spaces for Elliptic
      Problems, SINUM, 39(3), 902-931, 2001.

    This is a '%Vector' integrator, i.e. defined for FE spaces using multiple
    copies of a scalar FE space.
 */
class DGElasticityIntegrator : public BilinearFormIntegrator
{
public:
   DGElasticityIntegrator(real_t alpha_, real_t kappa_)
      : lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) { }

   DGElasticityIntegrator(Coefficient &lambda_, Coefficient &mu_,
                          real_t alpha_, real_t kappa_)
      : lambda(&lambda_), mu(&mu_), alpha(alpha_), kappa(kappa_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;

protected:
   Coefficient *lambda, *mu;
   real_t alpha, kappa;

#ifndef MFEM_THREAD_SAFE
   // values of all scalar basis functions for one component of u (which is a
   // vector) at the integration point in the reference space
   Vector shape1, shape2;
   // values of derivatives of all scalar basis functions for one component
   // of u (which is a vector) at the integration point in the reference space
   DenseMatrix dshape1, dshape2;
   // Adjugate of the Jacobian of the transformation: adjJ = det(J) J^{-1}
   DenseMatrix adjJ;
   // gradient of shape functions in the real (physical, not reference)
   // coordinates, scaled by det(J):
   //    dshape_ps(jdof,jm) = sum_{t} adjJ(t,jm)*dshape(jdof,t)
   DenseMatrix dshape1_ps, dshape2_ps;
   Vector nor;  // nor = |weight(J_face)| n
   Vector nL1, nL2;  // nL1 = (lambda1 * ip.weight / detJ1) nor
   Vector nM1, nM2;  // nM1 = (mu1     * ip.weight / detJ1) nor
   Vector dshape1_dnM, dshape2_dnM; // dshape1_dnM = dshape1_ps . nM1
   // 'jmat' corresponds to the term: kappa <h^{-1} {lambda + 2 mu} [u], [v]>
   DenseMatrix jmat;
#endif

   static void AssembleBlock(
      const int dim, const int row_ndofs, const int col_ndofs,
      const int row_offset, const int col_offset,
      const real_t jmatcoef, const Vector &col_nL, const Vector &col_nM,
      const Vector &row_shape, const Vector &col_shape,
      const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
      DenseMatrix &elmat, DenseMatrix &jmat);
};

/** Integrator for the DPG form:$ \langle v, [w] \rangle $ over all faces (the interface) where
    the trial variable $v$ is defined on the interface and the test variable $w$ is
    defined inside the elements, generally in a DG space. */
class TraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, shape1, shape2;

public:
   TraceJumpIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

/** Integrator for the form:$ \langle v, [w \cdot n] \rangle $ over all faces (the interface) where
    the trial variable $v$ is defined on the interface and the test variable $w$ is
    in an $H(div)$-conforming space. */
class NormalTraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, normal, shape1_n, shape2_n;
   DenseMatrix shape1, shape2;

public:
   NormalTraceJumpIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override;
};

/** Integrator for the DPG form:$ \langle v, w \rangle $ over a face (the interface) where
    the trial variable $v$ is defined on the interface
    ($H^{-1/2}$ i.e., $v := u \cdot n$ normal trace of $H(div)$)
    and the test variable $w$ is in an $H^1$-conforming space. */
class TraceIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, shape;
public:
   TraceIntegrator() { }
   void AssembleTraceFaceMatrix(int elem,
                                const FiniteElement &trial_face_fe,
                                const FiniteElement &test_fe,
                                FaceElementTransformations &Trans,
                                DenseMatrix &elmat);
};

/** Integrator for the form: $ \langle v, w \cdot n \rangle $ over a face (the interface) where
    the trial variable $v$ is defined on the interface ($H^{1/2}$, i.e., trace of $H^1$)
    and the test variable $w$ is in an $H(div)$-conforming space. */
class NormalTraceIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, normal, shape_n;
   DenseMatrix shape;

public:
   NormalTraceIntegrator() { }
   void AssembleTraceFaceMatrix(int ielem,
                                const FiniteElement &trial_face_fe,
                                const FiniteElement &test_fe,
                                FaceElementTransformations &Trans,
                                DenseMatrix &elmat) override;
};


/** Integrator for the form: $\langle v, w \times n \rangle$ over a face (the interface)
 *  In 3D the trial variable $v$ is defined on the interface ($H^{-1/2}$(curl), trace of $H(curl$)
 *  In 2D it's defined on the interface ($H^{1/2}$, trace of $H^1$)
 *  The test variable $w$ is in an $H(curl$-conforming space. */
class TangentTraceIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix face_shape, shape, shape_n;
   Vector normal;
   Vector temp;

   void cross_product(const Vector & x, const DenseMatrix & Y, DenseMatrix & Z)
   {
      int dim = x.Size();
      MFEM_VERIFY(Y.Width() == dim, "Size mismatch");
      int dimc = dim == 3 ? dim : 1;
      int h = Y.Height();
      Z.SetSize(h,dimc);
      if (dim == 3)
      {
         for (int i = 0; i<h; i++)
         {
            Z(i,0) = x(2) * Y(i,1) - x(1) * Y(i,2);
            Z(i,1) = x(0) * Y(i,2) - x(2) * Y(i,0);
            Z(i,2) = x(1) * Y(i,0) - x(0) * Y(i,1);
         }
      }
      else
      {
         for (int i = 0; i<h; i++)
         {
            Z(i,0) = x(1) * Y(i,0) - x(0) * Y(i,1);
         }
      }
   }

public:
   TangentTraceIntegrator() { }
   void AssembleTraceFaceMatrix(int elem,
                                const FiniteElement &trial_face_fe,
                                const FiniteElement &test_fe,
                                FaceElementTransformations &Trans,
                                DenseMatrix &elmat);
};

/** Abstract class to serve as a base for local interpolators to be used in the
    DiscreteLinearOperator class. */
class DiscreteInterpolator : public BilinearFormIntegrator { };


/** Class for constructing the gradient as a DiscreteLinearOperator from an
    $H^1$-conforming space to an $H(curl$-conforming space. The range space can be
    vector $L_2$ space as well. */
class GradientInterpolator : public DiscreteInterpolator
{
public:
   GradientInterpolator() : dofquad_fe(NULL) { }
   virtual ~GradientInterpolator() { delete dofquad_fe; }

   void AssembleElementMatrix2(const FiniteElement &h1_fe,
                               const FiniteElement &nd_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   { nd_fe.ProjectGrad(h1_fe, Trans, elmat); }

   using BilinearFormIntegrator::AssemblePA;

   /** @brief Setup method for PA data.

       @param[in] trial_fes   $H^1$ Lagrange space
       @param[in] test_fes    $H(curl$ Nedelec space
    */
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;

private:
   /// 1D finite element that generates and owns the 1D DofToQuad maps below
   FiniteElement *dofquad_fe;

   bool B_id; // is the B basis operator (maps_C_C) the identity?
   const DofToQuad *maps_C_C; // one-d map with Lobatto rows, Lobatto columns
   const DofToQuad *maps_O_C; // one-d map with Legendre rows, Lobatto columns
   int dim, ne, o_dofs1D, c_dofs1D;
};


/** Class for constructing the identity map as a DiscreteLinearOperator. This
    is the discrete embedding matrix when the domain space is a subspace of
    the range space. Otherwise, a dof projection matrix is constructed. */
class IdentityInterpolator : public DiscreteInterpolator
{
protected:
   const int vdim;

public:
   /** @brief Construct an identity interpolator.

       @param[in]  vdim_  Vector dimension (number of components) in the domain
                          and range FE spaces.
   */
   IdentityInterpolator(int vdim_ = 1) : vdim(vdim_) { }

   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   {
      if (vdim == 1)
      {
         ran_fe.Project(dom_fe, Trans, elmat);
         return;
      }
      DenseMatrix elmat_block;
      ran_fe.Project(dom_fe, Trans, elmat_block);
      elmat.SetSize(vdim*elmat_block.Height(), vdim*elmat_block.Width());
      elmat = 0_r;
      for (int i = 0; i < vdim; i++)
      {
         elmat.SetSubMatrix(i*elmat_block.Height(), i*elmat_block.Width(),
                            elmat_block);
      }
   }

   using BilinearFormIntegrator::AssemblePA;
   void AssemblePA(const FiniteElementSpace &trial_fes,
                   const FiniteElementSpace &test_fes) override;

   void AddMultPA(const Vector &x, Vector &y) const override;
   void AddMultTransposePA(const Vector &x, Vector &y) const override;

private:
   /// 1D finite element that generates and owns the 1D DofToQuad maps below
   std::unique_ptr<FiniteElement> dofquad_fe;

   const DofToQuad *maps_C_C; // one-d map with Lobatto rows, Lobatto columns
   const DofToQuad *maps_O_C; // one-d map with Legendre rows, Lobatto columns
   int dim, ne, o_dofs1D, c_dofs1D;

   Vector pa_data;
};


/** @brief Class identical to IdentityInterpolator with the exception that it
    requires the vector dimension (number of components) to be specified during
    construction. */
class VectorIdentityInterpolator : public IdentityInterpolator
{
public:
   VectorIdentityInterpolator(int vdim_) : IdentityInterpolator(vdim_) { }
};


/** Class for constructing the (local) discrete curl matrix which can be used
    as an integrator in a DiscreteLinearOperator object to assemble the global
    discrete curl matrix. */
class CurlInterpolator : public DiscreteInterpolator
{
public:
   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   { ran_fe.ProjectCurl(dom_fe, Trans, elmat); }
};


/** Class for constructing the (local) discrete divergence matrix which can
    be used as an integrator in a DiscreteLinearOperator object to assemble
    the global discrete divergence matrix.

    Note: Since the dofs in the L2_FECollection are nodal values, the local
    discrete divergence matrix (with an $H(div)$-type domain space) will depend on
    the transformation. On the other hand, the local matrix returned by
    VectorFEDivergenceIntegrator is independent of the transformation. */
class DivergenceInterpolator : public DiscreteInterpolator
{
public:
   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override
   { ran_fe.ProjectDiv(dom_fe, Trans, elmat); }
};


/** A trace face interpolator class for interpolating the normal component of
    the domain space, e.g. vector $H^1$, into the range space, e.g. the trace of
    $H(div)$ which uses FiniteElement::INTEGRAL map type. */
class NormalInterpolator : public DiscreteInterpolator
{
public:
   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
};

/** Interpolator of a scalar coefficient multiplied by a scalar field onto
    another scalar field. Note that this can produce inaccurate fields unless
    the target is sufficiently high order. */
class ScalarProductInterpolator : public DiscreteInterpolator
{
public:
   ScalarProductInterpolator(Coefficient & sc) : Q(&sc) { }

   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;

protected:
   Coefficient *Q;
};

/** Interpolator of a scalar coefficient multiplied by a vector field onto
    another vector field. Note that this can produce inaccurate fields unless
    the target is sufficiently high order. */
class ScalarVectorProductInterpolator : public DiscreteInterpolator
{
public:
   ScalarVectorProductInterpolator(Coefficient & sc)
      : Q(&sc) { }

   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
protected:
   Coefficient *Q;
};

/** Interpolator of a vector coefficient multiplied by a scalar field onto
    another vector field. Note that this can produce inaccurate fields unless
    the target is sufficiently high order. */
class VectorScalarProductInterpolator : public DiscreteInterpolator
{
public:
   VectorScalarProductInterpolator(VectorCoefficient & vc)
      : VQ(&vc) { }

   void AssembleElementMatrix2(const FiniteElement &dom_fe,
                               const FiniteElement &ran_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
protected:
   VectorCoefficient *VQ;
};

/** Interpolator of the 2D cross product between a vector coefficient and an
    $H(curl$-conforming field onto an $L_2$-conforming field. */
class ScalarCrossProductInterpolator : public DiscreteInterpolator
{
public:
   ScalarCrossProductInterpolator(VectorCoefficient & vc)
      : VQ(&vc) { }

   void AssembleElementMatrix2(const FiniteElement &nd_fe,
                               const FiniteElement &l2_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
protected:
   VectorCoefficient *VQ;
};

/** Interpolator of the cross product between a vector coefficient and an
    $H(curl$-conforming field onto an $H(div)$-conforming field. The range space
    can also be vector $L_2$. */
class VectorCrossProductInterpolator : public DiscreteInterpolator
{
public:
   VectorCrossProductInterpolator(VectorCoefficient & vc)
      : VQ(&vc) { }

   void AssembleElementMatrix2(const FiniteElement &nd_fe,
                               const FiniteElement &rt_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
protected:
   VectorCoefficient *VQ;
};

/** Interpolator of the inner product between a vector coefficient and an
    $H(div)$-conforming field onto an $L_2$-conforming field. The range space can
    also be $H^1$. */
class VectorInnerProductInterpolator : public DiscreteInterpolator
{
public:
   VectorInnerProductInterpolator(VectorCoefficient & vc) : VQ(&vc) { }

   void AssembleElementMatrix2(const FiniteElement &rt_fe,
                               const FiniteElement &l2_fe,
                               ElementTransformation &Trans,
                               DenseMatrix &elmat) override;
protected:
   VectorCoefficient *VQ;
};

}
#endif
