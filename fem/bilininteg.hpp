// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "libceed/ceed.hpp"

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

   using NonlinearFormIntegrator::AssemblePA;

   /// Method defining partial assembly.
   /** The result of the partial assembly is stored internally so that it can be
       used later in the methods AddMultPA() and AddMultTransposePA(). */
   virtual void AssemblePA(const FiniteElementSpace &fes);
   /** Used with BilinearFormIntegrators that have different spaces. */
   virtual void AssemblePA(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes);

   virtual void AssemblePAInteriorFaces(const FiniteElementSpace &fes);

   virtual void AssemblePABoundaryFaces(const FiniteElementSpace &fes);

   /// Assemble diagonal and add it to Vector @a diag.
   virtual void AssembleDiagonalPA(Vector &diag);

   /// Method for partially assembled action.
   /** Perform the action of integrator on the input @a x and add the result to
       the output @a y. Both @a x and @a y are E-vectors, i.e. they represent
       the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssemblePA() has been
       called. */
   virtual void AddMultPA(const Vector &x, Vector &y) const;

   /// Method for partially assembled transposed action.
   /** Perform the transpose action of integrator on the input @a x and add the
       result to the output @a y. Both @a x and @a y are E-vectors, i.e. they
       represent the element-wise discontinuous version of the FE space.

       This method can be called only after the method AssemblePA() has been
       called. */
   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;

   /// Given a particular Finite Element computes the element matrix elmat.
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   /** Compute the local matrix representation of a bilinear form
       a(u,v) defined on different trial (given by u) and test
       (given by v) spaces. The rows in the local matrix correspond
       to the test dofs and the columns -- to the trial dofs. */
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   /** Abstract method used for assembling TraceFaceIntegrators in a
       MixedBilinearForm. */
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   /// Perform the local action of the BilinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   virtual void AssembleElementGrad(const FiniteElement &el,
                                    ElementTransformation &Tr,
                                    const Vector &elfun, DenseMatrix &elmat)
   { AssembleElementMatrix(el, Tr, elmat); }

   virtual void AssembleFaceGrad(const FiniteElement &el1,
                                 const FiniteElement &el2,
                                 FaceElementTransformations &Tr,
                                 const Vector &elfun, DenseMatrix &elmat)
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
    */
   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u,
                                   const FiniteElement &fluxelem,
                                   Vector &flux, bool with_coef = true) { }

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
   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL)
   { return 0.0; }

   virtual ~BilinearFormIntegrator() { }
};

class TransposeIntegrator : public BilinearFormIntegrator
{
private:
   int own_bfi;
   BilinearFormIntegrator *bfi;

   DenseMatrix bfi_elmat;

public:
   TransposeIntegrator (BilinearFormIntegrator *_bfi, int _own_bfi = 1)
   { bfi = _bfi; own_bfi = _own_bfi; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   using BilinearFormIntegrator::AssemblePA;

   virtual void AssemblePA(const FiniteElementSpace& fes)
   {
      bfi->AssemblePA(fes);
   }

   virtual void AssemblePAInteriorFaces(const FiniteElementSpace &fes)
   {
      bfi->AssemblePAInteriorFaces(fes);
   }

   virtual void AssemblePABoundaryFaces(const FiniteElementSpace &fes)
   {
      bfi->AssemblePABoundaryFaces(fes);
   }

   virtual void AddMultTransposePA(const Vector &x, Vector &y) const
   {
      bfi->AddMultPA(x, y);
   }

   virtual void AddMultPA(const Vector& x, Vector& y) const
   {
      bfi->AddMultTransposePA(x, y);
   }

   virtual ~TransposeIntegrator() { if (own_bfi) { delete bfi; } }
};

class LumpedIntegrator : public BilinearFormIntegrator
{
private:
   int own_bfi;
   BilinearFormIntegrator *bfi;

public:
   LumpedIntegrator (BilinearFormIntegrator *_bfi, int _own_bfi = 1)
   { bfi = _bfi; own_bfi = _own_bfi; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

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

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~InverseIntegrator() { if (own_integrator) { delete integrator; } }
};

/// Integrator defining a sum of multiple Integrators.
class SumIntegrator : public BilinearFormIntegrator
{
private:
   int own_integrators;
   DenseMatrix elem_mat;
   Array<BilinearFormIntegrator*> integrators;

public:
   SumIntegrator(int own_integs = 1) { own_integrators = own_integs; }

   void AddIntegrator(BilinearFormIntegrator *integ)
   { integrators.Append(integ); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual ~SumIntegrator();
};

/** An abstract class for integrating the product of two scalar basis functions
    with an optional scalar coefficient. */
class MixedScalarIntegrator: public BilinearFormIntegrator
{
public:

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   /// Support for use in BilinearForm. Can be used only when appropriate.
   virtual void AssembleElementMatrix(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat)
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

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   /// Support for use in BilinearForm. Can be used only when appropriate.
   virtual void AssembleElementMatrix(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat)
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
   MixedVectorIntegrator(VectorCoefficient &dq, bool diag = true)
      : same_calc_shape(false), Q(NULL), VQ(diag?NULL:&dq), DQ(diag?&dq:NULL),
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


   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcVShape(Trans, shape); }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcVShape(Trans, shape); }

   Coefficient *Q;
   VectorCoefficient *VQ;
   VectorCoefficient *DQ;
   MatrixCoefficient *MQ;

private:

#ifndef MFEM_THREAD_SAFE
   Vector V;
   Vector D;
   DenseMatrix M;
   DenseMatrix test_shape;
   DenseMatrix trial_shape;
   DenseMatrix test_shape_tmp;
#endif

};

/** An abstract class for integrating the product of a scalar basis function and
    the inner product of a vector basis function with a vector coefficient. In
    2D the inner product can be replaced with a cross product. */
class MixedScalarVectorIntegrator: public BilinearFormIntegrator
{
public:

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

protected:

   MixedScalarVectorIntegrator(VectorCoefficient &vq, bool _transpose = false,
                               bool _cross_2d = false)
      : VQ(&vq), transpose(_transpose), cross_2d(_cross_2d) {}

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


   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcVShape(Trans, shape); }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysShape(Trans, shape); }

   VectorCoefficient *VQ;
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

/** Class for integrating the bilinear form a(u,v) := (Q u, v) in either 1D, 2D,
    or 3D and where Q is an optional scalar coefficient, u and v are each in H1
    or L2. */
class MixedScalarMassIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarMassIntegrator() { same_calc_shape = true; }
   MixedScalarMassIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) { same_calc_shape = true; }
};

/** Class for integrating the bilinear form a(u,v) := (Q u, v) in either 2D, or
    3D and where Q is a vector coefficient, u is in H1 or L2 and v is in H(Curl)
    or H(Div). */
class MixedVectorProductIntegrator : public MixedScalarVectorIntegrator
{
public:
   MixedVectorProductIntegrator(VectorCoefficient &vq)
      : MixedScalarVectorIntegrator(vq) {}
};

/** Class for integrating the bilinear form a(u,v) := (Q D u, v) in 1D where Q
    is an optional scalar coefficient, u is in H1, and v is in L2. */
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
             "and the trial space must implement CaldDShape.";
   }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      trial_fe.CalcPhysDShape(Trans, dshape);
   }
};

/** Class for integrating the bilinear form a(u,v) := -(Q u, D v) in 1D where Q
    is an optional scalar coefficient, u is in L2, and v is in H1. */
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

/** Class for integrating the bilinear form a(u,v) := (Q div u, v) in either 2D
    or 3D where Q is an optional scalar coefficient, u is in H(Div), and v is a
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
             "Trial must be H(Div) and the test space must be a "
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

/** Class for integrating the bilinear form a(u,v) := (V div u, v) in either 2D
    or 3D where V is a vector coefficient, u is in H(Div), and v is a vector
    field. */
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

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := -(Q u, div v) in either 2D
    or 3D where Q is an optional scalar coefficient, u is in L2 or H1, and v is
    in H(Div). */
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

   virtual void CalcTestShape(const FiniteElement & test_fe,
                              ElementTransformation &Trans,
                              Vector & shape)
   {
      test_fe.CalcPhysDivShape(Trans, shape);
      shape *= -1.0;
   }
};

/** Class for integrating the bilinear form a(u,v) := (Q curl u, v) in 2D where
    Q is an optional scalar coefficient, u is in H(Curl), and v is in L2 or
    H1. */
class MixedScalarCurlIntegrator : public MixedScalarIntegrator
{
public:
   MixedScalarCurlIntegrator() {}
   MixedScalarCurlIntegrator(Coefficient &q)
      : MixedScalarIntegrator(q) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 2 && test_fe.GetDim() == 2 &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL  &&
              test_fe.GetRangeType()  == mfem::FiniteElement::SCALAR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedScalarCurlIntegrator:  "
             "Trial must be H(Curl) and the test space must be a "
             "scalar field";
   }

   inline virtual int GetIntegrationOrder(const FiniteElement & trial_fe,
                                          const FiniteElement & test_fe,
                                          ElementTransformation &Trans)
   { return trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW() - 1; }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      Vector & shape)
   {
      DenseMatrix dshape(shape.GetData(), shape.Size(), 1);
      trial_fe.CalcPhysCurlShape(Trans, dshape);
   }
};

/** Class for integrating the bilinear form a(u,v) := (Q u, curl v) in 2D where
    Q is an optional scalar coefficient, u is in L2 or H1, and v is in
    H(Curl). */
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

/** Class for integrating the bilinear form a(u,v) := (Q u, v) in either 2D or
    3D and where Q is an optional coefficient (of type scalar, matrix, or
    diagonal matrix) u and v are each in H(Curl) or H(Div). */
class MixedVectorMassIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorMassIntegrator() { same_calc_shape = true; }
   MixedVectorMassIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) { same_calc_shape = true; }
   MixedVectorMassIntegrator(VectorCoefficient &dq)
      : MixedVectorIntegrator(dq, true) { same_calc_shape = true; }
   MixedVectorMassIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) { same_calc_shape = true; }
};

/** Class for integrating the bilinear form a(u,v) := (V x u, v) in 3D and where
    V is a vector coefficient u and v are each in H(Curl) or H(Div). */
class MixedCrossProductIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossProductIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) { same_calc_shape = true; }
};

/** Class for integrating the bilinear form a(u,v) := (V . u, v) in 2D or 3D and
    where V is a vector coefficient u is in H(Curl) or H(Div) and v is in H1 or
    L2. */
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

/** Class for integrating the bilinear form a(u,v) := (-V . u, Div v) in 2D or
    3D and where V is a vector coefficient u is in H(Curl) or H(Div) and v is in
    RT. */
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

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form a(u,v) := (V x u, Grad v) in 3D and
    where V is a vector coefficient u is in H(Curl) or H(Div) and v is in H1. */
class MixedWeakDivCrossIntegrator : public MixedVectorIntegrator
{
public:
   MixedWeakDivCrossIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form a(u,v) := (Q Grad u, Grad v) in 3D
    or in 2D and where Q is a scalar or matrix coefficient u and v are both in
    H1. */
class MixedGradGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedGradGradIntegrator() { same_calc_shape = true; }
   MixedGradGradIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) { same_calc_shape = true; }
   MixedGradGradIntegrator(VectorCoefficient &dq)
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x Grad u, Grad v) in 3D
    or in 2D and where V is a vector coefficient u and v are both in H1. */
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (Q Curl u, Curl v) in 3D
    and where Q is a scalar or matrix coefficient u and v are both in
    H(Curl). */
class MixedCurlCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCurlCurlIntegrator() { same_calc_shape = true; }
   MixedCurlCurlIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) { same_calc_shape = true; }
   MixedCurlCurlIntegrator(VectorCoefficient &dq)
      : MixedVectorIntegrator(dq, true) { same_calc_shape = true; }
   MixedCurlCurlIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) { same_calc_shape = true; }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x Curl u, Curl v) in 3D
    and where V is a vector coefficient u and v are both in H(Curl). */
class MixedCrossCurlCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossCurlCurlIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) { same_calc_shape = true; }

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x Curl u, Grad v) in 3D
    and where V is a vector coefficient u is in H(Curl) and v is in H1. */
class MixedCrossCurlGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossCurlGradIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x Grad u, Curl v) in 3D
    and where V is a scalar coefficient u is in H1 and v is in H(Curl). */
class MixedCrossGradCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossGradCurlIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x u, Curl v) in 3D and
    where V is a vector coefficient u is in H(Curl) or H(Div) and v is in
    H(Curl). */
class MixedWeakCurlCrossIntegrator : public MixedVectorIntegrator
{
public:
   MixedWeakCurlCrossIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x u, Curl v) in 2D and
    where V is a vector coefficient u is in H(Curl) or H(Div) and v is in
    H(Curl). */
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

/** Class for integrating the bilinear form a(u,v) := (V x Grad u, v) in 3D or
    in 2D and where V is a vector coefficient u is in H1 and v is in H(Curl) or
    H(Div). */
class MixedCrossGradIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossGradIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysDShape(Trans, shape); }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   { test_fe.CalcVShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x Curl u, v) in 3D and
    where V is a vector coefficient u is in H(Curl) and v is in H(Curl) or
    H(Div). */
class MixedCrossCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedCrossCurlIntegrator(VectorCoefficient &vq)
      : MixedVectorIntegrator(vq, false) {}

   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
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

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   { trial_fe.CalcPhysCurlShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x Curl u, v) in 2D and
    where V is a vector coefficient u is in H(Curl) and v is in H(Curl) or
    H(Div). */
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

/** Class for integrating the bilinear form a(u,v) := (V x Grad u, v) in 2D and
    where V is a vector coefficient u is in H1 and v is in H1 or L2. */
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

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (V x u, v) in 2D and where
    V is a vector coefficient u is in ND or RT and v is in H1 or L2. */
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

/** Class for integrating the bilinear form a(u,v) := (V x z u, v) in 2D and
    where V is a vector coefficient u is in H1 or L2 and v is in ND or RT. */
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

/** Class for integrating the bilinear form a(u,v) := (V . Grad u, v) in 2D or
    3D and where V is a vector coefficient, u is in H1 and v is in H1 or L2. */
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

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (-V . Grad u, Div v) in 2D
    or 3D and where V is a vector coefficient, u is in H1 and v is in RT. */
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

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (-V Div u, Grad v) in 2D
    or 3D and where V is a vector coefficient, u is in RT and v is in H1. */
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

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }

   inline virtual void CalcShape(const FiniteElement & scalar_fe,
                                 ElementTransformation &Trans,
                                 Vector & shape)
   { scalar_fe.CalcPhysDivShape(Trans, shape); }
};

/** Class for integrating the bilinear form a(u,v) := (-V u, Grad v) in 2D or 3D
    and where V is a vector coefficient, u is in H1 and v is in H1. */
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

   inline virtual void CalcVShape(const FiniteElement & vector_fe,
                                  ElementTransformation &Trans,
                                  DenseMatrix & shape)
   { vector_fe.CalcPhysDShape(Trans, shape); shape *= -1.0; }
};

/** Class for integrating the bilinear form a(u,v) := (Q grad u, v) in either 2D
    or 3D and where Q is an optional coefficient (of type scalar, matrix, or
    diagonal matrix) u is in H1 and v is in H(Curl) or H(Div). */
class MixedVectorGradientIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorGradientIntegrator() {}
   MixedVectorGradientIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorGradientIntegrator(VectorCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorGradientIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDerivType() == mfem::FiniteElement::GRAD &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedVectorGradientIntegrator:  "
             "Trial spaces must be H1 and the test space must be a "
             "vector field in 2D or 3D";
   }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   {
      trial_fe.CalcPhysDShape(Trans, shape);
   }

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes);

   virtual void AddMultPA(const Vector&, Vector&) const;

private:
   DenseMatrix Jinv;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, dofs1D, quad1D;
};

/** Class for integrating the bilinear form a(u,v) := (Q curl u, v) in 3D and
    where Q is an optional coefficient (of type scalar, matrix, or diagonal
    matrix) u is in H(Curl) and v is in H(Div) or H(Curl). */
class MixedVectorCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorCurlIntegrator() {}
   MixedVectorCurlIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorCurlIntegrator(VectorCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorCurlIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
              trial_fe.GetDerivType() == mfem::FiniteElement::CURL  &&
              test_fe.GetRangeType()  == mfem::FiniteElement::VECTOR );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedVectorCurlIntegrator:  "
             "Trial space must be H(Curl) and the test space must be a "
             "vector field in 3D";
   }

   inline virtual void CalcTrialShape(const FiniteElement & trial_fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix & shape)
   {
      trial_fe.CalcPhysCurlShape(Trans, shape);
   }
};

/** Class for integrating the bilinear form a(u,v) := (Q u, curl v) in 3D and
    where Q is an optional coefficient (of type scalar, matrix, or diagonal
    matrix) u is in H(Div) or H(Curl) and v is in H(Curl). */
class MixedVectorWeakCurlIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorWeakCurlIntegrator() {}
   MixedVectorWeakCurlIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorWeakCurlIntegrator(VectorCoefficient &dq)
      : MixedVectorIntegrator(dq, true) {}
   MixedVectorWeakCurlIntegrator(MatrixCoefficient &mq)
      : MixedVectorIntegrator(mq) {}

protected:
   inline virtual bool VerifyFiniteElementTypes(
      const FiniteElement & trial_fe,
      const FiniteElement & test_fe) const
   {
      return (trial_fe.GetDim() == 3 && test_fe.GetDim() == 3 &&
              trial_fe.GetRangeType() == mfem::FiniteElement::VECTOR &&
              test_fe.GetDerivType()  == mfem::FiniteElement::CURL );
   }

   inline virtual const char * FiniteElementTypeFailureMessage() const
   {
      return "MixedVectorWeakCurlIntegrator:  "
             "Trial space must be vector field in 3D and the "
             "test space must be H(Curl)";
   }

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   {
      test_fe.CalcPhysCurlShape(Trans, shape);
   }
};

/** Class for integrating the bilinear form a(u,v) := - (Q u, grad v) in either
    2D or 3D and where Q is an optional coefficient (of type scalar, matrix, or
    diagonal matrix) u is in H(Div) or H(Curl) and v is in H1. */
class MixedVectorWeakDivergenceIntegrator : public MixedVectorIntegrator
{
public:
   MixedVectorWeakDivergenceIntegrator() {}
   MixedVectorWeakDivergenceIntegrator(Coefficient &q)
      : MixedVectorIntegrator(q) {}
   MixedVectorWeakDivergenceIntegrator(VectorCoefficient &dq)
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

   inline virtual void CalcTestShape(const FiniteElement & test_fe,
                                     ElementTransformation &Trans,
                                     DenseMatrix & shape)
   {
      test_fe.CalcPhysDShape(Trans, shape);
      shape *= -1.0;
   }
};

/** Class for integrating the bilinear form a(u,v) := (Q grad u, v) where Q is a
    scalar coefficient, and v is a vector with components v_i in the same space
    as u. */
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
   GradientIntegrator(Coefficient *_q) :
      Q{_q}, trial_maps{NULL}, test_maps{NULL}, geom{NULL}
   { }
   GradientIntegrator(Coefficient &q) :
      Q{&q}, trial_maps{NULL}, test_maps{NULL}, geom{NULL}
   { }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes);

   virtual void AddMultPA(const Vector &x, Vector &y) const;
   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};

/** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v) where Q
    can be a scalar or a matrix coefficient. */
class DiffusionIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;

private:
   Vector vec, pointflux, shape;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, dshapedxt, invdfdx, mq;
   DenseMatrix te_dshape, te_dshapedxt;
#endif

   // PA extension
   const FiniteElementSpace *fespace;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, dofs1D, quad1D;
   Vector pa_data;

#ifdef MFEM_USE_CEED
   // CEED extension
   CeedData* ceedDataPtr;
#endif

public:
   /// Construct a diffusion integrator with coefficient Q = 1
   DiffusionIntegrator()
   {
      Q = NULL;
      MQ = NULL;
      maps = NULL;
      geom = NULL;
#ifdef MFEM_USE_CEED
      ceedDataPtr = NULL;
#endif
   }

   /// Construct a diffusion integrator with a scalar coefficient q
   DiffusionIntegrator(Coefficient &q)
      : Q(&q)
   {
      MQ = NULL;
      maps = NULL;
      geom = NULL;
#ifdef MFEM_USE_CEED
      ceedDataPtr = NULL;
#endif
   }

   /// Construct a diffusion integrator with a matrix coefficient q
   DiffusionIntegrator(MatrixCoefficient &q)
      : MQ(&q)
   {
      Q = NULL;
      maps = NULL;
      geom = NULL;
#ifdef MFEM_USE_CEED
      ceedDataPtr = NULL;
#endif
   }

   virtual ~DiffusionIntegrator()
   {
#ifdef MFEM_USE_CEED
      delete ceedDataPtr;
#endif
   }

   /** Given a particular Finite Element computes the element stiffness matrix
       elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   /** Given a trial and test Finite Element computes the element stiffness
       matrix elmat. */
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   /// Perform the local action of the BilinearFormIntegrator
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u, const FiniteElement &fluxelem,
                                   Vector &flux, bool with_coef = true);

   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL);

   using BilinearFormIntegrator::AssemblePA;

   virtual void AssemblePA(const FiniteElementSpace &fes);

   virtual void AssembleDiagonalPA(Vector &diag);

   virtual void AddMultPA(const Vector&, Vector&) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe);

   void SetupPA(const FiniteElementSpace &fes, const bool force = false);
};

/** Class for local mass matrix assembling a(u,v) := (Q u, v) */
class MassIntegrator: public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   Coefficient *Q;
   // PA extension
   const FiniteElementSpace *fespace;
   Vector pa_data;
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

#ifdef MFEM_USE_CEED
   // CEED extension
   CeedData* ceedDataPtr;
#endif

public:
   MassIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir)
   {
      Q = NULL;
      maps = NULL;
      geom = NULL;
#ifdef MFEM_USE_CEED
      ceedDataPtr = NULL;
#endif
   }

   /// Construct a mass integrator with coefficient q
   MassIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(&q)
   {
      maps = NULL;
      geom = NULL;
#ifdef MFEM_USE_CEED
      ceedDataPtr = NULL;
#endif
   }

   virtual ~MassIntegrator()
   {
#ifdef MFEM_USE_CEED
      delete ceedDataPtr;
#endif
   }
   /** Given a particular Finite Element computes the element mass matrix
       elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   using BilinearFormIntegrator::AssemblePA;

   virtual void AssemblePA(const FiniteElementSpace &fes);

   virtual void AssembleDiagonalPA(Vector &diag);

   virtual void AddMultPA(const Vector&, Vector&) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);

   void SetupPA(const FiniteElementSpace &fes, const bool force = false);
};

class BoundaryMassIntegrator : public MassIntegrator
{
public:
   BoundaryMassIntegrator(Coefficient &q) : MassIntegrator(q) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;

   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/// alpha (q . grad u, v)
class ConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *Q;
   double alpha;
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
   ConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(&q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);

   using BilinearFormIntegrator::AssemblePA;

   virtual void AssemblePA(const FiniteElementSpace&);

   virtual void AddMultPA(const Vector&, Vector&) const;

   static const IntegrationRule &GetRule(const FiniteElement &el,
                                         ElementTransformation &Trans);

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};

/// alpha (q . grad u, v) using the "group" FE discretization
class GroupConvectionIntegrator : public BilinearFormIntegrator
{
protected:
   VectorCoefficient *Q;
   double alpha;

private:
   DenseMatrix dshape, adjJ, Q_nodal, grad;
   Vector shape;

public:
   GroupConvectionIntegrator(VectorCoefficient &q, double a = 1.0)
      : Q(&q) { alpha = a; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

/** Class for integrating the bilinear form a(u,v) := (Q u, v),
    where u=(u1,...,un) and v=(v1,...,vn); ui and vi are defined
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
      : vdim(-1), Q(&q) { VQ = NULL; MQ = NULL; Q_order = qo; }
   VectorMassIntegrator(Coefficient &q, const IntegrationRule *ir)
      : BilinearFormIntegrator(ir), vdim(-1), Q(&q)
   { VQ = NULL; MQ = NULL; Q_order = 0; }
   /// Construct an integrator with diagonal coefficient q
   VectorMassIntegrator(VectorCoefficient &q, int qo = 0)
      : vdim(q.GetVDim()), VQ(&q) { Q = NULL; MQ = NULL; Q_order = qo; }
   /// Construct an integrator with matrix coefficient q
   VectorMassIntegrator(MatrixCoefficient &q, int qo = 0)
      : vdim(q.GetVDim()), MQ(&q) { Q = NULL; VQ = NULL; Q_order = qo; }

   int GetVDim() const { return vdim; }
   void SetVDim(int vdim) { this->vdim = vdim; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &fes);
   virtual void AssembleDiagonalPA(Vector &diag);
   virtual void AddMultPA(const Vector &x, Vector &y) const;
};


/** Class for integrating (div u, p) where u is a vector field given by
    VectorFiniteElement through Piola transformation (for RT elements); p is
    scalar function given by FiniteElement through standard transformation.
    Here, u is the trial function and p is the test function.

    Note: the element matrix returned by AssembleElementMatrix2 does NOT depend
    on the ElementTransformation Trans. */
class VectorFEDivergenceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
#ifndef MFEM_THREAD_SAFE
   Vector divshape, shape;
#endif

public:
   VectorFEDivergenceIntegrator() { Q = NULL; }
   VectorFEDivergenceIntegrator(Coefficient &q) { Q = &q; }
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


/** Integrator for `(-Q u, grad v)` for Nedelec (`u`) and H1 (`v`) elements.
    This is equivalent to a weak divergence of the Nedelec basis functions. */
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
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/** Integrator for (curl u, v) for Nedelec and RT elements. If the trial and
    test spaces are switched, assembles the form (u, curl v). */
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
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat) { }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// Class for integrating (Q D_i(u), v); u and v are scalars
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
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat)
   { AssembleElementMatrix2(el,el,Trans,elmat); }
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// Integrator for (curl u, curl v) for Nedelec elements
class CurlCurlIntegrator: public BilinearFormIntegrator
{
private:
   Vector vec, pointflux;
#ifndef MFEM_THREAD_SAFE
   DenseMatrix curlshape, curlshape_dFt, M;
   DenseMatrix vshape, projcurl;
#endif

protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:
   CurlCurlIntegrator() { Q = NULL; MQ = NULL; }
   /// Construct a bilinear form integrator for Nedelec elements
   CurlCurlIntegrator(Coefficient &q) : Q(&q) { MQ = NULL; }
   CurlCurlIntegrator(MatrixCoefficient &m) : MQ(&m) { Q = NULL; }

   /* Given a particular Finite Element, compute the
      element curl-curl matrix elmat */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);

   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u, const FiniteElement &fluxelem,
                                   Vector &flux, bool with_coef);

   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL);

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &fes);
   virtual void AddMultPA(const Vector &x, Vector &y) const;
   virtual void AssembleDiagonalPA(Vector& diag);
};

/** Integrator for (curl u, curl v) for FE spaces defined by 'dim' copies of a
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
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   /// Compute element energy: (1/2) (curl u, curl u)_E
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Tr,
                                   const Vector &elfun);
};

/** Integrator for (Q u, v), where Q is an optional coefficient (of type scalar,
    vector (diagonal matrix), or matrix), trial function u is in H(Curl) or
    H(Div), and test function v is in H(Curl), H(Div), or v=(v1,...,vn), where
    vi are in H1. */
class VectorFEMassIntegrator: public BilinearFormIntegrator
{
private:
   void Init(Coefficient *q, VectorCoefficient *vq, MatrixCoefficient *mq)
   { Q = q; VQ = vq; MQ = mq; }

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
   VectorCoefficient *VQ;
   MatrixCoefficient *MQ;

   // PA extension
   Vector pa_data;
   const DofToQuad *mapsO;         ///< Not owned. DOF-to-quad map, open.
   const DofToQuad *mapsC;         ///< Not owned. DOF-to-quad map, closed.
   const GeometricFactors *geom;   ///< Not owned
   int dim, ne, nq, dofs1D, quad1D;

public:
   VectorFEMassIntegrator() { Init(NULL, NULL, NULL); }
   VectorFEMassIntegrator(Coefficient *_q) { Init(_q, NULL, NULL); }
   VectorFEMassIntegrator(Coefficient &q) { Init(&q, NULL, NULL); }
   VectorFEMassIntegrator(VectorCoefficient *_vq) { Init(NULL, _vq, NULL); }
   VectorFEMassIntegrator(VectorCoefficient &vq) { Init(NULL, &vq, NULL); }
   VectorFEMassIntegrator(MatrixCoefficient *_mq) { Init(NULL, NULL, _mq); }
   VectorFEMassIntegrator(MatrixCoefficient &mq) { Init(NULL, NULL, &mq); }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &fes);
   virtual void AddMultPA(const Vector &x, Vector &y) const;
   virtual void AssembleDiagonalPA(Vector& diag);
};

/** Integrator for (Q div u, p) where u=(v1,...,vn) and all vi are in the same
    scalar FE space; p is also in a (different) scalar FE space.  */
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
   VectorDivergenceIntegrator(Coefficient *_q) :
      Q(_q), trial_maps(NULL), test_maps(NULL), geom(NULL)
   { }
   VectorDivergenceIntegrator(Coefficient &q) :
      Q(&q), trial_maps(NULL), test_maps(NULL), geom(NULL)
   { }

   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &trial_fes,
                           const FiniteElementSpace &test_fes);

   virtual void AddMultPA(const Vector &x, Vector &y) const;
   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;

   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe,
                                         ElementTransformation &Trans);
};

/// (Q div u, div v) for RT elements
class DivDivIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

private:
#ifndef MFEM_THREAD_SAFE
   Vector divshape;
#endif

public:
   DivDivIntegrator() { Q = NULL; }
   DivDivIntegrator(Coefficient &q) : Q(&q) { }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
};

/** Integrator for
      (Q grad u, grad v) = sum_i (Q grad u_i, grad v_i) e_i e_i^T
    for FE spaces defined by 'dim' copies of a scalar FE space. Where e_i
    is the unit vector in the i-th direction.  The resulting local element
    matrix is a block-diagonal matrix consisting of 'dim' copies of a scalar
    diffusion matrix in each diagonal block. */
class VectorDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;

   // PA extension
   const DofToQuad *maps;         ///< Not owned
   const GeometricFactors *geom;  ///< Not owned
   int dim, ne, dofs1D, quad1D;
   Vector pa_data;

private:
   DenseMatrix Jinv;
   DenseMatrix dshape;
   DenseMatrix gshape;
   DenseMatrix pelmat;

public:
   VectorDiffusionIntegrator() { Q = NULL; }
   VectorDiffusionIntegrator(Coefficient &q) { Q = &q; }

   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementVector(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const Vector &elfun, Vector &elvect);
   using BilinearFormIntegrator::AssemblePA;
   virtual void AssemblePA(const FiniteElementSpace &fes);
   virtual void AssembleDiagonalPA(Vector &diag);
   virtual void AddMultPA(const Vector &x, Vector &y) const;
};

/** Integrator for the linear elasticity form:
    a(u,v) = (lambda div(u), div(v)) + (2 mu e(u), e(v)),
    where e(v) = (1/2) (grad(v) + grad(v)^T).
    This is a 'Vector' integrator, i.e. defined for FE spaces
    using multiple copies of a scalar FE space. */
class ElasticityIntegrator : public BilinearFormIntegrator
{
protected:
   double q_lambda, q_mu;
   Coefficient *lambda, *mu;

private:
#ifndef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape, gshape, pelmat;
   Vector divshape;
#endif

public:
   ElasticityIntegrator(Coefficient &l, Coefficient &m)
   { lambda = &l; mu = &m; }
   /** With this constructor lambda = q_l * m and mu = q_m * m;
       if dim * q_l + 2 * q_m = 0 then trace(sigma) = 0. */
   ElasticityIntegrator(Coefficient &m, double q_l, double q_m)
   { lambda = NULL; mu = &m; q_lambda = q_l; q_mu = q_m; }

   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);

   /** Compute the stress corresponding to the local displacement @a u and
       interpolate it at the nodes of the given @a fluxelem. Only the symmetric
       part of the stress is stored, so that the size of @a flux is equal to
       the number of DOFs in @a fluxelem times dim*(dim+1)/2. In 2D, the order
       of the stress components is: s_xx, s_yy, s_xy. In 3D, it is: s_xx, s_yy,
       s_zz, s_xy, s_xz, s_yz. In other words, @a flux is the local vector for
       a FE space with dim*(dim+1)/2 vector components, based on the finite
       element @a fluxelem. */
   virtual void ComputeElementFlux(const FiniteElement &el,
                                   ElementTransformation &Trans,
                                   Vector &u,
                                   const FiniteElement &fluxelem,
                                   Vector &flux, bool with_coef = true);

   /** Compute the element energy (integral of the strain energy density)
       corresponding to the stress represented by @a flux which is a vector of
       coefficients multiplying the basis functions defined by @a fluxelem. In
       other words, @a flux is the local vector for a FE space with
       dim*(dim+1)/2 vector components, based on the finite element @a fluxelem.
       The number of components, dim*(dim+1)/2 is such that it represents the
       symmetric part of the (symmetric) stress tensor. The order of the
       components is: s_xx, s_yy, s_xy in 2D, and s_xx, s_yy, s_zz, s_xy, s_xz,
       s_yz in 3D. */
   virtual double ComputeFluxEnergy(const FiniteElement &fluxelem,
                                    ElementTransformation &Trans,
                                    Vector &flux, Vector *d_energy = NULL);
};

/** Integrator for the DG form:
    alpha < rho_u (u.n) {v},[w] > + beta < rho_u |u.n| [v],[w] >,
    where v and w are the trial and test variables, respectively, and rho/u are
    given scalar/vector coefficients. The vector coefficient, u, is assumed to
    be continuous across the faces and when given the scalar coefficient, rho,
    is assumed to be discontinuous. The integrator uses the upwind value of rho,
    rho_u, which is value from the side into which the vector coefficient, u,
    points. */
class DGTraceIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *rho;
   VectorCoefficient *u;
   double alpha, beta;
   // PA extension
   Vector pa_data;
   const DofToQuad *maps;             ///< Not owned
   const FaceGeometricFactors *geom;  ///< Not owned
   int dim, nf, nq, dofs1D, quad1D;

private:
   Vector shape1, shape2;

public:
   /// Construct integrator with rho = 1.
   DGTraceIntegrator(VectorCoefficient &_u, double a, double b)
   { rho = NULL; u = &_u; alpha = a; beta = b; }

   DGTraceIntegrator(Coefficient &_rho, VectorCoefficient &_u,
                     double a, double b)
   { rho = &_rho; u = &_u; alpha = a; beta = b; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

   using BilinearFormIntegrator::AssemblePA;

   virtual void AssemblePAInteriorFaces(const FiniteElementSpace &fes);

   virtual void AssemblePABoundaryFaces(const FiniteElementSpace &fes);

   virtual void AddMultTransposePA(const Vector &x, Vector &y) const;

   virtual void AddMultPA(const Vector&, Vector&) const;

   static const IntegrationRule &GetRule(Geometry::Type geom, int order,
                                         FaceElementTransformations &T);

private:
   void SetupPA(const FiniteElementSpace &fes, FaceType type);
};

/** Integrator for the DG form:

    - < {(Q grad(u)).n}, [v] > + sigma < [u], {(Q grad(v)).n} >
    + kappa < {h^{-1} Q} [u], [v] >,

    where Q is a scalar or matrix diffusion coefficient and u, v are the trial
    and test spaces, respectively. The parameters sigma and kappa determine the
    DG method to be used (when this integrator is added to the "broken"
    DiffusionIntegrator):
    * sigma = -1, kappa >= kappa0: symm. interior penalty (IP or SIPG) method,
    * sigma = +1, kappa > 0: non-symmetric interior penalty (NIPG) method,
    * sigma = +1, kappa = 0: the method of Baumann and Oden. */
class DGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;

   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   DGDiffusionIntegrator(const double s, const double k)
      : Q(NULL), MQ(NULL), sigma(s), kappa(k) { }
   DGDiffusionIntegrator(Coefficient &q, const double s, const double k)
      : Q(&q), MQ(NULL), sigma(s), kappa(k) { }
   DGDiffusionIntegrator(MatrixCoefficient &q, const double s, const double k)
      : Q(NULL), MQ(&q), sigma(s), kappa(k) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the DG elasticity form, for the formulations see:
    - PhD Thesis of Jonas De Basabe, High-Order Finite %Element Methods for
      Seismic Wave Propagation, UT Austin, 2009, p. 23, and references therein
    - Peter Hansbo and Mats G. Larson, Discontinuous Galerkin and the
      Crouzeix-Raviart %Element: Application to Elasticity, PREPRINT 2000-09,
      p.3

    \f[
    - \left< \{ \tau(u) \}, [v] \right> + \alpha \left< \{ \tau(v) \}, [u]
        \right> + \kappa \left< h^{-1} \{ \lambda + 2 \mu \} [u], [v] \right>
    \f]

    where \f$ \left<u, v\right> = \int_{F} u \cdot v \f$, and \f$ F \f$ is a
    face which is either a boundary face \f$ F_b \f$ of an element \f$ K \f$ or
    an interior face \f$ F_i \f$ separating elements \f$ K_1 \f$ and \f$ K_2 \f$.

    In the bilinear form above \f$ \tau(u) \f$ is traction, and it's also
    \f$ \tau(u) = \sigma(u) \cdot \vec{n} \f$, where \f$ \sigma(u) \f$ is
    stress, and \f$ \vec{n} \f$ is the unit normal vector w.r.t. to \f$ F \f$.

    In other words, we have
    \f[
    - \left< \{ \sigma(u) \cdot \vec{n} \}, [v] \right> + \alpha \left< \{
        \sigma(v) \cdot \vec{n} \}, [u] \right> + \kappa \left< h^{-1} \{
        \lambda + 2 \mu \} [u], [v] \right>
    \f]

    For isotropic media
    \f[
    \begin{split}
    \sigma(u) &= \lambda \nabla \cdot u I + 2 \mu \varepsilon(u) \\
              &= \lambda \nabla \cdot u I + 2 \mu \frac{1}{2} (\nabla u + \nabla
                 u^T) \\
              &= \lambda \nabla \cdot u I + \mu (\nabla u + \nabla u^T)
    \end{split}
    \f]

    where \f$ I \f$ is identity matrix, \f$ \lambda \f$ and \f$ \mu \f$ are Lame
    coefficients (see ElasticityIntegrator), \f$ u, v \f$ are the trial and test
    functions, respectively.

    The parameters \f$ \alpha \f$ and \f$ \kappa \f$ determine the DG method to
    use (when this integrator is added to the "broken" ElasticityIntegrator):

    - IIPG, \f$\alpha = 0\f$,
      C. Dawson, S. Sun, M. Wheeler, Compatible algorithms for coupled flow and
      transport, Comp. Meth. Appl. Mech. Eng., 193(23-26), 2565-2580, 2004.

    - SIPG, \f$\alpha = -1\f$,
      M. Grote, A. Schneebeli, D. Schotzau, Discontinuous Galerkin Finite
      %Element Method for the Wave Equation, SINUM, 44(6), 2408-2431, 2006.

    - NIPG, \f$\alpha = 1\f$,
      B. Riviere, M. Wheeler, V. Girault, A Priori Error Estimates for Finite
      %Element Methods Based on Discontinuous Approximation Spaces for Elliptic
      Problems, SINUM, 39(3), 902-931, 2001.

    This is a '%Vector' integrator, i.e. defined for FE spaces using multiple
    copies of a scalar FE space.
 */
class DGElasticityIntegrator : public BilinearFormIntegrator
{
public:
   DGElasticityIntegrator(double alpha_, double kappa_)
      : lambda(NULL), mu(NULL), alpha(alpha_), kappa(kappa_) { }

   DGElasticityIntegrator(Coefficient &lambda_, Coefficient &mu_,
                          double alpha_, double kappa_)
      : lambda(&lambda_), mu(&mu_), alpha(alpha_), kappa(kappa_) { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);

protected:
   Coefficient *lambda, *mu;
   double alpha, kappa;

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
      const double jmatcoef, const Vector &col_nL, const Vector &col_nM,
      const Vector &row_shape, const Vector &col_shape,
      const Vector &col_dshape_dnM, const DenseMatrix &col_dshape,
      DenseMatrix &elmat, DenseMatrix &jmat);
};

/** Integrator for the DPG form: < v, [w] > over all faces (the interface) where
    the trial variable v is defined on the interface and the test variable w is
    defined inside the elements, generally in a DG space. */
class TraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, shape1, shape2;

public:
   TraceJumpIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the form: < v, [w.n] > over all faces (the interface) where
    the trial variable v is defined on the interface and the test variable w is
    in an H(div)-conforming space. */
class NormalTraceJumpIntegrator : public BilinearFormIntegrator
{
private:
   Vector face_shape, normal, shape1_n, shape2_n;
   DenseMatrix shape1, shape2;

public:
   NormalTraceJumpIntegrator() { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                   const FiniteElement &test_fe1,
                                   const FiniteElement &test_fe2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Abstract class to serve as a base for local interpolators to be used in the
    DiscreteLinearOperator class. */
class DiscreteInterpolator : public BilinearFormIntegrator { };


/** Class for constructing the gradient as a DiscreteLinearOperator from an
    H1-conforming space to an H(curl)-conforming space. The range space can be
    vector L2 space as well. */
class GradientInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &h1_fe,
                                       const FiniteElement &nd_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { nd_fe.ProjectGrad(h1_fe, Trans, elmat); }
};


/** Class for constructing the identity map as a DiscreteLinearOperator. This
    is the discrete embedding matrix when the domain space is a subspace of
    the range space. Otherwise, a dof projection matrix is constructed. */
class IdentityInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { ran_fe.Project(dom_fe, Trans, elmat); }
};


/** Class for constructing the (local) discrete curl matrix which can be used
    as an integrator in a DiscreteLinearOperator object to assemble the global
    discrete curl matrix. */
class CurlInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { ran_fe.ProjectCurl(dom_fe, Trans, elmat); }
};


/** Class for constructing the (local) discrete divergence matrix which can
    be used as an integrator in a DiscreteLinearOperator object to assemble
    the global discrete divergence matrix.

    Note: Since the dofs in the L2_FECollection are nodal values, the local
    discrete divergence matrix (with an RT-type domain space) will depend on
    the transformation. On the other hand, the local matrix returned by
    VectorFEDivergenceIntegrator is independent of the transformation. */
class DivergenceInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat)
   { ran_fe.ProjectDiv(dom_fe, Trans, elmat); }
};


/** A trace face interpolator class for interpolating the normal component of
    the domain space, e.g. vector H1, into the range space, e.g. the trace of
    RT which uses FiniteElement::INTEGRAL map type. */
class NormalInterpolator : public DiscreteInterpolator
{
public:
   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/** Interpolator of a scalar coefficient multiplied by a scalar field onto
    another scalar field. Note that this can produce inaccurate fields unless
    the target is sufficiently high order. */
class ScalarProductInterpolator : public DiscreteInterpolator
{
public:
   ScalarProductInterpolator(Coefficient & sc) : Q(&sc) { }

   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);

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

   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
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

   virtual void AssembleElementMatrix2(const FiniteElement &dom_fe,
                                       const FiniteElement &ran_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
protected:
   VectorCoefficient *VQ;
};

/** Interpolator of the cross product between a vector coefficient and an
    H(curl)-conforming field onto an H(div)-conforming field. The range space
    can also be vector L2. */
class VectorCrossProductInterpolator : public DiscreteInterpolator
{
public:
   VectorCrossProductInterpolator(VectorCoefficient & vc)
      : VQ(&vc) { }

   virtual void AssembleElementMatrix2(const FiniteElement &nd_fe,
                                       const FiniteElement &rt_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
protected:
   VectorCoefficient *VQ;
};

/** Interpolator of the inner product between a vector coefficient and an
    H(div)-conforming field onto an L2-conforming field. The range space can
    also be H1. */
class VectorInnerProductInterpolator : public DiscreteInterpolator
{
public:
   VectorInnerProductInterpolator(VectorCoefficient & vc) : VQ(&vc) { }

   virtual void AssembleElementMatrix2(const FiniteElement &rt_fe,
                                       const FiniteElement &l2_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
protected:
   VectorCoefficient *VQ;
};

}

#endif
