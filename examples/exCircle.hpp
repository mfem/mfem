#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "algoim_quad.hpp"
#include <list>
/// Class for domain integration L(v) := (f, v)
using namespace mfem;
using namespace std;
//function that checks if an element is `cut` by `embedded circle` or  not
bool cutByCircle(Mesh *mesh, int &elemid);
//function that checks if an element is inside the `embedded circle` or  not
bool insideBoundary(Mesh *mesh, int &elemid);
// function to get element center
void GetElementCenter(Mesh *mesh, int id, mfem::Vector &cent);
// find bounding box for a given cut element
template <int N>
void findBoundingBox(Mesh *mesh, int id, blitz::TinyVector<double,N> &xmin, blitz::TinyVector<double,N> &xmax);
// get integration rule for cut elements
template <int N>
void GetCutElementIntRule(Mesh* mesh, vector<int> cutelems, 
                              std::map<int, IntegrationRule *> &cutSquareIntRules);
// get integration rule for cut segments
template <int N>
void GetCutSegmentIntRule(Mesh* mesh, vector<int> cutelems, vector<int> cutinteriorFaces,
                              std::map<int, IntegrationRule *> &cutSegmentIntRules, 
                              std::map<int, IntegrationRule *> &cutInteriorFaceIntRules);
// this is required for parameter verification test problem                        
// get integration rule for cut elements
template <int N>
void GetCutSegIntRule(Mesh* mesh, vector<int> cutelems, 
                              std::map<int, IntegrationRule *> &cutSegIntRules);
class CutMesh: public Mesh
{
protected:
   Array<Element *> elements;
public:   
  //CutMesh();
  //void updateMesh(Mesh* mesh, Mesh* &mesh2);
};
class CutDomainIntegrator : public DeltaLFIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
   std::map<int, IntegrationRule *> CutIntRules;
public:
   /// Constructs a domain integrator with a given Coefficient
   CutDomainIntegrator(Coefficient &QF, std::map<int, IntegrationRule *> CutSqIntRules)
      :CutIntRules(CutSqIntRules), DeltaLFIntegrator(QF, NULL), Q(QF), oa(1), ob(1) { }
   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect);

   //using LinearFormIntegrator::AssembleRHSElementVect;
};

/// Class for domain integration L(v) := (f, v)
class CutDomainLFIntegrator : public DeltaLFIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
   std::map<int, IntegrationRule *> CutIntRules;
   std::vector<bool> EmbeddedElements;
public:
   /// Constructs a domain integrator with a given Coefficient
   CutDomainLFIntegrator(Coefficient &QF, std::map<int, IntegrationRule *> CutSqIntRules,
                               std::vector<bool> EmbeddedElems, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is OK
      :CutIntRules(CutSqIntRules), EmbeddedElements(EmbeddedElems),
         DeltaLFIntegrator(QF), Q(QF), oa(a), ob(b) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);

   virtual void AssembleDeltaElementVect(const FiniteElement &fe,
                                         ElementTransformation &Trans,
                                         Vector &elvect);
};

/** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v) where Q
    can be a scalar or a matrix coefficient. */
class CutDiffusionIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;

private:
   Vector vec, pointflux, shape;
   std::vector<bool> EmbeddedElements;
   std::map<int, IntegrationRule *> CutIntRules;
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
   CutDiffusionIntegrator(Coefficient &q, std::map<int, IntegrationRule *> CutSqIntRules,  
                           std::vector<bool> EmbeddedElems):
                         Q(&q), CutIntRules(CutSqIntRules), EmbeddedElements(EmbeddedElems)
   {
      MQ = NULL;
      maps = NULL;
      geom = NULL;
#ifdef MFEM_USE_CEED
      ceedDataPtr = NULL;
#endif
   }
virtual ~CutDiffusionIntegrator()
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
   static const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                         const FiniteElement &test_fe);
};

/** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v) where Q
    can be a scalar or a matrix coefficient. */
class CutBoundaryFaceIntegrator: public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;
   std::map<int, IntegrationRule *> cutSegmentIntRules;
   Vector shape1, dshape1dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, mq, adjJ;
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
  CutBoundaryFaceIntegrator(Coefficient &q, const double s, const double k, 
                         std::map<int, IntegrationRule *> cutSegmentIntRules):
                         Q(&q), MQ(NULL), sigma(s), kappa(k),  
                         cutSegmentIntRules(cutSegmentIntRules)
  {
  }
virtual ~CutBoundaryFaceIntegrator()
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
class CutDGDiffusionIntegrator : public BilinearFormIntegrator
{
protected:
   Coefficient *Q;
   MatrixCoefficient *MQ;
   double sigma, kappa;
   std::vector<int> cutinteriorFaces;
   std::map<int, bool> immersedFaces;
   std::map<int, IntegrationRule *> cutInteriorFaceIntRules;
   // these are not thread-safe!
   Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
   DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

public:
   CutDGDiffusionIntegrator(Coefficient &q, const double s, const double k, 
                                    std::map<int, bool> immersedFaces, 
                                    std::vector<int> cutinteriorFaces, 
                                    std::map<int, IntegrationRule *> cutInteriorFaceIntRules
                           )
      : Q(&q), MQ(NULL), sigma(s), kappa(k), 
      immersedFaces(immersedFaces) , cutinteriorFaces(cutinteriorFaces), 
      cutInteriorFaceIntRules(cutInteriorFaceIntRules) { }
   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

class CutDomainNLFIntegrator : public NonlinearFormIntegrator
{
private:
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;
   std::map<int, IntegrationRule *> CutIntRules;
   std::vector<bool> EmbeddedElements;

public:
   /** @param[in]  */
   CutDomainNLFIntegrator(std::map<int, IntegrationRule *> CutSqIntRules, std::vector<bool> EmbeddedElems):
                                             CutIntRules(CutSqIntRules), EmbeddedElements(EmbeddedElems) { }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
       @param[in] el     Type of FiniteElement.
       @param[in] Ttr    Represents ref->target coordinates transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   // virtual void AssembleElementVector(const FiniteElement &el,
   //                                    ElementTransformation &Ttr,
   //                                    const Vector &elfun, Vector &elvect);

   // virtual void AssembleElementGrad(const FiniteElement &el,
   //                                  ElementTransformation &Ttr,
   //                                  const Vector &elfun, DenseMatrix &elmat);
};

class CutBoundaryNLFIntegrator : public NonlinearFormIntegrator
{
private:
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;
   std::map<int, IntegrationRule *> CutIntRules;
   std::vector<bool> EmbeddedElements;

public:
   /** @param[in]  */
   //CutBoundaryNLFIntegrator();
   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
       @param[in] el     Type of FiniteElement.
       @param[in] Ttr    Represents ref->target coordinates transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   // virtual void AssembleElementVector(const FiniteElement &el,
   //                                    ElementTransformation &Ttr,
   //                                    const Vector &elfun, Vector &elvect);

   // virtual void AssembleElementGrad(const FiniteElement &el,
   //                                  ElementTransformation &Ttr,
   //                                  const Vector &elfun, DenseMatrix &elmat);
};
class CutImmersedBoundaryNLFIntegrator: public NonlinearFormIntegrator
{
private:
   DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;
   std::map<int, IntegrationRule *> CutSegIntRules;
   std::vector<bool> EmbeddedElements;

public:
   /** @param[in]  */
   CutImmersedBoundaryNLFIntegrator(std::map<int, IntegrationRule *> CutSegIntRules, std::vector<bool> EmbeddedElems):
                                             CutSegIntRules(CutSegIntRules), EmbeddedElements(EmbeddedElems) { }

   /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
       @param[in] el     Type of FiniteElement.
       @param[in] Ttr    Represents ref->target coordinates transformation.
       @param[in] elfun  Physical coordinates of the zone. */
   virtual double GetElementEnergy(const FiniteElement &el,
                                   ElementTransformation &Ttr,
                                   const Vector &elfun);

   // virtual void AssembleElementVector(const FiniteElement &el,
   //                                    ElementTransformation &Ttr,
   //                                    const Vector &elfun, Vector &elvect);

   // virtual void AssembleElementGrad(const FiniteElement &el,
   //                                  ElementTransformation &Ttr,
   //                                  const Vector &elfun, DenseMatrix &elmat);
};

