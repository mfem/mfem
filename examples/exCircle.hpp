#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "algoim_quad.hpp"
#include <list>
/// Class for domain integration L(v) := (f, v)
using namespace mfem;
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
                              std::map<int, IntegrationRule *> &CutSquareIntRules);
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
   CutDomainLFIntegrator(Coefficient &QF, int a = 2, int b = 0, 
   std::map<int, IntegrationRule *> CutSqIntRules, std::vector<bool> EmbeddedElems)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is OK
      :  CutIntRules(CutSqIntRules), EmbeddedElements(EmbeddedElems),
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
