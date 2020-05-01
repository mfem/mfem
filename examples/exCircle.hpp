#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "algoim_quad.hpp"
/// Class for domain integration L(v) := (f, v)
using namespace mfem;
//function that checks if an element is `cut` by `embedded circle` or  not
bool cutByCircle(Mesh *mesh, int &elemid);
// function to get element center
void GetElementCenter(Mesh *mesh, int id, mfem::Vector &cent);
// find bounding box for a given cut element
template <int N>
void findBoundingBox(Mesh *mesh, int id, blitz::TinyVector<double,N> &xmin, blitz::TinyVector<double,N> &xmax);
// get integration rule for cut elements
template <int N>
void GetCutElementIntRule(Mesh* mesh, vector<int> cutelems, 
                              std::map<int, IntegrationRule *> &CutSquareIntRules);
class CutDomainIntegrator : public DeltaLFIntegrator
{
   Vector shape;
   Coefficient &Q;
   int oa, ob;
   int test;
   std::map<int, IntegrationRule *> CutIntRules;
public:
   /// Constructs a domain integrator with a given Coefficient
   CutDomainIntegrator(Coefficient &QF, std::map<int, IntegrationRule *> CutSqIntRules)
      :CutIntRules(CutSqIntRules),test(100), DeltaLFIntegrator(QF, NULL), Q(QF), oa(1), ob(1) { }
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