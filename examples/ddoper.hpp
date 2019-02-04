#ifndef DDOPER_HPP
#define DDOPER_HPP

#include "mfem.hpp"

using namespace mfem;
using namespace std;


class DDMInterfaceOperator : public Operator
{
public:
  DDMInterfaceOperator(const int numSubdomains_, ParMesh **pmeshSD_, const int orderND) :
    numSubdomains(numSubdomains_), pmeshSD(pmeshSD_), fec(orderND, pmeshSD_[0]->Dimension())
  {
    numInterfaces = 0;
    /*    
    fespace = new ParFiniteElementSpace*[numSubdomains];

    for (int s=0; s<numSubdomains; ++s)
      {
	if (pmeshSD[s] == NULL)
	  fespace[s] = NULL;
	else
	  fespace[s] = new ParFiniteElementSpace(pmeshSD[s], &fec);
      }
    */
  }

  virtual void Mult(const Vector & x, Vector & y) const
  {
    // x and y are vectors of DOF's on the subdomain interfaces and exterior boundary. 
    // Degrees of freedom in x and y are ordered as follows: x = [x_0, x_1, ..., x_{N-1}];
    // N = numSubdomains, and on subdomain m, x_m = [u_m^s, f_m, \rho_m];
    // u_m^s is the vector of DOF's of u on the surface of subdomain m, for a field u in a Nedelec space on subdomain m;
    // f_m is an auxiliary vector of DOF's in a Nedelec space on the surface of subdomain m;
    // \rho_m is an auxiliary vector of DOF's in an H^1 (actually H^{1/2}) FE space on the surface of subdomain m.
    // The surface of subdomain m is represented as the union of subdomain interfaces and a subset of the exterior boundary.
    // The surface bilinear forms and their matrices are defined on subdomain interfaces, not the entire subdomain boundary.
    // The surface DOF's for a subdomain are be defined by the entire subdomain mesh boundary, and we must use maps between
    // those surface DOF's and DOF's on the individual interfaces.

    
  }

  
private:
  
  const int numSubdomains;
  int numInterfaces;
  
  ParMesh **pmeshSD;

  ND_FECollection fec;
  ParFiniteElementSpace **fespace;

};
  
#endif  // DDOPER_HPP
