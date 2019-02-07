#ifndef DG_EULER
#define DG_EULER

#include "cons_law.hpp"
#include "ns_autogen.hpp"

namespace mfem
{
namespace dg
{

struct EulerFlux
{
   int dim = 0;
   int NComponents() const
   {
      return dim + 2;
   }
   void operator()(double *u, double *F) const
   {
      eulerF(dim, u, F);
   }
};

struct EulerNumericalFlux
{
   int dim = 0;
   int NComponents() const
   {
      return dim + 2;
   }
   void operator()(double *uL, double *uR, double *n, double *Fhat) const
   {
      eulerFhat(dim, uR, uL, n, Fhat);
   }
};

struct Euler : ConservationLaw<EulerFlux, EulerNumericalFlux>
{
   Euler(const PartialAssembly *pa_, int dim)
      : ConservationLaw(pa_, dim, dim + 2) { }
};

}
}

#endif