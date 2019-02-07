#ifndef DGPA_ADVECTION
#define DGPA_ADVECTION

#include "dg.hpp"
#include "cons_law.hpp"

namespace mfem
{
namespace dg
{

struct DAdv
{
   int dim;

   DAdv() : dim(0) { }
   int NComponents() const
   {
      return 1;
   }
   void operator()(double *u, double *F) const
   {
      for (int d = 0; d < dim; ++d)
      {
         F[d] = u[0];
      }
   }
};

struct DAdvUpwinding
{
   int dim;
   DAdvUpwinding() : dim(0) { }
   int NComponents() const
   {
      return 1;
   }
   void operator()(double *uL, double *uR, double *n, double *Fhat) const
   {
      double bDotN = 0.0;
      for (int d = 0; d < dim; ++d)
      {
         bDotN += 1.0*n[d];
      }
      Fhat[0] = 0.0;
      double u = (bDotN >= 0.0) ? uL[0] : uR[0];
      for (int d = 0; d < dim; ++d)
      {
         Fhat[0] += u*n[d];
      }
   }
};

struct Advection : ConservationLaw<DAdv, DAdvUpwinding>
{
   Advection(const PartialAssembly *pa_, int dim)
      : ConservationLaw(pa_, dim, 1) { }
};

}
}

#endif