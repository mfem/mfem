#ifndef DGPA_BURGERS
#define DGPA_BURGERS

#include "dg.hpp"
#include "cons_law.hpp"

namespace mfem
{
namespace dg
{

struct DBurgers
{
   int dim;

   DBurgers() : dim(0) { }
   int NComponents() const
   {
      return 1;
   }
   void operator()(double *u, double *F) const
   {
      for (int d = 0; d < dim; ++d)
      {
         F[d] = 0.5*u[0]*u[0];
      }
   }
};

struct DBurgersRiemann
{
   int dim;
   DBurgersRiemann() : dim(0) { }
   int NComponents() const
   {
      return 1;
   }
   void operator()(double *uL, double *uR, double *n, double *Fhat) const
   {
      double cL = 0.5*uL[0]*n[0];
      double cR = 0.5*uR[0]*n[0];
      double avg = 0.5*(cL + cR);

      if (cL > cR)
      {
         Fhat[0] = (avg > 0) ? cL*uL[0] : cR*uR[0];
      }
      else
      {
         if (cL > 0) Fhat[0] = cL*uL[0];
         else if (cL*cR < 0) Fhat[0] = 0;
         else Fhat[0] = cR*uR[0];
      }
   }
};

using Burgers = ConservationLaw<DBurgers, DBurgersRiemann>;

}
}

#endif