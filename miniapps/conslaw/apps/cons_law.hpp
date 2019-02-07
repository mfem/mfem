#ifndef DGPA_CONS_LAW
#define DGPA_CONS_LAW

#include "dg.hpp"

namespace mfem
{
namespace dg
{

template <typename F, typename Fhat>
class ConservationLaw : Operator
{
   const PartialAssembly *pa;
   GtDB<F> vol;
   BtDB_face<Fhat> face;
   int nc;
public:
   ConservationLaw(const PartialAssembly *pa_, int dim, int nc_)
      : pa(pa_), vol(pa), face(pa), nc(nc_)
   {
      vol.d.dim = dim;
      face.d.dim = dim;
   }
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      vol.Mult(x, y);
      face.Mult(x, y);
   }
   int Size() const
   {
      return pa->GetFES()->GetNDofs()*nc;
   }
};

}
}

#endif