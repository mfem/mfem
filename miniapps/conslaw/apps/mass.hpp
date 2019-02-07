#ifndef DG_MASS_OPER
#define DG_MASS_OPER

#include "dg.hpp"

namespace mfem
{
namespace dg
{

struct DId
{
   int NComponents() const
   {
      return 1;
   }
   void operator()(double *u, double *F) const
   {
      F[0] = u[0];
   }
};

class DGMassPA : Operator
{
   const PartialAssembly *pa;
   BtDB<DId> oper;
public:
   DGMassPA(const PartialAssembly *pa_) : pa(pa_), oper(pa) { }
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      oper.Mult(x, y);
   }
};

}
}

#endif