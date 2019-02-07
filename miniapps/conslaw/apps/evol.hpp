#ifndef DGPA_EVOL
#define DGPA_EVOL

#include "dg.hpp"

namespace mfem
{
namespace dg
{

template <typename Oper>
class PAEvolution : public TimeDependentOperator
{
private:
   MassInverse *Minv;
   Oper *oper;
   mutable Vector z;
public:
   PAEvolution(MassInverse *Minv_, Oper *oper_)
      : TimeDependentOperator(oper_->Size()), Minv(Minv_), oper(oper_) { }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      z.SetSize(x.Size());
      oper->Mult(x, z);
      Minv->Mult(z, y);
   }

   virtual ~PAEvolution() { }
};

}
}

#endif