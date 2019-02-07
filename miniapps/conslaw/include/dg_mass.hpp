#ifndef DGPA_MASS
#define DGPA_MASS

#include "dg_pa.hpp"

namespace mfem
{
namespace dg
{

class Mass : public Operator
{
   friend class MassInverse;

   const PartialAssembly *pa;
   const FiniteElementSpace *fes;
   DenseTensor M;
public:
   Mass(const PartialAssembly *pa_);
   void Mult(const Vector &x, Vector &y) const;
   const PartialAssembly *GetPA() const;
};

class MassInverse : public Operator
{
   const Mass *mass;
   const FiniteElementSpace *fes;
   DenseTensor Minv;
public:
   MassInverse(const Mass *mass_);
   virtual void Mult(const Vector &x, Vector &y) const;
};



} // namespace dg
} // namespace mfem

#endif