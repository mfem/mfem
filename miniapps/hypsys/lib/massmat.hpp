#ifndef HYPSYS_MASSMAT
#define HYPSYS_MASSMAT

#include "../../../mfem.hpp"

using namespace mfem;

class MassMatrixDG : public Operator
{
   friend class InverseMassMatrixDG;

   const FiniteElementSpace *fes;

public:
   DenseTensor M;

   MassMatrixDG(const FiniteElementSpace *fes_);
   ~MassMatrixDG() {};
   void Mult(const Vector &x, Vector &y) const;
};

class InverseMassMatrixDG : public Operator
{
   const MassMatrixDG *mass;
   const FiniteElementSpace *fes;

public:
   DenseTensor Minv;

   InverseMassMatrixDG(const MassMatrixDG *mass_);
   ~InverseMassMatrixDG() { };
   virtual void Mult(const Vector &x, Vector &y) const;
};

#endif
