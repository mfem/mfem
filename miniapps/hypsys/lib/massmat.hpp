#ifndef MFEM_MASS_MAT
#define MFEM_MASS_MAT

#include "mfem.hpp"

using namespace mfem;

class MassMatrixDG : public Operator
{
   friend class InverseMassMatrixDG;

   const FiniteElementSpace *fes;
   DenseTensor M;

public:
   MassMatrixDG(const FiniteElementSpace *fes_);
   ~MassMatrixDG() {};
   void Mult(const Vector &x, Vector &y) const;
};

class InverseMassMatrixDG : public Operator
{
   const MassMatrixDG *mass;
   const FiniteElementSpace *fes;
   DenseTensor Minv;

public:
   InverseMassMatrixDG(const MassMatrixDG *mass_);
   ~InverseMassMatrixDG() {};
   virtual void Mult(const Vector &x, Vector &y) const;
};

#endif
