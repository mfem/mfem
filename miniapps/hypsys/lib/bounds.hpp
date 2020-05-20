#ifndef HYPSYS_BOUNDS
#define HYPSYS_BOUNDS

#include "../../../mfem.hpp"

using namespace std;
using namespace mfem;

class Bounds
{
public:
   FiniteElementSpace *fes, *fesH1;
   GridFunction x_min, x_max;  // min max values for each H1 dof.
   Vector xi_min, xi_max; // min max values for each L2 dof
   Array<int> eldofs, DofMapH1;
   int nd, ne, NumEq;

   Bounds(FiniteElementSpace *fes_, FiniteElementSpace *fesH1_);

   virtual ~Bounds() { }

   void FillDofMap();

   virtual void ComputeBounds(const Vector &x);
   virtual void ComputeElementBounds(int n, int e, const Vector &x) = 0;
   virtual void ComputeSequentialBounds(int n, int e, const Vector &x) = 0;
};

class TightBounds : public Bounds
{
public:
   DenseMatrix ClosestNbrs;

   TightBounds(FiniteElementSpace *fes_, FiniteElementSpace *fesH1_);
   ~TightBounds() { }

   virtual void ComputeElementBounds(int n, int e, const Vector &x) override;
   virtual void ComputeSequentialBounds(int n, int e, const Vector &x) override;
};

class LooseBounds : public Bounds
{
public:
   LooseBounds(FiniteElementSpace *fes_, FiniteElementSpace *fesH1_);
   ~LooseBounds() { }

   virtual void ComputeElementBounds(int n, int e, const Vector &x) override;
   virtual void ComputeSequentialBounds(int n, int e, const Vector &x) override;
};

void FillClosestNbrs(const FiniteElement *el, DenseMatrix &ClosestNbrs);

#endif
