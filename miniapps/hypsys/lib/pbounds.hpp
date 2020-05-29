#ifndef HYPSYS_PBOUNDS
#define HYPSYS_PBOUNDS

#include "../../../mfem.hpp"
#include "bounds.hpp"

using namespace std;
using namespace mfem;

class ParBounds : public Bounds
{
public:
   ParFiniteElementSpace *pfes, *pfesH1;
   ParGridFunction px_min, px_max;  // min max values for each H1 dof.
   // Vector mom_min, mom_max;

   ParBounds(ParFiniteElementSpace *pfes_, ParFiniteElementSpace *pfesH1_);

   virtual ~ParBounds() { }

   virtual void ComputeBounds(const Vector &x) override;
};

class ParTightBounds : public ParBounds
{
public:
   DenseMatrix ClosestNbrs;

   ParTightBounds(ParFiniteElementSpace *pfes_, ParFiniteElementSpace *pfesH1_);
   ~ParTightBounds() { }

   virtual void ComputeElementBounds(int n, int e, const Vector &x) override;
   virtual void ComputeSequentialBounds(int n, int e, const Vector &x) override;
};

class ParLooseBounds : public ParBounds
{
public:
   ParLooseBounds(ParFiniteElementSpace *pfes_, ParFiniteElementSpace *pfesH1_);
   ~ParLooseBounds() { }

   virtual void ComputeElementBounds(int n, int e, const Vector &x) override;
   virtual void ComputeSequentialBounds(int n, int e, const Vector &x) override;
};

#endif