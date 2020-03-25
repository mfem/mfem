#ifndef HYPSYS_PARGALERKINEVOLUTION
#define HYPSYS_PARGALERKINEVOLUTION

#include "galerkin.hpp"
#include "../lib/pdofs.hpp"
#include "../lib/ptools.hpp"

using namespace std;
using namespace mfem;

class ParGalerkinEvolution : public GalerkinEvolution
{
public:
   mutable ParGridFunction x_gf_MPI;

   explicit ParGalerkinEvolution(ParFiniteElementSpace *pfes_,
                                 HyperbolicSystem *hyp_, DofInfo &dofs_);

   virtual ~ParGalerkinEvolution() { }

   void Mult(const Vector&x, Vector &y) const override;
   double ConvergenceCheck(double dt, double tol, const Vector &u) const override;
};

#endif