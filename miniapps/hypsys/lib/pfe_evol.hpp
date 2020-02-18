#ifndef HYPSYS_PFE_EVOL
#define HYPSYS_PFE_EVOL

#include <fstream>
#include <iostream>
#include "../../../mfem.hpp"
#include "fe_evol.hpp"
#include "massmat.hpp"
#include "pdofs.hpp"
#include "ptools.hpp"
#include "../apps/advection.hpp"
#include "../apps/burgers.hpp"
#include "../apps/shallowwater.hpp"

using namespace std;
using namespace mfem;


class ParFE_Evolution : public FE_Evolution
{
public:
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction x_gf_MPI;
   const int xSizeMPI;

   ParFE_Evolution(ParFiniteElementSpace *pfes_, HyperbolicSystem *hyp_,
                   DofInfo &dofs_, EvolutionScheme scheme_,
                   const Vector &LumpedMassMat_);

   virtual ~ParFE_Evolution() { };

   void FaceEval(const Vector &x, Vector &y1, Vector &y2, Vector &xMPI,
                 int e, int i, int k) const;
   void EvolveStandard(const Vector &x, Vector &y) const override;
   void EvolveMCL     (const Vector &x, Vector &y) const override;

   double ConvergenceCheck(double dt, double tol, const Vector &u) const override;
};

#endif
