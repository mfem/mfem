#ifndef HYPSYS_PFE_EVOL
#define HYPSYS_PFE_EVOL

#include "fe_evol.hpp"
#include "pdofs.hpp"
#include "ptools.hpp"

using namespace std;
using namespace mfem;


class ParFE_Evolution : public TimeDependentOperator
{
public:
   ParFiniteElementSpace *pfes;
   mutable ParGridFunction x_gf_MPI;
   EvolutionScheme scheme;

   // DG mass matrices.
   const MassMatrixDG *MassMat;
   const InverseMassMatrixDG *InvMassMat;

   mutable Vector z, uOld;

   explicit ParFE_Evolution(ParFiniteElementSpace *pfes_,
                            HyperbolicSystem *hyp_, DofInfo &dofs_,
                            EvolutionScheme scheme_);

   virtual ~ParFE_Evolution() { };

   void Mult(const Vector &x, Vector &y) const = 0;
   double ConvergenceCheck(double dt, double tol, const Vector &u) const;
};

#endif
