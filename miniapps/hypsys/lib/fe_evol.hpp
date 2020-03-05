#ifndef HYPSYS_FE_EVOL
#define HYPSYS_FE_EVOL

#include <fstream>
#include <iostream>
#include "../../../mfem.hpp"
#include "massmat.hpp"
#include "dofs.hpp"
#include "tools.hpp"
#include "../apps/advection.hpp"
#include "../apps/burgers.hpp"
#include "../apps/kpp.hpp"
#include "../apps/shallowwater.hpp"
#include "../apps/euler.hpp"

using namespace std;
using namespace mfem;


enum EvolutionScheme { Standard, MCL };

class FE_Evolution : public TimeDependentOperator
{
public:
   // General member variables.
   FiniteElementSpace *fes;
   const HyperbolicSystem *hyp;
   const DofInfo dofs;
   EvolutionScheme scheme;
   const IntegrationRule *IntRuleElem;
   const IntegrationRule *IntRuleFace;

   // Parameters that are needed repeatedly.
   int dim, nd, ne, nqe, nqf;

   // Shape function evaluations.
   DenseMatrix ShapeEval;
   DenseTensor DShapeEval;
   DenseTensor ShapeEvalFace;

   // Element and boundary integrals evaluated in quadrature points.
   DenseTensor ElemInt; // TODO better names
   DenseTensor BdrInt;
   DenseTensor OuterUnitNormals;

   // DG mass matrices.
   const Vector &LumpedMassMat;
   const MassMatrixDG *MassMat;
   const InverseMassMatrixDG *InvMassMat;

   // Tools to compute the discrete time derivative, needed repeatedly.
   mutable Array<int> vdofs;
   mutable Vector z, uOld, uElem, uEval, uNbrEval, NumFlux, normal;
   mutable DenseMatrix Flux, FluxNbr, mat1, mat2;
   mutable int DofInd, nbr;
   mutable double uNbr;

   mutable GridFunction inflow;

   FE_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                DofInfo &dofs_, EvolutionScheme scheme_,
                const Vector &LumpedMassMat_);

   virtual ~FE_Evolution()
   {
      delete MassMat;
      delete InvMassMat;
   }

   void Mult(const Vector &x, Vector &y) const override;
   void ElemEval(const Vector &uElem, Vector &uEval, int k) const;
   virtual void FaceEval(const Vector &x, Vector &y1, Vector &y2,
                         int e, int i, int k) const;
   void LaxFriedrichs(const Vector &x1, const Vector &x2, const Vector &normal,
                      Vector &y, int e, int k, int i) const;
   virtual void EvolveStandard(const Vector &x, Vector &y) const;
   virtual void EvolveMCL(const Vector &x, Vector &y) const;

   virtual double ConvergenceCheck(double dt, double tol, const Vector &u) const;
};

#endif
