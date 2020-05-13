#ifndef HYPSYS_MCL_EVOLUTION
#define HYPSYS_MCL_EVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class MCL_Evolution : public FE_Evolution
{
public:
   FiniteElementSpace *fesH1;

   double MassMatLumpedRef, tol = 0.;
   Vector DetJ;
   DenseTensor PrecGradOp, GradProd, Adjugates;
   DenseMatrix FaceMat, DistributionMatrix, MassMatLOR, Dof2LocNbr, MassMatRefInv;
   Bounds *bounds;

   mutable DenseTensor CTilde, CFull, NodalFluxes, uij;
   mutable DenseMatrix uFace, uNbrFace, mat3, DGFluxTerms, GalerkinRhs,
                       ElFlux, uDot, DTilde, wfi, BdrFlux, AntiDiffBdr, uijMin, uijMax;
   mutable Vector sif, vec1;

   explicit MCL_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                          DofInfo &dofs_);

   virtual ~MCL_Evolution() { delete bounds; delete fesH1; }

   void Mult(const Vector&x, Vector &y) const override;

   virtual void GetNodeVal(const Vector &uElem, Vector &uEval, int ) const;

   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
   void ComputePrecGradOp();
   void ComputeLORMassMatrix(DenseMatrix &RefMat, Geometry::Type gtype, bool UseDiagonalNbrs);
};

#endif
