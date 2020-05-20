#ifndef HYPSYS_MCL_EVOLUTION
#define HYPSYS_MCL_EVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class MCL_Evolution : public FE_Evolution
{
public:
   FiniteElementSpace *fesH1;

   double MassMatLumpedRef;
   const double dt;
   int NumLocNbr;
   Vector DetJ;
   DenseTensor PrecGradOp, GradProd, Adjugates;
   DenseMatrix FaceMat, DistributionMatrix, MassMatLOR, Dof2LocNbr, MassMatRefInv;
   Bounds *bounds;

   mutable DenseTensor CTilde, CFull, NodalFluxes, uij;
   mutable DenseMatrix uFace, uNbrFace, mat3, DGFluxTerms, GalerkinRhs,
                       ElFlux, uDot, DTilde, ufi, BdrFlux, AntiDiffBdr,
                       uijMin, uijMax, LimitedBarState;
   mutable Vector sif, vec1, diffusion, LimitedFluxState;

   explicit MCL_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                          DofInfo &dofs_, double dt_);

   virtual ~MCL_Evolution() { delete bounds; delete fesH1; }

   void Mult(const Vector&x, Vector &y) const override;

   virtual void GetNodeVal(const Vector &uElem, Vector &uEval, int ) const;
   virtual void GetFaceVal(const Vector &x, const Vector &xMPI, int e, int i) const;

   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
   void ComputeDissipativeMatrix(int e, const Vector &uElem) const;
   void ComputePrecGradOp();
   void ComputeLORMassMatrix(DenseMatrix &RefMat, Geometry::Type gtype, bool UseDiagonalNbrs);
};

#endif
