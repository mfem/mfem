#ifndef HYPSYS_MCL_EVOLUTION
#define HYPSYS_MCL_EVOLUTION

#include "fe_evol.hpp"

using namespace std;
using namespace mfem;

class MCL_Evolution : public FE_Evolution
{
public:
   DenseTensor PrecGradOp, GradProd, Adjugates;
   DenseMatrix FaceMat, DistributionMatrix, MassMatLOR, Dof2LocNbr;

   mutable DenseTensor CTilde, CFull;
   mutable DenseMatrix uFace, uNbrFace;
   mutable Vector C_eij;
   mutable Array<int> eldofs;

   explicit MCL_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                          DofInfo &dofs_);

   virtual ~MCL_Evolution() { }

   void Mult(const Vector&x, Vector &y) const override;

   virtual void GetNodeVal(const Vector &uElem, Vector &uEval, int ) const;
   virtual void FaceTerm(const Vector &x, Vector &y1, Vector &y2,
                         const Vector &xMPI, const Vector &normal,
                         int e, int i, int k) const;
   virtual void LinearFluxLumping(const Vector &x1, const Vector &x2, const Vector &normal,
                                  Vector &y, int e, int j, int i) const;
   void ComputeTimeDerivative(const Vector &x, Vector &y,
                              const Vector &xMPI = serial) const;
   void ComputeLORMassMatrix(DenseMatrix &RefMat, Geometry::Type gtype, bool UseDiagonalNbrs);
};

#endif
