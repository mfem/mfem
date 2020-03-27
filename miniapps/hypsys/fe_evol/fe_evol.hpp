#ifndef HYPSYS_FE_EVOL
#define HYPSYS_FE_EVOL

#include "../lib/lib.hpp"
#include "../apps/hyperbolic_system.hpp"

using namespace std;
using namespace mfem;

// Auxiliary, only used to pass a non-used reference as parameter.
static Vector serial;

class FE_Evolution : public TimeDependentOperator
{
public:
   // General member variables.
   FiniteElementSpace *fes;
   const HyperbolicSystem *hyp;
   const DofInfo dofs;
   const IntegrationRule *IntRuleElem;
   const IntegrationRule *IntRuleFace;

   // Parameters that are needed repeatedly.
   int dim, nd, ne, nqe, nqf;
   const int xSizeMPI;

   // Shape function evaluations.
   DenseMatrix ShapeEval;
   DenseTensor DShapeEval;
   DenseTensor ShapeEvalFace;

   // Element and boundary integrals evaluated in quadrature points.
   DenseTensor ElemInt;
   DenseTensor BdrInt;
   DenseTensor OuterUnitNormals;

   // DG mass matrices.
   const MassMatrixDG *MassMat;
   const InverseMassMatrixDG *InvMassMat;
   Vector LumpedMassMat;

   // Tools to compute the discrete time derivative, needed repeatedly.
   mutable Array<int> vdofs;
   mutable Vector z, uOld, uElem, uEval, uNbrEval, NumFlux, normal;
   mutable DenseMatrix Flux, FluxNbr, mat1, mat2;
   mutable int DofInd, nbr;
   mutable double uNbr;

   mutable GridFunction inflow;

   FE_Evolution(FiniteElementSpace *fes_, HyperbolicSystem *hyp_,
                DofInfo &dofs_);

   virtual ~FE_Evolution()
   {
      delete MassMat;
      delete InvMassMat;
   }

   void Mult(const Vector &x, Vector &y) const = 0;
   virtual void ElemEval(const Vector &uElem, Vector &uEval, int k) const;
   virtual void FaceEval(const Vector &x, Vector &y1, Vector &y2,
                         const Vector &xMPI, const Vector &normal,
                         int e, int i, int k) const;
   void LaxFriedrichs(const Vector &x1, const Vector &x2, const Vector &normal,
                      Vector &y, int e, int k, int i) const;
   virtual double ConvergenceCheck(double dt, double tol, const Vector &u) const;
};

#endif
