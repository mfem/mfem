#ifndef MFEM_SHALLOW_WATER
#define MFEM_SHALLOW_WATER

#include "../lib/hypsys.hpp"

class VolumeTerms
{
public:
   FiniteElementSpace *fes;
   DenseMatrix shape;
   DenseTensor dShape;
   DenseMatrix FluxEval;
   const int dim;
   const int NumEq;
   const IntegrationRule *ir;

   explicit VolumeTerms(FiniteElementSpace *_fes, const int _dim) : fes(_fes),
      dim(_dim), NumEq(_dim+1)
   {
      const FiniteElement *el = fes->GetFE(0);
      const int nd = el->GetDof();
      const int order = el->GetOrder();
      ir = &IntRules.Get(el->GetGeomType(), 2*order);
      const int nq = ir->GetNPoints();

      FluxEval.SetSize(NumEq, dim);

      Vector phi(nd);
      DenseMatrix dPhi(nd, dim);

      shape.SetSize(nq, nd);
      dShape.SetSize(nd, dim, nq);

      for (int k = 0; k < nq; k++)
      {
         const IntegrationPoint &ip = ir->IntPoint(k);
         el->CalcShape(ip, phi);
         shape.SetRow(k, phi);
         el->CalcDShape(ip, dPhi);
         dShape(k) = dPhi;
      }
   }

   virtual ~VolumeTerms() { };

   void EvalFluxFunction(const Vector &u,
                         DenseMatrix &FluxEval); // TODO this should belong to SWE
   void AssembleElementVolumeTerms(const int e, const DenseMatrix &uEl,
                                   DenseMatrix &VolTerms);
};

class ShallowWater
{
public:
   FiniteElementSpace *fes;
   double t0 = 0.;
   double tFinal;
   const int dim;
   const int NumEq;
   bool SolutionKnown = true;
   bool WriteErrors = false;

   VolumeTerms *vol;
   SparseMatrix K;
   Vector b;

   explicit ShallowWater(const Vector &bbmin, const Vector &bbmax,
                         const int config,
                         const double tEnd, const int _dim, FiniteElementSpace *_fes);
   virtual ~ShallowWater() { delete vol; };

   void PreprocessProblem(FiniteElementSpace *fes, GridFunction &u);
   void PostprocessProblem(const GridFunction &u, Array<double> &errors);
};

#endif
