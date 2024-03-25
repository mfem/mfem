#include "stress_integrator.hpp"

using namespace mfem;

void StressIntegrator::AssembleElementMatrix(
   const FiniteElement &el,
   ElementTransformation &Tr,
   DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   dim = el.GetDim();
   vdim = dim;

   dshape.SetSize(dof, dim);
   S.SetSize(vdim, dim);

   elmat.SetSize(vdim * dof);
   pelmat.SetSize(dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &DiffusionIntegrator::GetRule(el,el);
   }

   elmat = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      el.CalcPhysDShape(Tr, dshape);

      double nu = Q->Eval(Tr, ip);

      for (int n = 0; n < dof; n++)
      {
         for (int m = 0; m < dof; m++)
         {
            pelmat(n, m) = nu * (dshape(n, m) + dshape(m, n));
         }
      }
      for (int k = 0; k < vdim; ++k)
      {
         elmat.AddMatrix(pelmat, dof*k, dof*k);
      }
   }
}
