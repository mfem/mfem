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

      double JxW = Tr.Weight() * ip.weight;
      // T.InverseJacobian();

      double nu = Q->Eval(Tr, ip);

      for (int i = 0; i < dof; i++)
      {
         for (int j = 0; j < dof; j++)
         {
            pelmat(i,j) = nu * (dshape(i, j) + dshape(j, i));
         }
      }

      // MultABt(Tr.InverseJacobian(), S, );

      // Mult_a_AAt(w, dshapedxt, pelmat);
      for (int k = 0; k < vdim; ++k)
      {
         elmat.AddMatrix(pelmat, dof*k, dof*k);
      }
   }
}
