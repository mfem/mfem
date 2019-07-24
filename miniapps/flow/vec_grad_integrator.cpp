#include "vec_grad_integrator.hpp"

namespace mfem
{
void VectorGradientIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe,
   const FiniteElement &test_fe,
   ElementTransformation &Trans,
   DenseMatrix &elmat)
{
   int dim = test_fe.GetDim();
   int trial_dof = trial_fe.GetDof();
   int test_dof = test_fe.GetDof();
   double c;

   Vector d_col;

   dshape.SetSize(trial_dof, dim);
   gshape.SetSize(trial_dof, dim);
   Jadj.SetSize(dim);
   shape.SetSize(test_dof);

   elmat.SetSize(dim * test_dof, trial_dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&trial_fe) + test_fe.GetOrder();
      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;

   elmat_comp.SetSize(test_dof, trial_dof);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trial_fe.CalcDShape(ip, dshape);
      test_fe.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), Jadj);

      Mult(dshape, Jadj, gshape);

      c = ip.weight;
      if (Q)
      {
         c *= Q->Eval(Trans, ip);
      }
      shape *= c;

      for (int d = 0; d < dim; ++d)
      {
         gshape.GetColumnReference(d, d_col);
         MultVWt(shape, d_col, elmat_comp);
         for (int jj = 0; jj < trial_dof; ++jj)
         {
            for (int ii = 0; ii < test_dof; ++ii)
            {
               elmat(d * test_dof + ii, jj) += elmat_comp(ii, jj);
            }
         }
      }
   }
}

} // namespace mfem
