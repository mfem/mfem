#include "mortarintegrator.hpp"

namespace mfem
{
void L2MortarIntegrator::AssembleElementMatrix(
   const FiniteElement     &trial,
   const IntegrationRule   &trial_ir,
   ElementTransformation   &trial_Trans,
   const FiniteElement     &test,
   const IntegrationRule   &test_ir,
   ElementTransformation   &test_Trans,
   DenseMatrix          &elmat
)
{
   int tr_nd = trial.GetDof();
   int te_nd = test.GetDof();
   double w;

   Vector shape, te_shape;

   elmat.SetSize (te_nd, tr_nd);
   shape.SetSize (tr_nd);
   te_shape.SetSize (te_nd);

   elmat = 0.0;
   for (int i = 0; i < test_ir.GetNPoints(); i++)
   {
      const IntegrationPoint &trial_ip = trial_ir.IntPoint(i);
      const IntegrationPoint &test_ip  = test_ir.IntPoint(i);
      test_Trans.SetIntPoint (&test_ip);

      trial.CalcShape(trial_ip, shape);
      test.CalcShape(test_ip, te_shape);

      w = test_Trans.Weight() * test_ip.weight;

      te_shape *= w;
      AddMultVWt(te_shape, shape, elmat);
   }
}


void VectorL2MortarIntegrator::AssembleElementMatrix(
   const FiniteElement     &trial,
   const IntegrationRule   &trial_ir,
   ElementTransformation   &trial_Trans,
   const FiniteElement     &test,
   const IntegrationRule   &test_ir,
   ElementTransformation   &test_Trans,
   DenseMatrix          &elmat
)
{
   if ( test.GetRangeType() == FiniteElement::SCALAR && VQ )
   {
      // assume test is scalar FE and trial is vector FE
      int dim  = test.GetDim();
      int trial_dof = trial.GetDof();
      int test_dof = test.GetDof();
      double w;

      if (MQ)
         mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                    "   is not implemented for tensor materials");

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof, dim);
      Vector shape(test_dof);
      Vector D(dim);
#else
      trial_vshape.SetSize(trial_dof, dim);
      shape.SetSize(test_dof);
      D.SetSize(dim);
#endif

      elmat.SetSize (test_dof, trial_dof);

      elmat = 0.0;
      for (int i = 0; i < test_ir.GetNPoints(); i++)
      {
         const IntegrationPoint &trial_ip = trial_ir.IntPoint(i);
         const IntegrationPoint &test_ip  = test_ir.IntPoint(i);

         trial_Trans.SetIntPoint(&trial_ip);
         test_Trans.SetIntPoint (&test_ip);

         trial.CalcVShape(trial_Trans, trial_vshape);
         test.CalcShape(test_ip, shape);

         w = test_ip.weight * test_Trans.Weight();
         VQ->Eval(D, test_Trans, test_ip);
         D *= w;

         for (int d = 0; d < dim; d++)
         {
            for (int j = 0; j < test_dof; j++)
            {
               for (int k = 0; k < trial_dof; k++)
               {
                  elmat(j, k) += D[d] * shape(j) * trial_vshape(k, d);
               }
            }
         }
      }
   }
   else if ( test.GetRangeType() == FiniteElement::SCALAR )
   {
      // assume test is scalar FE and trial is vector FE
      int dim       = test.GetDim();
      int trial_dof = trial.GetDof();
      int test_dof  = test.GetDof();
      double w;

      if (VQ || MQ)
         mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                    "   is not implemented for vector/tensor permeability");

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof, dim);
      Vector shape(test_dof);
#else
      trial_vshape.SetSize(trial_dof, dim);
      shape.SetSize(test_dof);
#endif

      elmat.SetSize (dim*test_dof, trial_dof);


      elmat = 0.0;
      for (int i = 0; i < test_ir.GetNPoints(); i++)
      {
         const IntegrationPoint &trial_ip = trial_ir.IntPoint(i);
         const IntegrationPoint &test_ip  = test_ir.IntPoint(i);

         trial_Trans.SetIntPoint(&trial_ip);
         test_Trans.SetIntPoint (&test_ip);

         trial.CalcVShape(trial_Trans, trial_vshape);
         test.CalcShape(test_ip, shape);

         w = test_ip.weight * test_Trans.Weight();

         if (Q)
         {
            w *= Q -> Eval (test_Trans, test_ip);
         }

         for (int d = 0; d < dim; d++)
         {
            for (int j = 0; j < test_dof; j++)
            {
               for (int k = 0; k < trial_dof; k++)
               {
                  elmat(d * test_dof + j, k) += w * shape(j) * trial_vshape(k, d);
               }
            }
         }
      }
   }
   else
   {
      // assume both test and trial are vector FE
      int dim  = test.GetDim();
      int trial_dof = trial.GetDof();
      int test_dof = test.GetDof();
      double w;

      if (VQ || MQ)
         mfem_error("VectorFEMassIntegrator::AssembleElementMatrix2(...)\n"
                    "   is not implemented for vector/tensor permeability");

#ifdef MFEM_THREAD_SAFE
      DenseMatrix trial_vshape(trial_dof, dim);
      DenseMatrix test_vshape(test_dof,dim);
#else
      trial_vshape.SetSize(trial_dof, dim);
      test_vshape.SetSize(test_dof,dim);
#endif

      elmat.SetSize (test_dof, trial_dof);

      elmat = 0.0;
      for (int i = 0; i < test_ir.GetNPoints(); i++)
      {
         const IntegrationPoint &trial_ip = trial_ir.IntPoint(i);
         const IntegrationPoint &test_ip  = test_ir.IntPoint(i);

         trial_Trans.SetIntPoint(&trial_ip);
         test_Trans.SetIntPoint (&test_ip);

         trial.CalcVShape(trial_Trans, trial_vshape);
         test.CalcVShape(test_Trans, test_vshape);

         w = test_ip.weight * test_Trans.Weight();
         if (Q)
         {
            w *= Q -> Eval (test_Trans, test_ip);
         }

         for (int d = 0; d < dim; d++)
         {
            for (int j = 0; j < test_dof; j++)
            {
               for (int k = 0; k < trial_dof; k++)
               {
                  elmat(j, k) += w * test_vshape(j, d) * trial_vshape(k, d);
               }
            }
         }

         // test_vshape *= w;
         // AddMultABt (test_vshape, trial_vshape, elmat);
      }
   }
}

}
