#include "mfem.hpp"
#include "error.hpp"

namespace mfem
{

double CalculateH10Error2(GridFunction *sol, VectorCoefficient *exgrad,
                          Array<double> *elemError, Array<int> *elemRef,
                          int intOrder)
{
   const FiniteElementSpace *fes = sol->FESpace();
   Mesh* mesh = fes->GetMesh();

   Vector e_grad, a_grad, el_dofs, q_grad;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   const FiniteElement *fe;
   ElementTransformation *transf;

   int dim = mesh->Dimension();
   e_grad.SetSize(dim);
   a_grad.SetSize(dim);
   q_grad.SetSize(dim);
   Jinv.SetSize(dim);

   double error = 0.0;
   if (elemError) { elemError->SetSize(mesh->GetNE()); }
   if (elemRef)   { elemRef->SetSize(mesh->GetNE()); }

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      int fdof = fe->GetDof();
      transf = mesh->GetElementTransformation(i);
      el_dofs.SetSize(fdof);
      dshape.SetSize(fdof, dim);
      dshapet.SetSize(fdof, dim);

      fes->GetElementVDofs(i, vdofs);
      for (int k = 0; k < fdof; k++)
      {
         el_dofs(k) = (vdofs[k] >= 0) ? (*sol)(vdofs[k])
                                      : -(*sol)(-1-vdofs[k]);
      }

      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intOrder);

      // integrate the H^1_0 error
      double el_err = 0.0, a_dxyz[3] = { 0, 0, 0 };
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);

         transf->SetIntPoint(&ip);
         CalcInverse(transf->Jacobian(), Jinv);
         double w = ip.weight * transf->Weight();

         exgrad->Eval(e_grad, *transf, ip);

         fe->CalcDShape(ip, dshape);
         Mult(dshape, Jinv, dshapet);
         dshapet.MultTranspose(el_dofs, a_grad);

         e_grad -= a_grad;
         el_err += w * (e_grad * e_grad);

         // anisotropic indicators
         transf->Jacobian().MultTranspose(e_grad, q_grad);
         for (int k = 0; k < dim; k++)
         {
            a_dxyz[k] += w * (q_grad[k] * q_grad[k]);
         }
      }

      error += el_err;
      if (elemError)
      {
         (*elemError)[i] = fabs(el_err);
      }

      // determine what type of anisotropic refinement (if any) is suitable
      if (elemRef)
      {
         double sum = 0;
         for (int k = 0; k < dim; k++)
         {
            sum += a_dxyz[k];
         }

         const double thresh = 0.2 * 3/dim;
         int ref = 0;
         for (int k = 0; k < dim; k++)
         {
            if (a_dxyz[k] / sum > thresh)
            {
               ref |= (1 << k);
            }
         }

         (*elemRef)[i] = ref;
      }
   }

   return error;
}


} // namespace mfem
