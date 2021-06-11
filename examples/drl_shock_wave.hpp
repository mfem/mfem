#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

const double alpha =  200.0; // standard params
const double center = -0.05;
const double radius = 0.7;
template<typename T> T sqr(T x) { return x*x; }
double layer2_exsol(Vector &p)
{
   double x = p(0), y = p(1);
   double r = sqrt(sqr(x - center) + sqr(y - center));
   return atan(alpha * (r - radius));
}

void layer2_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   double r = sqrt(sqr(x - center) + sqr(y - center));
   double u = r * (sqr(alpha) * sqr(r - radius) + 1);
   grad(0) = alpha * (x - center) / u;
   grad(1) = alpha * (y - center) / u;
}

double layer2_laplace(Vector &p)
{
   double x = p(0), y = p(1);
   double r = sqr(y - center) + sqr(x - center);
   double u = sqr(alpha) * sqr(sqrt(r) - radius) + 1;

   return 2 * pow(alpha,3) * (sqrt(r) - radius) * sqr(y - center) / (r * sqr(u))
          + alpha * sqr(y - center) / (pow(r, 1.5) * u)
          - 2 * alpha / (sqrt(r) * u)
          + 2 * pow(alpha,3) * (sqrt(r) - radius) * sqr(x - center) / (r * sqr(u))
          + alpha * sqr(x - center) / (pow(r, 1.5) * u);
}

static double safeSqrt(double x)
{
   if (x < 0.0)
      return -sqrt(-x);
   else
      return sqrt(x);
}

double CalculateH10Error(GridFunction *sol, VectorCoefficient *exgrad,
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
   if (elemError) elemError->SetSize(mesh->GetNE());
   if (elemRef)   elemRef->SetSize(mesh->GetNE());

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
         if (vdofs[k] >= 0)
            el_dofs(k) =  (*sol)(vdofs[k]);
         else
            el_dofs(k) = -(*sol)(-1-vdofs[k]);

      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intOrder);

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

         transf->Jacobian().MultTranspose(e_grad, q_grad);
         for (int k = 0; k < dim; k++)
         {
            a_dxyz[k] += w * (q_grad[k] * q_grad[k]);
         }
      }

      error += el_err;
      if (elemError)
         (*elemError)[i] = sqrt(fabs(el_err));

      if (elemRef)
      {
         double sum = 0;
         for (int k = 0; k < dim; k++)
            sum += a_dxyz[k];

         const double thresh = 0.2 * 3/dim;
         int ref = 0;
         for (int k = 0; k < dim; k++)
            if (a_dxyz[k] / sum > thresh)
               ref |= (1 << k);

         (*elemRef)[i] = ref;
      }
   }

   return safeSqrt(error);
}