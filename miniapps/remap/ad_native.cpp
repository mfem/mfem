#include "ad_native.hpp"

namespace mfem
{

void ADFunction::Gradient(const Vector &x, ElementTransformation &Tr,
                           const IntegrationPoint &ip,
                           Vector &J) const
{
   ProcessParameters(Tr, ip);
   Gradient(x, J);
}
void ADFunction::Gradient(const Vector &x, Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Gradient: x.Size() must match n_input");
   J.SetSize(x.Size());
   ADVector x_ad(x);
   for (int i=0; i < x.Size(); i++)
   {
      x_ad[i].gradient = 1.0;
      ADReal_t result = (*this)(x_ad);
      J[i] = result.gradient;
      x_ad[i].gradient = 0.0;
   }
}

void ADFunction::Hessian(const Vector &x, ElementTransformation &Tr,
                          const IntegrationPoint &ip,
                          DenseMatrix &H) const
{
   ProcessParameters(Tr, ip);
   Hessian(x, H);
}

void ADFunction::Hessian(const Vector &x, DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Hessian: x.Size() must match n_input");
   H.SetSize(x.Size(), x.Size());
   AD2Vector x_ad(x);
   for (int i=0; i<x.Size(); i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient.value = 1.0;
         AD2Real_t result = (*this)(x_ad);
         H(j, i) = result.gradient.gradient;
         H(i, j) = result.gradient.gradient;
         x_ad[j].gradient.value = 0.0; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}

} // namespace mfem
