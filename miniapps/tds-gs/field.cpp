#include "mfem.hpp"
#include "field.hpp"

using namespace std;
using namespace mfem;

double FieldCoefficient::Eval(ElementTransformation & T,
                              const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double r(x(0));
   double z(x(1));
   int Component = 0;
   
   if (comp == 0) {
     // psi_z / R
     // psi_z = 0.0;
     // psi->GetDerivative(1, 1, psi_z);
     return psi_z->GetValue(T, ip, Component) / r;
   } else if (comp == 1) {
     // f / R
     double alpha = model->get_alpha_bar();
     double f_x = model->get_f_x();
     double psi_val = psi->GetValue(T, ip, Component);
     return (f_x + alpha * (psi_x - psi_val)) / r;
     // if ((psi_val > psi_ma) && (psi_val < psi_x)) {
     //   // return alpha * (f_x + alpha * (psi_x - psi_val)) / r;
     // } else {
     //   return 0.0;
     // }
     
   } else {
     // - psi_r / R
     // GridFunction psi_r(&fespace);
     // psi_r = 0.0;
     // psi->GetDerivative(1, 0, psi_r);
     return - psi_r->GetValue(T, ip, Component) / r;
   }

}
