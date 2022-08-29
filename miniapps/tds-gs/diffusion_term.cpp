#include "mfem.hpp"
#include "diffusion_term.hpp"
using namespace mfem;
using namespace std;


double DiffusionIntegratorCoefficient::Eval(ElementTransformation & T,
                                            const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double ri(x(0));

   return 1.0 / (ri * model->get_mu());
}
