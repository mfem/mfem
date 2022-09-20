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
   // if (true) {
   //   return 1.0;
   // }

   // const int *v = T.mesh->GetElement(T.ElementNo)->GetVertices();
   // double *r1 = T.mesh->GetVertex(v[0]);
   // double *r2 = T.mesh->GetVertex(v[1]);
   // double *r3 = T.mesh->GetVertex(v[2]);
   // double min_r = min(min(r1[0], r2[0]), r3[0]);
   // if (min_r < 0.1) {
   //   return min_r / (ri * model->get_mu());
   // }

   return 1.0 / (ri * model->get_mu());
}
