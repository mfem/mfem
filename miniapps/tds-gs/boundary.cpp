#include "mfem.hpp"
#include "boundary.hpp"

using namespace mfem;
using namespace std;

double BoundaryCoefficient::Eval(ElementTransformation & T,
                                 const IntegrationPoint & ip)
{
   double x_[3];
   Vector x(x_, 3);
   T.Transform(ip, x);
   double xr(x(0));
   double xz(x(1));

   if (order == 1) {
     double delta_p = sqrt(pow(xr, 2.0) + pow(rho_gamma + xz, 2.0));
     double delta_m = sqrt(pow(xr, 2.0) + pow(rho_gamma - xz, 2.0));

     if (xr == 0.0) {
       // coefficient blows up at r=0, however psi=0 at r=0
       return 0.0;
     }
     return (1.0 / delta_p + 1.0 / delta_m - 1.0 / rho_gamma) / (xr * model->get_mu());
   } else {

     double yr(y(0));
     double yz(y(1));

     double kxy = sqrt((4.0 * xr * yr)
                       / (pow(xr + yr, 2.0) + pow(xz - yz, 2.0)));
     
     double K = elliptic_fk(kxy);
     double E = elliptic_ek(kxy);

     return kxy * (E * (2.0 - pow(kxy, 2.0)) / (2.0 - 2.0 * pow(kxy, 2.0)) - K)
       / (4.0 * M_PI * pow(xr * yr, 1.5) * model->get_mu());
   }
}

double DoubleIntegralCoefficient::Eval(ElementTransformation & T,
                                       const IntegrationPoint & ip)
{

  double y_[3];
  Vector y(y_, 3);
  T.Transform(ip, y);
  boundary_coeff->SetY(y);

  BilinearForm bf(fespace);
  LinearForm lf(fespace);

  bf.AddBoundaryIntegrator(new BoundaryMassIntegrator(*boundary_coeff));
  // bf.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  bf.Assemble();
  bf.Mult(*psi, lf);

  return lf(*ones);
  // } else {
  //  LinearForm lf(fespace);
  //  lf.AddBoundaryIntegrator(BoundaryLFIntegrator(*boundary_coeff));
  //  lf.Assemble()
  //  bf.
  // }
}
