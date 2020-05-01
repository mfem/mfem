#include "mfem.hpp"
#include "exact.hpp"

namespace mfem
{

// L-shape domain problem exact solution (2D)

double lshape_exsol(const Vector &p)
{
   double x = p(0), y = p(1);
   double r = sqrt(x*x + y*y);
   double a = atan2(y, x);
   if (a < 0) a += 2*M_PI;
   return pow(r, 2.0/3.0) * sin(2.0*a/3.0);
}

void lshape_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1);
   double a = atan2(y, x);
   if (a < 0) a += 2*M_PI;
   double theta23 = 2.0/3.0*a;
   double r23 = pow(x*x + y*y, 2.0/3.0);
   grad(0) = 2.0/3.0*x*sin(theta23)/(r23) - 2.0/3.0*y*cos(theta23)/(r23);
   grad(1) = 2.0/3.0*y*sin(theta23)/(r23) + 2.0/3.0*x*cos(theta23)/(r23);
}

double lshape_laplace(const Vector &p)
{
   return 0;
}


// Fichera's corner problem exact solution (3D)

double fichera_exsol(const Vector &p)
{
   double x = p(0), y = p(1), z = p(2);
   return pow(x*x + y*y + z*z, 0.25);
}

void fichera_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1), z = p(2);
   grad(0) = 0.5 * x * pow(x*x + y*y + z*z, -0.75);
   grad(1) = 0.5 * y * pow(x*x + y*y + z*z, -0.75);
   grad(2) = 0.5 * z * pow(x*x + y*y + z*z, -0.75);
}

double fichera_laplace(const Vector &p)
{
   double x = p(0), y = p(1), z = p(2);
   return -0.75 * pow(x*x + y*y + z*z, -0.75);
}


// inner layer problem exact solution (2D)

#if 1
const double alpha = 200.0; // standard params
const double center = -0.05;
const double radius = 0.7;
#elif 0
const double alpha = 400.0; // cube centered
const double center = 0.5;
const double radius = 0.3;
#elif 1
const double alpha = 80.0; // nurbs ball
const double center = -1;
const double radius = 1.8;
#else
const double alpha = 80.0; // hcurl
const double center = -0.05;
const double radius = 0.7;
#endif

template<typename T> T sqr(T x) { return x*x; }

double layer2_exsol(const Vector &p)
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

double layer2_laplace(const Vector &p)
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


// inner layer problem exact solution (3D)

double layer3_exsol(const Vector &p)
{
   double x = p(0), y = p(1), z = p(2);
   double r = sqrt(sqr(x - center) + sqr(y - center) + sqr(z - center));
   return atan(alpha * (r - radius));
}

void layer3_exgrad(const Vector &p, Vector &grad)
{
   double x = p(0), y = p(1), z = p(2);
   double t1 = x * x;
   double t4 = center * center;
   double t6 = y * y;
   double t9 = z * z;
   double t13 = sqrt(t1 - 0.2e1 * x * center + 0.3e1 * t4 + t6
                     - 0.2e1 * y * center + t9 - 0.2e1 * z * center);
   double t17 = alpha * alpha;
   double t19 = pow(t13 - radius, 0.2e1);
   grad(0) = alpha / t13 * (x - center) / (0.1e1 + t17 * t19);
   grad(1) = alpha / t13 * (y - center) / (0.1e1 + t17 * t19);
   grad(2) = alpha / t13 * (z - center) / (0.1e1 + t17 * t19);
}

double layer3_laplace(const Vector &p)
{
   double x = p(0), y = p(1), z = p(2);

   double t1 = x * x;
   double t4 = center * center;
   double t6 = y * y;
   double t9 = z * z;
   double t12 = t1 - 0.2e1 * x * center + 0.3e1 * t4
          + t6 - 0.2e1 * y * center + t9 - 0.2e1 * z * center;
   double t13 = sqrt(t12);
   double t16 = alpha / t13 / t12;
   double t18 = 0.4e1 * pow(x - center, 0.2e1);
   double t19 = alpha * alpha;
   double t20 = t13 - radius;
   double t21 = t20 * t20;
   double t23 = 0.1e1 + t19 * t21;
   double t24 = 0.1e1 / t23;
   double t34 = alpha * t19 / t12;
   double t35 = t23 * t23;
   double t36 = 0.1e1 / t35;
   double t42 = 0.4e1 * pow(y - center, 0.2e1);
   double t51 = 0.4e1 * pow(z - center, 0.2e1);
   double t59 = -t16 * t18 * t24 / 0.4e1
                + 0.3e1 * alpha / t13 * t24
                - t34 * t18 * t36 * t20 / 0.2e1
                - t16 * t42 * t24 / 0.4e1
                - t34 * t42 * t36 * t20 / 0.2e1
                - t16 * t51 * t24 / 0.4e1
                - t34 * t51 * t36 * t20 / 0.2e1;

   return -t59;
}


// hcurl

#if 0
const double kappa = M_PI;

void hcurl_exsol(const Vector &x, Vector &E)
{
   E(0) = sin(kappa * x(1));
   E(1) = sin(kappa * x(2));
   E(2) = sin(kappa * x(0));
}

void hcurl_exrhs(const Vector &x, Vector &f)
{
   f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
   f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
   f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
}
#else

void hcurl_exsol(const Vector &x, Vector &E)
{
   Vector a(x), b(x), c(x);
   a(0) = x(1); a(1) = x(2); // yz
   b(0) = x(0); b(1) = x(2); // xz

   E(0) = layer2_exsol(a);
   E(1) = 0;//layer2_exsol(b);
   E(2) = 0;//layer2_exsol(c);
}

void hcurl_exrhs(const Vector &x, Vector &f)
{
   Vector a(x), b(x), c(x);
   a(0) = x(1); a(1) = x(2); // yz
   b(0) = x(0); b(1) = x(2); // xz

   f(0) = layer2_laplace(a) + layer2_exsol(a);
   f(1) = 0;//layer2_laplace(b) + layer2_exsol(b);
   f(2) = 0;//layer2_laplace(c) + layer2_exsol(c);
}

#endif

// hdiv

void hdiv_exsol(const Vector &p, Vector &F)
{
   double x,y,z;
   int dim = p.Size();

   x = p(0);
   y = p(1);
   if (dim == 3) { z = p(2); }

   F(0) = cos(M_PI*x) * sin(M_PI*y);
   F(1) = cos(M_PI*y) * sin(M_PI*x);
   if (dim == 3) { F(2) = 0.0; }

   (void) z;
}

void hdiv_exrhs(const Vector &p, Vector &f)
{
   double x,y,z;
   int dim = p.Size();

   x = p(0);
   y = p(1);
   if (dim == 3) { z = p(2); }

   double temp = 1 + 2*M_PI*M_PI;

   f(0) = temp * cos(M_PI*x) * sin(M_PI*y);
   f(1) = temp * cos(M_PI*y) * sin(M_PI*x);
   if (dim == 3) { f(2) = 0; }

   (void) z;
}

} // namespace mfem
