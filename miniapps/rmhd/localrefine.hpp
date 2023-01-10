#include "mfem.hpp"

using namespace std;
using namespace mfem;

double yrefine=0.2;

bool region(const Vector &p, const int lev)
{
   const double region_eps = 1e-8;
   const double x = p(0), y = p(1);

   if(lev==0)
      return std::max(-y-yrefine, y - yrefine)<region_eps;
   else
   {
      double ynew=0.8*yrefine;
      double xcenter=0.2, xedge=0.9;
      return (fabs(y)<ynew+region_eps && (fabs(x)<xcenter+region_eps || fabs(x)>xedge-region_eps) );
   }
}

bool yregion(const Vector &x, const double y0)
{
   return std::max(-x(1)-y0, x(1) - y0)<1e-8;
}

bool xyregion(const Vector &x, const double x0, const double y0)
{
   return std::max(-x(1)-y0, x(1) - y0)<1e-8 && 
         (std::max(-x(0)-x0, x(0) - x0)<1e-8 || (1.-x0-x(0))<1e-8 || (-1+x0-x(0))>1e-8) ;
}

bool center_region(const Vector &x, const double x0, const double y0)
{
   return std::max(-x(1)-y0, x(1) - y0)<1e-8 && std::max(-x(0)-x0, x(0) - x0)<1e-8;
}
