

#include "mfem.hpp"
#include "BCData.hpp"

namespace mfem
{

BCData::BCData()
{
   // TODO constructor stub
}

BCData::~BCData()
{
   // TODO destructor stub
}


void BCData::setDirBCs(const Vector& x, double t, Vector& y)
{ 
   // this routine applies a Dirichlet BC displacement INCREMENT. The 
   // values contained in essDisp are TOTAL displacement. The slope of the 
   // linear time function is essDisp[i]/t_final (tf). Slope*dt is the 
   // increment in displacement. The scale[i] factor is used to apply zero 
   // displacement increment for homogeneous Dirichlet BCs.
   double fac = 1.0 / tf;
   y = 0.; 
   y[0] = fac * essDisp[0] * scale[0] * dt; 
   y[1] = fac * essDisp[1] * scale[1] * dt;
   y[2] = fac * essDisp[2] * scale[2] * dt; 
//   printf("BCData dt: %f \n", dt);
//     printf("BCData y(0,1,2) %f %f %f \n", y[0], y[1], y[2]);
//   printf("BCData essDisp: %f %f %f \n", essDisp[0], essDisp[1], essDisp[2]);
}

void BCData::setScales()
{
   switch (compID)
   {
      case -1 : scale[0] = 1.0;
                scale[1] = 1.0;
                scale[2] = 1.0;
                break;
      case  1 : scale[0] = 1.0;
                scale[1] = 0.0;
                scale[2] = 0.0;
                break;
      case  2 : scale[0] = 0.0;
                scale[1] = 1.0;
                scale[2] = 0.0;
                break;
      case  3 : scale[0] = 0.0;
                scale[1] = 0.0;
                scale[2] = 1.0;
                break;
      case  4 : scale[0] = 1.0;
                scale[1] = 1.0;
                scale[2] = 0.0;
                break;
      case  5 : scale[0] = 0.0;
                scale[1] = 1.0;
                scale[2] = 1.0;
                break;
      case  6 : scale[0] = 1.0;
                scale[1] = 0.0;
                scale[2] = 1.0;
                break;
      case  7 : scale[0] = 0.0;
                scale[1] = 0.0;
                scale[2] = 0.0;
                break;
   } 
}

void BCData::getComponents(int id, Array<int> &component)
{
   switch (id)
   {
      case -1 : component[0] = 0;
                component[1] = 1;
                component[2] = 2;
                break;
      case  1 : component[0] = 0;
                component[1] = -1;
                component[2] = -1;
                break;
      case  2 : component[0] = -1;
                component[1] = 1;
                component[2] = -1;
                break;
      case  3 : component[0] = -1;
                component[1] = -1;
                component[2] = 2;
                break;
      case  4 : component[0] = 0;
                component[1] = 1;
                component[2] = -1;
                break;
      case  5 : component[0] = -1;
                component[1] = 1;
                component[2] = 2;
                break;
      case  6 : component[0] = 0;
                component[1] = -1;
                component[2] = 2;
                break;
      case  7 : component[0] = -1;
                component[1] = -1;
                component[2] = -1;
                break;
   } 
}


}
