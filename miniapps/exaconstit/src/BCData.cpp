

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
   //When doing the velocity based methods we only
   //need to do the below.
   y = 0.0;
   y[0] = essDisp[0] * scale[0];
   y[1] = essDisp[1] * scale[1];
   y[2] = essDisp[2] * scale[2];
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
