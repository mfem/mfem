
#ifndef BCDATA
#define BCDATA

#include "mfem.hpp"
#include "../../linalg/vector.hpp"
#include <fstream>

namespace mfem
{

class BCData
{
public:
   BCData();
   ~BCData();

   // scales for nonzero Dirichlet BCs
   double essDisp[3];
   double scale[3];
   int compID;
   double dt, tf;

   void setDirBCs(Vector& y);
   
   void setScales();
  
   static void getComponents(int id, Array<int> &component);
};
}
#endif
