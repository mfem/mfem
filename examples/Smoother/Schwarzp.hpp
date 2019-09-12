#include "mfem.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


struct par_patch_nod_info 
{
   int mynrpatch;
   vector<Array<int>> vert_contr;
   vector<Array<int>> edge_contr;
   vector<Array<int>> face_contr;
   vector<Array<int>> elem_contr;
   // constructor
   par_patch_nod_info(ParMesh * cpmesh_, int ref_levels_);
private:
   int ref_levels=0;;
   ParMesh pmesh;
};


bool its_a_patch(int iv, Array<int> patch_ids);
