#include "mfem.hpp"

using namespace std;
using namespace mfem;

//this is an AMR update function for VSize (instead of TrueVSize)
//It is only called in the initial stage of AMR to generate an adaptive mesh
void AMRUpdate(BlockVector &S, BlockVector &S_tmp,
               Array<int> &offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j);

//this is an update function for block vector of TureVSize
void AMRUpdateTrue(BlockVector &S, 
               Array<int> &true_offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j,
               ParGridFunction *pre,
               ParGridFunction *dpsidt=NULL);

//this is an update function for block vector of TureVSize
void AMRUpdateTrue(BlockVector &S, 
               Array<int> &true_offset,
               ParGridFunction &phi,
               ParGridFunction &psi,
               ParGridFunction &w,
               ParGridFunction &j);

