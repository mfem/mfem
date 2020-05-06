#ifndef HYPSYS_PDOFS
#define HYPSYS_PDOFS

#include "../../../mfem.hpp"
#include "dofs.hpp"

using namespace std;
using namespace mfem;

class ParDofInfo : public DofInfo
{
public:
   ParMesh *pmesh;
   ParFiniteElementSpace *pfes;

   ParDofInfo(ParFiniteElementSpace *pfes_);

   ~ParDofInfo() { }

   // NOTE: This approach will not work for meshes with hanging h- or p-nodes.
   void FillNeighborDofs();
};

#endif
