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

   ParGridFunction px_min, px_max;

   ParDofInfo(ParFiniteElementSpace *pfes_sltn,
              ParFiniteElementSpace *pfes_bounds);

   ~ParDofInfo() { }

   // Computes the admissible interval of values for each DG dof from the
   // values of all elements that feature the dof at its physical location.
   // Assumes that xe_min and xe_max are already computed.
   void ComputeBounds();

   // NOTE: This approach will not work for meshes with hanging nodes.
   void FillNeighborDofs();
};

#endif
