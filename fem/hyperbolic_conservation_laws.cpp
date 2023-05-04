#include "hyperbolic_conservation_laws.hpp"

namespace mfem
{
void FluxReconErrorEstimator::ComputeEstimates()
{
   FiniteElementSpace* fespace = solution.FESpace();
   Mesh* mesh = fespace->GetMesh();
   const int sdim = mesh->Dimension();
   const int vdim = fespace->GetVDim();
   const int numElem = mesh->GetNE();
   const int base_order = fespace->GetElementOrder(0);
   RT_FECollection *RTfec = new RT_FECollection(base_order, sdim);
   FiniteElementSpace RT(mesh, RTfec);
   if (fespace->IsVariableOrder())
   {
      for (int i=0; i<numElem; i++)
      {
         RT.SetElementOrder(i, fespace->GetElementOrder(i));
      }
      RT.Update(false);
   }
   Array<GridFunction*> fluxes(vdim);
   for (auto &flux:fluxes)
   {
      flux = new GridFunction(&RT);
   }
   Array<int> dofs;
   DenseMatrix flux_val;
   for (int i=0; i< numElem; i++)
   {
      fespace->GetElementInteriorDofs(i, dofs);
      auto fe = fespace->GetFE(i);

   }
   mfem_error("WIP");
}
}