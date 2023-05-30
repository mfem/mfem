#include "hyperbolic_conservation_laws.hpp"

namespace mfem
{
void FluxReconErrorEstimator::ComputeEstimates()
{
   // FiniteElementSpace* fespace = solution.FESpace();
   // Mesh* mesh = fespace->GetMesh();
   // const int sdim = mesh->Dimension();
   // const int vdim = fespace->GetVDim();
   // const int numElem = mesh->GetNE();
   // const int numFace = mesh->GetNumFaces();
   // const int base_order = fespace->GetElementOrder(0);
   // FiniteElementSpace RT(mesh, RTfec);
   // for (int i=0; i<numElem; i++)
   // {
   //    RT.SetElementOrder(i, fespace->GetElementOrder(i));
   // }
   // RT.Update(false);

   // // Create gridfunction for each row of the flux
   // Array<GridFunction*> fluxes(vdim);
   // for (auto &flux:fluxes)
   // {
   //    flux = new GridFunction(&RT);
   // }

   // Array<int> dofs;
   // DenseMatrix flux_val;
   // for (int i=0; i< numFace; i++)
   // {
   //    FiniteElement* fe = RT->GetFE(i);
      

   // }
   // delete fluxes;
   mfem_error("WIP");
}
}