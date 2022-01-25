#include "../common/pfem_extras.hpp"
#include "../common/mesh_extras.hpp"

using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;
using common::DivergenceFreeProjector;

class BFieldAdvector
{
   public:
   BFieldAdvector(ParMesh *pmesh_old, ParMesh *pmesh_new, int order);
   void SetMeshes(ParMesh *pmesh_old, ParMesh *pmesh_new);
   void Advect(ParGridFunction* b_old, ParGridFunction* b_new);

   ParGridFunction* GetVectorPotential() {return a;}
   ParGridFunction* GetReconstructedB() {return recon_b;}

   private:
   void CleanInternals();
   void ComputeCleanCurlB(ParGridFunction* b);
   void ComputeA(ParGridFunction* b);

   int order;
   ParMesh *pmeshOld, *pmeshNew;         //The old/source mesh and new/target mesh
   H1_ParFESpace *H1FESpaceOld, *H1FESpaceNew;
   ND_ParFESpace *NDFESpaceOld, *NDFESpaceNew;
   RT_ParFESpace *RTFESpaceOld, *RTFESpaceNew;
   L2_ParFESpace *L2FESpaceOld, *L2FESpaceNew;
   ParDiscreteGradOperator *grad;
   ParDiscreteCurlOperator *curl;
   ParMixedBilinearForm *weakCurl;
   ParBilinearForm *curlCurl;
   DivergenceFreeProjector *divFreeProj;
   ParGridFunction *a;
   ParGridFunction *curl_b;
   ParGridFunction *clean_curl_b;
   ParGridFunction *recon_b;
}