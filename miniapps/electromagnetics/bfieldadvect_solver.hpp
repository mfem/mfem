// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../common/pfem_extras.hpp"
#include "../common/mesh_extras.hpp"

namespace mfem
{

using common::H1_ParFESpace;
using common::ND_ParFESpace;
using common::RT_ParFESpace;
using common::L2_ParFESpace;
using common::ParDiscreteGradOperator;
using common::ParDiscreteCurlOperator;
using common::DivergenceFreeProjector;

namespace electromagnetics
{

class BFieldAdvector
{
   public:
   BFieldAdvector(ParMesh *pmesh_old, ParMesh *pmesh_new, int order);
   void SetMeshes(ParMesh *pmesh_old, ParMesh *pmesh_new);
   void Advect(ParGridFunction* b_old, ParGridFunction* b_new);

   ParGridFunction* GetVectorPotential() {return a;}
   ParGridFunction* GetCurlB() {return curl_b;}
   ParGridFunction* GetCleanCurlB() {return clean_curl_b;}
   ParGridFunction* GetReconstructedB() {return recon_b;}

   private:
   void CleanInternals();
   void ComputeCleanCurlB(ParGridFunction* b);
   void ComputeA(ParGridFunction* b);

   int order;
   ParMesh *pmeshOld, *pmeshNew;         //The old/source mesh and new/target mesh
   H1_ParFESpace *H1FESpaceOld, *H1FESpaceNew;
   ND_ParFESpace *HCurlFESpaceOld, *HCurlFESpaceNew;
   RT_ParFESpace *HDivFESpaceOld, *HDivFESpaceNew;
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
};

} // namespace electromagnetics

} // namespace mfem