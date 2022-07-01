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
#include "bfield_remhos.hpp"

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
   void SetMesh(ParMesh *pmesh_old, ParMesh *pmesh_new);
   void SetMeshNodes(ParGridFunction *old_nodes, ParGridFunction *new_nodes);
   void Advect(ParGridFunction* b_old, ParGridFunction* b_new);

   ParGridFunction* GetVectorPotential() {return a;}
   ParGridFunction* GetCurlB() {return curl_b;}
   ParGridFunction* GetCleanCurlB() {return clean_curl_b;}
   ParGridFunction* GetReconstructedB() {return recon_b;}
   ParGridFunction* GetA() {return a;}
   ParGridFunction* GetANew() {return a_new;}

   private:
   void CleanInternals();
   void ComputeCleanCurlB(ParGridFunction* b);
   void ComputeA(ParGridFunction* b);

   /// Given a grid function on the old and new meshes interpolate a field from the old to the new
   /** The fieldtype variable goes from 0 forms to 3 forms in order 0-H1, 1-H(curl), 2-H(div), 3-L2
   **/
   void FindPtsInterpolateToTargetMesh(const ParGridFunction *old_gf, ParGridFunction *new_gf, int fieldtype);
   void RemhosRemap(ParGridFunction* b_old, ParGridFunction* b_new);

   int order;
   MPI_Comm myComm;
   ParMesh *pmeshOld, *pmeshNew;
   H1_ParFESpace *H1FESpaceOld, *H1FESpaceNew;
   ND_ParFESpace *HCurlFESpaceOld, *HCurlFESpaceNew;
   RT_ParFESpace *HDivFESpaceOld, *HDivFESpaceNew;
   L2_ParFESpace *L2FESpaceOld, *L2FESpaceNew;
   ParDiscreteGradOperator *grad;
   ParDiscreteCurlOperator *curl_old;
   ParDiscreteCurlOperator *curl_new;
   ParMixedBilinearForm *weakCurl;
   HypreParMatrix *WC;
   ParBilinearForm *m1;
   ParBilinearForm *curlCurl;
   DivergenceFreeProjector *divFreeProj;
   ParGridFunction *a;
   ParGridFunction *a_new;
   ParGridFunction *curl_b;
   ParGridFunction *clean_curl_b;
   ParGridFunction *recon_b;
};


class AdvectionOperator : public TimeDependentOperator
{
private:
   BilinearForm &Mbf, &ml;
   ParBilinearForm &Kbf;
   ParBilinearForm &M_HO, &K_HO;
   Vector &lumpedM;

   Vector start_mesh_pos, start_submesh_pos;
   GridFunction &mesh_pos, *submesh_pos, &mesh_vel, &submesh_vel;

   mutable ParGridFunction x_gf;

   double dt;
   mutable double dt_est;
   Assembly &asmbl;

   LowOrderMethod &lom;
   DofInfo &dofs;

   HOSolver *ho_solver;
   LOSolver *lo_solver;
   FCTSolver *fct_solver;

public:
   AdvectionOperator(int size, BilinearForm &Mbf_, BilinearForm &_ml,
                     Vector &_lumpedM,
                     ParBilinearForm &Kbf_,
                     ParBilinearForm &M_HO_, ParBilinearForm &K_HO_,
                     GridFunction &pos, GridFunction *sub_pos,
                     GridFunction &vel, GridFunction &sub_vel,
                     Assembly &_asmbl, LowOrderMethod &_lom, DofInfo &_dofs,
                     HOSolver *hos, LOSolver *los, FCTSolver *fct);

   virtual void Mult(const Vector &x, Vector &y) const;

   void SetDt(double dt_) { dt = dt_; dt_est = dt; }
   double GetTimeStepEstimate() { return dt_est; }

   void SetRemapStartPos(const Vector &m_pos, const Vector &sm_pos)
   {
      start_mesh_pos    = m_pos;
      start_submesh_pos = sm_pos;
   }

   virtual ~AdvectionOperator() { }
};

} // namespace electromagnetics

} // namespace mfem