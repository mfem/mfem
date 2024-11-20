// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TMOP_AMR_HPP
#define MFEM_TMOP_AMR_HPP

#include "tmop_tools.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"
#include "estimators.hpp"
#include "../mesh/mesh_operators.hpp"

namespace mfem
{

class TMOPRefinerEstimator : public AnisotropicErrorEstimator
{
protected:
   Mesh *mesh; // not owned
   NonlinearForm *nlf; // not owned
   int order;
   int amrmetric;
   Array<IntegrationRule *> TriIntRule, QuadIntRule, TetIntRule, HexIntRule;
   long current_sequence;
   Vector error_estimates;
   Array<int> aniso_flags;
   // An element is refined only if
   // [mean TMOPEnergy(children)]*energy_scaling_factor < TMOPEnergy(parent)
   real_t energy_scaling_factor;
   GridFunction *spat_gf;   // If specified, can be used to specify the
   real_t spat_gf_critical; // region where hr-adaptivity is done.

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = mesh->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates. For an element E in the mesh,
   /// error(E) = TMOPEnergy(E)*energy_scaling_factor-Mean(TMOPEnergy(ChildofE)),
   /// where TMOPEnergy of Children of E is obtained by assuming the element E
   /// is refined using the refinement type being considered based on the TMOP
   /// mesh quality metric.
   void ComputeEstimates();

   /// Construct the integration rules to model how each element type is split
   /// using different refinement types. ref_type = 0 is the original element
   /// and reftype \ in [1, 7] represent different refinement type based on
   /// NCMesh class.
   void SetQuadIntRules(); // supports ref_type = 1 to 3.
   void SetTriIntRules(); // currently supports only isotropic refinement.
   void SetHexIntRules(); // currently supports only isotropic refinement.
   void SetTetIntRules(); // currently supports only isotropic refinement.

   /// Get TMOP energy for each element corresponding to the refinement type
   /// specified.
   void GetTMOPRefinementEnergy(int reftype, Vector &el_energy_vec);

   /// Use a mesh to setup an integration rule that will mimic the different
   /// refinement types.
   IntegrationRule* SetIntRulesFromMesh(Mesh &meshsplit);
public:
   TMOPRefinerEstimator(Mesh &mesh_, NonlinearForm &nlf_, int order_,
                        int amrmetric_) :
      mesh(&mesh_), nlf(&nlf_), order(order_), amrmetric(amrmetric_),
      TriIntRule(0), QuadIntRule(0), TetIntRule(0), HexIntRule(0),
      current_sequence(-1), error_estimates(), aniso_flags(),
      energy_scaling_factor(1.), spat_gf(NULL), spat_gf_critical(0.)
   {
      if (mesh->Dimension() == 2)
      {
         SetQuadIntRules();
         SetTriIntRules();
      }
      else
      {
         SetHexIntRules();
         SetTetIntRules();
      }
   }

   ~TMOPRefinerEstimator()
   {
      for (int i = 0; i < QuadIntRule.Size(); i++) { delete QuadIntRule[i]; }
      for (int i = 0; i < TriIntRule.Size();  i++) { delete TriIntRule[i]; }
      for (int i = 0; i < HexIntRule.Size();  i++) { delete HexIntRule[i]; }
      for (int i = 0; i < TetIntRule.Size();  i++) { delete TetIntRule[i]; }
   }

   /// Get TMOP-based errors for each element in the mesh computed based on the
   /// refinement types being considered.
   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }
   /// For anisotropic refinements, get the refinement type (e.g., x or y)
   const Array<int> &GetAnisotropicFlags() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Scaling factor for the TMOP refinement energy. An element is refined if
   /// [mean TMOPEnergy(children)]*energy_scaling_factor < TMOPEnergy(parent)
   void SetEnergyScalingFactor(real_t scale) { energy_scaling_factor = scale; }

   /// Spatial indicator function (eta) that can be used to prevent elements
   /// from being refined even if the energy criterion is met. Using this,
   /// an element E is not refined if mean(@a spat_gf(E)) < @a spat_gf_critical.
   void SetSpatialIndicator(GridFunction &spat_gf_,
                            real_t spat_gf_critical_ = 0.5)
   { spat_gf = &spat_gf_; spat_gf_critical = spat_gf_critical_; }
   void SetSpatialIndicatorCritical(real_t val_) { spat_gf_critical = val_; }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }
};

class TMOPDeRefinerEstimator : public ErrorEstimator
{
protected:
   Mesh *mesh;
   NonlinearForm *nlf;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh;
   ParNonlinearForm *pnlf;
#endif
   int order;
   int amrmetric;
   long current_sequence;
   Vector error_estimates;
   bool serial;

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = mesh->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates. For a given element E in the mesh,
   /// error(E) = TMOPEnergy(parent_of_E)-TMOPEnergy(E). Children element of an
   /// element are derefined if the mean TMOP energy of children is greater than
   /// the TMOP energy associated with their parent.
   void ComputeEstimates();

   void GetTMOPDerefinementEnergy(Mesh &cmesh,
                                  TMOP_Integrator &tmopi,
                                  Vector &el_energy_vec);

   bool GetDerefineEnergyForIntegrator(TMOP_Integrator &tmopi,
                                       Vector &fine_energy);
public:
   TMOPDeRefinerEstimator(Mesh &mesh_, NonlinearForm &nlf_) :
      mesh(&mesh_), nlf(&nlf_),
      current_sequence(-1), error_estimates(), serial(true)   { }
#ifdef MFEM_USE_MPI
   TMOPDeRefinerEstimator(ParMesh &pmesh_, ParNonlinearForm &pnlf_) :
      mesh(&pmesh_), nlf(&pnlf_), pmesh(&pmesh_), pnlf(&pnlf_),
      current_sequence(-1), error_estimates(), serial(false) { }
#endif

   ~TMOPDeRefinerEstimator() { }

   const Vector &GetLocalErrors() override
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   void Reset() override { current_sequence = -1; }
};

// hr-adaptivity using TMOP.
// If hr-adaptivity is disabled, r-adaptivity is done once using the
// TMOPNewtonSolver.
// Otherwise, "hr_iter" iterations of r-adaptivity are done followed by
// "h_per_r_iter" iterations of h-adaptivity after each r-adaptivity iteration.
// The solver terminates early if an h-adaptivity iteration does not
// refine/derefine any element in the mesh.
class TMOPHRSolver
{
protected:
   Mesh *mesh;
   NonlinearForm *nlf;
   TMOPNewtonSolver *tmopns;
   GridFunction *x;
   Array<GridFunction *> gridfuncarr;
   Array<FiniteElementSpace *> fespacearr;
   bool move_bnd, hradaptivity;
   const int mesh_poly_deg, amr_metric_id;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh;
   ParNonlinearForm *pnlf;
   Array<ParGridFunction *> pgridfuncarr;
   Array<ParFiniteElementSpace *> pfespacearr;
#endif
   bool serial;

   // All are owned.
   TMOPRefinerEstimator *tmop_r_est;
   ThresholdRefiner *tmop_r;
   TMOPDeRefinerEstimator *tmop_dr_est;
   ThresholdDerefiner *tmop_dr;

   int hr_iter, h_per_r_iter;

   void Update();
#ifdef MFEM_USE_MPI
   void ParUpdate();
#endif
   void UpdateNonlinearFormAndBC(Mesh *mesh, NonlinearForm *nlf);

#ifdef MFEM_USE_MPI
   // Rebalance ParMesh such that all the children elements are moved to the same
   // MPI rank where the parent will be if the mesh were to be derefined.
   void RebalanceParNCMesh();
#endif

public:
   TMOPHRSolver(Mesh &mesh_, NonlinearForm &nlf_,
                TMOPNewtonSolver &tmopns_, GridFunction &x_,
                bool move_bnd_, bool hradaptivity_,
                int mesh_poly_deg_, int amr_metric_id_,
                int hr_iter_ = 5, int h_per_r_iter_ = 1);
#ifdef MFEM_USE_MPI
   TMOPHRSolver(ParMesh &pmesh_, ParNonlinearForm &pnlf_,
                TMOPNewtonSolver &tmopns_, ParGridFunction &x_,
                bool move_bnd_, bool hradaptivity_,
                int mesh_poly_deg_, int amr_metric_id_,
                int hr_iter_ = 5, int h_per_r_iter_ = 1);
#endif

   void Mult();

   /// These are used to update spaces and functions that are not owned by the
   /// TMOPIntegrator or DiscreteAdaptTC. The owned ones are updated in the
   /// functions UpdateAfterMeshTopologyChange() of both classes.
   void AddGridFunctionForUpdate(GridFunction *gf) { gridfuncarr.Append(gf); }
   void AddFESpaceForUpdate(FiniteElementSpace *fes) { fespacearr.Append(fes); }

#ifdef MFEM_USE_MPI
   void AddGridFunctionForUpdate(ParGridFunction *pgf_)
   {
      pgridfuncarr.Append(pgf_);
   }
   void AddFESpaceForUpdate(ParFiniteElementSpace *pfes_)
   {
      pfespacearr.Append(pfes_);
   }
#endif

   ~TMOPHRSolver()
   {
      if (!hradaptivity) { return; }
      delete tmop_dr;
      delete tmop_dr_est;
      delete tmop_r;
      delete tmop_r_est;
   }

   /// Total number of hr-adaptivity iterations. At each iteration, we do an
   /// r-adaptivity iteration followed by a number of h-adaptivity iterations.
   void SetHRAdaptivityIterations(int iter) { hr_iter = iter; }

   /// Total number of h-adaptivity iterations per r-adaptivity iteration.
   void SetHAdaptivityIterations(int iter) { h_per_r_iter = iter; }
};

}
#endif
