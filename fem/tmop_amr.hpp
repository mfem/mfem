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

#ifndef MFEM_TMOP_AMR_HPP
#define MFEM_TMOP_AMR_HPP

#include "tmop_tools.hpp"
#include "nonlinearform.hpp"
#include "pnonlinearform.hpp"
#include "estimators.hpp"
#include "../mesh/mesh_operators.hpp"
#include <fstream>
#include <iostream>

namespace mfem
{

class TMOPRefinerEstimator : public AnisotropicErrorEstimator
{
protected:
   Mesh *mesh; //not-owned
   NonlinearForm *nlf; //not-owned
   int order;
   int amrmetric;
   Array<IntegrationRule *> TriIntRule, QuadIntRule, TetIntRule, HexIntRule;
   long current_sequence;
   Vector error_estimates;
   Array<int> aniso_flags;
   double energy_scaling_factor; // an element is refined only if
   // [mean E(children)]*factor < E(parent)
   GridFunction *spat_gf;          // If specified, can be used to specify the
   double spat_gf_critical;        // the region where hr-adaptivity is done.

   /// Check if the mesh of the solution was modified.
   bool MeshIsModified()
   {
      long mesh_sequence = mesh->GetSequence();
      MFEM_ASSERT(mesh_sequence >= current_sequence, "");
      return (mesh_sequence > current_sequence);
   }

   /// Compute the element error estimates.
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

   /// Destructor
   ~TMOPRefinerEstimator()
   {
      for (int i = 0; i < QuadIntRule.Size(); i++) { delete QuadIntRule[i]; }
      for (int i = 0; i < TriIntRule.Size();  i++) { delete TriIntRule[i]; }
      for (int i = 0; i < HexIntRule.Size();  i++) { delete HexIntRule[i]; }
      for (int i = 0; i < TetIntRule.Size();  i++) { delete TetIntRule[i]; }
   }

   /// Get TMOP-based errors for each element in the mesh computed based on the
   /// refinement types being considered.
   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }
   /// For anisotropic refinements, get the refinement type (e.g., x or y)
   virtual const Array<int> &GetAnisotropicFlags()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return aniso_flags;
   }

   /// Scaling factor for the TMOP refinement energy. Used to tighten the refinement
   /// criterion. An element is refined only if
   /// [mean E(children)]*energy_refuction_factor < E(parent)
   void SetEnergyScalingFactor(double factor_) { energy_scaling_factor = factor_; }

   /// Used to set a space-dependent function that can make elements from being
   /// refined.
   void SetSpatialIndicator(GridFunction &spat_gf_) { spat_gf = &spat_gf_; }
   void SetSpatialIndicatorCritical(double val_) { spat_gf_critical = val_; }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }
};



// TMOPRefiner is ThresholdRefiner with total_error_fraction = 0.;
class TMOPRefiner : public ThresholdRefiner
{
public:
   /// Construct a ThresholdRefiner using the given ErrorEstimator.
   TMOPRefiner(TMOPRefinerEstimator &est) : ThresholdRefiner(est)
   {
      SetTotalErrorFraction(0.);
   }
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

   /// Compute the element error estimates.
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

   // destructor
   ~TMOPDeRefinerEstimator()
   {
   }

   virtual const Vector &GetLocalErrors()
   {
      if (MeshIsModified()) { ComputeEstimates(); }
      return error_estimates;
   }

   /// Reset the error estimator.
   virtual void Reset() { current_sequence = -1; }
};


class TMOPAMRSolver
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

   TMOPRefinerEstimator *tmop_r_est;
   TMOPRefiner *tmop_r;
   TMOPDeRefinerEstimator *tmop_dr_est;
   ThresholdDerefiner *tmop_dr;

   void Update();
#ifdef MFEM_USE_MPI
   void ParUpdate();
#endif

public:
   TMOPAMRSolver(Mesh &mesh_,
                 NonlinearForm &nlf_,
                 TMOPNewtonSolver &tmopns_,
                 GridFunction &x_,
                 bool move_bnd_,
                 bool hradaptivity_,
                 int mesh_poly_deg_,
                 int amr_metric_id_);
#ifdef MFEM_USE_MPI
   TMOPAMRSolver(ParMesh &pmesh_,
                 ParNonlinearForm &pnlf_,
                 TMOPNewtonSolver &tmopns_,
                 ParGridFunction &x_,
                 bool move_bnd_,
                 bool hradaptivity_,
                 int mesh_poly_deg_,
                 int amr_metric_id_);
#endif

   void Mult();

   void AddGridFunctionForUpdate(GridFunction *gf_)
   {
      gridfuncarr.Append(gf_);
   }
#ifdef MFEM_USE_MPI
   void AddGridFunctionForUpdate(ParGridFunction *pgf_)
   {

      pgridfuncarr.Append(pgf_);
   }
#endif
   void AddFESpaceForUpdate(FiniteElementSpace *fes_)
   {

      fespacearr.Append(fes_);
   }
#ifdef MFEM_USE_MPI
   void AddFESpaceForUpdate(ParFiniteElementSpace *pfes_)
   {
      pfespacearr.Append(pfes_);
   }
#endif

#ifdef MFEM_USE_MPI
   // Rebalance ParMesh such that all the children elements are moved to the same
   // MPI rank where the parent will be if the mesh were to be derefined.
   void RebalanceParNCMesh();
#endif

   ~TMOPAMRSolver()
   {
      if (!hradaptivity) { return; }
      delete tmop_dr;
      delete tmop_dr_est;
      delete tmop_r;
      delete tmop_r_est;
   }
};

}
#endif
