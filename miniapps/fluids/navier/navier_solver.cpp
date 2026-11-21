// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "navier_solver.hpp"
#include "kernels/shear_stress_integrator.hpp"
#include <iomanip>

using namespace mfem;
using namespace navier;

NavierSolver::NavierSolver(ParMesh *mesh, int order, real_t kin_vis,
                           const real_t density, bool verbose_, bool variable_viscosity_)
   : pmesh(mesh), order(order), kin_vis(kin_vis), density(density),
     gll_rules(0, Quadrature1D::GaussLobatto), verbose(verbose_), variable_viscosity(variable_viscosity_)
{
   // Fully periodic meshes have no boundary attributes.
   max_bdr_attributes = pmesh->bdr_attributes.Size() > 0 ?
                        pmesh->bdr_attributes.Max() : 0;

   vfec = new H1_FECollection(order, pmesh->Dimension());
   pfec = new H1_FECollection(order);
   vfes = new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
   pfes = new ParFiniteElementSpace(pmesh, pfec);

   gll_ir = gll_rules.Get(mesh->GetTypicalElementGeometry(), 2 * order - 1);
   gll_ir_face = gll_rules.Get(mesh->GetTypicalFaceGeometry(), 2 * order - 1);
   gll_ir_nl = gll_rules.Get(mesh->GetTypicalElementGeometry(),
                             (int)(ceil(1.5 * (2 * order - 1))));

   // Check if fully periodic mesh
   if (!(pmesh->bdr_attributes.Size() == 0))
   {
      vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr = 0;

      vel_ess_attr_x.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr_x = 0;
      vel_ess_attr_y.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr_y = 0;
      vel_ess_attr_z.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr_z = 0;

      pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      pres_ess_attr = 0;

      robin_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      robin_ess_attr = 0;

      amp_robin_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      amp_robin_ess_attr = 0;

      neumann_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      neumann_ess_attr = 0;
   }

   int vfes_truevsize = vfes->GetTrueVSize();
   int pfes_truevsize = pfes->GetTrueVSize();

   un.SetSize(vfes_truevsize);
   un = 0.0;
   un_next.SetSize(vfes_truevsize);
   un_next = 0.0;
   unm1.SetSize(vfes_truevsize);
   unm1 = 0.0;
   unm2.SetSize(vfes_truevsize);
   unm2 = 0.0;
   fn.SetSize(vfes_truevsize);
   Nun.SetSize(vfes_truevsize);
   Nun = 0.0;
   Nunm1.SetSize(vfes_truevsize);
   Nunm1 = 0.0;
   Nunm2.SetSize(vfes_truevsize);
   Nunm2 = 0.0;
   u_ext.SetSize(vfes_truevsize);
   Fext.SetSize(vfes_truevsize);
   FText.SetSize(vfes_truevsize);
   Lext.SetSize(vfes_truevsize);
   resu.SetSize(vfes_truevsize);

   tmp1.SetSize(vfes_truevsize);

   pn.SetSize(pfes_truevsize);
   pn = 0.0;
   resp.SetSize(pfes_truevsize);
   resp = 0.0;
   FText_bdr.SetSize(pfes_truevsize);
   g_bdr.SetSize(pfes_truevsize);

   un_gf.SetSpace(vfes);
   un_gf = 0.0;
   un_next_gf.SetSpace(vfes);
   un_next_gf = 0.0;
   u_ext_gf.SetSpace(vfes);
   u_ext_gf = 0.0;

   Lext_gf.SetSpace(vfes);
   curlu_gf.SetSpace(vfes);
   curlcurlu_gf.SetSpace(vfes);
   FText_gf.SetSpace(vfes);
   resu_gf.SetSpace(vfes);

   pn_gf.SetSpace(pfes);
   pn_gf = 0.0;
   resp_gf.SetSpace(pfes);

   kin_vis_gf.SetSpace(pfes);
   ConstantCoefficient kvcoeff_tmp(kin_vis);
   kin_vis_gf.ProjectCoefficient(kvcoeff_tmp);
   kv.SetSize(pfes_truevsize);
   grad_nu_sym_grad_uext.SetSize(vfes_truevsize);

   hpfrt_tdofs.SetSize(vfes_truevsize);
   hpfrt_tdofs = 0.0;

   cur_step = 0;

   if (verbose)
      PrintInfo();
}


void NavierSolver::SetMeshVelocity(ParGridFunction *mesh_velocity_gf, bool own)
{
   wg_gf = mesh_velocity_gf;
   wg_coef = new VectorGridFunctionCoefficient(wg_gf);

   own_mesh_velocity = own;
}

void NavierSolver::Setup(real_t dt)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
   }

   sw_setup.Start();

   ///<---- Get the essential true dofs for velocity and pressure.
   vfes->GetEssentialTrueDofs(vel_ess_attr_x,vel_ess_tdof_x,0);
   vfes->GetEssentialTrueDofs(vel_ess_attr_y,vel_ess_tdof_y,1);
   vfes->GetEssentialTrueDofs(vel_ess_attr_z,vel_ess_tdof_z,2);
   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   vel_ess_tdof.Append(vel_ess_tdof_x);
   vel_ess_tdof.Append(vel_ess_tdof_y);
   vel_ess_tdof.Append(vel_ess_tdof_z);
   vel_ess_tdof.Sort();
   vel_ess_tdof.Unique();
   
   // For pressure get both Pressure-Dirichlet and Neumann BCs
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);
   for (const auto &neumann_dbc : neumann_dbcs)
   {
      Array<int> neumann_tdof;
      pfes->GetEssentialTrueDofs(neumann_dbc.attr, neumann_tdof);
      pres_ess_tdof.Append(neumann_tdof);
   }
   pres_ess_tdof.Sort();
   pres_ess_tdof.Unique();

   // Check if the pressure problem is well-posed (otherwise we need to use OrthoSolver and Orthogonalize the rhs).
   well_posed = !pres_dbcs.empty() || !robin_dbcs.empty() || !amp_robin_dbcs.empty() || !neumann_dbcs.empty();

   nlcoeff.constant = -1.0;
   N = new ParNonlinearForm(vfes);
   auto *nlc_nlfi = new VectorConvectionNLFIntegrator(nlcoeff, wg_coef);
   nlc_nlfi->SetIntRule(&gll_ir_nl);
   N->AddDomainIntegrator(nlc_nlfi);
   // N->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   //N->Setup();  // Setup done in the Assemble() method.

   // Velocity mass matrix
   Mv_form = new ParBilinearForm(vfes);
   auto *mv_blfi = new VectorMassIntegrator;
   mv_blfi->SetIntRule(&gll_ir);
   Mv_form->AddDomainIntegrator(mv_blfi);
   Mv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   // Pressure Laplacian matrix + Robin bcs
   Sp_form = new ParBilinearForm(pfes);
   auto *sp_blfi = new DiffusionIntegrator;
   sp_blfi->SetIntRule(&gll_ir);
   Sp_form->AddDomainIntegrator(sp_blfi);
   for (auto &robin_dbc : robin_dbcs)
   {
      auto *robin_mass_bdr_bnlfi = new MassIntegrator(*robin_dbc.rho_beta);
      robin_mass_bdr_bnlfi->SetIntRule(&gll_ir_face);
      Sp_form->AddBoundaryIntegrator(robin_mass_bdr_bnlfi, robin_dbc.attr);
   }
   for (auto &amp_robin_dbc : amp_robin_dbcs)
   {
      auto *robin_mass_bdr_bnlfi = new MassIntegrator(*amp_robin_dbc.rho_beta);
      robin_mass_bdr_bnlfi->SetIntRule(&gll_ir_face);
      Sp_form->AddBoundaryIntegrator(robin_mass_bdr_bnlfi, amp_robin_dbc.attr);
   }
   Sp_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   D_form = new ParMixedBilinearForm(vfes, pfes);
   auto *vd_mblfi = new VectorDivergenceIntegrator();
   vd_mblfi->SetIntRule(&gll_ir);
   D_form->AddDomainIntegrator(vd_mblfi);
   D_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   G_form = new ParMixedBilinearForm(pfes, vfes);
   auto *g_mblfi = new GradientIntegrator();
   g_mblfi->SetIntRule(&gll_ir);
   G_form->AddDomainIntegrator(g_mblfi);
   G_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   H_bdfcoeff.constant = 1.0 / dt;
   H_form = new ParBilinearForm(vfes);

   auto *hmv_blfi = new VectorMassIntegrator(H_bdfcoeff);
   hmv_blfi->SetIntRule(&gll_ir);

   BilinearFormIntegrator *hdv_blfi = nullptr;
   kin_vis_gf_coeff.SetGridFunction(&kin_vis_gf);
   if (variable_viscosity)
   {
      hdv_blfi = new ShearStressIntegrator(kin_vis_gf_coeff, gll_ir);
   }
   else
   {
      hdv_blfi = new VectorDiffusionIntegrator(kin_vis_gf_coeff);
   }
   hdv_blfi->SetIntRule(&gll_ir);
   H_form->AddDomainIntegrator(hmv_blfi);
   H_form->AddDomainIntegrator(hdv_blfi);
   for (auto &robin_dbc : robin_dbcs)
   {
      auto *robin_mass_bdr_bnlfi = new VectorMassIntegrator(*robin_dbc.alphaP_over_rho);   
      robin_mass_bdr_bnlfi->SetIntRule(&gll_ir_face);   // same rule as (alpha u_s, v) and the PPE Robin mass
      H_form->AddBoundaryIntegrator(robin_mass_bdr_bnlfi, robin_dbc.attr);
   }
   for (auto &amp_robin_dbc : amp_robin_dbcs)
   {
      auto *robin_tangential_mass_bdr_bnlfi = new VectorMassIntegrator(*amp_robin_dbc.tangent_outer_product); // alpha_s/rho (u \cdot t, w \cdot t)
      auto *robin_normal_mass_bdr_bnlfi = new VectorMassIntegrator(*amp_robin_dbc.normal_outer_product);      // alpha_p/rho (u \cdot n, w \cdot n)
      robin_tangential_mass_bdr_bnlfi->SetIntRule(&gll_ir_face);   // same rule as the AMP rhs integrators
      robin_normal_mass_bdr_bnlfi->SetIntRule(&gll_ir_face);
      H_form->AddBoundaryIntegrator(robin_tangential_mass_bdr_bnlfi, amp_robin_dbc.attr);
      H_form->AddBoundaryIntegrator(robin_normal_mass_bdr_bnlfi, amp_robin_dbc.attr);
   }
   // NOTE: for now we can't use partial assembly with Robin bcs since VectorMassIntegrator and the AMP tangential version have no AssemblePABoundary
   //H_form->SetAssemblyLevel(AssemblyLevel::PARTIAL); 

   HMv_form = new ParBilinearForm(vfes);
   auto *hmv_blfi2 = new VectorMassIntegrator(H_bdfcoeff);
   hmv_blfi2->SetIntRule(&gll_ir);
   HMv_form->AddDomainIntegrator(hmv_blfi2);
   HMv_form->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   // Pressure Poisson rhs terms
   FText_gfcoeff = new VectorGridFunctionCoefficient(&FText_gf);
   FText_bdr_form = new ParLinearForm(pfes);
   auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
   ftext_bnlfi->SetIntRule(&gll_ir_face);
   FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, vel_ess_attr);
   FText_bdr_form->UseFastAssembly(true);


   g_bdr_form = new ParLinearForm(pfes);
   for (auto &vel_dbc : vel_dbcs)
   {
      auto *gbdr_bnlfi = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
      gbdr_bnlfi->SetIntRule(&gll_ir_face);
      g_bdr_form->AddBoundaryIntegrator(gbdr_bnlfi, vel_dbc.attr);
   }
   g_bdr_form->UseFastAssembly(true);

   // NOTE: For now we can keep it like this, but one way to improve it is to project ShearStressNormalTractionCoefficient to a GridFunction,
   // And then use a GridFunctionCoefficient wherever we need it. 
   // This way we can avoid multiple evaluations of the same coefficient tau \cdot n, e.g. when we want to compute (tau \cdot n) \cdot n.
   // tau must be a dynamic stress (rho nu (grad u + grad u^T)) like p and g, which it is combined with
   // in the Robin PPE term (rho beta (n·tau)·n) and in the Neumann pressure value p = (n·tau)·n - g·n.
   dyn_vis_coeff = new ProductCoefficient(density, kin_vis_gf_coeff);
   shear_stress_normal_traction_coeff = new ShearStressNormalTractionCoefficient(&u_ext_gf, dyn_vis_coeff, variable_viscosity); // tau \cdot n

   // Set up derived pressure Dirichlet coefficients for Neumann BCs.
   // Done here because shear_stress_normal_traction_coeff must exist first.
   for (auto &neumann_dbc : neumann_dbcs)
   {
      neumann_dbc.SetCoefficients(shear_stress_normal_traction_coeff,
                                   pmesh->Dimension(),
                                   &u_ext_gf);
   }

   // Pressure p^{n+1} on traction-type boundaries (Neumann, Robin, AMP Robin), see the (p n, v) terms below.
   pn_bdr_coeff = new GridFunctionCoefficient(&pn_gf);

   // Portion of the Neumann BC applying to the Momentum equation
   g_vel_neumann_bdr_form = new ParLinearForm(vfes);
   for (auto &neumann_dbc : neumann_dbcs)
   {
      auto *g_neumann_bnlfi = new VectorBoundaryLFIntegrator(*neumann_dbc.coeff);
      g_neumann_bnlfi->SetIntRule(&gll_ir_face);
      g_vel_neumann_bdr_form->AddBoundaryIntegrator(g_neumann_bnlfi, neumann_dbc.attr);

      // The momentum uses the strong pressure gradient (grad p, v), whose natural BC is tau·n, not sigma·n.
      // sigma·n = g  <=>  tau·n = g + p n, so the traction data needs the (p n, v) term as well.
      auto *pn_neumann_bnlfi = new VectorBoundaryFluxLFIntegrator(*pn_bdr_coeff);    // (p, v·n)
      pn_neumann_bnlfi->SetIntRule(&gll_ir_face);
      g_vel_neumann_bdr_form->AddBoundaryIntegrator(pn_neumann_bnlfi, neumann_dbc.attr);
   }
   g_vel_neumann_bdr_form->UseFastAssembly(true);

   g_pres_robin_bdr_form = new ParLinearForm(pfes);        // u \cdot n
   t_pres_robin_bdr_form = new ParLinearForm(pfes);        // g \cdot n  = (sigma \cdot n) \cdot n
   tau_pres_robin_bdr_form = new ParLinearForm(pfes);      // (n \cdot tau ) \cdot n   
   g_vel_robin_bdr_form = new ParLinearForm(vfes);         // (alpha u_s + g + p n, v)
   
   g_vel_tangent_amp_robin_bdr_form = new ParLinearForm(vfes); 
   g_vel_norm_amp_robin_bdr_form = new ParLinearForm(vfes);    // ((alpha_p u_s + g)·n + p, v·n)

   for (auto &robin_dbc : robin_dbcs)
   {
      // Robin Pressure terms
      // u \cdot n term
      auto *g_robin_bdr_bnlfi = new BoundaryNormalLFIntegrator(*robin_dbc.GetCoeff(0)); 
      g_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      g_pres_robin_bdr_form->AddBoundaryIntegrator(g_robin_bdr_bnlfi, robin_dbc.attr); 

      // beta g \cdot n term
      auto *t_robin_bdr_bnlfi = new BoundaryNormalLFIntegrator(*robin_dbc.beta_g); 
      t_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      t_pres_robin_bdr_form->AddBoundaryIntegrator(t_robin_bdr_bnlfi, robin_dbc.attr);

      // tau \cdot n term
      robin_dbc.SetShearStressCoefficient(shear_stress_normal_traction_coeff);
      auto *tau_robin_bdr_bnlfi = new BoundaryNormalLFIntegrator(*robin_dbc.beta_tau);
      tau_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      tau_pres_robin_bdr_form->AddBoundaryIntegrator(tau_robin_bdr_bnlfi, robin_dbc.attr);

      // Robin Velocity terms
      // alpha u + sigma \cdot n = alpha u_s + g, with sigma = -p I + tau.
      // The momentum uses the strong pressure gradient (grad p, v), whose natural BC is tau·n, not sigma·n.
      // Hence the BC imposed by the velocity solve is  alpha u + tau·n = alpha u_s + g + p n.
      auto *g_vel_robin_bdr_bnlfi1 = new VectorBoundaryLFIntegrator(*robin_dbc.GetCoeff(1));  // (g, v), g = sigma_s \cdot n_f
      g_vel_robin_bdr_bnlfi1->SetIntRule(&gll_ir_face);
      g_vel_robin_bdr_form->AddBoundaryIntegrator(g_vel_robin_bdr_bnlfi1, robin_dbc.attr);

      auto *pn_vel_robin_bdr_bnlfi = new VectorBoundaryFluxLFIntegrator(*pn_bdr_coeff);      // (p, v·n)
      pn_vel_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      g_vel_robin_bdr_form->AddBoundaryIntegrator(pn_vel_robin_bdr_bnlfi, robin_dbc.attr);

      auto *g_vel_robin_bdr_bnlfi2 = new VectorBoundaryLFIntegrator(*robin_dbc.alphaP_u);  //   (alpha u_s, v) 
      g_vel_robin_bdr_bnlfi2->SetIntRule(&gll_ir_face);
      g_vel_robin_bdr_form->AddBoundaryIntegrator(g_vel_robin_bdr_bnlfi2, robin_dbc.attr);
   }
   for (auto &amp_robin_dbc : amp_robin_dbcs)
   {
      // Robin Pressure terms (same as the Robin BCs above)
      // u \cdot n term
      auto *g_robin_bdr_bnlfi = new BoundaryNormalLFIntegrator(*amp_robin_dbc.GetCoeff(0)); 
      g_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      g_pres_robin_bdr_form->AddBoundaryIntegrator(g_robin_bdr_bnlfi, amp_robin_dbc.attr); 

      // beta g \cdot n term
      auto *t_robin_bdr_bnlfi = new BoundaryNormalLFIntegrator(*amp_robin_dbc.beta_g); 
      t_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      t_pres_robin_bdr_form->AddBoundaryIntegrator(t_robin_bdr_bnlfi, amp_robin_dbc.attr);

      // tau \cdot n term
      amp_robin_dbc.SetShearStressCoefficient(shear_stress_normal_traction_coeff);
      auto *tau_robin_bdr_bnlfi = new BoundaryNormalLFIntegrator(*amp_robin_dbc.beta_tau);
      tau_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      tau_pres_robin_bdr_form->AddBoundaryIntegrator(tau_robin_bdr_bnlfi, amp_robin_dbc.attr);

      // Robin Velocity terms.   NOTE: this is NOT the same as the Robin BCs above!
      // Tangential AMP Robin BCs
      auto *g_vel_tangent_robin_bdr_bnlfi = new VectorBoundaryTangentialAMPLFIntegrator(amp_robin_dbc.GetCoeff(1), amp_robin_dbc.GetCoeff(0), amp_robin_dbc.alphaS);
      g_vel_tangent_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      g_vel_tangent_amp_robin_bdr_form->AddBoundaryIntegrator(g_vel_tangent_robin_bdr_bnlfi, amp_robin_dbc.attr);

      // Normal AMP Robin BCs
      auto *g_vel_norm_robin_bdr_bnlfi = new VectorBoundaryNormalAMPLFIntegrator(amp_robin_dbc.GetCoeff(1), amp_robin_dbc.GetCoeff(0), amp_robin_dbc.alphaP);
      g_vel_norm_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      g_vel_norm_amp_robin_bdr_form->AddBoundaryIntegrator(g_vel_norm_robin_bdr_bnlfi, amp_robin_dbc.attr);

      // Strong pressure gradient: the normal condition needs (p, v·n) too (the tangential one has no pressure).
      auto *pn_vel_norm_robin_bdr_bnlfi = new VectorBoundaryFluxLFIntegrator(*pn_bdr_coeff);  // (p, v·n)
      pn_vel_norm_robin_bdr_bnlfi->SetIntRule(&gll_ir_face);
      g_vel_norm_amp_robin_bdr_form->AddBoundaryIntegrator(pn_vel_norm_robin_bdr_bnlfi, amp_robin_dbc.attr);
   }
   g_pres_robin_bdr_form->UseFastAssembly(true);
   t_pres_robin_bdr_form->UseFastAssembly(true);
   tau_pres_robin_bdr_form->UseFastAssembly(false);        // Disable it for now as it fails with fast assembly
   g_vel_robin_bdr_form->UseFastAssembly(true);
   g_vel_tangent_amp_robin_bdr_form->UseFastAssembly(true); // This is doesn't support fast assembly, but we keep it for future development (either way use fast assembly is gonna be disabled inside)
   g_vel_norm_amp_robin_bdr_form->UseFastAssembly(true);

   if (!robin_dbcs.empty())
   {
      auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
      ftext_bnlfi->SetIntRule(&gll_ir_face);
      FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, robin_ess_attr);
   }
   if (!amp_robin_dbcs.empty())
   {
      auto *ftext_bnlfi = new BoundaryNormalLFIntegrator(*FText_gfcoeff);
      ftext_bnlfi->SetIntRule(&gll_ir_face);
      FText_bdr_form->AddBoundaryIntegrator(ftext_bnlfi, amp_robin_ess_attr);
   }

   // Acceleration terms
   f_form = new ParLinearForm(vfes);
   for (auto &accel_term : accel_terms)
   {
      auto *vdlfi = new VectorDomainLFIntegrator(*accel_term.coeff);
      // @TODO: This order should always be the same as the nonlinear forms one!
      // const IntegrationRule &ir = IntRules.Get(vfes->GetTypicalElementGeometry(),
      //                                          4 * order);
      // vdlfi->SetIntRule(&ir);
      vdlfi->SetIntRule(&gll_ir);
      f_form->AddDomainIntegrator(vdlfi);
   }
   f_form->UseFastAssembly(true);

   // Assemble the operators and setup solvers/preconditioners
   UpdateForms();
   Assemble(true);
   UpdateSolvers();
   full_update = false; // After the first call to Setup, we don't need to do a full update anymore. (Unless Update() is called)

   // If the initial condition was set, it has to be aligned with dependent
   // Vectors and GridFunctions
   un_gf.GetTrueDofs(un);
   un_next = un;
   un_next_gf.SetFromTrueDofs(un_next);

   // Set initial time step in the history array
   dthist[0] = dt;

   if (filter_method != FilterMethod::NONE)
   {
      vfec_filter = new H1_FECollection(order - filter_cutoff_modes,
                                        pmesh->Dimension());
      vfes_filter = new ParFiniteElementSpace(pmesh,
                                              vfec_filter,
                                              pmesh->Dimension());

      u_filter_basis_gf.SetSpace(vfes_filter);
      u_filter_basis_gf = 0.0;

      u_low_modes_gf.SetSpace(vfes);
      u_low_modes_gf = 0.0;

      hpfrt_gf.SetSpace(vfes);
      hpfrt_gf = 0.0;
   }

   sw_setup.Stop();
}

void NavierSolver::UpdateSpaces()
{
   // TODO: check because we also added other ParGridFunctions e.g. u_ext_gf
   constexpr int num_vfes_tvecs = 16;
   std::array<Vector *, num_vfes_tvecs> vfes_tvecs;
   vfes_tvecs[0] = &un;
   vfes_tvecs[1] = &un_next;
   vfes_tvecs[2] = &unm1;
   vfes_tvecs[3] = &unm2;
   vfes_tvecs[4] = &fn;
   vfes_tvecs[5] = &Nun;
   vfes_tvecs[6] = &Nunm1;
   vfes_tvecs[7] = &Nunm2;
   vfes_tvecs[8] = &u_ext;
   vfes_tvecs[9] = &Fext;
   vfes_tvecs[10] = &FText;
   vfes_tvecs[11] = &Lext;
   vfes_tvecs[12] = &resu;
   vfes_tvecs[13] = &tmp1;
   vfes_tvecs[14] = &grad_nu_sym_grad_uext;
   vfes_tvecs[15] = &hpfrt_tdofs;

   std::array<ParGridFunction, num_vfes_tvecs> vfes_gfs;
   for (size_t i = 0; i < vfes_gfs.size(); i++)
   {
      vfes_gfs[i].SetSpace(vfes);
      vfes_gfs[i].SetFromTrueDofs(*vfes_tvecs[i]);
   }

   vfes->Update();

   for (size_t i = 0; i < vfes_gfs.size(); i++)
   {
      vfes_gfs[i].Update();
      vfes_gfs[i].GetTrueDofs(*vfes_tvecs[i]);
   }

   constexpr int num_pfes_tvecs = 5;
   std::array<Vector *, num_pfes_tvecs> pfes_tvecs;
   pfes_tvecs[0] = &pn;
   pfes_tvecs[1] = &resp;
   pfes_tvecs[2] = &FText_bdr;
   pfes_tvecs[3] = &g_bdr;
   pfes_tvecs[4] = &kv;

   std::array<ParGridFunction, num_pfes_tvecs> pfes_gfs;
   for (size_t i = 0; i < pfes_gfs.size(); i++)
   {
      pfes_gfs[i].SetSpace(pfes);
      pfes_gfs[i].SetFromTrueDofs(*pfes_tvecs[i]);
   }

   pfes->Update();

   for (size_t i = 0; i < pfes_gfs.size(); i++)
   {
      pfes_gfs[i].Update();
      pfes_gfs[i].GetTrueDofs(*pfes_tvecs[i]);
   }

   un_gf.Update();
   un_next_gf.Update();
   Lext_gf.Update();
   curlu_gf.Update();
   curlcurlu_gf.Update();
   FText_gf.Update();
   resu_gf.Update();
   pn_gf.Update();
   resp_gf.Update();
   kin_vis_gf.Update();
}

void NavierSolver::UpdateForms(bool full_update)
{
   // NOTE: in the FSI problem we're calling UpdateForms after the mesh is deformed.
   // It might not be ideal to assemble in the Step() unless the dt is changed too.
   // Otherwise, we are re-assembling e.g. the H_form at each subiteration
   // Maybe we can split up the UpdateForms and Assemble calls? And call Assemble before the convergence loop?

   // Bilinear forms
   if (full_update)
   {
      Mv_form->Update();
      D_form->Update();
      G_form->Update();
   }

   if ( full_update || !robin_dbcs.empty() || !amp_robin_dbcs.empty() )
   { // If we don't need a full update, and we don't have Robin bcs, the pressure operator is unchanged.
      Sp_form->Update();
   }

   H_form->Update();
   HMv_form->Update();
   N->Update();

   // Linear forms
   f_form->Update();
   FText_bdr_form->Update();
   g_bdr_form->Update();
   g_pres_robin_bdr_form->Update();
   t_pres_robin_bdr_form->Update();
   tau_pres_robin_bdr_form->Update(); // Not used yet
   g_vel_robin_bdr_form->Update();
   g_vel_tangent_amp_robin_bdr_form->Update();
   g_vel_norm_amp_robin_bdr_form->Update();
   g_vel_neumann_bdr_form->Update();

   // Evaluators
   mean_evaluator.reset(new MeanEvaluator(*pfes, gll_ir));

   curl_evaluator.reset(new CurlEvaluator(*vfes));

   delete stress_evaluator;
   stress_evaluator = new ShearStressEvaluator(*kin_vis_gf.ParFESpace(),
                                               *un_gf.ParFESpace(),
                                               gll_ir);

   Vector ones(pfes->GetTrueVSize());
   ones = 1.0;
   volume = mean_evaluator->ComputeIntegral(ones);
}

void NavierSolver::AssembleConstantOperators()
{
   Array<int> empty;

   Mv_form->Assemble();
   Mv_form->FormSystemMatrix(empty, Mv);

   D_form->Assemble();
   D_form->FormRectangularSystemMatrix(empty, empty, D);

   G_form->Assemble();
   G_form->FormRectangularSystemMatrix(empty, empty, G);
}

void NavierSolver::AssembleTDOperators()
{
   H_form->Update();
   H_form->Assemble();
   H_form->FormSystemMatrix(vel_ess_tdof, H);

   if (full_update || !robin_dbcs.empty() || !amp_robin_dbcs.empty())
   { // If we don't need a full update, and we don't have Robin bcs, the pressure operator is unchanged.
      Sp_form->Update();
      Sp_form->Assemble();
      Sp_form->FormSystemMatrix(pres_ess_tdof, Sp);
   }

   HMv_form->Assemble();

   N->Setup();
}

void NavierSolver::Assemble( bool full_update_ )
{
   AssembleTDOperators();

   if (full_update_)
   {
      AssembleConstantOperators();
      full_update = false; // After this is called we don't need to do a full update anymore. Unless triggered by Update()
   }
}

void NavierSolver::UpdateSolvers()
{
   Array<int> empty;

   Vector diag_Mv_pa(vfes->GetTrueVSize());
   Mv_form->AssembleDiagonal(diag_Mv_pa);

   MvInvPC.reset(new OperatorJacobiSmoother(diag_Mv_pa, empty));

   MvInv.reset(new CGSolver(vfes->GetComm()));
   MvInv->iterative_mode = false;
   MvInv->SetOperator(*Mv);
   MvInv->SetPreconditioner(*MvInvPC);
   MvInv->SetPrintLevel(pl_mvsolve);
   MvInv->SetRelTol(1e-12);
   MvInv->SetMaxIter(200);

   delete lor;
   lor = new ParLORDiscretization(*Sp_form, pres_ess_tdof);
   delete SpInvPC;
   SpInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
   SpInvPC->SetSystemsOptions(pmesh->Dimension());           // Why without this i get PCG: The preconditioner is not positive definite? After action of preconditoner vector norm is 1e18
   SpInvPC->SetPrintLevel(pl_amg);
   SpInvPC->Mult(resp, pn);
   delete SpInvOrthoPC;
   SpInvOrthoPC = new OrthoSolver(vfes->GetComm());
   SpInvOrthoPC->SetSolver(*SpInvPC);
   delete SpInv;
   SpInv = new CGSolver(vfes->GetComm());
   SpInv->iterative_mode = true;
   SpInv->SetOperator(*Sp);
   if (!well_posed)
   {
      SpInv->SetPreconditioner(*SpInvOrthoPC);
   }
   else
   {
      SpInv->SetPreconditioner(*SpInvPC);
   }
   SpInv->SetPrintLevel(pl_spsolve);
   SpInv->SetRelTol(rtol_spsolve);
   SpInv->SetMaxIter(600);

   delete HInvPC;
   Vector diag_HMv_pa(vfes->GetTrueVSize());
   HMv_form->AssembleDiagonal(diag_HMv_pa);
   HInvPC = new OperatorJacobiSmoother(diag_HMv_pa, vel_ess_tdof);
   delete HInv;
   HInv = new CGSolver(vfes->GetComm());
   HInv->iterative_mode = true;
   HInv->SetOperator(*H);
   HInv->SetPreconditioner(*HInvPC);
   HInv->SetPrintLevel(pl_hsolve);
   HInv->SetRelTol(rtol_hsolve);
   HInv->SetMaxIter(300);
}

void NavierSolver::UpdateTimestepHistory(real_t dt)
{
   // Rotate values in time step history
   dthist[2] = dthist[1];
   dthist[1] = dthist[0];
   dthist[0] = dt;

   // Rotate values in nonlinear extrapolation history
   Nunm2 = Nunm1;
   Nunm1 = Nun;

   // Rotate values in solution history
   unm2 = unm1;
   unm1 = un;

   // Update the current solution and corresponding GridFunction
   un_next_gf.GetTrueDofs(un_next);
   un = un_next;
   un_gf.SetFromTrueDofs(un);
}

void NavierSolver::Step(real_t &time, real_t dt, int current_step,
                        bool provisional)
{
   sw_step.Start();

   SetTimeIntegrationCoefficients(current_step);

   H_bdfcoeff.constant = bd0 / dt;

   //////////////////////////////////////////////////////////////////////////
   /// ----- Set the current time for boundary conditions and forcing terms.
   ////////////////////////////////////////////////////////////////////////// 

   // Velocity Dirichlet bcs
   for (auto &vel_dbc : vel_dbcs)
   {
      vel_dbc.coeff->SetTime(time + dt);
   }
   for (auto &vel_dbc : vel_dbcs_xyz)
   {
      vel_dbc.coeff->SetTime(time + dt);
   }

   // Pressure Dirichlet bcs
   for (auto &pres_dbc : pres_dbcs)
   {
      pres_dbc.coeff->SetTime(time + dt);
   }

   // Robin bcs
   for (auto &robin_dbc : robin_dbcs)
   {
      robin_dbc.SetTime(time + dt);
      real_t new_beta = H_bdfcoeff.constant / robin_dbc.alphaP->constant;
      robin_dbc.SetBeta(new_beta);
   }
   for (auto &amp_robin_dbc : amp_robin_dbcs)
   {
      amp_robin_dbc.SetTime(time + dt);
      real_t new_beta = H_bdfcoeff.constant / amp_robin_dbc.alphaP->constant;
      amp_robin_dbc.SetBeta(new_beta);
   }

   // Forcing terms
   for (auto &accel_term : accel_terms)
   {
      accel_term.coeff->SetTime(time + dt);
   }

   // Neumann bcs
   for (auto &neumann_dbc : neumann_dbcs)
   {
      neumann_dbc.SetTime(time + dt);
   }

   //////////////////////////////////////////////////////////////
   /// ----- Update necessary forms, operators and solvers.
   //////////////////////////////////////////////////////////////

   UpdateForms(full_update);   
   Assemble(full_update);
   UpdateSolvers();

   // Extrapolated f^{n+1}.
   f_form->Assemble();
   f_form->ParallelAssemble(fn);

   // Nonlinear extrapolated terms.
   sw_extrap.Start();

   N->Mult(un, Nun);
   N->Mult(unm1, Nunm1);
   N->Mult(unm2, Nunm2);

   {
      const auto d_Nun = Nun.Read();
      const auto d_Nunm1 = Nunm1.Read();
      const auto d_Nunm2 = Nunm2.Read();
      auto d_Fext = Fext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      mfem::forall(Fext.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Fext[i] = ab1_ * d_Nun[i] +
                     ab2_ * d_Nunm1[i] +
                     ab3_ * d_Nunm2[i];
      });
   }

   Fext.Add(1.0, fn);

   if (filter_method == FilterMethod::HPFRT_LOS)
   {
      u_filter_basis_gf.ProjectGridFunction(un_gf);
      u_low_modes_gf.ProjectGridFunction(u_filter_basis_gf);
      const auto d_un_gf = un_gf.Read();
      const auto d_u_low_modes_gf = u_low_modes_gf.Read();
      auto d_hpfrt_gf = hpfrt_gf.ReadWrite();
      MFEM_FORALL(i,
                  un_gf.Size(),
                  d_hpfrt_gf[i] = -10.0 * (d_un_gf[i] - d_u_low_modes_gf[i]););

      hpfrt_gf.ParallelProject(hpfrt_tdofs);
   }
   else if (filter_method == FilterMethod::HPFRT_LPOLY)
   {
      BuildHPFForcing(un_gf);
   }

   // Fext = M^{-1} (F(u^{n}) + f^{n+1} + RT^{n+1})
   if (filter_method != FilterMethod::NONE)
   {
      Fext.Add(1.0, hpfrt_tdofs);
   }
   MvInv->Mult(Fext, tmp1);
   Fext.Set(1.0, tmp1);

   // Compute BDF terms.
   {
      const real_t bd1idt = -bd1 / dt;
      const real_t bd2idt = -bd2 / dt;
      const real_t bd3idt = -bd3 / dt;
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_Fext = Fext.ReadWrite();
      mfem::forall(Fext.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_Fext[i] += bd1idt * d_un[i] +
                      bd2idt * d_unm1[i] +
                      bd3idt * d_unm2[i];
      });
   }

   sw_extrap.Stop();

   // Pressure Poisson.
   sw_curlcurl.Start();
   {
      const auto d_un = un.Read();
      const auto d_unm1 = unm1.Read();
      const auto d_unm2 = unm2.Read();
      auto d_u_ext = u_ext.Write();
      const auto ab1_ = ab1;
      const auto ab2_ = ab2;
      const auto ab3_ = ab3;
      mfem::forall(Lext.Size(), [=] MFEM_HOST_DEVICE (int i)
      {
         d_u_ext[i] = ab1_ * d_un[i] +
                      ab2_ * d_unm1[i] +
                      ab3_ * d_unm2[i];
      });
   }

   u_ext_gf.SetFromTrueDofs(u_ext);

   // Compute curl(u_ext) and curl(curl(u_ext))
   {
      curl_evaluator->EnablePA(true);
      curl_evaluator->ComputeCurlCurl(u_ext, tmp1);
      curlcurlu_gf.SetFromTrueDofs(tmp1);
   }

   // (curl curl u)_i *= nu
   for (int d = 0; d < curlcurlu_gf.FESpace()->GetVDim(); d++)
   {
      for (int i = 0; i < curlcurlu_gf.FESpace()->GetNDofs(); i++)
      {
         int idx = Ordering::Map<Ordering::byNODES>(
                      curlcurlu_gf.FESpace()->GetNDofs(),
                      curlcurlu_gf.FESpace()->GetVDim(),
                      i,
                      d);
         curlcurlu_gf(idx) *= kin_vis_gf(i);
      }
   }
   curlcurlu_gf.GetTrueDofs(Lext);

   // Vorticity of the extrapolated velocity, only if it was requested through
   // GetCurrentVorticity().
   if (compute_vorticity)
   {
      if (pmesh->Dimension() == 2)
      {
         ComputeCurl2D(u_ext_gf, curlu_gf);
      }
      else
      {
         ComputeCurl3D(u_ext_gf, curlu_gf);
      }
   }

   sw_curlcurl.Stop();

   // \tilde{F} = F - \nu CurlCurl(u)
   FText.Set(-1.0, Lext);
   FText.Add(1.0, Fext);

   if (variable_viscosity)
   {
      kin_vis_gf.GetTrueDofs(kv);
      stress_evaluator->Apply(kv, u_ext, grad_nu_sym_grad_uext);
      FText.Add(1.0, grad_nu_sym_grad_uext);
   }

   // p_r = - \nabla \cdot FText
   D->Mult(FText, resp);
   resp.Neg(); 

   // Add boundary terms.            
   // 1. On Dirichlet boundaries add Neumann bc:   \nabla p_r \cdot n = density * [ - bd0 / dt * uD \cdot n + FText \cdot n ] 
   FText_gf.SetFromTrueDofs(FText);
   FText_bdr_form->Assemble();
   FText_bdr_form->ParallelAssemble(FText_bdr);      // NOTE: this is adding also the same contribution on the Robin boundaries
   resp.Add(1.0, FText_bdr);

   g_bdr_form->Assemble();
   g_bdr_form->ParallelAssemble(g_bdr);
   resp.Add(- bd0 / dt, g_bdr);

   // 2. On Robin boundaries
   g_pres_robin_bdr_form->Assemble();
   g_pres_robin_bdr_form->ParallelAssemble(g_bdr);
   resp.Add(- bd0 / dt, g_bdr);

   t_pres_robin_bdr_form->Assemble();
   t_pres_robin_bdr_form->ParallelAssemble(g_bdr);
   resp.Add(- 1.0, g_bdr);

   tau_pres_robin_bdr_form->Assemble();
   tau_pres_robin_bdr_form->ParallelAssemble(g_bdr);
   resp.Add(1.0, g_bdr);

   resp *= density;

   // Orthogonalize and project the pressure dirichlet boundary conditions.
   if (!well_posed)
   {
      Orthogonalize(resp);
   }

   for (auto &pres_dbc : pres_dbcs)
   {
      pn_gf.ProjectBdrCoefficient(*pres_dbc.coeff, pres_dbc.attr);
   }

   // Neumann BCs: project derived pressure Dirichlet p = (τ·n)·n̂ - g·n̂.
   // u_ext_gf is already set above, so shear_stress_normal_traction_coeff is current.
   for (auto &neumann_dbc : neumann_dbcs)
   {
      pn_gf.ProjectBdrCoefficient(*neumann_dbc.pres_coeff, neumann_dbc.attr);
   }

   pfes->GetRestrictionMatrix()->MultTranspose(resp, resp_gf);

   Vector X1, B1;
   auto *SpC = Sp.As<ConstrainedOperator>();
   EliminateRHS(*Sp_form, *SpC, pres_ess_tdof, pn_gf, resp_gf, X1, B1, 1);
   sw_spsolve.Start();
   SpInv->Mult(B1, X1);
   sw_spsolve.Stop();
   iter_spsolve = SpInv->GetNumIterations();
   res_spsolve = SpInv->GetFinalNorm();
   Sp_form->RecoverFEMSolution(X1, resp_gf, pn_gf);

   // If the boundary conditions on the pressure are pure Neumann remove the
   // nullspace by removing the mean of the pressure solution. This is also
   // ensured by the OrthoSolver wrapper for the preconditioner which removes
   // the nullspace after every application.
   if (!well_posed)
   {
      MeanZero(pn_gf);
   }

   pn_gf.GetTrueDofs(pn);

   // Project velocity.
   G->Mult(pn, resu);
   resu *= (1.0/density);
   resu.Neg();
   Mv->Mult(Fext, tmp1);
   resu.Add(1.0, tmp1);

   g_vel_robin_bdr_form->Assemble();
   g_vel_robin_bdr_form->ParallelAssemble(tmp1);
   resu.Add(1.0/density, tmp1);

   g_vel_tangent_amp_robin_bdr_form->Assemble();
   g_vel_tangent_amp_robin_bdr_form->ParallelAssemble(tmp1);
   resu.Add(1.0/density, tmp1);

   g_vel_norm_amp_robin_bdr_form->Assemble();
   g_vel_norm_amp_robin_bdr_form->ParallelAssemble(tmp1);
   resu.Add(1.0/density, tmp1);

   // Neumann BCs: add natural traction (g + p n, v)_Γ to the Helmholtz RHS.
   if (!neumann_dbcs.empty())
   {
      g_vel_neumann_bdr_form->Assemble();
      g_vel_neumann_bdr_form->ParallelAssemble(tmp1);
      resu.Add(1.0/density, tmp1);
   }

   // un_next_gf = un_gf;

   for (auto &vel_dbc : vel_dbcs)
   {
      un_next_gf.ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }

   ParGridFunction tmp_gf(vfes);              // temporary velocity gf for projection
   Vector          tmp_vec(vfes->GetTrueVSize());   // temporary velocity vector for projection
   Array<int>      tmp_tdofs;
   un_next_gf.GetTrueDofs(un_next);
   for (auto &vel_dbc : vel_dbcs_xyz)
   {
      VectorArrayCoefficient tmp_coeff(pmesh->Dimension());            // Set coefficient with right component
      tmp_coeff.Set(vel_dbc.dir, vel_dbc.coeff, false);
      tmp_coeff.SetTime(time + dt);                                    // Eval(i,..) forwards tmp_coeff's own time to the component
      tmp_gf.ProjectBdrCoefficient(tmp_coeff, vel_dbc.attr);           // Project on dummy gf
      tmp_gf.GetTrueDofs(tmp_vec);

      vfes->GetEssentialTrueDofs(vel_dbc.attr,tmp_tdofs,vel_dbc.dir);  // Update solution dofs
      for(int i=0;i<tmp_tdofs.Size();i++)
      {
         un_next[tmp_tdofs[i]]=tmp_vec[tmp_tdofs[i]];
      }      
   }   
   un_next_gf.SetFromTrueDofs(un_next);

   vfes->GetRestrictionMatrix()->MultTranspose(resu, resu_gf);

   Vector X2, B2;
   auto *HC = H.As<ConstrainedOperator>();
   EliminateRHS(*H_form, *HC, vel_ess_tdof, un_next_gf, resu_gf, X2, B2, 1);   // NOTE: fails here because H_form is assmebled without PA
   sw_hsolve.Start();
   HInv->Mult(B2, X2);
   sw_hsolve.Stop();
   iter_hsolve = HInv->GetNumIterations();
   res_hsolve = HInv->GetFinalNorm();
   H_form->RecoverFEMSolution(X2, resu_gf, un_next_gf);

   un_next_gf.GetTrueDofs(un_next);

   // If the current time step is not provisional, accept the computed solution
   // and update the time step history by default.
   if (!provisional)
   {
      UpdateTimestepHistory(dt);
      time += dt;
   }

   sw_step.Stop();

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(7) << "" << std::setw(3) << "It" << std::setw(8)
                << "Resid" << std::setw(12) << "Reltol"
                << "\n";
      mfem::out << std::setw(5) << "PRES " << std::setw(5) << std::fixed
                << iter_spsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_spsolve << "   " << rtol_spsolve
                << "\n";
      mfem::out << std::setw(5) << "HELM " << std::setw(5) << std::fixed
                << iter_hsolve << "   " << std::setw(3) << std::setprecision(2)
                << std::scientific << res_hsolve << "   " << rtol_hsolve
                << "\n";
      mfem::out << std::setprecision(8);
      mfem::out << std::fixed;
   }
}

void NavierSolver::ComputeCurl3D(ParGridFunction &u, ParGridFunction &cu)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         curl.SetSize(3);
         curl(0) = grad(2, 1) - grad(1, 2);
         curl(1) = grad(0, 2) - grad(2, 0);
         curl(2) = grad(1, 0) - grad(0, 1);

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

void NavierSolver::ComputeCurl2D(ParGridFunction &u,
                                 ParGridFunction &cu,
                                 bool assume_scalar)
{
   FiniteElementSpace *fes = u.FESpace();

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector vals;
   Vector loc_data;
   int vdim = fes->GetVDim();
   DenseMatrix grad_hat;
   DenseMatrix dshape;
   DenseMatrix grad;
   Vector curl;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      fes->GetElementVDofs(e, vdofs);
      u.GetSubVector(vdofs, loc_data);
      vals.SetSize(vdofs.Size());
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();
      int dim = el->GetDim();
      dshape.SetSize(elndofs, dim);

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         // Eval and GetVectorGradientHat.
         el->CalcDShape(tr->GetIntPoint(), dshape);
         grad_hat.SetSize(vdim, dim);
         DenseMatrix loc_data_mat(loc_data.GetData(), elndofs, vdim);
         MultAtB(loc_data_mat, dshape, grad_hat);

         const DenseMatrix &Jinv = tr->InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);

         if (assume_scalar)
         {
            curl.SetSize(2);
            curl(0) = grad(0, 1);
            curl(1) = -grad(0, 0);
         }
         else
         {
            curl.SetSize(2);
            curl(0) = grad(1, 0) - grad(0, 1);
            curl(1) = 0.0;
         }

         for (int j = 0; j < curl.Size(); ++j)
         {
            vals(elndofs * j + dof) = curl(j);
         }
      }

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof = vdofs[j];
         cu(ldof) += vals[j];
         zones_per_vdof[ldof]++;
      }
   }

   // Communication.

   // Count the zones globally.
   GroupCommunicator &gcomm = u.ParFESpace()->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast(zones_per_vdof);

   // Accumulate for all vdofs.
   gcomm.Reduce<real_t>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<real_t>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

int NavierSolver::PredictTimestep(real_t dt_min, real_t dt_max,
                                  real_t cfl_target, real_t &dt)
{
   int flag = 0;
   real_t previous_dt = dt;
   real_t cflmax = 1.2 * cfl_target;
   real_t cflmin = 0.8 * cfl_target;

   real_t vel_max = ComputeCFL(un_next_gf, 1.0);
   real_t cfl = dt * vel_max;
   real_t vel_cfl_old = vel_cfl;
   vel_cfl = vel_max;

   // first timestep, initialize values
   if (dthist[1] == 0.0)
   {
      cfl_old = cfl;
   }

   real_t cflpred = 2.0*cfl-cfl_old;

   if (cfl > cflmax || cflpred > cflmax || cfl < cflmin)
   {
      real_t A = (vel_cfl - vel_cfl_old) / dt;
      real_t B = vel_cfl;
      real_t C = -cfl_target;
      real_t discr = (B*B) - 4.0 * A * C;

      if (discr <= 0.0)
      {
         dt = dt * (cfl_target/cfl);
      }
      else if (std::fabs(vel_cfl - vel_cfl_old)/vel_cfl < 0.001)
      {
         dt = dt * (cfl_target/cfl);
      }
      else
      {
         real_t dtlow = (-B+sqrt(discr) )/(2.0*A);
         real_t dthi = (-B-sqrt(discr) )/(2.0*A);
         if (dthi > 0.0 && dtlow > 0.0)
         {
            dt = std::min(dthi, dtlow);
         }
         else if (dthi <= 0.0 && dtlow <= 0.0)
         {
            dt = dt * (cfl_target/cfl);
         }
         else
         {
            dt = std::max(dthi, dtlow);
         }
      }

      if (previous_dt/dt < 0.2)
      {
         dt = previous_dt * 5.0;
      }
   }

   if (Mpi::Root())
   {
      if (dt < dt_min)
      {
         MFEM_ABORT("Minimum timestep reached, likely unstable.");
      }
   }

   if (cfl > cflmax || cflpred > cflmax)
   {
      flag = -1;
   }

   dt = std::min(1.2*previous_dt, dt);
   dt = std::max(0.8*previous_dt, dt);
   dt = std::min(dt_max, dt);
   cfl_old = cfl;

   return flag;
}

void NavierSolver::MeanZero(ParGridFunction &v)
{
   Vector tvec;
   v.GetTrueDofs(tvec);
   mean_evaluator->MakeMeanZero(tvec);
   v.Distribute(tvec);
}

void NavierSolver::EliminateRHS(ParBilinearForm &A_form,
                                ConstrainedOperator &constrainedA,
                                const Array<int> &ess_tdof_list,
                                Vector &x,
                                Vector &b,
                                Vector &X,
                                Vector &B,
                                int copy_interior)
{
   const Operator *Po = A_form.GetOutputProlongation();
   const Operator *Pi = A_form.GetProlongation();
   const Operator *Ri = A_form.GetRestriction();
   A_form.InitTVectors(Po, Ri, Pi, x, b, X, B);
   if (!copy_interior)
   {
      X.SetSubVectorComplement(ess_tdof_list, 0.0);
   }

   auto assembly = A_form.GetAssemblyLevel();
   if (assembly == AssemblyLevel::LEGACY)
   {
      A_form.EliminateVDofsInRHS(ess_tdof_list, X, B);
   }
   else
   {
      constrainedA.EliminateRHS(X, B);
   }
}

void NavierSolver::Orthogonalize(Vector &v)
{
   real_t loc_sum = v.Sum();
   real_t global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MFEM_MPI_REAL_T, MPI_SUM, pfes->GetComm());
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, pfes->GetComm());

   v -= global_sum / static_cast<real_t>(global_size);
}

real_t NavierSolver::ComputeCFL(ParGridFunction &u, real_t dt)
{
   ParMesh *pmesh_u = u.ParFESpace()->GetParMesh();
   FiniteElementSpace *fes = u.FESpace();
   int vdim = fes->GetVDim();

   Vector ux, uy, uz;
   Vector ur, us, ut;
   real_t cflx = 0.0;
   real_t cfly = 0.0;
   real_t cflz = 0.0;
   real_t cflm = 0.0;
   real_t cflmax = 0.0;

   // The integration rule here has to conform with the nonlinear term.
   auto ir = gll_ir;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      u.GetValues(e, ir, ux, 1);
      ur.SetSize(ux.Size());
      u.GetValues(e, ir, uy, 2);
      us.SetSize(uy.Size());
      if (vdim == 3)
      {
         u.GetValues(e, ir, uz, 3);
         ut.SetSize(uz.Size());
      }

      real_t hmin = pmesh_u->GetElementSize(e, 1) /
                    (real_t) fes->GetTypicalFE()->GetOrder();

      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         cflx = fabs(dt * ux(i) / hmin);
         cfly = fabs(dt * uy(i) / hmin);

         if (vdim == 3)
         {
            cflz = fabs(dt * ut(i) / hmin);
         }
         cflm = cflx + cfly + cflz;
         cflmax = fmax(cflmax, cflm);
      }
   }

   real_t cflmax_global = 0.0;
   MPI_Allreduce(&cflmax,
                 &cflmax_global,
                 1,
                 MFEM_MPI_REAL_T,
                 MPI_MAX,
                 pmesh_u->GetComm());

   return cflmax_global;
}

void NavierSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attrs, bool own)
{
   vel_dbcs.emplace_back(attrs, coeff, own);

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierSolver:: Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attrs.Size(); ++i)
      {
         if (attrs[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attrs.Size(); ++i)
   {
      MFEM_ASSERT(CheckBCAttr(i, attrs) == 0,
                  "Duplicate boundary definition detected.");
      if (attrs[i] == 1)
      {
         vel_ess_attr[i] = 1;
      }
   }
}

int NavierSolver::CheckBCAttr(int i, Array<int> &attrs)
{
   // Check if the attribute is already defined for a boundary condition.
   return (vel_ess_attr[i] || pres_ess_attr[i] || robin_ess_attr[i] || amp_robin_ess_attr[i] || neumann_ess_attr[i]) && attrs[i];
}

void NavierSolver::AddVelDirichletBC(VecFuncT *f, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr, true);
}


void NavierSolver::AddVelDirichletBC(Coefficient *coeff, Array<int> &attrs, int component, bool own)
{
   // Add bc container to list of componentwise velocity bcs
   vel_dbcs_xyz.emplace_back(attrs, coeff, component, own);

   // Check for duplicate and add attributes for current bc to global list (for that specific component)
   std::string dir_string;
   for (int i = 0; i < attrs.Size(); ++i)
   {
      switch (component) {
            case 0: // x 
               dir_string = "x";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_x[i]) && attrs[i]) == 0,
                           "Duplicate boundary definition for x component detected.");
               if (attrs[i] == 1){vel_ess_attr_x[i] = 1;}
               break;
            case 1: // y
               dir_string = "y";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_y[i]) && attrs[i]) == 0,
                           "Duplicate boundary definition for y component detected.");
               if (attrs[i] == 1){vel_ess_attr_y[i] = 1;}
               break;
            case 2: // z
               dir_string = "z";
               MFEM_ASSERT(( (vel_ess_attr[i] || vel_ess_attr_z[i]) && attrs[i]) == 0,
                           "Duplicate boundary definition for z component detected.");
               if (attrs[i] == 1){vel_ess_attr_z[i] = 1;}
               break;
            default:;
         }      
   }

   // Output
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC ( " << dir_string << " component) to attributes: " << std::endl;
      for (int i = 0; i < attrs.Size(); ++i)
      {
         if (attrs[i] == 1)
         {
            mfem::out << i << ", ";
         }
      }
      mfem::out << std::endl;
   }
}


void NavierSolver::AddVelDirichletBC(Coefficient *coeff, int attr, int component, bool own)
{
   Array<int> attrs(max_bdr_attributes);
   attrs = 0;
   attrs[attr - 1] = 1;
   AddVelDirichletBC(coeff, attrs, component, own);
}


void NavierSolver::AddPresDirichletBC(Coefficient *coeff, Array<int> &attrs, bool own)
{
   pres_dbcs.emplace_back(attrs, coeff, own);

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierSolver:: Adding Pressure Dirichlet BC to attributes ";
      for (int i = 0; i < attrs.Size(); ++i)
      {
         if (attrs[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attrs.Size(); ++i)
   {
      MFEM_ASSERT(CheckBCAttr(i, attrs) == 0,
                  "Duplicate boundary definition detected.");
      if (attrs[i] == 1)
      {
         pres_ess_attr[i] = 1;
      }
   }
}

void NavierSolver::AddPresDirichletBC(ScalarFuncT *f, Array<int> &attr)
{
   AddPresDirichletBC(new FunctionCoefficient(f), attr, true);
}


void NavierSolver::AddNeumannBC(VectorCoefficient *traction_coeff, Array<int> &attrs, bool own, bool use_shear_stress,
                                bool use_div, Coefficient *alpha_div,
                                bool own_alpha_div)
{
   neumann_dbcs.emplace_back(attrs, traction_coeff, own, use_shear_stress,
                             use_div, alpha_div, own_alpha_div);

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierSolver:: Adding Neumann BC to attributes ";
      for (int i = 0; i < attrs.Size(); ++i)
      {
         if (attrs[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attrs.Size(); ++i)
   {
      MFEM_ASSERT(CheckBCAttr(i, attrs) == 0,
                  "Duplicate boundary definition detected.");
      if (attrs[i] == 1)
      {
         neumann_ess_attr[i] = 1;
      }
   }
}

void NavierSolver::AddNeumannBC(VecFuncT *traction_f, Array<int> &attrs)
{
   AddNeumannBC(new VectorFunctionCoefficient(pmesh->Dimension(), traction_f), attrs, true);
}


void NavierSolver::AddNeumannBC(VectorCoefficient *traction_coeff, int attr, bool own, bool use_shear_stress, 
                                bool use_div, Coefficient *alpha_div,
                                bool own_alpha_div)
{
   Array<int> attrs(max_bdr_attributes);
   attrs = 0;
   attrs[attr - 1] = 1;
   AddNeumannBC(traction_coeff, attrs, own, use_shear_stress, use_div, alpha_div, own_alpha_div);
}

void NavierSolver::AddNeumannBC(VecFuncT *traction_f, int attr)
{
   AddNeumannBC(new VectorFunctionCoefficient(pmesh->Dimension(), traction_f), attr, true);
}

/// Add a Robin boundary condition 
void NavierSolver::AddRobinBC(real_t alpha, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, Array<int> &attrs, bool own)
{
   MFEM_ASSERT(attrs.Size() >= max_bdr_attributes,  "Size of attributes array does not match mesh attributes.");

   // Append to the existing Robin boundary conditions.
   // -1 for the coefficients sigma_f \cdot n accounts for the fact that it's computed with the solid normal and not the fluid normal.
   std::vector<VectorCoefficient *> coeffs = {vel_coeff, traction_coeff};
   std::vector<real_t> scaling = {1.0, -1.0}; 
   robin_dbcs.emplace_back(attrs, alpha, density, coeffs, scaling, own);   

   // Check for duplicate
   for (int i = 0; i < attrs.Size(); ++i)
   {
      MFEM_ASSERT(CheckBCAttr(i, attrs) == 0,
                  "Duplicate boundary definition detected.");
      if (attrs[i] == 1)
      {
         robin_ess_attr[i] = 1;
      }
   }

   // Output information about the added Robin boundary condition.
   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierSolver:: Adding Robin BC with alpha=" << alpha << ", density=" << density << " to attributes ";
      for (int i = 0; i < attrs.Size(); ++i)
      {
         if (attrs[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierSolver::AddRobinBC(real_t alpha, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, int attr, bool own)
{
   Array<int> attrs(max_bdr_attributes);
   attrs = 0;
   attrs[attr - 1] = 1;
   AddRobinBC(alpha, vel_coeff, traction_coeff, attrs, own);
}

void NavierSolver::AddRobinBC(real_t alpha, VecFuncT *vel_f, VecFuncT *traction_f, Array<int> &attrs)
{
   AddRobinBC(alpha,
              new VectorFunctionCoefficient(pmesh->Dimension(), vel_f),
              new VectorFunctionCoefficient(pmesh->Dimension(), traction_f),
              attrs, true);
}

void NavierSolver::AddRobinBC(real_t alpha, VecFuncT *vel_f, VecFuncT *traction_f, int attr)
{
   AddRobinBC(alpha,
              new VectorFunctionCoefficient(pmesh->Dimension(), vel_f),
              new VectorFunctionCoefficient(pmesh->Dimension(), traction_f),
              attr, true);
}


// Add Robin boundary condition (AMP version)
void NavierSolver::AddAMPRobinBC(real_t alpha_s, real_t alpha_p, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, Array<int> &attrs, bool own)
{
   MFEM_ASSERT(attrs.Size() >= max_bdr_attributes,  "Size of attributes array does not match mesh attributes.");

   // Append to the existing Robin boundary conditions.
   // -1 for the coefficients sigma_f \cdot n accounts for the fact that it's computed with the solid normal and not the fluid normal.
   std::vector<VectorCoefficient *> coeffs = {vel_coeff, traction_coeff};
   std::vector<real_t> scaling = {1.0, -1.0}; 
   amp_robin_dbcs.emplace_back(attrs, alpha_s, alpha_p, density, coeffs, scaling, own);   

   // Check for duplicate
   for (int i = 0; i < attrs.Size(); ++i)
   {
      MFEM_ASSERT(CheckBCAttr(i, attrs) == 0,
                  "Duplicate boundary definition detected.");
      if (attrs[i] == 1)
      {
         amp_robin_ess_attr[i] = 1;
      }
   }

   // Output information about the added Robin boundary condition.
   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierSolver:: Adding AMP Robin BC with alpha_s=" << alpha_s << ", alpha_p=" << alpha_p << ", density=" << density << " to attributes ";
      for (int i = 0; i < attrs.Size(); ++i)
      {
         if (attrs[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierSolver::AddAMPRobinBC(real_t alpha_s, real_t alpha_p, VectorCoefficient *vel_coeff, VectorCoefficient *traction_coeff, int attr, bool own)
{
   Array<int> attrs(max_bdr_attributes);
   attrs = 0;
   attrs[attr - 1] = 1;
   AddAMPRobinBC(alpha_s, alpha_p, vel_coeff, traction_coeff, attrs, own);
}

void NavierSolver::AddAMPRobinBC(real_t alpha_s, real_t alpha_p, VecFuncT *vel_f, VecFuncT *traction_f, Array<int> &attrs)
{
   AddAMPRobinBC(alpha_s, alpha_p,
              new VectorFunctionCoefficient(pmesh->Dimension(), vel_f),
              new VectorFunctionCoefficient(pmesh->Dimension(), traction_f),
              attrs, true);
}

void NavierSolver::AddAMPRobinBC(real_t alpha_s, real_t alpha_p, VecFuncT *vel_f, VecFuncT *traction_f, int attr)
{
   AddAMPRobinBC(alpha_s, alpha_p,
              new VectorFunctionCoefficient(pmesh->Dimension(), vel_f),
              new VectorFunctionCoefficient(pmesh->Dimension(), traction_f),
              attr, true);
}


// Setters for the Robin boundary condition coefficients.
void NavierSolver::SetRobinBCAlpha(real_t alpha)
{
   for (auto &robin_dbc : robin_dbcs)
   {
      robin_dbc.SetAlphaP(alpha);
   }
}

void NavierSolver::SetAMPRobinBCAlphaP(real_t alpha_p)
{
   for (auto &amp_robin_dbc : amp_robin_dbcs)
   {
      amp_robin_dbc.SetAlphaP(alpha_p);
   }
}

void NavierSolver::SetAMPRobinBCAlphaS(real_t alpha_s)
{
   for (auto &amp_robin_dbc : amp_robin_dbcs)
   {
      amp_robin_dbc.SetAlphaS(alpha_s);
   }
}


// Add an acceleration term to the Navier-Stokes equations.
void NavierSolver::AddAccelTerm(VectorCoefficient *coeff, Array<int> &attr)
{
   accel_terms.emplace_back(attr, coeff);

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NavierSolver:: Adding Acceleration term to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }
}

void NavierSolver::AddAccelTerm(VecFuncT *f, Array<int> &attr)
{
   AddAccelTerm(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

void NavierSolver::SetTimeIntegrationCoefficients(int step)
{
   // Maximum BDF order to use at current time step
   // step + 1 <= order <= max_bdf_order
   int bdf_order = std::min(step + 1, max_bdf_order);

   // Ratio of time step history at dt(t_{n}) - dt(t_{n-1})
   real_t rho1 = 0.0;

   // Ratio of time step history at dt(t_{n-1}) - dt(t_{n-2})
   real_t rho2 = 0.0;

   rho1 = dthist[0] / dthist[1];

   if (bdf_order == 3)
   {
      rho2 = dthist[1] / dthist[2];
   }

   if (step == 0 && bdf_order == 1)
   {
      bd0 = 1.0;
      bd1 = -1.0;
      bd2 = 0.0;
      bd3 = 0.0;
      ab1 = 1.0;
      ab2 = 0.0;
      ab3 = 0.0;
   }
   else if (step >= 1 && bdf_order == 2)
   {
      bd0 = (1.0 + 2.0 * rho1) / (1.0 + rho1);
      bd1 = -(1.0 + rho1);
      bd2 = pow(rho1, 2.0) / (1.0 + rho1);
      bd3 = 0.0;
      ab1 = 1.0 + rho1;
      ab2 = -rho1;
      ab3 = 0.0;
   }
   else if (step >= 2 && bdf_order == 3)
   {
      bd0 = 1.0 + rho1 / (1.0 + rho1)
            + (rho2 * rho1) / (1.0 + rho2 * (1 + rho1));
      bd1 = -1.0 - rho1 - (rho2 * rho1 * (1.0 + rho1)) / (1.0 + rho2);
      bd2 = pow(rho1, 2.0) * (rho2 + 1.0 / (1.0 + rho1));
      bd3 = -(pow(rho2, 3.0) * pow(rho1, 2.0) * (1.0 + rho1))
            / ((1.0 + rho2) * (1.0 + rho2 + rho2 * rho1));
      ab1 = ((1.0 + rho1) * (1.0 + rho2 * (1.0 + rho1))) / (1.0 + rho2);
      ab2 = -rho1 * (1.0 + rho2 * (1.0 + rho1));
      ab3 = (pow(rho2, 2.0) * rho1 * (1.0 + rho1)) / (1.0 + rho2);
   }
}

void NavierSolver::PrintTimingData()
{
   real_t my_rt[6], rt_max[6];

   my_rt[0] = sw_setup.RealTime();
   my_rt[1] = sw_step.RealTime();
   my_rt[2] = sw_extrap.RealTime();
   my_rt[3] = sw_curlcurl.RealTime();
   my_rt[4] = sw_spsolve.RealTime();
   my_rt[5] = sw_hsolve.RealTime();

   MPI_Reduce(my_rt, rt_max, 6, MFEM_MPI_REAL_T, MPI_MAX, 0, pmesh->GetComm());

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << std::setw(10) << "SETUP" << std::setw(10) << "STEP"
                << std::setw(10) << "EXTRAP" << std::setw(10) << "CURLCURL"
                << std::setw(10) << "PSOLVE" << std::setw(10) << "HSOLVE"
                << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << my_rt[0]
                << std::setw(10) << my_rt[1] << std::setw(10) << my_rt[2]
                << std::setw(10) << my_rt[3] << std::setw(10) << my_rt[4]
                << std::setw(10) << my_rt[5] << "\n";

      mfem::out << std::setprecision(3) << std::setw(10) << " " << std::setw(10)
                << my_rt[1] / my_rt[1] << std::setw(10) << my_rt[2] / my_rt[1]
                << std::setw(10) << my_rt[3] / my_rt[1] << std::setw(10)
                << my_rt[4] / my_rt[1] << std::setw(10) << my_rt[5] / my_rt[1]
                << "\n";

      mfem::out << std::setprecision(8);
   }
}

void NavierSolver::PrintInfo()
{
   int mesh_ne = vfes->GetParMesh()->GetGlobalNE();
   int fes_size0 = vfes->GlobalTrueVSize();
   int fes_size1 = pfes->GlobalTrueVSize();

   if (pmesh->GetMyRank() == 0)
   {
      mfem::out << "NAVIER version: " << NAVIER_VERSION << std::endl
                << "MFEM version: " << MFEM_VERSION << std::endl
                << "MFEM GIT: " << MFEM_GIT_STRING << std::endl
                << "Mesh #EL: " << mesh_ne << std::endl
                << "Velocity #DOFs: " << fes_size0 << std::endl
                << "Pressure #DOFs: " << fes_size1 << std::endl;
   }
}

void NavierSolver::EvaluateLegendrePolynomial(const int poly_order,
                                              const real_t x,
                                              Vector &L)
{
   L(0) = 1.0;
   L(1) = x;

   for (int i = 2; i < poly_order; i++)
   {
      L(i) = ((2*i-1) * x * L(i-1) - (i-1) * L(i-2)) / (real_t)i;
   }
}

void NavierSolver::EvaluateLegendrePolynomialShifted(const int poly_order,
                                                     const real_t x,
                                                     Vector &L)
{
   EvaluateLegendrePolynomial(poly_order, 2.0 * x - 1.0, L);
}

void NavierSolver::EvaluateMonomialBasis(const int poly_order, const real_t x,
                                         Vector &L)
{
   for (int i = 0; i < poly_order; i++)
   {
      L(i) = std::pow(x, i);
   }
}

void NavierSolver::EnableFilter(FilterMethod f)
{
   filter_method = f;
}

void NavierSolver::BuildHPFForcing(ParGridFunction &vel_gf)
{
   const int hpf_kco = filter_cutoff_modes;
   const real_t hpf_weight = filter_alpha;
   const real_t hpf_chi = -1.0 * hpf_weight;
   const int NN = order + 1;

   DenseMatrix D_filter;
   D_filter.Diag(1.0, NN);

   const int k0 = NN - hpf_kco;
   // Amplitude
   real_t a;

   for (int k = k0; k < NN; k++)
   {
      a = (pow((real_t)(k+1 - k0), 2.0))/pow((real_t)(hpf_kco), 2.0);
      D_filter(k, k) = 1.0 - a;
   }

   // Legendre Basis interpolation matrix
   DenseMatrix V(NN), Vinv(NN);
   Vector Vi(NN);

   for (int i = 0; i < NN; i++)
   {
      EvaluateLegendrePolynomialShifted(NN, gll_ir.IntPoint(i).x, Vi);
      V.SetRow(i, Vi);
   }

   DenseMatrix DfVinv(NN), F(NN), Ft(NN);
   Vinv = V;
   Vinv.Invert();

   // Construct F = V * D * V^-1
   Mult(D_filter, Vinv, DfVinv);
   Mult(V, DfVinv, F);

   Ft.Transpose(F);

   // compute u - F u F'
   Array<int> vdofs, dofs;
   Vector u_e, w1;
   auto nodal_fe = dynamic_cast<const NodalFiniteElement*>(vfes->GetTypicalFE());
   if (!nodal_fe)
   {
      MFEM_ABORT("internal error");
   }
   auto dofs_lex_ordering = nodal_fe->GetLexicographicOrdering();
   int ndofs = nodal_fe->GetDof();
   dofs.SetSize(ndofs);
   int dim = nodal_fe->GetDim();
   DenseMatrix W1, W2(NN);

   // hpfrt E-vector
   auto h1v_element_restriction = vfes->GetElementRestriction(
                                     ElementDofOrdering::LEXICOGRAPHIC);
   Vector hpfrt_e(h1v_element_restriction->Height());
   Vector hpfrt_edual(h1v_element_restriction->Height());

   for (int e = 0; e < vfes->GetNE(); ++e)
   {
      vfes->GetElementVDofs(e, vdofs);
      for (int d = 0; d < dim; d++)
      {
         // Retrieve input field in single dimension d
         for (int lex_idx = 0; lex_idx < ndofs; lex_idx++)
         {
            int scalar_h1_idx = dofs_lex_ordering[lex_idx];
            int vector_h1_idx = scalar_h1_idx + ndofs * d;
            dofs[lex_idx] = vdofs[vector_h1_idx];
         }

         vel_gf.GetSubVector(dofs, u_e);

         // Copy 1D input field
         w1 = u_e;
         // Make 1D input field available as a matrix. Assumes that the 1D
         // field is ordered the same as the quadrature rule.
         W1.UseExternalData(w1.GetData(), NN, NN);
         // W2 = F * W1
         Mult(F, W1, W2);
         // W1 = W2 * F'
         Mult(W2, Ft, W1);
         // In vector form, perform u_e = u_e - w1
         subtract(u_e, w1, u_e);
         // Set the dofs of the high-pass filtered GridFunction in the E-vector
         hpfrt_e.SetVector(u_e, ndofs * (e * dim + d));
      }
   }
   auto mv_integ = (*Mv_form->GetDBFI())[0];
   hpfrt_edual = 0.0;
   mv_integ->AddMultPA(hpfrt_e, hpfrt_edual);
   h1v_element_restriction->MultTranspose(hpfrt_edual, hpfrt_gf);

   hpfrt_gf.ParallelAssemble(hpfrt_tdofs);
   hpfrt_tdofs *= hpf_chi;
}

void NavierSolver::Update()
{
   UpdateForms(true);  // This might be avoided, as full_update = true will trigger Update of all forms in the next Step() call
   full_update = true; // Trigger Assembly of all forms in the next Step() call
   //UpdateSolvers(); -- >This is called inside the Step() already
}

NavierSolver::~NavierSolver()
{
   delete FText_gfcoeff;
   delete shear_stress_normal_traction_coeff;
   delete dyn_vis_coeff;
   delete pn_bdr_coeff;
   delete g_bdr_form;
   delete FText_bdr_form;
   delete g_pres_robin_bdr_form;
   delete t_pres_robin_bdr_form;
   delete tau_pres_robin_bdr_form;
   delete g_vel_robin_bdr_form;
   delete g_vel_tangent_amp_robin_bdr_form;
   delete g_vel_norm_amp_robin_bdr_form;
   delete g_vel_neumann_bdr_form;
   delete mass_lf;
   delete Mv_form;
   delete N;
   delete Sp_form;
   delete D_form;
   delete G_form;
   delete HInvPC;
   delete HInv;
   delete H_form;
   delete SpInv;
   delete SpInvOrthoPC;
   delete SpInvPC;
   delete lor;
   delete f_form;
   delete vfec;
   delete pfec;
   delete vfes;
   delete pfes;
   delete vfec_filter;
   delete vfes_filter;
   delete HMv_form;
   delete stress_evaluator;

   delete wg_coef;

   if (own_mesh_velocity)
   {
      delete wg_gf;
   }
}

real_t NavierSolver::NekNorm(ParGridFunction &u, int type, bool is_vector)
{
   real_t m = 0.0;
   if (component_mass_lf == nullptr)
   {
      onecoeff.constant = 1.0;
      component_mass_lf = new ParLinearForm(pfes);
      component_mass_lf->AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      component_mass_lf->Assemble();
   }

   if (is_vector)
   {
      ParGridFunction ux(pfes), uy(pfes), umag(pfes);

      for (int i = 0; i < ux.Size(); ++i)
      {
         ux(i) = u(vfes->DofToVDof(i, 0));
         uy(i) = u(vfes->DofToVDof(i, 1));
         umag(i) = pow(ux(i), 2.0) + pow(uy(i), 2.0);
      }
      if (type == 1)
      {
         for (int i = 0; i < umag.Size(); ++i)
         {
            m = std::max(m, std::abs(sqrt(umag(i))));
         }
         return m;
      }
      return sqrt(component_mass_lf->operator()(umag) / volume);
   }

   ParGridFunction w(pfes);
   for (int i = 0; i < w.Size(); ++i)
   {
      w(i) = pow(u(i), 2.0);
      m = std::max(m, std::abs(u(i)));
   }
   if (type == 1)
   {
      return m;
   }
   return sqrt(component_mass_lf->operator()(w) / volume);
}
