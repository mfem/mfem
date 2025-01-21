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

#include "tmop_amr.hpp"

namespace mfem
{

using namespace mfem;

void TMOPRefinerEstimator::ComputeEstimates()
{
   bool iso = false;
   bool aniso = false;
   if (amrmetric == 1 || amrmetric == 2 || amrmetric == 58)
   {
      aniso = true;
   }
   if (amrmetric == 55 || amrmetric == 56 || amrmetric == 77 ||
       amrmetric == 315 || amrmetric == 316 || amrmetric == 321)
   {
      iso = true;
   }
   if (amrmetric == 7 || amrmetric == 9)
   {
      iso = true; aniso = true;
   }

   MFEM_VERIFY(iso || aniso, "Metric type not supported in hr-adaptivity.");

   const int dim = mesh->Dimension();
   const int num_ref_types = 3 + 4*(dim-2);
   const int NEorig = mesh->GetNE();

   aniso_flags.SetSize(NEorig);
   error_estimates.SetSize(NEorig);
   Vector amr_base_energy(NEorig), amr_temp_energy(NEorig);
   error_estimates = 1.*std::numeric_limits<float>::max();
   aniso_flags = -1;
   GetTMOPRefinementEnergy(0, amr_base_energy);

   for (int i = 1; i < num_ref_types+1; i++)
   {
      if ( dim == 2 && i < 3  && aniso != true ) { continue; }
      if ( dim == 2 && i == 3 && iso   != true ) { continue; }
      if ( dim == 3 && i < 7  && aniso != true ) { continue; }
      if ( dim == 3 && i == 7 && iso   != true ) { continue; }

      GetTMOPRefinementEnergy(i, amr_temp_energy);

      for (int e = 0; e < NEorig; e++)
      {
         if ( amr_temp_energy(e) < error_estimates(e) )
         {
            error_estimates(e) = amr_temp_energy(e);
            aniso_flags[e] = i;
         }
      }
   }
   error_estimates *= energy_scaling_factor;

   if (spat_gf)
   {
      L2_FECollection avg_fec(0, mesh->Dimension());
      FiniteElementSpace avg_fes(spat_gf->FESpace()->GetMesh(), &avg_fec);
      GridFunction elem_avg(&avg_fes);
      spat_gf->GetElementAverages(elem_avg);
      for (int i = 0; i < amr_base_energy.Size(); i++)
      {
         if (elem_avg(i) < spat_gf_critical) { amr_base_energy(i) = 0.; }
      }
   }

   error_estimates -= amr_base_energy;
   error_estimates *= -1; // error = E(parent) - scaling_factor*mean(E(children))
   current_sequence = mesh->GetSequence();
}

void TMOPRefinerEstimator::GetTMOPRefinementEnergy(int reftype,
                                                   Vector &el_energy_vec)
{
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   const int NE = fes->GetNE();
   GridFunction *xdof = mesh->GetNodes();
   xdof->SetTrueVector();
   xdof->SetFromTrueVector();

   el_energy_vec.SetSize(NE);
   el_energy_vec = std::numeric_limits<float>::max();

   for (int e = 0; e < NE; e++)
   {
      Geometry::Type gtype = fes->GetFE(e)->GetGeomType();
      DenseMatrix tr, xsplit;
      IntegrationRule *irule = NULL;

      if ( (gtype == Geometry::TRIANGLE && reftype > 0 && reftype < 3) ||
           (gtype == Geometry::CUBE && reftype > 0 && reftype < 7) ||
           (gtype == Geometry::TETRAHEDRON && reftype > 0 && reftype < 7) )
      {
         continue;
      }

      switch (gtype)
      {
         case Geometry::TRIANGLE:
         {
            int ref_access = reftype == 0 ? 0 : 1;
            xdof->GetVectorValues(e, *TriIntRule[ref_access], xsplit, tr);
            irule = TriIntRule[ref_access];
            break;
         }
         case Geometry::TETRAHEDRON:
         {
            int ref_access = reftype == 0 ? 0 : 1;
            xdof->GetVectorValues(e, *TetIntRule[ref_access], xsplit, tr);
            irule = TetIntRule[ref_access];
            break;
         }
         case Geometry::SQUARE:
         {
            MFEM_VERIFY(QuadIntRule[reftype], " Integration rule does not exist.");
            xdof->GetVectorValues(e, *QuadIntRule[reftype], xsplit, tr);
            irule = QuadIntRule[reftype];
            break;
         }
         case Geometry::CUBE:
         {
            int ref_access = reftype == 0 ? 0 : 1;
            xdof->GetVectorValues(e, *HexIntRule[ref_access], xsplit, tr);
            irule = HexIntRule[ref_access];
            break;
         }
         default:
            MFEM_ABORT("Incompatible geometry type!");
      }
      xsplit.Transpose();

      el_energy_vec(e) = 0.; // Re-set to 0

      // The data format is xe1,xe2,..xen,ye1,ye2..yen.
      // We will reformat it inside GetRefinementElementEnergy
      Vector elfun(xsplit.GetData(), xsplit.NumCols()*xsplit.NumRows());

      Array<NonlinearFormIntegrator*> &integs = *(nlf->GetDNFI());
      TMOP_Integrator *ti  = NULL;
      TMOPComboIntegrator *co = NULL;
      for (int i = 0; i < integs.Size(); i++)
      {
         ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
         if (ti)
         {
            el_energy_vec(e) = ti->GetRefinementElementEnergy(*fes->GetFE(e),
                                                              *mesh->GetElementTransformation(e),
                                                              elfun,
                                                              *irule);
         }
         co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
         if (co)
         {
            Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
            for (int j = 0; j < ati.Size(); j++)
            {
               el_energy_vec(e) += ati[j]->GetRefinementElementEnergy(*fes->GetFE(e),
                                                                      *mesh->GetElementTransformation(e),
                                                                      elfun,
                                                                      *irule);
            }
         }
      }
   }
}

void TMOPRefinerEstimator::SetHexIntRules()
{
   HexIntRule.SetSize(1+1);
   // Reftype = 0 -> original element
   Mesh meshsplit = Mesh::MakeCartesian3D(1, 1, 1, Element::HEXAHEDRON);
   Mesh base_mesh_copy(meshsplit);
   HexIntRule[0] = SetIntRulesFromMesh(meshsplit);
   meshsplit.Clear();

   // Reftype = 7
   for (int i = 7; i < 8; i++)
   {
      Array<Refinement> marked_elements;
      Mesh mesh_ref(base_mesh_copy);
      for (int e = 0; e < mesh_ref.GetNE(); e++)
      {
         marked_elements.Append(Refinement(e, i));
      }
      mesh_ref.GeneralRefinement(marked_elements, 1, 0);
      HexIntRule[1] = SetIntRulesFromMesh(mesh_ref);
      mesh_ref.Clear();
   }
}

void TMOPRefinerEstimator::SetQuadIntRules()
{
   QuadIntRule.SetSize(3+1);

   // Reftype = 0 -> original element
   Mesh meshsplit = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   Mesh base_mesh_copy(meshsplit);
   QuadIntRule[0] = SetIntRulesFromMesh(meshsplit);
   meshsplit.Clear();

   // Reftype = 1-3
   for (int i = 1; i < 4; i++)
   {
      Array<Refinement> marked_elements;
      Mesh mesh_ref(base_mesh_copy);
      for (int e = 0; e < mesh_ref.GetNE(); e++)
      {
         marked_elements.Append(Refinement(e, i));
      }
      mesh_ref.GeneralRefinement(marked_elements, 1, 0);
      QuadIntRule[i] = SetIntRulesFromMesh(mesh_ref);
      mesh_ref.Clear();
   }
}

void TMOPRefinerEstimator::SetTriIntRules()
{
   TriIntRule.SetSize(1+1);

   // Reftype = 0 // original element
   const int Nvert = 3, NEsplit = 1;
   Mesh meshsplit(2, Nvert, NEsplit, 0, 2);
   const real_t tri_v[3][2] =
   {
      {0, 0}, {1, 0}, {0, 1}
   };
   const int tri_e[1][3] =
   {
      {0, 1, 2}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit.AddVertex(tri_v[j]);
   }
   meshsplit.AddTriangle(tri_e[0], 1);
   meshsplit.FinalizeTriMesh(1, 1, true);

   Mesh base_mesh_copy(meshsplit);
   TriIntRule[0] = SetIntRulesFromMesh(meshsplit);
   meshsplit.Clear();

   // no anisotropic refinements for triangle
   // Reftype = 3
   for (int i = 1; i < 2; i++)
   {
      Array<Refinement> marked_elements;
      Mesh mesh_ref(base_mesh_copy);
      for (int e = 0; e < mesh_ref.GetNE(); e++)
      {
         marked_elements.Append(Refinement(e, i));
      }
      mesh_ref.GeneralRefinement(marked_elements, 1, 0);
      TriIntRule[i] = SetIntRulesFromMesh(mesh_ref);
      mesh_ref.Clear();
   }
}

void TMOPRefinerEstimator::SetTetIntRules()
{
   TetIntRule.SetSize(1+1);

   // Reftype = 0 // original element
   const int Nvert = 4, NEsplit = 1;
   Mesh meshsplit(3, Nvert, NEsplit, 0, 3);
   const real_t tet_v[4][3] =
   {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
   };
   const int tet_e[1][4] =
   {
      {0, 1, 2, 3}
   };

   for (int j = 0; j < Nvert; j++)
   {
      meshsplit.AddVertex(tet_v[j]);
   }
   meshsplit.AddTet(tet_e[0], 1);
   meshsplit.FinalizeTetMesh(1, 1, true);

   Mesh base_mesh_copy(meshsplit);
   TetIntRule[0] = SetIntRulesFromMesh(meshsplit);
   meshsplit.Clear();

   // no anisotropic refinements for triangle
   // Reftype = 7
   for (int i = 1; i < 2; i++)
   {
      Array<Refinement> marked_elements;
      Mesh mesh_ref(base_mesh_copy);
      for (int e = 0; e < mesh_ref.GetNE(); e++)
      {
         marked_elements.Append(Refinement(e, i)); // ref_type will default to 7
      }
      mesh_ref.GeneralRefinement(marked_elements, 1, 0);
      TetIntRule[i] = SetIntRulesFromMesh(mesh_ref);
      mesh_ref.Clear();
   }
}

IntegrationRule* TMOPRefinerEstimator::SetIntRulesFromMesh(Mesh &meshsplit)
{
   const int dim = meshsplit.Dimension();
   H1_FECollection fec(order, dim);
   FiniteElementSpace nodal_fes(&meshsplit, &fec, dim);
   meshsplit.SetNodalFESpace(&nodal_fes);

   const int NEsplit = meshsplit.GetNE();
   const int dof_cnt = nodal_fes.GetTypicalFE()->GetDof();
   const int pts_cnt = NEsplit * dof_cnt;

   DenseMatrix pos(dof_cnt, dim);
   Vector posV(pos.Data(), dof_cnt * dim);
   Array<int> xdofs(dof_cnt * dim);

   // Create an IntegrationRule on the nodes of the reference submesh.
   IntegrationRule *irule = new IntegrationRule(pts_cnt);
   GridFunction *nodesplit = meshsplit.GetNodes();

   int pt_id = 0;
   for (int i = 0; i < NEsplit; i++)
   {
      nodal_fes.GetElementVDofs(i, xdofs);
      nodesplit->GetSubVector(xdofs, posV);
      for (int j = 0; j < dof_cnt; j++)
      {
         if (dim == 2)
         {
            irule->IntPoint(pt_id).Set2(pos(j, 0), pos(j, 1));
         }
         else if (dim == 3)
         {
            irule->IntPoint(pt_id).Set3(pos(j, 0), pos(j, 1), pos(j, 2));
         }
         pt_id++;
      }
   }
   return irule;
}

bool TMOPDeRefinerEstimator::GetDerefineEnergyForIntegrator(
   TMOP_Integrator &tmopi,
   Vector &fine_energy)
{
   DiscreteAdaptTC *tcd = tmopi.GetDiscreteAdaptTC();
   fine_energy.SetSize(mesh->GetNE());

   if (serial)
   {
      Mesh meshcopy(*mesh);
      FiniteElementSpace *tcdfes = NULL;
      if (tcd)
      {
         tcdfes = new FiniteElementSpace(*tcd->GetTSpecFESpace(), &meshcopy);
      }

      Vector local_err(meshcopy.GetNE());
      local_err = 0.;
      real_t threshold = std::numeric_limits<float>::max();
      meshcopy.DerefineByError(local_err, threshold, 0, 1);

      if (meshcopy.GetGlobalNE() == mesh->GetGlobalNE())
      {
         delete tcdfes;
         return false;
      }

      if (tcd)
      {
         tcdfes->Update();
         tcd->SetTspecDataForDerefinement(tcdfes);
      }

      Vector coarse_energy(meshcopy.GetNE());
      GetTMOPDerefinementEnergy(meshcopy, tmopi, coarse_energy);
      if (tcd) { tcd->ResetDerefinementTspecData(); }
      GetTMOPDerefinementEnergy(*mesh, tmopi, fine_energy);

      const CoarseFineTransformations &dtrans =
         meshcopy.ncmesh->GetDerefinementTransforms();

      Table coarse_to_fine;
      dtrans.MakeCoarseToFineTable(coarse_to_fine);

      Array<int> tabrow;
      for (int pe = 0; pe < coarse_to_fine.Size(); pe++)
      {
         coarse_to_fine.GetRow(pe, tabrow);
         int nchild = tabrow.Size();
         real_t parent_energy = coarse_energy(pe);
         for (int fe = 0; fe < nchild; fe++)
         {
            int child = tabrow[fe];
            MFEM_VERIFY(child < mesh->GetNE(), " invalid coarse to fine mapping");
            fine_energy(child) -= parent_energy;
         }
      }
      delete tcdfes;
   }
   else
   {
#ifdef MFEM_USE_MPI
      ParMesh meshcopy(*pmesh);
      ParFiniteElementSpace *tcdfes = NULL;
      if (tcd)
      {
         tcdfes = new ParFiniteElementSpace(*tcd->GetTSpecParFESpace(), meshcopy);
      }

      Vector local_err(meshcopy.GetNE());
      local_err = 0.;
      real_t threshold = std::numeric_limits<float>::max();
      meshcopy.DerefineByError(local_err, threshold, 0, 1);

      if (meshcopy.GetGlobalNE() == pmesh->GetGlobalNE())
      {
         delete tcdfes;
         return false;
      }

      if (tcd)
      {
         tcdfes->Update();
         tcd->SetTspecDataForDerefinement(tcdfes);
      }

      Vector coarse_energy(meshcopy.GetNE());
      GetTMOPDerefinementEnergy(meshcopy, tmopi, coarse_energy);
      if (tcd) { tcd->ResetDerefinementTspecData(); }
      GetTMOPDerefinementEnergy(*pmesh, tmopi, fine_energy);

      const CoarseFineTransformations &dtrans =
         meshcopy.pncmesh->GetDerefinementTransforms();

      Table coarse_to_fine;
      dtrans.MakeCoarseToFineTable(coarse_to_fine);

      Array<int> tabrow;
      for (int pe = 0; pe < meshcopy.GetNE(); pe++)
      {
         coarse_to_fine.GetRow(pe, tabrow);
         int nchild = tabrow.Size();
         real_t parent_energy = coarse_energy(pe);
         for (int fe = 0; fe < nchild; fe++)
         {
            int child = tabrow[fe];
            MFEM_VERIFY(child < pmesh->GetNE(), " invalid coarse to fine mapping");
            fine_energy(child) -= parent_energy;
         }
      }
      delete tcdfes;
#endif
   }

   // error_estimate(e) = energy(parent_of_e)-energy(e)
   // Negative energy means derefinement is desirable.
   fine_energy *= -1;
   return true;
}

void TMOPDeRefinerEstimator::ComputeEstimates()
{
   Array<NonlinearFormIntegrator*> &integs = *(nlf->GetDNFI());
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   error_estimates.SetSize(mesh->GetNE());
   error_estimates = 0.;
   Vector fine_energy(mesh->GetNE());

   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         bool deref = GetDerefineEnergyForIntegrator(*ti, fine_energy);
         if (!deref) { error_estimates = 1; return; }
         error_estimates += fine_energy;
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            bool deref = GetDerefineEnergyForIntegrator(*ati[j], fine_energy);
            if (!deref) { error_estimates = 1; return; }
            error_estimates += fine_energy;
         }
      }
   }
}

void TMOPDeRefinerEstimator::GetTMOPDerefinementEnergy(Mesh &cmesh,
                                                       TMOP_Integrator &tmopi,
                                                       Vector &el_energy_vec)
{
   const int cNE = cmesh.GetNE();
   el_energy_vec.SetSize(cNE);
   const FiniteElementSpace *fespace = cmesh.GetNodalFESpace();

   GridFunction *cxdof = cmesh.GetNodes();

   Array<int> vdofs;
   Vector el_x;
   const FiniteElement *fe;
   ElementTransformation *T;

   for (int j = 0; j < cNE; j++)
   {
      fe = fespace->GetFE(j);
      fespace->GetElementVDofs(j, vdofs);
      T = cmesh.GetElementTransformation(j);
      cxdof->GetSubVector(vdofs, el_x);
      el_energy_vec(j) = tmopi.GetDerefinementElementEnergy(*fe, *T, el_x);
   }
}


TMOPHRSolver::TMOPHRSolver(Mesh &mesh_, NonlinearForm &nlf_,
                           TMOPNewtonSolver &tmopns_, GridFunction &x_,
                           bool move_bnd_, bool hradaptivity_,
                           int mesh_poly_deg_, int amr_metric_id_,
                           int hr_iter_, int h_per_r_iter_) :
   mesh(&mesh_), nlf(&nlf_), tmopns(&tmopns_), x(&x_),
   gridfuncarr(), fespacearr(),
   move_bnd(move_bnd_), hradaptivity(hradaptivity_),
   mesh_poly_deg(mesh_poly_deg_), amr_metric_id(amr_metric_id_),
   serial(true), hr_iter(hr_iter_), h_per_r_iter(h_per_r_iter_)
{
   if (!hradaptivity) { return; }
   tmop_r_est = new TMOPRefinerEstimator(*mesh, *nlf, mesh_poly_deg,
                                         amr_metric_id);
   tmop_r = new ThresholdRefiner(*tmop_r_est);
   tmop_r->SetTotalErrorFraction(0.0);
   tmop_r_est->SetEnergyScalingFactor(1.);
   tmop_dr_est= new TMOPDeRefinerEstimator(*mesh, *nlf);
   tmop_dr = new ThresholdDerefiner(*tmop_dr_est);
   AddGridFunctionForUpdate(x);
}

#ifdef MFEM_USE_MPI
TMOPHRSolver::TMOPHRSolver(ParMesh &pmesh_, ParNonlinearForm &pnlf_,
                           TMOPNewtonSolver &tmopns_, ParGridFunction &px_,
                           bool move_bnd_, bool hradaptivity_,
                           int mesh_poly_deg_, int amr_metric_id_,
                           int hr_iter_, int h_per_r_iter_) :
   mesh(&pmesh_), nlf(&pnlf_), tmopns(&tmopns_), x(&px_),
   gridfuncarr(), fespacearr(),
   move_bnd(move_bnd_), hradaptivity(hradaptivity_),
   mesh_poly_deg(mesh_poly_deg_), amr_metric_id(amr_metric_id_),
   pmesh(&pmesh_), pnlf(&pnlf_), pgridfuncarr(), pfespacearr(),
   serial(false), hr_iter(hr_iter_), h_per_r_iter(h_per_r_iter_)
{
   if (!hradaptivity) { return; }
   tmop_r_est = new TMOPRefinerEstimator(*pmesh, *pnlf, mesh_poly_deg,
                                         amr_metric_id);
   tmop_r = new ThresholdRefiner(*tmop_r_est);
   tmop_r->SetTotalErrorFraction(0.0);
   tmop_r_est->SetEnergyScalingFactor(1.);
   tmop_dr_est= new TMOPDeRefinerEstimator(*pmesh, *pnlf);
   tmop_dr = new ThresholdDerefiner(*tmop_dr_est);
   AddGridFunctionForUpdate(&px_);
}
#endif

void TMOPHRSolver::Mult()
{
   Vector b(0);
#ifdef MFEM_USE_MPI
   int myid = 0;
#endif
   if (serial)
   {
      tmopns->SetOperator(*nlf);
   }
   else
   {
#ifdef MFEM_USE_MPI
      myid = pnlf->ParFESpace()->GetMyRank();
      tmopns->SetOperator(*pnlf);
#endif
   }
   if (!hradaptivity)
   {
      tmopns->Mult(b, x->GetTrueVector());
      x->SetFromTrueVector();
      return;
   }

   bool radaptivity = true;

   tmop_dr->Reset();
   tmop_r->Reset();

   if (serial)
   {
      for (int i_hr = 0; i_hr < hr_iter; i_hr++)
      {
         if (!radaptivity)
         {
            break;
         }
         mfem::out << i_hr << " r-adaptivity iteration.\n";

         tmopns->SetOperator(*nlf);
         tmopns->Mult(b, x->GetTrueVector());
         x->SetFromTrueVector();

         mfem::out << "TMOP energy after r-adaptivity: " <<
                   nlf->GetGridFunctionEnergy(*x)/mesh->GetNE() <<
                   ", Elements: " << mesh->GetNE() << std::endl;

         for (int i_h = 0; i_h < h_per_r_iter; i_h++)
         {
            // Derefinement step.
            if (mesh->ncmesh)
            {
               tmop_dr->Apply(*mesh);
               Update();
            }
            mfem::out << "TMOP energy after derefinement: " <<
                      nlf->GetGridFunctionEnergy(*x)/mesh->GetNE() <<
                      ", Elements: " << mesh->GetNE() << std::endl;

            // Refinement step.
            tmop_r->Apply(*mesh);
            Update();
            mfem::out << "TMOP energy after   refinement: " <<
                      nlf->GetGridFunctionEnergy(*x)/mesh->GetNE() <<
                      ", Elements: " << mesh->GetNE() << std::endl;

            if (!tmop_dr->Derefined() && tmop_r->Stop())
            {
               radaptivity = false;
               mfem::out << "AMR stopping criterion satisfied. Stop.\n";
               break;
            }
         } //n_h
      } //n_hr
   }
   else
   {
#ifdef MFEM_USE_MPI
      int NEGlob;
      real_t tmopenergy;
      for (int i_hr = 0; i_hr < hr_iter; i_hr++)
      {
         if (!radaptivity)
         {
            break;
         }
         if (myid == 0) { mfem::out << i_hr << " r-adaptivity iteration.\n"; }
         tmopns->SetOperator(*pnlf);
         tmopns->Mult(b, x->GetTrueVector());
         x->SetFromTrueVector();

         NEGlob = pmesh->GetGlobalNE();
         tmopenergy = pnlf->GetParGridFunctionEnergy(*x) / NEGlob;
         if (myid == 0)
         {
            mfem::out << "TMOP energy after r-adaptivity: " << tmopenergy <<
                      ", Elements: " << NEGlob << std::endl;
         }

         for (int i_h = 0; i_h < h_per_r_iter; i_h++)
         {
            // Derefinement step.
            if (pmesh->pncmesh)
            {
               RebalanceParNCMesh();
               ParUpdate();

               tmop_dr->Apply(*pmesh);
               ParUpdate();
            }
            NEGlob = pmesh->GetGlobalNE();
            tmopenergy = pnlf->GetParGridFunctionEnergy(*x) / NEGlob;
            if (myid == 0)
            {
               mfem::out << "TMOP energy after derefinement: " << tmopenergy <<
                         ", Elements: " << NEGlob << std::endl;
            }

            // Refinement step.
            tmop_r->Apply(*pmesh);
            ParUpdate();
            NEGlob = pmesh->GetGlobalNE();
            tmopenergy = pnlf->GetParGridFunctionEnergy(*x) / NEGlob;
            if (myid == 0)
            {
               mfem::out << "TMOP energy after   refinement: " << tmopenergy <<
                         ", Elements: " << NEGlob << std::endl;
            }

            if (!tmop_dr->Derefined() && tmop_r->Stop())
            {
               radaptivity = false;
               if (myid == 0)
               {
                  mfem::out << "AMR stopping criterion satisfied. Stop.\n";
               }
               break;
            }
         } // n_r limit
      } // n_hr
#endif
   }
}

#ifdef MFEM_USE_MPI
void TMOPHRSolver::RebalanceParNCMesh()
{
   ParNCMesh *pncmesh = pmesh->pncmesh;
   if (pncmesh)
   {
      const Table &dreftable = pncmesh->GetDerefinementTable();
      Array<int> drefs, new_ranks;
      for (int i = 0; i < dreftable.Size(); i++)
      {
         drefs.Append(i);
      }
      pncmesh->GetFineToCoarsePartitioning(drefs, new_ranks);
      pmesh->Rebalance(new_ranks);
   }
}
#endif

void TMOPHRSolver::Update()
{
   // Update FESpace
   for (int i = 0; i < fespacearr.Size(); i++)
   {
      fespacearr[i]->Update();
   }
   // Update nodal GF
   for (int i = 0; i < gridfuncarr.Size(); i++)
   {
      gridfuncarr[i]->Update();
      gridfuncarr[i]->SetTrueVector();
      gridfuncarr[i]->SetFromTrueVector();
   }

   // Update Discrete Indicator for all the TMOP_Integrators in NonLinearForm
   Array<NonlinearFormIntegrator*> &integs = *(nlf->GetDNFI());
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->UpdateAfterMeshTopologyChange();
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->UpdateAfterMeshTopologyChange(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->UpdateAfterMeshTopologyChange();
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->UpdateAfterMeshTopologyChange(); }
         }
      }
   }

   // Update the Nonlinear form and set Essential BC.
   UpdateNonlinearFormAndBC(mesh, nlf);
}

#ifdef MFEM_USE_MPI
void TMOPHRSolver::ParUpdate()
{
   // Update FESpace
   for (int i = 0; i < pfespacearr.Size(); i++)
   {
      pfespacearr[i]->Update();
   }
   // Update nodal GF
   for (int i = 0; i < pgridfuncarr.Size(); i++)
   {
      pgridfuncarr[i]->Update();
      pgridfuncarr[i]->SetTrueVector();
      pgridfuncarr[i]->SetFromTrueVector();
   }

   // Update Discrete Indicator
   Array<NonlinearFormIntegrator*> &integs = *(nlf->GetDNFI());
   TMOP_Integrator *ti  = NULL;
   TMOPComboIntegrator *co = NULL;
   DiscreteAdaptTC *dtc = NULL;
   for (int i = 0; i < integs.Size(); i++)
   {
      ti = dynamic_cast<TMOP_Integrator *>(integs[i]);
      if (ti)
      {
         ti->ParUpdateAfterMeshTopologyChange();
         dtc = ti->GetDiscreteAdaptTC();
         if (dtc) { dtc->ParUpdateAfterMeshTopologyChange(); }
      }
      co = dynamic_cast<TMOPComboIntegrator *>(integs[i]);
      if (co)
      {
         Array<TMOP_Integrator *> ati = co->GetTMOPIntegrators();
         for (int j = 0; j < ati.Size(); j++)
         {
            ati[j]->ParUpdateAfterMeshTopologyChange();
            dtc = ati[j]->GetDiscreteAdaptTC();
            if (dtc) { dtc->ParUpdateAfterMeshTopologyChange(); }
         }
      }
   }

   // Update the Nonlinear form and set Essential BC.
   UpdateNonlinearFormAndBC(pmesh, pnlf);
}
#endif

void TMOPHRSolver::UpdateNonlinearFormAndBC(Mesh *mesh_, NonlinearForm *nlf_)
{
   const FiniteElementSpace &fes = *mesh_->GetNodalFESpace();

   // Update Nonlinear form and Set Essential BC
   nlf_->Update();
   const int dim = fes.GetTypicalFE()->GetDim();
   if (move_bnd == false)
   {
      Array<int> ess_bdr(mesh_->bdr_attributes.Max());
      ess_bdr = 1;
      nlf_->SetEssentialBC(ess_bdr);
   }
   else
   {
      const int nd  = fes.GetBE(0)->GetDof();
      int n = 0;
      for (int i = 0; i < mesh_->GetNBE(); i++)
      {
         const int attr = mesh_->GetBdrElement(i)->GetAttribute();
         MFEM_VERIFY(!(dim == 2 && attr == 3),
                     "Boundary attribute 3 must be used only for 3D meshes. "
                     "Adjust the attributes (1/2/3/4 for fixed x/y/z/all "
                     "components, rest for free nodes), or use -fix-bnd.");
         if (attr == 1 || attr == 2 || attr == 3) { n += nd; }
         if (attr == 4) { n += nd * dim; }
      }
      Array<int> ess_vdofs(n), vdofs;
      n = 0;
      for (int i = 0; i < mesh_->GetNBE(); i++)
      {
         const int attr = mesh_->GetBdrElement(i)->GetAttribute();
         fes.GetBdrElementVDofs(i, vdofs);
         if (attr == 1) // Fix x components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
         else if (attr == 2) // Fix y components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+nd]; }
         }
         else if (attr == 3) // Fix z components.
         {
            for (int j = 0; j < nd; j++)
            { ess_vdofs[n++] = vdofs[j+2*nd]; }
         }
         else if (attr == 4) // Fix all components.
         {
            for (int j = 0; j < vdofs.Size(); j++)
            { ess_vdofs[n++] = vdofs[j]; }
         }
      }
      nlf_->SetEssentialVDofs(ess_vdofs);
   }
}

}
