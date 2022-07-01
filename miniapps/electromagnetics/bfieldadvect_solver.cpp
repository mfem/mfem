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

#include "bfieldadvect_solver.hpp"


//notes
//Look at DivergenceFreeProjector to pull th divergence out of a vecrot in Hcurl
// Matrix-Vector Multiplication AddMult(x,y,val):  y = y + val*A*x, Mult(x, y): y = A*x

using namespace std;

namespace mfem
{

namespace electromagnetics
{

BFieldAdvector::BFieldAdvector(ParMesh *pmesh_old, ParMesh *pmesh_new, int order_) :
   order(order_),
   pmeshOld(nullptr),
   pmeshNew(nullptr),
   H1FESpaceOld(nullptr),
   HCurlFESpaceOld(nullptr),
   HDivFESpaceOld(nullptr),
   L2FESpaceOld(nullptr),   
   H1FESpaceNew(nullptr),
   HCurlFESpaceNew(nullptr),
   HDivFESpaceNew(nullptr),
   L2FESpaceNew(nullptr),
   grad(nullptr),
   curl_old(nullptr),
   curl_new(nullptr),
   weakCurl(nullptr),
   WC(nullptr),
   m1(nullptr),
   curlCurl(nullptr),
   divFreeProj(nullptr),
   a(nullptr),
   a_new(nullptr),
   curl_b(nullptr),
   clean_curl_b(nullptr),
   recon_b(nullptr)
{
   myComm = pmesh_old->GetComm();
   SetMesh(pmesh_old, pmesh_new);
}


void BFieldAdvector::SetMesh(ParMesh *pmesh_old, ParMesh *pmesh_new)
{
   CleanInternals();

   pmeshOld = pmesh_old;
   pmeshOld->EnsureNodes();
   pmeshNew = pmesh_new;
   pmeshNew->EnsureNodes();   

   //Set up the various spaces on the meshes
   H1FESpaceOld    = new H1_ParFESpace(pmesh_old,order,pmesh_old->Dimension());
   HCurlFESpaceOld = new ND_ParFESpace(pmesh_old,order,pmesh_old->Dimension());
   HDivFESpaceOld  = new RT_ParFESpace(pmesh_old,order,pmesh_old->Dimension());
   L2FESpaceOld    = new L2_ParFESpace(pmesh_old,order,pmesh_old->Dimension());
   H1FESpaceNew    = new H1_ParFESpace(pmesh_new,order,pmesh_new->Dimension());
   HCurlFESpaceNew = new ND_ParFESpace(pmesh_new,order,pmesh_new->Dimension());
   HDivFESpaceNew  = new RT_ParFESpace(pmesh_new,order,pmesh_new->Dimension());
   L2FESpaceNew    = new L2_ParFESpace(pmesh_new,order,pmesh_new->Dimension());   

   //Discrete Differential Operators
   grad = new ParDiscreteGradOperator(H1FESpaceOld, HCurlFESpaceOld);
   grad->Assemble();
   grad->Finalize();
   curl_old = new ParDiscreteCurlOperator(HCurlFESpaceOld, HDivFESpaceOld);
   curl_old->Assemble();
   curl_old->Finalize();   
   curl_new = new ParDiscreteCurlOperator(HCurlFESpaceOld, HDivFESpaceOld);
   curl_new->Assemble();
   curl_new->Finalize();

   //Weak curl operator for taking the curl of B living in Hdiv
   ConstantCoefficient oneCoef(1.0);
   weakCurl = new ParMixedBilinearForm(HDivFESpaceOld, HCurlFESpaceOld);
   weakCurl->AddDomainIntegrator(new VectorFECurlIntegrator(oneCoef));
   weakCurl->Assemble();
   weakCurl->Finalize();   
   WC = weakCurl->ParallelAssemble();

   m1 = new ParBilinearForm(HCurlFESpaceOld);
   m1->AddDomainIntegrator(new VectorFEMassIntegrator(oneCoef));
   m1->Assemble();
   m1->Finalize();

   //CurlCurl operator
   curlCurl  = new ParBilinearForm(HCurlFESpaceOld);
   curlCurl->AddDomainIntegrator(new CurlCurlIntegrator(oneCoef));
   curlCurl->Assemble();
   curlCurl->Finalize();

   //Projector to clean the divergence out of vectors in Hcurl
   int irOrder = H1FESpaceOld->GetElementTransformation(0)->OrderW()+ 2 * order;   
   divFreeProj = new DivergenceFreeProjector(*H1FESpaceOld, *HCurlFESpaceOld,
                                              irOrder, NULL, NULL, grad);

   // Build internal grid functions on the spaces
   a  = new ParGridFunction(HCurlFESpaceOld);            //Vector potential A in HCurl
   a_new = new ParGridFunction(HCurlFESpaceNew);         //Vector potential A in Hcurl on the new mesh
   curl_b = new ParGridFunction(HCurlFESpaceOld);        //curl B in Hcurl from the weak curl
   clean_curl_b = new ParGridFunction(HCurlFESpaceOld);  //B in Hcurl
   recon_b = new ParGridFunction(HDivFESpaceOld);        //Reconstructed B from A
}


void BFieldAdvector::CleanInternals()
{
   if (H1FESpaceOld != nullptr) delete H1FESpaceOld;
   if (HCurlFESpaceOld != nullptr) delete HCurlFESpaceOld;
   if (HDivFESpaceOld != nullptr) delete HDivFESpaceOld;
   if (L2FESpaceOld != nullptr) delete L2FESpaceOld;
   if (H1FESpaceNew != nullptr) delete H1FESpaceNew;
   if (HCurlFESpaceNew != nullptr) delete HCurlFESpaceNew;
   if (HDivFESpaceNew != nullptr) delete HDivFESpaceNew;
   if (L2FESpaceNew != nullptr) delete L2FESpaceNew;   

   if (grad != nullptr) delete grad;
   if (curl_old != nullptr) delete curl_old;
   if (curl_new != nullptr) delete curl_new;

   if (weakCurl != nullptr) delete weakCurl;
   if (divFreeProj != nullptr) delete divFreeProj;
   if (curlCurl != nullptr) delete curlCurl;

   if (a != nullptr) delete a;
   if (a_new != nullptr) delete a_new;
   if (curl_b != nullptr) delete curl_b;
   if (clean_curl_b != nullptr) delete clean_curl_b;
   if (recon_b != nullptr) delete recon_b;
}


void BFieldAdvector::Advect(ParGridFunction* b_old, ParGridFunction* b_new)
{
   ComputeA(b_old);
   FindPtsInterpolateToTargetMesh(a, a_new, 1);
   curl_new->Mult(*a_new, *b_new);
}

//Solve Curl Curl A = Curl B for A using AMS
void BFieldAdvector::ComputeA(ParGridFunction* b)
{
   Array<int> ess_bdr;
   ess_bdr.SetSize(pmeshOld->bdr_attributes.Max());
   ess_bdr = 0;   // All outer surfaces
   //ess_bdr[0] = 1;
   Array<int> ess_bdr_tdofs;
   HCurlFESpaceOld->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

   //Set up a linear form with a curl operator on B
   VectorGridFunctionCoefficient b_coef(b);
   ParLinearForm rhs(HCurlFESpaceOld);
   rhs.AddDomainIntegrator(new VectorFEDomainLFCurlIntegrator(b_coef));
   rhs.Assemble();

   // Apply Dirichlet BCs to matrix and right hand side and otherwise
   // prepare the linear system
   HypreParMatrix M;
   Vector A, RHS;
   *a = 0;

   //curlCurl->FormLinearSystem(ess_bdr_tdofs, *a, *clean_curl_b, M, A, RHS);
   curlCurl->FormLinearSystem(ess_bdr_tdofs, *a, rhs, M, A, RHS);

   // Define and apply a parallel PCG solver for M A = RHS with the AMS
   // preconditioner from hypre.
   HypreAMS ams(M, HCurlFESpaceOld);
   ams.SetSingularProblem();

   HyprePCG pcg(M);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(50);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(ams);
   pcg.Mult(RHS, A);

   // Extract the parallel grid function corresponding to the finite
   // element approximation A. This is the local solution on each
   // processor.
   curlCurl->RecoverFEMSolution(A, rhs, *a);

   //Compute the reconstructed b field for comparison
   curl_old->Mult(*a, *recon_b);

   //
   Vector diff(*b);
   diff -= *recon_b;    //diff = b - recon_b
   std::cout << "L2 Error in reconstructed B field on old mesh:  " << diff.Norml2() << std::endl;

}


//Given b in Hdiv compute the curl of b_ in Hcurl
//and then clean any divergence out of it
void BFieldAdvector::ComputeCleanCurlB(ParGridFunction* b)
{
   HypreParMatrix M1;
   ParGridFunction rhs(HCurlFESpaceOld);
   Vector RHS(HCurlFESpaceOld->GetTrueVSize());
   Vector X(RHS.Size());
   Vector P(RHS.Size());

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (pmeshOld->bdr_attributes.Size())
   {
      ess_bdr.SetSize(pmeshOld->bdr_attributes.Max());
      ess_bdr = 1;
      HCurlFESpaceOld->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   b->GetTrueDofs(P);
   WC->Mult(P,RHS);
   rhs.SetFromTrueDofs(RHS);
   *curl_b = 0.0;
   m1->FormLinearSystem(ess_tdof_list, *curl_b, rhs, M1, X, RHS);
   
   HypreDiagScale Jacobi(M1);
   HyprePCG pcg(M1);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(1000);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(Jacobi);
   X = 0.0;
   pcg.Mult(RHS, X);

   m1->RecoverFEMSolution(X, rhs, *curl_b);

   divFreeProj->Mult(*curl_b, *clean_curl_b);
}


void BFieldAdvector::FindPtsInterpolateToTargetMesh(const ParGridFunction *old_gf, ParGridFunction *new_gf, int fieldtype)
{
   MFEM_ASSERT(fieldtype >= 0 && fieldtype <= 3, "Method expects a field type of 0, 1, 2, or 3");

   int dim = pmeshOld->Dimension();
   int vdim = old_gf->VectorDim();
   int num_target_elem = pmeshOld->GetNE();
   ParFiniteElementSpace *target_fes = new_gf->ParFESpace();

   // Loop through the elements in case we have a mixed mesh
   int num_target_pts = 0;
   for (int e = 0; e < num_target_elem; ++e)
   {
      num_target_pts += target_fes->GetFE(e)->GetNodes().GetNPoints();
   }

   //Extract the target points from the nodes of the elements of the 
   //new mesh and then line them up in a vector V(x1,x2,...,y1,y2...,z1,z2...)
   Vector vxyz(num_target_pts*dim);
   int vxyz_pos = 0;
   for (int e = 0; e < num_target_elem; e++)
   {
      const FiniteElement *fe = target_fes->GetFE(e);
      const IntegrationRule ir = fe->GetNodes();
      int elem_num_nodes = fe->GetNodes().GetNPoints();
      ElementTransformation *trans = target_fes->GetElementTransformation(e);

      DenseMatrix pos;
      trans->Transform(ir, pos);
      Vector rowx(vxyz.GetData() + vxyz_pos, elem_num_nodes),
             rowy(vxyz.GetData() + num_target_pts + vxyz_pos, elem_num_nodes),
             rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(vxyz.GetData() + 2*num_target_pts + vxyz_pos, elem_num_nodes);
      }
      pos.GetRow(0, rowx);
      pos.GetRow(1, rowy);
      if (dim == 3) { pos.GetRow(2, rowz); }
      vxyz_pos += elem_num_nodes;
   }

   // Interpolate the values at the new_gf nodes
   Vector interp_vals(num_target_pts*vdim);
   FindPointsGSLIB finder(myComm);
   finder.Setup(*pmeshOld);
   finder.Interpolate(vxyz, *old_gf, interp_vals);


   //I'll need to integrat this in and think about the differences between
   //The H1/L2 versions and the ND/RT versions.
   //It may be a good idea to have isH1, isL2, isND, and isRT methods in the FEC
   // Project the interpolated values to the target FiniteElementSpace.
   if (fieldtype == 0 || fieldtype == 3) // H1 or L2
   {
      if ((fieldtype == 0) || fieldtype == 3)
      {
         (*new_gf) = interp_vals;
      }
      else // H1 - but mesh order != GridFunction order
      {
         Array<int> vdofs;
         //Vector vals;
         int ivals_pos = 0;
         for (int e = 0; e < num_target_elem; e++)
         {
            const FiniteElement *fe = target_fes->GetFE(e);
            int elem_num_nodes = fe->GetNodes().GetNPoints();
            Vector elem_dof_vals(elem_num_nodes*vdim);

            target_fes->GetElementVDofs(e, vdofs);
            //vals.SetSize(vdofs.Size());
            for (int j = 0; j < elem_num_nodes; j++)
            {
               for (int d = 0; d < vdim; d++)
               {
                  // Arrange values byNodes
                  elem_dof_vals(j+d*elem_num_nodes) = interp_vals(d*num_target_pts + ivals_pos + j);
               }
            }
            new_gf->SetSubVector(vdofs, elem_dof_vals);
            ivals_pos += elem_num_nodes;
         }
      }
   }
   else // H(div) or H(curl)
   {
      std::cout << "Writing data into H(curl)/H(Div) dofs" <<std::endl;
      Array<int> vdofs;
      Vector vals;
      int ivals_pos = 0;
      for (int e = 0; e < num_target_elem; e++)
      {
         const FiniteElement *fe = target_fes->GetFE(e);
         int elem_num_nodes = fe->GetNodes().GetNPoints();
         Vector elem_dof_vals(elem_num_nodes*vdim);
         target_fes->GetElementVDofs(e, vdofs);
         vals.SetSize(vdofs.Size());
         for (int j = 0; j < elem_num_nodes; j++)
         {
            for (int d = 0; d < vdim; d++)
            {
               // Arrange values byVDim
               elem_dof_vals(j*vdim+d) = interp_vals(d*num_target_pts + ivals_pos + j);
            }
         }
         fe->ProjectFromNodes(elem_dof_vals,
                              *target_fes->GetElementTransformation(e),
                              vals);
         new_gf->SetSubVector(vdofs, vals);
         ivals_pos += elem_num_nodes;
      }
   }
}


void BFieldAdvector::RemhosRemap(ParGridFunction* b_old, ParGridFunction* b_new)
{
#if 1
   const int NE  = pmeshOld->GetNE();
   const int mesh_order = pmeshOld->GetNodalFESpace()->GetMaxElementOrder();
   const int dim = pmeshOld->Dimension();
   const bool verify_bounds = false;
   const bool forced_bounds = false;
   const int problem_num = 11;
   const int vis_steps = 10;
   const int bounds_type = 0;
   double t_final = 1.0;


   FiniteElementCollection *mesh_fec = new H1_FECollection(mesh_order, dim, BasisType::GaussLobatto);

   // Define the ODE solver used for time integration.
   ODESolver *ode_solver = new RK3SSPSolver;
   

   // Current mesh positions.
   ParFiniteElementSpace mesh_pfes(pmeshOld, mesh_fec, dim);
   ParGridFunction x(&mesh_pfes);
   pmeshOld->SetNodalGridFunction(&x);

   ParFiniteElementSpace mesh_pfes_new(pmeshNew, mesh_fec, dim);
   ParGridFunction x_new(&mesh_pfes);
   pmeshOld->SetNodalGridFunction(&x_new);   

   // Store initial mesh positions.
   Vector x0(x.Size());
   x0 = x;

   // Initial time step estimate.
   // Since we are talking about a local change lets do a fixed number of steps
   // TODO:  Ask Vladimir about this assumption 
   double dt = 0.25;

   // Mesh velocity.
   // Obtain the mesh velocity by moving the mesh to the final
   // mesh positions, and taking the displacement vector.
   // The mesh motion resembles a time-dependent deformation, e.g., similar to
   // a deformation that is obtained by a Lagrangian simulation.
   GridFunction v_gf(x.FESpace());
   VectorGridFunctionCoefficient v_mesh_coeff(&v_gf);
   ParGridFunction v(&mesh_pfes);
   v = x_new - x;
   double t = 0.0;
   while (t < t_final)
   {
      t += dt;
      // Move the mesh nodes.
      x.Add(std::min(dt, t_final-t), v);
      //No need to update v here since the pseudo velocity doesn't change
      //During this remap step
      //TODO:  For higher order we will need a velocity function that can 
      //change over the course of the pseudo-timestep
   }
   add(x, -1.0, x0, v_gf); // Pseudotime velocity.
   x = x0; // Return the mesh to the initial configuration.

   const int btype = BasisType::Positive;
   DG_FECollection fec(order, dim, btype);
   ParFiniteElementSpace pfes(pmeshOld, &fec);

   ParGridFunction inflow_gf(&pfes);
   inflow_gf = 0.0;

   // Set up the bilinear and linear forms corresponding to the DG
   // discretization.
   ParBilinearForm m(&pfes);
   m.AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm M_HO(&pfes);
   M_HO.AddDomainIntegrator(new MassIntegrator);

   ParBilinearForm k(&pfes);
   ParBilinearForm K_HO(&pfes);
   k.AddDomainIntegrator(new ConvectionIntegrator(v_mesh_coeff));
   K_HO.AddDomainIntegrator(new ConvectionIntegrator(v_mesh_coeff));

   auto dgt_i = new DGTraceIntegrator(v_mesh_coeff, -1.0, -0.5);
   auto dgt_b = new DGTraceIntegrator(v_mesh_coeff, -1.0, -0.5);
   K_HO.AddInteriorFaceIntegrator(new TransposeIntegrator(dgt_i));
   K_HO.AddBdrFaceIntegrator(new TransposeIntegrator(dgt_b));
   K_HO.KeepNbrBlock(true);

   K_HO.SetAssemblyLevel(AssemblyLevel::FULL);
   M_HO.Assemble();
   K_HO.Assemble(0);
   M_HO.Finalize();
   K_HO.Finalize(0);

   // Compute the lumped mass matrix.
   Vector lumpedM;
   ParBilinearForm ml(&pfes);
   ml.AddDomainIntegrator(new LumpedIntegrator(new MassIntegrator));
   ml.Assemble();
   ml.Finalize();
   ml.SpMat().GetDiag(lumpedM);

   m.Assemble();
   m.Finalize();
   int skip_zeros = 0;
   k.Assemble(skip_zeros);
   k.Finalize(skip_zeros);

   // Store topological dof data.
   DofInfo dofs(pfes, bounds_type);

   // Precompute data required for high and low order schemes. This could be put
   // into a separate routine. I am using a struct now because the various
   // schemes require quite different information.
   LowOrderMethod lom;
   lom.subcell_scheme = false;

   lom.pk = NULL;
   lom.smap = SparseMatrix_Build_smap(k.SpMat());
   lom.D = k.SpMat();
   lom.coef = &v_mesh_coeff;

   // Face integration rule.
   const FaceElementTransformations *ft =
      pmeshOld->GetFaceElementTransformations(0);
   const int el_order = pfes.GetFE(0)->GetOrder();
   int ft_order = ft->Elem1->OrderW() + 2 * el_order;
   if (pfes.GetFE(0)->Space() == FunctionSpace::Pk) { ft_order++; }
   lom.irF = &IntRules.Get(ft->FaceGeom, ft_order);

   DG_FECollection fec0(0, dim, btype);
   DG_FECollection fec1(1, dim, btype);

   ParMesh *subcell_mesh = NULL;
   lom.SubFes0 = NULL;
   lom.SubFes1 = NULL;
   FiniteElementCollection *fec_sub = NULL;
   ParFiniteElementSpace *pfes_sub = NULL;;
   ParGridFunction *xsub = NULL;
   ParGridFunction v_sub_gf;
   VectorGridFunctionCoefficient v_sub_coef;
   Vector x0_sub;

   if (order > 1)
   {
      // The mesh corresponding to Bezier subcells of order p is constructed.
      // NOTE: The mesh is assumed to consist of quads or hexes.
      MFEM_VERIFY(order > 1, "This code should not be entered for order = 1.");

      // Get a uniformly refined mesh.
      const int btype = BasisType::ClosedUniform;
      subcell_mesh = new ParMesh(ParMesh::MakeRefined(*pmeshOld, order, btype));

      // Check if the mesh is periodic.
      const L2_FECollection *L2_coll = dynamic_cast<const L2_FECollection *>
                                       (pmeshOld->GetNodes()->FESpace()->FEColl());
      // Standard non-periodic mesh.
      // Note that the fine mesh is always linear.
      fec_sub = new H1_FECollection(1, dim, BasisType::ClosedUniform);
      pfes_sub = new ParFiniteElementSpace(subcell_mesh, fec_sub, dim);
      xsub = new ParGridFunction(pfes_sub);
      subcell_mesh->SetCurvature(1);
      subcell_mesh->SetNodalGridFunction(xsub);
 
      lom.SubFes0 = new FiniteElementSpace(subcell_mesh, &fec0);
      lom.SubFes1 = new FiniteElementSpace(subcell_mesh, &fec1);

      // Submesh velocity.
      v_sub_gf.SetSpace(pfes_sub);
      v_sub_gf.ProjectCoefficient(v_mesh_coeff);

      // Zero it out on boundaries (not moving boundaries).
      Array<int> ess_bdr, ess_vdofs;
      if (subcell_mesh->bdr_attributes.Size() > 0)
      {
         ess_bdr.SetSize(subcell_mesh->bdr_attributes.Max());
      }
      ess_bdr = 1;
      xsub->ParFESpace()->GetEssentialVDofs(ess_bdr, ess_vdofs);
      for (int i = 0; i < ess_vdofs.Size(); i++)
      {
         if (ess_vdofs[i] == -1) { v_sub_gf(i) = 0.0; }
      }
      v_sub_coef.SetGridFunction(&v_sub_gf);

      // Store initial submesh positions.
      x0_sub = *xsub;

      lom.subcellCoeff = &v_sub_coef;
      lom.VolumeTerms = new MixedConvectionIntegrator(v_sub_coef);
   } else { subcell_mesh = pmeshOld; }

   Assembly asmbl(dofs, lom, inflow_gf, pfes, subcell_mesh, 1);

   // Setup the initial conditions.
   const int vsize = pfes.GetVSize();
   Array<int> offset(2);      //2 because we are assuming product_sync = 0
   for (int i = 0; i < offset.Size(); i++) { offset[i] = i*vsize; }
   BlockVector S(offset, Device::GetMemoryType());
   // Primary scalar field is u.
   ParGridFunction u(&pfes);
   u.MakeRef(&pfes, S, offset[0]);
   u = *b_old;    //Set u to the pre remap b state
   u.SyncAliasMemory(S);

   //No Product sync

   //No Smoothness indicator
   SmoothnessIndicator *smth_indicator = NULL;

   // Setup of the high-order solver
   HOSolver *ho_solver = new LocalInverseHOSolver(pfes, M_HO, K_HO);

   // Setup the low order solver
   const bool time_dep = true;
   Array<int> lo_smap = SparseMatrix_Build_smap(k.SpMat());
   LOSolver *lo_solver = new DiscreteUpwind(pfes, k.SpMat(), lo_smap,
                                            lumpedM, asmbl, time_dep);

   // Setup of the FCT solver.
   Array<int> K_HO_smap;
   FCTSolver *fct_solver = NULL;
   K_HO.SpMat().HostReadI();
   K_HO.SpMat().HostReadJ();
   K_HO.SpMat().HostReadData();
   K_HO_smap = SparseMatrix_Build_smap(K_HO.SpMat());
   const int fct_iterations = 1;
   fct_solver = new FluxBasedFCT(pfes, smth_indicator, dt, K_HO.SpMat(),
                                 K_HO_smap, M_HO.SpMat(), fct_iterations);

   AdvectionOperator adv(S.Size(), m, ml, lumpedM, k, M_HO, K_HO,
                         x, xsub, v_gf, v_sub_gf, asmbl, lom, dofs,
                         ho_solver, lo_solver, fct_solver);

   t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   double umin, umax;
   GetMinMax(u, umin, umax);

   adv.SetRemapStartPos(x0, x0_sub);
   

   ParGridFunction res = u;
   double residual;
   double s_min_glob = numeric_limits<double>::infinity(),
          s_max_glob = -numeric_limits<double>::infinity();

   // Time-integration (loop over the time iterations, ti, with a time-step dt).
   bool done = false;
   BlockVector Sold(S);
   int ti_total = 0, ti = 0;
   while (done == false)
   {
      double dt_real = min(dt, t_final - t);

      // This also resets the time step estimate when automatic dt is on.
      adv.SetDt(dt_real);
      if (lo_solver)  { lo_solver->UpdateTimeStep(dt_real); }
      if (fct_solver) { fct_solver->UpdateTimeStep(dt_real); }

      Sold = S;
      ode_solver->Step(S, t, dt_real);
      ti++;
      ti_total++;

      //Time step control is fixed

      // S has been modified, update the alias
      u.SyncMemory(S);

      // Monotonicity check for debug purposes mainly.
      if (verify_bounds && forced_bounds && smth_indicator == NULL)
      {
         double umin_new, umax_new;
         GetMinMax(u, umin_new, umax_new);
         if (problem_num % 10 != 6 && problem_num % 10 != 7)
         {
            if (pmeshOld->GetMyRank() == 0)
            {
               MFEM_VERIFY(umin_new > umin - 1e-12,
                           "Undershoot of " << umin - umin_new);
               MFEM_VERIFY(umax_new < umax + 1e-12,
                           "Overshoot of " << umax_new - umax);
            }
            umin = umin_new;
            umax = umax_new;
         }
         else
         {
            if (pmeshOld->GetMyRank() == 0)
            {
               MFEM_VERIFY(umin_new > 0.0 - 1e-12,
                           "Undershoot of " << 0.0 - umin_new);
               MFEM_VERIFY(umax_new < 1.0 + 1e-12,
                           "Overshoot of " << umax_new - 1.0);
            }
         }
      }

      x0.HostReadWrite(); v_sub_gf.HostReadWrite();
      x.HostReadWrite();
      add(x0, t, v_gf, x);
      x0_sub.HostReadWrite(); v_sub_gf.HostReadWrite();
      MFEM_VERIFY(xsub != NULL,
                  "xsub == NULL/This code should not be entered for order = 1.");
      xsub->HostReadWrite();
      add(x0_sub, t, v_sub_gf, *xsub);

      done = (t >= t_final - 1.e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         if (pmeshOld->GetMyRank() == 0)
         {
            std::cout << "time step: " << ti << ", time: " << t
                 << ", dt: " << dt << ", residual: " << residual << std::endl;
         }
      }
   }

   std::cout << "Total time steps: " << ti_total
        << " (" << ti_total-ti << " repeated)." << std::endl;

   *b_new = u;
#endif
}




AdvectionOperator::AdvectionOperator(int size, BilinearForm &Mbf_,
                                     BilinearForm &_ml, Vector &_lumpedM,
                                     ParBilinearForm &Kbf_,
                                     ParBilinearForm &M_HO_, ParBilinearForm &K_HO_,
                                     GridFunction &pos, GridFunction *sub_pos,
                                     GridFunction &vel, GridFunction &sub_vel,
                                     Assembly &_asmbl,
                                     LowOrderMethod &_lom, DofInfo &_dofs,
                                     HOSolver *hos, LOSolver *los, FCTSolver *fct) :
   TimeDependentOperator(size), Mbf(Mbf_), ml(_ml), Kbf(Kbf_),
   M_HO(M_HO_), K_HO(K_HO_),
   lumpedM(_lumpedM),
   start_mesh_pos(pos.Size()), start_submesh_pos(sub_vel.Size()),
   mesh_pos(pos), submesh_pos(sub_pos),
   mesh_vel(vel), submesh_vel(sub_vel),
   x_gf(Kbf.ParFESpace()),
   asmbl(_asmbl), lom(_lom), dofs(_dofs),
   ho_solver(hos), lo_solver(los), fct_solver(fct)
{ 
   MFEM_VERIFY(fct_solver && ho_solver && lo_solver, "Bfield Remhos advector requires FCT, ho and lo solvers.");
}

void AdvectionOperator::Mult(const Vector &X, Vector &Y) const
{
   MFEM_VERIFY(ho_solver && lo_solver, "FCT requires HO and LO solvers.");

   // Move the mesh positions.
   const double t = GetTime();
   add(start_mesh_pos, t, mesh_vel, mesh_pos);
   if (submesh_pos)
   {
      add(start_submesh_pos, t, submesh_vel, *submesh_pos);
   }
   // Reset precomputed geometric data.
   Mbf.FESpace()->GetMesh()->DeleteGeometricFactors();

   // Reassemble on the new mesh. Element contributions.
   // Currently needed to have the sparse matrices used by the LO methods.
   Mbf.BilinearForm::operator=(0.0);
   Mbf.Assemble();
   Kbf.BilinearForm::operator=(0.0);
   Kbf.Assemble(0);
   ml.BilinearForm::operator=(0.0);
   ml.Assemble();
   lumpedM.HostReadWrite();
   ml.SpMat().GetDiag(lumpedM);

   M_HO.BilinearForm::operator=(0.0);
   M_HO.Assemble();
   K_HO.BilinearForm::operator=(0.0);
   K_HO.Assemble(0);

   if (lom.pk)
   {
      lom.pk->BilinearForm::operator=(0.0);
      lom.pk->Assemble();
   }

   // Face contributions.
   asmbl.bdrInt = 0.;
   Mesh *mesh = M_HO.FESpace()->GetMesh();
   const int dim = mesh->Dimension(), ne = mesh->GetNE();
   Array<int> bdrs, orientation;
   FaceElementTransformations *Trans;

   for (int k = 0; k < ne; k++)
   {
      if (dim == 1)      { mesh->GetElementVertices(k, bdrs); }
      else if (dim == 2) { mesh->GetElementEdges(k, bdrs, orientation); }
      else if (dim == 3) { mesh->GetElementFaces(k, bdrs, orientation); }

      for (int i = 0; i < dofs.numBdrs; i++)
      {
         Trans = mesh->GetFaceElementTransformations(bdrs[i]);
         asmbl.ComputeFluxTerms(k, i, Trans, lom);
      }
   }

   const int size = Kbf.ParFESpace()->GetVSize();
   const int NE   = Kbf.ParFESpace()->GetNE();

   // Needed because X and Y are allocated on the host by the ODESolver.
   X.Read(); Y.Read();

   Vector u, d_u;
   Vector* xptr = const_cast<Vector*>(&X);
   u.MakeRef(*xptr, 0, size);
   d_u.MakeRef(Y, 0, size);
   Vector du_HO(u.Size()), du_LO(u.Size());

   x_gf = u;
   x_gf.ExchangeFaceNbrData();

   if (fct_solver)
   {
      MFEM_VERIFY(ho_solver && lo_solver, "FCT requires HO and LO solvers.");

      lo_solver->CalcLOSolution(u, du_LO);
      ho_solver->CalcHOSolution(u, du_HO);

      dofs.ComputeElementsMinMax(u, dofs.xe_min, dofs.xe_max, NULL, NULL);
      dofs.ComputeBounds(dofs.xe_min, dofs.xe_max, dofs.xi_min, dofs.xi_max);
      fct_solver->CalcFCTSolution(x_gf, lumpedM, du_HO, du_LO,
                                  dofs.xi_min, dofs.xi_max, d_u);
   }

   d_u.SyncAliasMemory(Y);

   // Remap the product field, if there is a product field.
   if (X.Size() > size)
   {
      Vector us, d_us;
      us.MakeRef(*xptr, size, size);
      d_us.MakeRef(Y, size, size);

      x_gf = us;
      x_gf.ExchangeFaceNbrData();

      if (fct_solver)
      {
         MFEM_VERIFY(ho_solver && lo_solver, "FCT requires HO and LO solvers.");

         Vector d_us_HO(us.Size()), d_us_LO;
         if (fct_solver->NeedsLOProductInput())
         {
            d_us_LO.SetSize(us.Size());
            lo_solver->CalcLOSolution(us, d_us_LO);
         }
         ho_solver->CalcHOSolution(us, d_us_HO);

         // Compute the ratio s = us_old / u_old, and old active dofs.
         Vector s(size);
         Array<bool> s_bool_el, s_bool_dofs;
         ComputeRatio(NE, us, u, s, s_bool_el, s_bool_dofs);

         // Bounds for s, based on the old values (and old active dofs).
         // This doesn't consider s values from the old inactive dofs, because
         // there were no bounds restriction on them at the previous time step.
         dofs.ComputeElementsMinMax(s, dofs.xe_min, dofs.xe_max,
                                    &s_bool_el, &s_bool_dofs);
         dofs.ComputeBounds(dofs.xe_min, dofs.xe_max,
                            dofs.xi_min, dofs.xi_max, &s_bool_el);

         // Evolve u and get the new active dofs.
         Vector u_new(size);
         add(1.0, u, dt, d_u, u_new);
         Array<bool> s_bool_el_new, s_bool_dofs_new;
         ComputeBoolIndicators(NE, u_new, s_bool_el_new, s_bool_dofs_new);

         fct_solver->CalcFCTProduct(x_gf, lumpedM, d_us_HO, d_us_LO,
                                    dofs.xi_min, dofs.xi_max,
                                    u_new,
                                    s_bool_el_new, s_bool_dofs_new, d_us);
      }

      d_us.SyncAliasMemory(Y);
   }
}

} // namespace electromagnetics
} // namespace mfem