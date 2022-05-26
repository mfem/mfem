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

namespace mfem
{

namespace electromagnetics
{

BFieldAdvector::BFieldAdvector(ParMesh *pmesh_old, ParMesh *pmesh_new, int order_) :
   order(order_),
   pmeshOld(nullptr),
   pmeshNew(nullptr),
   H1FESpaceOld(nullptr),
   H1FESpaceNew(nullptr),
   HCurlFESpaceOld(nullptr),
   HCurlFESpaceNew(nullptr),
   HDivFESpaceOld(nullptr),
   HDivFESpaceNew(nullptr),
   L2FESpaceOld(nullptr),
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
   MFEM_ASSERT(pmesh_old->GetComm() == pmesh_new->GetComm(), "The old and new mesh must be in the same MPI COMMM");
   myComm = pmesh_old->GetComm();
   SetMeshes(pmesh_old, pmesh_new);
}


void BFieldAdvector::SetMeshes(ParMesh *pmesh_old, ParMesh *pmesh_new)
{
   CleanInternals();

   pmeshOld = pmesh_old;
   pmeshNew = pmesh_new;

   pmeshOld->EnsureNodes();
   pmeshNew->EnsureNodes();

   if (pmeshNew->GetNodes() == NULL) { pmeshNew->SetCurvature(1); }
   pmeshNewOrder = pmeshNew->GetNodes()->FESpace()->GetElementOrder(0);

   //Set up the various spaces on the meshes
   H1FESpaceOld    = new H1_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   HCurlFESpaceOld = new ND_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   HDivFESpaceOld  = new RT_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   L2FESpaceOld    = new L2_ParFESpace(pmeshOld,order,pmeshOld->Dimension());
   H1FESpaceNew    = new H1_ParFESpace(pmeshNew,order,pmeshNew->Dimension());
   HCurlFESpaceNew = new ND_ParFESpace(pmeshNew,order,pmeshNew->Dimension());
   HDivFESpaceNew  = new RT_ParFESpace(pmeshNew,order,pmeshNew->Dimension());
   L2FESpaceNew    = new L2_ParFESpace(pmeshNew,order,pmeshNew->Dimension());

   //Discrete Differential Operators
   grad = new ParDiscreteGradOperator(H1FESpaceOld, HCurlFESpaceOld);
   grad->Assemble();
   grad->Finalize();
   curl_old = new ParDiscreteCurlOperator(HCurlFESpaceOld, HDivFESpaceOld);
   curl_old->Assemble();
   curl_old->Finalize();   
   curl_new = new ParDiscreteCurlOperator(HCurlFESpaceNew, HDivFESpaceNew);
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

   int dim = pmeshNew->Dimension();
   int vdim = old_gf->VectorDim();
   int num_target_elem = pmeshNew->GetNE();
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
      if ((fieldtype == 0 && order == pmeshNewOrder) || fieldtype == 3)
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


} // namespace electromagnetics
} // namespace mfem