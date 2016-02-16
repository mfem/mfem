// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "volta_solver.hpp"

using namespace std;
namespace mfem
{
using namespace miniapps;

namespace electromagnetics
{

VoltaSolver::VoltaSolver(ParMesh & pmesh, int order,
                         Array<int> & dbcs, Vector & dbcv,
                         Array<int> & nbcs, Vector & nbcv,
                         double (*eps    )(const Vector&),
                         double (*phi_bc )(const Vector&),
                         double (*rho_src)(const Vector&),
                         void   (*p_src  )(const Vector&, Vector&))
   : myid_(0),
     num_procs_(1),
     order_(order),
     pmesh_(&pmesh),
     dbcs_(&dbcs),
     dbcv_(&dbcv),
     nbcs_(&nbcs),
     nbcv_(&nbcv),
     visit_dc_(NULL),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     divEpsGrad_(NULL),
     h1Mass_(NULL),
     h1SurfMass_(NULL),
     hCurlMass_(NULL),
     hDivMass_(NULL),
     hCurlHDivEps_(NULL),
     hCurlHDiv_(NULL),
     Grad_(NULL),
     phi_(NULL),
     rho_(NULL),
     sigma_(NULL),
     e_(NULL),
     d_(NULL),
     p_(NULL),
     epsCoef_(NULL),
     phiBCCoef_(NULL),
     rhoCoef_(NULL),
     pCoef_(NULL),
     eps_(eps),
     phi_bc_(phi_bc),
     rho_src_(rho_src),
     p_src_(p_src)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());

   // Select surface attributes for Dirichlet BCs
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr_ = 0;   // Deselect all outer surfaces
   for (int i=0; i<dbcs_->Size(); i++)
   {
      ess_bdr_[(*dbcs_)[i]-1] = 1;
   }

   // Setup various coefficients

   // Potential on outer surface
   if ( phi_bc_ != NULL )
   {
      phiBCCoef_ = new FunctionCoefficient(*phi_bc_);
   }

   // Permittivity Coefficient
   if ( eps_ == NULL )
   {
      epsCoef_ = new ConstantCoefficient(epsilon0_);
   }
   else
   {
      epsCoef_ = new FunctionCoefficient(eps_);
   }

   // Volume Charge Density
   if ( rho_src_ != NULL )
   {
      rhoCoef_ = new FunctionCoefficient(rho_src_);
   }

   // Polarization
   if ( p_src_ != NULL )
   {
      pCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             p_src_);
   }

   // Bilinear Forms
   divEpsGrad_  = new ParBilinearForm(H1FESpace_);
   divEpsGrad_->AddDomainIntegrator(new DiffusionIntegrator(*epsCoef_));

   hDivMass_ = new ParBilinearForm(HDivFESpace_);
   hDivMass_->AddDomainIntegrator(new VectorFEMassIntegrator);

   hCurlHDivEps_ = new ParMixedBilinearForm(HCurlFESpace_,HDivFESpace_);
   hCurlHDivEps_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsCoef_));

   // Assemble Matrices
   divEpsGrad_->Assemble();
   divEpsGrad_->Finalize();

   hDivMass_->Assemble();
   hDivMass_->Finalize();

   hCurlHDivEps_->Assemble();
   hCurlHDivEps_->Finalize();

   // Discrete Grad operator
   Grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);

   // Build grid functions
   phi_ = new ParGridFunction(H1FESpace_);
   d_   = new ParGridFunction(HDivFESpace_);
   e_   = new ParGridFunction(HCurlFESpace_);

   if ( rho_src_ )
   {
      rho_ = new ParGridFunction(H1FESpace_);

      h1Mass_ = new ParBilinearForm(H1FESpace_);
      h1Mass_->AddDomainIntegrator(new MassIntegrator);
      h1Mass_->Assemble();
      h1Mass_->Finalize();
   }

   if ( p_src_ )
   {
      p_ = new ParGridFunction(HCurlFESpace_);

      hCurlMass_  = new ParBilinearForm(HCurlFESpace_);
      hCurlMass_->AddDomainIntegrator(new VectorFEMassIntegrator);
      hCurlMass_->Assemble();
      hCurlMass_->Finalize();

      hCurlHDiv_ = new ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_);
      hCurlHDiv_->AddDomainIntegrator(new VectorFEMassIntegrator);
      hCurlHDiv_->Assemble();
      hCurlHDiv_->Finalize();
   }

   if ( nbcs_->Size() > 0 )
   {
      sigma_ = new ParGridFunction(H1FESpace_);

      h1SurfMass_  = new ParBilinearForm(H1FESpace_);
      h1SurfMass_->AddBoundaryIntegrator(new MassIntegrator);
      h1SurfMass_->Assemble();
      h1SurfMass_->Finalize();
   }
}

VoltaSolver::~VoltaSolver()
{
   delete epsCoef_;
   delete phiBCCoef_;
   delete rhoCoef_;
   delete pCoef_;

   delete phi_;
   delete rho_;
   delete sigma_;
   delete d_;
   delete e_;
   delete p_;

   delete Grad_;

   delete divEpsGrad_;
   delete h1Mass_;
   delete h1SurfMass_;
   delete hCurlMass_;
   delete hDivMass_;
   delete hCurlHDivEps_;
   delete hCurlHDiv_;

   delete H1FESpace_;
   delete HCurlFESpace_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
VoltaSolver::GetProblemSize()
{
   return H1FESpace_->GlobalTrueVSize();
}

void
VoltaSolver::PrintSizes()
{
   HYPRE_Int size_h1 = H1FESpace_->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }
}

void
VoltaSolver::Update()
{
   if (myid_ == 0) { cout << " Assembly ... " << flush; }

   // Inform the spaces that the mesh has changed
   H1FESpace_->Update();
   HCurlFESpace_->Update();
   HDivFESpace_->Update();

   // Inform the grid functions that the space has changed.
   phi_->Update();
   d_->Update();
   e_->Update();
   if ( rho_   ) { rho_->Update(); }
   if ( sigma_ ) { sigma_->Update(); }
   if ( p_     ) { p_->Update(); }

   // Inform the bilinear forms that the space has changed.
   divEpsGrad_->Update();
   divEpsGrad_->Assemble();
   divEpsGrad_->Finalize();

   hDivMass_->Update();
   hDivMass_->Assemble();
   hDivMass_->Finalize();

   hCurlHDivEps_->Update();
   hCurlHDivEps_->Assemble();
   hCurlHDivEps_->Finalize();

   if ( h1Mass_ )
   {
      h1Mass_->Update();
      h1Mass_->Assemble();
      h1Mass_->Finalize();
   }

   if ( h1SurfMass_ )
   {
      h1SurfMass_->Update();
      h1SurfMass_->Assemble();
      h1SurfMass_->Finalize();
   }

   if ( hCurlMass_ )
   {
      hCurlMass_->Update();
      hCurlMass_->Assemble();
      hCurlMass_->Finalize();
   }

   if ( hCurlHDiv_ )
   {
      hCurlHDiv_->Update();
      hCurlHDiv_->Assemble();
      hCurlHDiv_->Finalize();
   }

   // Inform the other objects that the space has changed.
   Grad_->Update();

   if (myid_ == 0) { cout << "done." << flush; }
}

void
VoltaSolver::Solve()
{
   if (myid_ == 0) { cout << "Running solver ... " << endl << flush; }

   // Initialize the electric potential with its boundary conditions
   *phi_ = 0.0;

   if ( dbcs_->Size() > 0 )
   {
      if ( phiBCCoef_ )
      {
         // Apply gradient boundary condition
         phi_->ProjectBdrCoefficient(*phiBCCoef_, ess_bdr_);
      }
      else
      {
         // Apply piecewise constant boundary condition
         Array<int> dbc_bdr_attr(pmesh_->bdr_attributes.Max());
         for (int i=0; i<dbcs_->Size(); i++)
         {
            ConstantCoefficient voltage((*dbcv_)[i]);
            dbc_bdr_attr = 0;
            dbc_bdr_attr[(*dbcs_)[i]-1] = 1;
            phi_->ProjectBdrCoefficient(voltage, dbc_bdr_attr);
         }
      }
   }

   // Initialize the RHS vector
   HypreParVector *RHS = new HypreParVector(H1FESpace_);
   *RHS = 0.0;

   // Initialize the volumetric charge density
   if ( rho_ )
   {
      rho_->ProjectCoefficient(*rhoCoef_);

      HypreParMatrix *MassH1 = h1Mass_->ParallelAssemble();
      HypreParVector *Rho    = rho_->ParallelProject();

      MassH1->Mult(*Rho,*RHS);

      delete MassH1;
      delete Rho;
   }

   // Initialize the Polarization
   HypreParVector *P = NULL;
   if ( p_ )
   {
      p_->ProjectCoefficient(*pCoef_);
      P = p_->ParallelProject();

      HypreParMatrix *MassHCurl = hCurlMass_->ParallelAssemble();
      HypreParVector *PD        = new HypreParVector(HCurlFESpace_);

      MassHCurl->Mult(*P,*PD);
      Grad_->MultTranspose(*PD,*RHS,-1.0,1.0);

      delete MassHCurl;
      delete PD;

   }

   // Initialize the surface charge density
   if ( sigma_ )
   {
      *sigma_ = 0.0;

      Array<int> nbc_bdr_attr(pmesh_->bdr_attributes.Max());
      for (int i=0; i<nbcs_->Size(); i++)
      {
         ConstantCoefficient sigma_coef((*nbcv_)[i]);
         nbc_bdr_attr = 0;
         nbc_bdr_attr[(*nbcs_)[i]-1] = 1;
         sigma_->ProjectBdrCoefficient(sigma_coef, nbc_bdr_attr);
      }

      HypreParMatrix *MassS = h1SurfMass_->ParallelAssemble();
      HypreParVector *Sigma = sigma_->ParallelProject();

      MassS->Mult(*Sigma,*RHS,1.0,1.0);

      delete MassS;
      delete Sigma;
   }

   // Apply Dirichlet BCs to matrix and right hand side
   HypreParMatrix *DivEpsGrad = divEpsGrad_->ParallelAssemble();
   HypreParVector *Phi        = phi_->ParallelProject();

   // Apply the boundary conditions to the assembled matrix and vectors
   if ( dbcs_->Size() > 0 )
   {
      // According to the selected surfaces
      divEpsGrad_->ParallelEliminateEssentialBC(ess_bdr_,
                                                *DivEpsGrad,
                                                *Phi, *RHS);
   }
   else
   {
      // No surfaces were labeled as Dirichlet so eliminate one DoF
      Array<int> dof_list(0);
      if ( myid_ == 0 )
      {
         dof_list.SetSize(1);
         dof_list[0] = 0;
      }
      DivEpsGrad->EliminateRowsCols(dof_list, *Phi, *RHS);
   }

   // Define and apply a parallel PCG solver for AX=B with the AMG
   // preconditioner from hypre.
   HypreSolver *amg = new HypreBoomerAMG(*DivEpsGrad);
   HyprePCG *pcg = new HyprePCG(*DivEpsGrad);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(*RHS, *Phi);

   delete amg;
   delete pcg;
   delete DivEpsGrad;
   delete RHS;

   // Extract the parallel grid function corresponding to the finite
   // element approximation Phi. This is the local solution on each
   // processor.
   *phi_ = *Phi;

   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi.
   HypreParVector *E = new HypreParVector(HCurlFESpace_);
   Grad_->Mult(*Phi,*E,-1.0);
   *e_ = *E;

   delete Phi;

   // Compute electric displacement (D) from E and P
   if (myid_ == 0) { cout << "Computing D ... " << flush; }

   HypreParMatrix *HCurlHDivEps = hCurlHDivEps_->ParallelAssemble();
   HypreParVector *ED = new HypreParVector(HDivFESpace_);
   HypreParVector *D  = new HypreParVector(HDivFESpace_);

   HCurlHDivEps->Mult(*E,*ED);

   if ( P )
   {
      HypreParMatrix *HCurlHDiv = hCurlHDiv_->ParallelAssemble();
      HCurlHDiv->Mult(*P,*ED,-1.0,1.0);
      delete HCurlHDiv;
   }

   HypreParMatrix * MassHDiv = hDivMass_->ParallelAssemble();

   HyprePCG * pcgM = new HyprePCG(*MassHDiv);
   pcgM->SetTol(1e-12);
   pcgM->SetMaxIter(500);
   pcgM->SetPrintLevel(0);
   HypreDiagScale *diagM = new HypreDiagScale;
   pcgM->SetPreconditioner(*diagM);
   pcgM->Mult(*ED,*D);

   *d_ = *D;

   if (myid_ == 0) { cout << "done." << flush; }

   delete diagM;
   delete pcgM;
   delete HCurlHDivEps;
   delete MassHDiv;
   delete E;
   delete ED;
   delete D;
   delete P;

   if (myid_ == 0) { cout << " Solver done. " << flush; }
}

void
VoltaSolver::GetErrorEstimates(Vector & errors)
{
   if (myid_ == 0) { cout << "Estimating Error ... " << flush; }

   // Space for the discontinuous (original) flux
   DiffusionIntegrator flux_integrator(*epsCoef_);
   L2_FECollection flux_fec(order_, pmesh_->Dimension());
   // ND_FECollection flux_fec(order_, pmesh_->Dimension());
   ParFiniteElementSpace flux_fes(pmesh_, &flux_fec, pmesh_->SpaceDimension());

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   RT_FECollection smooth_flux_fec(order_-1, pmesh_->Dimension());
   ParFiniteElementSpace smooth_flux_fes(pmesh_, &smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, *phi_,
                      smooth_flux_fes, flux_fes, errors, norm_p);

   if (myid_ == 0) { cout << "done." << flush; }
}

void
VoltaSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("Phi", phi_);
   visit_dc.RegisterField("D",     d_);
   visit_dc.RegisterField("E",     e_);
   if ( rho_   ) { visit_dc.RegisterField("Rho",     rho_); }
   if ( p_     ) { visit_dc.RegisterField("P",         p_); }
   if ( sigma_ ) { visit_dc.RegisterField("Sigma", sigma_); }
}

void
VoltaSolver::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " " << flush; }
   }
}

void
VoltaSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl << flush; }

   socks_["Phi"] = new socketstream;
   socks_["Phi"]->precision(8);

   socks_["D"] = new socketstream;
   socks_["D"]->precision(8);

   socks_["E"] = new socketstream;
   socks_["E"]->precision(8);

   if ( rho_)
   {
      socks_["Rho"] = new socketstream;
      socks_["Rho"]->precision(8);
   }
   if ( p_)
   {
      socks_["P"] = new socketstream;
      socks_["P"]->precision(8);
   }
   if ( sigma_)
   {
      socks_["Sigma"] = new socketstream;
      socks_["Sigma"]->precision(8);
   }
   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl << flush; }
}

void
VoltaSolver::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   VisualizeField(*socks_["Phi"], vishost, visport,
                  *phi_, "Electric Potential (Phi)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["D"], vishost, visport,
                  *d_, "Electric Displacement (D)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["E"], vishost, visport,
                  *e_, "Electric Field (E)", Wx, Wy, Ww, Wh);

   Wx = 0; Wy += offy; // next line

   if ( rho_ )
   {
      VisualizeField(*socks_["Rho"], vishost, visport,
                     *rho_, "Charge Density (Rho)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( p_ )
   {
      VisualizeField(*socks_["P"], vishost, visport,
                     *p_, "Electric Polarization (P)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( sigma_ )
   {
      VisualizeField(*socks_["Sigma"], vishost, visport,
                     *sigma_, "Surface Charge Density (Sigma)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if (myid_ == 0) { cout << " " << flush; }
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
