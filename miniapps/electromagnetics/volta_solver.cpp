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

#include "volta_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{

using namespace common;

namespace electromagnetics
{

VoltaSolver::VoltaSolver(ParMesh & pmesh, int order,
                         Array<int> & dbcs, Vector & dbcv,
                         Array<int> & nbcs, Vector & nbcv,
                         Coefficient & epsCoef,
                         double (*phi_bc )(const Vector&),
                         double (*rho_src)(const Vector&),
                         void   (*p_src  )(const Vector&, Vector&),
                         Vector & point_charges)
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
     L2FESpace_(NULL),
     divEpsGrad_(NULL),
     h1Mass_(NULL),
     h1SurfMass_(NULL),
     hDivMass_(NULL),
     hCurlHDivEps_(NULL),
     hCurlHDiv_(NULL),
     weakDiv_(NULL),
     rhod_(NULL),
     l2_vol_int_(NULL),
     rt_surf_int_(NULL),
     grad_(NULL),
     phi_(NULL),
     rho_src_(NULL),
     rho_(NULL),
     sigma_src_(NULL),
     e_(NULL),
     d_(NULL),
     p_src_(NULL),
     oneCoef_(1.0),
     epsCoef_(&epsCoef),
     phiBCCoef_(NULL),
     rhoCoef_(NULL),
     pCoef_(NULL),
     phi_bc_func_(phi_bc),
     rho_src_func_(rho_src),
     p_src_func_(p_src),
     point_charge_params_(point_charges),
     point_charges_(0)
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
   L2FESpace_    = new L2_ParFESpace(pmesh_,order-1,pmesh_->Dimension());

   // Select surface attributes for Dirichlet BCs
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr_ = 0;   // Deselect all outer surfaces
   for (int i=0; i<dbcs_->Size(); i++)
   {
      ess_bdr_[(*dbcs_)[i]-1] = 1;
   }

   // Setup various coefficients

   // Potential on outer surface
   if ( phi_bc_func_ != NULL )
   {
      phiBCCoef_ = new FunctionCoefficient(*phi_bc_func_);
   }

   // Volume Charge Density
   if ( rho_src_func_ != NULL )
   {
      rhoCoef_ = new FunctionCoefficient(rho_src_func_);
   }

   // Polarization
   if ( p_src_func_ != NULL )
   {
      pCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             p_src_func_);
   }

   // Bilinear Forms
   divEpsGrad_  = new ParBilinearForm(H1FESpace_);
   divEpsGrad_->AddDomainIntegrator(new DiffusionIntegrator(*epsCoef_));

   hDivMass_ = new ParBilinearForm(HDivFESpace_);
   hDivMass_->AddDomainIntegrator(new VectorFEMassIntegrator);

   hCurlHDivEps_ = new ParMixedBilinearForm(HCurlFESpace_,HDivFESpace_);
   hCurlHDivEps_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsCoef_));

   rhod_   = new ParLinearForm(H1FESpace_);

   l2_vol_int_ = new ParLinearForm(L2FESpace_);
   l2_vol_int_->AddDomainIntegrator(new DomainLFIntegrator(oneCoef_));

   rt_surf_int_ = new ParLinearForm(HDivFESpace_);
   rt_surf_int_->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator);

   // Discrete derivative operator
   grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
   div_  = new ParDiscreteDivOperator(HDivFESpace_, L2FESpace_);

   // Build grid functions
   phi_  = new ParGridFunction(H1FESpace_);
   d_    = new ParGridFunction(HDivFESpace_);
   e_    = new ParGridFunction(HCurlFESpace_);
   rho_  = new ParGridFunction(L2FESpace_);

   if ( point_charge_params_.Size() > 0 )
   {
      int dim = pmesh_->Dimension();
      int npts = point_charge_params_.Size() / (dim + 1);
      point_charges_.resize(npts);

      Vector cent(dim);
      for (int i=0; i<npts; i++)
      {
         for (int d=0; d<dim; d++)
         {
            cent[d] = point_charge_params_[(dim + 1) * i + d];
         }
         double s = point_charge_params_[(dim + 1) * i + dim];

         point_charges_[i] = new DeltaCoefficient();
         point_charges_[i]->SetScale(s);
         point_charges_[i]->SetDeltaCenter(cent);

         rhod_->AddDomainIntegrator(new DomainLFIntegrator(*point_charges_[i]));
      }
   }

   if ( rho_src_func_ )
   {
      rho_src_ = new ParGridFunction(H1FESpace_);

      h1Mass_ = new ParBilinearForm(H1FESpace_);
      h1Mass_->AddDomainIntegrator(new MassIntegrator);
   }

   if ( p_src_func_ )
   {
      p_src_ = new ParGridFunction(HCurlFESpace_);

      hCurlHDiv_ = new ParMixedBilinearForm(HCurlFESpace_, HDivFESpace_);
      hCurlHDiv_->AddDomainIntegrator(new VectorFEMassIntegrator);

      weakDiv_ = new ParMixedBilinearForm(HCurlFESpace_, H1FESpace_);
      weakDiv_->AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
   }

   if ( nbcs_->Size() > 0 )
   {
      sigma_src_ = new ParGridFunction(H1FESpace_);

      h1SurfMass_ = new ParBilinearForm(H1FESpace_);
      h1SurfMass_->AddBoundaryIntegrator(new MassIntegrator);
   }
}

VoltaSolver::~VoltaSolver()
{
   delete phiBCCoef_;
   delete rhoCoef_;
   delete pCoef_;

   delete phi_;
   delete rho_src_;
   delete rho_;
   delete rhod_;
   delete l2_vol_int_;
   delete rt_surf_int_;
   delete sigma_src_;
   delete d_;
   delete e_;
   delete p_src_;

   delete grad_;
   delete div_;

   delete divEpsGrad_;
   delete h1Mass_;
   delete h1SurfMass_;
   delete hDivMass_;
   delete hCurlHDivEps_;
   delete hCurlHDiv_;
   delete weakDiv_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;
   delete L2FESpace_;

   for (unsigned int i=0; i<point_charges_.size(); i++)
   {
      delete point_charges_[i];
   }

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
   HYPRE_Int size_l2 = L2FESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
      cout << "Number of L2      unknowns: " << size_l2 << endl;
   }
}

void VoltaSolver::Assemble()
{
   if (myid_ == 0) { cout << "Assembling ... " << flush; }

   divEpsGrad_->Assemble();
   divEpsGrad_->Finalize();

   hDivMass_->Assemble();
   hDivMass_->Finalize();

   hCurlHDivEps_->Assemble();
   hCurlHDivEps_->Finalize();

   *rhod_ = 0.0;
   rhod_->Assemble();

   l2_vol_int_->Assemble();
   rt_surf_int_->Assemble();

   grad_->Assemble();
   grad_->Finalize();

   div_->Assemble();
   div_->Finalize();

   if ( h1Mass_ )
   {
      h1Mass_->Assemble();
      h1Mass_->Finalize();
   }
   if ( h1SurfMass_ )
   {
      h1SurfMass_->Assemble();
      h1SurfMass_->Finalize();
   }
   if ( hCurlHDiv_ )
   {
      hCurlHDiv_->Assemble();
      hCurlHDiv_->Finalize();
   }
   if ( weakDiv_ )
   {
      weakDiv_->Assemble();
      weakDiv_->Finalize();
   }

   if (myid_ == 0) { cout << "done." << endl << flush; }
}

void
VoltaSolver::Update()
{
   if (myid_ == 0) { cout << "Updating ..." << endl; }

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   H1FESpace_->Update(false);
   HCurlFESpace_->Update(false);
   HDivFESpace_->Update(false);
   L2FESpace_->Update(false);

   // Inform the grid functions that the space has changed.
   phi_->Update();
   rhod_->Update();
   l2_vol_int_->Update();
   rt_surf_int_->Update();
   d_->Update();
   e_->Update();
   rho_->Update();
   if ( rho_src_   ) { rho_src_->Update(); }
   if ( sigma_src_ ) { sigma_src_->Update(); }
   if ( p_src_     ) { p_src_->Update(); }

   // Inform the bilinear forms that the space has changed.
   divEpsGrad_->Update();
   hDivMass_->Update();
   hCurlHDivEps_->Update();

   if ( h1Mass_ )     { h1Mass_->Update(); }
   if ( h1SurfMass_ ) { h1SurfMass_->Update(); }
   if ( hCurlHDiv_ )  { hCurlHDiv_->Update(); }
   if ( weakDiv_ )    { weakDiv_->Update(); }

   // Inform the other objects that the space has changed.
   grad_->Update();
   div_->Update();
}

void
VoltaSolver::Solve()
{
   if (myid_ == 0) { cout << "Running solver ... " << endl; }

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
            if ((*dbcs_)[i] <= dbc_bdr_attr.Size())
            {
               dbc_bdr_attr[(*dbcs_)[i]-1] = 1;
            }
            phi_->ProjectBdrCoefficient(voltage, dbc_bdr_attr);
         }
      }
   }

   // Initialize the volumetric charge density
   if ( rho_src_ )
   {
      rho_src_->ProjectCoefficient(*rhoCoef_);
      h1Mass_->AddMult(*rho_src_, *rhod_);
   }

   // Initialize the Polarization
   if ( p_src_ )
   {
      p_src_->ProjectCoefficient(*pCoef_);
      weakDiv_->AddMult(*p_src_, *rhod_);
   }

   // Initialize the surface charge density
   if ( sigma_src_ )
   {
      *sigma_src_ = 0.0;

      Array<int> nbc_bdr_attr(pmesh_->bdr_attributes.Max());
      for (int i=0; i<nbcs_->Size(); i++)
      {
         ConstantCoefficient sigma_coef((*nbcv_)[i]);
         nbc_bdr_attr = 0;
         if ((*nbcs_)[i] <= nbc_bdr_attr.Size())
         {
            nbc_bdr_attr[(*nbcs_)[i]-1] = 1;
         }
         sigma_src_->ProjectBdrCoefficient(sigma_coef, nbc_bdr_attr);
      }
      h1SurfMass_->AddMult(*sigma_src_, *rhod_);
   }

   // Determine the essential BC degrees of freedom
   if ( dbcs_->Size() > 0 )
   {
      // From user supplied boundary attributes
      H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }
   else
   {
      // Use the first DoF on processor zero by default
      if ( myid_ == 0 )
      {
         ess_bdr_tdofs_.SetSize(1);
         ess_bdr_tdofs_[0] = 0;
      }
   }

   // Apply essential BC and form linear system
   HypreParMatrix DivEpsGrad;
   HypreParVector Phi(H1FESpace_);
   HypreParVector RHS(H1FESpace_);

   divEpsGrad_->FormLinearSystem(ess_bdr_tdofs_, *phi_, *rhod_, DivEpsGrad,
                                 Phi, RHS);

   // Define and apply a parallel PCG solver for AX=B with the AMG
   // preconditioner from hypre.
   HypreBoomerAMG amg(DivEpsGrad);
   HyprePCG pcg(DivEpsGrad);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(500);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(amg);
   pcg.Mult(RHS, Phi);

   // Extract the parallel grid function corresponding to the finite
   // element approximation Phi. This is the local solution on each
   // processor.
   divEpsGrad_->RecoverFEMSolution(Phi, *rhod_, *phi_);

   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi.
   grad_->Mult(*phi_, *e_); *e_ *= -1.0;

   // Compute electric displacement (D) from E and P (if present)
   if (myid_ == 0) { cout << "Computing D ..." << flush; }

   ParGridFunction ed(HDivFESpace_);
   hCurlHDivEps_->Mult(*e_, ed);
   if ( p_src_ )
   {
      hCurlHDiv_->AddMult(*p_src_, ed, -1.0);
   }

   HypreParMatrix MassHDiv;
   Vector ED, D;

   Array<int> dbc_dofs_d;
   hDivMass_->FormLinearSystem(dbc_dofs_d, *d_, ed, MassHDiv, D, ED);

   HyprePCG pcgM(MassHDiv);
   pcgM.SetTol(1e-12);
   pcgM.SetMaxIter(500);
   pcgM.SetPrintLevel(0);
   HypreDiagScale diagM;
   pcgM.SetPreconditioner(diagM);
   pcgM.Mult(ED, D);

   hDivMass_->RecoverFEMSolution(D, ed, *d_);

   // Compute charge density from rho = Div(D)
   div_->Mult(*d_, *rho_);

   if (myid_ == 0) { cout << "done." << flush; }

   {
      // Compute total charge as volume integral of rho
      double charge_rho = (*l2_vol_int_)(*rho_);

      // Compute total charge as surface integral of D
      double charge_D = (*rt_surf_int_)(*d_);

      if (myid_ == 0)
      {
         cout << endl << "Total charge: \n"
              << "   Volume integral of charge density:   " << charge_rho
              << "\n   Surface integral of dielectric flux: " << charge_D
              << endl << flush;
      }
   }

   if (myid_ == 0) { cout << "Solver done. " << endl; }
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

   if (myid_ == 0) { cout << "done." << endl; }
}

void
VoltaSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("Phi", phi_);
   visit_dc.RegisterField("D",     d_);
   visit_dc.RegisterField("E",     e_);
   visit_dc.RegisterField("Rho", rho_);
   if ( rho_src_   ) { visit_dc.RegisterField("Rho Source",     rho_src_); }
   if ( p_src_     ) { visit_dc.RegisterField("P Source",         p_src_); }
   if ( sigma_src_ ) { visit_dc.RegisterField("Sigma Source", sigma_src_); }
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

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
VoltaSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   socks_["Phi"] = new socketstream;
   socks_["Phi"]->precision(8);

   socks_["D"] = new socketstream;
   socks_["D"]->precision(8);

   socks_["E"] = new socketstream;
   socks_["E"]->precision(8);

   socks_["Rho"] = new socketstream;
   socks_["Rho"]->precision(8);

   if ( rho_src_ )
   {
      socks_["RhoSrc"] = new socketstream;
      socks_["RhoSrc"]->precision(8);
   }
   if ( p_src_ )
   {
      socks_["PSrc"] = new socketstream;
      socks_["PSrc"]->precision(8);
   }
   if ( sigma_src_ )
   {
      socks_["SigmaSrc"] = new socketstream;
      socks_["SigmaSrc"]->precision(8);
   }
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

   VisualizeField(*socks_["E"], vishost, visport,
                  *e_, "Electric Field (E)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["D"], vishost, visport,
                  *d_, "Electric Displacement (D)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Rho"], vishost, visport,
                  *rho_, "Charge Density", Wx, Wy, Ww, Wh);
   Wx = 0; Wy += offy; // next line

   if ( rho_src_ )
   {
      VisualizeField(*socks_["RhoSrc"], vishost, visport,
                     *rho_src_, "Charge Density Source (Rho)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( p_src_ )
   {
      VisualizeField(*socks_["PSrc"], vishost, visport,
                     *p_src_, "Electric Polarization Source (P)",
                     Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( sigma_src_ )
   {
      VisualizeField(*socks_["SigmaSrc"], vishost, visport,
                     *sigma_src_, "Surface Charge Density Source (Sigma)",
                     Wx, Wy, Ww, Wh);
      // Wx += offx; // not used
   }
   if (myid_ == 0) { cout << " done." << endl; }
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
