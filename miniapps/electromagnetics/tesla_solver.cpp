// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tesla_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

using namespace common;

namespace electromagnetics
{

TeslaSolver::TeslaSolver(ParMesh & pmesh, int order,
                         Array<int> & kbcs,
                         Array<int> & vbcs, Vector & vbcv,
                         Coefficient & muInvCoef,
                         void   (*a_bc )(const Vector&, Vector&),
                         void   (*j_src)(const Vector&, Vector&),
                         void   (*m_src)(const Vector&, Vector&))
   : myid_(0),
     num_procs_(1),
     order_(order),
     pmesh_(&pmesh),
     visit_dc_(NULL),
     H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     curlMuInvCurl_(NULL),
     hCurlMass_(NULL),
     hDivHCurlMuInv_(NULL),
     weakCurlMuInv_(NULL),
     grad_(NULL),
     curl_(NULL),
     a_(NULL),
     b_(NULL),
     h_(NULL),
     jr_(NULL),
     j_(NULL),
     k_(NULL),
     m_(NULL),
     bd_(NULL),
     jd_(NULL),
     DivFreeProj_(NULL),
     SurfCur_(NULL),
     muInvCoef_(&muInvCoef),
     aBCCoef_(NULL),
     jCoef_(NULL),
     mCoef_(NULL),
     a_bc_(a_bc),
     j_src_(j_src),
     m_src_(m_src)
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

   int irOrder = H1FESpace_->GetElementTransformation(0)->OrderW()
                 + 2 * order;
   int geom = H1FESpace_->GetFE(0)->GetGeomType();
   const IntegrationRule * ir = &IntRules.Get(geom, irOrder);

   // Select surface attributes for Dirichlet BCs
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   non_k_bdr_.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr_ = 1;   // All outer surfaces
   non_k_bdr_ = 1; // Surfaces without applied surface currents

   for (int i=0; i<kbcs.Size(); i++)
   {
      non_k_bdr_[kbcs[i]-1] = 0;
   }

   // Setup various coefficients

   // Vector Potential on the outer surface
   if ( a_bc_ == NULL )
   {
      Vector Zero(3);
      Zero = 0.0;
      aBCCoef_ = new VectorConstantCoefficient(Zero);
   }
   else
   {
      aBCCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                               *a_bc_);
   }

   // Volume Current Density
   if ( j_src_ != NULL )
   {
      jCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             j_src_);
   }

   // Magnetization
   if ( m_src_ != NULL )
   {
      mCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             m_src_);
   }

   // Bilinear Forms
   curlMuInvCurl_  = new ParBilinearForm(HCurlFESpace_);
   curlMuInvCurl_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));

   BilinearFormIntegrator * hCurlMassInteg = new VectorFEMassIntegrator;
   hCurlMassInteg->SetIntRule(ir);
   hCurlMass_      = new ParBilinearForm(HCurlFESpace_);
   hCurlMass_->AddDomainIntegrator(hCurlMassInteg);

   BilinearFormIntegrator * hDivHCurlInteg =
      new VectorFEMassIntegrator(*muInvCoef_);
   hDivHCurlInteg->SetIntRule(ir);
   hDivHCurlMuInv_ = new ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_);
   hDivHCurlMuInv_->AddDomainIntegrator(hDivHCurlInteg);

   // Discrete Curl operator
   curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);

   // Build grid functions
   a_  = new ParGridFunction(HCurlFESpace_);
   b_  = new ParGridFunction(HDivFESpace_);
   h_  = new ParGridFunction(HCurlFESpace_);
   bd_ = new ParGridFunction(HCurlFESpace_);
   jd_ = new ParGridFunction(HCurlFESpace_);

   if ( jCoef_ || kbcs.Size() > 0 )
   {
      grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
   }
   if ( jCoef_ )
   {
      jr_          = new ParGridFunction(HCurlFESpace_);
      j_           = new ParGridFunction(HCurlFESpace_);
      DivFreeProj_ = new DivergenceFreeProjector(*H1FESpace_, *HCurlFESpace_,
                                                 irOrder, NULL, NULL, grad_);
   }

   if ( kbcs.Size() > 0 )
   {
      k_ = new ParGridFunction(HCurlFESpace_);

      // Object to solve the subproblem of computing surface currents
      SurfCur_ = new SurfaceCurrent(*H1FESpace_, *grad_,
                                    kbcs, vbcs, vbcv);
   }

   if ( mCoef_ )
   {
      m_ = new ParGridFunction(HDivFESpace_);

      weakCurlMuInv_ = new ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_);
      weakCurlMuInv_->AddDomainIntegrator(
         new VectorFECurlIntegrator(*muInvCoef_));
   }
}

TeslaSolver::~TeslaSolver()
{
   delete jCoef_;
   delete mCoef_;
   delete aBCCoef_;

   delete DivFreeProj_;
   delete SurfCur_;

   delete a_;
   delete b_;
   delete h_;
   delete jr_;
   delete j_;
   delete k_;
   delete m_;
   delete bd_;
   delete jd_;

   delete grad_;
   delete curl_;

   delete curlMuInvCurl_;
   delete hCurlMass_;
   delete hDivHCurlMuInv_;
   delete weakCurlMuInv_;

   delete H1FESpace_;
   delete HCurlFESpace_;
   delete HDivFESpace_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
TeslaSolver::GetProblemSize()
{
   return HCurlFESpace_->GlobalTrueVSize();
}

void
TeslaSolver::PrintSizes()
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
TeslaSolver::Assemble()
{
   if (myid_ == 0) { cout << "Assembling ..." << flush; }

   curlMuInvCurl_->Assemble();
   curlMuInvCurl_->Finalize();

   hDivHCurlMuInv_->Assemble();
   hDivHCurlMuInv_->Finalize();

   hCurlMass_->Assemble();
   hCurlMass_->Finalize();

   curl_->Assemble();
   curl_->Finalize();

   if ( grad_ )
   {
      grad_->Assemble();
      grad_->Finalize();
   }
   if ( weakCurlMuInv_ )
   {
      weakCurlMuInv_->Assemble();
      weakCurlMuInv_->Finalize();
   }

   if (myid_ == 0) { cout << " done." << endl; }
}

void
TeslaSolver::Update()
{
   if (myid_ == 0) { cout << "Updating ..." << endl; }

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   H1FESpace_->Update(false);
   HCurlFESpace_->Update(false);
   HDivFESpace_->Update(false);

   HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);

   // Inform the grid functions that the space has changed.
   a_->Update();
   h_->Update();
   b_->Update();
   bd_->Update();
   jd_->Update();
   if ( jr_ ) { jr_->Update(); }
   if ( j_  ) {  j_->Update(); }
   if ( k_  ) {  k_->Update(); }
   if ( m_  ) {  m_->Update(); }

   // Inform the bilinear forms that the space has changed.
   curlMuInvCurl_->Update();
   hCurlMass_->Update();
   hDivHCurlMuInv_->Update();
   if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   curl_->Update();
   if ( grad_        ) { grad_->Update(); }
   if ( DivFreeProj_ ) { DivFreeProj_->Update(); }
   if ( SurfCur_     ) { SurfCur_->Update(); }
}

void
TeslaSolver::Solve()
{
   if (myid_ == 0) { cout << "Running solver ... " << endl; }

   // Initialize the magnetic vector potential with its boundary conditions
   *a_ = 0.0;

   // Apply surface currents if available
   if ( k_ )
   {
      SurfCur_->ComputeSurfaceCurrent(*k_);
      *a_ = *k_;
   }

   // Apply uniform B boundary condition on remaining surfaces
   a_->ProjectBdrCoefficientTangent(*aBCCoef_, non_k_bdr_);

   // Initialize the RHS vector to zero
   *jd_ = 0.0;

   // Initialize the volumetric current density
   if ( jr_ )
   {
      jr_->ProjectCoefficient(*jCoef_);

      // Compute the discretely divergence-free portion of jr_
      DivFreeProj_->Mult(*jr_, *j_);

      // Compute the dual of j_
      hCurlMass_->AddMult(*j_, *jd_);
   }

   // Initialize the Magnetization
   if ( m_ )
   {
      m_->ProjectCoefficient(*mCoef_);
      weakCurlMuInv_->AddMult(*m_, *jd_, mu0_);
   }

   // Apply Dirichlet BCs to matrix and right hand side and otherwise
   // prepare the linear system
   HypreParMatrix CurlMuInvCurl;
   HypreParVector A(HCurlFESpace_);
   HypreParVector RHS(HCurlFESpace_);

   curlMuInvCurl_->FormLinearSystem(ess_bdr_tdofs_, *a_, *jd_, CurlMuInvCurl,
                                    A, RHS);

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.
   HypreAMS ams(CurlMuInvCurl, HCurlFESpace_);
   ams.SetSingularProblem();

   HyprePCG pcg (CurlMuInvCurl);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(50);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(ams);
   pcg.Mult(RHS, A);

   // Extract the parallel grid function corresponding to the finite
   // element approximation A. This is the local solution on each
   // processor.
   curlMuInvCurl_->RecoverFEMSolution(A, *jd_, *a_);

   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi.
   curl_->Mult(*a_, *b_);

   // Compute magnetic field (H) from B and M
   if (myid_ == 0) { cout << "Computing H ... " << flush; }

   hDivHCurlMuInv_->Mult(*b_, *bd_);
   if ( m_ )
   {
      hDivHCurlMuInv_->AddMult(*m_, *bd_, -1.0 * mu0_);
   }

   HypreParMatrix MassHCurl;
   Vector BD, H;

   Array<int> dbc_dofs_h;
   hCurlMass_->FormLinearSystem(dbc_dofs_h, *h_, *bd_, MassHCurl, H, BD);

   HyprePCG pcgM(MassHCurl);
   pcgM.SetTol(1e-12);
   pcgM.SetMaxIter(500);
   pcgM.SetPrintLevel(0);
   HypreDiagScale diagM;
   pcgM.SetPreconditioner(diagM);
   pcgM.Mult(BD, H);

   hCurlMass_->RecoverFEMSolution(H, *bd_, *h_);

   if (myid_ == 0) { cout << "done." << flush; }

   if (myid_ == 0) { cout << " Solver done. " << endl; }
}

void
TeslaSolver::GetErrorEstimates(Vector & errors)
{
   if (myid_ == 0) { cout << "Estimating Error ... " << flush; }

   // Space for the discontinuous (original) flux
   CurlCurlIntegrator flux_integrator(*muInvCoef_);
   RT_FECollection flux_fec(order_-1, pmesh_->SpaceDimension());
   ParFiniteElementSpace flux_fes(pmesh_, &flux_fec);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   ND_FECollection smooth_flux_fec(order_, pmesh_->Dimension());
   ParFiniteElementSpace smooth_flux_fes(pmesh_, &smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, *a_,
                      smooth_flux_fes, flux_fes, errors, norm_p);

   if (myid_ == 0) { cout << "done." << endl; }
}

void
TeslaSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("A", a_);
   visit_dc.RegisterField("B", b_);
   visit_dc.RegisterField("H", h_);
   if ( j_ ) { visit_dc.RegisterField("J", j_); }
   if ( k_ ) { visit_dc.RegisterField("K", k_); }
   if ( m_ ) { visit_dc.RegisterField("M", m_); }
   if ( SurfCur_ ) { visit_dc.RegisterField("Psi", SurfCur_->GetPsi()); }
}

void
TeslaSolver::WriteVisItFields(int it)
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
TeslaSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   socks_["A"] = new socketstream;
   socks_["A"]->precision(8);

   socks_["B"] = new socketstream;
   socks_["B"]->precision(8);

   socks_["H"] = new socketstream;
   socks_["H"]->precision(8);

   if ( j_ )
   {
      socks_["J"] = new socketstream;
      socks_["J"]->precision(8);
   }
   if ( k_ )
   {
      socks_["K"] = new socketstream;
      socks_["K"]->precision(8);

      socks_["Psi"] = new socketstream;
      socks_["Psi"]->precision(8);
   }
   if ( m_ )
   {
      socks_["M"] = new socketstream;
      socks_["M"]->precision(8);
   }
   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl; }
}

void
TeslaSolver::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   VisualizeField(*socks_["A"], vishost, visport,
                  *a_, "Vector Potential (A)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["B"], vishost, visport,
                  *b_, "Magnetic Flux Density (B)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["H"], vishost, visport,
                  *h_, "Magnetic Field (H)", Wx, Wy, Ww, Wh);
   Wx += offx;

   if ( j_ )
   {
      VisualizeField(*socks_["J"], vishost, visport,
                     *j_, "Current Density (J)", Wx, Wy, Ww, Wh);
   }

   Wx = 0; Wy += offy; // next line

   if ( k_ )
   {
      VisualizeField(*socks_["K"], vishost, visport,
                     *k_, "Surface Current Density (K)", Wx, Wy, Ww, Wh);
      Wx += offx;

      VisualizeField(*socks_["Psi"], vishost, visport,
                     *SurfCur_->GetPsi(),
                     "Surface Current Potential (Psi)", Wx, Wy, Ww, Wh);
      Wx += offx;
   }
   if ( m_ )
   {
      VisualizeField(*socks_["M"], vishost, visport,
                     *m_, "Magnetization (M)", Wx, Wy, Ww, Wh);
      // Wx += offx; // not used
   }
   if (myid_ == 0) { cout << " done." << endl; }
}

SurfaceCurrent::SurfaceCurrent(ParFiniteElementSpace & H1FESpace,
                               ParDiscreteGradOperator & grad,
                               Array<int> & kbcs,
                               Array<int> & vbcs, Vector & vbcv)
   : H1FESpace_(&H1FESpace),
     grad_(&grad),
     kbcs_(&kbcs),
     vbcs_(&vbcs),
     vbcv_(&vbcv),
     s0_(NULL),
     psi_(NULL),
     rhs_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_rank(H1FESpace_->GetParMesh()->GetComm(), &myid_);

   s0_ = new ParBilinearForm(H1FESpace_);
   s0_->AddBoundaryIntegrator(new DiffusionIntegrator);
   s0_->Assemble();
   s0_->Finalize();
   S0_ = new HypreParMatrix;

   ess_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   ess_bdr_ = 0;
   for (int i=0; i<vbcs_->Size(); i++)
   {
      ess_bdr_[(*vbcs_)[i]-1] = 1;
   }
   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);

   non_k_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   non_k_bdr_ = 1;
   for (int i=0; i<kbcs_->Size(); i++)
   {
      non_k_bdr_[(*kbcs_)[i]-1] = 0;
   }

   psi_ = new ParGridFunction(H1FESpace_);
   rhs_ = new ParGridFunction(H1FESpace_);

   pcg_ = NULL;
   amg_ = NULL;
}

SurfaceCurrent::~SurfaceCurrent()
{
   delete psi_;
   delete rhs_;

   delete pcg_;
   delete amg_;

   delete S0_;

   delete s0_;
}

void
SurfaceCurrent::InitSolver() const
{
   delete pcg_;
   delete amg_;

   amg_ = new HypreBoomerAMG(*S0_);
   amg_->SetPrintLevel(0);
   pcg_ = new HyprePCG(*S0_);
   pcg_->SetTol(1e-14);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*amg_);
}

void
SurfaceCurrent::ComputeSurfaceCurrent(ParGridFunction & k)
{
   if (myid_ == 0) { cout << "Computing K ... " << flush; }

   // Apply piecewise constant voltage boundary condition
   *psi_ = 0.0;
   *rhs_ = 0.0;
   Array<int> vbc_bdr_attr(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   for (int i=0; i<vbcs_->Size(); i++)
   {
      ConstantCoefficient voltage((*vbcv_)[i]);
      vbc_bdr_attr = 0;
      vbc_bdr_attr[(*vbcs_)[i]-1] = 1;
      psi_->ProjectBdrCoefficient(voltage, vbc_bdr_attr);
   }

   // Apply essential BC and form linear system
   s0_->FormLinearSystem(ess_bdr_tdofs_, *psi_, *rhs_, *S0_, Psi_, RHS_);

   // Solve the linear system for Psi
   if ( pcg_ == NULL ) { this->InitSolver(); }
   pcg_->Mult(RHS_, Psi_);

   // Compute the parallel grid function corresponding to Psi
   s0_->RecoverFEMSolution(Psi_, *rhs_, *psi_);

   // Compute the surface current from psi
   grad_->Mult(*psi_, k);

   // Force the tangential part of k to be zero away from the intended surfaces
   Vector vZero(3); vZero = 0.0;
   VectorConstantCoefficient Zero(vZero);
   k.ProjectBdrCoefficientTangent(Zero, non_k_bdr_);

   if (myid_ == 0) { cout << "done." << endl; }
}

void
SurfaceCurrent::Update()
{
   delete pcg_; pcg_ = NULL;
   delete amg_; amg_ = NULL;
   delete S0_;  S0_  = new HypreParMatrix;

   psi_->Update();
   rhs_->Update();

   s0_->Update();
   s0_->Assemble();
   s0_->Finalize();

   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
