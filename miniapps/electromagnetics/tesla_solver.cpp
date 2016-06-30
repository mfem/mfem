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

#include "tesla_solver.hpp"

using namespace std;

namespace mfem
{

using namespace miniapps;

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
     hDivMassMuInv_(NULL),
     hDivHCurlMuInv_(NULL),
     Grad_(NULL),
     Curl_(NULL),
     a_(NULL),
     b_(NULL),
     h_(NULL),
     j_(NULL),
     k_(NULL),
     m_(NULL),
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

   hCurlMass_      = new ParBilinearForm(HCurlFESpace_);
   hCurlMass_->AddDomainIntegrator(new VectorFEMassIntegrator);

   hDivHCurlMuInv_ = new ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_);
   hDivHCurlMuInv_->AddDomainIntegrator(new VectorFEMassIntegrator(*muInvCoef_));

   // Discrete Curl operator
   Curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);

   // Build grid functions
   a_ = new ParGridFunction(HCurlFESpace_);
   b_ = new ParGridFunction(HDivFESpace_);
   h_ = new ParGridFunction(HCurlFESpace_);

   if ( jCoef_ || kbcs.Size() > 0 )
   {
      Grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
   }
   if ( jCoef_ )
   {
      j_           = new ParGridFunction(HCurlFESpace_);
      DivFreeProj_ = new DivergenceFreeProjector(*H1FESpace_, *HCurlFESpace_,
                                                 *Grad_);
   }

   if ( kbcs.Size() > 0 )
   {
      k_ = new ParGridFunction(HCurlFESpace_);

      // Object to solve the subproblem of computing surface currents
      SurfCur_ = new SurfaceCurrent(*H1FESpace_, *HCurlFESpace_, *Grad_,
                                    kbcs, vbcs, vbcv);
   }

   if ( mCoef_ )
   {
      m_ = new ParGridFunction(HDivFESpace_);

      hDivMassMuInv_ = new ParBilinearForm(HDivFESpace_);
      hDivMassMuInv_->AddDomainIntegrator(
         new VectorFEMassIntegrator(*muInvCoef_));
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
   delete j_;
   delete k_;
   delete m_;

   delete Grad_;
   delete Curl_;

   delete curlMuInvCurl_;
   delete hCurlMass_;
   delete hDivMassMuInv_;
   delete hDivHCurlMuInv_;

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

   if ( hCurlMass_ )
   {
      hCurlMass_->Assemble();
      hCurlMass_->Finalize();
   }
   if ( hDivMassMuInv_ )
   {
      hDivMassMuInv_->Assemble();
      hDivMassMuInv_->Finalize();
   }
   if ( hDivHCurlMuInv_ )
   {
      hDivHCurlMuInv_->Assemble();
      hDivHCurlMuInv_->Finalize();
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

   // Inform the grid functions that the space has changed.
   a_->Update();
   h_->Update();
   b_->Update();
   if ( j_ ) { j_->Update(); }
   if ( k_ ) { k_->Update(); }
   if ( m_ ) { m_->Update(); }

   // Inform the bilinear forms that the space has changed.
   curlMuInvCurl_->Update();
   if ( hCurlMass_      ) { hCurlMass_->Update(); }
   if ( hDivMassMuInv_  ) { hDivMassMuInv_->Update(); }
   if ( hDivHCurlMuInv_ ) { hDivHCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   Curl_->Update();
   if ( Grad_        ) { Grad_->Update(); }
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

   // Initialize the RHS vector
   HypreParVector *RHS = new HypreParVector(HCurlFESpace_);
   *RHS = 0.0;

   // Initialize the volumetric current density
   if ( j_ )
   {
      j_->ProjectCoefficient(*jCoef_);

      HypreParMatrix *MassHCurl = hCurlMass_->ParallelAssemble();
      HypreParVector *J    = j_->ParallelProject();
      HypreParVector *JD   = new HypreParVector(HCurlFESpace_);

      MassHCurl->Mult(*J,*JD);
      DivFreeProj_->Mult(*JD, *RHS);

      delete MassHCurl;
      delete J;
      delete JD;
   }

   // Initialize the Magnetization
   HypreParVector *M = NULL;
   if ( m_ )
   {
      m_->ProjectCoefficient(*mCoef_);
      M = m_->ParallelProject();

      HypreParMatrix *MassHDiv = hDivMassMuInv_->ParallelAssemble();
      HypreParVector *MD   = new HypreParVector(HDivFESpace_);

      MassHDiv->Mult(*M,*MD);
      Curl_->MultTranspose(*MD,*RHS,mu0_,1.0);

      delete MassHDiv;
      delete MD;
   }

   // Apply Dirichlet BCs to matrix and right hand side
   HypreParMatrix *CurlMuInvCurl = curlMuInvCurl_->ParallelAssemble();
   HypreParVector *A             = a_->ParallelProject();

   // Apply the boundary conditions to the assembled matrix and vectors
   curlMuInvCurl_->ParallelEliminateEssentialBC(ess_bdr_,
                                                *CurlMuInvCurl,
                                                *A, *RHS);

   // Define and apply a parallel PCG solver for AX=B with the AMS
   // preconditioner from hypre.
   HypreAMS *ams = new HypreAMS(*CurlMuInvCurl, HCurlFESpace_);
   ams->SetSingularProblem();

   HyprePCG *pcg = new HyprePCG(*CurlMuInvCurl);
   pcg->SetTol(1e-12);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*ams);
   pcg->Mult(*RHS, *A);

   delete ams;
   delete pcg;
   delete CurlMuInvCurl;
   delete RHS;

   // Extract the parallel grid function corresponding to the finite
   // element approximation Phi. This is the local solution on each
   // processor.
   *a_ = *A;

   // Compute the negative Gradient of the solution vector.  This is
   // the magnetic field corresponding to the scalar potential
   // represented by phi.
   HypreParVector *B = new HypreParVector(HDivFESpace_);
   Curl_->Mult(*A,*B);
   *b_ = *B;

   // Compute magnetic field (H) from B and M
   if (myid_ == 0) { cout << "Computing H ... " << flush; }

   ParGridFunction bd(HCurlFESpace_);
   hDivHCurlMuInv_->Mult(*b_, bd);
   if ( m_ )
   {
      hDivHCurlMuInv_->AddMult(*m_, bd, -1.0 * mu0_);
   }

   HypreParMatrix MassHCurl;
   Vector BD, H;

   Array<int> dbc_dofs_h;
   hCurlMass_->FormLinearSystem(dbc_dofs_h, *h_, bd, MassHCurl, H, BD);

   HyprePCG * pcgM = new HyprePCG(MassHCurl);
   pcgM->SetTol(1e-12);
   pcgM->SetMaxIter(500);
   pcgM->SetPrintLevel(0);
   HypreDiagScale diagM;
   pcgM->SetPreconditioner(diagM);
   pcgM->Mult(BD,H);

   hCurlMass_->RecoverFEMSolution(H, bd, *h_);

   if (myid_ == 0) { cout << "done." << flush; }

   delete pcgM;
   delete A;
   delete B;
   delete M;

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
      Wx += offx;
   }
   if (myid_ == 0) { cout << " done." << endl; }
}

SurfaceCurrent::SurfaceCurrent(ParFiniteElementSpace & H1FESpace,
                               ParFiniteElementSpace & HCurlFESpace,
                               ParDiscreteGradOperator & Grad,
                               Array<int> & kbcs,
                               Array<int> & vbcs, Vector & vbcv)
   : H1FESpace_(&H1FESpace),
     HCurlFESpace_(&HCurlFESpace),
     Grad_(&Grad),
     kbcs_(&kbcs),
     vbcs_(&vbcs),
     vbcv_(&vbcv)
{
   // Initialize MPI variables
   MPI_Comm_rank(H1FESpace_->GetParMesh()->GetComm(), &myid_);

   s0_ = new ParBilinearForm(H1FESpace_);
   s0_->AddBoundaryIntegrator(new DiffusionIntegrator);
   s0_->Assemble();
   s0_->Finalize();
   S0_ = s0_->ParallelAssemble();

   amg_ = new HypreBoomerAMG(*S0_);
   amg_->SetPrintLevel(0);

   pcg_ = new HyprePCG(*S0_);
   pcg_->SetTol(1e-12);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*amg_);

   ess_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   ess_bdr_ = 0;
   for (int i=0; i<vbcs_->Size(); i++)
   {
      ess_bdr_[(*vbcs_)[i]-1] = 1;
   }

   non_k_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   non_k_bdr_ = 1;
   for (int i=0; i<kbcs_->Size(); i++)
   {
      non_k_bdr_[(*kbcs_)[i]-1] = 0;
   }

   psi_ = new ParGridFunction(H1FESpace_);
   *psi_ = 0.0;

   // Apply piecewise constant voltage boundary condition
   Array<int> vbc_bdr_attr(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   for (int i=0; i<vbcs_->Size(); i++)
   {
      ConstantCoefficient voltage((*vbcv_)[i]);
      vbc_bdr_attr = 0;
      vbc_bdr_attr[(*vbcs_)[i]-1] = 1;
      psi_->ProjectBdrCoefficient(voltage, vbc_bdr_attr);
   }

   PSI_ = psi_->ParallelProject();
   RHS_ = new HypreParVector(H1FESpace_);
   K_   = new HypreParVector(HCurlFESpace_);

   s0_->ParallelEliminateEssentialBC(ess_bdr_,
                                     *S0_,
                                     *PSI_, *RHS_);
}

SurfaceCurrent::~SurfaceCurrent()
{
   delete pcg_;
   delete amg_;
   delete S0_;
   delete PSI_;
   delete RHS_;
   delete K_;
   delete s0_;
   delete psi_;
}

void
SurfaceCurrent::ComputeSurfaceCurrent(ParGridFunction & k)
{
   if (myid_ == 0) { cout << "Computing K ... " << flush; }

   k = 0.0;
   pcg_->Mult(*RHS_, *PSI_);
   *psi_ = *PSI_;

   Grad_->Mult(*PSI_,*K_);
   k = *K_;

   Vector vZero(3); vZero = 0.0;
   VectorConstantCoefficient Zero(vZero);
   k.ProjectBdrCoefficientTangent(Zero,non_k_bdr_);

   if (myid_ == 0) { cout << "done." << endl; }
}

void
SurfaceCurrent::Update()
{
   delete pcg_;
   delete amg_;
   delete S0_;
   delete PSI_;
   delete RHS_;
   delete K_;

   psi_->Update();
   *psi_ = 0.0;

   s0_->Update();
   s0_->Assemble();
   s0_->Finalize();
   S0_ = s0_->ParallelAssemble();

   amg_ = new HypreBoomerAMG(*S0_);
   amg_->SetPrintLevel(0);

   pcg_ = new HyprePCG(*S0_);
   pcg_->SetTol(1e-12);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*amg_);

   // Apply piecewise constant voltage boundary condition
   Array<int> vbc_bdr_attr(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   for (int i=0; i<vbcs_->Size(); i++)
   {
      ConstantCoefficient voltage((*vbcv_)[i]);
      vbc_bdr_attr = 0;
      vbc_bdr_attr[(*vbcs_)[i]-1] = 1;
      psi_->ProjectBdrCoefficient(voltage, vbc_bdr_attr);
   }

   PSI_ = psi_->ParallelProject();
   RHS_ = new HypreParVector(H1FESpace_);
   K_   = new HypreParVector(HCurlFESpace_);

   s0_->ParallelEliminateEssentialBC(ess_bdr_,
                                     *S0_,
                                     *PSI_, *RHS_);
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
