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

#include "hertz_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;
namespace mfem
{
using namespace common;

namespace electromagnetics
{

// Used for combining scalar coefficients
double prodFunc(double a, double b) { return a * b; }

HertzSolver::HertzSolver(ParMesh & pmesh, int order, double freq,
                         HertzSolver::SolverType sol,
                         ComplexOperator::Convention conv,
                         Coefficient & epsCoef,
                         Coefficient & muInvCoef,
                         Coefficient * sigmaCoef,
                         Coefficient * etaInvCoef,
                         Array<int> & abcs,
                         Array<int> & dbcs,
                         void   (*e_r_bc )(const Vector&, Vector&),
                         void   (*e_i_bc )(const Vector&, Vector&),
                         void   (*j_r_src)(const Vector&, Vector&),
                         void   (*j_i_src)(const Vector&, Vector&))
   : myid_(0),
     num_procs_(1),
     order_(order),
     logging_(1),
     sol_(sol),
     conv_(conv),
     ownsEtaInv_(etaInvCoef == NULL),
     freq_(freq),
     pmesh_(&pmesh),
     // H1FESpace_(NULL),
     HCurlFESpace_(NULL),
     // HDivFESpace_(NULL),
     // curlMuInvCurl_(NULL),
     // hCurlMass_(NULL),
     // hDivHCurlMuInv_(NULL),
     // weakCurlMuInv_(NULL),
     // grad_(NULL),
     // curl_(NULL),
     a1_(NULL),
     b1_(NULL),
     // e_r_(NULL),
     // e_i_(NULL),
     e_(NULL),
     // b_(NULL),
     // h_(NULL),
     // j_r_(NULL),
     // j_i_(NULL),
     j_(NULL),
     jd_(NULL),
     // jd_r_(NULL),
     // jd_i_(NULL),
     // k_(NULL),
     // m_(NULL),
     // bd_(NULL),
     // jd_(NULL),
     // DivFreeProj_(NULL),
     // SurfCur_(NULL),
     epsCoef_(&epsCoef),
     muInvCoef_(&muInvCoef),
     sigmaCoef_(sigmaCoef),
     etaInvCoef_(etaInvCoef),
     omegaCoef_(new ConstantCoefficient(2.0 * M_PI * freq_)),
     negOmegaCoef_(new ConstantCoefficient(-2.0 * M_PI * freq_)),
     omega2Coef_(new ConstantCoefficient(pow(2.0 * M_PI * freq_, 2))),
     negOmega2Coef_(new ConstantCoefficient(-pow(2.0 * M_PI * freq_, 2))),
     massCoef_(NULL),
     posMassCoef_(NULL),
     lossCoef_(NULL),
     // gainCoef_(NULL),
     abcCoef_(NULL),
     jrCoef_(NULL),
     jiCoef_(NULL),
     erCoef_(NULL),
     eiCoef_(NULL),
     // mCoef_(NULL),
     // a_bc_(a_bc),
     j_r_src_(j_r_src),
     j_i_src_(j_i_src),
     e_r_bc_(e_r_bc),
     e_i_bc_(e_i_bc),
     // m_src_(m_src)
     dbcs_(&dbcs),
     visit_dc_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   // H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
   // HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());

   blockTrueOffsets_.SetSize(3);
   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   // int irOrder = H1FESpace_->GetElementTransformation(0)->OrderW()
   //            + 2 * order;
   // int geom = H1FESpace_->GetFE(0)->GetGeomType();
   // const IntegrationRule * ir = &IntRules.Get(geom, irOrder);
   /*
   // Select surface attributes for Dirichlet BCs
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   non_k_bdr_.SetSize(pmesh.bdr_attributes.Max());
   ess_bdr_ = 1;   // All outer surfaces
   non_k_bdr_ = 1; // Surfaces without applied surface currents

   for (int i=0; i<kbcs.Size(); i++)
   {
      non_k_bdr_[kbcs[i]-1] = 0;
   }
   */
   ess_bdr_.SetSize(pmesh.bdr_attributes.Max());
   if ( dbcs_ != NULL )
   {
      if ( dbcs_->Size() == 1 && (*dbcs_)[0] == -1 )
      {
         ess_bdr_ = 1;
      }
      else
      {
         ess_bdr_ = 0;
         for (int i=0; i<dbcs_->Size(); i++)
         {
            ess_bdr_[(*dbcs_)[i]-1] = 1;
         }
      }
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }
   // Setup various coefficients
   /*
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
   */
   massCoef_ = new TransformedCoefficient(negOmega2Coef_, epsCoef_, prodFunc);
   posMassCoef_ = new TransformedCoefficient(omega2Coef_, epsCoef_, prodFunc);
   if ( sigmaCoef_ )
   {
      lossCoef_ = new TransformedCoefficient(omegaCoef_, sigmaCoef_, prodFunc);
      // gainCoef_ = new TransformedCoefficient(omegaCoef_, sigmaCoef_, prodFunc);
   }

   // Impedance of free space
   if ( abcs.Size() > 0 )
   {
      if ( myid_ == 0 && logging_ > 0 )
      {
         cout << "Creating Admittance Coefficient" << endl;
      }

      abc_marker_.SetSize(pmesh.bdr_attributes.Max());
      if ( abcs.Size() == 1 && abcs[0] < 0 )
      {
         // Mark all boundaries as absorbing
         abc_marker_ = 1;
      }
      else
      {
         // Mark select boundaries as absorbing
         abc_marker_ = 0;
         for (int i=0; i<abcs.Size(); i++)
         {
            abc_marker_[abcs[i]-1] = 1;
         }
      }
      if ( etaInvCoef_ == NULL )
      {
         etaInvCoef_ = new ConstantCoefficient(sqrt(epsilon0_/mu0_));
      }
      abcCoef_ = new TransformedCoefficient(negOmegaCoef_, etaInvCoef_,
                                            prodFunc);
   }

   // Volume Current Density
   if ( j_r_src_ != NULL )
   {
      jrCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                              j_r_src_);
   }
   else
   {
      Vector j(3); j = 0.0;
      jrCoef_ = new VectorConstantCoefficient(j);
   }
   if ( j_i_src_ != NULL )
   {
      jiCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                              j_i_src_);
   }
   else
   {
      Vector j(3); j = 0.0;
      jiCoef_ = new VectorConstantCoefficient(j);
   }
   /*
   // Magnetization
   if ( m_src_ != NULL )
   {
      mCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                             m_src_);
   }
   */
   // Bilinear Forms
   a1_ = new ParSesquilinearForm(HCurlFESpace_, conv_);
   a1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_), NULL);
   a1_->AddDomainIntegrator(new VectorFEMassIntegrator(*massCoef_), NULL);
   if ( lossCoef_ )
   {
      a1_->AddDomainIntegrator(NULL, new VectorFEMassIntegrator(*lossCoef_));
   }
   if ( abcCoef_ )
   {
      a1_->AddBoundaryIntegrator(NULL, new VectorFEMassIntegrator(*abcCoef_),
                                 abc_marker_);
   }

   b1_ = new ParBilinearForm(HCurlFESpace_);
   b1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));
   b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));
   if ( lossCoef_ )
   {
      b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*lossCoef_));
   }
   /*
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
   */

   // Build grid functions
   e_  = new ParComplexGridFunction(HCurlFESpace_);
   *e_ = 0.0;
   // e_r_  = new ParGridFunction(HCurlFESpace_);
   // e_i_  = new ParGridFunction(HCurlFESpace_);
   // b_  = new ParGridFunction(HDivFESpace_);
   // h_  = new ParGridFunction(HCurlFESpace_);
   // bd_ = new ParGridFunction(HCurlFESpace_);
   // j_r_ = new ParGridFunction(HCurlFESpace_);
   // j_i_ = new ParGridFunction(HCurlFESpace_);
   j_ = new ParComplexGridFunction(HCurlFESpace_);
   j_->ProjectCoefficient(*jrCoef_, *jiCoef_);
   /*
   jd_r_ = new ParLinearForm(HCurlFESpace_);
   jd_i_ = new ParLinearForm(HCurlFESpace_);

   if ( jrCoef_ )
   {
     jd_r_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*jrCoef_));
   }
   if ( jiCoef_ )
   {
     jd_i_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*jiCoef_));
   }
   */
   jd_ = new ParComplexLinearForm(HCurlFESpace_, conv_);
   jd_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*jrCoef_),
                            new VectorFEDomainLFIntegrator(*jiCoef_));
   jd_->real().Vector::operator=(0.0);
   jd_->imag().Vector::operator=(0.0);
   /*
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
   */
}

HertzSolver::~HertzSolver()
{
   delete jrCoef_;
   delete jiCoef_;
   delete erCoef_;
   delete eiCoef_;
   delete massCoef_;
   delete posMassCoef_;
   delete lossCoef_;
   // delete gainCoef_;
   delete abcCoef_;
   if ( ownsEtaInv_ ) { delete etaInvCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   // delete DivFreeProj_;
   // delete SurfCur_;

   // delete e_r_;
   // delete e_i_;
   delete e_;
   // delete b_;
   // delete h_;
   delete j_;
   // delete j_r_;
   // delete j_i_;
   // delete j_;
   // delete k_;
   // delete m_;
   // delete bd_;
   delete jd_;
   // delete jd_r_;
   // delete jd_i_;

   // delete grad_;
   // delete curl_;

   delete a1_;
   delete b1_;
   // delete curlMuInvCurl_;
   // delete hCurlMass_;
   // delete hDivHCurlMuInv_;
   // delete weakCurlMuInv_;

   // delete H1FESpace_;
   delete HCurlFESpace_;
   // delete HDivFESpace_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
HertzSolver::GetProblemSize()
{
   return 2 * HCurlFESpace_->GlobalTrueVSize();
}

void
HertzSolver::PrintSizes()
{
   // HYPRE_Int size_h1 = H1FESpace_->GlobalTrueVSize();
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   // HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      // cout << "Number of H1      unknowns: " << size_h1 << endl;
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      // cout << "Number of H(Div)  unknowns: " << size_rt << endl;
   }
}

void
HertzSolver::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << flush; }

   // a0_->Assemble();
   // a0_->Finalize();

   a1_->Assemble();
   a1_->Finalize();

   b1_->Assemble();
   b1_->Finalize();

   jd_->Assemble();
   /*
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
   */
   if ( myid_ == 0 && logging_ > 0 ) { cout << " done." << endl; }
}

void
HertzSolver::Update()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Updating ..." << endl; }

   // Inform the spaces that the mesh has changed
   // Note: we don't need to interpolate any GridFunctions on the new mesh
   // so we pass 'false' to skip creation of any transformation matrices.
   // H1FESpace_->Update(false);
   HCurlFESpace_->Update();
   // HDivFESpace_->Update(false);

   if ( ess_bdr_.Size() > 0 )
   {
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }

   // Inform the grid functions that the space has changed.
   e_->Update();
   // e_r_->Update();
   // e_i_->Update();
   // h_->Update();
   // b_->Update();
   // bd_->Update();
   jd_->Update();
   // jd_i_->Update();
   // if ( jr_ ) { jr_->Update(); }
   if ( j_  ) {  j_->Update(); }
   // if ( j_r_  ) {  j_r_->Update(); }
   // if ( j_i_  ) {  j_i_->Update(); }
   // if ( k_  ) {  k_->Update(); }
   // if ( m_  ) {  m_->Update(); }

   // Inform the bilinear forms that the space has changed.
   // a0_->Update();
   a1_->Update();
   b1_->Update();
   // curlMuInvCurl_->Update();
   // hCurlMass_->Update();
   // hDivHCurlMuInv_->Update();
   // if ( weakCurlMuInv_ ) { weakCurlMuInv_->Update(); }

   // Inform the other objects that the space has changed.
   // curl_->Update();
   // if ( grad_        ) { grad_->Update(); }
   // if ( DivFreeProj_ ) { DivFreeProj_->Update(); }
   // if ( SurfCur_     ) { SurfCur_->Update(); }
}

void
HertzSolver::Solve()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Running solver ... " << endl; }

   // *e_ = 0.0;

   /// For testing
   // e_->ProjectCoefficient(*jrCoef_, *jiCoef_);
   /*
   ComplexHypreParMatrix * A1 =
      a1_->ParallelAssemble();

   if ( A1->hasRealPart() ) { A1->real().Print("A1_real.mat"); }
   if ( A1->hasImagPart() ) { A1->imag().Print("A1_imag.mat"); }

   HypreParMatrix * A1C = A1->GetSystemMatrix();
   A1C->Print("A1_combined.mat");

   HypreParVector ra(*A1C); ra.Randomize(123);
   HypreParVector A1ra(*A1C);
   HypreParVector Ara(*A1C);
   HypreParVector diff(*A1C);

   A1->Mult(ra, A1ra);
   A1C->Mult(ra, Ara);

   subtract(A1ra, Ara, diff);

   ra.Print("r.vec");
   A1ra.Print("A1r.vec");
   Ara.Print("Ar.vec");
   diff.Print("diff.vec");

   double nrm = Ara.Norml2();
   double nrm1 = A1ra.Norml2();
   double nrmdiff = diff.Norml2();

   if ( myid_ == 0 )
   {
      cout << "norms " << nrm << " " << nrm1 << " " << nrmdiff << endl;
   }
   */
   /*
   HYPRE_Int size = HCurlFESpace_->GetTrueVSize();
   Vector E(2*size), RHS(2*size);
   jd_->ParallelAssemble(RHS);
   e_->ParallelProject(E);
   */
   OperatorHandle A1;
   Vector E, RHS;
   cout << "Norm of jd (pre-fls): " << jd_->Norml2() << endl;
   a1_->FormLinearSystem(ess_bdr_tdofs_, *e_, *jd_, A1, E, RHS);

   cout << "Norm of jd (post-fls): " << jd_->Norml2() << endl;
   cout << "Norm of RHS: " << RHS.Norml2() << endl;

   OperatorHandle PCOp;
   b1_->FormSystemMatrix(ess_bdr_tdofs_, PCOp);

   /*
   #ifdef MFEM_USE_SUPERLU
   SuperLURowLocMatrix A_SuperLU(*A1C);
   SuperLUSolver solver(MPI_COMM_WORLD);
   solver.SetOperator(A_SuperLU);
   solver.Mult(RHS, E);
   #endif
   #ifdef MFEM_USE_STRUMPACK
   STRUMPACKRowLocMatrix A_STRUMPACK(*A1C);
   STRUMPACKSolver solver(0, NULL, MPI_COMM_WORLD);
   solver.SetOperator(A_STRUMPACK);
   solver.Mult(RHS, E);
   #endif
   */
   /*
   MINRESSolver minres(HCurlFESpace_->GetComm());
   minres.SetOperator(*A1);
   minres.SetRelTol(1e-6);
   minres.SetMaxIter(5000);
   minres.SetPrintLevel(1);
   // pcg.SetPreconditioner(ams);
   minres.Mult(RHS, E);
   */
   switch (sol_)
   {
      case GMRES:
      {
         GMRESSolver gmres(HCurlFESpace_->GetComm());
         gmres.SetOperator(*A1.Ptr());
         gmres.SetRelTol(1e-4);
         gmres.SetMaxIter(10000);
         gmres.SetPrintLevel(1);

         gmres.Mult(RHS, E);
      }
      break;
      case FGMRES:
      {
         // HypreParMatrix * B1 = b1_->ParallelAssemble();

         // HypreAMS ams(*B1, HCurlFESpace_);
         HypreAMS amsr(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()),
                       HCurlFESpace_);
         ScaledOperator amsi(&amsr,
                             (conv_ == ComplexOperator::HERMITIAN)?1.0:-1.0);

         BlockDiagonalPreconditioner BDP(blockTrueOffsets_);
         BDP.SetDiagonalBlock(0,&amsr);
         BDP.SetDiagonalBlock(1,&amsi);
         BDP.owns_blocks = 0;

         FGMRESSolver fgmres(HCurlFESpace_->GetComm());
         fgmres.SetPreconditioner(BDP);
         fgmres.SetOperator(*A1.Ptr());
         fgmres.SetRelTol(1e-4);
         fgmres.SetMaxIter(1000);
         fgmres.SetPrintLevel(1);

         fgmres.Mult(RHS, E);

         // delete B1;
      }
      break;
#ifdef MFEM_USE_SUPERLU
      case SUPERLU:
      {
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         SuperLURowLocMatrix A_SuperLU(*A1C);
         SuperLUSolver solver(MPI_COMM_WORLD);
         solver.SetOperator(A_SuperLU);
         solver.Mult(RHS, E);
         delete A1C;
         // delete A1Z;
      }
      break;
#endif
#ifdef MFEM_USE_STRUMPACK
      case STRUMPACK:
      {
         //A1.SetOperatorOwner(false);
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         STRUMPACKRowLocMatrix A_STRUMPACK(*A1C);
         STRUMPACKSolver solver(0, NULL, MPI_COMM_WORLD);
         solver.SetPrintFactorStatistics(true);
         solver.SetPrintSolveStatistics(false);
         solver.SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
         solver.SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
         solver.DisableMatching();
         solver.SetOperator(A_STRUMPACK);
         solver.SetFromCommandLine();
         solver.Mult(RHS, E);
         delete A1C;
         // delete A1Z;
      }
      break;
#endif
      default:
         break;
   };

   e_->Distribute(E);

   // delete A1;
   // delete A1C;
   /*
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
   */
   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << flush; }

   if ( myid_ == 0 && logging_ > 0 ) { cout << " Solver done. " << endl; }
}

void
HertzSolver::GetErrorEstimates(Vector & errors)
{
   if ( myid_ == 0 && logging_ > 0 )
   { cout << "Estimating Error ... " << flush; }

   // Space for the discontinuous (original) flux
   CurlCurlIntegrator flux_integrator(*muInvCoef_);
   RT_FECollection flux_fec(order_-1, pmesh_->SpaceDimension());
   ParFiniteElementSpace flux_fes(pmesh_, &flux_fec);

   // Space for the smoothed (conforming) flux
   double norm_p = 1;
   ND_FECollection smooth_flux_fec(order_, pmesh_->Dimension());
   ParFiniteElementSpace smooth_flux_fes(pmesh_, &smooth_flux_fec);

   L2ZZErrorEstimator(flux_integrator, e_->real(),
                      smooth_flux_fes, flux_fes, errors, norm_p);

   if ( myid_ == 0 && logging_ > 0 ) { cout << "done." << endl; }
}

void
HertzSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("Re(E)", &e_->real());
   visit_dc.RegisterField("Im(E)", &e_->imag());
   // visit_dc.RegisterField("Er", e_r_);
   // visit_dc.RegisterField("Ei", e_i_);
   // visit_dc.RegisterField("B", b_);
   // visit_dc.RegisterField("H", h_);
   if ( j_ )
   {
      visit_dc.RegisterField("Re(J)", &j_->real());
      visit_dc.RegisterField("Im(J)", &j_->imag());
   }
   // if ( j_r_ ) { visit_dc.RegisterField("Jr", j_r_); }
   // if ( j_i_ ) { visit_dc.RegisterField("Ji", j_i_); }
   // if ( k_ ) { visit_dc.RegisterField("K", k_); }
   // if ( m_ ) { visit_dc.RegisterField("M", m_); }
   // if ( SurfCur_ ) { visit_dc.RegisterField("Psi", SurfCur_->GetPsi()); }
}

void
HertzSolver::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      if ( j_ )
      {
         j_->ProjectCoefficient(*jrCoef_, *jiCoef_);
      }

      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

      if (myid_ == 0) { cout << " done." << endl; }
   }
}

void
HertzSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl; }

   socks_["Er"] = new socketstream;
   socks_["Er"]->precision(8);

   socks_["Ei"] = new socketstream;
   socks_["Ei"]->precision(8);

   // socks_["B"] = new socketstream;
   // socks_["B"]->precision(8);

   // socks_["H"] = new socketstream;
   // socks_["H"]->precision(8);

   if ( j_ )
   {
      socks_["Jr"] = new socketstream;
      socks_["Jr"]->precision(8);

      socks_["Ji"] = new socketstream;
      socks_["Ji"]->precision(8);
   }
   /*
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
   */
   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl; }
}

void
HertzSolver::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   VisualizeField(*socks_["Er"], vishost, visport,
                  e_->real(), "Electric Field, Re(E)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["Ei"], vishost, visport,
                  e_->imag(), "Electric Field, Im(E)", Wx, Wy, Ww, Wh);
   /*
   Wx += offx;
   VisualizeField(*socks_["B"], vishost, visport,
                  *b_, "Magnetic Flux Density (B)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["H"], vishost, visport,
                  *h_, "Magnetic Field (H)", Wx, Wy, Ww, Wh);
   Wx += offx;
   */
   Wx = 0; Wy += offy; // next line

   if ( j_ )
   {
      j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

      VisualizeField(*socks_["Jr"], vishost, visport,
                     j_->real(), "Current Density, Re(J)", Wx, Wy, Ww, Wh);
      Wx += offx;
      VisualizeField(*socks_["Ji"], vishost, visport,
                     j_->imag(), "Current Density, Im(J)", Wx, Wy, Ww, Wh);
   }

   Wx = 0; Wy += offy; // next line
   /*
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
   */
   if (myid_ == 0) { cout << " done." << endl; }
}
/*
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
*/
} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
