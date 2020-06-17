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
                         HertzSolver::SolverType sol, SolverOptions & sOpts,
                         HertzSolver::PrecondType prec,
                         ComplexOperator::Convention conv,
                         Coefficient & epsCoef,
                         Coefficient & muInvCoef,
                         Coefficient * sigmaCoef,
                         Coefficient * etaInvCoef,
                         Array<int> & abcs,
                         Array<int> & dbcs,
                         void (*e_r_bc )(const Vector&, Vector&),
                         void (*e_i_bc )(const Vector&, Vector&),
                         void (*j_r_src)(const Vector&, Vector&),
                         void (*j_i_src)(const Vector&, Vector&))
   : myid_(0),
     num_procs_(1),
     order_(order),
     logging_(1),
     sol_(sol),
     solOpts_(sOpts),
     prec_(prec),
     conv_(conv),
     ownsEtaInv_(etaInvCoef == NULL),
     freq_(freq),
     pmesh_(&pmesh),
     HCurlFESpace_(NULL),
     a1_(NULL),
     b1_(NULL),
     e_(NULL),
     e_t_(NULL),
     j_(NULL),
     jd_(NULL),
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
     abcCoef_(NULL),
     posAbcCoef_(NULL),
     jrCoef_(NULL),
     jiCoef_(NULL),
     erCoef_(NULL),
     eiCoef_(NULL),
     j_r_src_(j_r_src),
     j_i_src_(j_i_src),
     e_r_bc_(e_r_bc),
     e_i_bc_(e_i_bc),
     dbcs_(&dbcs),
     visit_dc_(NULL)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order Nedelec finite elements.
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());

   // Set the size of the 2x2 block representation of the complex linear system
   blockTrueOffsets_.SetSize(3);
   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   // Setup Dirichlet BC
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

      if (e_r_bc_)
      {
         erCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                                 e_r_bc_);
         if (e_i_bc_ == NULL)
         {
            Vector e(3); e = 0.0;
            eiCoef_ = new VectorConstantCoefficient(e);
         }
      }
      if (e_i_bc_)
      {
         eiCoef_ = new VectorFunctionCoefficient(pmesh_->SpaceDimension(),
                                                 e_i_bc_);
         if (e_r_bc_ == NULL)
         {
            Vector e(3); e = 0.0;
            erCoef_ = new VectorConstantCoefficient(e);
         }
      }
   }

   // Setup various coefficients
   massCoef_ = new TransformedCoefficient(negOmega2Coef_, epsCoef_, prodFunc);
   posMassCoef_ = new TransformedCoefficient(omega2Coef_, epsCoef_, prodFunc);
   if ( sigmaCoef_ )
   {
      lossCoef_ = new TransformedCoefficient(omegaCoef_, sigmaCoef_, prodFunc);
   }

   // Impedance of free space for the Absorbing boundary condition
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
      posAbcCoef_ = new TransformedCoefficient(omegaCoef_, etaInvCoef_,
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

   // Bilinear Forms
   // Primary system operator
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

   // Operator used with the perconditioner
   b1_ = new ParBilinearForm(HCurlFESpace_);
   b1_->AddDomainIntegrator(new CurlCurlIntegrator(*muInvCoef_));
   b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*posMassCoef_));
   if ( lossCoef_ )
   {
      b1_->AddDomainIntegrator(new VectorFEMassIntegrator(*lossCoef_));
   }
   if ( abcCoef_ )
   {
      b1_->AddBoundaryIntegrator(new VectorFEMassIntegrator(*posAbcCoef_),
                                 abc_marker_);
   }

   // Build grid functions
   // The solution vector is the Electric field
   e_   = new ParComplexGridFunction(HCurlFESpace_);
   e_t_ = new ParGridFunction(HCurlFESpace_);
   if (erCoef_ && eiCoef_)
   {
      e_->ProjectCoefficient(*erCoef_, *eiCoef_);
   }
   else
   {
      *e_ = 0.0;
   }

   // A GridFunction to visualize the volumetric current density
   j_ = new ParComplexGridFunction(HCurlFESpace_);
   j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

   // A LineatForm representation of the current denisty for the RHS
   jd_ = new ParComplexLinearForm(HCurlFESpace_, conv_);
   jd_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*jrCoef_),
                            new VectorFEDomainLFIntegrator(*jiCoef_));
   jd_->real().Vector::operator=(0.0);
   jd_->imag().Vector::operator=(0.0);
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
   delete abcCoef_;
   delete posAbcCoef_;
   if ( ownsEtaInv_ ) { delete etaInvCoef_; }
   delete omegaCoef_;
   delete negOmegaCoef_;
   delete omega2Coef_;
   delete negOmega2Coef_;

   delete e_;
   delete e_t_;
   delete j_;
   delete jd_;

   delete a1_;
   delete b1_;

   delete HCurlFESpace_;

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
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
   }
}

void
HertzSolver::Assemble()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Assembling ..." << flush; }

   a1_->Assemble();
   a1_->Finalize();

   b1_->Assemble();
   b1_->Finalize();

   jd_->Assemble();

   if ( myid_ == 0 && logging_ > 0 ) { cout << " done." << endl; }
}

void
HertzSolver::Update()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Updating ..." << endl; }

   // Inform the spaces that the mesh has changed
   HCurlFESpace_->Update();

   if ( ess_bdr_.Size() > 0 )
   {
      HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
   }

   blockTrueOffsets_[0] = 0;
   blockTrueOffsets_[1] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_[2] = HCurlFESpace_->TrueVSize();
   blockTrueOffsets_.PartialSum();

   // Inform the grid functions that the space has changed.
   e_->Update();
   if (erCoef_ && eiCoef_)
   {
      e_->ProjectCoefficient(*erCoef_, *eiCoef_);
   }

   j_->Update();
   j_->ProjectCoefficient(*jrCoef_, *jiCoef_);

   jd_->Update();

   // Inform the bilinear forms that the space has changed.
   a1_->Update();
   b1_->Update();
}

void
HertzSolver::Solve()
{
   if ( myid_ == 0 && logging_ > 0 ) { cout << "Running solver ... " << endl; }

   OperatorHandle A1;
   Vector E, RHS;

   a1_->FormLinearSystem(ess_bdr_tdofs_, *e_, *jd_, A1, E, RHS);

   OperatorHandle PCOp;
   b1_->FormSystemMatrix(ess_bdr_tdofs_, PCOp);

   tic_toc.Clear();
   tic_toc.Start();

   Operator * pcr = NULL;
   Operator * pci = NULL;
   BlockDiagonalPreconditioner * BDP = NULL;

   if (sol_ == FGMRES || sol_ == MINRES)
   {
      switch (prec_)
      {
         case INVALID_PC:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "No Preconditioner Requested" << endl;
            }
            break;
         case DIAG_SCALE:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Diagonal Scaling Preconditioner Requested" << endl;
            }
            pcr = new HypreDiagScale(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
            break;
         case PARASAILS:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "ParaSails Preconditioner Requested" << endl;
            }
            pcr = new HypreParaSails(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
            dynamic_cast<HypreParaSails*>(pcr)->SetSymmetry(1);
            break;
         case EUCLID:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "Euclid Preconditioner Requested" << endl;
            }
            pcr = new HypreEuclid(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()));
            if (solOpts_.euLvl != 1)
            {
               HypreSolver * pc = dynamic_cast<HypreSolver*>(pcr);
               HYPRE_EuclidSetLevel(*pc, solOpts_.euLvl);
            }
            break;
         case AMS:
            if ( myid_ == 0 && logging_ > 0 )
            {
               cout << "AMS Preconditioner Requested" << endl;
            }
            pcr = new HypreAMS(dynamic_cast<HypreParMatrix&>(*PCOp.Ptr()),
                               HCurlFESpace_);
            break;
         default:
            MFEM_ABORT("Requested preconditioner is not available.");
            break;
      }
      pci = pcr;

      if (pcr)
      {
         BDP = new BlockDiagonalPreconditioner(blockTrueOffsets_);
         BDP->SetDiagonalBlock(0, pcr);
         BDP->SetDiagonalBlock(1, pci);
         BDP->owns_blocks = 0;
      }
   }

   switch (sol_)
   {
      case GMRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "GMRES Solver Requested" << endl;
         }
         GMRESSolver gmres(HCurlFESpace_->GetComm());
         gmres.SetOperator(*A1.Ptr());
         gmres.SetRelTol(solOpts_.relTol);
         gmres.SetMaxIter(solOpts_.maxIter);
         gmres.SetKDim(solOpts_.kDim);
         gmres.SetPrintLevel(solOpts_.printLvl);

         gmres.Mult(RHS, E);
      }
      break;
      case FGMRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "FGMRES Solver Requested" << endl;
         }
         FGMRESSolver fgmres(HCurlFESpace_->GetComm());
         if (BDP) { fgmres.SetPreconditioner(*BDP); }
         fgmres.SetOperator(*A1.Ptr());
         fgmres.SetRelTol(solOpts_.relTol);
         fgmres.SetMaxIter(solOpts_.maxIter);
         fgmres.SetKDim(solOpts_.kDim);
         fgmres.SetPrintLevel(solOpts_.printLvl);

         fgmres.Mult(RHS, E);
      }
      break;
      case MINRES:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "MINRES Solver Requested" << endl;
         }
         MINRESSolver minres(HCurlFESpace_->GetComm());
         if (BDP) { minres.SetPreconditioner(*BDP); }
         minres.SetOperator(*A1.Ptr());
         minres.SetRelTol(solOpts_.relTol);
         minres.SetMaxIter(solOpts_.maxIter);
         minres.SetPrintLevel(solOpts_.printLvl);

         minres.Mult(RHS, E);
      }
      break;
#ifdef MFEM_USE_SUPERLU
      case SUPERLU:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "SuperLU Solver Requested" << endl;
         }
         ComplexHypreParMatrix * A1Z = A1.As<ComplexHypreParMatrix>();
         HypreParMatrix * A1C = A1Z->GetSystemMatrix();
         SuperLURowLocMatrix A_SuperLU(*A1C);
         SuperLUSolver solver(MPI_COMM_WORLD);
         solver.SetOperator(A_SuperLU);
         solver.Mult(RHS, E);
         delete A1C;
      }
      break;
#endif
#ifdef MFEM_USE_STRUMPACK
      case STRUMPACK:
      {
         if ( myid_ == 0 && logging_ > 0 )
         {
            cout << "STRUMPACK Solver Requested" << endl;
         }
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
      }
      break;
#endif
      default:
         break;
   };

   tic_toc.Stop();

   e_->Distribute(E);

   delete BDP;
   if (pci != pcr) { delete pci; }
   delete pcr;

   if ( myid_ == 0 && logging_ > 0 )
   {
      cout << " Solver done in " << tic_toc.RealTime() << " seconds." << endl;
   }
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

   if ( j_ )
   {
      visit_dc.RegisterField("Re(J)", &j_->real());
      visit_dc.RegisterField("Im(J)", &j_->imag());
   }
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

   if ( j_ )
   {
      socks_["Jr"] = new socketstream;
      socks_["Jr"]->precision(8);

      socks_["Ji"] = new socketstream;
      socks_["Ji"]->precision(8);
   }

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

   if (myid_ == 0) { cout << " done." << endl; }
}

void
HertzSolver::DisplayAnimationToGLVis()
{
   if (myid_ == 0) { cout << "Sending animation data to GLVis ..." << flush; }

   Vector zeroVec(3); zeroVec = 0.0;
   VectorConstantCoefficient zeroCoef(zeroVec);

   double norm_r = e_->real().ComputeMaxError(zeroCoef);
   double norm_i = e_->imag().ComputeMaxError(zeroCoef);

   *e_t_ = e_->real();

   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock(vishost, visport);
   sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
   sol_sock.precision(8);
   sol_sock << "solution\n" << *pmesh_ << *e_t_
            << "window_title 'Harmonic Solution (t = 0.0 T)'"
            << "valuerange 0.0 " << max(norm_r, norm_i) << "\n"
            << "autoscale off\n"
            << "keys cvvv\n"
            << "pause\n" << flush;
   if (myid_ == 0)
      cout << "GLVis visualization paused."
           << " Press space (in the GLVis window) to resume it.\n";
   int num_frames = 24;
   int i = 0;
   while (sol_sock)
   {
      double t = (double)(i % num_frames) / num_frames;
      ostringstream oss;
      oss << "Harmonic Solution (t = " << t << " T)";

      add( cos( 2.0 * M_PI * t), e_->real(),
           sin( 2.0 * M_PI * t), e_->imag(), *e_t_);
      sol_sock << "parallel " << num_procs_ << " " << myid_ << "\n";
      sol_sock << "solution\n" << *pmesh_ << *e_t_
               << "window_title '" << oss.str() << "'" << flush;
      i++;
   }
}

} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
