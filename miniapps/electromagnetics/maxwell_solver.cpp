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

#include "maxwell_solver.hpp"

using namespace std;

namespace mfem
{

using namespace miniapps;

namespace electromagnetics
{

MaxwellSolver::MaxwellSolver(ParMesh & pmesh, int order,
                             double (*eps     )(const Vector&),
                             double (*muInv   )(const Vector&),
                             void   (*j_src   )(const Vector&, double, Vector&),
                             Array<int> & dbcs,
                             void   (*dEdt_bc )(const Vector&, double, Vector&))
   : myid_(0),
     num_procs_(1),
     order_(order),
     pmesh_(&pmesh),
     visit_dc_(NULL),
     HCurlFESpace_(NULL),
     HDivFESpace_(NULL),
     hCurlMassEps_(NULL),
     hDivMassMuInv_(NULL),
     weakCurlMuInv_(NULL),
     Curl_(NULL),
     pcg_(NULL),
     e_(NULL),
     b_(NULL),
     j_(NULL),
     de_(NULL),
     rhs_(NULL),
     jd_(NULL),
     M1Eps_(NULL),
     M2MuInv_(NULL),
     NegCurl_(NULL),
     WeakCurlMuInv_(NULL),
     E_(NULL),
     B_(NULL),
     HD_(NULL),
     JD_(NULL),
     RHS_(NULL),
     epsCoef_(NULL),
     muInvCoef_(NULL),
     eCoef_(NULL),
     bCoef_(NULL),
     jCoef_(NULL),
     dEdtBCCoef_(NULL),
     eps_(eps),
     muInv_(muInv),
     j_src_(j_src),
     dEdt_bc_(dEdt_bc),
     dbcs_(&dbcs)
{
   // Initialize MPI variables
   MPI_Comm_size(pmesh_->GetComm(), &num_procs_);
   MPI_Comm_rank(pmesh_->GetComm(), &myid_);

   // Define compatible parallel finite element spaces on the parallel
   // mesh. Here we use arbitrary order H1, Nedelec, and Raviart-Thomas finite
   // elements.
   // H1FESpace_    = new H1_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HCurlFESpace_ = new ND_ParFESpace(pmesh_,order,pmesh_->Dimension());
   HDivFESpace_  = new RT_ParFESpace(pmesh_,order,pmesh_->Dimension());

   this->height = HCurlFESpace_->GlobalTrueVSize();
   this->width  = HDivFESpace_->GlobalTrueVSize();

   // Electric permittivity
   if ( eps_ == NULL )
   {
      epsCoef_ = new ConstantCoefficient(epsilon0_);
   }
   else
   {
      if ( myid_ == 0 )
      {
         cout << "Creating Permittivity Coefficient" << endl;
      }
      epsCoef_ = new FunctionCoefficient(eps_);
   }

   // Inverse of the magnetic permeability
   if ( muInv_ == NULL )
   {
      muInvCoef_ = new ConstantCoefficient(1.0/mu0_);
   }
   else
   {
      if ( myid_ == 0 )
      {
         cout << "Creating Permeability Coefficient" << endl;
      }
      muInvCoef_ = new FunctionCoefficient(muInv_);
   }

   // Electric Field Boundary Condition
   if ( dbcs.Size() > 0 )
   {
      HCurlFESpace_->GetEssentialTrueDofs(dbcs, dbc_dofs_);

      if ( dEdt_bc_ != NULL )
      {
         dEdtBCCoef_ = new VectorFunctionCoefficient(3,dEdt_bc_);
      }
      else
      {
         Vector ebc(3); ebc = 0.0;
         dEdtBCCoef_ = new VectorConstantCoefficient(ebc);
      }
   }

   // Bilinear Forms
   hCurlMassEps_  = new ParBilinearForm(HCurlFESpace_);
   hDivMassMuInv_ = new ParBilinearForm(HDivFESpace_);
   weakCurlMuInv_ = new ParMixedBilinearForm(HDivFESpace_,HCurlFESpace_);

   hCurlMassEps_->AddDomainIntegrator(new VectorFEMassIntegrator(*epsCoef_));
   hDivMassMuInv_->AddDomainIntegrator(new VectorFEMassIntegrator(*muInvCoef_));
   weakCurlMuInv_->AddDomainIntegrator(
      new MixedVectorWeakCurlIntegrator(*muInvCoef_));

   // Assemble Matrices
   hCurlMassEps_->Assemble();
   hDivMassMuInv_->Assemble();
   weakCurlMuInv_->Assemble();

   hCurlMassEps_->Finalize();
   hDivMassMuInv_->Finalize();
   weakCurlMuInv_->Finalize();

   // Create Linear Algebra Matrices
   M1Eps_   = hCurlMassEps_->ParallelAssemble();
   M2MuInv_ = hDivMassMuInv_->ParallelAssemble();
   WeakCurlMuInv_ = weakCurlMuInv_->ParallelAssemble();
   /*
   {
     ParMixedBilinearForm curlMuInv(HCurlFESpace_,HDivFESpace_);
     curlMuInv.AddDomainIntegrator(new VectorFECurlIntegrator(*muInvCoef_));
     curlMuInv.Assemble();
     curlMuInv.Finalize();
     HypreParMatrix * CurlMuInv = curlMuInv.ParallelAssemble();
     HypreParMatrix * WeakCurlMuInv = weakCurlMuInv_->ParallelAssemble();

     CurlMuInv->Print("CurlMuInv.mat");
     WeakCurlMuInv->Print("WeakCurlMuInv.mat");
     // delete CurlMuInv;
     // delete WeakCurlMuInv;
   }
   */
   /*
   {
      ConstantCoefficient etaCoef(1.0);
      ParBilinearForm m1eta(HCurlFESpace_);
      m1eta.AddBoundaryIntegrator(new VectorFEMassIntegrator(etaCoef));
      m1eta.Assemble();
      m1eta.Finalize();
      HypreParMatrix * M1eta = m1eta.ParallelAssemble();
      M1eta->Print("M1eta.mat");
      delete M1eta;
   }
   */

   Curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);
   Curl_->Assemble();
   Curl_->Finalize();
   NegCurl_ = Curl_->ParallelAssemble();
   // NegCurl_->Print("T12.mat");
   *NegCurl_ *= -1.0; // Beware this modifies the matrix stored within
   // the Curl_ object.

   // HypreParMatrix * MT12 = ParMult(M2MuInv_,NegCurl_);
   // MT12->Print("M2MuInvT12.mat");
   // delete MT12;

   // Build grid functions
   e_   = new ParGridFunction(HCurlFESpace_);
   de_  = new ParGridFunction(HCurlFESpace_);
   rhs_ = new ParGridFunction(HCurlFESpace_);
   b_   = new ParGridFunction(HDivFESpace_);

   E_ = e_->ParallelProject();
   B_ = b_->ParallelProject();

   HD_  = new HypreParVector(HDivFESpace_);
   RHS_ = new HypreParVector(HCurlFESpace_);

   // Eliminate essential BC dofs from M1Eps
   *de_ = 0.0;
   hCurlMassEps_->FormLinearSystem(dbc_dofs_,*de_,*rhs_,*M1Eps_,*E_,*RHS_);

   // Create Solver
   diagScale_ = new HypreDiagScale(*M1Eps_);

   pcg_ = new HyprePCG(*M1Eps_);
   pcg_->SetTol(1e-12);
   pcg_->SetMaxIter(1000);
   pcg_->SetLogging(0);
   pcg_->SetPreconditioner(*diagScale_);

   /*
   {
     // Just testing

     B_->Randomize(123);
     *b_ = *B_;

     M2MuInv_->Mult(*B_,*HD_);
     NegCurl_->MultTranspose(*HD_,*RHS_,-1.0,0.0);
     RHS_->Print("RHS.vec");

     weakCurlMuInv_->Mult(*b_,*e_);

     e_->ParallelAssemble(*E_);
     E_->Print("E.vec");
     E_->Add(1.0,*RHS_);
     double nrm = E_->Norml2();

     cout << "Norm of diff: " << nrm << endl;
   }
   */
   if ( j_src_)
   {
      if ( myid_ == 0 )
      {
         cout << "Creating Current Source" << endl;
      }
      jCoef_ = new VectorFunctionCoefficient(3,j_src_);

      j_  = new ParGridFunction(HCurlFESpace_);
      j_->ProjectCoefficient(*jCoef_);

      jd_ = new ParLinearForm(HCurlFESpace_);
      jd_->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*jCoef_));
      jd_->Assemble();

      JD_ = new HypreParVector(HCurlFESpace_);
   }
}

MaxwellSolver::~MaxwellSolver()
{
   delete epsCoef_;
   delete muInvCoef_;
   delete jCoef_;
   delete dEdtBCCoef_;

   delete E_;
   delete B_;
   delete HD_;
   delete RHS_;

   delete e_;
   delete b_;
   delete j_;
   delete de_;
   delete rhs_;
   delete jd_;

   delete Curl_;

   delete pcg_;
   delete diagScale_;

   delete M1Eps_;
   delete M2MuInv_;

   delete hCurlMassEps_;
   delete hDivMassMuInv_;
   delete weakCurlMuInv_;

   delete HCurlFESpace_;
   delete HDivFESpace_;

   map<string,socketstream*>::iterator mit;
   for (mit=socks_.begin(); mit!=socks_.end(); mit++)
   {
      delete mit->second;
   }
}

HYPRE_Int
MaxwellSolver::GetProblemSize()
{
   return HCurlFESpace_->GlobalTrueVSize();
}

void
MaxwellSolver::PrintSizes()
{
   HYPRE_Int size_nd = HCurlFESpace_->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace_->GlobalTrueVSize();
   if (myid_ == 0)
   {
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl << flush;
   }
}

void
MaxwellSolver::SetInitialEField(VectorCoefficient & EFieldCoef)
{
   eCoef_ = &EFieldCoef;
   e_->ProjectCoefficient(EFieldCoef);
   e_->ParallelProject(*E_);
}

void
MaxwellSolver::SetInitialBField(VectorCoefficient & BFieldCoef)
{
   bCoef_ = &BFieldCoef;
   b_->ProjectCoefficient(BFieldCoef);
   b_->ParallelProject(*B_);
}

void
MaxwellSolver::Mult(const Vector &B, Vector &dE) const
{
   // cout << "MaxwellSolver::Mult 0" << endl;
   // M2MuInv_->Mult(B,*HD_);
   // cout << "MaxwellSolver::Mult 1" << endl;
   // NegCurl_->MultTranspose(*HD_,*RHS_,-1.0,0.0);
   // cout << "MaxwellSolver::Mult 2" << endl;
   //WeakCurlMuInv_->Mult(B,*RHS_);

   *b_ = B;
   weakCurlMuInv_->Mult(*b_,*rhs_);

   if ( jd_ )
   {
      jCoef_->SetTime(t); // Is member data from mfem::TimeDependentOperator
      jd_->Assemble();
      // jd_->ParallelAssemble(*JD_);
      // *RHS_ -= *JD_;
      *rhs_ -= *jd_;
   }

   if ( dEdtBCCoef_ && dbcs_ )
   {
      de_->ProjectBdrCoefficientTangent(*dEdtBCCoef_,*dbcs_);
   }

   hCurlMassEps_->FormLinearSystem(dbc_dofs_,*de_,*rhs_,*M1Eps_,dE,*RHS_);
   /*
   if (diagScale_ == NULL)
   {
     diagScale_ = new HypreDiagScale(*M1Eps_);
   }
   if (pcg_ == NULL)
   {
     pcg_ = new HyprePCG(*M1Eps_);
     pcg_->SetTol(1.0e-12);
     pcg_->SetMaxIter(200);
     pcg_->SetPrintLevel(0);
     pcg_->SetPreconditioner(*diagScale_);
   }
   */
   pcg_->Mult(*RHS_,dE);
   // hCurlMassEps_->ReconverFEMSolution(dE,*rhs_,*de_);

   // cout << "MaxwellSolver::Mult 3" << endl;
}

void
MaxwellSolver::SyncGridFuncs()
{
   *e_ = *E_;
   *b_ = *B_;
}

double
MaxwellSolver::GetMaximumTimeStep() const
{
   HypreParVector * v0 = new HypreParVector(HCurlFESpace_);
   HypreParVector * v1 = new HypreParVector(HCurlFESpace_);
   HypreParVector * u0 = new HypreParVector(HDivFESpace_);
   HypreParVector * vTmp = NULL;

   v0->Randomize(1234);

   int iter = 0, nstep = 20;
   double dt0 = 1.0, dt1 = 1.0, change = 1.0, ptol = 0.001;

   while ( iter < nstep && change > ptol )
   {
      double normV0 = InnerProduct(*v0,*v0);
      *v0 /= sqrt(normV0);

      NegCurl_->Mult(*v0,*u0);
      M2MuInv_->Mult(*u0,*HD_);
      NegCurl_->MultTranspose(*HD_,*RHS_);

      pcg_->Mult(*RHS_,*v1);

      double lambda = InnerProduct(*v0,*v1);
      dt1 = 2.0/sqrt(lambda);
      change = fabs((dt1-dt0)/dt0);
      dt0 = dt1;

      if ( myid_ == 0 )
      {
         cout << iter << ":  " << dt0 << " " << change << endl;
      }

      vTmp = v0;
      v0   = v1;
      v1   = vTmp;

      iter++;
   }

   delete v0;
   delete v1;
   delete u0;

   return dt0;
}

double
MaxwellSolver::GetEnergy() const
{
   double energy = 0.0;

   M1Eps_->Mult(*E_,*RHS_);
   M2MuInv_->Mult(*B_,*HD_);

   energy = InnerProduct(*E_,*RHS_) + InnerProduct(*B_,*HD_);

   return 0.5 * energy;
}

void
MaxwellSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("E", e_);
   visit_dc.RegisterField("B", b_);
   if ( j_ )
   {
      visit_dc.RegisterField("J", j_);
   }
}

void
MaxwellSolver::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      // if (myid_ == 0) { cout << "Writing VisIt files ..." << flush; }

      if ( j_ )
      {
         jCoef_->SetTime(t); // Is member data from mfem::TimeDependentOperator
         j_->ProjectCoefficient(*jCoef_);
      }

      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(t);
      visit_dc_->Save();

      // if (myid_ == 0) { cout << " " << endl << flush; }
   }
}

void
MaxwellSolver::InitializeGLVis()
{
   if ( myid_ == 0 ) { cout << "Opening GLVis sockets." << endl << flush; }

   socks_["E"] = new socketstream;
   socks_["E"]->precision(8);

   socks_["B"] = new socketstream;
   socks_["B"]->precision(8);

   if ( j_ )
   {
      socks_["J"] = new socketstream;
      socks_["J"]->precision(8);
   }

   if ( myid_ == 0 ) { cout << "GLVis sockets open." << endl << flush; }
}

void
MaxwellSolver::DisplayToGLVis()
{
   if (myid_ == 0) { cout << "Sending data to GLVis ..." << flush; }

   char vishost[] = "localhost";
   int  visport   = 19916;

   int Wx = 0, Wy = 0; // window position
   int Ww = 350, Wh = 350; // window size
   int offx = Ww+10, offy = Wh+45; // window offsets

   VisualizeField(*socks_["E"], vishost, visport,
                  *e_, "Electric Field (E)", Wx, Wy, Ww, Wh);
   Wx += offx;

   VisualizeField(*socks_["B"], vishost, visport,
                  *b_, "Magnetic Flux Density (B)", Wx, Wy, Ww, Wh);

   if ( j_ )
   {
      Wx = 0;
      Wy += offy;

      jCoef_->SetTime(t); // Is member data from mfem::TimeDependentOperator
      j_->ProjectCoefficient(*jCoef_);

      VisualizeField(*socks_["J"], vishost, visport,
                     *j_, "Current Density (J)", Wx, Wy, Ww, Wh);
   }
   if (myid_ == 0) { cout << " " << flush; }
}

/// Returns the largest number less than or equal to dt which is of the form
///    ( p / n ) * 10^m
/// Where m, n, and p are integers: n is given (must be greater than 1), m is
/// arbitrary, and p is in the range 1 <= p < n.
/*
double
MaxwellSolver::SnapTimeStep(int n, double dt)
{
  MFEM_ASSERT(n > 1,"The integer must be greater than one.");
  double a = log10(n);
  double b = 1.0 + log10(dt) / a;
  int    c = (int)floor( a * b );
  double d = b - c / a;
  return floor(pow(n,d))*pow(10.0,c)/n;
}
*/
} // namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
