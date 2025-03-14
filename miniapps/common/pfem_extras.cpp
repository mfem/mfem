// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "pfem_extras.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace common
{

H1_ParFESpace::H1_ParFESpace(ParMesh *m,
                             const int p, const int space_dim, const int type,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new H1_FECollection(p,space_dim,type),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

H1_ParFESpace::~H1_ParFESpace()
{
   delete FEC_;
}

ND_ParFESpace::ND_ParFESpace(ParMesh *m, const int p, const int space_dim,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new ND_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

ND_ParFESpace::~ND_ParFESpace()
{
   delete FEC_;
}

RT_ParFESpace::RT_ParFESpace(ParMesh *m, const int p, const int space_dim,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new RT_FECollection(p-1,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

RT_ParFESpace::~RT_ParFESpace()
{
   delete FEC_;
}

L2_ParFESpace::L2_ParFESpace(ParMesh *m, const int p, const int space_dim,
                             int vdim, int order)
   : ParFiniteElementSpace(m, new L2_FECollection(p,space_dim),vdim,order)
{
   FEC_ = this->FiniteElementSpace::fec;
}

L2_ParFESpace::~L2_ParFESpace()
{
   delete FEC_;
}

ParDiscreteInterpolationOperator::~ParDiscreteInterpolationOperator()
{}

ParDiscreteGradOperator::ParDiscreteGradOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
   : ParDiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new GradientInterpolator);
}

ParDiscreteCurlOperator::ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
   : ParDiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new CurlInterpolator);
}

ParDiscreteDivOperator::ParDiscreteDivOperator(ParFiniteElementSpace *dfes,
                                               ParFiniteElementSpace *rfes)
   : ParDiscreteInterpolationOperator(dfes, rfes)
{
   this->AddDomainInterpolator(new DivergenceInterpolator);
}

IrrotationalNDProjector
::IrrotationalNDProjector(ParFiniteElementSpace   & H1FESpace,
                          ParFiniteElementSpace   & HCurlFESpace,
                          const int               & irOrder,
                          ParBilinearForm         * s0,
                          ParMixedBilinearForm    * weakDiv,
                          ParDiscreteGradOperator * grad)
   : H1FESpace_(&H1FESpace),
     HCurlFESpace_(&HCurlFESpace),
     s0_(s0),
     weakDiv_(weakDiv),
     grad_(grad),
     psi_(NULL),
     xDiv_(NULL),
     S0_(NULL),
     amg_(NULL),
     pcg_(NULL),
     ownsS0_(s0 == NULL),
     ownsWeakDiv_(weakDiv == NULL),
     ownsGrad_(grad == NULL)
{
   ess_bdr_.SetSize(H1FESpace_->GetParMesh()->bdr_attributes.Max());
   ess_bdr_ = 1;
   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);

   Geometry::Type geom = H1FESpace_->GetMesh()->GetTypicalElementGeometry();
   const IntegrationRule * ir = &IntRules.Get(geom, irOrder);

   if ( s0 == NULL )
   {
      s0_ = new ParBilinearForm(H1FESpace_);
      BilinearFormIntegrator * diffInteg = new DiffusionIntegrator;
      diffInteg->SetIntRule(ir);
      s0_->AddDomainIntegrator(diffInteg);
      s0_->Assemble();
      s0_->Finalize();
      S0_ = new HypreParMatrix;
   }
   if ( weakDiv_ == NULL )
   {
      weakDiv_ = new ParMixedBilinearForm(HCurlFESpace_, H1FESpace_);
      BilinearFormIntegrator * wdivInteg = new VectorFEWeakDivergenceIntegrator;
      wdivInteg->SetIntRule(ir);
      weakDiv_->AddDomainIntegrator(wdivInteg);
      weakDiv_->Assemble();
      weakDiv_->Finalize();
   }
   if ( grad_ == NULL )
   {
      grad_ = new ParDiscreteGradOperator(H1FESpace_, HCurlFESpace_);
      grad_->Assemble();
      grad_->Finalize();
   }

   psi_  = new ParGridFunction(H1FESpace_);
   xDiv_ = new ParGridFunction(H1FESpace_);
}

IrrotationalNDProjector::~IrrotationalNDProjector()
{
   delete psi_;
   delete xDiv_;

   delete amg_;
   delete pcg_;

   delete S0_;

   delete s0_;
   delete weakDiv_;
}

void
IrrotationalNDProjector::InitSolver() const
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
IrrotationalNDProjector::Mult(const Vector &x, Vector &y) const
{
   // Compute the divergence of x
   weakDiv_->Mult(x,*xDiv_); *xDiv_ *= -1.0;

   // Apply essential BC and form linear system
   *psi_ = 0.0;
   s0_->FormLinearSystem(ess_bdr_tdofs_, *psi_, *xDiv_, *S0_, Psi_, RHS_);

   // Solve the linear system for Psi
   if ( pcg_ == NULL ) { this->InitSolver(); }
   pcg_->Mult(RHS_, Psi_);

   // Compute the parallel grid function corresponding to Psi
   s0_->RecoverFEMSolution(Psi_, *xDiv_, *psi_);

   // Compute the irrotational portion of x
   grad_->Mult(*psi_, y);
}

void
IrrotationalNDProjector::Update()
{
   delete pcg_; pcg_ = NULL;
   delete amg_; amg_ = NULL;
   delete S0_;  S0_  = new HypreParMatrix;

   psi_->Update();
   xDiv_->Update();

   if ( ownsS0_ )
   {
      s0_->Update();
      s0_->Assemble();
      s0_->Finalize();
   }
   if ( ownsWeakDiv_ )
   {
      weakDiv_->Update();
      weakDiv_->Assemble();
      weakDiv_->Finalize();
   }
   if ( ownsGrad_ )
   {
      grad_->Update();
      grad_->Assemble();
      grad_->Finalize();
   }

   H1FESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
}

DivergenceFreeNDProjector
::DivergenceFreeNDProjector(ParFiniteElementSpace   & H1FESpace,
                            ParFiniteElementSpace   & HCurlFESpace,
                            const int               & irOrder,
                            ParBilinearForm         * s0,
                            ParMixedBilinearForm    * weakDiv,
                            ParDiscreteGradOperator * grad)
   : IrrotationalNDProjector(H1FESpace,HCurlFESpace, irOrder, s0, weakDiv, grad)
{}

DivergenceFreeNDProjector::~DivergenceFreeNDProjector()
{}

void
DivergenceFreeNDProjector::Mult(const Vector &x, Vector &y) const
{
   this->IrrotationalNDProjector::Mult(x, y);
   y  -= x;
   y *= -1.0;
}

void
DivergenceFreeNDProjector::Update()
{
   this->IrrotationalNDProjector::Update();
}

DivergenceFreeRTProjector
::DivergenceFreeRTProjector(ParFiniteElementSpace   & HCurlFESpace,
                            ParFiniteElementSpace   & HDivFESpace,
                            const int               & irOrder,
                            ParBilinearForm         * s1,
                            ParMixedBilinearForm    * weakCurl,
                            ParDiscreteCurlOperator * curl)
   : HCurlFESpace_(&HCurlFESpace),
     HDivFESpace_(&HDivFESpace),
     s1_(s1),
     weakCurl_(weakCurl),
     curl_(curl),
     psi_(NULL),
     xCurl_(NULL),
     S1_(NULL),
     pc_(NULL),
     pcg_(NULL),
     dim_(HCurlFESpace_->GetFE(0)->GetDim()),
     ownsS1_(s1 == NULL),
     ownsWeakCurl_(weakCurl == NULL),
     ownsCurl_(curl == NULL)
{
   ess_bdr_.SetSize(HCurlFESpace_->GetParMesh()->bdr_attributes.Max());
   ess_bdr_ = 1;
   HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);

   int geom = HCurlFESpace_->GetFE(0)->GetGeomType();
   const IntegrationRule * ir = &IntRules.Get(geom, irOrder);

   if ( s1 == NULL )
   {
      s1_ = new ParBilinearForm(HCurlFESpace_);
      BilinearFormIntegrator * ccInteg = (dim_==2) ?
                                         dynamic_cast<BilinearFormIntegrator*>(new DiffusionIntegrator) :
                                         dynamic_cast<BilinearFormIntegrator*>(new CurlCurlIntegrator);
      ccInteg->SetIntRule(ir);
      s1_->AddDomainIntegrator(ccInteg);
      s1_->Assemble();
      s1_->Finalize();
      S1_ = new HypreParMatrix;
   }
   if ( weakCurl_ == NULL )
   {
      weakCurl_ = new ParMixedBilinearForm(HDivFESpace_, HCurlFESpace_);
      BilinearFormIntegrator * wcurlInteg = new MixedVectorWeakCurlIntegrator;
      wcurlInteg->SetIntRule(ir);
      weakCurl_->AddDomainIntegrator(wcurlInteg);
      weakCurl_->Assemble();
      weakCurl_->Finalize();
   }
   if ( curl_ == NULL )
   {
      curl_ = new ParDiscreteCurlOperator(HCurlFESpace_, HDivFESpace_);
      curl_->Assemble();
      curl_->Finalize();
   }

   psi_   = new ParGridFunction(HCurlFESpace_);
   xCurl_ = new ParGridFunction(HCurlFESpace_);
}

DivergenceFreeRTProjector::~DivergenceFreeRTProjector()
{
   delete psi_;
   delete xCurl_;

   delete pc_;
   delete pcg_;

   delete S1_;

   delete s1_;
   delete weakCurl_;
}

void
DivergenceFreeRTProjector::InitSolver() const
{
   delete pcg_;
   delete pc_;

   if (dim_ == 2)
   {
      HypreBoomerAMG * amg = new HypreBoomerAMG(*S1_);
      amg->SetPrintLevel(0);
      pc_ = amg;
   }
   else
   {
      HypreAMS * ams = new HypreAMS(*S1_, HCurlFESpace_);
      ams->SetPrintLevel(0);
      pc_ = ams;
   }
   pcg_ = new HyprePCG(*S1_);
   pcg_->SetTol(1e-14);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*pc_);
}

void
DivergenceFreeRTProjector::Mult(const Vector &x, Vector &y) const
{
   // Compute the curl of x
   weakCurl_->Mult(x,*xCurl_);

   // Apply essential BC and form linear system
   *psi_ = 0.0;
   s1_->FormLinearSystem(ess_bdr_tdofs_, *psi_, *xCurl_, *S1_, Psi_, RHS_);

   // Solve the linear system for Psi
   if ( pcg_ == NULL ) { this->InitSolver(); }
   pcg_->Mult(RHS_, Psi_);

   // Compute the parallel grid function correspoinding to Psi
   s1_->RecoverFEMSolution(Psi_, *xCurl_, *psi_);

   // Compute the divergence free portion of x
   curl_->Mult(*psi_, y);
}

void
DivergenceFreeRTProjector::Update()
{
   delete pcg_; pcg_ = NULL;
   delete pc_;  pc_  = NULL;
   delete S1_;  S1_  = new HypreParMatrix;

   psi_->Update();
   xCurl_->Update();

   if ( ownsS1_ )
   {
      s1_->Update();
      s1_->Assemble();
      s1_->Finalize();
   }
   if ( ownsWeakCurl_ )
   {
      weakCurl_->Update();
      weakCurl_->Assemble();
      weakCurl_->Finalize();
   }
   if ( ownsCurl_ )
   {
      curl_->Update();
      curl_->Assemble();
      curl_->Finalize();
   }

   HCurlFESpace_->GetEssentialTrueDofs(ess_bdr_, ess_bdr_tdofs_);
}

IrrotationalRTProjector
::IrrotationalRTProjector(ParFiniteElementSpace   & HCurlFESpace,
                          ParFiniteElementSpace   & HDivFESpace,
                          const int               & irOrder,
                          ParBilinearForm         * s1,
                          ParMixedBilinearForm    * weakCurl,
                          ParDiscreteCurlOperator * curl)
   : DivergenceFreeRTProjector(HCurlFESpace, HDivFESpace, irOrder,
                               s1, weakCurl, curl)
{}

IrrotationalRTProjector::~IrrotationalRTProjector()
{}

void
IrrotationalRTProjector::Mult(const Vector &x, Vector &y) const
{
   this->DivergenceFreeRTProjector::Mult(x, y);
   y  -= x;
   y *= -1.0;
}

void
IrrotationalRTProjector::Update()
{
   this->DivergenceFreeRTProjector::Update();
}

void VisualizeMesh(socketstream &sock, const char *vishost, int visport,
                   ParMesh &pmesh, const char *title,
                   int x, int y, int w, int h, const char *keys)
{
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "mesh\n";
      }

      pmesh.PrintAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";
         if ( keys ) { sock << "keys " << keys << "\n"; }
         sock << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    const ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h, const char *keys, bool vec)
{
   ParMesh &pmesh = *gf.ParFESpace()->GetParMesh();
   MPI_Comm comm = pmesh.GetComm();

   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   bool newly_opened = false;
   int connection_failed;

   do
   {
      if (myid == 0)
      {
         if (!sock.is_open() || !sock)
         {
            sock.open(vishost, visport);
            sock.precision(8);
            newly_opened = true;
         }
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);
      gf.SaveAsOne(sock);

      if (myid == 0 && newly_opened)
      {
         sock << "window_title '" << title << "'\n"
              << "window_geometry "
              << x << " " << y << " " << w << " " << h << "\n";
         if ( keys ) { sock << "keys " << keys << "\n"; }
         else { sock << "keys maaAc"; }
         if ( vec ) { sock << "vvv"; }
         sock << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

} // namespace common

} // namespace mfem

#endif
