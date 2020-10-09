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

IrrotationalProjector
::IrrotationalProjector(ParFiniteElementSpace   & H1FESpace,
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

   int geom = H1FESpace_->GetFE(0)->GetGeomType();
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

IrrotationalProjector::~IrrotationalProjector()
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
IrrotationalProjector::InitSolver() const
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
IrrotationalProjector::Mult(const Vector &x, Vector &y) const
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
IrrotationalProjector::Update()
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

DivergenceFreeProjector
::DivergenceFreeProjector(ParFiniteElementSpace   & H1FESpace,
                          ParFiniteElementSpace   & HCurlFESpace,
                          const int               & irOrder,
                          ParBilinearForm         * s0,
                          ParMixedBilinearForm    * weakDiv,
                          ParDiscreteGradOperator * grad)
   : IrrotationalProjector(H1FESpace,HCurlFESpace, irOrder, s0, weakDiv, grad)
{}

DivergenceFreeProjector::~DivergenceFreeProjector()
{}

void
DivergenceFreeProjector::Mult(const Vector &x, Vector &y) const
{
   this->IrrotationalProjector::Mult(x, y);
   y  -= x;
   y *= -1.0;
}

void
DivergenceFreeProjector::Update()
{
   this->IrrotationalProjector::Update();
}

void VisualizeMesh(socketstream &sock, const char *vishost, int visport,
                   ParMesh &pmesh, const char *title,
                   int x, int y, int w, int h, const char *keys, bool vec)
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
         sock << "solution\n";
      }

      pmesh.PrintAsOne(sock);

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

void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
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
