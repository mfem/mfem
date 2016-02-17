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

#include "pfem_extras.hpp"

using namespace std;

namespace mfem
{

namespace miniapps
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
{
   delete pdlo_;
   delete mat_;
}

HYPRE_Int
ParDiscreteInterpolationOperator::Mult(HypreParVector &x, HypreParVector &y,
                                       double alpha, double beta)
{
   if ( !mat_ ) { this->createMatrix(); }
   return mat_->Mult( x, y, alpha, beta);
}

HYPRE_Int
ParDiscreteInterpolationOperator::Mult(HYPRE_ParVector x, HYPRE_ParVector y,
                                       double alpha, double beta)
{
   if ( !mat_ ) { this->createMatrix(); }
   return mat_->Mult( x, y, alpha, beta);
}

HYPRE_Int
ParDiscreteInterpolationOperator::MultTranspose(HypreParVector &x,
                                                HypreParVector &y,
                                                double alpha, double beta)
{
   if ( !mat_ ) { this->createMatrix(); }
   return mat_->MultTranspose( x, y, alpha, beta);
}

void
ParDiscreteInterpolationOperator::Mult(double a, const Vector &x,
                                       double b, Vector &y) const
{
   if ( !mat_ ) { this->createMatrix(); }
   mat_->Mult( a, x, b, y);
}

void
ParDiscreteInterpolationOperator::MultTranspose(double a, const Vector &x,
                                                double b, Vector &y) const
{
   if ( !mat_ ) { this->createMatrix(); }
   mat_->MultTranspose( a, x, b, y);
}

void
ParDiscreteInterpolationOperator::Mult(const Vector &x, Vector &y) const
{
   if ( !mat_ ) { this->createMatrix(); }
   mat_->Mult( x, y);
}

void
ParDiscreteInterpolationOperator::MultTranspose(const Vector &x,
                                                Vector &y) const
{
   if ( !mat_ ) { this->createMatrix(); }
   mat_->MultTranspose( x, y);
}

void
ParDiscreteInterpolationOperator::createMatrix() const
{
   pdlo_->Assemble();
   pdlo_->Finalize();
   delete mat_;
   mat_ = pdlo_->ParallelAssemble();
}

void
ParDiscreteInterpolationOperator::Update()
{
   pdlo_->Update();
   this->createMatrix();
}

HypreParMatrix *
ParDiscreteInterpolationOperator::ParallelAssemble()
{
   if ( !mat_ ) { this->createMatrix(); }
   HypreParMatrix * mat = mat_;
   mat_ = NULL;
   return mat;
}

ParDiscreteGradOperator::ParDiscreteGradOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new GradientInterpolator);
   this->createMatrix();
}

ParDiscreteCurlOperator::ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new CurlInterpolator);
   this->createMatrix();
}

ParDiscreteDivOperator::ParDiscreteDivOperator(ParFiniteElementSpace *dfes,
                                               ParFiniteElementSpace *rfes)
{
   pdlo_ = new ParDiscreteLinearOperator(dfes, rfes);
   pdlo_->AddDomainInterpolator(new DivergenceInterpolator);
   this->createMatrix();
}

IrrotationalProjector
::IrrotationalProjector(ParFiniteElementSpace & H1FESpace,
                        ParFiniteElementSpace & HCurlFESpace,
                        ParDiscreteInterpolationOperator & Grad)
   : H1FESpace_(&H1FESpace),
     HCurlFESpace_(&HCurlFESpace),
     Grad_(&Grad)
{
   ess_bdr_.SetSize(H1FESpace.GetParMesh()->bdr_attributes.Max());
   ess_bdr_ = 1;

   s0_ = new ParBilinearForm(&H1FESpace);
   s0_->AddDomainIntegrator(new DiffusionIntegrator());
   s0_->Assemble();
   s0_->Finalize();
   S0_ = s0_->ParallelAssemble();

   m1_ = new ParBilinearForm(&HCurlFESpace);
   m1_->AddDomainIntegrator(new VectorFEMassIntegrator());
   m1_->Assemble();
   m1_->Finalize();
   M1_ = m1_->ParallelAssemble();

   amg_ = new HypreBoomerAMG(*S0_);
   amg_->SetPrintLevel(0);
   pcg_ = new HyprePCG(*S0_);
   pcg_->SetTol(1e-14);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*amg_);

   xDiv_     = new HypreParVector(&H1FESpace);
   yPot_     = new HypreParVector(&H1FESpace);
   gradYPot_ = new HypreParVector(HCurlFESpace_);
}

IrrotationalProjector::~IrrotationalProjector()
{
   delete s0_;
   delete m1_;
   delete amg_;
   delete pcg_;
   delete S0_;
   delete M1_;
   delete xDiv_;
   delete yPot_;
   delete gradYPot_;
}

void
IrrotationalProjector::Mult(const Vector &x, Vector &y) const
{
   *yPot_ = 0.0;
   Grad_->MultTranspose(x,*xDiv_);
   s0_->ParallelEliminateEssentialBC(ess_bdr_,*S0_,*yPot_,*xDiv_);
   pcg_->Mult(*xDiv_,*yPot_);
   Grad_->Mult(*yPot_,*gradYPot_);
   M1_->Mult(*gradYPot_,y);
}

void
IrrotationalProjector::Update()
{
   delete S0_;
   delete M1_;
   delete gradYPot_;
   delete yPot_;
   delete xDiv_;
   delete pcg_;
   delete amg_;

   s0_->Update();
   m1_->Update();

   s0_->Assemble();
   s0_->Finalize();

   m1_->Assemble();
   m1_->Finalize();

   S0_ = s0_->ParallelAssemble();
   M1_ = m1_->ParallelAssemble();

   amg_ = new HypreBoomerAMG(*S0_);
   amg_->SetPrintLevel(0);
   pcg_ = new HyprePCG(*S0_);
   pcg_->SetTol(1e-14);
   pcg_->SetMaxIter(200);
   pcg_->SetPrintLevel(0);
   pcg_->SetPreconditioner(*amg_);

   xDiv_     = new HypreParVector(H1FESpace_);
   yPot_     = new HypreParVector(H1FESpace_);
   gradYPot_ = new HypreParVector(HCurlFESpace_);
}

DivergenceFreeProjector
::DivergenceFreeProjector(ParFiniteElementSpace & H1FESpace,
                          ParFiniteElementSpace & HCurlFESpace,
                          ParDiscreteInterpolationOperator & Grad)
   : IrrotationalProjector(H1FESpace,HCurlFESpace, Grad),
     HCurlFESpace_(&HCurlFESpace),
     xIrr_(NULL)
{
   xIrr_ = new HypreParVector(&HCurlFESpace);
}

DivergenceFreeProjector::~DivergenceFreeProjector()
{
   delete xIrr_;
}

void
DivergenceFreeProjector::Mult(const Vector &x, Vector &y) const
{
   this->IrrotationalProjector::Mult(x,*xIrr_);
   y  = x;
   y -= *xIrr_;
}

void
DivergenceFreeProjector::Update()
{
   delete xIrr_;

   this->IrrotationalProjector::Update();

   xIrr_ = new HypreParVector(HCurlFESpace_);
}


void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x, int y, int w, int h)
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
              << x << " " << y << " " << w << " " << h << "\n"
              << "keys maaAc" << endl;
      }

      if (myid == 0)
      {
         connection_failed = !sock && !newly_opened;
      }
      MPI_Bcast(&connection_failed, 1, MPI_INT, 0, comm);
   }
   while (connection_failed);
}

} // namespace miniapps

} // namespace mfem

#endif
