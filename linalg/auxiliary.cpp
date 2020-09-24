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

#include "linalg.hpp"
#include "../fem/pfespace.hpp"
#include "../fem/pbilinearform.hpp"

namespace mfem
{

GeneralAMS::GeneralAMS(const mfem::Operator& A,
                       const mfem::Operator& pi,
                       const mfem::Operator& g,
                       const mfem::Operator& pispacesolver,
                       const mfem::Operator& gspacesolver,
                       const mfem::Operator& smoother,
                       const mfem::Array<int>& ess_tdof_list)
   :
   mfem::Solver(A.Height()),
   A_(A),
   pi_(pi),
   g_(g),
   pispacesolver_(pispacesolver),
   gspacesolver_(gspacesolver),
   smoother_(smoother),
   ess_tdof_list_(ess_tdof_list),
   residual_time_(0.0),
   smooth_time_(0.0),
   gspacesolver_time_(0.0),
   pispacesolver_time_(0.0)
{
   // could assert a bunch of sizes...
}

GeneralAMS::~GeneralAMS()
{
}

void GeneralAMS::FormResidual(const mfem::Vector& rhs, const mfem::Vector& x,
                              mfem::Vector& residual) const
{
   chrono_.Clear();
   chrono_.Start();

   A_.Mult(x, residual);
   residual *= -1.0;
   residual += rhs;

   chrono_.Stop();
   residual_time_ += chrono_.RealTime();
}

void GeneralAMS::Mult(const Vector& x, Vector& y) const
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   // see hypre_ParCSRSubspacePrec() in hypre/src/parcsr_ls/ams.c
   // and also hypre_AMSSolve() in the same file

   // default cyle (cycle 1) is "01210"
   // ie, smooth, correct in space 1, correct in space 2, correct in space 1, smooth
   // Bi[0] = ams_data -> B_G;    HBi[0] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;
   // Bi[1] = ams_data -> B_Pi;   HBi[1] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGBlockSolve;
   // (suggests space 1 is G, space 2 is Pi)

   // but I think mfem::HypreAMS picks cycle 13, or 034515430
   // with 0 smooth, three separate coordinate Pi solves, grad solve, then back up

   // maybe we should do Pi-G-Pi instead of G-Pi-G (doesn't seem to make much difference)

   // cycle would be more like 0102010 if below is true
   const bool extra_smoothing = false;

   mfem::StopWatch chrono;

   MFEM_ASSERT(x.Size() == y.Size(), "Sizes don't match!");
   MFEM_ASSERT(A_.Height() == x.Size(), "Sizes don't match!");

   Vector residual(x.Size());
   residual = 0.0;
   y = 0.0;

   // smooth (exactly what smoother is HypreAMS using?)
   chrono.Clear();
   chrono.Start();
   smoother_.Mult(x, y);
   chrono.Stop();
   smooth_time_ += chrono.RealTime();

   // g-space correction
   FormResidual(x, y, residual);
   Vector gspacetemp(g_.Width());
   g_.MultTranspose(residual, gspacetemp);
   Vector gspacecorrection(g_.Width());
   gspacecorrection = 0.0;
   chrono.Clear();
   chrono.Start();
   gspacesolver_.Mult(gspacetemp, gspacecorrection);
   chrono.Stop();
   gspacesolver_time_ += chrono.RealTime();
   g_.Mult(gspacecorrection, residual);
   y += residual;

   Vector temp(x.Size());
   if (extra_smoothing)
   {
      FormResidual(x, y, residual);
      smoother_.Mult(residual, temp);
      y += temp;
   }

   // pi-space correction
   FormResidual(x, y, residual);
   Vector pispacetemp(pi_.Width());

   pi_.MultTranspose(residual, pispacetemp);

   Vector pispacecorrection(pi_.Width());
   pispacecorrection = 0.0;
   chrono.Clear();
   chrono.Start();
   pispacesolver_.Mult(pispacetemp, pispacecorrection);
   chrono.Stop();
   pispacesolver_time_ += chrono.RealTime();
   pi_.Mult(pispacecorrection, residual);
   y += residual;

   if (extra_smoothing)
   {
      FormResidual(x, y, residual);
      smoother_.Mult(residual, temp);
      y += temp;
   }

   // g-space correction
   FormResidual(x, y, residual);
   g_.MultTranspose(residual, gspacetemp);
   gspacecorrection = 0.0;
   chrono.Clear();
   chrono.Start();
   gspacesolver_.Mult(gspacetemp, gspacecorrection);
   chrono.Stop();
   gspacesolver_time_ += chrono.RealTime();
   g_.Mult(gspacecorrection, residual);
   y += residual;

   // smooth (don't need the residual if smoother_ has iterative_mode ?)
   FormResidual(x, y, residual);
   chrono.Clear();
   chrono.Start();
   smoother_.Mult(residual, temp);
   y += temp;
   chrono.Stop();
   smooth_time_ += chrono.RealTime();
}

// Pi-space constructor
MatrixFreeAuxiliarySpace::MatrixFreeAuxiliarySpace(
   mfem::ParMesh& mesh_lor,
   mfem::Coefficient* alpha_coeff,
   mfem::Coefficient* beta_coeff, Array<int>& ess_bdr,
   mfem::Operator& curlcurl_oper,
   mfem::Operator& pi,
   int cg_iterations)
   :
   Solver(pi.Width()),
   matfree_(NULL),
   cg_(NULL),
   inner_aux_iterations_(0)
{
   H1_FECollection * fec_lor = new H1_FECollection(1, mesh_lor.Dimension());
   ParFiniteElementSpace fespace_lor_d(&mesh_lor, fec_lor, mesh_lor.Dimension(),
                                       Ordering::byVDIM);

   // build LOR AMG v-cycle
   if (ess_bdr.Size())
   {
      fespace_lor_d.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);
   }
   ParBilinearForm a_space(&fespace_lor_d);

   // this choice of policy is super-important for the G-space solver, but
   // also can make some difference here
   const Matrix::DiagonalPolicy policy = Matrix::DIAG_KEEP;
   a_space.SetDiagonalPolicy(policy); // doesn't do anything, see Eliminate() below
   a_space.AddDomainIntegrator(new VectorDiffusionIntegrator(*alpha_coeff));
   a_space.AddDomainIntegrator(new VectorMassIntegrator(*beta_coeff));
   a_space.UsePrecomputedSparsity();
   a_space.Assemble();
   a_space.EliminateEssentialBC(ess_bdr, policy);
   a_space.Finalize();
   aspacematrix_ = a_space.ParallelAssemble();
   aspacematrix_->CopyRowStarts();
   aspacematrix_->CopyColStarts();

   SetupBoomerAMG(fespace_lor_d.GetMesh()->Dimension());

   if (cg_iterations > 0)
   {
      const bool super_duper_extra_verbose = false;
      SetupCG(curlcurl_oper, pi, cg_iterations, super_duper_extra_verbose);
   }
   else
   {
      SetupVCycle();
   }
   delete fec_lor;
}

// G-space constructor
MatrixFreeAuxiliarySpace::MatrixFreeAuxiliarySpace(
   mfem::ParMesh& mesh_lor,
   mfem::Coefficient* beta_coeff, Array<int>& ess_bdr,
   mfem::Operator& curlcurl_oper, mfem::Operator& g,
   int cg_iterations)
   :
   Solver(curlcurl_oper.Height()),
   matfree_(NULL),
   cg_(NULL),
   inner_aux_iterations_(0)
{
   H1_FECollection * fec_lor = new H1_FECollection(1, mesh_lor.Dimension());
   ParFiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   // build LOR AMG v-cycle
   ParBilinearForm a_space(&fespace_lor);

   // we need something like DIAG_ZERO in the solver, but explicitly doing
   // that makes BoomerAMG setup complain, so instead we constrain the boundary
   // in the CG solver
   const Matrix::DiagonalPolicy policy = Matrix::DIAG_ONE;

   a_space.SetDiagonalPolicy(policy);
   a_space.AddDomainIntegrator(new DiffusionIntegrator(*beta_coeff));

   a_space.UsePrecomputedSparsity();
   a_space.Assemble();
   if (ess_bdr.Size())
   {
      fespace_lor.GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);
   }

   // you have to use (serial) BilinearForm eliminate routines to get
   // diag policy DIAG_ZERO all the ParallelEliminateTDofs etc. routines
   // implicitly have a Matrix::DIAG_KEEP policy
   a_space.EliminateEssentialBC(ess_bdr, policy);
   a_space.Finalize();
   aspacematrix_ = a_space.ParallelAssemble();

   aspacematrix_->CopyRowStarts();
   aspacematrix_->CopyColStarts();

   SetupBoomerAMG(0);

   if (cg_iterations > 0)
   {
      // inner CG seems necessary in G-space solver
      SetupCG(curlcurl_oper, g, cg_iterations);
   }
   else
   {
      // this would probably be more efficient, but there are boundary condition issues?
      SetupVCycle();
   }

   delete fec_lor;
}

void MatrixFreeAuxiliarySpace::SetupCG(
   mfem::Operator& curlcurl_oper, mfem::Operator& conn,
   int inner_cg_iterations, bool very_verbose)
{
   MFEM_ASSERT(conn.Height() == curlcurl_oper.Width(),
               "Operators don't match!");
   matfree_ = new RAPOperator(conn, curlcurl_oper, conn);
   MFEM_ASSERT(matfree_->Height() == aspacepc_->Height(),
               "Operators don't match!");

   cg_ = new CGSolver(MPI_COMM_WORLD);
   cg_->SetOperator(*matfree_);
   cg_->SetPreconditioner(*aspacepc_);
   if (inner_cg_iterations > 99)
   {
      cg_->SetRelTol(1.e-14);
      cg_->SetMaxIter(100);
   }
   else
   {
      cg_->SetRelTol(0.0);
      cg_->SetMaxIter(inner_cg_iterations);
   }
   if (very_verbose)
      cg_->SetPrintLevel(1);
   else
      cg_->SetPrintLevel(-1);

   aspacewrapper_ = cg_;
}

void MatrixFreeAuxiliarySpace::SetupVCycle()
{
   aspacewrapper_ = aspacepc_;
}

class ZeroWrap : public Solver
{
public:
   ZeroWrap(HypreParMatrix& mat, Array<int>& ess_tdof_list) :
      Solver(mat.Height()), amg_(mat), ess_tdof_list_(ess_tdof_list)
   {
      amg_.SetPrintLevel(0);
   }

   void Mult(const Vector& x, Vector& y) const
   {
      amg_.Mult(x, y);
      for (int k : ess_tdof_list_)
      {
         y(k) = 0.0;
      }
   }

   void SetOperator(const Operator&) { }

private:
   HypreBoomerAMG amg_;
   Array<int>& ess_tdof_list_;
};

void MatrixFreeAuxiliarySpace::SetupBoomerAMG(int system_dimension)
{
   if (system_dimension == 0)
   {
      // boundary condition tweak for G-space solver
      aspacepc_ = new ZeroWrap(*aspacematrix_, ess_tdof_list_);
   }
   else // if (system_dimension > 0)
   {
      // Pi-space solver is a vector space
      HypreBoomerAMG* hpc = new HypreBoomerAMG(*aspacematrix_);
      hpc->SetSystemsOptions(system_dimension);
      hpc->SetPrintLevel(0);
      aspacepc_ = hpc;
   }
}

void MatrixFreeAuxiliarySpace::Mult(const mfem::Vector& x, mfem::Vector& y) const
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   y = 0.0;
   aspacewrapper_->Mult(x, y);
   if (cg_ && rank == 0)
   {
      int q = cg_->GetNumIterations();
      inner_aux_iterations_ += q;
   }
}

MatrixFreeAuxiliarySpace::~MatrixFreeAuxiliarySpace()
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   delete aspacematrix_;
   delete aspacepc_;
   delete matfree_;
   if (aspacepc_ != aspacewrapper_) delete aspacewrapper_;
   if (cg_ != aspacewrapper_) delete cg_;
}

MatrixFreeAMS::MatrixFreeAMS(
   ParBilinearForm& aform, Operator& oper, ParFiniteElementSpace& nd_fespace,
   Coefficient* alpha_coeff,
   Coefficient* beta_coeff, Array<int>& ess_bdr, int inner_pi_iterations,
   int inner_g_iterations)
   :
   Solver(oper.Height())
{
   int order = nd_fespace.GetFE(0)->GetOrder();
   ParMesh *mesh = nd_fespace.GetParMesh();
   int dim = mesh->Dimension();

   // smoother
   const double scale = 0.25; // not so clear what exactly to put here...
   Array<int> ess_tdof_list;
   nd_fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   smoother_ = new OperatorJacobiSmoother(aform, ess_tdof_list, scale);

   // get H1 space
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   h1_fespace_ = new ParFiniteElementSpace(mesh, h1_fec);
   h1_fespace_d_ = new ParFiniteElementSpace(mesh, h1_fec, dim, Ordering::byVDIM);

   // build G operator
   pa_grad_ = new ParDiscreteLinearOperator(h1_fespace_, &nd_fespace);
   pa_grad_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_grad_->AddDomainInterpolator(new GradientInterpolator);
   pa_grad_->Assemble();
   pa_grad_->FormRectangularSystemMatrix(G_);

   // build Pi operator
   pa_interp_ = new ParDiscreteLinearOperator(h1_fespace_d_, &nd_fespace);
   pa_interp_->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_interp_->AddDomainInterpolator(new IdentityInterpolator);
   pa_interp_->Assemble();
   pa_interp_->FormRectangularSystemMatrix(Pi_);

   // build LOR space
   ParMesh mesh_lor(mesh, order, BasisType::GaussLobatto);

   /* A lot depends on the quality of the auxiliary space solves.
      For high-contrast coefficients, and other difficult problems,
      inner iteration counts may need to be increased.
      
      Boundary conditions can matter as well (see DIAG_ZERO policy) */

   // build G space solver
   Gspacesolver_ = new MatrixFreeAuxiliarySpace(
      mesh_lor, beta_coeff, ess_bdr, oper, *G_, inner_g_iterations);

   // build Pi space solver
   Pispacesolver_ = new MatrixFreeAuxiliarySpace(
      mesh_lor, alpha_coeff, beta_coeff, ess_bdr, oper, *Pi_, inner_pi_iterations);

   general_ams_ = new GeneralAMS(oper, *Pi_, *G_, *Pispacesolver_,
                                 *Gspacesolver_, *smoother_, ess_tdof_list);

   delete h1_fec;
}

MatrixFreeAMS::~MatrixFreeAMS()
{
   delete smoother_;
   // delete serialPi_;
   // delete Pi_;
   // delete serialG_;
   // delete G_;
   delete pa_grad_;
   delete pa_interp_;
   delete Gspacesolver_;
   delete Pispacesolver_;
   delete general_ams_;
   delete h1_fespace_;
   delete h1_fespace_d_;
}

} // namespace mfem
