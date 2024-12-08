// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "linalg.hpp"
#include "../fem/pfespace.hpp"
#include "../fem/pbilinearform.hpp"

namespace mfem
{

GeneralAMS::GeneralAMS(const Operator& curlcurl_op_,
                       const Operator& pi_,
                       const Operator& gradient_,
                       const Operator& pispacesolver_,
                       const Operator& gspacesolver_,
                       const Operator& smoother_,
                       const Array<int>& ess_tdof_list_)
   :
   Solver(curlcurl_op_.Height()),
   curlcurl_op(curlcurl_op_),
   pi(pi_),
   gradient(gradient_),
   pispacesolver(pispacesolver_),
   gspacesolver(gspacesolver_),
   smoother(smoother_),
   ess_tdof_list(ess_tdof_list_)
{
}

GeneralAMS::~GeneralAMS()
{
}

void GeneralAMS::FormResidual(const Vector& rhs, const Vector& x,
                              Vector& residual) const
{
   curlcurl_op.Mult(x, residual);
   residual *= -1.0;
   residual += rhs;
}

/*
  This implementation follows that in hypre, see hypre_ParCSRSubspacePrec()
  in hypre/src/parcsr_ls/ams.c and also hypre_AMSSolve() in the same file.

  hypre's default cycle (cycle 1) is "01210", ie, smooth, correct in space
  1, correct in space 2, correct in space 1, smooth. Their space 1 is G and
  space 2 is Pi by default.

  The MFEM interface in mfem::HypreAMS though picks cycle 13, or 034515430,
  which separates the Pi-space solve into three separate (scalar) AMG solves
  instead of a single vector solve.

  We choose below the hypre default, but we have experimented with some other
  cycles; the short version is that they often work but the differences are
  generally not large.
*/
void GeneralAMS::Mult(const Vector& x, Vector& y) const
{
   MFEM_ASSERT(x.Size() == y.Size(), "Sizes don't match!");
   MFEM_ASSERT(curlcurl_op.Height() == x.Size(), "Sizes don't match!");

   Vector residual(x.Size());
   residual = 0.0;
   y = 0.0;

   // smooth
   smoother.Mult(x, y);

   // g-space correction
   FormResidual(x, y, residual);
   Vector gspacetemp(gradient.Width());
   gradient.MultTranspose(residual, gspacetemp);
   Vector gspacecorrection(gradient.Width());
   gspacecorrection = 0.0;
   gspacesolver.Mult(gspacetemp, gspacecorrection);
   gradient.Mult(gspacecorrection, residual);
   y += residual;

   // pi-space correction
   FormResidual(x, y, residual);
   Vector pispacetemp(pi.Width());

   pi.MultTranspose(residual, pispacetemp);

   Vector pispacecorrection(pi.Width());
   pispacecorrection = 0.0;
   pispacesolver.Mult(pispacetemp, pispacecorrection);
   pi.Mult(pispacecorrection, residual);
   y += residual;

   // g-space correction
   FormResidual(x, y, residual);
   gradient.MultTranspose(residual, gspacetemp);
   gspacecorrection = 0.0;
   gspacesolver.Mult(gspacetemp, gspacecorrection);
   gradient.Mult(gspacecorrection, residual);
   y += residual;

   // smooth
   FormResidual(x, y, residual);
   Vector temp(x.Size());
   smoother.Mult(residual, temp);
   y += temp;
}

// Pi-space constructor
MatrixFreeAuxiliarySpace::MatrixFreeAuxiliarySpace(
   ParMesh& mesh_lor, Coefficient* alpha_coeff,
   Coefficient* beta_coeff, MatrixCoefficient* beta_mcoeff, Array<int>& ess_bdr,
   Operator& curlcurl_oper, Operator& pi,
#ifdef MFEM_USE_AMGX
   bool useAmgX_,
#endif
   int cg_iterations) :
   Solver(pi.Width()),
   comm(mesh_lor.GetComm()),
   matfree(NULL),
   cg(NULL),
#ifdef MFEM_USE_AMGX
   useAmgX(useAmgX_),
#endif
   inner_aux_iterations(0)
{
   H1_FECollection * fec_lor = new H1_FECollection(1, mesh_lor.Dimension());
   ParFiniteElementSpace fespace_lor_d(&mesh_lor, fec_lor, mesh_lor.Dimension(),
                                       Ordering::byVDIM);

   // build LOR AMG v-cycle
   if (ess_bdr.Size())
   {
      fespace_lor_d.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   ParBilinearForm a_lor(&fespace_lor_d);

   // this choice of policy is important for the G-space solver, but
   // also can make some difference here
   const Matrix::DiagonalPolicy policy = Matrix::DIAG_KEEP;
   a_lor.SetDiagonalPolicy(policy);
   if (alpha_coeff == NULL)
   {
      a_lor.AddDomainIntegrator(new VectorDiffusionIntegrator);
   }
   else
   {
      a_lor.AddDomainIntegrator(new VectorDiffusionIntegrator(*alpha_coeff));
   }

   if (beta_mcoeff != NULL)
   {
      MFEM_VERIFY(beta_coeff == NULL, "Only one beta coefficient should be defined.");
      a_lor.AddDomainIntegrator(new VectorMassIntegrator(*beta_mcoeff));
   }
   else if (beta_coeff != NULL)
   {
      a_lor.AddDomainIntegrator(new VectorMassIntegrator(*beta_coeff));
   }
   else
   {
      a_lor.AddDomainIntegrator(new VectorMassIntegrator);
   }
   a_lor.UsePrecomputedSparsity();
   a_lor.Assemble();
   a_lor.EliminateEssentialBC(ess_bdr, policy);
   a_lor.Finalize();
   lor_matrix = a_lor.ParallelAssemble();
   lor_matrix->CopyRowStarts();
   lor_matrix->CopyColStarts();

   SetupAMG(fespace_lor_d.GetMesh()->Dimension());

   if (cg_iterations > 0)
   {
      SetupCG(curlcurl_oper, pi, cg_iterations);
   }
   else
   {
      SetupVCycle();
   }
   delete fec_lor;
}

/* G-space constructor

   The auxiliary space solves in general, and this one in particular,
   seem to be quite sensitive to handling of boundary conditions. Note
   some careful choices for Matrix::DiagonalPolicy and the ZeroWrapAMG
   object, as well as the use of a single CG iteration (instead of just
   an AMG V-cycle). Just a V-cycle may be more efficient in some cases,
   but we recommend the CG wrapper for robustness here. */
MatrixFreeAuxiliarySpace::MatrixFreeAuxiliarySpace(
   ParMesh& mesh_lor, Coefficient* beta_coeff,
   MatrixCoefficient* beta_mcoeff, Array<int>& ess_bdr, Operator& curlcurl_oper,
   Operator& g,
#ifdef MFEM_USE_AMGX
   bool useAmgX_,
#endif
   int cg_iterations)
   :
   Solver(curlcurl_oper.Height()),
   comm(mesh_lor.GetComm()),
   matfree(NULL),
   cg(NULL),
#ifdef MFEM_USE_AMGX
   useAmgX(useAmgX_),
#endif
   inner_aux_iterations(0)
{
   H1_FECollection * fec_lor = new H1_FECollection(1, mesh_lor.Dimension());
   ParFiniteElementSpace fespace_lor(&mesh_lor, fec_lor);

   // build LOR AMG v-cycle
   ParBilinearForm a_lor(&fespace_lor);

   // we need something like DIAG_ZERO in the solver, but explicitly doing
   // that makes BoomerAMG setup complain, so instead we constrain the boundary
   // in the CG solver
   const Matrix::DiagonalPolicy policy = Matrix::DIAG_ONE;

   a_lor.SetDiagonalPolicy(policy);

   if (beta_mcoeff != NULL)
   {
      MFEM_VERIFY(beta_coeff == NULL, "Only one beta coefficient should be defined.");
      a_lor.AddDomainIntegrator(new DiffusionIntegrator(*beta_mcoeff));
   }
   else if (beta_coeff != NULL)
   {
      a_lor.AddDomainIntegrator(new DiffusionIntegrator(*beta_coeff));
   }
   else
   {
      a_lor.AddDomainIntegrator(new DiffusionIntegrator);
   }
   a_lor.UsePrecomputedSparsity();
   a_lor.Assemble();
   if (ess_bdr.Size())
   {
      fespace_lor.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // you have to use (serial) BilinearForm eliminate routines to get
   // diag policy DIAG_ZERO all the ParallelEliminateTDofs etc. routines
   // implicitly have a Matrix::DIAG_KEEP policy
   a_lor.EliminateEssentialBC(ess_bdr, policy);
   a_lor.Finalize();
   lor_matrix = a_lor.ParallelAssemble();

   lor_matrix->CopyRowStarts();
   lor_matrix->CopyColStarts();

   SetupAMG(0);

   if (cg_iterations > 0)
   {
      SetupCG(curlcurl_oper, g, cg_iterations);
   }
   else
   {
      SetupVCycle();
   }

   delete fec_lor;
}

void MatrixFreeAuxiliarySpace::SetupCG(
   Operator& curlcurl_oper, Operator& conn,
   int inner_cg_iterations)
{
   MFEM_ASSERT(conn.Height() == curlcurl_oper.Width(),
               "Operators don't match!");
   matfree = new RAPOperator(conn, curlcurl_oper, conn);
   MFEM_ASSERT(matfree->Height() == lor_pc->Height(),
               "Operators don't match!");

   cg = new CGSolver(comm);
   cg->SetOperator(*matfree);
   cg->SetPreconditioner(*lor_pc);
   if (inner_cg_iterations > 99)
   {
      cg->SetRelTol(1.e-14);
      cg->SetMaxIter(100);
   }
   else
   {
      cg->SetRelTol(0.0);
      cg->SetMaxIter(inner_cg_iterations);
   }
   cg->SetPrintLevel(-1);

   aspacewrapper = cg;
}

void MatrixFreeAuxiliarySpace::SetupVCycle()
{
   aspacewrapper = lor_pc;
}

class ZeroWrapAMG : public Solver
{
public:
#ifdef MFEM_USE_AMGX
   ZeroWrapAMG(HypreParMatrix& mat, Array<int>& ess_tdof_list_,
               const bool useAmgX) :
#else
   ZeroWrapAMG(HypreParMatrix& mat, Array<int>& ess_tdof_list_) :
#endif
      Solver(mat.Height()), ess_tdof_list(ess_tdof_list_)
   {
#ifdef MFEM_USE_AMGX
      if (useAmgX)
      {
         const bool amgx_verbose = false;
         AmgXSolver *amgx = new AmgXSolver(mat.GetComm(),
                                           AmgXSolver::PRECONDITIONER,
                                           amgx_verbose);
         amgx->SetOperator(mat);
         amg_ = amgx;
      }
      else
#endif
      {
         HypreBoomerAMG *amg = new HypreBoomerAMG(mat);
         amg->SetPrintLevel(0);
         amg_ = amg;
      }
   }

   void Mult(const Vector& x, Vector& y) const override
   {
      amg_->Mult(x, y);

      auto Y = y.HostReadWrite();
      for (int k : ess_tdof_list)
      {
         Y[k] = 0.0;
      }
   }

   void SetOperator(const Operator&) override { }

   ~ZeroWrapAMG() override
   {
      delete amg_;
   }

private:
   Solver *amg_ = NULL;
   Array<int>& ess_tdof_list;
};

void MatrixFreeAuxiliarySpace::SetupAMG(int system_dimension)
{
   if (system_dimension == 0)
   {
      // boundary condition tweak for G-space solver
#ifdef MFEM_USE_AMGX
      lor_pc = new ZeroWrapAMG(*lor_matrix, ess_tdof_list, useAmgX);
#else
      lor_pc = new ZeroWrapAMG(*lor_matrix, ess_tdof_list);
#endif
   }
   else
   {
      // systems options for Pi-space solver
#ifdef MFEM_USE_AMGX
      if (useAmgX)
      {
         const bool amgx_verbose = false;
         AmgXSolver *amgx = new AmgXSolver(lor_matrix->GetComm(),
                                           AmgXSolver::PRECONDITIONER,
                                           amgx_verbose);
         amgx->SetOperator(*lor_matrix);
         lor_pc = amgx;
      }
      else
#endif
      {
         HypreBoomerAMG* hpc = new HypreBoomerAMG(*lor_matrix);
         hpc->SetSystemsOptions(system_dimension);
         hpc->SetPrintLevel(0);
         lor_pc = hpc;
      }
   }
}

void MatrixFreeAuxiliarySpace::Mult(const Vector& x, Vector& y) const
{
   int rank;
   MPI_Comm_rank(comm, &rank);

   y = 0.0;
   aspacewrapper->Mult(x, y);
   if (cg && rank == 0)
   {
      int q = cg->GetNumIterations();
      inner_aux_iterations += q;
   }
}

MatrixFreeAuxiliarySpace::~MatrixFreeAuxiliarySpace()
{
   delete lor_matrix;
   delete lor_pc;
   delete matfree;
   if (lor_pc != aspacewrapper) { delete aspacewrapper; }
   if (cg != aspacewrapper) { delete cg; }
}

/* As an implementation note, a lot depends on the quality of the auxiliary
   space solves. For high-contrast coefficients, and other difficult problems,
   inner iteration counts may need to be increased. Boundary conditions can
   matter as well (see DIAG_ZERO policy). */
MatrixFreeAMS::MatrixFreeAMS(
   ParBilinearForm& aform, Operator& oper, ParFiniteElementSpace& nd_fespace,
   Coefficient* alpha_coeff, Coefficient* beta_coeff,
   MatrixCoefficient* beta_mcoeff, Array<int>& ess_bdr,
#ifdef MFEM_USE_AMGX
   bool useAmgX,
#endif
   int inner_pi_iterations, int inner_g_iterations, Solver * nd_smoother) :
   Solver(oper.Height())
{
   int order = nd_fespace.GetTypicalFE()->GetOrder();
   ParMesh *mesh = nd_fespace.GetParMesh();
   int dim = mesh->Dimension();

   // smoother
   Array<int> ess_tdof_list;
   nd_fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   if (nd_smoother)
   {
      smoother = nd_smoother;
   }
   else
   {
      const double scale = 0.25;
      smoother = new OperatorJacobiSmoother(aform, ess_tdof_list, scale);
   }

   // get H1 space
   FiniteElementCollection *h1_fec = new H1_FECollection(order, dim);
   h1_fespace = new ParFiniteElementSpace(mesh, h1_fec);
   h1_fespace_d = new ParFiniteElementSpace(mesh, h1_fec, dim, Ordering::byVDIM);

   // build G operator
   pa_grad = new ParDiscreteLinearOperator(h1_fespace, &nd_fespace);
   pa_grad->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_grad->AddDomainInterpolator(new GradientInterpolator);
   pa_grad->Assemble();
   pa_grad->FormRectangularSystemMatrix(Gradient);

   // build Pi operator
   pa_interp = new ParDiscreteLinearOperator(h1_fespace_d, &nd_fespace);
   pa_interp->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   pa_interp->AddDomainInterpolator(new IdentityInterpolator);
   pa_interp->Assemble();
   pa_interp->FormRectangularSystemMatrix(Pi);

   // build LOR space
   ParMesh mesh_lor = ParMesh::MakeRefined(*mesh, order, BasisType::GaussLobatto);

   // build G space solver
   Gspacesolver = new MatrixFreeAuxiliarySpace(mesh_lor, beta_coeff,
                                               beta_mcoeff, ess_bdr, oper,
                                               *Gradient,
#ifdef MFEM_USE_AMGX
                                               useAmgX,
#endif
                                               inner_g_iterations);

   // build Pi space solver
   Pispacesolver = new MatrixFreeAuxiliarySpace(mesh_lor, alpha_coeff,
                                                beta_coeff, beta_mcoeff,
                                                ess_bdr, oper, *Pi,
#ifdef MFEM_USE_AMGX
                                                useAmgX,
#endif
                                                inner_pi_iterations);

   general_ams = new GeneralAMS(oper, *Pi, *Gradient, *Pispacesolver,
                                *Gspacesolver, *smoother, ess_tdof_list);

   delete h1_fec;
}

MatrixFreeAMS::~MatrixFreeAMS()
{
   delete smoother;
   delete pa_grad;
   delete pa_interp;
   delete Gspacesolver;
   delete Pispacesolver;
   delete general_ams;
   delete h1_fespace;
   delete h1_fespace_d;
}

} // namespace mfem

#endif
