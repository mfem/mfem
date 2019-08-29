//                       MFEM Example 5 - Parallel Version
//
// Compile with: make ex5p
//
// Sample runs:  mpirun -np 4 ex5p -m ../data/square-disc.mesh
//               mpirun -np 4 ex5p -m ../data/star.mesh
//               mpirun -np 4 ex5p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex5p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex5p -m ../data/escher.mesh
//               mpirun -np 4 ex5p -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include "mixed_fe_solvers.hpp"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <memory>

using namespace std;
using namespace mfem;

void SetOptions(IterativeSolver& solver, int print_lvl, int max_it,
                double atol, double rtol, bool iter_mode)
{
    solver.SetPrintLevel(print_lvl);
    solver.SetMaxIter(max_it);
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.iterative_mode = iter_mode;
}

void SetOptions(IterativeSolver& solver, const IterSolveParameters& param)
{
    SetOptions(solver, param.print_level, param.max_iter, param.abs_tol,
               param.rel_tol, param.iter_mode);
}

void PrintConvergence(const IterativeSolver& solver, bool verbose)
{
    if (!verbose) return;
    auto msg = solver.GetConverged() ? "converged in " : "did not converge in ";
    cout << "CG " << msg << solver.GetNumIterations() << " iterations. "
         << "Final residual norm is " << solver.GetFinalNorm() << ".\n";
}

void GetSubMatrix(const SparseMatrix& A, const Array<int>& rows,
                  const Array<int>& cols, DenseMatrix& sub_A)
{
    sub_A.SetSize(rows.Size(), cols.Size());
    A.GetSubMatrix(rows, cols, sub_A);
}

SparseMatrix AggToIntDof(const SparseMatrix& agg_elem, const SparseMatrix& elem_dof)
{
    unique_ptr<SparseMatrix> agg_dof(Mult(agg_elem, elem_dof));
    unique_ptr<SparseMatrix> dof_agg(Transpose(*agg_dof));

    int * intdof_agg_i = new int [dof_agg->NumRows()+1]();

    for (int i=0; i < dof_agg->NumRows(); ++i)
    {
        intdof_agg_i[i+1] = intdof_agg_i[i] + (dof_agg->RowSize(i) == 1 && dof_agg->GetRowEntries(i)[0] == 2.0);
    }
    const int nnz = intdof_agg_i[dof_agg->NumRows()];

    int * intdof_agg_j = new int[nnz];
    double * intdof_agg_data = new double[nnz];

    int counter = 0;
    for (int i=0; i< dof_agg->NumRows(); i++)
    {
        if (dof_agg->RowSize(i) == 1 && dof_agg->GetRowEntries(i)[0] == 2.0)
            intdof_agg_j[counter++] = dof_agg->GetRowColumns(i)[0];
    }

    std::fill_n(intdof_agg_data, nnz, 1);

    SparseMatrix intdof_agg(intdof_agg_i, intdof_agg_j, intdof_agg_data,
                            dof_agg->NumRows(), dof_agg->NumCols());

    unique_ptr<SparseMatrix> tmp(Transpose(intdof_agg));
    SparseMatrix agg_intdof;
    agg_intdof.Swap(*tmp);

    return agg_intdof;
}

Vector LocalSolution(const DenseMatrix& M,  const DenseMatrix& B, const Vector& F)
{
    DenseMatrix BT(B, 't');

    if (M.Size() > 0)
    {
        DenseMatrix MinvBT;
        DenseMatrixInverse M_solver(M);
        M_solver.Mult(BT, MinvBT);
        BT = MinvBT;
    }

    DenseMatrix BMinvBT(B.NumRows());
    Mult(B, BT, BMinvBT);

    BMinvBT.SetRow(0, 0);
    BMinvBT.SetCol(0, 0);
    BMinvBT(0, 0) = 1.;

    DenseMatrixInverse BMinvBT_solver(BMinvBT);

    double F0 = F[0];
    const_cast<Vector&>(F)[0] = 0;

    Vector u(B.NumRows());
    Vector sigma(B.NumCols());

    BMinvBT_solver.Mult(F, u);
    BT.Mult(u, sigma);

    const_cast<Vector&>(F)[0] = F0;

    return sigma;
}

SparseMatrix ElemToTrueDofs(const ParFiniteElementSpace& fes)
{
    const int num_elems = fes.GetNE();
    const int num_dofs = fes.GetVSize();
    int* I = const_cast<int*>(fes.GetElementToDofTable().GetI());

    const int nnz = I[num_elems];
    vector<double> D(nnz, 1.0);

    Array<int> J(nnz); //TODO: can we simply use J?
    copy_n(fes.GetElementToDofTable().GetJ(), nnz, J.begin());
    fes.AdjustVDofs(J);

    SparseMatrix el_dof(I, J.GetData(), D.data(), num_elems, num_dofs, 0, 0, 0);
    SparseMatrix true_dofs_restrict;
    fes.Dof_TrueDof_Matrix()->GetDiag(true_dofs_restrict);

    OperatorHandle elem_truedof(Mult(el_dof, true_dofs_restrict));
    return *elem_truedof.As<SparseMatrix>();
}

Vector MLDivPart(const HypreParMatrix& M,
                 const HypreParMatrix& B,
                 const Vector& F,
                 const Array<SparseMatrix>& agg_elem,
                 const Array<SparseMatrix>& elem_hdivdofs,
                 const Array<SparseMatrix>& elem_l2dofs,
                 const Array<OperatorHandle>& P_hdiv,
                 const Array<OperatorHandle>& P_l2,
                 const Array<int>& coarsest_ess_dofs)
{
    const unsigned int num_levels = elem_hdivdofs.Size() + 1;
    OperatorHandle B_l(const_cast<HypreParMatrix*>(&B), false);
    OperatorHandle M_l;//(M.NumRows() ? const_cast<HypreParMatrix*>(&M) : NULL, false);

    Array<Vector> sigma(num_levels);
    Vector F_l, F_a, trash, Pi_F_l, F_coarse, PT_F_l;
    Array<int> loc_hdivdofs, loc_l2dofs;
    SparseMatrix P_l2_l, B_l_diag, M_l_diag;
    DenseMatrix B_a, M_a;

    for (unsigned int l = 0; l < num_levels - 1; ++l)
    {
        OperatorHandle agg_l2dof(Mult(agg_elem[l], elem_l2dofs[l]));
        auto agg_hdivintdof = AggToIntDof(agg_elem[l], elem_hdivdofs[l]);

        // Right hand side: F_l = F - P_l2[l] (P_l2[l]^T P_l2[l])^{-1} P_l2[l]^T F
        F_l = l == 0 ? F : PT_F_l;
        PT_F_l.SetSize(P_l2[l]->NumCols());
        P_l2[l]->MultTranspose(F_l, PT_F_l);

        {
            P_l2[l].As<HypreParMatrix>()->GetDiag(P_l2_l);
            OperatorHandle PT_l2(Transpose(P_l2_l));
            OperatorHandle PTP_l2(Mult(*PT_l2.As<SparseMatrix>(), P_l2_l));

            F_coarse.SetSize(PT_F_l.Size());
            for(int m = 0; m < F_coarse.Size(); m++)
            {
                F_coarse[m] = PT_F_l[m] / PTP_l2.As<SparseMatrix>()->Elem(m, m);
            }
        }

        Pi_F_l.SetSize(F_l.Size());
        P_l2[l]->Mult(F_coarse, Pi_F_l);
        F_l -= Pi_F_l;

        sigma[l].SetSize(agg_hdivintdof.NumCols());
        sigma[l] = 0.0;

        B_l.As<HypreParMatrix>()->GetDiag(B_l_diag);
        if (M_l.Ptr())
        {
            M_l.As<HypreParMatrix>()->GetDiag(M_l_diag);
        }

        for (int agg = 0; agg < agg_hdivintdof.NumRows(); agg++)
        {
            agg_hdivintdof.GetRow(agg, loc_hdivdofs, trash);
            agg_l2dof.As<SparseMatrix>()->GetRow(agg, loc_l2dofs, trash);

            if (M_l.Ptr())
            {
                GetSubMatrix(M_l_diag, loc_hdivdofs, loc_hdivdofs, M_a);
            }
            GetSubMatrix(B_l_diag, loc_l2dofs, loc_hdivdofs, B_a);
            F_l.GetSubVector(loc_l2dofs, F_a);
            sigma[l].AddElementVector(loc_hdivdofs, LocalSolution(M_a, B_a, F_a));
        }  // loop over elements

        // Coarsen problem
        OperatorHandle B_finer(B_l.As<HypreParMatrix>(), B_l.OwnsOperator());
        B_l.SetOperatorOwner(false);
        B_l.MakeRAP(const_cast<OperatorHandle&>(P_l2[l]), B_finer, const_cast<OperatorHandle&>(P_hdiv[l]));

        if (M_l.Ptr())
        {
            OperatorHandle M_finer(M_l.As<HypreParMatrix>(), M_l.OwnsOperator());
            M_l.SetOperatorOwner(false);
            M_l.MakePtAP(M_finer, const_cast<OperatorHandle&>(P_hdiv[l]));
        }
    }  // loop over levels

    // The coarse problem:
    //    B_l->EliminateCols(coarsest_ess_dofs);

    //    if (M_l.Ptr())
    //    {
    //        for ( int k = 0; k < coarsest_ess_dofs.Size(); ++k)
    //            if (coarsest_ess_dofs[k] !=0)
    //                M_l->EliminateRowCol(k);
    //    }

    if (M_l.Ptr())
    {
        OperatorHandle BT_l(B_l.As<HypreParMatrix>()->Transpose());

        Array<int> block_offsets(3);
        block_offsets[0] = 0;
        block_offsets[1] = M_l->NumRows();
        block_offsets[2] = block_offsets[1] + B_l->NumRows();

        BlockOperator coarseMatrix(block_offsets);
        coarseMatrix.SetBlock(0,0, M_l.Ptr());
        coarseMatrix.SetBlock(0,1, BT_l.Ptr());
        coarseMatrix.SetBlock(1,0, B_l.Ptr());

        BlockVector true_rhs(block_offsets);
        true_rhs = 0.0;
        true_rhs.GetBlock(1)= PT_F_l;

        L2H1Preconditioner prec(*M_l.As<HypreParMatrix>(), *B_l.As<HypreParMatrix>(), block_offsets);

        MINRESSolver solver(B.GetComm());
        SetOptions(solver, 0, 500, 1e-12, 1e-9);
        solver.SetOperator(coarseMatrix);
        solver.SetPreconditioner(prec);

        sigma.Last().SetSize(block_offsets[2]);
        sigma.Last() = 0.0;
        solver.Mult(true_rhs, sigma.Last());
        sigma.Last().SetSize(B_l->NumCols());
    }
    else
    {
        BBTSolver BBT_solver(*B_l.As<HypreParMatrix>());

        Vector u_c(B_l->NumRows());
        BBT_solver.Mult(PT_F_l, u_c);

        sigma.Last().SetSize(B_l->NumCols());
        B_l->MultTranspose(u_c, sigma.Last());
    }

    for (int k = num_levels-2; k>=0; k--)
    {
        Vector P_sigma(P_hdiv[k]->NumRows());
        P_hdiv[k]->Mult(sigma[k+1], P_sigma);
        sigma[k] += P_sigma;
    }

    return sigma[0];
}

BBTSolver::BBTSolver(HypreParMatrix& B, IterSolveParameters param)
    : Solver(B.NumRows()),
      BT_(B.Transpose()),
      S_(ParMult(&B, BT_.As<HypreParMatrix>())),
      invS_(*S_.As<HypreParMatrix>()),
      S_solver_(B.GetComm())
{
    invS_.SetPrintLevel(0);
    SetOptions(S_solver_, param);
    S_solver_.SetOperator(*S_.As<HypreParMatrix>());
    S_solver_.SetPreconditioner(invS_);

    MPI_Comm_rank(B.GetComm(), &verbose_);
    verbose_ = (param.print_level) >= 0 && (verbose_ == 0);
}

void BBTSolver::Mult(const Vector &x, Vector &y) const
{
    S_solver_.Mult(x, y);
    PrintConvergence(S_solver_, verbose_);
}

InterpolationCollector::InterpolationCollector(const ParFiniteElementSpace& fes,
                                               int num_refine)
    : coarse_fes_(new ParFiniteElementSpace(fes)), refine_count_(num_refine)
{
    P_.SetSize(num_refine, OperatorHandle(Operator::Hypre_ParCSR));
}

void InterpolationCollector::CollectData(const ParFiniteElementSpace &fes)
{
    fes.GetTrueTransferOperator(*coarse_fes_, P_[--refine_count_]);
    P_[refine_count_].As<HypreParMatrix>()->Threshold(1e-16);
    if (refine_count_)
    {
        coarse_fes_->Update();
    }
    else
    {
        coarse_fes_.reset();
    }
}

HdivL2Hierarchy::HdivL2Hierarchy(const ParFiniteElementSpace& hdiv_fes,
                                 const ParFiniteElementSpace& l2_fes,
                                 int num_refine,
                                 const Array<int>& ess_bdr)
    : agg_el_(num_refine),
      el_hdivdofs_(num_refine),
      el_l2dofs_(num_refine),
      coarse_hdiv_fes_(new ParFiniteElementSpace(hdiv_fes)),
      coarse_l2_fes_(new ParFiniteElementSpace(l2_fes)),
      l2_coll_0(0, coarse_l2_fes_->GetParMesh()->SpaceDimension()),
      l2_fes_0_(new FiniteElementSpace(coarse_l2_fes_->GetParMesh(), &l2_coll_0)),
      refine_count_(num_refine)
{
    P_hdiv_.SetSize(num_refine, OperatorHandle(Operator::Hypre_ParCSR));
    P_l2_.SetSize(num_refine, OperatorHandle(Operator::Hypre_ParCSR));
    coarse_hdiv_fes_->GetEssentialVDofs(ess_bdr, coarse_ess_dofs_);
    l2_fes_0_->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
}

void HdivL2Hierarchy::CollectData(const ParFiniteElementSpace& hdiv_fes,
                                  const ParFiniteElementSpace& l2_fes)
{
    auto& elem_agg_l = (const SparseMatrix&)*l2_fes_0_->GetUpdateOperator();
    OperatorHandle agg_elem_l(Transpose(elem_agg_l));
    agg_el_[--refine_count_].Swap(*agg_elem_l.As<SparseMatrix>());

    el_hdivdofs_[refine_count_] = ElemToTrueDofs(hdiv_fes);
    el_l2dofs_[refine_count_] = ElemToTrueDofs(l2_fes);

    hdiv_fes.GetTrueTransferOperator(*coarse_hdiv_fes_, P_hdiv_[refine_count_]);
    l2_fes.GetTrueTransferOperator(*coarse_l2_fes_, P_l2_[refine_count_]);
    P_hdiv_[refine_count_].As<HypreParMatrix>()->Threshold(1e-16);
    P_l2_[refine_count_].As<HypreParMatrix>()->Threshold(1e-16);

    if (refine_count_)
    {
        coarse_hdiv_fes_->Update();
        coarse_l2_fes_->Update();
    }
    else
    {
        coarse_hdiv_fes_.reset();
        coarse_l2_fes_.reset();
        l2_fes_0_.reset();
    }
}

MLDivFreeSolver::MLDivFreeSolver(HdivL2Hierarchy& hierarchy, HypreParMatrix& M,
                                 HypreParMatrix& B, HypreParMatrix& C,
                                 MLDivFreeSolveParameters param)
    : h_(hierarchy), M_(M), B_(B), C_(C), BBT_solver_(B, param.BBT_solve_param),
      CTMC_solver_(C.GetComm()), param_(param), offsets_(3)
{
    offsets_[0] = 0;
    offsets_[1] = M.NumCols();
    offsets_[2] = offsets_[1] + B.NumRows();

    OperatorHandle MC(ParMult(&M_, &C));
    OperatorHandle CT(C.Transpose());
    CTMC_.Reset(ParMult(CT.As<HypreParMatrix>(), MC.As<HypreParMatrix>()));
    CTMC_solver_.SetOperator(*CTMC_);
    SetOptions(CTMC_solver_, param.CTMC_solve_param);
}

void MLDivFreeSolver::SetupMG(const InterpolationCollector& P)
{
    CTMC_prec_.Reset(new Multigrid(*CTMC_.As<HypreParMatrix>(), P.GetP()));
    CTMC_solver_.SetPreconditioner(*CTMC_prec_.As<Solver>());
}

void MLDivFreeSolver::SetupAMS(ParFiniteElementSpace& hcurl_fes)
{
    CTMC_prec_.Reset(new HypreAMS(*CTMC_.As<HypreParMatrix>(), &hcurl_fes));
    CTMC_prec_.As<HypreAMS>()->SetSingularProblem();
    CTMC_solver_.SetPreconditioner(*CTMC_prec_.As<Solver>());
}

void MLDivFreeSolver::SolveParticular(const Vector& rhs, Vector& sol) const
{
    if (param_.ml_part)
    {
        sol = MLDivPart(M_, B_, rhs, h_.agg_el_, h_.el_hdivdofs_, h_.el_l2dofs_,
                        h_.P_hdiv_, h_.P_l2_, h_.coarse_ess_dofs_);
    }
    else
    {
        Vector potential(rhs.Size());
        BBT_solver_.Mult(rhs, potential);
        B_.MultTranspose(potential, sol);
    }
}

void MLDivFreeSolver::SolveDivFree(const Vector &rhs, Vector& sol) const
{
    // Compute the right hand side for the divergence free solver problem
    Vector rhs_divfree(CTMC_->NumRows());
    C_.MultTranspose(rhs, rhs_divfree);

    // Solve the "potential" of divergence free solution
    Vector potential_divfree(CTMC_->NumRows());
    CTMC_solver_.Mult(rhs_divfree, potential_divfree);
    PrintConvergence(CTMC_solver_, param_.verbose);

    // Compute divergence free solution
    C_.Mult(potential_divfree, sol);
}

void MLDivFreeSolver::SolvePotential(const Vector& rhs, Vector& sol) const
{
    Vector rhs_p(B_.NumRows());
    B_.Mult(rhs, rhs_p);
    BBT_solver_.Mult(rhs_p, sol);
}

void MLDivFreeSolver::Mult(const Vector & x, Vector & y) const
{
    MFEM_VERIFY(x.Size() == offsets_[2], "MLDivFreeSolver: x size mismatch");
    MFEM_VERIFY(y.Size() == offsets_[2], "MLDivFreeSolver: y size mismatch");

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();

    BlockVector blk_x(BlockVector(x.GetData(), offsets_));
    BlockVector blk_y(y.GetData(), offsets_);

    Vector& particular_flux = blk_y.GetBlock(0);
    SolveParticular(blk_x.GetBlock(1), particular_flux);

    if (param_.verbose)
        cout << "Particular solution found in " << chrono.RealTime() << "s.\n";

    chrono.Clear();
    chrono.Start();

    Vector divfree_flux(C_.NumRows());
    M_.Mult(-1.0, particular_flux, 1.0, blk_x.GetBlock(0));
    SolveDivFree(blk_x.GetBlock(0), divfree_flux);

    if (param_.verbose)
        cout << "Divergence free solution found in " << chrono.RealTime() << "s.\n";

    blk_y.GetBlock(0) += divfree_flux;

    // Compute the right hand side for the pressure problem BB^T p = rhs_p
    chrono.Clear();
    chrono.Start();

    M_.Mult(-1.0, divfree_flux, 1.0, blk_x.GetBlock(0));
    SolvePotential(blk_x.GetBlock(0), blk_y.GetBlock(1));

    if (param_.verbose)
        cout << "Scalar potential found in " << chrono.RealTime() << "s.\n";
}

Multigrid::Multigrid(HypreParMatrix& op,
                     const Array<OperatorHandle>& P,
                     OperatorHandle coarse_solver)
    :
      Solver(op.GetNumRows()),
      P_(P),
      ops_(P.Size()+1),
      smoothers_(ops_.Size()),
      coarse_solver_(coarse_solver.Ptr(), false),
      correct_(ops_.Size()),
      resid_(ops_.Size())
{
    ops_[0].Reset(&op, false);
    smoothers_[0].Reset(new HypreSmoother(op));

    for (int l = 1; l < ops_.Size(); ++l)
    {
        ops_[l].MakePtAP(ops_[l-1], const_cast<OperatorHandle&>(P_[l-1]));
        smoothers_[l].Reset(new HypreSmoother(*ops_[l].As<HypreParMatrix>()));
        resid_[l].SetSize(ops_[l]->NumRows());
        correct_[l].SetSize(ops_[l]->NumRows());
    }
}

void Multigrid::Mult(const Vector& x, Vector& y) const
{
    resid_[0] = x;
    correct_[0].SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle(0);
}

void Multigrid::MG_Cycle(int level) const
{
    const HypreParMatrix* op_l = ops_[level].As<HypreParMatrix>();

    // PreSmoothing
    smoothers_[level]->Mult(resid_[level], correct_[level]);
    op_l->Mult(-1., correct_[level], 1., resid_[level]);

    // Coarse grid correction
    cor_cor_.SetSize(resid_[level].Size());
    if (level < P_.Size())
    {
        P_[level]->MultTranspose(resid_[level], resid_[level+1]);
        MG_Cycle(level+1);
        cor_cor_.SetSize(resid_[level].Size());
        P_[level]->Mult(correct_[level+1], cor_cor_);
        correct_[level] += cor_cor_;
        op_l->Mult(-1.0, cor_cor_, 1.0, resid_[level]);
    }
    else if (coarse_solver_.Ptr())
    {
        coarse_solver_->Mult(resid_[level], cor_cor_);
        correct_[level] += cor_cor_;
        op_l->Mult(-1.0, cor_cor_, 1.0, resid_[level]);
    }

    // PostSmoothing
    smoothers_[level]->Mult(resid_[level], cor_cor_);
    correct_[level] += cor_cor_;
}

L2H1Preconditioner::L2H1Preconditioner(HypreParMatrix& M,
                                       HypreParMatrix& B,
                                       const Array<int>& offsets)
    : BlockDiagonalPreconditioner(offsets)
{
    Vector Md;
    M.GetDiag(Md);
    OperatorHandle MinvBt(B.Transpose());
    MinvBt.As<HypreParMatrix>()->InvScaleRows(Md);
    S_.Reset(ParMult(&B, MinvBt.As<HypreParMatrix>()));
    S_.As<HypreParMatrix>()->CopyRowStarts();
    S_.As<HypreParMatrix>()->CopyColStarts();

    SetDiagonalBlock(0, new HypreDiagScale(M));
    SetDiagonalBlock(1, new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
    static_cast<HypreBoomerAMG&>(GetDiagonalBlock(1)).SetPrintLevel(0);
    owns_blocks = true;
}

