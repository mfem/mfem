
#include "mixed_fe_solvers.hpp"

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

SparseMatrix AggToInteriorDof(const SparseMatrix& agg_elem,
                              const SparseMatrix& elem_dof,
                              const Array<int>& bdr_dofs)
{
    unique_ptr<SparseMatrix> agg_dof(Mult(agg_elem, elem_dof));
    unique_ptr<SparseMatrix> dof_agg(Transpose(*agg_dof));

    int * I = new int [dof_agg->NumRows()+1]();
    int * J = new int[dof_agg->NumNonZeroElems()];

    Array<int> is_bdr;
    FiniteElementSpace::ListToMarker(bdr_dofs, elem_dof.NumCols(), is_bdr);

    int counter = 0;
    for (int i = 0; i < dof_agg->NumRows(); ++i)
    {
        if (dof_agg->RowSize(i) > 1 || is_bdr[i]) { I[i+1] = I[i]; continue; }
        I[i+1] = I[i] + 1;
        J[counter++] = dof_agg->GetRowColumns(i)[0];
    }

    double * D = new double[I[dof_agg->NumRows()]];
    std::fill_n(D, I[dof_agg->NumRows()], 1.0);

    SparseMatrix intdof_agg(I, J, D, dof_agg->NumRows(), dof_agg->NumCols());

    unique_ptr<SparseMatrix> tmp(Transpose(intdof_agg));
    SparseMatrix agg_intdof;
    agg_intdof.Swap(*tmp);

    return agg_intdof;
}

void MakeRHS(const HypreParMatrix& P_l2_l, Vector F_coarse, Vector& F_l)
{
    SparseMatrix P_l2;
    P_l2_l.GetDiag(P_l2);
    OperatorPtr PT_l2(Transpose(P_l2));
    OperatorPtr PTP_l2(Mult(*PT_l2.As<SparseMatrix>(), P_l2));

    for(int m = 0; m < F_coarse.Size(); m++)
    {
        F_coarse[m] /= PTP_l2.As<SparseMatrix>()->Elem(m, m); //TODO: high order
    }

    P_l2_l.Mult(-1.0, F_coarse, 1.0, F_l);
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
    const int nnz = fes.GetElementToDofTable().Size_of_connections();

    vector<double> D(nnz, 1.0);
    int* I = const_cast<int*>(fes.GetElementToDofTable().GetI());
    Array<int> J(nnz);
    copy_n(fes.GetElementToDofTable().GetJ(), nnz, J.begin());
    fes.AdjustVDofs(J);

    SparseMatrix el_dof(I, J, D.data(), fes.GetNE(), fes.GetVSize(), 0, 0, 0);
    SparseMatrix true_dofs_restrict;
    fes.Dof_TrueDof_Matrix()->GetDiag(true_dofs_restrict);
    OperatorPtr elem_truedof(Mult(el_dof, true_dofs_restrict));
    return *elem_truedof.As<SparseMatrix>();
}

Vector MLDivPart(const HypreParMatrix& M,
                 const HypreParMatrix& B,
                 const Vector& F,
                 const Array<SparseMatrix>& agg_elem,
                 const Array<SparseMatrix>& elem_hdivdofs,
                 const Array<SparseMatrix>& elem_l2dofs,
                 const Array<OperatorPtr>& P_hdiv,
                 const Array<OperatorPtr>& P_l2,
                 const Array<Array<int> >& bdr_hdivdofs)
{
    const unsigned int num_levels = elem_hdivdofs.Size() + 1;
    OperatorPtr B_l(const_cast<HypreParMatrix*>(&B), false);
    OperatorPtr M_l;//(M.NumRows() ? const_cast<HypreParMatrix*>(&M) : NULL, false);

    Array<Vector> sigma(num_levels);
    Vector F_l, F_a, trash, PT_F_l;
    Array<int> loc_hdivdofs, loc_l2dofs;
    SparseMatrix B_l_diag, M_l_diag;
    DenseMatrix B_a, M_a;

    for (unsigned int l = 0; l < num_levels - 1; ++l)
    {
        OperatorPtr agg_l2dof(Mult(agg_elem[l], elem_l2dofs[l]));
        auto agg_hdivdof = AggToInteriorDof(agg_elem[l], elem_hdivdofs[l], bdr_hdivdofs[l]);

        // Right hand side: F_l = F - P_l2[l] (P_l2[l]^T P_l2[l])^{-1} P_l2[l]^T F
        F_l = l == 0 ? F : PT_F_l;
        PT_F_l.SetSize(P_l2[l]->NumCols());
        P_l2[l]->MultTranspose(F_l, PT_F_l);

        MakeRHS(*P_l2[l].As<HypreParMatrix>(), PT_F_l, F_l);

        sigma[l].SetSize(agg_hdivdof.NumCols());
        sigma[l] = 0.0;

        if (M_l.Ptr()) M_l.As<HypreParMatrix>()->GetDiag(M_l_diag);
        B_l.As<HypreParMatrix>()->GetDiag(B_l_diag);

        for (int agg = 0; agg < agg_hdivdof.NumRows(); agg++)
        {
            agg_hdivdof.GetRow(agg, loc_hdivdofs, trash);
            agg_l2dof.As<SparseMatrix>()->GetRow(agg, loc_l2dofs, trash);

            if (M_l.Ptr()) GetSubMatrix(M_l_diag, loc_hdivdofs, loc_hdivdofs, M_a);
            GetSubMatrix(B_l_diag, loc_l2dofs, loc_hdivdofs, B_a);
            F_l.GetSubVector(loc_l2dofs, F_a);
            sigma[l].AddElementVector(loc_hdivdofs, LocalSolution(M_a, B_a, F_a));
        }  // loop over elements

        // Coarsen problem
        OperatorPtr B_finer(B_l.As<HypreParMatrix>(), B_l.OwnsOperator());
        B_l.SetOperatorOwner(false);
        B_l.MakeRAP(const_cast<OperatorPtr&>(P_l2[l]), B_finer, const_cast<OperatorPtr&>(P_hdiv[l]));

        if (M_l.Ptr())
        {
            OperatorPtr M_finer(M_l.As<HypreParMatrix>(), M_l.OwnsOperator());
            M_l.SetOperatorOwner(false);
            M_l.MakePtAP(M_finer, const_cast<OperatorPtr&>(P_hdiv[l]));
        }
    }  // loop over levels

    // The coarse problem:
//    B_l.As<HypreParMatrix>()->GetDiag(B_l_diag);
//    for (int dof : bdr_hdivdofs.Last()) B_l_diag.EliminateCol(dof);

    if (M_l.Ptr())
    {
        Array<int> block_offsets(3);
        block_offsets[0] = 0;
        block_offsets[1] = M_l->NumRows();
        block_offsets[2] = block_offsets[1] + B_l->NumRows();

//        OperatorPtr M_l_elim;
//        M_l_elim.EliminateRowsCols(M_l, bdr_hdivdofs.Last());
        OperatorPtr BT_l(B_l.As<HypreParMatrix>()->Transpose());

        BlockOperator coarseMatrix(block_offsets);
        coarseMatrix.SetBlock(0,0, M_l.Ptr());
        coarseMatrix.SetBlock(0,1, BT_l.Ptr());
        coarseMatrix.SetBlock(1,0, B_l.Ptr());

        BlockVector true_rhs(block_offsets);
        true_rhs.GetBlock(0) = 0.0;
        true_rhs.GetBlock(1)= PT_F_l;

        L2H1Preconditioner prec(*M_l.As<HypreParMatrix>(), *B_l.As<HypreParMatrix>(), block_offsets);

        MINRESSolver solver(B.GetComm());
        SetOptions(solver, 0, 500, 1e-12, 1e-9, false);
        solver.SetOperator(coarseMatrix);
        solver.SetPreconditioner(prec);

        sigma.Last().SetSize(block_offsets[2]);
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

    for (int l = num_levels-2; l>=0; l--) // add solutions from all levels
    {
        Vector P_sigma(P_hdiv[l]->NumRows());
        P_hdiv[l]->Mult(sigma[l+1], P_sigma);
        sigma[l] += P_sigma;
    }

    return sigma[0];
}

BBTSolver::BBTSolver(const HypreParMatrix& B, bool B_has_nullity_one,
                     IterSolveParameters param)
    : Solver(B.NumRows()),
      BT_(B.Transpose()),
      S_(ParMult(&B, BT_.As<HypreParMatrix>())),
      S_solver_(B.GetComm())
{
    MPI_Comm_rank(B.GetComm(), &verbose_);
    B_has_nullity_one_ = B_has_nullity_one && !verbose_; // verbose_ = MPI rank

    Array<int> ess_dofs(B_has_nullity_one_ ? 1 : 0);
    ess_dofs = 0;
    OperatorPtr S_elim;
    S_elim.EliminateRowsCols(S_, ess_dofs);

    invS_.Reset(new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
    invS_.As<HypreBoomerAMG>()->SetPrintLevel(0);

    SetOptions(S_solver_, param);
    S_solver_.SetOperator(*S_);
    S_solver_.SetPreconditioner(*invS_.As<HypreBoomerAMG>());

    verbose_ = (param.print_level) >= 0 && (verbose_ == 0);
}

void BBTSolver::Mult(const Vector &x, Vector &y) const
{
    double x_0 = x[0];
    if (B_has_nullity_one_) const_cast<Vector&>(x)[0] = 0.0;
    S_solver_.Mult(x, y);
    if (B_has_nullity_one_) const_cast<Vector&>(x)[0] = x_0;
    PrintConvergence(S_solver_, verbose_);
}

DivFreeSolverDataCollector::
DivFreeSolverDataCollector(const ParFiniteElementSpace& hdiv_fes,
                           const ParFiniteElementSpace& l2_fes,
                           int num_refine,
                           const Array<int>& ess_bdr,
                           const DivFreeSolverParameters& param)
    : l2_0_fec_(0, l2_fes.GetMesh()->Dimension()), ess_bdr_(ess_bdr), level_(num_refine),
      data_(hdiv_fes.GetOrder(0), const_cast<ParFiniteElementSpace&>(l2_fes).GetParMesh())
{
    data_.param = param;
    if (data_.param.ml_particular)
    {
        all_bdr_.SetSize(ess_bdr.Size(), 1);
        coarse_hdiv_fes_.reset(new ParFiniteElementSpace(hdiv_fes));
        coarse_l2_fes_.reset(new ParFiniteElementSpace(l2_fes));
        l2_0_fes_.reset(new FiniteElementSpace(l2_fes.GetMesh(), &l2_0_fec_));
        l2_0_fes_->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);

        data_.agg_el.SetSize(num_refine);
        data_.el_hdivdofs.SetSize(num_refine);
        data_.el_l2dofs.SetSize(num_refine);
        data_.P_hdiv.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
        data_.P_l2.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
        data_.bdr_hdivdofs.SetSize(num_refine+1);
        coarse_hdiv_fes_->GetEssentialTrueDofs(all_bdr_, data_.bdr_hdivdofs[level_]);
    }

    if (data_.param.MG_type == GeometricMG)
    {
        coarse_hcurl_fes_.reset(new ParFiniteElementSpace(data_.hcurl_fes));
        data_.P_curl.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
    }
}

void DivFreeSolverDataCollector::CollectData(const ParFiniteElementSpace& hdiv_fes,
                                             const ParFiniteElementSpace& l2_fes)
{
    auto hdiv_fes_ref = const_cast<ParFiniteElementSpace&>(hdiv_fes);
    data_.hcurl_fes.Update();
    --level_;

    if (data_.param.ml_particular)
    {
        auto& elem_agg_l = (const SparseMatrix&)*l2_0_fes_->GetUpdateOperator();
        OperatorPtr agg_elem_l(Transpose(elem_agg_l));
        data_.agg_el[level_].Swap(*agg_elem_l.As<SparseMatrix>());

        data_.el_hdivdofs[level_] = ElemToTrueDofs(hdiv_fes);
        data_.el_l2dofs[level_] = ElemToTrueDofs(l2_fes);

        hdiv_fes.GetTrueTransferOperator(*coarse_hdiv_fes_, data_.P_hdiv[level_]);
        l2_fes.GetTrueTransferOperator(*coarse_l2_fes_, data_.P_l2[level_]);
        data_.P_hdiv[level_].As<HypreParMatrix>()->Threshold(1e-16);
        data_.P_l2[level_].As<HypreParMatrix>()->Threshold(1e-16);

        hdiv_fes_ref.GetEssentialTrueDofs(all_bdr_, data_.bdr_hdivdofs[level_]);
        // TODO:  include shared true dofs in bdr_hdivdofs
        level_ ? coarse_hdiv_fes_->Update() : coarse_hdiv_fes_.reset();
        level_ ? coarse_l2_fes_->Update() : coarse_l2_fes_.reset();
    }

    if (data_.param.MG_type == GeometricMG)
    {
        data_.hcurl_fes.GetTrueTransferOperator(*coarse_hcurl_fes_, data_.P_curl[level_]);
        data_.P_curl[level_].As<HypreParMatrix>()->Threshold(1e-16);
        level_ ? coarse_hcurl_fes_->Update() : coarse_hcurl_fes_.reset();
    }

    if (level_ == 0)
    {
        Vector trash1(data_.hcurl_fes.GetVSize()), trash2(hdiv_fes.GetVSize());
        ParDiscreteLinearOperator curl(&data_.hcurl_fes, &hdiv_fes_ref);
        curl.AddDomainInterpolator(new CurlInterpolator);
        curl.Assemble();
        curl.EliminateTrialDofs(ess_bdr_, trash1, trash2);
        curl.Finalize();
        data_.discrete_curl.Reset(curl.ParallelAssemble());

        l2_0_fes_.reset();
    }
}

DivFreeSolver::DivFreeSolver(const HypreParMatrix &M, const HypreParMatrix& B,
                             const DivFreeSolverData& data)
    : Solver(M.NumRows()+B.NumRows()), M_(M), B_(B),
      BBT_solver_(B, data.param.B_has_nullity_one, data.param.BBT_solve_param),
      CTMC_solver_(B_.GetComm()),
      offsets_(3), data_(data)
{
    offsets_[0] = 0;
    offsets_[1] = M.NumCols();
    offsets_[2] = offsets_[1] + B.NumRows();

    OperatorPtr MC(ParMult(&M_, data.discrete_curl.As<HypreParMatrix>()));
    OperatorPtr CT(data.discrete_curl.As<HypreParMatrix>()->Transpose());
    CTMC_.Reset(ParMult(CT.As<HypreParMatrix>(), MC.As<HypreParMatrix>()));
    CTMC_.As<HypreParMatrix>()->CopyRowStarts();
    CTMC_.As<HypreParMatrix>()->EliminateZeroRows();
    CTMC_solver_.SetOperator(*CTMC_);

    if (data_.param.MG_type == AlgebraicMG)
    {
        auto hcurl_fes = const_cast<ParFiniteElementSpace*>(&data_.hcurl_fes);
        CTMC_prec_.Reset(new HypreAMS(*CTMC_.As<HypreParMatrix>(), hcurl_fes));
        CTMC_prec_.As<HypreAMS>()->SetSingularProblem();
    }
    else
    {
        CTMC_prec_.Reset(new Multigrid(*CTMC_.As<HypreParMatrix>(), data_.P_curl));
    }
    CTMC_solver_.SetPreconditioner(*CTMC_prec_.As<Solver>());
    SetOptions(CTMC_solver_, data_.param.CTMC_solve_param);
}

void DivFreeSolver::SolveParticular(const Vector& rhs, Vector& sol) const
{
    if (data_.param.ml_particular)
    {
        sol = MLDivPart(M_, B_, rhs, data_.agg_el, data_.el_hdivdofs, data_.el_l2dofs,
                        data_.P_hdiv, data_.P_l2, data_.bdr_hdivdofs);
    }
    else
    {
        Vector potential(rhs.Size());
        BBT_solver_.Mult(rhs, potential);
        B_.MultTranspose(potential, sol);
    }
}

void DivFreeSolver::SolveDivFree(const Vector &rhs, Vector& sol) const
{
    Vector rhs_divfree(CTMC_->NumRows());
    data_.discrete_curl->MultTranspose(rhs, rhs_divfree);

    Vector potential_divfree(CTMC_->NumRows());
    CTMC_solver_.Mult(rhs_divfree, potential_divfree);
    PrintConvergence(CTMC_solver_, data_.param.verbose);

    data_.discrete_curl->Mult(potential_divfree, sol);
}

void DivFreeSolver::SolvePotential(const Vector& rhs, Vector& sol) const
{
    Vector rhs_p(B_.NumRows());
    B_.Mult(rhs, rhs_p);
    BBT_solver_.Mult(rhs_p, sol);
}

void DivFreeSolver::Mult(const Vector & x, Vector & y) const
{
    MFEM_VERIFY(x.Size() == offsets_[2], "MLDivFreeSolver: x size is invalid");
    MFEM_VERIFY(y.Size() == offsets_[2], "MLDivFreeSolver: y size is invalid");

    StopWatch ch;
    ch.Start();

    BlockVector blk_x(BlockVector(x.GetData(), offsets_));
    BlockVector blk_y(y.GetData(), offsets_);

    Vector particular_flux(blk_y.BlockSize(0));
    SolveParticular(blk_x.GetBlock(1), particular_flux);
    blk_y.GetBlock(0) += particular_flux;

    if (data_.param.verbose)
        cout << "Particular solution found in " << ch.RealTime() << "s.\n";

    ch.Clear();
    ch.Start();

    Vector divfree_flux(blk_y.BlockSize(0));
    M_.Mult(-1.0, particular_flux, 1.0, blk_x.GetBlock(0));
    SolveDivFree(blk_x.GetBlock(0), divfree_flux);
    blk_y.GetBlock(0) += divfree_flux;

    if (data_.param.verbose)
        cout << "Divergence free solution found in " << ch.RealTime() << "s.\n";

    ch.Clear();
    ch.Start();

    M_.Mult(-1.0, divfree_flux, 1.0, blk_x.GetBlock(0));
    SolvePotential(blk_x.GetBlock(0), blk_y.GetBlock(1));

    if (data_.param.verbose)
        cout << "Scalar potential found in " << ch.RealTime() << "s.\n";
}

Multigrid::Multigrid(HypreParMatrix& op,
                     const Array<OperatorPtr>& P,
                     OperatorPtr coarse_solver)
    : Solver(op.GetNumRows()),
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
        ops_[l].MakePtAP(ops_[l-1], const_cast<OperatorPtr&>(P_[l-1]));
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
    OperatorPtr MinvBt(B.Transpose());
    MinvBt.As<HypreParMatrix>()->InvScaleRows(Md);
    S_.Reset(ParMult(&B, MinvBt.As<HypreParMatrix>()));
    S_.As<HypreParMatrix>()->CopyRowStarts();
    S_.As<HypreParMatrix>()->CopyColStarts();

    SetDiagonalBlock(0, new HypreDiagScale(M));
    SetDiagonalBlock(1, new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
    static_cast<HypreBoomerAMG&>(GetDiagonalBlock(1)).SetPrintLevel(0);
    owns_blocks = true;
}

