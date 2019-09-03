
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
    auto name = dynamic_cast<const CGSolver*>(&solver) ? "CG " : "MINRES ";
    auto msg = solver.GetConverged() ? "converged in " : "did not converge in ";
    cout << name << msg << solver.GetNumIterations() << " iterations. "
         << "Final residual norm is " << solver.GetFinalNorm() << ".\n";
}

HypreParMatrix* TwoStepsRAP(const HypreParMatrix& Rt, const HypreParMatrix& A,
                            const HypreParMatrix& P)
{
    OperatorPtr R(Rt.Transpose());
    OperatorPtr RA(ParMult(R.As<HypreParMatrix>(), &A));
    HypreParMatrix* RAP = ParMult(RA.As<HypreParMatrix>(), &P);
    RAP->CopyRowStarts();
    RAP->CopyColStarts();
    return RAP;
}

void GetRowColumnsRef(SparseMatrix& A, int row, Array<int>& cols)
{
    cols.MakeRef(A.GetRowColumns(row), A.RowSize(row));
}

void GetSubMatrix(const SparseMatrix& A, const Array<int>& rows,
                  const Array<int>& cols, DenseMatrix& sub_A)
{
    sub_A.SetSize(rows.Size(), cols.Size());
    A.GetSubMatrix(rows, cols, sub_A);
}

SparseMatrix GetSubMatrix(const SparseMatrix& A, const Array<int>& rows,
                          const Array<int>& cols, Array<int>& col_marker)
{
    if (rows.Size() == 0 || cols.Size() == 0)
    {
        SparseMatrix out(rows.Size(), cols.Size());
        out.Finalize();
        return out;
    }

    const int* i_A = A.GetI();
    const int* j_A = A.GetJ();
    const double* a_A = A.GetData();

    MFEM_ASSERT(rows.Size() && rows.Max() < A.NumRows(), "incompatible rows");
    MFEM_ASSERT(cols.Size() && cols.Max() < A.NumCols(), "incompatible rows");
    MFEM_ASSERT(col_marker.Size() >= A.NumCols(), "incompatible col_marker");

    for (int jcol = 0; jcol < cols.Size(); ++jcol)
        col_marker[cols[jcol]] = jcol;

    const int nrow_sub = rows.Size();
    const int ncol_sub = cols.Size();

    int* i_sub = new int[nrow_sub+1]();

    // Find the number of nnz.
    int nnz = 0;
    for (int i = 0; i < nrow_sub; ++i)
    {
        const int r = rows[i];

        for (int j = i_A[r]; j < i_A[r+1]; ++j)
            if (col_marker[j_A[j]] >= 0) ++nnz;

        i_sub[i+1] = nnz;
    }

    // Allocate memory
    int* j_sub = new int[nnz];
    double* a_sub = new double[nnz];

    // Fill in the matrix
    int count = 0;
    for (int i = 0; i < nrow_sub; ++i)
    {
        const int current_row = rows[i];
        for (int j = i_A[current_row]; j < i_A[current_row + 1]; ++j)
        {
            if (col_marker[j_A[j]] >= 0)
            {
                j_sub[count] = col_marker[j_A[j]];
                a_sub[count++] = a_A[j];
            }
        }
    }

    // Restore colMapper so it can be reused other times!
    for (int jcol = 0; jcol < cols.Size(); ++jcol)
        col_marker[cols[jcol]] = -1;

    return SparseMatrix(i_sub, j_sub, a_sub, nrow_sub, ncol_sub);
}

SparseMatrix* AggToInteriorDof(const SparseMatrix& agg_elem,
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
    return Transpose(intdof_agg);
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

BBTSolver::BBTSolver(const HypreParMatrix& B, bool B_has_nullity_one,
                     IterSolveParameters param)
    : Solver(B.NumRows()), BBT_solver_(B.GetComm())
{
    OperatorPtr BT(B.Transpose());
    BBT_.Reset(ParMult(&B, BT.As<HypreParMatrix>()));
    BBT_.As<HypreParMatrix>()->CopyColStarts();

    MPI_Comm_rank(B.GetComm(), &verbose_);
    B_has_nullity_one_ = B_has_nullity_one && !verbose_; // verbose_ = MPI rank

    Array<int> ess_dofs(B_has_nullity_one_ ? 1 : 0);
    ess_dofs = 0;
    OperatorPtr BBT_elim;
    BBT_elim.EliminateRowsCols(BBT_, ess_dofs);

    BBT_prec_.Reset(new HypreBoomerAMG(*BBT_.As<HypreParMatrix>()));
    BBT_prec_.As<HypreBoomerAMG>()->SetPrintLevel(0);

    SetOptions(BBT_solver_, param);
    BBT_solver_.SetOperator(*BBT_);
    BBT_solver_.SetPreconditioner(*BBT_prec_.As<HypreBoomerAMG>());

    verbose_ = (param.print_level) >= 0 && (verbose_ == 0);
}

void BBTSolver::Mult(const Vector &x, Vector &y) const
{
    double x_0 = x[0];
    if (B_has_nullity_one_) const_cast<Vector&>(x)[0] = 0.0;
    BBT_solver_.Mult(x, y);
    if (B_has_nullity_one_) const_cast<Vector&>(x)[0] = x_0;
    PrintConvergence(BBT_solver_, verbose_);
}

LocalSolver::LocalSolver(const DenseMatrix& B)
    : Solver(B.NumCols()), BT_(B, 't'), BBT_(B.NumRows())
{
    mfem::Mult(B, BT_, BBT_);
    BBT_.SetRow(0, 0);
    BBT_.SetCol(0, 0);
    BBT_(0, 0) = 1.;
    BBT_solver_.SetOperator(BBT_);
}

void LocalSolver::Mult(const Vector &x, Vector &y) const
{
    double x0 = x[0];
    const_cast<Vector&>(x)[0] = 0.0;

    Vector u(BT_.NumCols());
    BBT_solver_.Mult(x, u);

    y.SetSize(BT_.NumRows());
    BT_.Mult(u, y);
    const_cast<Vector&>(x)[0] = x0;
}

BlockDiagSolver::BlockDiagSolver(const OperatorPtr &A, const SparseMatrix& block_dof)
    : Solver(A->NumRows()), block_dof_ref_(const_cast<SparseMatrix&>(block_dof)),
      block_solver_(block_dof.NumRows())
{
    SparseMatrix A_diag;
    A.As<HypreParMatrix>()->GetDiag(A_diag);
    DenseMatrix sub_A;
    for(int block = 0; block < block_dof.NumRows(); block++)
    {
        GetRowColumnsRef(block_dof_ref_, block, local_dofs_);
        GetSubMatrix(A_diag, local_dofs_, local_dofs_, sub_A);
        block_solver_[block].SetOperator(sub_A);
    }
}

void BlockDiagSolver::Mult(const Vector &x, Vector &y) const
{
    y.SetSize(x.Size());
    y = 0.0;

    for(int block = 0; block < block_dof_ref_.NumRows(); block++)
    {
        GetRowColumnsRef(block_dof_ref_, block, local_dofs_);
        x.GetSubVector(local_dofs_, sub_rhs_);
        sub_sol_.SetSize(local_dofs_.Size());
        block_solver_[block].Mult(sub_rhs_, sub_sol_);
        y.AddElementVector(local_dofs_, sub_sol_);
    }
}

void AddSharedTDofs(const HypreParMatrix& dof_tdof, Array<int>& bdr_tdofs)
{
    OperatorPtr tdof_dof(dof_tdof.Transpose());
    OperatorPtr dof_tdof_dof(ParMult(&dof_tdof, tdof_dof.As<HypreParMatrix>()));

    SparseMatrix dof_is_shared, dof_is_owned;
    HYPRE_Int* trash;
    dof_tdof_dof.As<HypreParMatrix>()->GetOffd(dof_is_shared, trash);
    dof_tdof.GetDiag(dof_is_owned);

    Array<int> shared_tdofs;
    shared_tdofs.Reserve(dof_is_shared.NumNonZeroElems());
    for (int i = 0; i < dof_is_shared.NumRows(); ++i)
    {
        if (dof_is_shared.RowSize(i) && dof_is_owned.RowSize(i))
        {
            shared_tdofs.Append(dof_is_owned.GetRowColumns(i)[0]);
        }
    }
    bdr_tdofs.Append(shared_tdofs);
}

DivFreeSolverDataCollector::
DivFreeSolverDataCollector(int order, int num_refine, ParMesh *mesh,
                           const Array<int>& ess_bdr,
                           const DivFreeSolverParameters& param)
    : hdiv_fec_(order, mesh->Dimension()), l2_fec_(order, mesh->Dimension()),
      hcurl_fec_(order+1, mesh->Dimension()), l2_0_fec_(0, mesh->Dimension()),
      ess_bdr_(ess_bdr), level_(num_refine), order_(order)
{
    data_.param = param;
    if (data_.param.ml_particular)
    {
        all_bdr_.SetSize(ess_bdr.Size(), 1);
        hdiv_fes_.reset(new ParFiniteElementSpace(mesh, &hdiv_fec_));
        l2_fes_.reset(new ParFiniteElementSpace(mesh, &l2_fec_));
        coarse_hdiv_fes_.reset(new ParFiniteElementSpace(*hdiv_fes_));
        coarse_l2_fes_.reset(new ParFiniteElementSpace(*l2_fes_));
        l2_0_fes_.reset(new FiniteElementSpace(mesh, &l2_0_fec_));
        l2_0_fes_->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);

        data_.agg_el.SetSize(num_refine);
        data_.el_hdivdof.SetSize(num_refine);
        data_.el_l2dof.SetSize(num_refine+1);
        data_.P_hdiv.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
        data_.P_l2.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
        data_.bdr_hdivdofs.SetSize(num_refine);
        data_.el_l2dof[level_] = ElemToTrueDofs(*coarse_l2_fes_);
        hdiv_fes_->GetEssentialTrueDofs(ess_bdr_, data_.coarsest_ess_hdivdofs);
    }

    if (data_.param.MG_type == GeometricMG)
    {
        if (mesh->GetElement(0)->GetType() == Element::TETRAHEDRON && order)
            mesh->ReorientTetMesh();
        hcurl_fes_.reset(new ParFiniteElementSpace(mesh, &hcurl_fec_));
        coarse_hcurl_fes_.reset(new ParFiniteElementSpace(*hcurl_fes_));
        data_.P_curl.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
    }
}

void DivFreeSolverDataCollector::CollectData(ParMesh* mesh)
{
    --level_;

    auto GetP = [this](OperatorPtr& P, ParFiniteElementSpace& fes,
                       unique_ptr<ParFiniteElementSpace>& cfes)
    {
        fes.Update();
        fes.GetTrueTransferOperator(*cfes, P);
        P.As<HypreParMatrix>()->Threshold(1e-16);
        this->level_ ? cfes->Update() : cfes.reset();
    };

    if (data_.param.ml_particular)
    {
        GetP(data_.P_hdiv[level_], *hdiv_fes_, coarse_hdiv_fes_);
        GetP(data_.P_l2[level_], *l2_fes_, coarse_l2_fes_);

        auto& elem_agg_l = (const SparseMatrix&)*l2_0_fes_->GetUpdateOperator();
        OperatorPtr agg_elem_l(Transpose(elem_agg_l));
        data_.agg_el[level_].Swap(*agg_elem_l.As<SparseMatrix>());

        data_.el_hdivdof[level_] = ElemToTrueDofs(*hdiv_fes_);
        data_.el_l2dof[level_] = ElemToTrueDofs(*l2_fes_);

        hdiv_fes_->GetEssentialTrueDofs(all_bdr_, data_.bdr_hdivdofs[level_]);
        AddSharedTDofs(*hdiv_fes_->Dof_TrueDof_Matrix(), data_.bdr_hdivdofs[level_]);
    }

    if (data_.param.MG_type == GeometricMG)
        GetP(data_.P_curl[level_], *hcurl_fes_, coarse_hcurl_fes_);

    if (level_ == 0)
    {
        if (data_.param.MG_type == AlgebraicMG)
        {
            if (mesh->GetElement(0)->GetType() == Element::TETRAHEDRON && order_)
                mesh->ReorientTetMesh();
            hcurl_fes_.reset(new ParFiniteElementSpace(mesh, &hcurl_fec_));
        }

        if (data_.param.ml_particular == false)
        {
            hdiv_fes_.reset(new ParFiniteElementSpace(mesh, &hdiv_fec_));
            l2_fes_.reset(new ParFiniteElementSpace(mesh, &l2_fec_));
        }

        Vector trash1(hcurl_fes_->GetVSize()), trash2(hdiv_fes_->GetVSize());
        ParDiscreteLinearOperator curl(hcurl_fes_.get(), hdiv_fes_.get());
        curl.AddDomainInterpolator(new CurlInterpolator);
        curl.Assemble();
        curl.EliminateTrialDofs(ess_bdr_, trash1, trash2);
        curl.Finalize();
        data_.C.Reset(curl.ParallelAssemble());

        ParBilinearForm mass(l2_fes_.get());
        mass.AddDomainIntegrator(new MassIntegrator());
        mass.Assemble();
        mass.Finalize();
        data_.W.Reset(mass.ParallelAssemble());

        l2_0_fes_.reset();
    }
}

MLDivSolver::MLDivSolver(const HypreParMatrix& M, const HypreParMatrix &B,
                         const OperatorPtr &W, const DivFreeSolverData& data)
    : data_(data), agg_hdivdof_(data.P_l2.Size()), agg_l2dof_(data.P_l2.Size()),
      agg_solver_(data.P_l2.Size()), W_(data.P_l2.Size()+1), coarser_W_inv_(data.P_l2.Size())
{
    const unsigned int num_levels = data.el_l2dof.Size();

    W_[0] = W;
    OperatorPtr B_l(const_cast<HypreParMatrix*>(&B), false);
    OperatorPtr M_l;//(M.NumRows() ? const_cast<HypreParMatrix*>(&M) : NULL, false);

    Array<int> loc_hdivdofs, loc_l2dofs;
    SparseMatrix B_l_diag, M_l_diag;
    DenseMatrix B_a, M_a;

    for (unsigned int l = 0; l < num_levels - 1; ++l)
    {
        const HypreParMatrix& P_l2 = *data.P_l2[l].As<HypreParMatrix>();
        const HypreParMatrix& P_hdiv = *data.P_hdiv[l].As<HypreParMatrix>();

        agg_l2dof_[l].Reset(mfem::Mult(data.agg_el[l], data.el_l2dof[l]));
        agg_hdivdof_[l].Reset(
            AggToInteriorDof(data.agg_el[l], data.el_hdivdof[l], data.bdr_hdivdofs[l]));

        W_[l+1].Reset(TwoStepsRAP(P_l2, *W_[l].As<HypreParMatrix>(), P_l2));

        coarser_W_inv_[l].Reset(new BlockDiagSolver(W_[l+1], data.el_l2dof[l+1]));

        if (M_l.Ptr()) M_l.As<HypreParMatrix>()->GetDiag(M_l_diag);
        B_l.As<HypreParMatrix>()->GetDiag(B_l_diag);

        SparseMatrix& agg_hdivdof_l = *agg_hdivdof_[l].As<SparseMatrix>();
        SparseMatrix& agg_l2dof_l = *agg_l2dof_[l].As<SparseMatrix>();

        agg_solver_[l].SetSize(agg_l2dof_l.NumRows());

        for (int agg = 0; agg < agg_l2dof_l.NumRows(); agg++)
        {
            GetRowColumnsRef(agg_hdivdof_l, agg, loc_hdivdofs);
            GetRowColumnsRef(agg_l2dof_l, agg, loc_l2dofs);
            if (M_l.Ptr()) GetSubMatrix(M_l_diag, loc_hdivdofs, loc_hdivdofs, M_a);
            GetSubMatrix(B_l_diag, loc_l2dofs, loc_hdivdofs, B_a);
            agg_solver_[l][agg].Reset(new LocalSolver(B_a));
        }

        B_l.Reset(TwoStepsRAP(P_l2, *B_l.As<HypreParMatrix>(), P_hdiv), l < num_levels-2);
        if (M_l.Ptr()) M_l.Reset(TwoStepsRAP(P_hdiv, *M_l.As<HypreParMatrix>(), P_hdiv));
    }

    coarsest_B_.Reset(B_l.As<HypreParMatrix>());
    coarsest_B_.As<HypreParMatrix>()->GetDiag(B_l_diag);
    for (int dof : data.coarsest_ess_hdivdofs) B_l_diag.EliminateCol(dof);
    coarsest_solver_.Reset(new BBTSolver(*coarsest_B_.As<HypreParMatrix>()));
}

void MLDivSolver::Mult(const Vector & x, Vector & y) const
{
    y.SetSize(agg_hdivdof_[0]->NumCols());

    Array<Vector> sigma(W_.Size());
    sigma[0].SetDataAndSize(y.GetData(), y.Size());

    Array<int> loc_hdivdofs, loc_l2dofs;
    Vector F_l, PT_F_l, F_coarse, PF_coarse, F_a, sigma_a;

    for (unsigned int l = 0; l < W_.Size() - 1; ++l)
    {
        sigma[l].SetSize(agg_hdivdof_[l]->NumCols());
        sigma[l] = 0.0;

        // Right hand side: F_l = F - W_l P_l2[l] (W_{l+1})^{-1} P_l2[l]^T F
        F_l = l == 0 ? x : PT_F_l;
        PT_F_l.SetSize(W_[l+1]->NumRows());
        data_.P_l2[l]->MultTranspose(F_l, PT_F_l);
        coarser_W_inv_[l]->Mult(PT_F_l, F_coarse);
        PF_coarse.SetSize(W_[l]->NumRows());
        data_.P_l2[l]->Mult(F_coarse, PF_coarse);
        W_[l].As<HypreParMatrix>()->Mult(-1.0, PF_coarse, 1.0, F_l);

        SparseMatrix& agg_hdivdof_l = *agg_hdivdof_[l].As<SparseMatrix>();
        SparseMatrix& agg_l2dof_l = *agg_l2dof_[l].As<SparseMatrix>();

        for (int agg = 0; agg < agg_hdivdof_l.NumRows(); agg++)
        {
            GetRowColumnsRef(agg_hdivdof_l, agg, loc_hdivdofs);
            GetRowColumnsRef(agg_l2dof_l, agg, loc_l2dofs);
            F_l.GetSubVector(loc_l2dofs, F_a);
            agg_solver_[l][agg]->Mult(F_a, sigma_a);
            sigma[l].AddElementVector(loc_hdivdofs, sigma_a);
        }
    }

    Vector u_c(coarsest_B_->NumRows());
    coarsest_solver_->Mult(PT_F_l, u_c);
    sigma.Last().SetSize(coarsest_B_->NumCols());
    coarsest_B_->MultTranspose(u_c, sigma.Last());

    for (int l = W_.Size()-2; l>=0; l--) // add solutions from all levels
    {
        Vector P_sigma(data_.P_hdiv[l]->NumRows());
        data_.P_hdiv[l]->Mult(sigma[l+1], P_sigma);
        sigma[l] += P_sigma;
    }
}

DivFreeSolver::DivFreeSolver(const HypreParMatrix &M, const HypreParMatrix& B,
                             ParFiniteElementSpace* hcurl_fes,
                             const DivFreeSolverData& data)
    : Solver(M.NumRows()+B.NumRows()), M_(M), B_(B),
      BBT_solver_(B, data.param.B_has_nullity_one, data.param.BBT_solve_param),
      CTMC_solver_(B_.GetComm()),
      offsets_(3), data_(data)
{
    offsets_[0] = 0;
    offsets_[1] = M.NumCols();
    offsets_[2] = offsets_[1] + B.NumRows();

    if (data.param.ml_particular)
        particular_solver_.Reset(new MLDivSolver(M, B, data.W, data));

    OperatorPtr MC(ParMult(&M_, data.C.As<HypreParMatrix>()));
    OperatorPtr CT(data.C.As<HypreParMatrix>()->Transpose());
    CTMC_.Reset(ParMult(CT.As<HypreParMatrix>(), MC.As<HypreParMatrix>()));
    CTMC_.As<HypreParMatrix>()->CopyRowStarts();
    CTMC_.As<HypreParMatrix>()->EliminateZeroRows();
    CTMC_.As<HypreParMatrix>()->Threshold(1e-14);
    CTMC_solver_.SetOperator(*CTMC_);

    if (data_.param.MG_type == AlgebraicMG)
    {
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
    if (data_.param.ml_particular) { particular_solver_->Mult(rhs, sol); return; }

    Vector potential(rhs.Size());
    BBT_solver_.Mult(rhs, potential);
    B_.MultTranspose(potential, sol);
}

void DivFreeSolver::SolveDivFree(const Vector &rhs, Vector& sol) const
{
    Vector rhs_divfree(CTMC_->NumRows());
    data_.C->MultTranspose(rhs, rhs_divfree);

    Vector potential_divfree(CTMC_->NumRows());
    CTMC_solver_.Mult(rhs_divfree, potential_divfree);
    PrintConvergence(CTMC_solver_, data_.param.verbose);

    data_.C->Mult(potential_divfree, sol);
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
        auto P = const_cast<OperatorPtr&>(P_[l-1]).As<HypreParMatrix>();
        ops_[l].Reset(TwoStepsRAP(*P, *ops_[l-1].As<HypreParMatrix>(), *P));
        ops_[l].As<HypreParMatrix>()->Threshold(1e-14);
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

