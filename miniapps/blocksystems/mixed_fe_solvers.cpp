
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

HypreParMatrix* Mult(const OperatorPtr& A, const OperatorPtr& B, const OperatorPtr& C)
{
    OperatorPtr AB(ParMult(A.As<HypreParMatrix>(), B.As<HypreParMatrix>()));
    auto* ABC = ParMult(AB.As<HypreParMatrix>(), C.As<HypreParMatrix>());
    ABC->CopyRowStarts();
    ABC->CopyColStarts();
    return ABC;
}

HypreParMatrix* Mult(const SparseMatrix& A, const SparseMatrix& B,
                     const HypreParMatrix& C, Array<int>& row_starts)
{
    OperatorPtr AB(Mult(A, B));
    return C.LeftDiagMult(*AB.As<SparseMatrix>(), row_starts);
}


HypreParMatrix* TwoStepsRAP(const OperatorPtr& Rt, const OperatorPtr& A,
                            const OperatorPtr& P)
{
    return Mult(OperatorPtr(Rt.As<HypreParMatrix>()->Transpose()), A, P);
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
    PrintConvergence(BBT_solver_, false);
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

BlockDiagSolver::BlockDiagSolver(const OperatorPtr &A, SparseMatrix block_dof)
    : Solver(A->NumRows()), block_dof_(std::move(block_dof)),
      block_solver_(block_dof.NumRows())
{
    SparseMatrix A_diag;
    A.As<HypreParMatrix>()->GetDiag(A_diag);
    DenseMatrix sub_A;
    for(int block = 0; block < block_dof.NumRows(); block++)
    {
        GetRowColumnsRef(block_dof_, block, local_dofs_);
        GetSubMatrix(A_diag, local_dofs_, local_dofs_, sub_A);
        block_solver_[block].SetOperator(sub_A);
    }
}

void BlockDiagSolver::Mult(const Vector &x, Vector &y) const
{
    y.SetSize(x.Size());
    y = 0.0;

    for(int block = 0; block < block_dof_.NumRows(); block++)
    {
        GetRowColumnsRef(block_dof_, block, local_dofs_);
        x.GetSubVector(local_dofs_, sub_rhs_);
        sub_sol_.SetSize(local_dofs_.Size());
        block_solver_[block].Mult(sub_rhs_, sub_sol_);
        y.AddElementVector(local_dofs_, sub_sol_);
    }
}

SparseMatrix ElemToDof(const ParFiniteElementSpace& fes)
{
    int* I = new int[fes.GetNE()+1];
    copy_n(fes.GetElementToDofTable().GetI(), fes.GetNE()+1, I);
    Array<int> J(new int[I[fes.GetNE()]], I[fes.GetNE()]);
    copy_n(fes.GetElementToDofTable().GetJ(), J.Size(), J.begin());
    fes.AdjustVDofs(J);
    double* D = new double[J.Size()];
    fill_n(D, J.Size(), 1.0);
    return SparseMatrix(I, J, D, fes.GetNE(), fes.GetVSize());
}

DFSDataCollector::
DFSDataCollector(int order, int num_refine, ParMesh *mesh,
                 const Array<int>& ess_attr, const DFSParameters& param)
    : hdiv_fec_(order, mesh->Dimension()), l2_fec_(order, mesh->Dimension()),
      hcurl_fec_(order+1, mesh->Dimension()), l2_0_fec_(0, mesh->Dimension()),
      ess_bdr_attr_(ess_attr), level_(num_refine), order_(order)
{
    data_.param = param;
    if (data_.param.ml_particular)
    {
        all_bdr_attr_.SetSize(ess_attr.Size(), 1);
        hdiv_fes_.reset(new ParFiniteElementSpace(mesh, &hdiv_fec_));
        l2_fes_.reset(new ParFiniteElementSpace(mesh, &l2_fec_));
        coarse_hdiv_fes_.reset(new ParFiniteElementSpace(*hdiv_fes_));
        coarse_l2_fes_.reset(new ParFiniteElementSpace(*l2_fes_));
        l2_0_fes_.reset(new ParFiniteElementSpace(mesh, &l2_0_fec_));
        l2_0_fes_->SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
        el_l2dof_.SetSize(num_refine+1);
        el_l2dof_[level_] = ElemToDof(*coarse_l2_fes_);

        data_.agg_hdivdof.SetSize(num_refine);
        data_.agg_l2dof.SetSize(num_refine);
        data_.P_hdiv.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
        data_.P_l2.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
        data_.Q_l2.SetSize(num_refine);
        hdiv_fes_->GetEssentialTrueDofs(ess_attr, data_.coarsest_ess_hdivdofs);
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

SparseMatrix* AggToInteriorDof(const Array<int>& bdr_truedofs,
                               const SparseMatrix& agg_elem,
                               const SparseMatrix& elem_dof,
                               const HypreParMatrix& dof_truedof,
                               Array<int>& agg_starts)
{
    OperatorPtr agg_tdof(Mult(agg_elem, elem_dof, dof_truedof, agg_starts));
    OperatorPtr agg_tdof_T(agg_tdof.As<HypreParMatrix>()->Transpose());
    SparseMatrix tdof_agg, is_shared;
    HYPRE_Int* trash;
    agg_tdof_T.As<HypreParMatrix>()->GetDiag(tdof_agg);
    agg_tdof_T.As<HypreParMatrix>()->GetOffd(is_shared, trash);

    int * I = new int [tdof_agg.NumRows()+1]();
    int * J = new int[tdof_agg.NumNonZeroElems()];

    Array<int> is_bdr;
    FiniteElementSpace::ListToMarker(bdr_truedofs, tdof_agg.NumRows(), is_bdr);

    int counter = 0;
    for (int i = 0; i < tdof_agg.NumRows(); ++i)
    {
        bool agg_bdr = is_bdr[i] || is_shared.RowSize(i) || tdof_agg.RowSize(i)>1;
        if (agg_bdr) { I[i+1] = I[i]; continue; }
        I[i+1] = I[i] + 1;
        J[counter++] = tdof_agg.GetRowColumns(i)[0];
    }

    double * D = new double[I[tdof_agg.NumRows()]];
    std::fill_n(D, I[tdof_agg.NumRows()], 1.0);

    SparseMatrix intdof_agg(I, J, D, tdof_agg.NumRows(), tdof_agg.NumCols());
    return Transpose(intdof_agg);
}

void DFSDataCollector::MakeDofRelationTables(int level)
{
    Array<int> agg_starts(Array<int>(l2_0_fes_->GetDofOffsets(), 2));
    auto& elem_agg = (const SparseMatrix&)*l2_0_fes_->GetUpdateOperator();
    OperatorPtr agg_elem(Transpose(elem_agg));
    SparseMatrix& agg_el = *agg_elem.As<SparseMatrix>();

    el_l2dof_[level] = ElemToDof(*l2_fes_);
    data_.agg_l2dof[level].Reset(Mult(agg_el, el_l2dof_[level]));

    Array<int> bdr_tdofs;
    hdiv_fes_->GetEssentialTrueDofs(all_bdr_attr_, bdr_tdofs);
    auto tmp = AggToInteriorDof(bdr_tdofs, agg_el, ElemToDof(*hdiv_fes_),
                                *hdiv_fes_->Dof_TrueDof_Matrix(), agg_starts);
    data_.agg_hdivdof[level].Reset(tmp);
}

void DFSDataCollector::DataFinalize(ParMesh* mesh)
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
    curl.EliminateTrialDofs(ess_bdr_attr_, trash1, trash2);
    curl.Finalize();
    data_.C.Reset(curl.ParallelAssemble());

    ParBilinearForm mass(l2_fes_.get());
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();
    OperatorPtr W(mass.ParallelAssemble());

    for (int l = 0; l < data_.P_l2.Size(); ++l)
    {
        auto WP = ParMult(W.As<HypreParMatrix>(), data_.P_l2[l].As<HypreParMatrix>());
        WP->CopyRowStarts();
        OperatorPtr PT_l2(data_.P_l2[l].As<HypreParMatrix>()->Transpose());
        W.Reset(ParMult(PT_l2.As<HypreParMatrix>(), WP));
        W.As<HypreParMatrix>()->CopyRowStarts();
        auto cW_inv = new BlockDiagSolver(W, move(el_l2dof_[l+1]));
        data_.Q_l2[l].Reset(new ProductOperator(WP, cW_inv, true, true));
    }

    el_l2dof_.DeleteAll();
    l2_0_fes_.reset();
}

void DFSDataCollector::CollectData(ParMesh* mesh)
{
    --level_;

    auto GetP = [this](OperatorPtr& P, unique_ptr<ParFiniteElementSpace>& cfes,
                       ParFiniteElementSpace& fes, bool remove_zero)
    {
        fes.Update();
        fes.GetTrueTransferOperator(*cfes, P);
        if (remove_zero) P.As<HypreParMatrix>()->Threshold(1e-16);
        this->level_ ? cfes->Update() : cfes.reset();
    };

    if (data_.param.ml_particular)
    {
        GetP(data_.P_hdiv[level_], coarse_hdiv_fes_, *hdiv_fes_, true);
        GetP(data_.P_l2[level_], coarse_l2_fes_, *l2_fes_, false);
        MakeDofRelationTables(level_);
    }

    if (data_.param.MG_type == GeometricMG)
        GetP(data_.P_curl[level_], coarse_hcurl_fes_, *hcurl_fes_, true);

    if (level_ == 0) DataFinalize(mesh);
}

MLDivSolver::MLDivSolver(const HypreParMatrix& M, const HypreParMatrix &B, const DFSData& data)
    : data_(data), agg_solver_(data.P_l2.Size())
{
    const unsigned int num_levels = agg_solver_.Size()+1;

    OperatorPtr B_l(const_cast<HypreParMatrix*>(&B), false);
    OperatorPtr M_l;//(M.NumRows() ? const_cast<HypreParMatrix*>(&M) : NULL, false);

    Array<int> loc_hdivdofs, loc_l2dofs;
    SparseMatrix B_l_diag, M_l_diag;
    DenseMatrix B_a, M_a;

    for (unsigned int l = 0; l < num_levels-1; ++l)
    {
        if (M_l.Ptr()) M_l.As<HypreParMatrix>()->GetDiag(M_l_diag);
        B_l.As<HypreParMatrix>()->GetDiag(B_l_diag);

        SparseMatrix& agg_hdivdof_l = *data_.agg_hdivdof[l].As<SparseMatrix>();
        SparseMatrix& agg_l2dof_l = *data_.agg_l2dof[l].As<SparseMatrix>();

        agg_solver_[l].SetSize(agg_l2dof_l.NumRows());
        for (int agg = 0; agg < agg_l2dof_l.NumRows(); agg++)
        {
            GetRowColumnsRef(agg_hdivdof_l, agg, loc_hdivdofs);
            GetRowColumnsRef(agg_l2dof_l, agg, loc_l2dofs);
            if (M_l.Ptr()) GetSubMatrix(M_l_diag, loc_hdivdofs, loc_hdivdofs, M_a);
            GetSubMatrix(B_l_diag, loc_l2dofs, loc_hdivdofs, B_a);
            agg_solver_[l][agg].Reset(new LocalSolver(B_a));
        }

        B_l.Reset(TwoStepsRAP(data.P_l2[l], B_l, data.P_hdiv[l]), l < num_levels-2);
        if (M_l.Ptr()) M_l.Reset(TwoStepsRAP(data.P_hdiv[l], M_l, data.P_hdiv[l]));
    }

    coarsest_B_.Reset(B_l.As<HypreParMatrix>());
    coarsest_B_.As<HypreParMatrix>()->GetDiag(B_l_diag);
    for (int dof : data.coarsest_ess_hdivdofs) B_l_diag.EliminateCol(dof);
    coarsest_solver_.Reset(new BBTSolver(*coarsest_B_.As<HypreParMatrix>()));
}

void MLDivSolver::Mult(const Vector & x, Vector & y) const
{
    y.SetSize(data_.agg_hdivdof[0]->NumCols());

    Array<Vector> sigma(agg_solver_.Size()+1);
    sigma[0].SetDataAndSize(y.GetData(), y.Size());

    Array<int> loc_hdivdofs, loc_l2dofs;
    Vector F_l, PT_F_l, Pi_F_l, F_a, sigma_a;

    for (int l = 0; l < agg_solver_.Size(); ++l)
    {
        sigma[l].SetSize(data_.agg_hdivdof[l]->NumCols());
        sigma[l] = 0.0;

        // Right hand side: F_l = F - W_l P_l2[l] (W_{l+1})^{-1} P_l2[l]^T F
        F_l = l == 0 ? x : PT_F_l;
        PT_F_l.SetSize(data_.P_l2[l]->NumCols());
        data_.P_l2[l]->MultTranspose(F_l, PT_F_l);
        Pi_F_l.SetSize(data_.P_l2[l]->NumRows());
        data_.Q_l2[l]->Mult(PT_F_l, Pi_F_l);
        F_l -= Pi_F_l;

        SparseMatrix& agg_hdivdof_l = *data_.agg_hdivdof[l].As<SparseMatrix>();
        SparseMatrix& agg_l2dof_l = *data_.agg_l2dof[l].As<SparseMatrix>();

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

    for (int l = agg_solver_.Size()-1; l>=0; l--)
        data_.P_hdiv[l].As<HypreParMatrix>()->Mult(1., sigma[l+1], 1., sigma[l]);
}

DivFreeSolver::DivFreeSolver(const HypreParMatrix &M, const HypreParMatrix& B,
                             ParFiniteElementSpace* hcurl_fes, const DFSData& data)
    : DarcySolver(M.NumRows(), B.NumRows()), M_(M), B_(B),
      BBT_solver_(B, data.param.B_has_nullity_one, data.param.BBT_solve_param),
      CTMC_solver_(B_.GetComm()), data_(data)
{
    if (data.param.ml_particular)
        particular_solver_.Reset(new MLDivSolver(M, B, data));

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
    BlockVector blk_x(x.GetData(), offsets_);
    Vector x_blk0_copy(blk_x.GetBlock(0));
    BlockVector blk_y(y.GetData(), offsets_);

    Vector particular_flux(blk_y.BlockSize(0));
    SolveParticular(blk_x.GetBlock(1), particular_flux);
    blk_y.GetBlock(0) += particular_flux;

    if (data_.param.verbose)
        cout << "Particular solution found in " << ch.RealTime() << "s.\n";

    ch.Clear();
    ch.Start();

    Vector divfree_flux(blk_y.BlockSize(0));
    M_.Mult(-1.0, particular_flux, 1.0, x_blk0_copy);
    SolveDivFree(x_blk0_copy, divfree_flux);
    blk_y.GetBlock(0) += divfree_flux;

    if (data_.param.verbose)
        cout << "Divergence free solution found in " << ch.RealTime() << "s.\n";

    ch.Clear();
    ch.Start();

    M_.Mult(-1.0, divfree_flux, 1.0, x_blk0_copy);
    SolvePotential(x_blk0_copy, blk_y.GetBlock(1));

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
        ops_[l].Reset(TwoStepsRAP(P_[l-1], ops_[l-1], P_[l-1]));
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

BDPMinresSolver::BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                                 IterSolveParameters param)
    : DarcySolver(M.NumRows(), B.NumRows()), op_(offsets_), prec_(offsets_),
      BT_(B.Transpose()), solver_(M.GetComm())
{
    op_.SetBlock(0,0, &M);
    op_.SetBlock(0,1, BT_.As<HypreParMatrix>());
    op_.SetBlock(1,0, &B);

    Vector Md;
    M.GetDiag(Md);
    BT_.As<HypreParMatrix>()->InvScaleRows(Md);
    S_.Reset(ParMult(&B, BT_.As<HypreParMatrix>()));
    BT_.As<HypreParMatrix>()->ScaleRows(Md);

    prec_.SetDiagonalBlock(0, new HypreDiagScale(M));
    prec_.SetDiagonalBlock(1, new HypreBoomerAMG(*S_.As<HypreParMatrix>()));
    static_cast<HypreBoomerAMG&>(prec_.GetDiagonalBlock(1)).SetPrintLevel(0);
    prec_.owns_blocks = true;

    SetOptions(solver_, param);
    solver_.SetOperator(op_);
    solver_.SetPreconditioner(prec_);
}

void BDPMinresSolver::Mult(const Vector & x, Vector & y) const
{
    solver_.Mult(x, y);
    PrintConvergence(solver_, false);
}

