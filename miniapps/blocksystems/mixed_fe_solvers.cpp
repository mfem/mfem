
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

HypreParMatrix* Mult(const HypreParMatrix& A, const HypreParMatrix& B, const HypreParMatrix& C)
{
    OperatorPtr AB(ParMult(&A, &B));
    auto* ABC = ParMult(AB.As<HypreParMatrix>(), &C);
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


HypreParMatrix* TwoStepsRAP(const HypreParMatrix& Rt, const HypreParMatrix& A,
                            const HypreParMatrix& P)
{
    OperatorPtr R(Rt.Transpose());
    return Mult(*(R.As<HypreParMatrix>()), A, P);
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
    : Solver(B.NumCols()), BT_(B, 't'), local_system_(B.NumRows()), offset_(0)
{
    mfem::Mult(B, BT_, local_system_);
    local_system_.SetRow(0, 0.0);
    local_system_.SetCol(0, 0.0);
    local_system_(0, 0) = 1.;
    local_solver_.SetOperator(local_system_);
}

LocalSolver::LocalSolver(const DenseMatrix& M, const DenseMatrix& B)
    : Solver(M.NumRows()+B.NumRows()), local_system_(height), offset_(M.NumRows())
{
    local_system_.CopyMN(M, 0, 0);
    local_system_.CopyMN(B, offset_, 0);
    local_system_.CopyMNt(B, 0, offset_);

    local_system_.SetRow(offset_, 0.0);
    local_system_.SetCol(offset_, 0.0);
    local_system_(offset_, offset_) = -1.0;
    local_solver_.SetOperator(local_system_);
}

void LocalSolver::Mult(const Vector &x, Vector &y) const
{
    if (x.Size() == BT_.NumCols())
    {
        double x0 = x[0];
        const_cast<Vector&>(x)[0] = 0.0;

        Vector u(BT_.NumCols());
        local_solver_.Mult(x, u);

        y.SetSize(BT_.NumRows());
        BT_.Mult(u, y);
        const_cast<Vector&>(x)[0] = x0;
    }
    else
    {
//        Vector rhs(local_system_.NumRows());
//        for (int i = 0; i < offset_+1; i++)
//        {
//            rhs[i] = 0.0;
//        }
//        for (int i = offset_+1; i < local_system_.NumRows(); i++)
//        {
//            rhs[i] = x[i - offset_];
//        }

//        Vector sol(local_system_.NumRows());
//        local_solver_.Mult(rhs, sol);

//        y.SetSize(offset_);
//        for (int i = 0; i < offset_; i++)
//        {
//            y[i] = sol[i];
//        }

        double x0 = x[offset_];
        const_cast<Vector&>(x)[offset_] = 0.0;

        y.SetSize(local_system_.NumRows());
        local_solver_.Mult(x, y);

        const_cast<Vector&>(x)[offset_] = x0;
    }
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
        data_.C.SetSize(num_refine+1);
    }

    if (data_.param.MG_type == GeometricMG)
    {
        if (mesh->GetElement(0)->GetType() == Element::TETRAHEDRON && order)
            mesh->ReorientTetMesh();
        hcurl_fes_.reset(new ParFiniteElementSpace(mesh, &hcurl_fec_));
        coarse_hcurl_fes_.reset(new ParFiniteElementSpace(*hcurl_fes_));
        data_.P_hcurl.SetSize(num_refine, OperatorPtr(Operator::Hypre_ParCSR));
    }

    Vector trash1(hcurl_fes_->GetVSize()), trash2(hdiv_fes_->GetVSize());
    ParDiscreteLinearOperator curl(hcurl_fes_.get(), hdiv_fes_.get());
    curl.AddDomainInterpolator(new CurlInterpolator);
    curl.Assemble();
    curl.EliminateTrialDofs(ess_bdr_attr_, trash1, trash2);
    curl.Finalize();
    data_.C[level_].Reset(curl.ParallelAssemble());
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
        //TODO: maybe store the product is better if it needs to be applied in iterations
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
        GetP(data_.P_hcurl[level_], coarse_hcurl_fes_, *hcurl_fes_, true);

    Vector trash1(hcurl_fes_->GetVSize()), trash2(hdiv_fes_->GetVSize());
    ParDiscreteLinearOperator curl(hcurl_fes_.get(), hdiv_fes_.get());
    curl.AddDomainInterpolator(new CurlInterpolator);
    curl.Assemble();
    curl.EliminateTrialDofs(ess_bdr_attr_, trash1, trash2);
    curl.Finalize();
    data_.C[level_].Reset(curl.ParallelAssemble());

    if (level_ == 0) DataFinalize(mesh);
}

MLDivSolver::MLDivSolver(const HypreParMatrix& M, const HypreParMatrix &B, const DFSData& data)
    : data_(data), agg_solver_(data.P_l2.Size())
{
    const unsigned int num_levels = agg_solver_.Size()+1;

    OperatorPtr B_l(const_cast<HypreParMatrix*>(&B), false);
    OperatorPtr M_l(M.NumRows() ? const_cast<HypreParMatrix*>(&M) : NULL, false);

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
            if (M_l.Ptr()) { GetSubMatrix(M_l_diag, loc_hdivdofs, loc_hdivdofs, M_a); }
            GetSubMatrix(B_l_diag, loc_l2dofs, loc_hdivdofs, B_a);
//            agg_solver_[l][agg].Reset(new LocalSolver(B_a));

            agg_solver_[l][agg].Reset(new LocalSolver(M_a, B_a));
        }

        HypreParMatrix& P_hdiv_l = *data.P_hdiv[l].As<HypreParMatrix>();
        HypreParMatrix& P_l2_l = *data.P_l2[l].As<HypreParMatrix>();
        HypreParMatrix& B_l_ref = *B_l.As<HypreParMatrix>();

        B_l.Reset(TwoStepsRAP(P_l2_l, B_l_ref, P_hdiv_l), l < num_levels-2);
        if (M_l.Ptr())
        {
            HypreParMatrix& M_l_ref = *M_l.As<HypreParMatrix>();
            M_l.Reset(TwoStepsRAP(P_hdiv_l, M_l_ref, P_hdiv_l), l < num_levels-2);
        }
    }

    if (M_l.Ptr())
    {
        coarsest_M_.Reset(M_l.As<HypreParMatrix>());
        coarsest_M_.As<HypreParMatrix>()->GetDiag(M_l_diag);
    }
    coarsest_B_.Reset(B_l.As<HypreParMatrix>());
    coarsest_B_.As<HypreParMatrix>()->GetDiag(B_l_diag);
    for (int dof : data.coarsest_ess_hdivdofs)
    {
        B_l_diag.EliminateCol(dof);
        if (M_l.Ptr()) { M_l_diag.EliminateRowCol(dof); }
    }
//    coarsest_solver_.Reset(new BBTSolver(*coarsest_B_.As<HypreParMatrix>()));
    coarsest_solver_.Reset(new BDPMinresSolver(*coarsest_M_.As<HypreParMatrix>(),
                                               *coarsest_B_.As<HypreParMatrix>(),
                                               data_.param.BBT_solve_param));
}

MLDivSolver::MLDivSolver(const Array<OperatorPtr>& M,
                         const Array<OperatorPtr>& B,
                         const DFSData& data)
    : data_(data), agg_solver_(data.P_l2.Size())
{
    const unsigned int num_levels = agg_solver_.Size()+1;

    Array<int> loc_hdivdofs, loc_l2dofs;
    SparseMatrix B_l_diag, M_l_diag;
    DenseMatrix B_a, M_a;

    for (unsigned int l = 0; l < num_levels-1; ++l)
    {
        M[l].As<HypreParMatrix>()->GetDiag(M_l_diag);
        B[l].As<HypreParMatrix>()->GetDiag(B_l_diag);

        SparseMatrix& agg_hdivdof_l = *data_.agg_hdivdof[l].As<SparseMatrix>();
        SparseMatrix& agg_l2dof_l = *data_.agg_l2dof[l].As<SparseMatrix>();

        agg_solver_[l].SetSize(agg_l2dof_l.NumRows());
        for (int agg = 0; agg < agg_l2dof_l.NumRows(); agg++)
        {
            GetRowColumnsRef(agg_hdivdof_l, agg, loc_hdivdofs);
            GetRowColumnsRef(agg_l2dof_l, agg, loc_l2dofs);
            GetSubMatrix(M_l_diag, loc_hdivdofs, loc_hdivdofs, M_a);
            GetSubMatrix(B_l_diag, loc_l2dofs, loc_hdivdofs, B_a);
            agg_solver_[l][agg].Reset(new LocalSolver(M_a, B_a));
        }
    }
}

void MLDivSolver::Mult(const Vector & x, Vector & y) const
{
    y.SetSize(data_.agg_hdivdof[0]->NumCols());

    Array<Vector> sigma(agg_solver_.Size()+1);
    sigma[0].SetDataAndSize(y.GetData(), y.Size());

//    Array<Vector> sigma(agg_solver_.Size()+1);
//    sigma[0].SetSize(data_.P_hdiv[0]->NumRows());
//    sigma[0] = 0.0;

    Array<Vector> u(agg_solver_.Size()+1);
//    u[0].SetSize(data_.P_l2[0]->NumRows());
//    u[0] = 0.0;


    Array<int> loc_hdivdofs, loc_l2dofs;
    Vector F_l, PT_F_l, Pi_F_l, F_a, rhs_a, sol_a;

    for (int l = 0; l < agg_solver_.Size(); ++l)
    {
        sigma[l].SetSize(data_.agg_hdivdof[l]->NumCols());
        sigma[l] = 0.0;

        u[l].SetSize(data_.agg_l2dof[l]->NumCols());
        u[l] = 0.0;

        // Right hand side: F_l = F - W_l P_l2[l] (W_{l+1})^{-1} P_l2[l]^T F
        // TODO modularize this few lines
        F_l = l == 0 ? x : PT_F_l;
        PT_F_l.SetSize(data_.P_l2[l]->NumCols());
        data_.P_l2[l]->MultTranspose(F_l, PT_F_l);
        Pi_F_l.SetSize(data_.P_l2[l]->NumRows());
        data_.Q_l2[l]->Mult(PT_F_l, Pi_F_l);
        F_l -= Pi_F_l;

        auto& agg_hdivdof_l = *data_.agg_hdivdof[l].As<SparseMatrix>();
        auto& agg_l2dof_l = *data_.agg_l2dof[l].As<SparseMatrix>();

        for (int agg = 0; agg < agg_hdivdof_l.NumRows(); agg++)
        {
            GetRowColumnsRef(agg_hdivdof_l, agg, loc_hdivdofs);
            GetRowColumnsRef(agg_l2dof_l, agg, loc_l2dofs);
            F_l.GetSubVector(loc_l2dofs, F_a);

            rhs_a.SetSize(loc_hdivdofs.Size()+loc_l2dofs.Size());
            for (int i = 0; i < loc_hdivdofs.Size(); ++i)
            {
                rhs_a[i] = 0.0;
            }
            for (int i = 0; i < loc_l2dofs.Size(); ++i)
            {
                rhs_a[loc_hdivdofs.Size()+i] = F_a[i];
            }

//            agg_solver_[l][agg]->Mult(F_a, sigma_a);
//            sigma[l].AddElementVector(loc_hdivdofs, sigma_a);

            agg_solver_[l][agg]->Mult(rhs_a, sol_a);
            for (int i = 0; i < loc_hdivdofs.Size(); ++i)
            {
                sigma[l][loc_hdivdofs[i]] += sol_a[i];
            }
            for (int i = 0; i < loc_l2dofs.Size(); ++i)
            {
                u[l][loc_l2dofs[i]] += sol_a[loc_hdivdofs.Size()+i];
            }
        }
    }

//    Vector u_c(coarsest_B_->NumRows());
//    coarsest_solver_->Mult(PT_F_l, u_c);
//    sigma.Last().SetSize(coarsest_B_->NumCols());
//    coarsest_B_->MultTranspose(u_c, sigma.Last());

    Vector sol_c(coarsest_M_->NumRows()+coarsest_B_->NumRows());
    // TODO: nonzero first block if MG
    Vector rhs_c(coarsest_M_->NumRows()+coarsest_B_->NumRows());
    rhs_c = 0.0;
    for (int i = 0; i < PT_F_l.Size(); ++i)
    {
        rhs_c[coarsest_M_->NumRows()+i] = PT_F_l[i];
    }
    coarsest_solver_->Mult(rhs_c, sol_c);
    sigma.Last().SetSize(coarsest_B_->NumCols());
    u.Last().SetSize(coarsest_B_->NumRows());
    for (int i = 0; i < coarsest_M_->NumRows(); ++i)
    {
        sigma.Last()[i] = sol_c[i];
    }
    for (int i = 0; i < PT_F_l.Size(); ++i)
    {
        u.Last()[i] = sol_c[coarsest_M_->NumRows()+i];
    }


    for (int l = agg_solver_.Size()-1; l>=0; l--)
    {
        data_.P_hdiv[l].As<HypreParMatrix>()->Mult(1., sigma[l+1], 1., sigma[l]);
        data_.P_l2[l].As<HypreParMatrix>()->Mult(1., u[l+1], 1., u[l]);
    }
}

void MLDivSolver::Mult(int l, const Vector & x, Vector & y) const
{
    MFEM_ASSERT(l < agg_solver_.Size(), "level l should be less than number of levels!");

    Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = data_.agg_hdivdof[l]->NumCols();
    offsets[2] = offsets[1] + data_.agg_l2dof[l]->NumCols();

    y.SetSize(offsets[2]);
    y = 0.0;

    BlockVector blk_y(y.GetData(), offsets);
    BlockVector Pi_x(offsets); // aggregate-wise average free projection of x
    static_cast<Vector&>(Pi_x) = x;

    Array<int> hdivdofs_a, l2dofs_a;

    auto& agg_hdivdof_l = *data_.agg_hdivdof[l].As<SparseMatrix>();
    auto& agg_l2dof_l = *data_.agg_l2dof[l].As<SparseMatrix>();

    Vector coarse_vec(data_.P_l2[l]->NumCols());
    data_.P_l2[l]->MultTranspose(Pi_x.GetBlock(1), coarse_vec);

    Vector agg_average(Pi_x.BlockSize(1));
    data_.Q_l2[l]->Mult(coarse_vec, agg_average);
    Pi_x.GetBlock(1) -= agg_average;

    for (int agg = 0; agg < agg_hdivdof_l.NumRows(); agg++)
    {
        GetRowColumnsRef(agg_hdivdof_l, agg, hdivdofs_a);
        GetRowColumnsRef(agg_l2dof_l, agg, l2dofs_a);

        Array<int> offsets_a(3);
        offsets_a[0] = 0;
        offsets_a[1] = hdivdofs_a.Size();
        offsets_a[2] = offsets_a[1]+l2dofs_a.Size();

        BlockVector rhs_a(offsets_a), sol_a(offsets_a);
        Pi_x.GetBlock(0).GetSubVector(hdivdofs_a, rhs_a.GetBlock(0));
        Pi_x.GetBlock(1).GetSubVector(l2dofs_a, rhs_a.GetBlock(1));

        agg_solver_[l][agg]->Mult(rhs_a, sol_a);

        blk_y.GetBlock(0).AddElementVector(hdivdofs_a, sol_a.GetBlock(0));
        blk_y.GetBlock(1).AddElementVector(l2dofs_a, sol_a.GetBlock(1));
    }

    data_.Q_l2[l].As<ProductOperator>()->MultTranspose(blk_y.GetBlock(1), coarse_vec);
    data_.P_l2[l]->Mult(coarse_vec, agg_average);
    blk_y.GetBlock(1) -= agg_average;
}

HiptmairSmoother::HiptmairSmoother(BlockOperator& op,
                                   MLDivSolver& ml_div_solver,
                                   int level, HypreParMatrix& C)
    : Solver(op.NumRows()), level_(level), op_(op), ml_div_solver_(ml_div_solver)
{
    C_col_offsets_.SetSize(2);
    C_col_offsets_[0] = 0;
    C_col_offsets_[1] = C.NumCols();

    blk_C_.Reset(new BlockOperator(op.ColOffsets(), C_col_offsets_));
    blk_C_.As<BlockOperator>()->SetBlock(0, 0, &C);

    HypreParMatrix& M = dynamic_cast<HypreParMatrix&>(op.GetBlock(0, 0));
    div_kernel_system_.Reset(TwoStepsRAP(C, M, C));
    div_kernel_system_.As<HypreParMatrix>()->EliminateZeroRows();
    div_kernel_system_.As<HypreParMatrix>()->Threshold(1e-14);
    div_kernel_smoother_.Reset(new HypreSmoother(*div_kernel_system_.As<HypreParMatrix>()));
}

void HiptmairSmoother::MultTranspose(const Vector & x, Vector & y) const
{
    ml_div_solver_.Mult(level_, x, y);
    Vector resid = x;
    op_.Mult(y, resid);
    add(-1.0, resid, 1.0, x, resid);    // resid = x - A(y)

    Vector CT_r(blk_C_->NumCols());
    CT_r = 0.0;
    Vector div_kern_sol(CT_r);
    div_kern_sol = 0.0;

    blk_C_.As<BlockOperator>()->MultTranspose(resid, CT_r);
    div_kernel_smoother_->Mult(CT_r, div_kern_sol);

    Vector div_kernel_correction(blk_C_->NumRows());
    div_kernel_correction = 0.0;
    blk_C_.As<BlockOperator>()->Mult(div_kern_sol, div_kernel_correction);

    y += div_kernel_correction;
}

void HiptmairSmoother::Mult(const Vector & x, Vector & y) const
{
    Vector CT_r(blk_C_->NumCols());
    CT_r = 0.0;
    Vector div_kern_sol(CT_r);

    blk_C_.As<BlockOperator>()->MultTranspose(x, CT_r);
    div_kernel_smoother_->Mult(CT_r, div_kern_sol);

    Vector div_kernel_correction(blk_C_->NumRows());
    div_kernel_correction = 0.0;
    blk_C_.As<BlockOperator>()->Mult(div_kern_sol, y);

    Vector resid = x;
    op_.Mult(y, resid);
    add(-1.0, resid, 1.0, x, resid);    // resid = x - A(y)

    ml_div_solver_.Mult(level_, resid, div_kernel_correction);
    y += div_kernel_correction;
}

DivFreeSolver::DivFreeSolver(const HypreParMatrix &M, const HypreParMatrix& B,
                             ParFiniteElementSpace* hcurl_fes, const DFSData& data)
    : DarcySolver(M.NumRows(), B.NumRows()), data_(data), M_(M), B_(B),
      BBT_solver_(B, data.param.B_has_nullity_one, data.param.BBT_solve_param),
      CTMC_solver_(B_.GetComm()), block_solver_(B_.GetComm())
{
    HypreParMatrix& C_0 = *data.C[0].As<HypreParMatrix>();

    if (data.param.coupled_solve)
    {
        BT_.Reset(B.Transpose());

        coarse_offsets_.SetSize(data.P_l2.Size()+1);
        coarse_offsets_[0].MakeRef(DarcySolver::offsets_);

        ops_.SetSize(data.P_l2.Size()+1);
        ops_[0].Reset(new BlockOperator(coarse_offsets_[0]));
        ops_[0].As<BlockOperator>()->SetBlock(0, 0, const_cast<HypreParMatrix*>(&M));
        ops_[0].As<BlockOperator>()->SetBlock(1, 0, const_cast<HypreParMatrix*>(&B));
        ops_[0].As<BlockOperator>()->SetBlock(0, 1, BT_.Ptr());

        blk_Ps_.SetSize(data.P_l2.Size());
        smoothers_.SetSize(data.P_l2.Size()+1);
        smoothers_[0].Reset(new HiptmairSmoother(*ops_[0].As<BlockOperator>(),
                            *particular_solver_.As<MLDivSolver>(), 0, C_0));

        for (int l = 0; l < data.P_l2.Size(); ++l)
        {
            HypreParMatrix& P_hdiv_l = *data.P_hdiv[l].As<HypreParMatrix>();
            HypreParMatrix& P_l2_l = *data.P_l2[l].As<HypreParMatrix>();

            auto& M_l = dynamic_cast<HypreParMatrix&>(ops_[l].As<BlockOperator>()->GetBlock(0, 0));
            auto& B_l = dynamic_cast<HypreParMatrix&>(ops_[l].As<BlockOperator>()->GetBlock(1, 0));

            HypreParMatrix* coarse_M = TwoStepsRAP(P_hdiv_l, M_l, P_hdiv_l);
            HypreParMatrix* coarse_B = TwoStepsRAP(P_l2_l, B_l, P_hdiv_l);

            coarse_offsets_[l+1].SetSize(3);
            coarse_offsets_[l+1][0] = 0;
            coarse_offsets_[l+1][1] = coarse_M->NumRows();
            coarse_offsets_[l+1][2] = coarse_M->NumRows() + coarse_B->NumRows();

            if (l == data.P_l2.Size()-1)
            {
                SparseMatrix M_l_diag, B_l_diag;
                coarse_M->GetDiag(M_l_diag);
                coarse_B->GetDiag(B_l_diag);
                for (int dof : data.coarsest_ess_hdivdofs)
                {
                    M_l_diag.EliminateRowCol(dof);
                    B_l_diag.EliminateCol(dof);
                }
            }

            ops_[l+1].Reset(new BlockOperator(coarse_offsets_[l+1]));
            ops_[l+1].As<BlockOperator>()->SetBlock(0, 0, coarse_M);
            ops_[l+1].As<BlockOperator>()->SetBlock(1, 0, coarse_B);
            ops_[l+1].As<BlockOperator>()->SetBlock(0, 1, coarse_B->Transpose());
            ops_[l+1].As<BlockOperator>()->owns_blocks = true;
        }

        particular_solver_.Reset(new MLDivSolver(M, B, data));

        for (int l = 0; l < data.P_l2.Size()+1; ++l)
        {
            if(l < data.P_l2.Size())
            {
                blk_Ps_[l].Reset(new BlockOperator(coarse_offsets_[l], coarse_offsets_[l+1]));
                blk_Ps_[l].As<BlockOperator>()->SetBlock(0, 0, data.P_hdiv[l].Ptr());
                blk_Ps_[l].As<BlockOperator>()->SetBlock(1, 1, data.P_l2[l].Ptr());

                smoothers_[l].Reset(new HiptmairSmoother(*ops_[l].As<BlockOperator>(),
                                      *particular_solver_.As<MLDivSolver>(), l, *data.C[l].As<HypreParMatrix>()));
            }
            else
            {
                auto& M_l = dynamic_cast<HypreParMatrix&>(ops_[l].As<BlockOperator>()->GetBlock(0, 0));
                auto& B_l = dynamic_cast<HypreParMatrix&>(ops_[l].As<BlockOperator>()->GetBlock(1, 0));
                smoothers_[l].Reset(new BDPMinresSolver(M_l, B_l, data_.param.BBT_solve_param));
            }
        }

        CTMC_prec_.Reset(new AbstractMultigrid(ops_, blk_Ps_, smoothers_));
        block_solver_.SetOperator(*ops_[0].Ptr());
        block_solver_.SetPreconditioner(*CTMC_prec_.As<Solver>());
        SetOptions(block_solver_, data_.param.CTMC_solve_param);
        block_solver_.SetMaxIter(100);
        block_solver_.SetPrintLevel(1);
    }

    if (data.param.ml_particular && !data.param.coupled_solve)
    {
        particular_solver_.Reset(new MLDivSolver(M, B, data));
    }

    if (data.param.coupled_solve) { return; }

    CTMC_.Reset(TwoStepsRAP(C_0, M, C_0));
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
        CTMC_prec_.Reset(new AbstractMultigrid(*CTMC_.As<HypreParMatrix>(), data_.P_hcurl));
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
    data_.C[0]->MultTranspose(rhs, rhs_divfree);

    Vector potential_divfree(CTMC_->NumRows());
    CTMC_solver_.Mult(rhs_divfree, potential_divfree);
    PrintConvergence(CTMC_solver_, data_.param.verbose);

    data_.C[0]->Mult(potential_divfree, sol);
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

    BlockVector blk_x(x.GetData(), offsets_);
    BlockVector blk_y(y.GetData(), offsets_);

    if (data_.param.coupled_solve == false)
    {
        StopWatch ch;
        ch.Start();

        Vector x_blk0_copy(blk_x.GetBlock(0));
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
    else
    {
        Vector resid(y.Size()), correction(y.Size());
        ops_[0]->Mult(y, resid);
        add(-1.0, resid, 1.0, x, resid);

        correction = 0.0;
        block_solver_.Mult(resid, correction);
        y += correction;
    }
}

AbstractMultigrid::AbstractMultigrid(HypreParMatrix& op,
                                     const Array<OperatorPtr>& Ps)
    : Solver(op.GetNumRows()),
      Ps_(Ps),
      ops_(Ps.Size()+1),
      smoothers_(ops_.Size()),
      correct_(ops_.Size()),
      resid_(ops_.Size())
{
    ops_[0].Reset(&op, false);
    smoothers_[0].Reset(new HypreSmoother(op));

    for (int l = 1; l < ops_.Size(); ++l)
    {
        HypreParMatrix* P = Ps[l-1].Is<HypreParMatrix>();
        MFEM_ASSERT(P, "P needs to be of type HypreParMatrix");

        ops_[l].Reset(TwoStepsRAP(*P, *ops_[l-1].As<HypreParMatrix>(), *P));
        ops_[l].As<HypreParMatrix>()->Threshold(1e-14);
        smoothers_[l].Reset(new HypreSmoother(*ops_[l].As<HypreParMatrix>()));
        resid_[l].SetSize(ops_[l]->NumRows());
        correct_[l].SetSize(ops_[l]->NumRows());
    }
}

AbstractMultigrid::AbstractMultigrid(const Array<OperatorPtr>& ops,
                  const Array<OperatorPtr>& Ps,
                  const Array<OperatorPtr>& smoothers)
    : Solver(ops[0]->NumRows()),
      Ps_(Ps),
      ops_(Ps.Size()+1),
      smoothers_(ops_.Size()),
      correct_(ops_.Size()),
      resid_(ops_.Size())
{
    for (int l = 0; l < ops_.Size(); ++l)
    {
        ops_[l].Reset(ops[l].Ptr(), false);
        smoothers_[l].Reset(smoothers[l].Ptr(), false);
        resid_[l].SetSize(ops_[l]->NumRows());
        correct_[l].SetSize(ops_[l]->NumRows());
    }
}

void AbstractMultigrid::Mult(const Vector& x, Vector& y) const
{
    resid_[0] = x;
    correct_[0].SetDataAndSize(y.GetData(), y.Size());

    for (int l = 1; l < ops_.Size(); ++l)
    {
        resid_[l] = 0.0;
        correct_[l] = 0.0;
    }

    MG_Cycle(0);
}

void AbstractMultigrid::MG_Cycle(int level) const
{
    // PreSmoothing
    smoothers_[level]->Mult(resid_[level], correct_[level]);

    if (level == Ps_.Size()) { return; }

    Vector Ax(ops_[level]->NumCols());
    ops_[level]->Mult(correct_[level], Ax);
    resid_[level] -= Ax;

    // Coarse grid correction
    Ps_[level]->MultTranspose(resid_[level], resid_[level+1]);
    MG_Cycle(level+1);
    cor_cor_.SetSize(resid_[level].Size());
    Ps_[level]->Mult(correct_[level+1], cor_cor_);
    correct_[level] += cor_cor_;

    ops_[level]->Mult(cor_cor_, Ax);
    resid_[level] -= Ax;

    // PostSmoothing
    smoothers_[level]->MultTranspose(resid_[level], cor_cor_);
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

