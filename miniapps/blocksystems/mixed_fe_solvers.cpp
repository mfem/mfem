
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

void GetRowColumnsRef(const SparseMatrix& A, int row, Array<int>& cols)
{
    cols.MakeRef(const_cast<int*>(A.GetRowColumns(row)), A.RowSize(row));
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
    const double x0 = x[offset_];
    const_cast<Vector&>(x)[offset_] = 0.0;

    y.SetSize(local_system_.NumRows());
    local_solver_.Mult(x, y);

    const_cast<Vector&>(x)[offset_] = x0;
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
        OperatorPtr PT_l2(data_.P_l2[l].As<HypreParMatrix>()->Transpose());
        auto PTW = ParMult(PT_l2.As<HypreParMatrix>(), W.As<HypreParMatrix>(), true);
        W.Reset(ParMult(PTW, data_.P_l2[l].As<HypreParMatrix>()));
        auto cW_inv = new BlockDiagSolver(W, move(el_l2dof_[l+1]));
        //TODO: maybe store the product is better if it needs to be applied in iterations
        data_.Q_l2[l].Reset(new ProductOperator(cW_inv, PTW, true, true));
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

SchwarzSmoother::SchwarzSmoother(const HypreParMatrix& M,
                                 const HypreParMatrix& B,
                                 const SparseMatrix& agg_hdivdof,
                                 const SparseMatrix& agg_l2dof,
                                 const HypreParMatrix& P_l2,
                                 const HypreParMatrix& Q_l2)
    : Solver(M.NumRows() + B.NumRows()), agg_hdivdof_(agg_hdivdof),
      agg_l2dof_(agg_l2dof), solvers_loc_(agg_l2dof.NumRows())
{
    coarse_l2_projector_.Reset(new ProductOperator(&P_l2, &Q_l2, false, false));

    offsets_loc_.SetSize(3, 0);
    offsets_.SetSize(3, 0);
    offsets_[1] = M.NumRows();
    offsets_[2] = M.NumRows() + B.NumRows();

    SparseMatrix M_diag, B_diag;
    M.GetDiag(M_diag);
    B.GetDiag(B_diag);

    DenseMatrix B_loc, M_loc;

    for (int agg = 0; agg < solvers_loc_.Size(); agg++)
    {
        GetRowColumnsRef(agg_hdivdof_, agg, hdivdofs_loc_);
        GetRowColumnsRef(agg_l2dof_, agg, l2dofs_loc_);
        GetSubMatrix(M_diag, hdivdofs_loc_, hdivdofs_loc_, M_loc);
        GetSubMatrix(B_diag, l2dofs_loc_, hdivdofs_loc_, B_loc);
        solvers_loc_[agg].Reset(new LocalSolver(M_loc, B_loc));
    }
}

void SchwarzSmoother::Mult(const Vector & x, Vector & y) const
{
    y.SetSize(offsets_[2]);
    y = 0.0;

    BlockVector blk_y(y.GetData(), offsets_);
    BlockVector Pi_x(offsets_); // aggregate-wise average free projection of x
    static_cast<Vector&>(Pi_x) = x;

    // Right hand side: F_l = F - W_l P_l2[l] (W_{l+1})^{-1} P_l2[l]^T F
    Vector coarse_l2_projection(Pi_x.BlockSize(1));
    coarse_l2_projector_->MultTranspose(Pi_x.GetBlock(1), coarse_l2_projection);

    Pi_x.GetBlock(1) -= coarse_l2_projection;

    for (int agg = 0; agg < solvers_loc_.Size(); agg++)
    {
        GetRowColumnsRef(agg_hdivdof_, agg, hdivdofs_loc_);
        GetRowColumnsRef(agg_l2dof_, agg, l2dofs_loc_);

        offsets_loc_[1] = hdivdofs_loc_.Size();
        offsets_loc_[2] = offsets_loc_[1]+l2dofs_loc_.Size();

        BlockVector rhs_loc(offsets_loc_), sol_loc(offsets_loc_);
        Pi_x.GetBlock(0).GetSubVector(hdivdofs_loc_, rhs_loc.GetBlock(0));
        Pi_x.GetBlock(1).GetSubVector(l2dofs_loc_, rhs_loc.GetBlock(1));

        solvers_loc_[agg]->Mult(rhs_loc, sol_loc);

        blk_y.GetBlock(0).AddElementVector(hdivdofs_loc_, sol_loc.GetBlock(0));
        blk_y.GetBlock(1).AddElementVector(l2dofs_loc_, sol_loc.GetBlock(1));
    }

    coarse_l2_projector_->Mult(blk_y.GetBlock(1), coarse_l2_projection);
    blk_y.GetBlock(1) -= coarse_l2_projection;
}

KernelSmoother::KernelSmoother(const BlockOperator& op, const HypreParMatrix& kernel_map)
    : Solver(op.NumRows()), offsets_(2)
{
    offsets_[0] = 0;
    offsets_[1] = kernel_map.NumCols();

    BlockOperator* blk_map = new BlockOperator(op.ColOffsets(), offsets_);
    blk_map->SetBlock(0, 0, const_cast<HypreParMatrix*>(&kernel_map));
    blk_kernel_map_.Reset(blk_map);

    const HypreParMatrix& M = dynamic_cast<const HypreParMatrix&>(op.GetBlock(0, 0));
    kernel_system_.Reset(TwoStepsRAP(kernel_map, M, kernel_map));
    kernel_system_.As<HypreParMatrix>()->EliminateZeroRows();
    kernel_system_.As<HypreParMatrix>()->Threshold(1e-14);
    kernel_smoother_.Reset(new HypreSmoother(*kernel_system_.As<HypreParMatrix>()));
}

void KernelSmoother::Mult(const Vector & x, Vector & y) const
{
    Vector kernel_rhs(blk_kernel_map_->NumCols());
    blk_kernel_map_.As<BlockOperator>()->MultTranspose(x, kernel_rhs);

    Vector kernel_sol(kernel_rhs.Size());
    kernel_smoother_->Mult(kernel_rhs, kernel_sol);

    y.SetSize(blk_kernel_map_->NumRows());
    blk_kernel_map_.As<BlockOperator>()->Mult(kernel_sol, y);
}

void ProductSolver::Mult(int i, int j, const Vector & x, Vector & y) const
{
    y.SetSize(x.Size());
    y = 0.0;
    solvers_[i]->Mult(x, y);

    Vector resid(x.Size());
    resid = 0.0;
    op_->Mult(y, resid);
    add(-1.0, resid, 1.0, x, resid);    // resid = x - A(y)

    Vector correction(x.Size());
    correction = 0.0;

    solvers_[j]->Mult(resid, correction);
    y += correction;
}

DivFreeSolver::DivFreeSolver(const HypreParMatrix &M, const HypreParMatrix& B,
                             ParFiniteElementSpace* hcurl_fes, const DFSData& data)
    : DarcySolver(M.NumRows(), B.NumRows()), data_(data), BT_(B.Transpose()),
      BBT_solver_(B, data.param.B_has_nullity_one, data.param.BBT_solve_param),
      ops_offsets_(data.P_l2.Size()+1), ops_(ops_offsets_.Size()),
      blk_Ps_(ops_.Size()-1), smoothers_(ops_.Size())
{
    ops_offsets_[0].MakeRef(DarcySolver::offsets_);
    ops_[0].Reset(new BlockOperator(ops_offsets_[0]));
    ops_[0].As<BlockOperator>()->SetBlock(0, 0, const_cast<HypreParMatrix*>(&M));
    ops_[0].As<BlockOperator>()->SetBlock(1, 0, const_cast<HypreParMatrix*>(&B));
    ops_[0].As<BlockOperator>()->SetBlock(0, 1, BT_.Ptr());

    for (int l = 0; l < data.P_l2.Size(); ++l)
    {
        HypreParMatrix& P_hdiv_l = *data.P_hdiv[l].As<HypreParMatrix>();
        HypreParMatrix& P_l2_l = *data.P_l2[l].As<HypreParMatrix>();

        auto& M_f = dynamic_cast<HypreParMatrix&>(ops_[l].As<BlockOperator>()->GetBlock(0, 0));
        auto& B_f = dynamic_cast<HypreParMatrix&>(ops_[l].As<BlockOperator>()->GetBlock(1, 0));

        auto S0 = new SchwarzSmoother(M_f, B_f, *data.agg_hdivdof[l].As<SparseMatrix>(),
                                      *data.agg_l2dof[l].As<SparseMatrix>(),
                                      P_l2_l, *data.Q_l2[l].As<HypreParMatrix>());
        if (data.param.coupled_solve)
        {
            auto S1 = new KernelSmoother(*ops_[l].As<BlockOperator>(),
                                         *data.C[l].As<HypreParMatrix>());
            smoothers_[l].Reset(new ProductSolver(ops_[l].Ptr(), S0, S1, false, true, true));
        }
        else
        {
            smoothers_[l].Reset(S0);
        }

        HypreParMatrix* M_c = TwoStepsRAP(P_hdiv_l, M_f, P_hdiv_l);
        HypreParMatrix* B_c = TwoStepsRAP(P_l2_l, B_f, P_hdiv_l);

        ops_offsets_[l+1].SetSize(3, 0);
        ops_offsets_[l+1][1] = M_c->NumRows();
        ops_offsets_[l+1][2] = M_c->NumRows() + B_c->NumRows();

        blk_Ps_[l].Reset(new BlockOperator(ops_offsets_[l], ops_offsets_[l+1]));
        blk_Ps_[l].As<BlockOperator>()->SetBlock(0, 0, &P_hdiv_l);
        blk_Ps_[l].As<BlockOperator>()->SetBlock(1, 1, &P_l2_l);

        if (l < data.P_l2.Size()-1)
        {
            ops_[l+1].Reset(new BlockOperator(ops_offsets_[l+1]));
            ops_[l+1].As<BlockOperator>()->SetBlock(0, 0, M_c);
            ops_[l+1].As<BlockOperator>()->SetBlock(1, 0, B_c);
            ops_[l+1].As<BlockOperator>()->SetBlock(0, 1, B_c->Transpose());
            ops_[l+1].As<BlockOperator>()->owns_blocks = true;
        }
        else
        {
            SparseMatrix M_c_diag, B_c_diag;
            M_c->GetDiag(M_c_diag);
            B_c->GetDiag(B_c_diag);
            for (int dof : data.coarsest_ess_hdivdofs)
            {
                M_c_diag.EliminateRowCol(dof);
                B_c_diag.EliminateCol(dof);
            }

            const IterSolveParameters& param = data.param.BBT_solve_param;
            smoothers_[l+1].Reset(new BDPMinresSolver(*M_c, *B_c, true, param));
            smoothers_[l+1].As<BDPMinresSolver>()->SetEssZeroDofs(data.coarsest_ess_hdivdofs);
        }
    }

    if (data.param.coupled_solve)
    {
        solver_.Reset(new GMRESSolver(B.GetComm()));
        solver_.As<GMRESSolver>()->SetOperator(*ops_[0].Ptr());
        prec_.Reset(new AbstractMultigrid(ops_, blk_Ps_, smoothers_));
    }
    else
    {
        solver_.Reset(new CGSolver(B.GetComm()));
        HypreParMatrix& C_0 = *data.C[0].As<HypreParMatrix>();
        CTMC_.Reset(TwoStepsRAP(C_0, M, C_0));
        CTMC_.As<HypreParMatrix>()->EliminateZeroRows();
        CTMC_.As<HypreParMatrix>()->Threshold(1e-14);
        solver_.As<CGSolver>()->SetOperator(*CTMC_);

        if (data_.param.MG_type == AlgebraicMG)
        {
            prec_.Reset(new HypreAMS(*CTMC_.As<HypreParMatrix>(), hcurl_fes));
            prec_.As<HypreAMS>()->SetSingularProblem();
        }
        else
        {
            prec_.Reset(new AbstractMultigrid(*CTMC_.As<HypreParMatrix>(), data_.P_hcurl));
        }
    }

    solver_.As<IterativeSolver>()->SetPreconditioner(*prec_.As<Solver>());
    SetOptions(*solver_.As<IterativeSolver>(), data_.param);
}

void DivFreeSolver::SolveParticular(const Vector& rhs, Vector& sol) const
{
    Array<Vector> rhss(smoothers_.Size());
    Array<Vector> sols(smoothers_.Size());

    rhss[0].SetDataAndSize(const_cast<Vector&>(rhs), rhs.Size());
    sols[0].SetDataAndSize(sol, sol.Size());

    for (int l = 0; l < blk_Ps_.Size(); ++l)
    {
        rhss[l+1].SetSize(blk_Ps_[l]->NumCols());
        sols[l+1].SetSize(blk_Ps_[l]->NumCols());
        blk_Ps_[l]->MultTranspose(rhss[l], rhss[l+1]);
    }

    for (int l = 0; l < smoothers_.Size(); ++l)
    {
        smoothers_[l]->Mult(rhss[l], sols[l]);
    }

    for (int l = blk_Ps_.Size()-1; l >= 0; l--)
    {
        Vector P_sol(blk_Ps_[l]->NumRows());
        blk_Ps_[l]->Mult(sols[l+1], P_sol);
        sols[l] += P_sol;
    }
}

void DivFreeSolver::SolveDivFree(const Vector &rhs, Vector& sol) const
{
    Vector rhs_divfree(CTMC_->NumRows());
    data_.C[0]->MultTranspose(rhs, rhs_divfree);

    Vector potential_divfree(CTMC_->NumRows());
    solver_->Mult(rhs_divfree, potential_divfree);
    PrintConvergence(*solver_.As<CGSolver>(), data_.param.verbose);

    data_.C[0]->Mult(potential_divfree, sol);
}

void DivFreeSolver::SolvePotential(const Vector& rhs, Vector& sol) const
{
    Vector rhs_p(BT_->NumCols());
    BT_->MultTranspose(rhs, rhs_p);
    BBT_solver_.Mult(rhs_p, sol);
}

void DivFreeSolver::Mult(const Vector & x, Vector & y) const
{
    MFEM_VERIFY(x.Size() == offsets_[2], "MLDivFreeSolver: x size is invalid");
    MFEM_VERIFY(y.Size() == offsets_[2], "MLDivFreeSolver: y size is invalid");

    BlockVector blk_y(y, offsets_);

    BlockVector resid(offsets_);
    ops_[0]->Mult(y, resid);
    add(1.0, x, -1.0, resid, resid);

    BlockVector correction(offsets_);
    correction = 0.0;

    if (data_.param.coupled_solve)
    {
        solver_->Mult(resid, correction);
        y += correction;
    }
    else
    {
        StopWatch ch;
        ch.Start();

        SolveParticular(resid, correction);
        blk_y += correction;

        if (data_.param.verbose)
            cout << "Particular solution found in " << ch.RealTime() << "s.\n";

        ch.Clear();
        ch.Start();

        ops_[0]->Mult(y, resid);
        add(1.0, x, -1.0, resid, resid);

        SolveDivFree(resid.GetBlock(0), correction.GetBlock(0));
        blk_y.GetBlock(0) += correction.GetBlock(0);

        if (data_.param.verbose)
            cout << "Divergence free solution found in " << ch.RealTime() << "s.\n";

        ch.Clear();
        ch.Start();

        auto M = dynamic_cast<HypreParMatrix&>(ops_[0].As<BlockOperator>()->GetBlock(0,0));
        M.Mult(-1.0, correction.GetBlock(0), 1.0, resid.GetBlock(0));
        SolvePotential(resid.GetBlock(0), correction.GetBlock(1));
        blk_y.GetBlock(1) += correction.GetBlock(1);

        if (data_.param.verbose)
            cout << "Scalar potential found in " << ch.RealTime() << "s.\n";
    }
}

AbstractMultigrid::AbstractMultigrid(HypreParMatrix& op,
                                     const Array<OperatorPtr>& Ps)
    : Solver(op.GetNumRows()), Ps_(Ps), ops_(Ps.Size()+1),
      smoothers_(ops_.Size()), correct_(ops_.Size()), resid_(ops_.Size()),
      smoothers_are_symmetric_(true)
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
                                     const Array<OperatorPtr>& smoothers,
                                     bool smoothers_are_symmetric)
    : Solver(ops[0]->NumRows()), Ps_(Ps), ops_(Ps.Size()+1),
      smoothers_(ops_.Size()), correct_(ops_.Size()), resid_(ops_.Size()),
      smoothers_are_symmetric_(smoothers_are_symmetric)
{
    for (int l = 0; l < ops_.Size(); ++l)
    {
        ops_[l].Reset(ops[l].Ptr(), false);
        smoothers_[l].Reset(smoothers[l].Ptr(), false);
        resid_[l].SetSize(smoothers[l]->NumRows());
        correct_[l].SetSize(smoothers[l]->NumRows());
    }
}

void AbstractMultigrid::Mult(const Vector& x, Vector& y) const
{
    resid_[0] = x;
    correct_[0].SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle(0);
}

void AbstractMultigrid::MG_Cycle(int level) const
{
    // PreSmoothing
    smoothers_[level]->Mult(resid_[level], correct_[level]);

    if (level == Ps_.Size()) { return; }

    // Coarse grid correction
    Vector Ax(ops_[level]->NumCols());
    ops_[level]->Mult(correct_[level], Ax);
    resid_[level] -= Ax;

    Ps_[level]->MultTranspose(resid_[level], resid_[level+1]);
    MG_Cycle(level+1);
    cor_cor_.SetSize(resid_[level].Size());
    Ps_[level]->Mult(correct_[level+1], cor_cor_);
    correct_[level] += cor_cor_;

    // PostSmoothing
    ops_[level]->Mult(cor_cor_, Ax);
    resid_[level] -= Ax;

    if (smoothers_are_symmetric_)
    {
        smoothers_[level]->Mult(resid_[level], cor_cor_);
    }
    else
    {
        smoothers_[level]->MultTranspose(resid_[level], cor_cor_);
    }
    correct_[level] += cor_cor_;
}

BDPMinresSolver::BDPMinresSolver(HypreParMatrix& M, HypreParMatrix& B,
                                 bool own_input, IterSolveParameters param)
    : DarcySolver(M.NumRows(), B.NumRows()), op_(offsets_), prec_(offsets_),
      BT_(B.Transpose(), !own_input), solver_(M.GetComm())
{
    op_.SetBlock(0,0, &M);
    op_.SetBlock(0,1, BT_.As<HypreParMatrix>());
    op_.SetBlock(1,0, &B);
    op_.owns_blocks = own_input;

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
    for (int dof : ess_zero_dofs_) { y[dof] = 0.0; }
    PrintConvergence(solver_, false);
}

