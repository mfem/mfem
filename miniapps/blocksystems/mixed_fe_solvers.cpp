
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

void W_Inverse(const HypreParMatrix& W_coarse, const SparseMatrix& agg_l2dof,
               const Vector& PT_F_l, Vector& F_coarse)
{
    SparseMatrix W;
    W_coarse.GetDiag(W);

    Array<int> agg_l2dofs;
    Vector sub_F_coarse, sub_PT_F_l, trash;
    DenseMatrix sub_W;
    DenseMatrixInverse sub_W_solver;

    F_coarse.SetSize(PT_F_l.Size());

    F_coarse = 0.0;
    for(int agg = 0; agg < agg_l2dof.NumRows(); agg++)
    {
        agg_l2dof.GetRow(agg, agg_l2dofs, trash); // TODO: make custom GetRowColumnsRef
        GetSubMatrix(W, agg_l2dofs, agg_l2dofs, sub_W);
        PT_F_l.GetSubVector(agg_l2dofs, sub_PT_F_l);
        sub_F_coarse.SetSize(agg_l2dofs.Size());
        sub_W_solver.SetOperator(sub_W);
        sub_W_solver.Mult(sub_PT_F_l, sub_F_coarse);
        F_coarse.AddElementVector(agg_l2dofs, sub_F_coarse);
    }
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
                 const HypreParMatrix& W,
                 const Vector& F,
                 const Array<SparseMatrix>& agg_elem,
                 const Array<SparseMatrix>& elem_hdivdof,
                 const Array<SparseMatrix>& elem_l2dof,
                 const Array<OperatorPtr>& P_hdiv,
                 const Array<OperatorPtr>& P_l2,
                 const Array<Array<int> >& bdr_hdivdofs,
                 const Array<int>& coarsest_ess_hdivdofs)
{
    const unsigned int num_levels = elem_hdivdof.Size() + 1;

    Array<OperatorHandle> Ws(num_levels);
    Ws[0].Reset(const_cast<HypreParMatrix*>(&W), false);
    OperatorPtr B_l(const_cast<HypreParMatrix*>(&B), false);
    OperatorPtr M_l;//(M.NumRows() ? const_cast<HypreParMatrix*>(&M) : NULL, false);

    int myid;
    MPI_Comm_rank(B.GetComm(), &myid);

    Array<Vector> sigma(num_levels);
    Vector F_l, PT_F_l, F_coarse, F_a, PF_coarse, trash;
    Array<int> loc_hdivdofs, loc_l2dofs, col_marker(B.NumCols());
    SparseMatrix B_l_diag, M_l_diag;
    DenseMatrix B_a, M_a;

    col_marker = -1;

    for (unsigned int l = 0; l < num_levels - 1; ++l)
    {
        OperatorPtr agg_l2dof(Mult(agg_elem[l], elem_l2dof[l]));
        auto agg_hdivdof = AggToInteriorDof(agg_elem[l], elem_hdivdof[l], bdr_hdivdofs[l]);

        Ws[l+1].MakeRAP(const_cast<OperatorPtr&>(P_l2[l]), Ws[l], const_cast<OperatorPtr&>(P_l2[l]));

        // Right hand side: F_l = F - W_l P_l2[l] (W_{l+1})^{-1} P_l2[l]^T F
        F_l = l == 0 ? F : PT_F_l;
        PT_F_l.SetSize(P_l2[l]->NumCols());
        P_l2[l]->MultTranspose(F_l, PT_F_l);
        W_Inverse(*Ws[l+1].As<HypreParMatrix>(), elem_l2dof[l+1], PT_F_l, F_coarse);

        PF_coarse.SetSize(P_l2[l]->NumRows());
        P_l2[l]->Mult(F_coarse, PF_coarse);
        Ws[l].As<HypreParMatrix>()->Mult(-1.0, PF_coarse, 1.0, F_l);

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
    B_l.As<HypreParMatrix>()->GetDiag(B_l_diag);
    for (int dof : coarsest_ess_hdivdofs) B_l_diag.EliminateCol(dof);

    if (M_l.Ptr())
    {
        Array<int> block_offsets(3);
        block_offsets[0] = 0;
        block_offsets[1] = M_l->NumRows();
        block_offsets[2] = block_offsets[1] + B_l->NumRows();

        OperatorPtr M_l_elim;
        M_l_elim.EliminateRowsCols(M_l, coarsest_ess_hdivdofs);
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

    OperatorPtr MC(ParMult(&M_, data.C.As<HypreParMatrix>()));
    OperatorPtr CT(data.C.As<HypreParMatrix>()->Transpose());
    CTMC_.Reset(ParMult(CT.As<HypreParMatrix>(), MC.As<HypreParMatrix>()));
    CTMC_.As<HypreParMatrix>()->CopyRowStarts();
    CTMC_.As<HypreParMatrix>()->EliminateZeroRows();
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
    if (data_.param.ml_particular)
    {
        sol = MLDivPart(M_, B_, *data_.W.As<HypreParMatrix>(), rhs, data_.agg_el,
                        data_.el_hdivdof, data_.el_l2dof, data_.P_hdiv, data_.P_l2,
                        data_.bdr_hdivdofs, data_.coarsest_ess_hdivdofs);
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

