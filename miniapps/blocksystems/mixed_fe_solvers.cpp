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

void GetSubMatrix(const HypreParMatrix& A, const Array<int> rows,
                  const Array<int> cols, DenseMatrix& sub_A)
{
    sub_A.SetSize(rows.Size(), cols.Size());
    SparseMatrix A_diag_block;
    A.GetDiag(A_diag_block);
    A_diag_block.GetSubMatrix(rows, cols, sub_A);
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
    int* I = const_cast<int*>(fes.GetElementToDofTable().GetI());

    const int nnz = I[num_elems];
    vector<double> D(nnz, 1.0);

    Array<int> J(nnz); //TODO: can we simply use J?
    copy_n(fes.GetElementToDofTable().GetJ(), nnz, J.begin());
    fes.AdjustVDofs(J);

    SparseMatrix elem_dof(I, J.GetData(), D.data(), num_elems, fes.GetVSize());
    SparseMatrix true_dofs_restrict;
    fes.Dof_TrueDof_Matrix()->GetDiag(true_dofs_restrict);

    OperatorHandle elem_truedof(Mult(elem_dof, true_dofs_restrict));
    return *elem_truedof.As<SparseMatrix>();
}

Vector MLDivPart(const HypreParMatrix& M,
                 const HypreParMatrix& B,
                 const Vector& F,
                 const vector<SparseMatrix>& agg_elem,
                 const vector<SparseMatrix>& elem_hdivdofs,
                 const vector<SparseMatrix>& elem_l2dofs,
                 const vector<OperatorHandle>& P_hdiv,
                 const vector<OperatorHandle>& P_l2,
                 const HypreParMatrix& coarse_hdiv_d_td,
                 const HypreParMatrix& coarse_l2_d_td,
                 const Array<int> &coarsest_ess_dofs)
{
    const unsigned int num_levels = elem_hdivdofs.size() + 1;

    vector<Vector> sigma(num_levels);
    Vector F_l, Pi_F_l, F_coarse, PT_F_l(F);
    OperatorHandle B_l(Add(1.0, B, 0.0, B));
    OperatorHandle M_l(M.NumRows() ? Add(1.0, M, 0.0, M) : NULL);

    for (unsigned int l = 0; l < num_levels - 1; ++l)
    {
        unique_ptr<SparseMatrix> agg_l2dof(Mult(agg_elem[l], elem_l2dofs[l]));
        auto agg_hdivintdof = AggToIntDof(agg_elem[l], elem_hdivdofs[l]);

        // Right hand side: F_l = F - P_l2[l] (P_l2[l]^T P_l2[l])^{-1} P_l2[l]^T F
        F_l = PT_F_l;
        PT_F_l.SetSize(P_l2[l]->NumCols());
        P_l2[l]->MultTranspose(F_l, PT_F_l);

        unique_ptr<SparseMatrix> PT_l2(Transpose(P_l2[l]));
        unique_ptr<SparseMatrix> PTP_l2(Mult(*PT_l2, P_l2[l]));

        F_coarse.SetSize(P_l2[l].NumCols());
        for(int m = 0; m < F_coarse.Size(); m++)
        {
            F_coarse[m] = PT_F_l[m] / PTP_l2->Elem(m, m);
        }

        Pi_F_l.SetSize(P_l2[l].NumRows());
        P_l2[l].Mult(F_coarse, Pi_F_l);
        F_l -= Pi_F_l;

        DenseMatrix B_a;
        DenseMatrix M_a;

        Vector F_a;
        Vector trash;

        Array<int> loc_hdivdofs;
        Array<int> loc_l2dofs;

        sigma[l].SetSize(agg_hdivintdof.NumCols());
        sigma[l] = 0.0;

        for (int agg = 0; agg < agg_hdivintdof.NumRows(); agg++)
        {
            agg_hdivintdof.GetRow(agg, loc_hdivdofs, trash);
            agg_l2dof->GetRow(agg, loc_l2dofs, trash);

            if (M_l.Ptr())
            {
                GetSubMatrix(*M_l.As<HypreParMatrix>(), loc_hdivdofs, loc_hdivdofs, M_a);
            }

            GetSubMatrix(*B_l.As<HypreParMatrix>(), loc_l2dofs, loc_hdivdofs, B_a);

            F_l.GetSubVector(loc_l2dofs, F_a);

            Vector sigma_loc = LocalSolution(M_a, B_a, F_a);

            sigma[l].AddElementVector(loc_hdivdofs, sigma_loc);
        }  // loop over elements

        B_l.MakeRAP(const_cast<OperatorHandle&>(P_l2[l]), B_l, const_cast<OperatorHandle&>(P_hdiv[l]));
        if (M_l.Ptr())
        {
            M_l.MakePtAP(M_l, const_cast<OperatorHandle&>(P_hdiv[l]));
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

    OperatorHandle BT_l(B_l->Transpose());

    Vector true_sigma_c(B_l->NumCols());

    if (M_l.Ptr())
    {
//        unique_ptr<HypreParMatrix> d_td_M(coarse_hdiv_d_td.LeftDiagMult(*M_l));
//        unique_ptr<HypreParMatrix> d_td_T(coarse_hdiv_d_td.Transpose());
//        unique_ptr<HypreParMatrix> M_Coarse(ParMult(d_td_T.get(), d_td_M.get()));

        Array<int> block_offsets(3);
        block_offsets[0] = 0;
        block_offsets[1] = M_l->NumRows();
        block_offsets[2] = block_offsets[1] + B_l->NumRows();

        BlockOperator coarseMatrix(block_offsets);
        coarseMatrix.SetBlock(0,0, M_l.Ptr());
        coarseMatrix.SetBlock(0,1, BT_l.Ptr());
        coarseMatrix.SetBlock(1,0, B_l.Ptr());

        BlockVector trueX(block_offsets), trueRhs(block_offsets);
        trueRhs = 0.0;
        trueRhs.GetBlock(1)= PT_F_l;

        L2H1Preconditioner darcyPr(*M_l.As<HypreParMatrix>(), *B_l.As<HypreParMatrix>(), block_offsets);

        MINRESSolver solver(coarse_l2_d_td.GetComm());
        SetOptions(solver, 0, 500, 1e-12, 1e-9);
        solver.SetOperator(coarseMatrix);
        solver.SetPreconditioner(darcyPr);
        trueX = 0.0;
        solver.Mult(trueRhs, trueX);
        true_sigma_c = trueX.GetBlock(0);
    }
    else
    {
        BBTSolver BBT_solver(*B_Coarse);

        Vector u_c(B_Coarse->Height());
        u_c = 0.0;
        BBT_solver.Mult(PT_F_l, u_c);
        BT_coarse->Mult(u_c, true_sigma_c);
    }

    sigma[num_levels-1].SetSize(B_l->Width());
    coarse_hdiv_d_td.Mult(true_sigma_c, sigma[num_levels-1]);

    Vector P_sigma;
    for (int k = num_levels-2; k>=0; k--)
    {
        P_sigma.SetSize(P_hdiv[k].NumRows());
        P_hdiv[k].Mult(sigma[k+1], P_sigma);
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

InterpolationCollector::InterpolationCollector(ParFiniteElementSpace& fes,
                                               int num_refine)
    : coarse_fes_(fes.GetParMesh(), fes.FEColl()), refine_count_(num_refine)
{
    P_.SetSize(num_refine, OperatorHandle(Operator::Hypre_ParCSR));
}

void InterpolationCollector::CollectData(ParFiniteElementSpace& fes)
{
    fes.Update();
    fes.GetTrueTransferOperator(coarse_fes_, P_[--refine_count_]);
    coarse_fes_.Update();
}

HdivL2Hierarchy::HdivL2Hierarchy(ParFiniteElementSpace& hdiv_fes,
                                 ParFiniteElementSpace& l2_fes,
                                 int num_refine,
                                 const Array<int>& ess_bdr)
    : hdiv_fes_(hdiv_fes),
      l2_fes_(l2_fes),
      coarse_hdiv_fes_(hdiv_fes.GetParMesh(), hdiv_fes.FEColl()),
      coarse_l2_fes_(l2_fes.GetParMesh(), l2_fes.FEColl()),
      l2_coll_0(0, l2_fes.GetParMesh()->SpaceDimension()),
      l2_fes_0_(l2_fes.GetParMesh(), &l2_coll_0),
      agg_el_(num_refine),
      el_hdivdofs_(num_refine),
      el_l2dofs_(num_refine),
      P_hdiv_(num_refine, OperatorHandle(Operator::Hypre_ParCSR)),
      P_l2_(num_refine, OperatorHandle(Operator::Hypre_ParCSR)),
      refine_count_(num_refine)
{
    coarse_hdiv_fes_.GetEssentialVDofs(ess_bdr, coarse_ess_dofs_);
    hdiv_fes_.SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
    l2_fes_.SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
    l2_fes_0_.SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
}

void HdivL2Hierarchy::CollectData()
{
//    P_hdiv_[--refine_count_] = (const SparseMatrix&)*hdiv_fes_.GetUpdateOperator();
//    P_l2_[refine_count_] = (const SparseMatrix&)*l2_fes_.GetUpdateOperator();
    hdiv_fes_.GetTrueTransferOperator(coarse_hdiv_fes_, P_hdiv_[--refine_count_]);
    l2_fes_.GetTrueTransferOperator(coarse_l2_fes_, P_l2_[refine_count_]);
    P_hdiv_[refine_count_].As<HypreParMatrix>()->Threshold(1e-16);
    P_l2_[refine_count_].As<HypreParMatrix>()->Threshold(1e-16);

    auto& elem_agg_l = (const SparseMatrix&)*l2_fes_0_.GetUpdateOperator();
    OperatorHandle agg_elem_l(Transpose(elem_agg_l));
    agg_el_[refine_count_].Swap(*agg_elem_l.As<SparseMatrix>());

    el_hdivdofs_[refine_count_] = ElemToTrueDofs(hdiv_fes_);
    el_l2dofs_[refine_count_] = ElemToTrueDofs(l2_fes_);

    coarse_hdiv_fes_.Update();
    coarse_l2_fes_.Update();
}

MLDivFreeSolver::MLDivFreeSolver(HdivL2Hierarchy& hierarchy, HypreParMatrix& M,
                                 HypreParMatrix& B, HypreParMatrix& BT, HypreParMatrix& C,
                                 MLDivFreeSolveParameters param)
    : h_(hierarchy), M_(M), B_(B), BT_(BT), C_(C), CT_(C.Transpose()),
      BBT_solver_(B, param.BBT_solve_param), CTMC_solver_(C.GetComm()),
      param_(param), offsets_(3)
{
        offsets_[0] = 0;
        offsets_[1] = M.NumCols();
        offsets_[2] = offsets_[1] + B.NumRows();

        unique_ptr<HypreParMatrix> MC(ParMult(&M_, &C));
        CTMC_.Reset(ParMult(CT_.As<HypreParMatrix>(), MC.get()));
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

void MLDivFreeSolver::SolveParticularSolution(const Vector& blk_rhs_1,
                                              Vector& true_flux_part) const
{
    if (param_.ml_part)
    {
        MFEM_VERIFY(B_fine_.NumRows() > 0, "MLDivFreeSolver: op is not set!");
        Vector sigma_part = MLDivPart(M_fine_, B_fine_, blk_rhs_1,
                                      h_.agg_el_, h_.el_hdivdofs_, h_.el_l2dofs_,
                                      h_.P_hdiv_, h_.P_l2_,
                                      *h_.coarse_hdiv_fes_.Dof_TrueDof_Matrix(),
                                      *h_.coarse_l2_fes_.Dof_TrueDof_Matrix(),
                                      h_.coarse_ess_dofs_);

        SparseMatrix true_hdiv_dof_restrict;
        h_.hdiv_fes_.Dof_TrueDof_Matrix()->GetDiag(true_hdiv_dof_restrict);
        true_hdiv_dof_restrict.MultTranspose(sigma_part, true_flux_part);
    }
    else
    {
        Vector potential(blk_rhs_1.Size());
        BBT_solver_.Mult(blk_rhs_1, potential);
        BT_.Mult(potential, true_flux_part);
    }
}

void MLDivFreeSolver::SolveDivFreeSolution(const Vector& true_flux_part,
                                           Vector& blk_rhs_0,
                                           Vector& true_flux_divfree) const
{
    // Compute the right hand side for the divergence free solver problem
    Vector rhs_divfree(CTMC_->NumRows());
    M_.Mult(-1.0, true_flux_part, 1.0, blk_rhs_0);
    CT_->Mult(blk_rhs_0, rhs_divfree);

    // Solve the "potential" of divergence free solution
    Vector potential_divfree(CTMC_->NumRows());
    CTMC_solver_.Mult(rhs_divfree, potential_divfree);
    PrintConvergence(CTMC_solver_, param_.verbose);

    // Compute divergence free solution
    C_.Mult(potential_divfree, true_flux_divfree);
}

void MLDivFreeSolver::SolvePotential(const Vector& true_flux_divfree,
                                     Vector& blk_rhs_0,
                                     Vector& potential) const
{
    Vector rhs_p(B_.NumRows());
    M_.Mult(-1.0, true_flux_divfree, 1.0, blk_rhs_0);
    B_.Mult(blk_rhs_0, rhs_p);
    BBT_solver_.Mult(rhs_p, potential);
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

    Vector& true_flux_part = blk_y.GetBlock(0);
    SolveParticularSolution(blk_x.GetBlock(1), true_flux_part);

    if (param_.verbose)
        cout << "Particular solution found in " << chrono.RealTime() << "s.\n";

    chrono.Clear();
    chrono.Start();

    Vector true_flux_divfree(C_.NumRows());
    SolveDivFreeSolution(true_flux_part, blk_x.GetBlock(0), true_flux_divfree);

    if (param_.verbose)
        cout << "Divergence free solution found in " << chrono.RealTime() << "s.\n";

    blk_y.GetBlock(0) += true_flux_divfree;

    // Compute the right hand side for the pressure problem BB^T p = rhs_p
    chrono.Clear();
    chrono.Start();

    SolvePotential(true_flux_divfree, blk_x.GetBlock(0), blk_y.GetBlock(1));

    if (param_.verbose)
        cout << "Pressure solution found in " << chrono.RealTime() << "s.\n";
}

void MLDivFreeSolver::SetOperator(const Operator &op)
{
    const SparseMatrix* mat = dynamic_cast<const SparseMatrix*>(&op);
    MFEM_VERIFY(mat, "MLDivFreeSolver: op needs to be a SparseMatrix");
    B_fine_.MakeRef(*mat);
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

