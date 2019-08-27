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

mfem::SparseMatrix GetSubMatrix(const SparseMatrix& A, const Array<int>& rows,
                                const Array<int>& cols, mfem::Array<int>& col_marker)
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

    assert(rows.Size() && rows.Max() < A.Height());
    assert(cols.Size() && cols.Max() < A.Width());
    assert(col_marker.Size() >= A.Width());

    for (int jcol(0); jcol < cols.Size(); ++jcol)
        col_marker[cols[jcol]] = jcol;

    const int nrow_sub = rows.Size();
    const int ncol_sub = cols.Size();

    int* i_sub = new int[nrow_sub + 1];
    i_sub[0] = 0;

    // Find the number of nnz.
    int nnz = 0;
    for (int i = 0; i < nrow_sub; ++i)
    {
        const int current_row = rows[i];

        for (int j = i_A[current_row]; j < i_A[current_row + 1]; ++j)
        {
            if (col_marker[j_A[j]] >= 0)
                ++nnz;
        }

        i_sub[i + 1] = nnz;
    }

    // Allocate memory
    int* j_sub = new int[nnz];
    double* a_sub = new double[nnz];

    // Fill in the matrix
    int count = 0;
    for (int i(0); i < nrow_sub; ++i)
    {
        const int current_row = rows[i];

        for (int j = i_A[current_row]; j < i_A[current_row + 1]; ++j)
        {
            if (col_marker[j_A[j]] >= 0)
            {
                j_sub[count] = col_marker[j_A[j]];
                a_sub[count] = a_A[j];
                count++;
            }
        }
    }

    // Restore colMapper so it can be reused other times!
    for (int jcol(0); jcol < cols.Size(); ++jcol)
        col_marker[cols[jcol]] = -1;

    return SparseMatrix(i_sub, j_sub, a_sub, nrow_sub, ncol_sub);
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

    //    unique_ptr<SparseMatrix> BT(Transpose(B));
    //    unique_ptr<SparseMatrix> BMinvBT(Mult(B, *BT));
    //    UMFPackSolver BMinvBT_solver(*BMinvBT);

    double F0 = F[0];
    const_cast<Vector&>(F)[0] = 0;

    Vector u(B.NumRows());
    Vector sigma(B.NumCols());

    BMinvBT_solver.Mult(F, u);
    BT.Mult(u, sigma);

    const_cast<Vector&>(F)[0] = F0;

    return sigma;
}

SparseMatrix ElemToDofs(const FiniteElementSpace& fes)
{
    int * I = new int[fes.GetNE()+1];
    copy_n(fes.GetElementToDofTable().GetI(), fes.GetNE()+1, I);

    const int nnz = I[fes.GetNE()];
    int * J = new int[nnz];
    copy_n(fes.GetElementToDofTable().GetJ(), nnz, J);

    double * data = new double[nnz];
    fill_n(data, nnz, 1.0);

    Array<int> dofs(J, nnz);
    fes.AdjustVDofs(dofs);

    return SparseMatrix(I, J, data, fes.GetNE(), fes.GetVSize());
}

Vector div_part(const unsigned int num_levels,
                const SparseMatrix& M_fine,
                const SparseMatrix& B_fine,
                const Vector& F_fine,
                const vector<SparseMatrix>& agg_elem,
                const vector<SparseMatrix>& elem_hdivdofs,
                const vector<SparseMatrix>& elem_l2dofs,
                const vector<SparseMatrix>& P_hdiv,
                const vector<SparseMatrix>& P_l2,
                const HypreParMatrix& coarse_hdiv_d_td,
                const HypreParMatrix& coarse_l2_d_td,
                Array<int>& coarsest_ess_dofs)
{
    vector<Vector> sigma(num_levels);

    Vector F_l, Pi_F_l, F_coarse, PT_F_l(F_fine);
    unique_ptr<SparseMatrix> B_l(new SparseMatrix(B_fine));
    unique_ptr<SparseMatrix> M_l(M_fine.NumRows() ? new SparseMatrix(M_fine) : nullptr);

    for (unsigned int l = 0; l < num_levels - 1; ++l)
    {
        unique_ptr<SparseMatrix> agg_l2dof(Mult(agg_elem[l], elem_l2dofs[l]));
        auto agg_hdivintdof = AggToIntDof(agg_elem[l], elem_hdivdofs[l]);

        // Right hand side: F_l = F - P_l2[l] (P_l2[l]^T P_l2[l])^{-1} P_l2[l]^T F

        F_l = PT_F_l;

        PT_F_l.SetSize(P_l2[l].NumCols());
        P_l2[l].MultTranspose(F_l, PT_F_l);

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

        DenseMatrix sub_B;
        DenseMatrix sub_M;

        Vector sub_F;
        Vector trash;

        Array<int> loc_hdivdofs;
        Array<int> loc_l2dofs;

        Array<int> col_marker(B_l->NumCols());
        col_marker = -1;

        sigma[l].SetSize(agg_hdivintdof.NumCols());
        sigma[l] = 0.0;

        for( int e = 0; e < agg_hdivintdof.NumRows(); e++)
        {
            agg_hdivintdof.GetRow(e, loc_hdivdofs, trash);
            agg_l2dof->GetRow(e, loc_l2dofs, trash);

            if (M_l)
            {
                sub_M.SetSize(loc_hdivdofs.Size());
                M_l->GetSubMatrix(loc_hdivdofs, loc_hdivdofs, sub_M);
            }

            sub_B.SetSize(loc_l2dofs.Size(), loc_hdivdofs.Size());
            B_l->GetSubMatrix(loc_l2dofs, loc_hdivdofs, sub_B);
            //            auto sub_B = GetSubMatrix(*B_l, loc_l2dofs, loc_hdivdofs, col_marker);

            F_l.GetSubVector(loc_l2dofs, sub_F);

            Vector sigma_loc = LocalSolution(sub_M, sub_B, sub_F);

            sigma[l].AddElementVector(loc_hdivdofs, sigma_loc);
        }  // loop over elements

        B_l.reset(RAP(P_l2[l], *B_l, P_hdiv[l]));
        if (M_l)
        {
            M_l.reset(RAP(P_hdiv[l], *M_l, P_hdiv[l]));
        }
    }  // loop over levels

    // The coarse problem:
    B_l->EliminateCols(coarsest_ess_dofs);
    if (M_l)
    {
        for ( int k = 0; k < coarsest_ess_dofs.Size(); ++k)
            if (coarsest_ess_dofs[k] !=0)
                M_l->EliminateRowCol(k);
    }

    unique_ptr<HypreParMatrix> B_Coarse(
                coarse_hdiv_d_td.LeftDiagMult(*B_l, coarse_l2_d_td.GetColStarts()));
    unique_ptr<HypreParMatrix> BT_coarse(B_Coarse->Transpose());

    Vector true_sigma_c(B_Coarse->NumCols());

    if (M_l)
    {
        unique_ptr<HypreParMatrix> d_td_M(coarse_hdiv_d_td.LeftDiagMult(*M_l));
        unique_ptr<HypreParMatrix> d_td_T(coarse_hdiv_d_td.Transpose());
        unique_ptr<HypreParMatrix> M_Coarse(ParMult(d_td_T.get(), d_td_M.get()));

        Array<int> block_offsets(3); // number of variables + 1
        block_offsets[0] = 0;
        block_offsets[1] = M_Coarse->Width();
        block_offsets[2] = B_Coarse->Height();
        block_offsets.PartialSum();

        BlockOperator coarseMatrix(block_offsets);
        coarseMatrix.SetBlock(0,0, M_Coarse.get());
        coarseMatrix.SetBlock(0,1, BT_coarse.get());
        coarseMatrix.SetBlock(1,0, B_Coarse.get());

        BlockVector trueX(block_offsets), trueRhs(block_offsets);
        trueRhs =0;
        trueRhs.GetBlock(1)= PT_F_l;

        Vector Md;
        M_Coarse->GetDiag(Md);
        BT_coarse->InvScaleRows(Md);
        unique_ptr<HypreParMatrix> S(ParMult(B_Coarse.get(), BT_coarse.get()));
        BT_coarse->ScaleRows(Md);

        HypreSmoother invM(*M_Coarse);
        HypreBoomerAMG invS(*S);
        invS.SetPrintLevel(0);

        BlockDiagonalPreconditioner darcyPr(block_offsets);
        darcyPr.SetDiagonalBlock(0, &invM);
        darcyPr.SetDiagonalBlock(1, &invS);

        MINRESSolver solver(coarse_l2_d_td.GetComm());
        solver.SetAbsTol(1.e-12);
        solver.SetRelTol(1.e-9);
        solver.SetMaxIter(500);
        solver.SetOperator(coarseMatrix);
        solver.SetPreconditioner(darcyPr);
        solver.SetPrintLevel(0);
        trueX = 0.0;
        solver.Mult(trueRhs, trueX);
        true_sigma_c = trueX.GetBlock(0);
    }
    else
    {
        unique_ptr<HypreParMatrix> S(ParMult(B_Coarse.get(), BT_coarse.get()));
        HypreBoomerAMG invS(*S);
        invS.SetPrintLevel(0);

        Vector u_c(B_Coarse->Height());
        u_c = 0.0;

        CGSolver solver(coarse_l2_d_td.GetComm());
        solver.SetAbsTol(1.e-12);
        solver.SetRelTol(1.e-9);
        solver.SetMaxIter(500);
        solver.SetOperator(*S);
        solver.SetPreconditioner(invS);
        solver.SetPrintLevel(0);

        solver.Mult(PT_F_l, u_c);
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

InterpolationCollector::InterpolationCollector(ParFiniteElementSpace& fes,
                                               int num_refine)
    : fes_(fes), coarse_fes_(fes.GetParMesh(), fes.FEColl()), ref_count_(0)
{
    P_.SetSize(num_refine, OperatorHandle(Operator::Hypre_ParCSR));
    fes_.SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
    coarse_fes_.SetUpdateOperatorType(Operator::MFEM_SPARSEMAT);
}

void InterpolationCollector::Collect()
{
    //        auto P_loc = ((const SparseMatrix*)fes_.GetUpdateOperator());
    //        auto d_td_coarse = coarse_fes_.Dof_TrueDof_Matrix();
    //        auto RP_loc = Mult(*fes_.GetRestrictionMatrix(), *P_loc);

    //        P_.Append(d_td_coarse->LeftDiagMult(*RP_loc, fes_.GetTrueDofOffsets()));
    //        P_.Last()->CopyColStarts();
    //        P_.Last()->CopyRowStarts();
    //        delete RP_loc;

    fes_.Update();
    fes_.GetTrueTransferOperator(coarse_fes_, P_[ref_count_++]);
    coarse_fes_.Update();
}

//    InterpolationCollector::~InterpolationCollector()
//    {
//        for (auto& P_ptr : P_)
//            delete P_ptr;
//    }

Multigrid::Multigrid(HypreParMatrix& Op,
                     const Array<OperatorHandle>& P,
                     Solver* CoarsePrec)
    :
      Solver(Op.GetNumRows()),
      P_(P),
      Ops_(P.Size()+1),
      Smoothers_(Ops_.Size()),
      current_level(Ops_.Size()-1),
      correction(Ops_.Size()),
      residual(Ops_.Size()),
      CoarseSolver(NULL),
      CoarsePrec_(CoarsePrec)
{
    if (CoarsePrec)
    {
        CoarseSolver = new CGSolver(Op.GetComm());
        CoarseSolver->SetRelTol(1e-8);
        CoarseSolver->SetMaxIter(50);
        CoarseSolver->SetPrintLevel(0);
        CoarseSolver->SetOperator(*Ops_[0]);
        CoarseSolver->SetPreconditioner(*CoarsePrec);
    }

    Ops_.Last().Reset(&Op, false);
    for (int l = Ops_.Size()-1; l > 0; --l)
    {
        Ops_[l-1].MakePtAP(Ops_[l], const_cast<OperatorHandle&>(P_[l-1]));
        // Two steps RAP
        //            unique_ptr<HypreParMatrix> PT( P[l-1]->Transpose() );
        //            unique_ptr<HypreParMatrix> AP( ParMult(Ops_[l], P[l-1]) );
        //            Ops_[l-1] = ParMult(PT.get(), AP.get());
        //            Ops_[l-1]->CopyRowStarts();
    }

    for (int l = 0; l < Ops_.Size(); ++l)
    {
        Smoothers_[l] = new HypreSmoother(*Ops_[l].As<HypreParMatrix>());
        residual[l].SetSize(Ops_[l]->NumRows());
        if (l < Ops_.Size()-1)
            correction[l].SetSize(Ops_[l]->NumRows());
    }
}

Multigrid::~Multigrid()
{
    for (int l = 0; l < Ops_.Size(); ++l)
    {
        delete Smoothers_[l];
    }
}

void Multigrid::Mult(const Vector& x, Vector& y) const
{
    residual.back() = x;
    correction.back().SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    // PreSmoothing
    auto& Operator_l = *Ops_[current_level].As<HypreParMatrix>();
    const HypreSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = residual[current_level];
    Vector& correction_l = correction[current_level];

    Smoother_l.Mult(residual_l, correction_l);
    Operator_l.Mult(-1.0, correction_l, 1.0, residual_l);

    // Coarse grid correction
    if (current_level > 0)
    {
        auto& P_l = *P_[current_level-1].As<HypreParMatrix>();

        P_l.MultTranspose(residual_l, residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
        }
    }

    // PostSmoothing
    Smoother_l.Mult(residual_l, cor_cor);
    correction_l += cor_cor;
}

