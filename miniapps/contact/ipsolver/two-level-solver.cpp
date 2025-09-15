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

#include "two-level-solver.hpp"

namespace mfem
{

TwoLevelAMGSolver::TwoLevelAMGSolver(MPI_Comm comm_): Solver()
{
    Init(comm_);
}

TwoLevelAMGSolver::TwoLevelAMGSolver(const Operator & Op, const Operator & P_)
: Solver()
{
   auto APtr = dynamic_cast<const HypreParMatrix *>(&Op);
   MFEM_VERIFY(APtr, "Operator: Not a compatible matrix type");
   Init(APtr->GetComm());
   auto PPtr = dynamic_cast<const HypreParMatrix *>(&P_);
   MFEM_VERIFY(PPtr, "Transfer Map: not a compatible matrix type");
   
   SetOperator(Op);
   SetContactTransferMap(P_);
}

void TwoLevelAMGSolver::Init(MPI_Comm comm_)
{
    comm=comm_;
    MPI_Comm_size(comm, &numProcs);
    MPI_Comm_rank(comm, &myid);
}

void TwoLevelAMGSolver::SetOperator(const Operator & Op)
{
    A = dynamic_cast<const HypreParMatrix *>(&Op);
    height = A->Height();
    width = A->Width();
    InitAMG();
}

void TwoLevelAMGSolver::SetContactTransferMap(const Operator & P)
{
    Pc = dynamic_cast<const HypreParMatrix *>(&P);
    InitCoarseSolver();
}

void TwoLevelAMGSolver::SetNonContactTransferMap(const Operator & P)
{
    Pnc = dynamic_cast<const HypreParMatrix *>(&P);
}

void TwoLevelAMGSolver::InitAMG()
{
    chrono.Restart();
    amg = new HypreBoomerAMG(*A);
    amg->SetPrintLevel(0);
    amg->SetSystemsOptions(3);
    amg->SetRelaxType(relax_type);

    chrono.Stop();
    if (myid == 0)
    {
        std::cout << "AMG in AMGF setup took " << chrono.RealTime() << " seconds.\n";
    }

    // Setup AMG
    Vector b(A->NumRows());
    b = 1.0;
    Vector z(A->NumRows());
    z = 0.0;
    chrono.Restart();
    amg->Mult(b, z);
    chrono.Stop();
    if (myid == 0)
    {
        std::cout << "AMG in AMGF setup mult took " << chrono.RealTime() << " seconds.\n";
    }
    // chrono.Restart();
    // amg->Mult(b, z);
    // chrono.Stop();
    // if (myid == 0)
    // {
    //     std::cout << "AMG in AMGF mult after setup mult took " << chrono.RealTime() << " seconds.\n";
    // }
    // chrono.Restart();
    // amg->Mult(b, z);
    // chrono.Stop();
    // if (myid == 0)
    // {
    //     std::cout << "AMG in AMGF mult after setup mult took " << chrono.RealTime() << " seconds.\n";
    // }
}

void TwoLevelAMGSolver::InitCoarseSolver()
{
    chrono.Restart();
    Ac = RAP(A, Pc);
    chrono.Stop();

    double A_nnz = ((double)A->NNZ());
    double Ac_nnz = ((double)Ac->NNZ());
    if (A_nnz < 0.0) // it's a hack, only works if overflow a little bit
    {
        A_nnz += 2147483648.0 * 2;
    }
    if (Ac_nnz < 0.0)
    {
        Ac_nnz += 2147483648.0 * 2;
    }

    if (myid == 0)
    {
        std::cout << "RAP(A, Pc) took " << chrono.RealTime() << " seconds.\n";
        std::cout << "A->NNZ() = " << A_nnz << "\n";
        std::cout << "Ac->NNZ() = " << Ac_nnz << "\n";
    }
    operator_complexity = 1.0 + (Ac_nnz / A_nnz);

    chrono.Restart();
#ifdef MFEM_USE_MUMPS
    Mcoarse = new MUMPSSolver(comm);
    auto M = dynamic_cast<MUMPSSolver *>(Mcoarse);
    M->SetPrintLevel(0);
    M->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
    M->SetOperator(*Ac);
#else
#ifdef MFEM_USE_MKL_CPARDISO
    Mcoarse = new CPardisoSolver(comm);
    auto M = dynamic_cast<CPardisoSolver *>(Mcoarse);
    M->SetMatrixType(CPardisoSolver::MatType::REAL_NONSYMMETRIC);
    M->SetOperator(*Ac);
#else
    MFEM_VERIFY(false, "TwoLevelSolver will only work for an mfem build that uses mumps or mkl_cpardiso");
#endif
#endif
    chrono.Stop();
    coarse_setup_time = chrono.RealTime();
    if (myid == 0)
    {
        std::cout << "Coarse solver setup took " << chrono.RealTime() << " seconds.\n";
    }
    chrono.Clear();
}


void TwoLevelAMGSolver::Mult(const Vector & b, Vector & x) const
{
    MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent x and y size");

    x = 0.0;
    Vector z(x);
    amg->Mult(b, z);
    x+=z;
    Vector rc(Pc->Width());
    Vector xc(Pc->Width());
    if (additive)
    {
        Pc->MultTranspose(b,rc);
        Mcoarse->Mult(rc,xc);
        Pc->Mult(xc,z);
    }
    else
    {
        Vector r(b.Size());
        // 2. Compute Residual r = b - A x
        A->Mult(x,r);
        r.Neg(); r+=b;
        // 3. Restrict to subspace
        Pc->MultTranspose(r,rc);
        // 4. Solve on the subspace
        chrono.Start();
        Mcoarse->Mult(rc,xc);
        chrono.Stop();
        // 5. Transfer to fine space
        Pc->Mult(xc,z);
        // 6. Update Correction
        x+=z;
        // 7. Compute Residual r = b - A x
        A->Mult(x,r);
        r.Neg(); r+=b;
        // 8. Post V-Cycle 
        amg->Mult(r, z);
    }
    x+= z;
}

} // namespace
