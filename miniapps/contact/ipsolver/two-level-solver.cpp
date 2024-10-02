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
    InitMumps();
}

void TwoLevelAMGSolver::SetNonContactTransferMap(const Operator & P)
{
    Pi = dynamic_cast<const HypreParMatrix *>(&P);
    // Ai = RAP(A, Pi);
    // Si = new HypreBoomerAMG(*Ai);
    // Si->SetPrintLevel(0);
    // Si->SetSystemsOptions(3);
    // Si->SetRelaxType(relax_type);
}

void TwoLevelAMGSolver::InitAMG()
{
    S = new HypreBoomerAMG(*A);
    S->SetPrintLevel(0);
    S->SetSystemsOptions(3);
    S->SetRelaxType(relax_type);
}

void TwoLevelAMGSolver::InitMumps()
{
    Ac = RAP(A, Pc);
    Sc = new MUMPSSolver(comm);
    Sc->SetPrintLevel(0);
    Sc->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
    Sc->SetOperator(*Ac);
}

void TwoLevelAMGSolver::Mult(const Vector & b, Vector & x) const
{
    MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent x and y size");

    x = 0.0;
    Vector z(x);
    S->Mult(b, z);
    x+=z;
    Vector rc(Pc->Width());
    Vector xc(Pc->Width());
    Vector r(b);
    // 2. Compute Residual r = b - A x
    A->AddMult(x,r,-1.0);
    // 3. Restrict to subspace
    Pc->MultTranspose(r,rc);
    // 4. Solve on the subspace
    Sc->Mult(rc,xc);
    // 5. Transfer to fine space
    Pc->Mult(xc,z);
    // 6. Update Correction
    x+=z;
    // 7. Compute Residual r = b - A x = r - A z
    A->AddMult(z,r,-1.0);
    // 8. Post V-Cycle 
    S->Mult(r, z);
    x+= z;

    // Vector z(x.Size());
    // Vector rc(Pc->Width());
    // Vector xc(Pc->Width());
    // Pc->MultTranspose(b,rc);
    // Sc->Mult(rc,xc);
    // Pc->Mult(xc,z);
    // x+=z;
    // Vector r(b);
    // A->AddMult(z,r,-1.0);
    // S->Mult(r, z);
    // x+=z;
    // A->AddMult(z,r,-1.0);
    // Pc->MultTranspose(r,rc);
    // Sc->Mult(rc,xc);
    // Pc->Mult(xc,z);
    // x+=z;



}


TwoLevelContactSolver::TwoLevelContactSolver(MPI_Comm comm_): Solver()
{
    Init(comm_);
}

TwoLevelContactSolver::TwoLevelContactSolver(const Operator & A_, const Operator & D_, const Operator & Pi_, const Operator & Pc_)
: Solver()
{
   auto APtr = dynamic_cast<const HypreParMatrix *>(&A_);
   MFEM_VERIFY(APtr, "Operator: Not a compatible matrix type");
   Init(APtr->GetComm());

   A = dynamic_cast<const HypreParMatrix *>(&A_);
   JtDJ = dynamic_cast<const HypreParMatrix *>(&D_);
   K = ParAdd(A, JtDJ);  // A + Ju^T D Ju
   height = A->Height();
   width = A->Width(); 
   SetContactTransferMap(Pc_);
   SetNonContactTransferMap(Pi_);

}

void TwoLevelContactSolver::Init(MPI_Comm comm_)
{
    comm=comm_;
    MPI_Comm_size(comm, &numProcs);
    MPI_Comm_rank(comm, &myid);
}

void TwoLevelContactSolver::SetOperator(const Operator & Op) { }

void TwoLevelContactSolver::SetNonContactTransferMap(const Operator & P)
{
    Pi = dynamic_cast<const HypreParMatrix *>(&P);
    Ai = RAP(K,Pi);
    InitAMG();
}


void TwoLevelContactSolver::SetContactTransferMap(const Operator & P)
{
    Pc = dynamic_cast<const HypreParMatrix *>(&P);
    Ac = RAP(K,Pc);
    InitMumps();
}

void TwoLevelContactSolver::InitAMG()
{
    Si = new HypreBoomerAMG(*Ai);
    Si->SetPrintLevel(0);
    Si->SetSystemsOptions(3);
    Si->SetRelaxType(relax_type);
}

void TwoLevelContactSolver::InitMumps()
{
    Sc = new MUMPSSolver(comm);
    Sc->SetPrintLevel(0);
    Sc->SetMatrixSymType(MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
    Sc->SetOperator(*Ac);
}

void TwoLevelContactSolver::Mult(const Vector & b, Vector & x) const
{
    MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent x and y size");

    x = 0.0;
    Vector z(x);
    Vector r(b);
    Vector ri(Pi->Width());
    Vector zi(Pi->Width()); zi = 0.0;
    Pi->MultTranspose(r,ri);
    Si->Mult(ri, zi);
    Pi->Mult(zi,z);
    x+=z;

    // // 2. Compute Residual r = b - A x
    K->AddMult(z,r,-1.0);
    Vector rc(Pc->Width());
    Vector zc(Pc->Width());
    // 3. Restrict to subspace
    Pc->MultTranspose(r,rc);
    // 4. Solve on the subspace
    Sc->Mult(rc,zc);
    // 5. Transfer to fine space
    Pc->Mult(zc,z);
    // 6. Update Correction
    x+=z;
    // 7. Compute Residual r = b - A x
    K->AddMult(z,r,-1.0);
    // 8. Post V-Cycle 
    Pi->MultTranspose(r,ri);
    Si->Mult(ri, zi);
    Pi->Mult(zi,z);
    x+=z;
}



} // namespace
