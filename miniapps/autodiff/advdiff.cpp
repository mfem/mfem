#include "advdiff.hpp"
#include "petsc.h"

namespace mfem {


void AdvectionDiffusionMXSolver::DirectSolver(mfem::BlockOperator& A)
{
    delete psol; psol=nullptr;
    delete prec; prec=nullptr;
    delete pmat; pmat=nullptr;

    mfem::HypreParMatrix* A00=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,0)));
    mfem::HypreParMatrix* A01=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,1)));
    mfem::HypreParMatrix* A02=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,2)));
    mfem::HypreParMatrix* A10=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,0)));
    mfem::HypreParMatrix* A11=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,1)));
    mfem::HypreParMatrix* A12=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,2)));
    mfem::HypreParMatrix* A20=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(2,0)));
    mfem::HypreParMatrix* A21=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(2,1)));
    mfem::HypreParMatrix* A22=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(2,2)));

    Array2D< HypreParMatrix * > bm(3,3);
    bm(0,0)=A00; bm(0,1)=A01; bm(0,2)=A02;
    bm(1,0)=A10; bm(1,1)=A11; bm(1,2)=A12;
    bm(2,0)=A20; bm(2,1)=A21; bm(2,2)=A22;

    HypreParMatrix* MM=mfem::HypreParMatrixFromBlocks(bm);

    mfem::PetscParMatrix* pmat=new mfem::PetscParMatrix(MM,mfem::Operator::PETSC_MATAIJ);
    mfem::PetscLinearSolver*   psol=new mfem::PetscLinearSolver(pmesh->GetComm());
    psol->SetOperator(*pmat);
    psol->SetAbsTol(abs_tol);
    psol->SetRelTol(rel_tol);
    psol->SetMaxIter(max_iter);
    psol->SetPrintLevel(print_level);
    psol->Mult(rhs, sol);

    delete MM;
}

void AdvectionDiffusionMXSolver::PETSCSolver(mfem::BlockOperator& A)
{
    delete psol; psol=nullptr;
    delete prec; prec=nullptr;
    delete pmat; pmat=nullptr;

    pmat= new mfem::PetscParMatrix(pmesh->GetComm(),&A, mfem::Operator::PETSC_MATAIJ);
    // construct the preconditioner
    prec = new mfem::PetscFieldSplitSolver(pmesh->GetComm(),*pmat,"prec_");
    // construct the linear solver
    psol = new mfem::PetscLinearSolver(pmesh->GetComm());

    psol->SetOperator(*pmat);
    psol->SetPreconditioner(*prec);
    psol->SetAbsTol(abs_tol);
    psol->SetRelTol(rel_tol);
    psol->SetMaxIter(max_iter);
    psol->SetPrintLevel(print_level);
    psol->Mult(rhs, sol);

}

void AdvectionDiffusionMXSolver::FSolve()
{
    sol=0.0;

    if(nfin.size()==0)
    {
        //add the domain integrator
        nfin.push_back(new mfem::AdvectionDiffusionMX(dicoef,vecoef,mucoef,incoef));
        nf->AddDomainIntegrator(nfin[nfin.size()-1]);


        //add BC face integrators
        for(auto it=bcc.begin();it!=bcc.end();it++)
        {
            mfem::AdvectionDiffusionMX* iin=new mfem::AdvectionDiffusionMX(dicoef,vecoef,mucoef,incoef);
            nfin.push_back(iin);
            iin->SetDirichletBCCoeficient(it->first);
            nf->AddBdrFaceIntegrator(iin,it->second);
        }


        nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);
    }

    // set the RHS
    nf->Mult(sol,rhs);
    rhs.Neg();

    mfem::BlockOperator& A=nf->GetGradient(sol);

    DirectSolver(A);


}



void AdvectionDiffusionMXSolver::ASolve(BlockVector &rhs)
{

}



}
