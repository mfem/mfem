#include "stokes.hpp"
#include "petsc.h"

namespace mfem {

void StokesSolver::FSolve()
{

    if(pmat!=nullptr)
    {
        delete psol;
        delete prec;
        delete pmat;
        ess_tdofv.DeleteAll();
    }


    sol=0.0;
    // Set the BC
    ess_tdofv.DeleteAll();
    Array<int> ess_tdofx;
    Array<int> ess_tdofy;
    Array<int> ess_tdofz;
    Array<int> ess_tdofp;
    int dim=pmesh->Dimension();
    {
        for(auto it=bccx.begin();it!=bccx.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,0);
            ess_tdofx.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(0, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsx from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofx[ii]]=rhs.GetBlock(0)[ess_tdofx[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofx);


        for(auto it=bccy.begin();it!=bccy.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,1);
            ess_tdofy.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(1, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsy from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofy[ii]]=rhs.GetBlock(0)[ess_tdofy[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofy);

        for(auto it=bccz.begin();it!=bccz.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,2);
            ess_tdofz.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);

            pcoeff.Set(2, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsz from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofz.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofz[ii]]=rhs.GetBlock(0)[ess_tdofz[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofz);
    }

    if(nfin==nullptr)
    {
        nfin=new StokesIntegratorTH(viscosity,bpenal,load);
        nf->AddDomainIntegrator(nfin);
        nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);
    }
    // set the RHS
    nf->Mult(sol,rhs);
    rhs.Neg();

    mfem::BlockOperator& A=nf->GetGradient(sol);

    mfem::HypreParMatrix* A00=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,0)));
    mfem::HypreParMatrix* A01=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,1)));
    mfem::HypreParMatrix* A10=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,0)));
    mfem::HypreParMatrix* A11=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,1)));

    mfem::HypreParMatrix* A00elim=A00->EliminateRowsCols(ess_tdofv);
    mfem::HypreParMatrix* A11elim=A11->EliminateRowsCols(ess_tdofp);
    mfem::HypreParMatrix* A01elim=A01->EliminateCols(ess_tdofp); A01->EliminateRows(ess_tdofv);
    mfem::HypreParMatrix* A10elim=A10->EliminateCols(ess_tdofv); A10->EliminateRows(ess_tdofp);

    //copy BC to RHS

    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        rhs.GetBlock(0)[ess_tdofv[ii]]=sol.GetBlock(0)[ess_tdofv[ii]];
    }

    delete psol;
    delete prec;
    delete pmat;
    pmat= new mfem::PetscParMatrix(pmesh->GetComm(),&A, mfem::Operator::PETSC_MATAIJ);
    //set the local block size of the matrix
    Mat sub;
    MatNestGetSubMat(pmat->operator petsc::Mat(),0,0,&sub);
    MatSetBlockSize(sub,dim);
    // construct the preconditioner
    prec = new mfem::PetscFieldSplitSolver(pmesh->GetComm(),*pmat,"prec_");
    // construct the linear solver
    psol = new mfem::PetscLinearSolver(pmesh->GetComm());

    /*
    {
        std::fstream out("full.mat",std::ios::out);
        pmat->PrintMatlab(out);
        out.close();
    }
    */


    psol->SetOperator(*pmat);
    psol->SetPreconditioner(*prec);
    psol->SetAbsTol(abs_tol);
    psol->SetRelTol(rel_tol);
    psol->SetMaxIter(max_iter);
    psol->SetPrintLevel(print_level);
    psol->Mult(rhs, sol);

    //psol->GetConverged();

    delete A11elim;
    delete A01elim;
    delete A10elim;
    delete A00elim;

}


void StokesSolver::ASolve(BlockVector& arhs)
{
    if(pmat==nullptr)
    {
        MFEM_ABORT("StokesSolve::Adjoint - The forward solver should be called first!!!")
    }

    //set BC
    rhs=arhs;
    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        adj.GetBlock(0)[ess_tdofv[ii]]=0.0;
        rhs.GetBlock(0)[ess_tdofv[ii]]=0.0;
    }

    psol->Mult(rhs, adj);
}

void StokesSolver::GradD(Vector &grad)
{
    if(dfes==nullptr)
    {
        MFEM_ABORT("StokesSolve::GradD - The design space in not set!!!")
    }

    //set vector size
    grad.SetSize(dfes->GetTrueVSize());
    grad=0.0;

    fvelocity.SetFromTrueDofs(sol.GetBlock(0));
    avelocity.SetFromTrueDofs(adj.GetBlock(0));

    mfem::ParLinearForm lf(dfes);
    lf.AddDomainIntegrator(new StokesGradIntergrator(fvelocity, avelocity
                                                     ,*(ltopopt.fcoef),
                                                     vfes->GetOrder(0)));
    lf.ParallelAssemble(grad);
}


}
