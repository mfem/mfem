#include "rand_eigensolver.hpp"

namespace mfem{

void RandomizedSubspaceIteration::Solve()
{

    if(A==nullptr){return;}

    int myrank=0;
#ifdef MFEM_USE_MPI
    MPI_Comm_rank(comm,&myrank);
#endif
    std::random_device rd;
    std::mt19937 generator(rd());


    // Create a normal distribution object
    std::normal_distribution<real_t> distribution(0.0, 1.0);



    std::vector<Vector> omega; omega.resize(num_modes);

    //populate omega with standard Gaussian RV
    for(int i=0;i<num_modes;i++){
        omega[i].SetSize(A->NumCols());
        Vector& cv=omega[i];

        for(int j=0;j<A->NumCols();j++)
        {
            cv[j]=distribution(generator);
        }
    }

    if(ess_tdofs!=nullptr){
        for(int j=0;j<num_modes;j++){
            Vector& cv=omega[j];
            for(int i=0;i<ess_tdofs->Size();i++){
                cv[(*ess_tdofs)[i]]=0.0;
            }
        }
    }


    //initialize the modes
    for(int i=0;i<num_modes;i++){
        A->Mult(omega[i],modes[i]);
    }

    real_t gp;

    //Orthogonalize the modes
    for(int i=0;i<num_modes;i++){
       for(int j=0;j<i;j++){
#ifdef MFEM_USE_MPI
           gp=InnerProduct (comm, modes[i], modes[j]);
#else
           gp=InnerProduct (modes[i], modes[j]);
#endif
           modes[i].Add(-gp,modes[j]);
       }

#ifdef MFEM_USE_MPI
       gp=InnerProduct (comm, modes[i], modes[i]);
#else
       gp=InnerProduct (modes[i], modes[i]);
#endif
       if(fabs(gp)>std::numeric_limits<real_t>::epsilon()){
           modes[i]/=sqrt(gp); //scale the vector
       }
    }


    for(int it=0;it<iter;it++){
        for(int i=0;i<num_modes;i++){
            if(symmetric){
                A->Mult(modes[i],omega[i]);
            }else{
                A->MultTranspose(modes[i],omega[i]);
            }

            for(int j=0;j<i;j++){
     #ifdef MFEM_USE_MPI
                gp=InnerProduct (comm, omega[i], omega[j]);
     #else
                gp=InnerProduct (omega[i], omega[j]);
     #endif
                omega[i].Add(-gp,omega[j]);
            }

     #ifdef MFEM_USE_MPI
            gp=InnerProduct (comm, omega[i], omega[i]);
     #else
            gp=InnerProduct (omega[i], omega[i]);
     #endif
            if(fabs(gp)>std::numeric_limits<real_t>::epsilon()){
                omega[i]/=sqrt(gp); //scale the vector
            }
        }

        //test the product
        /*
         if(myrank==0){std::cout<<std::endl;}
        for(int i=0;i<num_modes;i++){
            for(int j=0;j<num_modes;j++){
                real_t bb=InnerProduct(comm,omega[i],modes[j]);
                if(myrank==0){std::cout<<bb<<" ";}

            }
            if(myrank==0){std::cout<<std::endl;}
        }
        */


        for(int i=0;i<num_modes;i++){
            A->Mult(omega[i],modes[i]);
            for(int j=0;j<i;j++){
     #ifdef MFEM_USE_MPI
                gp=InnerProduct (comm, modes[i], modes[j]);
     #else
                gp=InnerProduct (modes[i], modes[j]);
     #endif
                modes[i].Add(-gp,modes[j]);
            }

     #ifdef MFEM_USE_MPI
            gp=InnerProduct (comm, modes[i], modes[i]);
     #else
            gp=InnerProduct (modes[i], modes[i]);
     #endif
            if(fabs(gp)>std::numeric_limits<real_t>::epsilon()){
                modes[i]/=sqrt(gp); //scale the vector
            }
        }
    }


}


/// Solves the generalized eigenproble using fixed number of iterations
/// Does not use adaptive check of the precision just run the
/// specified number of iterations
void AdaptiveRandomizedGenEig::SolveNA()
{
    if(A==nullptr){return;}

    int myrank=0;
    MPI_Comm_rank(comm,&myrank);

    std::random_device rd;
    std::mt19937 generator(rd());

    // Create a normal distribution object
    std::normal_distribution<real_t> distribution(0.0, 1.0);

    std::vector<Vector> omega; omega.resize(num_modes);

    //populate omega with standard Gaussian RV
    for(int i=0;i<num_modes;i++){
        omega[i].SetSize(A->NumCols());
        Vector& cv=omega[i];

        for(int j=0;j<A->NumCols();j++)
        {
            cv[j]=distribution(generator);
        }
    }

    //initialize the modes
    for(int i=0;i<num_modes;i++){
        A->Mult(omega[i],modes[i]);
        iB->Mult(modes[i], omega[i]);
    }
    //Orthogonalize the modes using A inner product
    //Ortho(A,omega,modes);
    if(B==nullptr){
        Ortho(omega,modes);
    }else{
        DenseMatrix AA(num_modes);
        DenseMatrix BB(num_modes);

        for(int i=0;i<num_modes;i++){
            A->Mult(omega[i],modes[i]);
            AA(i,i)=InnerProduct(comm, omega[i], modes[i]);
            for(int j=0;j<i;j++){
                AA(i,j)=InnerProduct(comm, omega[i], modes[j]);
                AA(j,i)=AA(i,j);
            }
        }

        for(int i=0;i<num_modes;i++){
            B->Mult(omega[i],modes[i]);
            BB(i,i)=InnerProduct(comm, omega[i], modes[i]);
            for(int j=0;j<i;j++){
                BB(i,j)=InnerProduct(comm, omega[i], modes[j]);
                BB(j,i)=BB(i,j);
            }
        }

        DenseMatrixGeneralizedEigensystem eig(AA,BB,false,true);
        eig.Eval();
        DenseMatrix& evecs=eig.RightEigenvectors();

        for(int i=0;i<num_modes;i++){
            modes[i]=0.0;
            for(int j=0;j<num_modes;j++){
                modes[i].Add(evecs(j,i),omega[j]);
            }
        }
    }




    for(int it=0;it<max_iter;it++){
        for(int i=0;i<num_modes;i++){
            iB->Mult(modes[i],omega[i]);
            A->Mult(omega[i], modes[i]);
        }

        //Ortho(A,modes,omega);
        Ortho(modes,omega);

        for(int i=0;i<num_modes;i++){
            A->Mult(omega[i],modes[i]);
            iB->Mult(modes[i], omega[i]);
        }
        //Ortho(A,omega,modes);
        if(B==nullptr){
            Ortho(omega,modes);
        }else{
            DenseMatrix AA(num_modes);
            DenseMatrix BB(num_modes);

            for(int i=0;i<num_modes;i++){
                A->Mult(omega[i],modes[i]);
                AA(i,i)=InnerProduct(comm, omega[i], modes[i]);
                for(int j=0;j<i;j++){
                    AA(i,j)=InnerProduct(comm, omega[i], modes[j]);
                    AA(j,i)=AA(i,j);
                }
            }

            for(int i=0;i<num_modes;i++){
                B->Mult(omega[i],modes[i]);
                BB(i,i)=InnerProduct(comm, omega[i], modes[i]);
                for(int j=0;j<i;j++){
                    BB(i,j)=InnerProduct(comm, omega[i], modes[j]);
                    BB(j,i)=BB(i,j);
                }
            }

            DenseMatrixGeneralizedEigensystem eig(AA,BB,false,true);
            eig.Eval();
            DenseMatrix& evecs=eig.RightEigenvectors();
            Vector& evals=eig.EigenvaluesRealPart();

            for(int i=0;i<num_modes;i++){
                modes[i]=0.0;
                for(int j=0;j<num_modes;j++){
                    modes[i].Add(evecs(j,i),omega[j]);
                }
            }
        }

    }

    Ortho(A,modes);


}

void AdaptiveRandomizedGenEig::Ortho(const Operator* C,
                                      std::vector<Vector>& vecs)
{
    Vector tv(vecs[0]);
    DenseMatrix AA(vecs.size(),vecs.size());
    real_t gp;
    for(size_t i=0;i<vecs.size();i++){
        C->Mult(vecs[i],tv);
        for(size_t j=0;j<i;j++){
            gp=InnerProduct (comm, tv, vecs[j]);
            AA(i,j)=gp;
            AA(j,i)=gp;
        }
        gp=InnerProduct (comm, tv, vecs[i]);
        AA(i,i)=gp;
    }

    Vector sv(vecs.size());

    CholeskyFactors chol(AA.GetData());
    chol.Factor(AA.NumRows());

    int myrank;
    MPI_Comm_rank(comm,&myrank);

    for(int i=(vecs.size()-1);i>=0;i=i-1){
        sv=0.0; sv[i]=1.0;
        chol.USolve(vecs.size(),1,sv.GetData());
        (vecs[i])*=sv[i];
        for(int j=(i-1);j>=0;j--){
            vecs[i].Add(sv[j],vecs[j]);
        }
    }
}


//return the othogonalized vectors in orth
void AdaptiveRandomizedGenEig::Ortho(const Operator* B,
                                      std::vector<Vector>& vecs,
                                      std::vector<Vector>& orth)
{
    real_t gp;


    DenseMatrix A(vecs.size(),vecs.size());
    for(size_t i=0;i<vecs.size();i++){
        orth[i].SetSize(vecs[i].Size());
        B->Mult(vecs[i],orth[i]);
        for(size_t j=0;j<i;j++){
            gp=InnerProduct (comm, orth[i], vecs[j]);
            A(i,j)=gp;
            A(j,i)=gp;
        }
        gp=InnerProduct (comm, orth[i], vecs[i]);
        A(i,i)=gp;
    }

    Vector eval; eval.SetSize(vecs.size());
    DenseMatrix evec; evec.SetSize(vecs.size(),vecs.size());
    A.Eigensystem(eval,evec);

    int nn=vecs.size()-1;

    for(size_t i=0;i<vecs.size();i++){
        orth[nn-i]=real_t(0.0);
        for(size_t j=0;j<vecs.size();j++){
            orth[nn-i].Add(evec(j,i)/sqrt(eval(i)),vecs[j]);
        }
    }
}

void AdaptiveRandomizedGenEig::Ortho( std::vector<Vector>& vecs,
                                      std::vector<Vector>& orth)
{
    real_t gp;
    DenseMatrix A(vecs.size(),vecs.size());
    for(size_t i=0;i<vecs.size();i++){
        for(size_t j=0;j<i;j++){
            gp=InnerProduct (comm, vecs[i], vecs[j]);
            A(i,j)=gp;
            A(j,i)=gp;
        }
        gp=InnerProduct (comm, vecs[i], vecs[i]);
        A(i,i)=gp;
    }

    Vector eval; eval.SetSize(vecs.size());
    DenseMatrix evec; evec.SetSize(vecs.size(),vecs.size());
    A.Eigensystem(eval,evec);

    int nn=vecs.size()-1;

    for(size_t i=0;i<vecs.size();i++){
        orth[nn-i]=real_t(0.0);
        for(size_t j=0;j<vecs.size();j++){
            orth[nn-i].Add(evec(j,i)/sqrt(eval(i)),vecs[j]);
        }
    }
}

}
