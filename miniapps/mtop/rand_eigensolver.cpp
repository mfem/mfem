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
        iB->Mult(omega[i],modes[i]);
    }

    real_t gp;
    Vector tv(modes[0]);
    Vector bv(modes[0]);

    //Orthogonalize the modes using A inner product
    for(int i=0;i<num_modes;i++){
        A->Mult(modes[i],tv);
        iB->Mult(tv,modes[i]);
        for(int j=0;j<i;j++){
            gp=InnerProduct (comm, modes[i], modes[j]);
            modes[i].Add(-gp,modes[j]);
        }

        A->Mult(modes[i],tv);
        gp=InnerProduct (comm, tv, modes[i]);
        if(fabs(gp)>std::numeric_limits<real_t>::epsilon()){
            modes[i]/=sqrt(gp); //scale the vector
        }
    }

    for(int it=0;it<max_iter;it++){
        for(int i=0;i<num_modes;i++){
            A->Mult(modes[i],tv);
            iB->Mult(tv,modes[i]);
            for(int j=0;j<i;j++){
                gp=InnerProduct (comm, modes[i], modes[j]);
                modes[i].Add(-gp,modes[j]);
            }

            A->Mult(modes[i],tv);
            gp=InnerProduct (comm, tv, modes[i]);
            if(fabs(gp)>std::numeric_limits<real_t>::epsilon()){
                modes[i]/=sqrt(gp); //scale the vector
            }
        }
    }
}

void AdaptiveRandomizedGenEig::OrthoB(Operator* B,
                                      std::vector<Vector>& vecs)
{
    Vector tv(vecs[0]);

    real_t gp;

    for(int i=0;i<vecs.size();i++){
        for(int j=0;j<i;j++){
            B->Mult(vecs[i],tv);
            gp=InnerProduct (comm, tv, vecs[j]);
            vecs[i].Add(-gp,vecs[j]);
        }
        B->Mult(vecs[i],tv);
        gp=InnerProduct (comm, tv, vecs[i]);
        if(fabs(gp)>std::numeric_limits<real_t>::epsilon()){
            vecs[i]/=sqrt(gp); //scale the vector
        }
    }
}


//return the othogonalized vectors in orth
void AdaptiveRandomizedGenEig::OrthoB(Operator* B,
                                      std::vector<Vector>& vecs,
                                      std::vector<Vector>& orth)
{
    real_t gp;


    DenseMatrix A(vecs.size(),vecs.size());
    for(int i=0;i<vecs.size();i++){
        orth[i].SetSize(vecs[i].Size());
        B->Mult(vecs[i],orth[i]);
        for(int j=0;j<i;j++){
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

    for(int i=0;i<vecs.size();i++){
        orth[i]=real_t(0.0);
        for(int j=0;j<vecs.size();j++){
            orth[i].Add(evec(j,i)/sqrt(eval(i)),vecs[j]);
        }
    }
}



}
