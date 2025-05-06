#ifndef RAND_EIGENSOLVER_HPP
#define RAND_EIGENSOLVER_HPP

#include "mfem.hpp"


namespace mfem{

class RandomizedSubspaceIteration
{
public:

    RandomizedSubspaceIteration()
    {
#ifdef MFEM_USE_MPI
        comm=MPI_COMM_WORLD;
#endif
        num_modes=1;
        A=nullptr;
        modes.resize(num_modes);
        symmetric=false;
        ess_tdofs=nullptr;
        iter=1;
    }

#ifdef MFEM_USE_MPI
    RandomizedSubspaceIteration(MPI_Comm comm_, bool symm=false)
    {
        comm=comm_;
        num_modes=1;
        A=nullptr;
        modes.resize(num_modes);
        symmetric=symm;
        ess_tdofs=nullptr;
        iter=1;
    }
#endif

    void SetConstrDOFs(mfem::Array<int>& ess_tdofs_)
    {
        ess_tdofs=&ess_tdofs_;
    }

    void SetNumModes(int num_)
    {
        num_modes=num_;
        modes.resize(num_modes);
        if(A!=nullptr)
        {
            for(int i=0;i<num_modes;i++){
                modes[i].SetSize(A->NumRows());
            }
        }
    }

    int GetNumModes() const
    {
        return num_modes;
    }

    void SetNumIter(int it_){
        iter=it_;
    }

    int GetNumIter() const
    {
        return iter;
    }

    void SetOperator(const Operator& A_, bool symm=false)
    {
        A=&A_;
        for(int i=0;i<num_modes;i++){
            modes[i].SetSize(A->NumRows());
        }
        symmetric=symm;
    }

    void Solve();

    void GetMode(int i, Vector& q){
        if((i<num_modes)&&(A!=nullptr))
        {
            q=modes[i];
        }
    }

    const std::vector<Vector>& GetModes() const
    {
        return modes;
    }

private:
#ifdef MFEM_USE_MPI
    MPI_Comm comm;
#endif

    mfem::Array<int>* ess_tdofs;

    int num_modes;
    int iter;
    const Operator* A;
    std::vector<Vector> modes;

    bool symmetric;

};

class AdaptiveRandomizedGenEig
{
public:
    AdaptiveRandomizedGenEig(MPI_Comm comm_):comm(comm_)
    {
        num_modes=1;
        max_iter=10;
        eps=1e-6;

    }

    void SetOperators(const Operator& A_, const Operator& iB_)
    {
        A=&A_;
        iB=&iB_;

        for(int i=0;i<num_modes;i++){
            modes[i].SetSize(A->NumRows());
        }
    }

    void SetNumModes(int num_)
    {
        num_modes=num_;
        modes.resize(num_modes);
        if(A!=nullptr)
        {
            for(int i=0;i<num_modes;i++){
                modes[i].SetSize(A->NumRows());
            }
        }
    }

    int GetNumModes() const
    {
        return num_modes;
    }

    void SetNumIter(int it_){
        max_iter=it_;
    }

    int GetNumIter() const
    {
        return max_iter;
    }

    void Solve(bool flag_adaptive=false);


    void GetMode(int i, Vector& q){
        if((i<num_modes)&&(A!=nullptr))
        {
            q=modes[i];
        }
    }

    const std::vector<Vector>& GetModes() const
    {
        return modes;
    }


    void OrthoB(Operator* B,
                std::vector<Vector>& vecs);
private:
    MPI_Comm comm;

    const Operator* A;
    const Operator* iB;

    int num_modes;
    int max_iter;
    real_t eps;

    std::vector<Vector> modes;
    Vector evals;

    void SolveNA(); //non-addaptive solve

};


}
#endif // RAND_EIGENSOLVER_HPP
