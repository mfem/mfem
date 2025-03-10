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



};
#endif // RAND_EIGENSOLVER_HPP
